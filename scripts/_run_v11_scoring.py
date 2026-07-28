# -*- coding: utf-8 -*-
"""v11 run1 の全採点ランナー（v10.2の採点系をv11のパス/構成で実行）。

構成（evalアセットはv10共用のため、集合ごとにDSが異なる）:
  1. step12 val      : DS=v11（plan/foa/scene）、SPLIT=val（1,200本、純静穏含む）
     ※車の採点はtrack0=plan条件の車が基準。複数車クリップの車リードは参考値
     ※v11 planに assignment_scenario.csv は無い（評価枠はv10共用）ため、roster読込用に
       v10の同名ファイルを一時複製し実行後に削除（行はv9.1以来同一シード）
  2. step12 scenario : DS=v10（交差点20本）→ OUT/scenario/
  3. step16相当      : v10a部=DS v10 / **1台対照=v11 valの n_car==1 & n_warnings==0**
     （step16の1台対照はv10 valを読む設計でv11予測と名簿が合わないため、
      本ランナー内にv11版を実装。集計ロジックはstep16と同一）
  4. step15 scn2 / step17 halluc / step18 probe: CLI引数でパス差し替え（subprocess）

出力: out/step12_notify_v11/
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import soundfile as sf  # noqa: E402

from outdoor_seld.calibration import frame_spl_a  # noqa: E402

DS11 = ROOT / "out" / "dataset_outdoor_siren_v11"
DS10 = ROOT / "out" / "dataset_outdoor_siren_v10"
PRED = ROOT / "out" / "predictions_v11"
OUT = ROOT / "out" / "step12_notify_v11"
CAR = 4


def load_m12():
    spec = importlib.util.spec_from_file_location(
        "step12", ROOT / "scripts" / "step12_notify_v9.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_pred_multi(path):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. step12 val（v11） ----
    print("=== 1/6 step12 val (DS=v11) ===", flush=True)
    m12 = load_m12()
    m12.DS, m12.PRED, m12.OUT, m12.SPLIT = DS11, PRED, OUT, "val"
    tmp = DS11 / "plan" / "assignment_scenario.csv"
    created = False
    if not tmp.exists():
        shutil.copyfile(DS10 / "plan" / "assignment_scenario.csv", tmp)
        created = True
    try:
        m12.main()
    finally:
        if created:
            tmp.unlink()

    # ---- 2. step12 scenario（v10、交差点20本） ----
    print("=== 2/6 step12 scenario (DS=v10) ===", flush=True)
    m12b = load_m12()
    m12b.DS, m12b.PRED, m12b.OUT, m12b.SPLIT = DS10, PRED, OUT / "scenario", "scenario"
    m12b.main()

    # ---- 3. step16相当（v10a=DS v10、1台対照=v11 val n_car==1&warn0） ----
    print("=== 3/6 v10a同時検出 (step16のv11版) ===", flush=True)
    m12c = load_m12()
    pred = load_pred_multi(PRED / "v10a_all.csv")
    plan = list(csv.DictReader(open(DS10 / "plan" / "assignment_v10a.csv")))
    simu = {2: [0, 0], 3: [0, 0, 0]}
    per_car, warn_ok, warn_n = [], 0, 0
    fires_by_ncars = defaultdict(list)
    for row in plan:
        clip = row["clip_id"]
        ncars = int(row["scenario"][-1])
        labels = defaultdict(list)
        for line in open(DS10 / "metadata" / f"{clip}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                labels[int(q[0])].append((int(q[1]), int(q[2]),
                                          float(q[3]), float(q[4])))
        p = pred[clip]
        for k, evs in labels.items():
            cars = [e for e in evs if e[0] == CAR]
            npred = len([e for e in p.get(k, []) if e[0] == CAR])
            if len(cars) == 2:
                simu[2][1] += 1
                simu[2][0] += int(npred >= 2)
            elif len(cars) == 3:
                simu[3][2] += 1
                simu[3][0] += int(npred >= 2)
                simu[3][1] += int(npred >= 3)
        scene = json.loads((DS10 / "work" / clip / "scene.json").read_text())
        for src in scene["sources"]:
            if src["class"] != "car_drive":
                continue
            tr = src["track"]
            hit = tot = 0
            for k, evs in labels.items():
                mine = [e for e in evs if e[0] == CAR and e[1] == tr]
                if not mine:
                    continue
                tot += 1
                gaz = mine[0][2]
                if any(abs((a - gaz + 180) % 360 - 180) <= 30.0
                       for c, a, _ in p.get(k, []) if c == CAR):
                    hit += 1
            if tot >= 5:
                per_car.append(hit / tot)
        mix = np.asarray(sf.read(DS10 / "foa" / f"{clip}.flac")[0], np.float64).T
        lv = frame_spl_a(mix[0], 24000)
        fires = m12c.fire_events(p, lv)
        fires_by_ncars[ncars].append(sum(1 for _, c, _ in fires if c == CAR))
        if row["w1_class"]:
            warn_n += 1
            wci = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
                   "crossing": 5}[row["w1_class"]]
            warn_ok += int(any(c == wci for _, c, _ in fires))

    core11 = list(csv.DictReader(open(DS11 / "plan" / "assignment_core.csv")))
    val_caronly = [r["clip_id"] for r in core11
                   if r["split"] == "fold2" and r["n_warnings"] == "0"
                   and r["n_car"] == "1"]
    pred_val = load_pred_multi(PRED / "val_all.csv")
    for clip in val_caronly:
        mix = np.asarray(sf.read(DS11 / "foa" / f"{clip}.flac")[0], np.float64).T
        lv = frame_spl_a(mix[0], 24000)
        fires_by_ncars[1].append(
            sum(1 for _, c, _ in m12c.fire_events(pred_val[clip], lv) if c == CAR))

    rep = ["# v10a 交通量・複数車 採点（v11 run1。2026-07-28）",
           "（1台対照はv11 valの n_car==1 & n_warnings==0"
           f" {len(val_caronly)}本。v10a本体はv10評価アセット=ビット同一）", ""]
    rep += ["## 1. 複数車の同時検出（フレーム単位）",
            f"- 2台同時ラベルのフレーム: 両方同時に報告 {simu[2][0]}/{simu[2][1]} "
            f"({simu[2][0]/max(simu[2][1],1):.1%})",
            f"- 3台同時: 2台以上 {simu[3][0]/max(simu[3][2],1):.1%} / 3台 "
            f"{simu[3][1]/max(simu[3][2],1):.1%}",
            f"- 車ごとの方向つきフレームカバー率（±30°）: "
            f"中央値 {np.median(per_car):.1%} / p10 {np.percentile(per_car, 10):.1%} "
            f"（n={len(per_car)}台）", ""]
    rep.append("## 2. 車通知の発火回数分布（10秒あたり、ルールv1・不応期5s）")
    rep.append("")
    rep.append("| 台数 | クリップ数 | 発火0回 | 1回 | 2回 | 3回以上 | 平均 |")
    rep.append("| --- | --- | --- | --- | --- | --- | --- |")
    for n in (1, 2, 3):
        f = np.array(fires_by_ncars[n])
        rep.append(f"| {n}台 | {len(f)} | {(f == 0).mean():.0%} | "
                   f"{(f == 1).mean():.0%} | {(f == 2).mean():.0%} | "
                   f"{(f >= 3).mean():.0%} | {f.mean():.2f} |")
    f1 = np.array(fires_by_ncars[1])
    f23 = np.array(fires_by_ncars[2] + fires_by_ncars[3])
    rep += ["",
            f"- 交通量モード候補「10秒窓で2回以上」: 交通量(2-3台)成立 {(f23 >= 2).mean():.0%}、"
            f"1台での誤成立 {(f1 >= 2).mean():.0%}", ""]
    rep += ["## 3. 交通量下の警告音①の生存",
            f"- 警告音ありの交通クリップ {warn_n}本中、警告音の通知 {warn_ok}本 "
            f"({warn_ok/max(warn_n,1):.0%})", ""]
    (OUT / "v10a_summary.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))

    # ---- 4-6. scn2 / halluc / probe（CLI差し替え） ----
    py = sys.executable
    jobs = [
        ("4/6 scn2", [py, str(ROOT / "scripts" / "step15_scn2_score.py"),
                      "--ds", "dataset_outdoor_siren_v10",
                      "--pred", "predictions_v11/scn2_all.csv",
                      "--out", "step12_notify_v11",
                      "--title", "追加5シナリオ 採点（v11 run1、ルールv1。2026-07-28）"]),
        ("5/6 halluc", [py, str(ROOT / "scripts" / "step17_halluc_score.py"),
                        "--pred", "out/predictions_v11",
                        "--out", "out/step12_notify_v11",
                        "--title", "v11 run1"]),
        ("6/6 probe", [py, str(ROOT / "scripts" / "step18_probe_score.py"),
                       "--ds", "out/dataset_outdoor_siren_v10",
                       "--pred", "out/predictions_v11",
                       "--out", "out/step12_notify_v11",
                       "--title", "v11 run1"]),
    ]
    for name, cmd in jobs:
        print(f"=== {name} ===", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        assert r.returncode == 0, (name, r.returncode)
    print("\nALL SCORING DONE ->", OUT)


if __name__ == "__main__":
    main()
