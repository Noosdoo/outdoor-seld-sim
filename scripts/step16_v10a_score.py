# -*- coding: utf-8 -*-
"""Step 16: v10a 交通量・複数車の採点＋交通量モードのしきい値設計。

出すもの:
  1. 複数車の同時検出（フレーム単位: GT台数 vs 予測台数、方向つきカバー率）
  2. 車通知の発火頻度分布: 1台(valの車のみクリップ) vs 2台 vs 3台
     → 「交通量モード」切替ルールの実データ設計
  3. 交通量下でも警告音①の通知が生きるか
出力: out/step12_notify_v9_1/v10a_summary.md
"""
from __future__ import annotations

import csv
import importlib.util
import json
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

spec = importlib.util.spec_from_file_location(
    "step12", ROOT / "scripts" / "step12_notify_v9.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

DS = ROOT / "out" / "dataset_outdoor_siren_v9_1"
PRED_DIR = ROOT / "out" / "predictions_v9_1"
OUT = ROOT / "out" / "step12_notify_v9_1"
CAR = 4
FPS = 10


def load_labels_tracked(clip):
    """{frame: [(class, track, az, el)]}"""
    out = defaultdict(list)
    for line in open(DS / "metadata" / f"{clip}.csv"):
        p = line.strip().split(",")
        if len(p) == 5:
            out[int(p[0])].append((int(p[1]), int(p[2]),
                                   float(p[3]), float(p[4])))
    return out


def load_pred_multi(path):
    """{clip: {frame: [(class, az, el)]}}（同一クラス複数行を保持）"""
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def car_fire_count(pred_clip, mix_level):
    return sum(1 for _, c, _ in m12.fire_events(pred_clip, mix_level) if c == CAR)


def main():
    pred = load_pred_multi(PRED_DIR / "v10a_all.csv")
    plan = list(csv.DictReader(open(DS / "plan" / "assignment_v10a.csv")))

    # ---- 1. 複数車の同時検出＋方向つきカバー率 ----
    simu = {2: [0, 0], 3: [0, 0, 0]}   # GT台数 -> [pred>=2できた, 総数] / 3台は[>=2,>=3,総数]
    per_car = []                        # (フレーム再現率) per car
    warn_ok, warn_n = 0, 0
    fires_by_ncars = defaultdict(list)
    for row in plan:
        clip = row["clip_id"]
        ncars = int(row["scenario"][-1])
        labels = load_labels_tracked(clip)
        p = pred[clip]
        # 同時検出
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
        # 方向つきカバー率（車ごと: 自分のラベルフレームで±30°以内の車予測があるか）
        scene = json.loads((DS / "work" / clip / "scene.json").read_text())
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
        # 発火頻度と警告音生存
        mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
        lv = frame_spl_a(mix[0], 24000)
        fires = m12.fire_events(p, lv)
        fires_by_ncars[ncars].append(
            sum(1 for _, c, _ in fires if c == CAR))
        if row["w1_class"]:
            warn_n += 1
            wci = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
                   "crossing": 5}[row["w1_class"]]
            warn_ok += int(any(c == wci for _, c, _ in fires))

    # ---- 2. 1台の対照: valの「車のみ」クリップ（n_warnings=0）----
    core = list(csv.DictReader(open(DS / "plan" / "assignment_core.csv")))
    val_caronly = [r["clip_id"] for r in core
                   if r["split"] == "fold2" and r["n_warnings"] == "0"]
    pred_val = load_pred_multi(PRED_DIR / "val_all.csv")
    for clip in val_caronly:
        mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
        fires_by_ncars[1].append(
            car_fire_count(pred_val[clip], frame_spl_a(mix[0], 24000)))

    rep = ["# v10a 交通量・複数車 採点（v9.1 run1。2026-07-18）", ""]
    rep += ["## 1. 複数車の同時検出（フレーム単位）→ **発見: 同一クラスは1台ずつしか報告されない**",
            f"- 2台同時ラベルのフレーム: 両方同時に報告 {simu[2][0]}/{simu[2][1]} "
            f"({simu[2][0]/max(simu[2][1],1):.1%})",
            f"- 3台同時: 2台以上 {simu[3][0]/max(simu[3][2],1):.1%} / 3台 "
            f"{simu[3][1]/max(simu[3][2],1):.1%}",
            "- **原因の切り分け済み**: デコーダ（multi_accdoa_to_dcase_format）は方位15°超なら"
            "別イベントとして書ける実装 → 書かれていない=モデルが同一クラス複数トラックを"
            "活性化していない。**v9.1の学習データは設計上車1台のため、同一クラス多重の"
            "教師例がゼロ**（異クラス同時85.9-100%は学習済みで良好、という対比）。"
            "mACCDOAの能力の問題ではなく学習分布の帰結",
            f"- 車ごとの方向つきフレームカバー率（±30°、逐次的にどれかは追えている）: "
            f"中央値 {np.median(per_car):.1%} / p10 {np.percentile(per_car, 10):.1%} "
            f"（n={len(per_car)}台）＝支配的な1台を追い、支配が入れ替わると乗り移る挙動", ""]

    rep.append("## 2. 車通知の発火回数分布（10秒あたり、ルールv1・不応期5s）")
    rep.append("")
    rep.append("| 台数 | クリップ数 | 発火0回 | 1回 | 2回 | 3回以上 | 平均 |")
    rep.append("| --- | --- | --- | --- | --- | --- | --- |")
    for n in (1, 2, 3):
        f = np.array(fires_by_ncars[n])
        rep.append(f"| {n}台 | {len(f)} | {(f == 0).mean():.0%} | "
                   f"{(f == 1).mean():.0%} | {(f == 2).mean():.0%} | "
                   f"{(f >= 3).mean():.0%} | {f.mean():.2f} |")
    f1, f23 = np.array(fires_by_ncars[1]), np.array(
        fires_by_ncars[2] + fires_by_ncars[3])
    rule_fp = (f1 >= 2).mean()
    rule_tp = (f23 >= 2).mean()
    rep += ["",
            "**交通量モード切替ルールの設計（この分布から）**:",
            f"- 候補「10秒窓で車通知2回以上→モードON」: 交通量(2-3台)での成立 "
            f"{rule_tp:.0%}、1台での誤成立 {rule_fp:.0%}",
            "- 実運用では窓を連続時間に伸ばす（例: 30秒窓で3回以上）ことで"
            "1台の誤成立をさらに下げられる（クリップ10秒の制約内での測定である点は明記）", ""]

    rep += ["## 3. 交通量下の警告音①の生存",
            f"- 警告音ありの交通クリップ {warn_n}本中、警告音の通知 {warn_ok}本 "
            f"({warn_ok/max(warn_n,1):.0%})", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v10a_summary.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
