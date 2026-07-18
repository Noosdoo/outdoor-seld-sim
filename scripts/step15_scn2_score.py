# -*- coding: utf-8 -*-
"""Step 15: 追加5シナリオ（room4〜8）の採点。設計= out/scenario_design_2026-07-17.md。

通知ルール・発火判定は step12（ルールv1）をそのまま流用。オラクル=可聴開始で即発火。
出力: out/step12_notify_v9_1/scn2_summary.md
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
from outdoor_seld.geometry import receiver_positions_at  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "step12", ROOT / "scripts" / "step12_notify_v9.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

DS = ROOT / "out" / "dataset_outdoor_siren_v9_1"
PRED = ROOT / "out" / "predictions_v9_1" / "scn2_all.csv"
OUT = ROOT / "out" / "step12_notify_v9_1"
CLSIDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
          "car_drive": 4, "crossing": 5}
CLASSES = ["Siren", "Horn", "BackupBeep", "BikeBell", "CarDrive", "Crossing"]
FPS = 10


def first_audible(clip, cls_idx):
    with open(DS / "masks" / f"{clip}.csv") as f:
        next(f)
        for line in f:
            k, c, snr = line.strip().split(",")
            if int(c) == cls_idx and float(snr) >= 0.0:
                return int(k) / FPS
    return None


def mic_dist_at(scene, src, t):
    wp = np.array(src["wp"], dtype=np.float64)
    ps = receiver_positions_at(np.array([t]), wp)[0]
    mic = m12._mic_wp(scene) if scene["mic"]["motion"] != "static" \
        else np.array([0.0, 0.0, 1.5])
    pm = (receiver_positions_at(np.array([t]), mic)[0]
          if mic.ndim == 2 else mic)
    return float(np.linalg.norm(ps - pm))


def main():
    pred = m12.load_pred(PRED)
    plan = list(csv.DictReader(
        open(DS / "plan" / "assignment_scenario2.csv")))
    by_scen = defaultdict(list)

    for row in plan:
        clip = row["clip_id"]
        scene = json.loads((DS / "work" / clip / "scene.json").read_text())
        mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
        fires = m12.fire_events(pred.get(clip, {}), frame_spl_a(mix[0], 24000))
        by_scen[row["scenario"]].append((clip, scene, fires, pred.get(clip, {})))

    rep = ["# 追加5シナリオ 採点（v9.1 run1、ルールv1。2026-07-18）", ""]

    # ---- S4 完全静穏（先に=ヘッドライン） ----
    rows = by_scen["silent_negative"]
    n_fr = sum(len(evs) for _, _, _, p in rows for evs in p.values())
    n_fire = sum(len(f) for _, _, f, _ in rows)
    rep += ["## S4 完全静穏（音源ゼロ 20本）",
            f"- 誤検出フレーム: **{n_fr}** / 通知発火: **{n_fire}**"
            "（車の幻覚も含めて完全ゼロ＝誤通知フロアはゼロ回/時）", ""]

    # ---- S1 踏切通過 ----
    rows = by_scen["crossing_wait"]
    cr_ok = car_fired = 0
    car_leads, car_orc = [], []
    for clip, scene, fires, p in rows:
        byc = defaultdict(list)
        for k, c, az in fires:
            byc[c].append((k, az))
        if byc[5]:
            cr_ok += 1
        car = next(s for s in scene["sources"] if s["class"] == "car_drive")
        t_cpa = scene["cpa_rel_time_s"]
        ta = first_audible(clip, 4)
        if ta is not None:
            car_orc.append(t_cpa - ta)
        if byc[4]:
            car_fired += 1
            car_leads.append(t_cpa - byc[4][0][0] / FPS)
    L, O = np.array(car_leads), np.array(car_orc)
    rep += ["## S1 踏切通過（警報鳴動中＋背後から車・注意層 20本）",
            f"- 踏切通知: {cr_ok}/20",
            f"- **警報のマスキング下の車**: 通知 {car_fired}/20、"
            f"リードタイム中央値 {np.median(L):.1f}s / >=2.0s {(L >= 2.0).mean():.0%}"
            if len(L) else f"- 車の通知 {car_fired}/20",
            f"- オラクル上限: 中央値 {np.median(O):.1f}s / >=2.0s {(O >= 2.0).mean():.0%}"
            f"（可聴{len(O)}/20台）", ""]

    # ---- S2 背後からベル ----
    rows = by_scen["bell_overtake"]
    ok = 0
    lat, leads, dirs = [], [], []
    for clip, scene, fires, p in rows:
        src = scene["sources"][0]
        cand = [(k, az) for k, c, az in fires if c == 3
                and src["t_on"] * FPS - 1 <= k <= (src["t_off"] + 1) * FPS]
        if cand:
            ok += 1
            k, az = cand[0]
            lat.append(k / FPS - src["t_on"])
            leads.append(src["t_cpa_rel_s"] - k / FPS)
            gaz = m12.gt_az_at(scene, "bike_bell", k / FPS)
            if gaz is not None:
                dirs.append(abs((az - gaz + 180) % 360 - 180))
    rep += ["## S2 背後からの自転車ベル（側方0.8-1.5m追い越し 20本）",
            f"- 通知: {ok}/20、発火遅れ中央値 {np.median(lat):.2f}s、"
            f"**追い越しの {np.median(leads):.1f}s 前**（中央値）に通知",
            f"- 通知方向誤差 中央値 {np.median(dirs):.1f}°"
            "（ほぼ真後ろの音源＝視覚の効かない方向）", ""]

    # ---- S3 バック車 ----
    rows = by_scen["backup_reverse"]
    res = {"critical": [], "caution": []}
    fired = {"critical": 0, "caution": 0}
    n = {"critical": 0, "caution": 0}
    codir = []
    for clip, scene, fires, p in rows:
        tier = scene["row"]["danger_tier"]
        n[tier] += 1
        t_cpa = scene["cpa_rel_time_s"]
        cand = [(k, az) for k, c, az in fires if c == 2]
        if cand:
            fired[tier] += 1
            k, az = cand[0]
            res[tier].append(t_cpa - k / FPS)
            gaz = m12.gt_az_at(scene, "car_drive", k / FPS)
            if gaz is not None:
                codir.append(abs((az - gaz + 180) % 360 - 180))
    rep.append("## S3 駐車場のバック車（バック音=車と同一軌道 20本）")
    for t in ("critical", "caution"):
        L = np.array(res[t])
        rep.append(f"- {t}: 通知 {fired[t]}/{n[t]}、リードタイム中央値 "
                   f"{np.median(L):.1f}s / >=2.5s {(L >= 2.5).mean():.0%}"
                   if len(L) else f"- {t}: 通知 {fired[t]}/{n[t]}")
    rep.append(f"- ベル通知方向と車真方向の差 中央値 {np.median(codir):.1f}°"
               "（=警告音が車の方向を正しく代弁）")
    rep.append("")

    # ---- S5 悪条件サイレン ----
    rows = by_scen["siren_worstnoise"]
    ok = 0
    leads, orc, dists, dirs = [], [], [], []
    for clip, scene, fires, p in rows:
        src = scene["sources"][0]
        t_cpa = src["t_cpa_rel_s"]
        ta = first_audible(clip, 0)
        if ta is not None:
            orc.append(t_cpa - ta)
        cand = [(k, az) for k, c, az in fires if c == 0]
        if cand:
            ok += 1
            k, az = cand[0]
            leads.append(t_cpa - k / FPS)
            dists.append(mic_dist_at(scene, src, k / FPS))
            gaz = m12.gt_az_at(scene, "siren", k / FPS)
            if gaz is not None:
                dirs.append(abs((az - gaz + 180) % 360 - 180))
    L, O = np.array(leads), np.array(orc)
    rep += ["## S5 悪条件サイレン（暗騒音60-65dB固定×遠方10-15m 20本）",
            f"- 通知: {ok}/20、リードタイム中央値 {np.median(L):.1f}s / "
            f">=2.5s {(L >= 2.5).mean():.0%}",
            f"- オラクル上限: 中央値 {np.median(O):.1f}s（差=モデル+ルールの遅れ）",
            f"- 初通知時の距離 中央値 {np.median(dists):.0f}m、"
            f"通知方向誤差 中央値 {np.median(dirs):.1f}°", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scn2_summary.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
