# -*- coding: utf-8 -*-
"""v19 plan = 新 4 場面の割当表（2026-09-08）。v16 の列をそのまま使い、v16 の 18,000 行には触らない（追加分だけ別 csv）。

各場面 750 場面 × 2 高さ（v16 と同じ高さ増強）= 1,500 本。train 1,200 / val 300（5 場面に 1 つを val。旧 fold1/fold2）。
  clip_id: {train,val}_syn_clip{N}（2026-09-22 の規則。旧 fold{1,2}_room1_mix{N}）  N = 20001 + k·2000 + i（h_copy=1）、+1000（h_copy=2）。
  k = 場面番号（fast 0 / static 1 / turn 2 / platoon 3 / launch 4）
  seed: 9190000000 + k·100000 + i（同一場面の 2 高さは同じ seed = 同じ音源・軌跡）
  motion: 歩行 70% / 静止 30%（v13 の D4 と同じ比率）。danger_tier: critical/caution/safe を巡回。car_side: L/R を交互
  mic_z: 一様 1.4〜2.1 m（h_copy=1 は rng([seed, 20260908])、h_copy=2 は rng([seed, 20260918])）
  rain: 無し（試作は場面の効果だけを見る。採るときは v2 の雨の規則を掛ける）
使い方: python scripts/step10_v19_plan.py [--scene fast|static|turn|platoon|all] [--n-scenes 750]
  → out/dataset_outdoor_siren_v19/plan/assignment_v19_<scene>.csv と assignment_v19.csv（全場面）
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = ROOT / "out/dataset_outdoor_siren_v19/plan"
SCENES = ["fast", "static", "turn", "platoon", "launch",   # launch = 発進（2026-09-21・変更票 #36 の (b)・k=4 → mix 28001〜。既定は adopt=false）
          "kickfar"]                                        # kickfar = キックの遠い横距離 2.5〜8 m（2026-09-28・変更票 #42・k=5 → mix 30001〜）
COLUMNS = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side", "w2_class", "w2_side", "danger_tier",
           "car_side", "scenario", "seed", "scene_type", "n_car", "rain", "rain_rate", "rain_dba", "grammar",
           "close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db", "mic_z", "h_copy"]
MIX_BASE = 20001
MIX_STEP = 2000
SEED_BASE = 9190000000
SEED_STEP = 100000
SALT_H1, SALT_H2 = 20260908, 20260918
MIC_Z_RANGE = (1.4, 2.1)
WALK_FRAC = 0.7
N_CAR = {"fast": "1", "static": "0", "turn": "1", "platoon": "3", "launch": "1", "kickfar": "0"}
TIERS = ("critical", "caution", "safe")


def _arg(name, default):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def build(scene: str, n_scenes: int) -> list:
    k = SCENES.index(scene)
    rows = []
    for i in range(n_scenes):
        seed = SEED_BASE + k * SEED_STEP + i
        split = "val" if i % 5 == 4 else "train"          # 2026-09-22 改名: 旧 fold2/fold1
        rng_m = np.random.default_rng([seed, 7])
        motion = "walk" if rng_m.random() < WALK_FRAC else "static"
        base = {"split": split, "motion": motion, "n_warnings": "0", "w1_class": "", "w1_side": "", "w2_class": "",
                "w2_side": "", "danger_tier": TIERS[i % 3], "car_side": "L" if i % 2 else "R",
                "scenario": f"v19{scene}", "seed": str(seed), "scene_type": "v19", "n_car": N_CAR[scene],
                "rain": "", "rain_rate": "", "rain_dba": "", "grammar": "v19", "close_slow": "",
                "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": ""}
        for h, salt in ((1, SALT_H1), (2, SALT_H2)):
            r = dict(base)
            n = MIX_BASE + k * MIX_STEP + i + (1000 if h == 2 else 0)
            r["clip_id"] = f"{split}_syn_clip{n:05d}"
            r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*MIC_Z_RANGE)):.3f}"
            r["h_copy"] = str(h)
            rows.append(r)
    return rows


def write(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    scene = _arg("--scene", "all")
    n_scenes = int(_arg("--n-scenes", "750"))
    scenes = SCENES if scene == "all" else [scene]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L = ["# v19 plan 台帳（自動生成: step10_v19_plan.py）", ""]
    allrows = []
    for sc in scenes:
        rows = build(sc, n_scenes)
        write(OUT_DIR / f"assignment_v19_{sc}.csv", rows)
        allrows += rows
        c = Counter(r["split"] for r in rows)
        z = np.array([float(r["mic_z"]) for r in rows])
        L.append(f"- v19{sc}: {len(rows):,} 本（train {c['train']:,} / val {c['val']:,}）・mic_z 平均 {z.mean():.2f} m・"
                 f"歩行 {sum(r['motion'] == 'walk' for r in rows) / len(rows):.0%}・mix {rows[0]['clip_id'][-5:]}〜{rows[-1]['clip_id'][-5:]}")
    if scene == "all":
        write(OUT_DIR / "assignment_v19.csv", allrows)
    assert len({r["clip_id"] for r in allrows}) == len(allrows)
    (OUT_DIR / "README_plan_v19.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
