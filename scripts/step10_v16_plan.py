# -*- coding: utf-8 -*-
"""v16 plan = v15 plan ＋ 高さ増強（各場面をもう 1 つの高さでも描く）— 2026-09-05。

背景（v15 宣言 §5.5〜5.6）: マイク高さ 1.4〜2.1 m を幅で学習した v15/v15c は、全体で距離誤差 10.6%（目標 9.6%）、
高さ固定の v15b は自分の高さでは 7.9% だが他の高さでは 19% と崩れる。ラベル定義（3D/水平）は原因でなく、
「帯あたりの学習例が薄い（3,000 本/帯）」が残る仮説。→ **同じ場面（同一 seed＝同じ音源・軌跡・雨・騒音）を
別の高さでもう 1 回描く**。高さ以外が同一の対を与えるので、モデルが「高さの違いだけ」を学びやすい（高さ増強）。

  - 元の 9,000 行（clip_id・seed・mic_z すべて v15 と同一。v15 の音とラベルは v16 でも同一）
  - 複製 9,000 行: clip_id = mix 番号 +10000（fold1_room1_mix0001 → fold1_room1_mix10001）、split・seed・他列は同一、
    mic_z だけ rng([seed, SALT16]) の一様 1.4〜2.1 m で引き直す。列 h_copy=2
  → 18,000 行（fold1 14,400 / fold2 3,600）。fold2 のうち mix≤9000 の 1,800 本は v15 val そのもの（主指標はここで測る）

使い方: python scripts/step10_v16_plan.py   → out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_V15 = ROOT / "out/dataset_outdoor_siren_v15/plan/assignment_v15.csv"
OUT_DIR = ROOT / "out/dataset_outdoor_siren_v16/plan"
SALT16 = 20260917
MIC_Z_RANGE = (1.4, 2.1)
OFFSET = 10000


def main() -> int:
    with open(SRC_V15, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    assert len(base) == 9000, len(base)
    rows = []
    for r in base:
        r = dict(r); r["h_copy"] = "1"; r["grammar"] = "v16"
        rows.append(r)
    sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import make, parse   # 2026-09-22 改名: 写しは番号 +OFFSET（train_syn_clip00001 → train_syn_clip10001）
    for r in base:
        split, source, n = parse(r["clip_id"])
        assert n < OFFSET, r["clip_id"]
        c = dict(r)
        c["clip_id"] = make(split, source, n + OFFSET)
        rng = np.random.default_rng([int(r["seed"]), SALT16])
        c["mic_z"] = f"{float(rng.uniform(*MIC_Z_RANGE)):.3f}"
        c["h_copy"] = "2"; c["grammar"] = "v16"
        rows.append(c)
    assert len({r["clip_id"] for r in rows}) == len(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "assignment_v16.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    z1 = np.array([float(r["mic_z"]) for r in rows if r["h_copy"] == "1"])
    z2 = np.array([float(r["mic_z"]) for r in rows if r["h_copy"] == "2"])
    L = ["# v16 plan台帳（自動生成: step10_v16_plan.py）", "",
         "土台= v15 plan（v14 の場面 ＋ D13 高さ 1.4〜2.1 m）。各場面を **2 つの高さ**で描く（同一 seed・高さだけ別）。",
         f"- 元 9,000 行: mic_z 平均 {z1.mean():.2f} m（v15 と同一）",
         f"- 複製 9,000 行（mix+{OFFSET}）: mic_z 平均 {z2.mean():.2f} m、seed×{SALT16}、|z2−z1| 中央値 {np.median(np.abs(z2 - z1)):.2f} m", ""]
    for split in ("train", "val"):
        n = sum(1 for r in rows if r["split"] == split)
        L.append(f"- {split}: {n:,} 本")
    (OUT_DIR / "README_plan_v16.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", OUT_DIR / "assignment_v16.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
