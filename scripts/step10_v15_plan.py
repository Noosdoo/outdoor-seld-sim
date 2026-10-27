# -*- coding: utf-8 -*-
"""v15 plan = v14 plan ＋ D13「マイクの高さを実録に合わせる」（2026-09-04・本人「実測した高さに学習データを合わせましょう」）。

背景: 実録はヘルメット装着（身長 183 cm → マイク中心 約 2.0〜2.1 m）。学習データは固定 1.5 m。高さが違うと
至近の 3D 距離が +0.2〜0.4 m、仰角が −20°→−35° に変わる（機材監査 §7）。実機はネックバンドで 1.4〜1.5 m。
→ 学習データのマイク高さを行ごとに引く。既定は 1.4〜2.1 m の一様（実機とヘルメットの両方を覆う）。
   初日リハで実測したら --fixed <m> か --range a,b で作り直す（seed × SALT15 で決定論的）。

変更はマイク高さ（mic_z 列）だけ。横距離・危険層・速度・雨・文法・至近低速（v14）は不変。
⚠️ 危険層（critical 0.6〜1.5 m など）は 1.5 m のマイクで引いた 3D 最接近なので、マイクが高いほど実際の 3D 最接近は大きくなる
   （横距離は同じ）。通知規則の閾値は 3D なので、装着高さが変わるなら「規則の入力を水平距離にする」検討が要る（設計案 D13 の注）。

使い方: python scripts/step10_v15_plan.py [--range 1.4,2.1 | --fixed 2.05]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_V14 = ROOT / "out/dataset_outdoor_siren_v14/plan/assignment_v14.csv"
OUT_DIR = ROOT / "out/dataset_outdoor_siren_v15/plan"
SALT15 = 20260915
MIC_Z_RANGE = (1.4, 2.1)     # 既定。実測後に差し替える


def main() -> int:
    a = sys.argv
    fixed = float(a[a.index("--fixed") + 1]) if "--fixed" in a else None
    rng_lo, rng_hi = MIC_Z_RANGE
    if "--range" in a:
        rng_lo, rng_hi = (float(x) for x in a[a.index("--range") + 1].split(","))
    with open(SRC_V14, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        rng = np.random.default_rng([int(r["seed"]), SALT15])
        z = fixed if fixed is not None else float(rng.uniform(rng_lo, rng_hi))
        r["mic_z"] = f"{z:.3f}"
        r["grammar"] = "v15"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "assignment_v15.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    zs = np.array([float(r["mic_z"]) for r in rows])
    L = ["# v15 plan台帳（自動生成: step10_v15_plan.py）", "",
         "土台= v14 plan（v13 の文法・雨・歩行 70%・至近低速 D10 は不変）。",
         ("D13 マイク高さ: 固定 " + f"{fixed:.2f} m" if fixed is not None else f"D13 マイク高さ: {rng_lo}〜{rng_hi} m の一様（seed×{SALT15}）")
         + f"。平均 {zs.mean():.2f} m、最小 {zs.min():.2f}、最大 {zs.max():.2f}", ""]
    for split in ("fold1", "fold2"):
        n = sum(1 for r in rows if r["split"] == split)
        L.append(f"- {split}: {n:,} 本")
    (OUT_DIR / "README_plan_v15.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", OUT_DIR / "assignment_v15.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
