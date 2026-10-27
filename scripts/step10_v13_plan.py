# -*- coding: utf-8 -*-
"""step10_v13_plan.py — v13（合成データ修正の束）の割当表。

設計= md/design/v13設計書_合成データ修正の束_2026-09-02.md §1-2。
v11 core（fold1/fold2）＋ v12ext（fold1/fold2）の行を**そのまま土台**にし（clip_id・seed・
クラス・危険層・n_car・場面は不変）、seedハッシュで決定論的に:
  S7: motion を 静止30:歩行70 に再割当（旧: 50:50）
  R1: 20% のクリップに雨（light/moderate/heavy = 50/35/15、dB(A)= U(45,50)/U(50,58)/U(60,68) 仮置き）
  S5: grammar='v13' 列（文法の版）
fold3（test）は載せない（生成しない）。fold20 は無関係。

出力: out/dataset_outdoor_siren_v13/plan/assignment_v13.csv ＋ plan_ledger_v13.md
使い方: python scripts/step10_v13_plan.py
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

SRC_CORE = ROOT / "out/dataset_outdoor_siren_v11/plan/assignment_core.csv"
SRC_EXT = ROOT / "out/dataset_outdoor_siren_v12/plan/assignment_v12ext.csv"
OUT_DIR = ROOT / "out/dataset_outdoor_siren_v13/plan"

WALK_FRAC = 0.70                 # S7
RAIN_FRAC = 0.20                 # R1
RAIN_KIND_P = {"light": 0.50, "moderate": 0.35, "heavy": 0.15}                       # 既定は旧。本番は params（step10_v2final_plan／step10_v27_plan の _p13 が上書き・点検 K9）
RAIN_DBA = {"light": (45.0, 50.0), "moderate": (50.0, 58.0), "heavy": (60.0, 68.0)}   # ⚠️旧の仮置き（9/2）。本番は params の D2_RAIN_DBA（#70 r13: 34〜58／45〜61／49〜64・佐藤ら 2020 図 9）
RAIN_RATE = {"light": 150.0, "moderate": 600.0, "heavy": 2500.0}                       # rain.INTENSITY
SALT = 20260913


def assign(row: dict) -> dict:
    rng = np.random.default_rng([int(row["seed"]), SALT])
    r = dict(row)
    r["motion"] = "walk" if rng.random() < WALK_FRAC else "static"
    if rng.random() < RAIN_FRAC:
        kinds, ps = zip(*RAIN_KIND_P.items())
        kind = str(rng.choice(kinds, p=ps))
        r["rain"] = kind
        r["rain_rate"] = f"{RAIN_RATE[kind]:.0f}"
        r["rain_dba"] = f"{rng.uniform(*RAIN_DBA[kind]):.1f}"
    else:
        r["rain"], r["rain_rate"], r["rain_dba"] = "", "", ""
    r["grammar"] = "v13"
    return r


sys.path.insert(0, str(ROOT / "src"))
from outdoor_seld.clip_ids import split_to_new, to_new  # noqa: E402  （2026-09-22 改名: 土台の plan が旧名でも新名でも新名で出す）


def main() -> int:
    rows = []
    for src in (SRC_CORE, SRC_EXT):
        with open(src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] in ("fold1", "fold2", "train", "val"):
                    r["clip_id"], r["split"] = to_new(r["clip_id"]), split_to_new(r["split"])
                    rows.append(assign(r))
    cols = list(rows[0].keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "assignment_v13.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    L = ["# v13 plan台帳（自動生成: step10_v13_plan.py）", "",
         f"土台= v11 core + v12ext の train/val（旧 fold1/fold2。clip_id・seed・クラス・危険層・n_car・場面は不変。名前は 2026-09-22 の規則 <用途>_<出所>_clip<番号>）。",
         f"再割当= motion（歩行{WALK_FRAC:.0%}）・雨（{RAIN_FRAC:.0%}）を seed×{SALT} で決定論的に。", ""]
    for split in ("train", "val"):
        rs = [r for r in rows if r["split"] == split]
        L += [f"## {split}（{len(rs):,}本）", "",
              f"- motion: {dict(Counter(r['motion'] for r in rs))}",
              f"- rain: {dict(Counter(r['rain'] or 'none' for r in rs))}",
              f"- scenario: {dict(Counter(r['scenario'] for r in rs))}",
              f"- n_car: {dict(sorted(Counter(r['n_car'] for r in rs).items()))}",
              f"- 警告クラス: {dict(Counter([r['w1_class'] for r in rs if r['w1_class']] + [r['w2_class'] for r in rs if r['w2_class']]))}",
              ""]
    (OUT_DIR / "plan_ledger_v13.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", OUT_DIR / "assignment_v13.csv", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
