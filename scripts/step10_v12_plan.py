# -*- coding: utf-8 -*-
"""step10_v12_plan.py — v12追加行のplan生成（既存7,200行は完全据置）。

設計= md/design/v12設計書_2026-08-05.md 追記2/3。追加は3シナリオ:
  v12kick : キックボード（特定小型原付: 車道≤20km/h・歩道≤6km/h）
  v12bike : バイク（原付=法定30km/h / 軽・小型二輪=市街地30〜60km/h）
  v12train: 第4種踏切の列車通過（複数点音源・警笛50%）
規模: fold1(学習) kick900+bike900+train600 / fold2(val) kick225+bike225+train150
出力: out/dataset_outdoor_siren_v12/plan/assignment_v12ext.csv + plan_ledger_v12.md
seeds: 9.1G帯の新規系列（既存planと排他なことを検証して書き出す）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN_V12 = ROOT / "out" / "dataset_outdoor_siren_v12" / "plan"
FIELDS = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side",
          "w2_class", "w2_side", "danger_tier", "car_side", "scenario", "seed",
          "scene_type", "n_car"]

SPEC = [  # (fold, scenario, 件数, mix開始番号)
    ("fold1", "v12kick", 900, 4801),
    ("fold1", "v12bike", 900, 5701),
    ("fold1", "v12train", 600, 6601),
    ("fold2", "v12kick", 225, 1201),
    ("fold2", "v12bike", 225, 1426),
    ("fold2", "v12train", 150, 1651),
]
SEED_BASE = 9_100_000_000


def main() -> None:
    # 既存seedとの排他検証
    import step11_v11_render as m11v
    m9 = m11v.m9
    existing = {r["seed"] for r in m9.load_plan("core")}

    PLAN_V12.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for fold, scen, count, start in SPEC:
        for j in range(count):
            seed = SEED_BASE + idx * 7919
            assert seed not in existing
            rows.append({
                "clip_id": f"{fold}_room1_mix{start + j:04d}",
                "split": fold,
                "motion": "walk" if (idx % 2) else "static",   # 50:50決定論
                "n_warnings": 0, "w1_class": "", "w1_side": "",
                "w2_class": "", "w2_side": "", "danger_tier": "",
                "car_side": "", "scenario": scen, "seed": seed,
                "scene_type": "v12ext", "n_car": 0,
            })
            idx += 1

    out_csv = PLAN_V12 / "assignment_v12ext.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # 員数台帳（3,342会計の教訓: 生成時に自動出力）
    ledger = ["# v12 plan台帳（自動生成: step10_v12_plan.py）", "",
              "| 区分 | シナリオ | 本数 | clip_id範囲 |", "|---|---|---|---|",
              "| 本体(v11 core据置) | v11core | 7,200 | fold1 mix0001-4800 / fold2 0001-1200 / fold3 0001-1200 |"]
    for fold, scen, count, start in SPEC:
        ledger.append(f"| 追加 | {scen} | {count} | {fold} mix{start:04d}-{start+count-1:04d} |")
    total = 7200 + sum(c for _, _, c, _ in SPEC)
    ledger += ["", f"**合計 {total:,} 本**（学習 {4800+2400:,} / val {1200+600:,} / "
               "test 1,200 素のまま）", "",
               "評価専用の拡充（大型車600・長距離300・キックボード300・バイク300・"
               "第4種踏切150）は本体生成・基準線の後に別planで追加する（本台帳に追記）。"]
    (PLAN_V12 / "plan_ledger_v12.md").write_text("\n".join(ledger), encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows) + ledger. total={total:,}")


if __name__ == "__main__":
    main()
