# -*- coding: utf-8 -*-
"""v12評価専用セット拡充のplan生成（設計= v12設計書§5 + 本人指示2026-08-06）。

4セット・計1,500本（すべて新fold番号=学習/val/testと不交差、員数台帳自動出力）:
  heavy600   : 大型車600 — v11core同型幾何で車を全て大型化（至近/選別のE系評価の常設化）
  far300     : 長距離接近300 — サイレン/車を100〜300mからスポーン（E1b先行・
               「聞こえる前」の検知距離測定。v12の都市背景+低音強化の土俵で）
  kick300    : キックボード300 — 基本150（単独接近）+苦手150（背景騒音上限側）
  bike300    : バイク300 — 基本150 + 苦手150（同上）
出力: out/dataset_outdoor_siren_v12_eval/plan/assignment_v12eval.csv + 台帳
シナリオ名: e_heavy / e_far / e_kick / e_bike（サンプラは step11_v12_render 側に追加予定）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN = ROOT / "out" / "dataset_outdoor_siren_v12_eval" / "plan"
FIELDS = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side",
          "w2_class", "w2_side", "danger_tier", "car_side", "scenario", "seed",
          "scene_type", "n_car"]
SPEC = [  # (fold, scenario, 件数, 開始mix, 補足)
    ("fold10", "e_heavy", 600, 1, "大型車・safe幾何(3.2-15m)常設E系評価"),
    ("fold11", "e_far", 300, 1, "100-300mスポーンの長距離接近"),
    ("fold12", "e_kick", 300, 1, "キックボード 基本150+苦手150"),
    ("fold13", "e_bike", 300, 1, "バイク 基本150+苦手150"),
]
SEED_BASE = 9_500_000_000


def main() -> None:
    import step11_v11_render as m11v
    existing = {r["seed"] for r in m11v.m9.load_plan("core")}
    v12ext = ROOT / "out" / "dataset_outdoor_siren_v12" / "plan" / "assignment_v12ext.csv"
    with open(v12ext, newline="") as f:
        existing |= {int(r["seed"]) for r in csv.DictReader(f)}

    PLAN.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for fold, scen, count, start, _ in SPEC:
        for j in range(count):
            seed = SEED_BASE + idx * 104729
            assert seed not in existing
            rows.append({
                "clip_id": f"{fold}_room1_mix{start + j:04d}",
                "split": fold, "motion": "walk" if (idx % 2) else "static",
                "n_warnings": 0, "w1_class": "", "w1_side": "", "w2_class": "",
                "w2_side": "", "danger_tier": "", "car_side": "",
                "scenario": scen, "seed": seed, "scene_type": "v12eval",
                "n_car": 0})
            idx += 1
    out_csv = PLAN / "assignment_v12eval.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ledger = ["# v12評価専用拡充 plan台帳（自動生成）", "",
              "| セット | シナリオ | fold | 本数 | 目的 |", "|---|---|---|---|---|"]
    for fold, scen, count, _, note in SPEC:
        ledger.append(f"| {scen} | {scen} | {fold} | {count} | {note} |")
    ledger += ["", f"**合計 {len(rows):,} 本**（学習・val・testのどのfoldとも不交差）",
               "", "サンプラ実装・生成・推論・採点は次バッチ（サンプラ仕様は本plan起票時の",
               "設計= md/design/v12設計書 §5 と本ファイルdocstringに従う）"]
    (PLAN / "eval_ledger_v12.md").write_text("\n".join(ledger), encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows) + ledger")


if __name__ == "__main__":
    main()
