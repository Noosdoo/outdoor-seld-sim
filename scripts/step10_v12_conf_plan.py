# -*- coding: utf-8 -*-
"""v12確定評価セット(conf)のplan生成（事前登録= md/design/v12確定評価セット_事前登録_2026-08-10.md）。

fold2の全1,800行（core1,200 + ext600）をテンプレートに、設計フィールドは完全保存、
clip_id/split/seedのみ差し替え（fold20・SEED_BASE=12G帯・全既存planと排他）。
出力: out/dataset_outdoor_siren_v12_conf/plan/assignment_v12conf.csv + 台帳
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN = ROOT / "out" / "dataset_outdoor_siren_v12_conf" / "plan"
FIELDS = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side",
          "w2_class", "w2_side", "danger_tier", "car_side", "scenario", "seed",
          "scene_type", "n_car"]
SEED_BASE = 12_000_000_000
STEP = 104729


def main() -> None:
    core = list(csv.DictReader(open(
        ROOT / "out" / "dataset_outdoor_siren_v11" / "plan" /
        "assignment_core.csv")))
    ext = list(csv.DictReader(open(
        ROOT / "out" / "dataset_outdoor_siren_v12" / "plan" /
        "assignment_v12ext.csv")))
    tmpl = [r for r in core if r["clip_id"].startswith("fold2_")] + \
           [r for r in ext if r["clip_id"].startswith("fold2_")]
    assert len(tmpl) == 1800, len(tmpl)

    # 既存シードとの排他（core/ext/eval全plan）
    existing = {int(r["seed"]) for r in core} | {int(r["seed"]) for r in ext}
    evalp = ROOT / "out" / "dataset_outdoor_siren_v12_eval" / "plan" / \
        "assignment_v12eval.csv"
    existing |= {int(r["seed"]) for r in csv.DictReader(open(evalp))}

    PLAN.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(tmpl):
        seed = SEED_BASE + i * STEP
        assert seed not in existing, seed
        r = dict(t)
        r["clip_id"] = f"fold20_room1_mix{i + 1:04d}"
        r["split"] = "fold20"
        r["seed"] = seed
        rows.append(r)

    out_csv = PLAN / "assignment_v12conf.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_core = sum(1 for r in rows[:1200])
    scen = {}
    for r in rows[1200:]:
        scen[r["scenario"]] = scen.get(r["scenario"], 0) + 1
    ledger = ["# v12確定評価セット plan台帳（自動生成・事前登録の実体）", "",
              f"- 総数 {len(rows):,}本（fold20_room1_mix0001-1800）",
              f"- core（fold2 coreの鏡像・新シード）: {n_core}本 = mix0001-1200",
              f"- ext（fold2 extの鏡像・新シード）: {scen} = mix1201-1800",
              f"- SEED_BASE={SEED_BASE:,} step={STEP}（core/ext/eval全planと排他をassert済み）",
              "- 設計フィールドはテンプレート行を完全保存（clip_id/split/seedのみ差替）",
              "- 事前登録: md/design/v12確定評価セット_事前登録_2026-08-10.md"]
    (PLAN / "conf_ledger.md").write_text("\n".join(ledger), encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows) + ledger; ext={scen}")


if __name__ == "__main__":
    main()
