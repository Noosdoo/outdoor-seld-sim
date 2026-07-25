# -*- coding: utf-8 -*-
"""step10_v9_2_ctrl2_plan.py — v9.2均衡対照（ctrl2）の割当表。

設計= md/design/v9_2_ctrl2_design_2026-07-22.md。
通常クリップ270本（fold1_room4、車1台/本=車イベント270台）。行の構造式は
v9.2の旧ctrl（step10_v9_plan.py build_v92のctrl部）と同一で、本数を180→270に
拡大しただけ（警告0/1/2=81/149/40 ≒ 30/55/15%）。

出力: out/dataset_outdoor_siren_v9_2_ctrl2/plan/assignment_ctrl2.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step10_v9_plan as v9  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v9_2_ctrl2" / "plan"

# +50000はv9.2(20260717系)とv10.2(20260722系)が使用済みのためオフセットを70000に変更
GLOBAL_SEED = 20260723
OFFSET = 70000
N = 270
N_W = [0] * 81 + [1] * 149 + [2] * 40   # ≒30/55/15%


def build_ctrl2() -> list:
    base = GLOBAL_SEED * 613 + OFFSET
    rows = []
    for i in range(N):
        nw = N_W[i]
        w1 = v9.WARN_CLASSES[i % 5] if nw >= 1 else ""
        w2 = v9.WARN_CLASSES[(i + 2) % 5] if nw == 2 else ""
        if w2 and w2 == w1:
            w2 = v9.WARN_CLASSES[(i + 3) % 5]
        rows.append({"clip_id": f"fold1_room4_mix{i+1:03d}", "split": "fold1",
                     "motion": "static" if i % 2 == 0 else "walk",
                     "n_warnings": nw,
                     "w1_class": w1, "w1_side": v9.SIDES[i % 2] if w1 else "",
                     "w2_class": w2, "w2_side": v9.SIDES[(i + 1) % 2] if w2 else "",
                     "danger_tier": v9.TIERS[i % 3],
                     "car_side": v9.SIDES[(i // 3) % 2],
                     "scenario": "normal", "seed": base + i})
    return rows


def load_all_existing_seeds() -> set:
    seeds = set()
    for plan_dir in [ROOT / "out" / "dataset_outdoor_siren_v9" / "plan",
                     ROOT / "out" / "dataset_outdoor_siren_v9_2_add" / "plan",
                     ROOT / "out" / "dataset_outdoor_siren_v10" / "plan",
                     ROOT / "out" / "dataset_outdoor_siren_v10_2_add" / "plan"]:
        for f in sorted(plan_dir.glob("assignment_*.csv")):
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    seeds.add(int(r["seed"]))
    return seeds


def main() -> int:
    r1, r2 = build_ctrl2(), build_ctrl2()
    c1, c2 = v9.to_csv(r1), v9.to_csv(r2)
    assert hashlib.md5(c1.encode()).hexdigest() == \
        hashlib.md5(c2.encode()).hexdigest(), "determinism check failed"

    assert len(r1) == N
    assert sum(1 for r in r1 if r["n_warnings"] == 0) == 81
    assert sum(1 for r in r1 if r["n_warnings"] == 1) == 149
    assert sum(1 for r in r1 if r["n_warnings"] == 2) == 40
    assert all(r["car_side"] for r in r1), "車1台/本が前提"
    assert all(r["scenario"] == "normal" for r in r1)
    assert all(not r["w2_class"] or r["w2_class"] != r["w1_class"] for r in r1)
    ids = [r["clip_id"] for r in r1]
    seeds = [r["seed"] for r in r1]
    assert len(set(ids)) == len(ids) and len(set(seeds)) == len(seeds)
    existing = load_all_existing_seeds()
    overlap = set(seeds) & existing
    assert not overlap, f"seed collision: {sorted(overlap)[:5]}"

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / "assignment_ctrl2.csv").write_text(c1, encoding="utf-8")
    md5 = hashlib.md5(c1.encode()).hexdigest()
    rep = ["# v9.2 ctrl2（均衡対照）割当表 検算レポート", "",
           f"- GLOBAL_SEED={GLOBAL_SEED} offset={OFFSET} / md5={md5}",
           f"- {N}本・車イベント{N}台（1台/本）・警告0/1/2=81/149/40",
           f"- 既存全plan（v9/v9_2_add/v10/v10_2_add、{len(existing)}シード）との衝突ゼロ",
           "- 決定論: 2回構築でmd5一致 / ID・seed一意", ""]
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(rep) + "\n",
                                                   encoding="utf-8")
    print("\n".join(rep))
    print("ALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
