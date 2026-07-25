# -*- coding: utf-8 -*-
"""step10_v10_2_ctrl_plan.py — v10.2均衡対照（2本）の割当表。

設計= md/design/v10_2_ctrl_design_2026-07-22.md（本人指示「対象元も対象先も合わせて
公平な比較を」→クリップ数一致と車イベント数一致の2対照で挟み撃ち）:

  ctrlclip: 通常675本 (fold1_room5) — クリップ数をv10.2追加分と一致（車675台）
  ctrlev  : 通常1,012本 (fold1_room3) — 車イベント数をv10.2追加分と一致（車1,012台）

行の構造式はv9.2旧ctrlと同一（警告0/1/2≒30/55/15%、車1台/本）。
出力: out/dataset_outdoor_siren_v10_2ctrl_add/plan/assignment_{ctrlclip,ctrlev}.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step10_v9_plan as v9  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v10_2ctrl_add" / "plan"

GLOBAL_SEED = 20260723
ARMS = {
    # name: (room, n, offset, n_w分布[0個,1個,2個])
    "ctrlev": ("fold1_room3", 1012, 80000, (304, 556, 152)),
    "ctrlclip": ("fold1_room5", 675, 90000, (203, 371, 101)),
}


def build_arm(room: str, n: int, offset: int, nw_split: tuple) -> list:
    base = GLOBAL_SEED * 613 + offset
    n_w = [0] * nw_split[0] + [1] * nw_split[1] + [2] * nw_split[2]
    assert len(n_w) == n
    rows = []
    for i in range(n):
        nw = n_w[i]
        w1 = v9.WARN_CLASSES[i % 5] if nw >= 1 else ""
        w2 = v9.WARN_CLASSES[(i + 2) % 5] if nw == 2 else ""
        if w2 and w2 == w1:
            w2 = v9.WARN_CLASSES[(i + 3) % 5]
        rows.append({"clip_id": f"{room}_mix{i+1:03d}", "split": "fold1",
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
    for name in ["dataset_outdoor_siren_v9", "dataset_outdoor_siren_v9_2_add",
                 "dataset_outdoor_siren_v9_2_ctrl2", "dataset_outdoor_siren_v10",
                 "dataset_outdoor_siren_v10_2_add"]:
        for f in sorted((ROOT / "out" / name / "plan").glob("assignment_*.csv")):
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    seeds.add(int(r["seed"]))
    return seeds


def main() -> int:
    existing = load_all_existing_seeds()
    # v10.2追加分の車イベント数（plan実測）
    v10_2 = list(csv.DictReader(open(
        ROOT / "out" / "dataset_outdoor_siren_v10_2_add" / "plan"
        / "assignment_v10_2add.csv")))
    cars_v10_2 = sum((2 if r["scenario"] == "traffic2" else
                      3 if r["scenario"] == "traffic3" else
                      1 if r["car_side"] else 0) for r in v10_2)
    assert cars_v10_2 == 1012 and len(v10_2) == 675

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    rep = ["# v10.2 均衡対照（2本）割当表 検算レポート", "",
           f"- GLOBAL_SEED={GLOBAL_SEED} / v10.2追加分の実測: 675本・車{cars_v10_2}台", ""]
    for name, (room, n, offset, nw_split) in ARMS.items():
        r1 = build_arm(room, n, offset, nw_split)
        r2 = build_arm(room, n, offset, nw_split)
        c1 = v9.to_csv(r1)
        assert hashlib.md5(c1.encode()).hexdigest() == \
            hashlib.md5(v9.to_csv(r2).encode()).hexdigest(), f"{name}: 決定論NG"
        assert len(r1) == n
        for k in (0, 1, 2):
            assert sum(1 for r in r1 if r["n_warnings"] == k) == nw_split[k]
        assert all(r["car_side"] for r in r1)
        assert all(r["scenario"] == "normal" for r in r1)
        assert all(not r["w2_class"] or r["w2_class"] != r["w1_class"] for r in r1)
        seeds = [r["seed"] for r in r1]
        assert len(set(seeds)) == len(seeds)
        overlap = set(seeds) & existing
        assert not overlap, f"{name}: seed collision {sorted(overlap)[:5]}"
        existing |= set(seeds)   # 相互衝突もここで検出される
        all_rows += r1
        (PLAN_DIR / f"assignment_{name}.csv").write_text(c1, encoding="utf-8")
        md5 = hashlib.md5(c1.encode()).hexdigest()
        rep.append(f"- {name}: {room} {n}本・車{n}台・警告0/1/2="
                   f"{nw_split[0]}/{nw_split[1]}/{nw_split[2]}・offset={offset}・md5={md5}")
    ids = [r["clip_id"] for r in all_rows]
    assert len(set(ids)) == len(ids)
    rep += ["- ctrlclip: クリップ数675=v10.2と一致 / ctrlev: 車1012台=v10.2と一致",
            "- 既存全plan＋相互のシード衝突ゼロ / 決定論md5一致 / ID一意", ""]
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(rep) + "\n",
                                                   encoding="utf-8")
    print("\n".join(rep))
    print("ALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
