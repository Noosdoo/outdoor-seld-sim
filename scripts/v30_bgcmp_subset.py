# -*- coding: utf-8 -*-
"""背景の形の小さな比べ（決め直し 5・本人 10/26 20:03・仕様 §9-7 の 2）の学習用の部分集合を選ぶ。描かない。一覧を先に固定してコミットする。

選び方（決まった手順・種 2026）:
  - plan（out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv・v30 も同じ plan）の train の行だけ（val・評価の本は使わない）
  - 高さの双子（同じ seed の h_copy 1／2）は両方を一緒に選ぶ（単位 = 組・train は全部 2 本の組）
  - scenario の割合を train 全体と同じにする: 組の数 6,000（= 12,000 本）を scenario ごとの組の数に比例して配る（最大剰余法・同点は scenario 名の順）
  - scenario の中は組を random.Random(2026) で並べ替えて先頭から取る（scenario 名の順に処理）
出力: md/design/v30_bgcmp_本の一覧_2026-10-26.csv（列 clip_id,scenario,seed,h_copy・plan の順）と、標準出力に sha256・本数・scenario の内訳
使い方: python scripts/v30_bgcmp_subset.py --out md/design/v30_bgcmp_本の一覧_2026-10-26.csv [--n-rows 12000]
試験: scripts/test_v30_bgcmp.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
SEED = 2026
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pairs_of(rows: list[dict]) -> dict:
    """train の行を seed の組に（h_copy 1／2 の 2 本）。組が 2 本でなければ止める。"""
    by = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by[r["seed"]].append(r)
    bad = {s: len(v) for s, v in by.items() if len(v) != 2 or {x["h_copy"] for x in v} != {"1", "2"} or len({x["scenario"] for x in v}) != 1}
    if bad:
        raise SystemExit(f"高さの双子の組になっていない seed がある: {list(bad.items())[:3]}（{len(bad)} 組）")
    return by


def allocate(counts: dict, total: int) -> dict:
    """組の数を scenario に比例して配る（最大剰余法・同点は名前の順）。"""
    n = sum(counts.values())
    raw = {k: total * v / n for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rest = total - sum(base.values())
    for k in sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))[:rest]:
        base[k] += 1
    return base


def select(rows: list[dict], n_rows: int = 12000, seed: int = SEED) -> list[dict]:
    assert n_rows % 2 == 0, "高さの双子の組で選ぶので偶数"
    by = pairs_of(rows)
    sc_pairs = defaultdict(list)
    for s, v in by.items():
        sc_pairs[v[0]["scenario"]].append(s)
    alloc = allocate({k: len(v) for k, v in sc_pairs.items()}, n_rows // 2)
    rng = random.Random(seed)
    take = set()
    for sc in sorted(sc_pairs):
        seeds = sorted(sc_pairs[sc], key=int)
        rng.shuffle(seeds)
        take |= set(seeds[:alloc[sc]])
    return [r for r in rows if r["split"] == "train" and r["seed"] in take]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-rows", type=int, default=12000)
    ap.add_argument("--plan", type=Path, default=PLAN)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"一覧が既にある: {a.out}（固定した一覧は上書きしない）")
    rows = list(csv.DictReader(open(a.plan, encoding="utf-8")))
    sel = select(rows, a.n_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_id", "scenario", "seed", "h_copy"])
        w.writerows([[r["clip_id"], r["scenario"], r["seed"], r["h_copy"]] for r in sel])
    tr = Counter(r["scenario"] for r in rows if r["split"] == "train")
    sc = Counter(r["scenario"] for r in sel)
    print(f"{len(sel)} 本 → {a.out}・sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    print("scenario（部分集合／train 全体）: " + "・".join(f"{k} {sc[k]}/{tr[k]}" for k in sorted(tr)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
