# -*- coding: utf-8 -*-
"""test_clip_ids.py — 識別子の対応関数の検査（2026-09-22）。
  往復の同一性（to_old(to_new(x)) == x）・新名の形・5 桁・対象外の旧名を拒む・plan の全行で to_new が単射（v2final 24,000 行・v16 18,000 行・v12 manifest）。
使い方: python scripts/test_clip_ids.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld import clip_ids as C  # noqa: E402

bad = []


def check(cond, msg):
    if not cond:
        bad.append(msg)


# 1. 形と往復
for old, new in [("fold1_room1_mix0001", "train_syn_clip00001"), ("fold1_room1_mix28028", "train_syn_clip28028"),
                 ("fold2_room1_mix10001", "val_syn_clip10001"), ("fold20_room1_mix1800", "eval_syn_clip01800"),
                 ("fold2_room1_mix0007", "val_syn_clip00007")]:
    check(C.to_new(old) == new, f"to_new({old}) = {C.to_new(old)} ≠ {new}")
    check(C.to_old(new) == old, f"to_old({new}) = {C.to_old(new)} ≠ {old}")
    check(C.to_new(new) == new and C.compat(old) == new, f"compat/idempotent: {old}")
    check(C.is_new(new) and not C.is_new(old) and C.is_old(old), f"is_new/is_old: {old} {new}")
    check(C.number(old) == C.number(new), f"number: {old}")
check(C.make("eval", "field", 1) == "eval_field_clip00001", "make field")
check(C.parse("eval_field_clip00001") == ("eval", "field", 1), "parse field")
check(C.make_field(3, 12) == "eval_field_clip03012" and C.parse_field("eval_field_clip03012") == (3, 12), "make_field/parse_field")
check(C.make_field(1, 1) == "eval_field_clip01001" and C.make_field(99, 999) == "eval_field_clip99999", "make_field 範囲")
for bad_args in ((0, 1), (100, 1), (1, 0), (1, 1000)):
    try:
        C.make_field(*bad_args); bad.append(f"make_field が範囲外を受け付けた: {bad_args}")
    except AssertionError:
        pass
for s in ("fold3_room1_mix0001", "fold10_room1_mix0001", "fold1_room2_mix0001", "train_syn_clip1", "foo"):
    try:
        C.to_new(s)
        bad.append(f"拒むべき名前を受け付けた: {s}")
    except ValueError:
        pass
try:
    C.to_old("eval_field_clip00001")
    bad.append("実録の新名に旧名を作った")
except ValueError:
    pass
check(C.split_to_new("fold1") == "train" and C.split_to_new("val") == "val", "split_to_new")

# 2. plan の全行で単射・split の整合
plans = [ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv",
         ROOT / "out/dataset_outdoor_siren_v16/plan/assignment_v16.csv",
         ROOT / "out/dataset_outdoor_siren_v12/plan/manifest_val_syn.csv"]
for p in plans:
    if not p.exists():
        print(f"skip（無い）: {p}")
        continue
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["clip_id"] for r in rows]
    news = []
    for r in rows:
        n = C.to_new(r["clip_id"])
        news.append(n)
        if "split" in r:
            check(C.parse(n)[0] == C.split_to_new(r["split"]), f"{p.name}: {r['clip_id']} の split {r['split']} と名前が合わない")
    check(len(set(news)) == len(set(ids)) == len(ids), f"{p.name}: 新名が重複（{len(ids)} 行 → {len(set(news))} 種）")
    # 往復: 旧名の plan なら to_old(to_new(x)) == x、新名の plan なら to_new(to_old(x)) == x
    check(all((C.to_old(n) == o) if C.is_old(o) else (C.to_new(C.to_old(n)) == n) for n, o in zip(news, ids)), f"{p.name}: 往復で戻らない行がある")
    print(f"{p.name}: {len(ids):,} 行 OK（例 {ids[0]} → {news[0]}、{ids[-1]} → {news[-1]}）")

print("=" * 60)
print("合格" if not bad else "不合格")
for b in bad:
    print("  NG", b)
sys.exit(1 if bad else 0)
