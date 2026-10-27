# -*- coding: utf-8 -*-
"""共通クラスでの集計（変更票 #73・Astra R73-04）: 正解（metadata_dist）と予測（7 列 csv）を、指定したクラスだけに絞った写しを作る。
採点器（統一採点器・知覚層・step20）は変えず、絞った写しを渡して「相手と共通のクラスだけの成績」を出す。9 クラス全体の成績とは別に併記する。

使い方:
  python scripts/_filter_classes.py --classes 0,1,3,4,7,8 --meta-in <metadata_dist> --meta-out <dir> [--pred-in <csv> --pred-out <csv>] [--manifest <clip_id の名簿>]
  python scripts/_filter_classes.py --selftest
決まり: 行の class（2 列目）が集合に無ければ落とす。本（clip）は残す（行が 0 になった本は「正解なし」の本として扱われる）。名簿があればその本だけ。出力に記録（json）を添える。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_classes(s: str) -> set[int]:
    cs = {int(x) for x in s.split(",") if x.strip() != ""}
    if not cs or any(c < 0 or c > 8 for c in cs):
        raise SystemExit(f"クラスの集合が変: {s}")
    return cs


def filter_meta(meta_in: Path, meta_out: Path, classes: set[int], clips: list[str] | None) -> dict:
    meta_out.mkdir(parents=True, exist_ok=True)
    names = [f"{c}.csv" for c in clips] if clips is not None else sorted(p.name for p in meta_in.glob("*.csv"))
    n_in = n_out = 0
    for name in names:
        src = meta_in / name
        if not src.is_file():
            raise FileNotFoundError(src)
        kept = []
        for line in src.read_text(encoding="utf-8").splitlines():
            q = line.strip().split(",")
            if len(q) < 2 or not q[0]:
                continue
            n_in += 1
            if int(float(q[1])) in classes:
                kept.append(line.rstrip("\n") + "\n"); n_out += 1
        (meta_out / name).write_text("".join(kept), encoding="utf-8")
    return {"n_files": len(names), "rows_in": n_in, "rows_out": n_out}


def filter_pred(pred_in: Path, pred_out: Path, classes: set[int], clips: list[str] | None) -> dict:
    keep = set(clips) if clips is not None else None
    n_in = n_out = 0
    with open(pred_in, encoding="utf-8") as f, open(pred_out, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            q = line.strip().split(",")
            if len(q) < 3 or not q[0]:
                continue
            n_in += 1
            if keep is not None and q[0] not in keep:
                continue
            if int(float(q[2])) in classes:
                g.write(line.rstrip("\n") + "\n"); n_out += 1
    return {"rows_in": n_in, "rows_out": n_out, "sha256_out": hashlib.sha256(pred_out.read_bytes()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes"); ap.add_argument("--meta-in", type=Path); ap.add_argument("--meta-out", type=Path)
    ap.add_argument("--pred-in", type=Path); ap.add_argument("--pred-out", type=Path); ap.add_argument("--manifest", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    classes = parse_classes(a.classes)
    clips = None
    if a.manifest:
        with open(a.manifest, encoding="utf-8", newline="") as f:
            clips = [r["clip_id"] for r in csv.DictReader(f)]
    rec = {"classes": sorted(classes)}
    if a.meta_in:
        rec["meta"] = filter_meta(a.meta_in, a.meta_out, classes, clips)
        (a.meta_out / "_filter_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.pred_in:
        rec["pred"] = filter_pred(a.pred_in, a.pred_out, classes, clips)
        a.pred_out.with_suffix(".filter_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False))
    return 0


def selftest() -> int:
    import tempfile
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="filtcls_"))
    mi = tmp / "meta"; mi.mkdir()
    (mi / "a.csv").write_text("0,4,0,10,0,2.0\n1,6,0,20,0,3.0\n2,0,0,30,0,5.0\n", encoding="utf-8")
    (mi / "b.csv").write_text("0,2,0,10,0,2.0\n", encoding="utf-8")
    pred = tmp / "p.csv"; pred.write_text("a,0,4,0,10,0,2.0\na,1,6,0,20,0,3.0\nb,0,2,0,10,0,2.0\nc,0,4,0,10,0,2.0\n", encoding="utf-8")
    man = tmp / "m.csv"; man.write_text("clip_id\na\nb\n", encoding="utf-8")
    r = filter_meta(mi, tmp / "mo", {0, 1, 3, 4, 7, 8}, ["a", "b"])
    ok.append(("正解: a は 2 行残り（4・0）・b は 0 行（2 = バックブザーは対象外）・本は残る", r["rows_in"] == 4 and r["rows_out"] == 2 and (tmp / "mo/b.csv").is_file() and (tmp / "mo/b.csv").read_text() == ""))
    r2 = filter_pred(pred, tmp / "po.csv", {0, 1, 3, 4, 7, 8}, ["a", "b"])
    ok.append(("予測: 名簿の本だけ・クラス 6 と 2 を落とす → 1 行", r2["rows_in"] == 4 and r2["rows_out"] == 1 and (tmp / "po.csv").read_text() == "a,0,4,0,10,0,2.0\n"))
    try:
        parse_classes("4,9"); ok.append(("クラス 9 は拒否", False))
    except SystemExit:
        ok.append(("クラス 9 は拒否", True))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
