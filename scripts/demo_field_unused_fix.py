# -*- coding: utf-8 -*-
"""開発用の学内の本（台帳で使わないの本・デモの組）の固定記録（Astra R5-30-04・2026-10-27 Opus）。

採点の前に、最終の一覧（md/design/demo_field_unused_list_2026-10-26.csv・19 本）を入口にして、次のものを 1 つの読み取り専用の json にまとめる:
  - 一覧・台帳（ledger・events_fixed・session_fixed）・道具の版（準備・この道具・step19h・step19d・変換・step19k・calibration）の sha
  - 日付ごとのファイルの sha: 本人の読みの状態（reads_A.state.json）・録音側の拍手とベル（audio_reads_A.json）・reads_A.csv・sync_A.csv・
    除外区間メモ.csv・読む表（ann_demo_video_takes.csv）・events_demo.csv・sessions_demo.csv・index.csv
  - 本ごとの値: clip_id・区分・条件の写し元・向きの直し・線の上書き・変換した音の sha・その本の読みの行の sha（行を並べ替えてから）・
    目印の数（相手・歩く人）・拍手（音と動画の秒）・録音側の拍手とベルの鳴り始め・除外区間（①・入り・出）
--check（既定）は今のファイルを記録と比べ、**本ごとの値の違い**（意味のある変化）と**ファイル全体の sha の違い**（並びや書式だけの変化も含む）を分けて出す。
--write は記録が無ければ作る。記録と違うときは --reason が要り、履歴に理由と旧新の sha を残して書き直す（理由なしでは書かない・終了コード 3）。
記録はファイルの読み取り専用の印を付けて置く。以後の段（切り出し・正解・推論・採点）は、この記録の 19 本だけを入口にし（fixed_takes）、
フォルダの中を全部拾う方式にしない（Astra R5-30-03）。一覧に無いファイルが置き場にあれば知らせる（使わない）。

使い方: python scripts/demo_field_unused_fix.py [--write --reason "…"] [--list …] [--base out/_demo_field_unused_149] [--record md/design/demo_field_unused_fixed_record.json]
終了コード: --check 0 = 一致・1 = 違いあり・2 = 記録が無い ／ --write 0 = 書いた（または同じ）・3 = 違うのに理由が無い
試験: scripts/test_demo_field_unused_fix.py
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEF_LIST = E2E / "md" / "design" / "demo_field_unused_list_2026-10-26.csv"
DEF_BASE = E2E / "out" / "_demo_field_unused_149"
DEF_RECORD = E2E / "md" / "design" / "demo_field_unused_fixed_record.json"
DATE_FILES = (("reads", "reads_A.state.json"), ("reads", "audio_reads_A.json"), ("reads", "reads_A.csv"), ("reads", "sync_A.csv"),
              (".", "除外区間メモ.csv"), (".", "ann_demo_video_takes.csv"), (".", "events_demo.csv"), (".", "sessions_demo.csv"), (".", "index.csv"))
TOOLS = ("scripts/demo_field_unused_prepare.py", "scripts/demo_field_unused_fix.py", "scripts/step19h_video_marks.py", "scripts/step19d_field_csv_to_ann.py",
         "scripts/step19_realsmoke_convert.py", "scripts/step19k_field_perception.py", "src/outdoor_seld/calibration.py",
         "scripts/field_exclusion.py")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p) -> str:
    return sha_bytes(Path(p).read_bytes())


def read_csv(p) -> list:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build(list_path: Path, base: Path, ledger_dir: Path, root: Path = E2E) -> dict:
    """今のファイルから記録の中身（made_at・history を除く）を作る。一覧の本だけを見る。"""
    lst = read_csv(list_path)
    by = {}
    for r in lst:
        by.setdefault(r["日付"].strip(), []).append(r)
    out = {"list": {"path": list_path.relative_to(root).as_posix() if list_path.is_relative_to(root) else str(list_path), "sha256": sha(list_path), "n": len(lst)},
           "tools": {t: (sha(root / t) if (root / t).is_file() else None) for t in TOOLS},
           "index_json_sha256": sha(base / "index.json") if (base / "index.json").is_file() else None,
           "dates": {}, "takes": [], "stray": []}
    for d, rows in sorted(by.items()):
        dd = base / d
        files = {n: (sha(dd / sub / n) if (dd / sub / n).is_file() else None) for sub, n in DATE_FILES}
        led = {k: (sha(ledger_dir / f"{k}_{d}.csv") if (ledger_dir / f"{k}_{d}.csv").is_file() else None) for k in ("ledger", "events_fixed", "session_fixed")}
        out["dates"][d] = {"files": files, "raw_ledger": led}
        idx = {r["原本"]: r for r in read_csv(dd / "index.csv")} if (dd / "index.csv").is_file() else {}
        reads = read_csv(dd / "reads" / "reads_A.csv") if (dd / "reads" / "reads_A.csv").is_file() else []
        sync = {r["take"]: r for r in read_csv(dd / "reads" / "sync_A.csv")} if (dd / "reads" / "sync_A.csv").is_file() else {}
        aud = json.loads((dd / "reads" / "audio_reads_A.json").read_text(encoding="utf-8")) if (dd / "reads" / "audio_reads_A.json").is_file() else {}
        memo = {r["原本"]: r for r in read_csv(dd / "除外区間メモ.csv")} if (dd / "除外区間メモ.csv").is_file() else {}
        listed = set()
        for r in rows:
            o = r["原本"].strip()
            ix = idx.get(o, {})
            conv = dd / (ix.get("conv") or f"conv/{Path(o).stem}_conv.flac")
            listed.add(conv.name)
            rr = sorted(tuple(x[k] for k in sorted(x)) for x in reads if x.get("take") == o)
            m = memo.get(o)
            out["takes"].append({
                "clip_id": r["clip_id"].strip(), "日付": d, "原本": o, "区分": r["区分"].strip(), "台本ID": r["台本ID"].strip(),
                "条件の写し元": ix.get("条件の写し元"), "条件の写し元の台本ID": (r.get("条件の写し元の台本ID") or "").strip() or r["台本ID"].strip(),
                "向きの直し_deg": r["向きの直し_deg"].strip(), "線の上書き_m": (r.get("線の上書き_m") or "").strip(),
                "conv": conv.relative_to(base).as_posix() if conv.is_relative_to(base) else str(conv), "conv_sha256": sha(conv) if conv.is_file() else None,
                "reads_rows_sha256": sha_bytes(canon(rr).encode("utf-8")), "n_marks": len([x for x in reads if x.get("take") == o and x.get("cls") != "mic"]),
                "n_marks_mic": len([x for x in reads if x.get("take") == o and x.get("cls") == "mic"]),
                "sync": {k: sync.get(o, {}).get(k) for k in ("audio_s", "video_s")},
                "audio_reads": aud.get(o, {}),
                "除外区間": ({"①": m.get("自転車が最後まで通った（○／×）", ""), "入り_音の秒": m.get("除外の入り_音の秒", ""), "出_音の秒": m.get("除外の出_音の秒", "")} if m else None)})
        if (dd / "conv").is_dir():
            out["stray"] += [f"{d}/conv/{p.name}" for p in sorted((dd / "conv").glob("*.flac")) if p.name not in listed]
    return out


def compare(old: dict, new: dict) -> tuple[list, list]:
    """戻り = (本ごとの値の違い, ファイル全体の sha などの違い)。"""
    per, whole = [], []
    ot = {t["clip_id"]: t for t in old.get("takes", [])}
    nt = {t["clip_id"]: t for t in new.get("takes", [])}
    for c in sorted(set(ot) | set(nt)):
        if c not in ot or c not in nt:
            per.append(f"{c}: 一覧の本が {'増えた' if c in nt else '減った'}")
            continue
        for k in sorted(set(ot[c]) | set(nt[c])):
            if ot[c].get(k) != nt[c].get(k):
                per.append(f"{c}（{nt[c]['原本']}）の {k}: {ot[c].get(k)} → {nt[c].get(k)}")
    for k in ("list", "tools", "index_json_sha256"):
        if old.get(k) != new.get(k):
            whole.append(f"{k}: {canon(old.get(k))[:120]} → {canon(new.get(k))[:120]}")
    for d in sorted(set(old.get("dates", {})) | set(new.get("dates", {}))):
        if old.get("dates", {}).get(d) != new.get("dates", {}).get(d):
            o, n = old.get("dates", {}).get(d, {}), new.get("dates", {}).get(d, {})
            for g in ("files", "raw_ledger"):
                for f in sorted(set(o.get(g, {})) | set(n.get(g, {}))):
                    if o.get(g, {}).get(f) != n.get(g, {}).get(f):
                        whole.append(f"{d} の {f}: sha {str(o.get(g, {}).get(f))[:12]} → {str(n.get(g, {}).get(f))[:12]}")
    return per, whole


def content_of(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("made_at", "history", "rule", "content_sha256")}


def fixed_takes(record: Path = DEF_RECORD) -> list:
    """以後の段の入口: 記録の 19 本だけを返す（フォルダの中を拾わない）。"""
    return json.loads(Path(record).read_text(encoding="utf-8"))["takes"]


def write_record(path: Path, rec: dict) -> None:
    if path.exists():
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IREAD)                                            # 読み取り専用の印


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=DEF_LIST)
    ap.add_argument("--base", type=Path, default=DEF_BASE)
    ap.add_argument("--record", type=Path, default=DEF_RECORD)
    ap.add_argument("--ledger-dir", type=Path, default=E2E / "raw" / "ledger")
    ap.add_argument("--root", type=Path, default=E2E, help="試験用（道具の sha を取る根）")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    new = build(a.list.resolve(), a.base.resolve(), a.ledger_dir.resolve(), a.root.resolve())
    if new["stray"]:
        print(f"[知らせ] 一覧に無い変換音が置き場にある（使わない・以後の段は記録の 19 本だけ）: {new['stray']}")
    if not a.record.exists():
        if not a.write:
            print(f"記録が無い: {a.record}（--write で作る）")
            return 2
        rec = {"rule": "開発用の学内の本（台帳で使わないの本）の固定記録（Astra R5-30-04）。本番（評価の本）は入れない。以後の段はこの記録の本だけを入口にする",
               "made_at": datetime.datetime.now().isoformat(timespec="seconds"), **new, "history": []}
        rec["content_sha256"] = sha_bytes(canon(content_of(rec)).encode("utf-8"))
        write_record(a.record, rec)
        print(f"記録を作った: {a.record}（{len(new['takes'])} 本・content {rec['content_sha256'][:12]}）")
        return 0
    old = json.loads(a.record.read_text(encoding="utf-8"))
    per, whole = compare(content_of(old), new)
    if not per and not whole:
        print(f"一致: {len(new['takes'])} 本・本ごとの値もファイル全体の sha も記録と同じ（content {old.get('content_sha256', '')[:12]}）")
        return 0
    print(f"本ごとの値の違い（意味のある変化）: {len(per)} 件")
    for x in per[:40]:
        print("  " + x)
    print(f"ファイル全体の sha などの違い（並びや書式だけの変化も含む）: {len(whole)} 件")
    for x in whole[:40]:
        print("  " + x)
    if not a.write:
        return 1
    if not a.reason.strip():
        print("❌ 記録と違うので、書き直すには --reason が要る（理由と旧新の sha を履歴に残す）。書かなかった")
        return 3
    body = {**new}
    new_sha = sha_bytes(canon(body).encode("utf-8"))
    hist = list(old.get("history", [])) + [{"at": datetime.datetime.now().isoformat(timespec="seconds"), "reason": a.reason.strip(),
                                             "old_content_sha256": old.get("content_sha256"), "new_content_sha256": new_sha,
                                             "per_take_changes": per[:200], "whole_changes": whole[:200]}]
    rec = {"rule": old.get("rule"), "made_at": datetime.datetime.now().isoformat(timespec="seconds"), **new, "history": hist, "content_sha256": new_sha}
    write_record(a.record, rec)
    print(f"記録を書き直した（理由: {a.reason.strip()}・{old.get('content_sha256', '')[:12]} → {new_sha[:12]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
