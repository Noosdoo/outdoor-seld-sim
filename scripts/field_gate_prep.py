# -*- coding: utf-8 -*-
"""field_gate_prep.py — 採点の門の前に、実録の読みを正解づくり用に整える（2026-10-27 Opus・#69 r49・本人 10/27 12:35 (c)・13:09 (d)・14:13 案 A）。
デモの組（demo_field_unused_chain.py）で入れた形を、本番の学内の本の段（step19l → step19k → field_score_members）でも同じに使うための道具。
この道具は読みの表（csv・json）だけを読み書きする。録音・動画・推論・採点には触れない。本番の本に掛けるのは採点の門（本人の合図）のあと。

  1. 読みを足す（step19l の前）:
     - ベルの本（events の class = bike_bell）: 前輪の読み（cls bike_bell）を cls bicycle としても写す（ベルを鳴らす自転車は実際に走っている → 自転車の走行音の正解も付ける・(d)）。
       step19l がふつうの自転車の本と同じに車軸の中点へずらす（session の備考の軸の間）。
     - 走りながらのクラクション E3（events の class = horn で、台本ID が E3- か、向きと線がある本）: 前輪の読み（cls horn）を cls car_drive としても写す（車の走行音の正解）。
       horn の行は step19l がホーンの位置へ前にずらす（備考「ホーンの位置: 前輪の軸から前へ N cm」）。
     - 止めた車の E2（向きも線も無い horn）は動画の読みが無いので何もしない（step19m の読みのまま）。
  2. 警告音の正解の区間の表（step19k の --warn-interval・take,class,on_s,end_s・全長の録音の秒）:
     - ベルの本と E3: 本人が波形で読んだ鳴り始め（b = bell_onset_s）と鳴り終わり（e = bell_end_s）（audio_reads_<人>.json・E3 も同じ鍵）。
     - E2: step19m の reads_fixed（押しの始まりと終わりの 2 点の t）の最小〜最大。
     読みが無い・鳴り終わり ≤ 鳴り始めの本があれば止める（一覧を出す・黙って通過の区間のまま使わない）。
使い方:
  python scripts/field_gate_prep.py --reads <reads_A.csv> --audio-reads <audio_reads_A.json> --events <events.csv> [--reads-fixed <reads_fixed.csv>] \
      --out-reads <reads_A_gate.csv> --out-warn <warn_interval.csv>
  そのあと: step19l_when_where.py --reads <reads_A_gate.csv> … → field_score_members.py … --warn-interval <warn_interval.csv>
終了コード: 0 = できた・2 = 読み待ちの本がある・入力の誤り（何も書かない）
試験: scripts/test_field_gate_prep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WARN_COLS = ["take", "class", "on_s", "end_s"]


def read_csv(p: Path) -> list:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _name(s: str) -> str:
    return Path(str(s or "").strip()).name


def is_moving_horn(e: dict) -> bool:
    """走りながらのクラクション E3: class = horn で、台本ID が E3- か、向きと線がある（止めた E2 は向きも線も空）。"""
    if (e.get("class") or "").strip() != "horn":
        return False
    sid = (e.get("台本ID") or e.get("条件ID") or "").strip().upper()
    if sid.startswith("E3-"):
        return True
    return bool((e.get("direction") or "").strip() and (e.get("line_m") or "").strip()) and not sid.startswith("E2-")


def take_sets(events: list) -> tuple[set, set]:
    bell = {_name(e.get("原本")) for e in events if (e.get("class") or "").strip() == "bike_bell" and _name(e.get("原本"))}
    e3 = {_name(e.get("原本")) for e in events if is_moving_horn(e) and _name(e.get("原本"))}
    return bell, e3


def add_reads(reads: list, bell: set, e3: set) -> tuple[list, dict]:
    """ベルの本の bike_bell の行 → bicycle の写し・E3 の horn の行 → car_drive の写し（元の行は残す）。戻り = (足した後の行, 数)。"""
    extra, n = [], {"bicycle_from_bell": 0, "car_from_e3": 0}
    for r in reads:
        t = _name(r.get("take"))
        c = (r.get("cls") or "").strip().lower()
        if t in bell and c == "bike_bell":
            extra.append(dict(r, cls="bicycle")); n["bicycle_from_bell"] += 1
        elif t in e3 and c == "horn":
            extra.append(dict(r, cls="car_drive")); n["car_from_e3"] += 1
    return reads + extra, n


def warn_intervals(bell: set, e3: set, aud: dict, fixed: list | None) -> tuple[list, list]:
    """戻り = (区間の行, 読み待ち・誤りの一覧)。"""
    rows, bad = [], []
    for take, cls in sorted([(t, "bike_bell") for t in bell] + [(t, "horn") for t in e3]):
        a = aud.get(take, {})
        on, end = a.get("bell_onset_s"), a.get("bell_end_s")
        if on is None or end is None:
            bad.append(f"{take}（{'ベル' if cls == 'bike_bell' else '走りながらのクラクション'}）: 鳴り始め（b）か鳴り終わり（e）の読みが無い"); continue
        if float(end) <= float(on):
            bad.append(f"{take}: 鳴り終わり {float(end):.3f} s が鳴り始め {float(on):.3f} s 以前"); continue
        rows.append({"take": take, "class": cls, "on_s": f"{float(on):.4f}", "end_s": f"{float(end):.4f}"})
    if fixed:
        span = {}
        for r in fixed:
            if (r.get("cls") or "").strip() != "horn" or not (r.get("t") or "").strip():
                continue
            t = _name(r.get("take")); v = float(r["t"])
            lo, hi = span.get(t, (v, v)); span[t] = (min(lo, v), max(hi, v))
        for t, (lo, hi) in sorted(span.items()):
            if t in e3:
                continue
            if hi <= lo:
                bad.append(f"{t}（E2）: 読みの始まりと終わりが同じ"); continue
            rows.append({"take": t, "class": "horn", "on_s": f"{lo:.4f}", "end_s": f"{hi:.4f}"})
    return rows, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", type=Path, required=True)
    ap.add_argument("--audio-reads", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--reads-fixed", type=Path, default=None)
    ap.add_argument("--out-reads", type=Path, required=True)
    ap.add_argument("--out-warn", type=Path, required=True)
    a = ap.parse_args(argv)
    reads = read_csv(a.reads)
    if reads and "converted" in reads[0]:
        print("止める: --reads が step19l の変換済み（converted の列がある）。step19h の生の読みを渡す（この道具は step19l の前）"); return 2
    events = read_csv(a.events)
    aud = json.loads(a.audio_reads.read_text(encoding="utf-8"))
    fixed = read_csv(a.reads_fixed) if a.reads_fixed else None
    bell, e3 = take_sets(events)
    out_reads, n = add_reads(reads, bell, e3)
    warn, bad = warn_intervals(bell, e3, aud, fixed)
    # Astra R142-04（10/27）: ベルと E3 の本には、写す元の前輪の読み（bike_bell・horn）が要る（無ければ走行音の正解も音源の位置も作れない）
    have = {(_name(r.get("take")), (r.get("cls") or "").strip().lower()) for r in reads}
    bad += [f"{t}（ベル）: 前輪の読み（cls bike_bell）が無い" for t in sorted(bell) if (t, "bike_bell") not in have]
    bad += [f"{t}（走りながらのクラクション）: 前輪の読み（cls horn）が無い" for t in sorted(e3) if (t, "horn") not in have]
    if bad:
        print(f"止める: 読み待ち・誤り {len(bad)} 本（何も書かない）")
        for b in bad:
            print("  " + b)
        return 2
    cols = list(reads[0].keys()) if reads else ["take", "frame", "cls", "track", "x", "y", "z", "t"]
    for p, rows, cs in ((a.out_reads, out_reads, cols), (a.out_warn, warn, WARN_COLS)):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cs); w.writeheader(); w.writerows(rows)
    print(f"ベルの本 {len(bell)}・走りながらのクラクション {len(e3)}・足した読み 自転車 {n['bicycle_from_bell']} 行／車 {n['car_from_e3']} 行 -> {a.out_reads}")
    print(f"警告音の区間 {len(warn)} 本（ベル {sum(r['class'] == 'bike_bell' for r in warn)}・クラクション {sum(r['class'] == 'horn' for r in warn)}） -> {a.out_warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
