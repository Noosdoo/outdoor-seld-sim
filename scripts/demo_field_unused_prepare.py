# -*- coding: utf-8 -*-
"""学内の実録で「使わない」にした本を、開発用のデモの組（家の 13 本の隣）に足す準備（2026-10-26 Opus・本人 22:02 の決定・Fable の仕様）。

本番（台帳で「本」の本＝評価に使う本）は決して入れない（本人 21:55「本番は決して入れないでくださいね」）。守り（どの段でも最初に通す）:
  1. 明示の一覧（--list の csv・列 = 日付,原本,台本ID,区分,向きの直し_deg,線の上書き_m,clip_id）だけを受ける。ワイルドカード（* ? [ ]）は一覧のパスにも原本名にも不可。
     raw/ を探して本を集めることはしない（一覧に書いた原本名だけを 1 本ずつ開く）。
  2. 一覧の本を 1 本ずつ台帳 raw/ledger/ledger_<日付>.csv で引く。次のどれかが 1 本でもあれば、終了コード 2 で止める（何も書かない）:
     台帳で「本」（評価に使う本）・役割が「使わない」で始まらない（校正・点検・台帳に無い）・台本ID が台帳の役割の文に無い（取り違え）・
     原本名の日付が一覧の日付と違う・events_fixed_<日付>.csv で条件ID が入っている（評価の本）・区分や向きの直しの値が決まりの外・clip_id の形や重複。
  3. 書くのは --out（既定 out/_demo_field_unused_149・名前は _demo_field_unused で始まること）の下だけ。raw/ の中は不可。

やること（--convert のときだけ・守りを通ったあと。無ければ守りの確かめだけ）:
  - セッションの校正（session_fixed の 校正原本・LAeq_dB・暗騒音区間_秒）から 149 の約束（v12_149）でゲインを 1 回求め
    （step19_realsmoke_convert.calibration_record）、本ごとに全長を 24 kHz の FOA に変換（convert_with_gain）→ <out>/<日付>/conv/<原本の stem>_conv.flac。
    向きの直し_deg は変換の yaw（10/22 の歩く比較の 1 回目〔マイクが 180° 逆向きの間〕だけ 180・縦軸まわりの回転＝上下は同じという前提・069 と
    ベルの方位で確かめた。本人 22:28: 向きを確かめられた本は「入れる」に上げる＝「入れる」でも 180 がありうる）。
    歩行の本は step19k が歩く向きから正面を決めて三脚の向きの欄を使わないので、向きは正解側ではなく音の側で直す（静止も同じ扱いにそろえる）。
  - 正解を作るための条件の行 <out>/<日付>/events_demo.csv: 同じ原本の events の行があればそれ、無ければ同じ台本ID の評価の本の行の条件を写す
    （原本・動画を差し替え・拍手とラップ秒と条件ID は空・線の上書き）。評価の本の音には触らない（表の値を写すだけ）。
  - 本人が読む表（step19d --video-only → <out>/<日付>/ann_demo_video_takes.csv・step19h の入力）。セッションの表は sessions_demo.csv（除外の一覧を外す・用途 = 開発用デモ）。
  - <out>/<日付>/index.csv（本ごと: clip_id・区分・使わない理由・条件の写し元・ゲイン・約束・向きの直し・sha256）と <out>/index.json（台帳・一覧・events・session・道具の sha256）。
10 秒の切り出し（前輪が x = 0 を通った秒を 8 秒目）・推論・採点は、本人の読み（step19h）のあと（別の段で足す）。

一覧の任意の列 条件の写し元の台本ID（本人 22:41）: 実際の走り方が台本と違った本は、実際に近い台本の評価の本の条件を写す（空なら台本ID）。
区分「入れる（除外区間あり）」（本人 22:56 案 A）: 車などが通ってやり直した本。本人が <out>/<日付>/除外区間メモ.csv に ①自転車が最後まで通ったか ②車などが画面にいた
  時間（入りと出・step19h の画面の「音 … s」の秒）を書き、採点ではその時間は**どのクラスも採点しない（自転車も・本人 23:05 案 1）**＝除外区間の外だけで
  入れるの本と同じ指標を測る（マイクのそばの声のような大きい音で自転車が埋もれた時間を、モデルの成績に数えない）。①が × なら参考に下げる。
  メモの表は無いときだけ作る（作り直しでも本人の書いた値を上書きしない）。
一覧の任意の列 読みの備考（2026-10-26 Fable 22:3x）: 読む表の 備考（step19h の画面の一番上に出る）・events の 特記・index に写す。
  例: W03 の 1 回目（080・081）は自転車が台本と逆の右を通った見込み → 動画で実際に通った線を確かめる（違えば 線の上書き_m に入れて作り直し、step19h --export で書き出し直す）。
使い方: python scripts/demo_field_unused_prepare.py --list md/design/demo_field_unused_list_2026-10-26.csv [--convert] [--out out/_demo_field_unused_149]
試験: scripts/test_demo_field_unused_prepare.py
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CONVENTION = "v12_149"
WILD = re.compile(r"[*?\[\]]")
ORIG_RE = re.compile(r"^(\d{6})_(\d{3})\.WAV$")
CLIP_RE = re.compile(r"^tune_field_clip8\d{4}$")
KUBUN = ("入れる", "入れる（除外区間あり）", "参考", "参考（向き）")   # 入れる（除外区間あり）= 本人 22:56 案 A・23:05 案 1: 車や声が入った時間（除外区間）はどのクラスも採点しない（自転車も）
MEMO_COLS = ("原本", "clip_id", "動画", "区分", "自転車が最後まで通った（○／×）", "除外の入り_音の秒", "除外の出_音の秒", "メモ")
COLS = ("日付", "原本", "台本ID", "区分", "向きの直し_deg", "線の上書き_m", "clip_id")
CLEAR = ("ラップ秒", "拍手_音側秒", "拍手_動画側秒", "拍手2_音側秒", "拍手2_動画側秒", "拍手_録音側秒", "終わり秒", "条件ID")
OUT_PREFIX = "_demo_field_unused"
RULE = ("本番（台帳で「本」＝評価に使う本）は入れない（本人 21:55）。一覧に書いた原本だけ・台帳の役割が「使わない」の本だけ。"
        "評価の本が 1 本でもあれば終了コード 2 で何も書かない。評価の本からは条件の表の値だけを写す（音には触らない）")


class GuardError(Exception):
    """守りで止める（終了コード 2・何も書かない）。"""


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p) -> list:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_ledger(ledger_dir: Path, d: str) -> dict:
    """台帳 → {録音名: 行}。録音名は一意が前提（Astra R5-30-03・10/27）: 辞書にする前に重複を数え、1 つでもあれば理由によらず止める
    （同じ録音名で「本」と「使わない」が並ぶと、後の行が前の行を消して評価の本を見落とすため）。"""
    p = ledger_dir / f"ledger_{d}.csv"
    if not p.is_file():
        raise GuardError(f"台帳が無い: {p}")
    rows = [r for r in read_csv(p) if (r.get("録音") or "").strip() and not r["録音"].startswith("#")]
    names = [r["録音"].strip() for r in rows]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise GuardError(f"台帳 {p.name} に同じ録音名が 2 行以上ある（理由によらず止める・R5-30-03）: {dup[:5]}")
    return {r["録音"].strip(): r for r in rows}


def load_events(ledger_dir: Path, d: str) -> list:
    """events_fixed_<日付>.csv。条件ID の確かめ（評価の本の検出）と条件の写しに使うので、ファイルが無いときは空の表として許さず止める（R5-30-03）。"""
    p = ledger_dir / f"events_fixed_{d}.csv"
    if not p.is_file():
        raise GuardError(f"events が無い: {p}（条件ID の確かめができないので、空の表として扱わずに止める）")
    return read_csv(p)


def load_list(path: Path) -> list:
    if WILD.search(str(path)):
        raise GuardError(f"一覧のパスにワイルドカード: {path}")
    if not path.is_file():
        raise GuardError(f"一覧が無い: {path}")
    rows = read_csv(path)
    if not rows:
        raise GuardError("一覧が空")
    miss = [c for c in COLS if c not in rows[0]]
    if miss:
        raise GuardError(f"一覧の列が足りない: {miss}")
    return rows


def has_script_id(role: str, sid: str) -> bool:
    return bool(sid) and re.search(r"(?<![0-9A-Za-z])" + re.escape(sid) + r"(?![0-9])", role) is not None


def guard(rows: list, ledger_dir: Path) -> dict:
    """一覧の全行を台帳と照らす。1 つでも外れがあれば GuardError（全部の外れを並べる）。戻り = {日付: 台帳}。"""
    errs, ledgers, events = [], {}, {}
    seen_o, seen_c = set(), set()
    for i, r in enumerate(rows, start=2):
        d, o, sid, kb = (r["日付"] or "").strip(), (r["原本"] or "").strip(), (r["台本ID"] or "").strip(), (r["区分"] or "").strip()
        yaw, line, cid = (r["向きの直し_deg"] or "").strip(), (r["線の上書き_m"] or "").strip(), (r["clip_id"] or "").strip()
        at = f"一覧 {i} 行目 {o or '（原本なし）'}"
        if WILD.search(o) or not ORIG_RE.match(o):
            errs.append(f"{at}: 原本名の形が違う（ワイルドカード不可・YYMMDD_NNN.WAV）"); continue
        if not re.fullmatch(r"2026\d{4}", d) or ORIG_RE.match(o).group(1) != d[2:]:
            errs.append(f"{at}: 原本名の日付が一覧の日付 {d} と違う"); continue
        if o in seen_o:
            errs.append(f"{at}: 原本が重複")
        seen_o.add(o)
        if kb not in KUBUN:
            errs.append(f"{at}: 区分 {kb!r} は {KUBUN} のどれか")
        if yaw not in ("0", "180") or (kb == "参考（向き）" and yaw != "180"):
            errs.append(f"{at}: 向きの直し_deg は 0 か 180（区分「参考（向き）」は必ず 180・本人 22:28 から「入れる」でも 180 可＝向きを確かめて上げた本）")
        if line:
            try:
                v = float(line)
                if not (0.0 < v <= 20.0):
                    raise ValueError
            except ValueError:
                errs.append(f"{at}: 線の上書き_m は 0〜20 m の数")
        if not CLIP_RE.match(cid) or cid in seen_c:
            errs.append(f"{at}: clip_id {cid!r} は tune_field_clip8NNNN の形で重複なし（評価の eval_field_ と取り違えない）")
        seen_c.add(cid)
        try:
            led = ledgers.setdefault(d, load_ledger(ledger_dir, d))
        except GuardError as ex:
            errs.append(f"{at}: {ex}"); continue
        lr = led.get(o)
        if lr is None:
            errs.append(f"{at}: 台帳 ledger_{d}.csv に無い"); continue
        role = (lr.get("役割") or "").strip()
        if role.startswith("本"):
            errs.append(f"{at}: 台帳で「本」（評価に使う本）→ 入れない（{role[:30]}）"); continue
        if not role.startswith("使わない"):
            errs.append(f"{at}: 台帳の役割が「使わない」でない（{role[:30]}）"); continue
        if not has_script_id(role, sid):
            errs.append(f"{at}: 台本ID {sid} が台帳の役割の文に無い（取り違えの疑い: {role[:40]}）")
        ev = events.setdefault(d, load_events(ledger_dir, d))
        if any((e.get("原本") or "").strip() == o and (e.get("条件ID") or "").strip() for e in ev):
            errs.append(f"{at}: events_fixed_{d}.csv で条件ID が入っている（評価の本）→ 入れない")
    if errs:
        raise GuardError("\n".join(errs))
    return ledgers


def condition_rows(r: dict, ev: list, led: dict) -> tuple[list, str]:
    """この本の条件の行。同じ原本の行があればそれ、無ければ同じ台本ID の評価の本（台帳で「本」）の行を写す。戻り = (行, 写し元の原本)。
    一覧の任意の列 条件の写し元の台本ID（2026-10-26 本人 22:41）: 実際の走り方が台本と違った本は、実際に近い台本の評価の本から写す
    （例 W03-歩行 の 1 回目 081 は、実際は W04 と同じ向き〔歩く人は反対向き・自転車は後ろから右・線 2.0 m〕＝ W04-歩行 を写す）。守りの台本ID の照合は台帳の台本ID のまま。"""
    o = r["原本"].strip()
    sid = (r.get("条件の写し元の台本ID") or "").strip() or r["台本ID"].strip()
    own = [e for e in ev if (e.get("原本") or "").strip() == o]
    if own:
        return own, o
    cands = sorted({(e.get("原本") or "").strip() for e in ev
                    if (e.get("台本ID") or "").strip() == sid and (led.get((e.get("原本") or "").strip(), {}).get("役割") or "").startswith("本")})
    if len(cands) != 1:
        raise SystemExit(f"{o}: 台本ID {sid} の評価の本が 1 本に決まらない（{cands}）→ 条件の行を作れない（終了コード 3）")
    return [e for e in ev if (e.get("原本") or "").strip() == cands[0]], cands[0]


def demo_events(rows_d: list, ev: list, led: dict) -> tuple[list, dict]:
    out, src_of = [], {}
    for n, r in enumerate(rows_d, start=1):
        src, src_name = condition_rows(r, ev, led)
        o = r["原本"].strip()
        src_of[o] = src_name
        for e in src:
            x = dict(e)
            x["原本"] = o
            x["動画原本"] = (led[o].get("動画") or "").strip()
            x["take_id"] = str(900 + n)
            for c in CLEAR:
                if c in x:
                    x[c] = ""
            x["台本ID"] = r["台本ID"].strip()
            if (r["線の上書き_m"] or "").strip():
                v = f"{float(r['線の上書き_m']):.1f}"
                x["line_m"] = v
                if (x.get("横距離m") or "").strip():
                    x["横距離m"] = v
            x["特記"] = (f"デモ用（台帳で使わないの本・評価に使わない）・条件の写し元 {src_name}" + (f"・線 {r['線の上書き_m']} m に上書き" if (r["線の上書き_m"] or "").strip() else "")
                       + (f"・読みの備考: {(r.get('読みの備考') or '').strip()}" if (r.get("読みの備考") or "").strip() else ""))
            out.append(x)
    return out, src_of


def add_notes(path: Path, notes: dict) -> int:
    """step19d が書いた読む表に 備考 の列を足す（原本ごと・歩く人の行も同じ原本なら同じ備考）。戻り = 備考を入れた行の数。"""
    rows = read_csv(path)
    cols = list(rows[0].keys()) if rows else []
    if "備考" not in cols:
        cols.append("備考")
    n = 0
    for r in rows:
        r["備考"] = notes.get((r.get("take") or "").strip(), "")
        n += bool(r["備考"])
    write_csv(path, rows, cols)
    return n


def write_memo(path: Path, rows_d: list, led: dict) -> int:
    """本人が書く除外区間のメモの表（案 A・本人 22:56）。既にあれば上書きしない（本人の書いた値を守る）。戻り = 書いた行の数（既にあれば −1）。
    行 = 区分が「入れる（除外区間あり）」の本と、読みの備考のある「参考」の本（最後まで通っていれば上げられる本）。"""
    if path.exists():
        return -1
    out = [{"原本": r["原本"].strip(), "clip_id": r["clip_id"].strip(), "動画": (led[r["原本"].strip()].get("動画") or "").strip(), "区分": r["区分"].strip(),
            "自転車が最後まで通った（○／×）": "", "除外の入り_音の秒": "", "除外の出_音の秒": "", "メモ": ""}
           for r in rows_d if r["区分"].strip() == "入れる（除外区間あり）" or (r["区分"].strip() == "参考" and (r.get("読みの備考") or "").strip())]
    if out:
        write_csv(path, out, list(MEMO_COLS))
    return len(out)


def write_csv(p: Path, rows: list, cols=None) -> None:
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=E2E / "out" / "_demo_field_unused_149")
    ap.add_argument("--convert", action="store_true", help="守りを通ったら変換・条件の行・読む表・index を書く（無ければ守りの確かめだけ）")
    ap.add_argument("--overwrite", action="store_true", help="既にある置き場に書き直す")
    ap.add_argument("--no-readprep", action="store_true", help="step19d --video-only を呼ばない（試験用）")
    ap.add_argument("--raw-dir", type=Path, default=E2E / "raw", help="試験用（既定 raw/）")
    ap.add_argument("--ledger-dir", type=Path, default=None, help="試験用（既定 <raw>/ledger）")
    a = ap.parse_args(argv)
    ledger_dir = a.ledger_dir or (a.raw_dir / "ledger")
    try:
        if WILD.search(str(a.out)) or not a.out.name.startswith(OUT_PREFIX):
            raise GuardError(f"--out の名前は {OUT_PREFIX} で始めること（評価の置き場に書かない）: {a.out}")
        if a.raw_dir.resolve() in a.out.resolve().parents or a.out.resolve() == a.raw_dir.resolve():
            raise GuardError(f"--out が raw/ の中: {a.out}")
        rows = load_list(a.list)
        ledgers = guard(rows, ledger_dir)
    except GuardError as ex:
        print("❌ 守りで止めた（終了コード 2・何も書いていない）:\n" + str(ex))
        return 2
    by = {}
    for r in rows:
        by.setdefault(r["日付"].strip(), []).append(r)
    n_k = {k: sum(r["区分"].strip() == k for r in rows) for k in KUBUN}
    print(f"守り OK: {len(rows)} 本（{n_k}）・台帳で全部「使わない」・評価の本 0")
    if not a.convert:
        return 0
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:
        print(f"❌ 置き場 {a.out} に既に中身がある（書き直すなら --overwrite）")
        return 3
    from step19_realsmoke_convert import calibration_record, convert_with_gain
    info = {"tool": "scripts/demo_field_unused_prepare.py", "tool_sha256": sha(__file__), "rule": RULE, "convention": CONVENTION,
            "list": str(a.list), "list_sha256": sha(a.list), "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "n": n_k, "dates": {}}
    for d, rows_d in sorted(by.items()):
        led = ledgers[d]
        ev = load_events(ledger_dir, d)
        sess_p = ledger_dir / f"session_fixed_{d}.csv"
        sessions = {s["session_id"]: s for s in read_csv(sess_p)}
        dd = a.out / d
        conv_dir = dd / "conv"
        evd, src_of = demo_events(rows_d, ev, led)
        write_csv(dd / "events_demo.csv", evd, list(ev[0].keys()) if ev else None)
        used_s = sorted({e["session_id"] for e in evd})
        # 2026-10-27: 用途 = 調整用（step19d が評価の注釈に入れず <out>_tuning.csv に出す・家のデモと同じ扱い）。
        #   備考は除外の一覧だけ外し、自転車・車の長さ（軸の間）とベルの位置は残す（step19l の「いつ」と「どこ」が読む）
        sdemo = [dict(sessions[s], 用途="調整用",
                      備考="デモ用（台帳で使わないの本だけ・評価に使わない）／" + "／".join(x for x in (sessions[s].get("備考") or "").split("／") if x.strip() and not re.match(r"\s*除外[:：]", x)))
                 for s in used_s]
        write_csv(dd / "sessions_demo.csv", sdemo, list(sdemo[0].keys()))
        gains, idx = {}, []
        sess_of = {e["原本"]: e["session_id"] for e in evd}
        for r in rows_d:
            o = r["原本"].strip()
            s = sessions[sess_of[o]]
            cal = s["校正原本"].strip()
            if cal not in gains:
                w0, w1 = (float(x) for x in s["暗騒音区間_秒"].split("-"))
                gains[cal] = calibration_record(a.raw_dir / cal, float(s["LAeq_dB"]), (w0, w1), Path(cal).stem, conv_dir, convention=CONVENTION)
            g = gains[cal]
            yaw = float(r["向きの直し_deg"])
            outp = convert_with_gain(a.raw_dir / o, conv_dir, g["gain_db"], 0.0, 0.0, yaw)
            idx.append({"clip_id": r["clip_id"].strip(), "日付": d, "原本": o, "台本ID": r["台本ID"].strip(), "区分": r["区分"].strip(),
                        "使わない理由（台帳）": led[o]["役割"].strip(), "動画": (led[o].get("動画") or "").strip(), "条件の写し元": src_of[o],
                        "向きの直し_deg": r["向きの直し_deg"].strip(), "線の上書き_m": (r["線の上書き_m"] or "").strip(), "session_id": sess_of[o],
                        "校正原本": cal, "gain_db": g["gain_db"], "convention": g["convention"], "conv": f"conv/{outp.name}", "conv_sha256": sha(outp),
                        "読みの備考": (r.get("読みの備考") or "").strip(), "条件の写し元の台本ID": (r.get("条件の写し元の台本ID") or "").strip() or r["台本ID"].strip()})
        write_csv(dd / "index.csv", idx)
        nm = write_memo(dd / "除外区間メモ.csv", rows_d, led)
        print(f"{d}: 除外区間メモ " + ("は既にある（本人の書いた値を守って上書きしない）" if nm < 0 else f"{nm} 行を用意"))
        info["dates"][d] = {"ledger_sha256": sha(ledger_dir / f"ledger_{d}.csv"), "events_fixed_sha256": sha(ledger_dir / f"events_fixed_{d}.csv"),
                            "session_fixed_sha256": sha(sess_p), "n": len(rows_d), "calibration": {k: v["gain_db"] for k, v in gains.items()},
                            # 本人の読みとメモの sha（Astra R5-30-04・10/27）: この index を作った時点の控え。正式な固定は demo_field_unused_fix.py の固定記録
                            "reads_sha256": {n: sha(dd / sub / n) for sub, n in (("reads", "reads_A.state.json"), ("reads", "audio_reads_A.json"),
                                                                                  ("reads", "reads_A.csv"), ("reads", "sync_A.csv"), (".", "除外区間メモ.csv"))
                                             if (dd / sub / n).is_file()}}
        if not a.no_readprep:
            rc = subprocess.run([sys.executable, str(E2E / "scripts" / "step19d_field_csv_to_ann.py"), "--session", str(dd / "sessions_demo.csv"),
                                 "--events", str(dd / "events_demo.csv"), "--out", str(dd / "ann_demo.csv"), "--video-only", "--audio-dir", str(a.raw_dir)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(rc.stdout[-1500:], rc.stderr[-800:])
            if rc.returncode != 0:
                print(f"❌ step19d --video-only が止まった（{d}・rc {rc.returncode}）")
                return 4
            nn = add_notes(dd / "ann_demo_video_takes.csv", {r["原本"].strip(): (r.get("読みの備考") or "").strip() for r in rows_d})
            print(f"読む表に備考 {nn} 行（step19h の画面の一番上に出る）")
        print(f"{d}: {len(rows_d)} 本を変換 → {dd}")
    (a.out / "index.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {a.out}/index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
