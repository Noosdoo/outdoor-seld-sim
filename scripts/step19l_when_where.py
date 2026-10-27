# -*- coding: utf-8 -*-
"""step19l_when_where.py — 動画の目印の読み（step19h の reads）から、距離クラス（車・自転車）の「いつ」と「どこ」を作る（2026-10-12・Fable・変更票 #59 r3〜r7「今の決まり」）。

決まり（変更票 #59・今の決まり・r6 時点のまとめ）:
  - 読むのは前輪の接地点だけ（step19h の生の読み・x は線に沿った位置 [m]・原点 = 静止マイク・t = 録音の秒）。
  - 「いつ」= 前輪と後輪の真ん中（車軸の中点）が肩の線（静止の本は x = 0、歩く比較の歩行の本はその時刻の本人の位置 x_m(t)）を越えた時刻:
      x_w(t) − σ·L_mid − x_m(t) = 0   （σ = 進む向き ＋1／−1・L_mid = 測った軸の間 ÷ 2・軌道は読んだ目印の間の直線）
    読んだ目印の外へは延ばさない。交わらない・止まった・何度も交わる・読みが足りない本は**欠測**として報告する（成績を見て除外や録り直しを決めない）。
  - 「どこ」= 同じ中点の軌道。この道具が「前輪 → 中点」のずらしを **1 回だけ**行い、変換済みの reads（reads_mid_*.csv・印と長さと出典の列つき）を書く。
    step19f／step19k はそれを読む（生の読みは残す・2 回目の変換は止める）。ずらすのは距離クラス（car／bicycle）だけ。mic（本人）は変えない。
    ベルは前輪から後ろへ（備考「ベルの位置」・#69 r6）、走りながらのクラクション E3（動画の読みのある horn）は前輪から前へ（備考「ホーンの位置: 前輪の軸から前へ N cm」・#69 r49）ずらす。
  - 長さ: 記録シートの session.csv の備考「自転車の長さ: 軸の間 X cm」「車の長さ: 車種 M・軸の間 X cm」（Version 66）から取り出す。
    要る長さ（録った本のクラスで決まる）の欠け・0 以下・範囲外（自転車 70〜150 cm・乗用車 150〜450 cm）があれば止める（rc 2）。決まった値に黙って置き換えない。
  - 時刻の不確かさの見込み: 読み ±0.1 s（仮定・未測定）÷ 相対速度 … を本ごとに併記する（上限ではない）。

使い方:
  python scripts/step19l_when_where.py --reads <reads_A.csv> --session <session.csv> --events <events.csv> --out <出力dir> [--reader A]
    → <out>/when_where_<reader>.csv（本ごとの t_cpa・σ・L・状態・欠測の理由・見込みの不確かさ）
      <out>/reads_mid_<reader>.csv（中点にずらした reads・step19f／step19k の入力）
      <out>/when_where_<reader>.md（人が読む一覧）
  そのあと: python scripts/step19d_field_csv_to_ann.py ... --when-csv <out>/when_where_<reader>.csv   （ラップ秒の空いた距離クラスの正例を埋める）
  python scripts/step19l_when_where.py --selftest <一時dir>
終了コード: 0 = 全部できた（欠測があっても 0・欠測は表に出る）、2 = 長さの欠けや変換済みの reads など、止めるべき入力。
"""
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _field_orientation import mic_field_xy  # noqa: E402  2026-10-25（敵対レビュー 軽 10）: マイクの現場の座標は step19d と同じ関数（旧は static_mic_x・walker_line_m）
from _field_hidden_fill import hidden_intervals, fill_hidden, strip_fill_flag, hold_ends, HIDE_MAX_S as FILL_MAX_S  # noqa: E402  変更票 #69 r31 ②・r34（2026-10-23）

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIST_CLS_NAMES = {"car": "car", "car_drive": "car", "bicycle": "bicycle", "kick": "kick", "kickboard": "kick", "bike": "bike", "motorcycle": "bike"}
NEED_LENGTH = {"car": "car", "bicycle": "bike"}            # クラス → 要る長さ（kick／bike は実録の台本に無い → 来たら止める）
MIC_CLS = "mic"
READ_UNCERT_S = 0.1                                         # 目印の読みの不確かさ（仮定・未測定・結果に「見込み」と書く）
MIN_REL_SPEED = 0.1                                         # これ未満の相対速度で交わる本は「止まった」= 欠測
STOP_DWELL_S = 0.5                                          # 中点が線の上（f = 0）にこれより長く居たら「止まった」= 欠測（AR12-11・読みの刻みより長い）
MIN_TRAVEL_M = 0.5                                          # 読みの x の幅がこれ未満なら向きが決まらない = 欠測
RANGE_CM = {"bike": (70.0, 150.0), "car": (150.0, 450.0)}  # 記録シートと同じ範囲（Version 65）
CONVERTED_MARK = "midpoint"
FRONT_MARK = "front"                                       # B3（R17-06）: 前の車軸（前輪の読みのまま）の感度の reads の印
BELL_MARK = "bell_offset"                                  # #69 r6: ベルの位置（前輪の軸から後ろへ N cm）をずらした印
HORN_MARK = "horn_offset"                                  # #69 r49（10/27）: 走りながらのクラクション E3 のホーンの位置（前輪の軸から前へ N cm）をずらした印
HIDE_MAX_S = FILL_MAX_S                                    # 変更票 #69 r31 ②（2026-10-23）: 隠れた区間は前後の読みに 2 次式（等加速度）を最小二乗で当てて埋める・上限 3 s・両側に読みがある区間だけ
                                                           #   （旧 B4 R17-01「1 秒以内を線で埋める」を置き換え・_field_hidden_fill.py）。埋められない区間に交わりがあれば欠測
RANGE_M = 10.0                                             # 変更票 #69 r31 ①: 知覚の正解の範囲（検出フレーム率の分母）= 道に沿って ±10（中点が入っている時刻 t_in10_s〜t_out10_s を出す）
OUT_COLS = ["take", "session_id", "cls", "track", "sigma", "L_mid_m", "L_source", "n_marks", "n_mic_marks", "mic", "t_cpa_s", "status", "reason",
            "x_cross_m", "rel_speed_mps", "dt_est_s", "t_first_s", "x_first_m", "t_last_s", "mic_x0_m", "n_hidden", "t_cpa_front_s", "status_front", "reason_front", "dt_front_s",
            "n_hidden_filled", "n_hidden_blocked", "t_in10_s", "t_out10_s", "range_note"]


# ---------------------------------------------------------------- 入力
def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _stem(name: str) -> str:
    return Path(str(name or "").strip()).stem.lower()


def _num(s: str, where: str) -> float:
    """長さの数字。読めない誤記（26.3.0 など）は ValueError を投げ、build が「止める一覧」に入れる（AR12-10）"""
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"長さの数字が読めない: 「{s}」（{where}）")
    return v


def lengths_from_note(note: str) -> dict:
    """session.csv の備考 → {"bike": (cm, 文, ""), "car": (cm, 文, 車種)}（無いものは入れない）。
    全角の数字・コロンは NFKC で半角に。車種は「軸の間」の直前の「・」までを貪欲に取る（車種に「・」「／」が入っても読める・AR12-10）。"""
    import unicodedata
    out = {}
    note = unicodedata.normalize("NFKC", note or "")
    m = re.search(r"自転車の長さ:\s*軸の間\s*(\S+?)\s*cm", note)
    if m:
        out["bike"] = (_num(m.group(1), "自転車の長さ"), m.group(0), "")
    m = re.search(r"(?<!自転)車の長さ:\s*(?:車種\s*(.+?)\s*・\s*(?=軸の間))?軸の間\s*(\S+?)\s*cm", note)   # 「自転車の長さ」に引っかからないように
    if m and m.group(2) != "?":
        out["car"] = (_num(m.group(2), "車の長さ"), m.group(0), (m.group(1) or "").strip())
    return out


def bell_offset_from_note(note: str):
    """session.csv の備考「ベルの位置: 前輪の軸から後ろへ N cm」→ N [cm]（無ければ None・変更票 #69 r6・2026-10-20 B2）。全角は NFKC で半角に。"""
    import unicodedata
    note = unicodedata.normalize("NFKC", note or "")
    m = re.search(r"ベルの位置:\s*前輪の軸から後ろへ\s*(\S+?)\s*cm", note)
    return _num(m.group(1), "ベルの位置") if m else None


def horn_offset_from_note(note: str):
    """session.csv の備考「ホーンの位置: 前輪の軸から前へ N cm」→ N [cm]（無ければ None・#69 r49・2026-10-27）。
    当日に巻尺で測って機材のメモに書き、その夜に台帳（session_fixed）の備考へ写す（クラクションの高さと同じ流れ）。全角は NFKC で半角に。"""
    import unicodedata
    note = unicodedata.normalize("NFKC", note or "")
    m = re.search(r"ホーンの位置:\s*前輪の軸から前へ\s*(\S+?)\s*cm", note)
    return _num(m.group(1), "ホーンの位置") if m else None


def session_lengths(sessions: list[dict]) -> dict:
    """session_id → lengths_from_note(備考)"""
    out = {}
    for r in sessions:
        sid = (r.get("session_id") or "").strip()
        if not sid or sid.startswith("←"):
            continue
        try:
            out[sid] = lengths_from_note(r.get("備考", "") or r.get("note", ""))
            out[sid]["bell_cm"] = bell_offset_from_note(r.get("備考", "") or r.get("note", ""))
            out[sid]["horn_cm"] = horn_offset_from_note(r.get("備考", "") or r.get("note", ""))
        except ValueError as e:
            out[sid] = {"_error": str(e)}
    return out


def take_index(events: list[dict]) -> dict:
    """原本の stem（小文字）→ {"session_id", "state"(静止/歩行), "events": {event_id: class}}"""
    idx = {}
    for r in events:
        sid = (r.get("session_id") or "").strip()
        if not sid or sid.startswith("←") or not r.get("原本"):
            continue
        st = _stem(r["原本"])
        state = (r.get("状態") or "").strip()
        if st not in idx:
            mx, my = mic_field_xy(r, state)                       # B2: 静止の本人の x（R97-01）・Version 123 の マイク_x_m・マイク_y_m（無ければ今までの決まり）
        d = idx.setdefault(st, {"session_id": sid, "state": state, "events": {},
                                "mic_x0": (0.0 if state == "歩行" else float(mx)), "mic_y": float(my)})
        d["events"][str(r.get("event_id") or "")] = (r.get("class") or "").strip()
    return idx


# ---------------------------------------------------------------- 幾何
def _interp(pts: list[tuple[float, float]], t: float) -> float:
    """折れ線の値（pts は (t, x) 昇順・範囲内の t だけ）"""
    for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return x0 if t1 == t0 else x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    raise ValueError("範囲外")


def crossings(opp: list[tuple[float, float]], mic: list[tuple[float, float]] | None, mic_x0: float = 0.0) -> tuple[list[tuple[float, float, float]], float, float]:
    """f(t) = x_opp(t) − x_mic(t) の零点。戻り = ([(t, 相対速度 df/dt, x_opp)], 重なる区間の始め, 終わり)。
    opp／mic は (t, x) の昇順の折れ線。mic が None なら x_mic = mic_x0（静止・既定 0・歩く比較の静止の本は 歩く人の静止x_m）。読んだ目印の外へは延ばさない（重なる区間の中だけ）。
    折れ線のまま解く（等速を仮定しない・R97-03）。"""
    lo = opp[0][0] if mic is None else max(opp[0][0], mic[0][0])
    hi = opp[-1][0] if mic is None else min(opp[-1][0], mic[-1][0])
    if hi <= lo:
        return [], lo, hi
    knots = sorted({t for t, _ in opp if lo <= t <= hi} | ({t for t, _ in mic if lo <= t <= hi} if mic else set()) | {lo, hi})
    f = [(_interp(opp, t) - (mic_x0 if mic is None else _interp(mic, t))) for t in knots]
    roots = []
    # 符号が変わる所だけを交わりとする（節に乗った 0 は、その前後で符号が変わるときだけ。「触れて戻る」は数えない・AR12-11）
    i = 0
    while i < len(knots) - 1:
        t0, t1, f0, f1 = knots[i], knots[i + 1], f[i], f[i + 1]
        slope = (f1 - f0) / (t1 - t0) if t1 != t0 else 0.0
        if f0 != 0.0 and f1 != 0.0 and f0 * f1 < 0:
            tc = t0 - f0 / slope
            roots.append((tc, slope, _interp(opp, tc)))
        elif f0 == 0.0:
            j = i
            while j < len(f) and f[j] == 0.0:
                j += 1
            before = f[i - 1] if i > 0 else None
            after = f[j] if j < len(f) else None
            if before is not None and after is not None and before * after < 0:
                dwell = knots[j - 1] - knots[i]                                        # 0 に居た長さ。STOP_DWELL_S より長ければ「線の上で止まった」= 相対速度 0 で返す（欠測になる）
                s_in = (f[i] - f[i - 1]) / (knots[i] - knots[i - 1]) if knots[i] != knots[i - 1] else 0.0
                s_out = (f[j] - f[j - 1]) / (knots[j] - knots[j - 1]) if knots[j] != knots[j - 1] else 0.0
                slope0 = 0.0 if dwell > STOP_DWELL_S else (s_in if abs(s_in) < abs(s_out) else s_out)
                roots.append((t0, slope0, _interp(opp, t0)))
            i = max(j - 1, i + 1); continue
        i += 1
    return roots, lo, hi


def when_for_take(pts_raw: list[tuple[float, float]], mic_pts: list[tuple[float, float]] | None, L_mid: float, mic_x0: float = 0.0, blocked=()) -> dict:
    """1 本（1 相手）の「いつ」。pts_raw = 前輪の (t, x) 昇順。mic_pts = 本人の (t, x) 昇順か None（静止・x = mic_x0）。
    blocked = 1 秒より長く隠れた区間 [(隠れる前の最後の読み, 見えてからの最初の読み)]（B4・R17-01）。交わりがその中なら欠測。L_mid = 0 なら前の車軸（B3 の感度）。"""
    res = dict(sigma="", n_marks=len(pts_raw), n_mic_marks=(len(mic_pts) if mic_pts else 0), t_cpa_s="", status="", reason="",
               x_cross_m="", rel_speed_mps="", dt_est_s="", t_first_s=(f"{pts_raw[0][0]:.3f}" if pts_raw else ""), x_first_m=(f"{pts_raw[0][1]:g}" if pts_raw else ""), t_last_s=(f"{pts_raw[-1][0]:.3f}" if pts_raw else ""))
    if len(pts_raw) < 2:
        res.update(status="欠測", reason=f"読みが足りない（目印 {len(pts_raw)} 点・2 点以上要る）"); return res
    dx = pts_raw[-1][1] - pts_raw[0][1]
    if abs(dx) < MIN_TRAVEL_M:
        res.update(status="欠測", reason=f"動いていない／向きが決まらない（x の幅 {abs(dx):.2f} m < {MIN_TRAVEL_M} m）"); return res
    sigma = 1 if dx > 0 else -1
    res["sigma"] = sigma
    if mic_pts is not None and len(mic_pts) < 2:
        res.update(status="欠測", reason=f"本人（mic）の読みが足りない（{len(mic_pts)} 点・歩行の本は 2 点以上要る）"); return res
    if len({t for t, _ in pts_raw}) != len(pts_raw) or (mic_pts is not None and len({t for t, _ in mic_pts}) != len(mic_pts)):
        res.update(status="欠測", reason="同じ時刻の読みが 2 つある（重複読み・どちらが正しいか決められない）"); return res   # AR12-11
    mid = [(t, x - sigma * L_mid) for t, x in pts_raw]
    t_in, t_out, rnote = range_times(mid)                                 # r31 ①: 中点が ±10 に入っている時刻（検出フレーム率の分母の窓）
    res.update(t_in10_s=t_in, t_out10_s=t_out, range_note=rnote)
    if mic_pts is not None:
        # r34（2026-10-22 本人）: 歩く人の最初の読み（∓10・動き始め）より前はそこに、最後の読み（±10・止まった）より後ろはそこに立っていた
        mic_pts = hold_ends(mic_pts, min(pts_raw[0][0], mic_pts[0][0]) - 1.0, max(pts_raw[-1][0], mic_pts[-1][0]) + 1.0)
    roots, lo, hi = crossings(mid, mic_pts, mic_x0)
    if hi <= lo:
        res.update(status="欠測", reason="相手と本人の読みの時刻が重ならない"); return res
    if not roots:
        res.update(status="欠測", reason=f"交わらない（読んだ目印の範囲 {lo:.2f}〜{hi:.2f} s の中で中点が肩の線を越えない）"); return res
    if len(roots) > 1:
        res.update(status="欠測", reason=f"何度も交わる（{len(roots)} 回: {', '.join(f'{t:.2f}' for t, _, _ in roots)} s）"); return res
    tc, slope, xc = roots[0]
    if slope != 0.0 and (slope > 0) != (sigma > 0):
        res.update(status="欠測", reason=f"交わる向きが進む向きと逆（σ {sigma:+d}・相対速度の符号 {'+' if slope > 0 else '-'}）"); return res   # AR12-11
    if abs(slope) < MIN_REL_SPEED:
        res.update(status="欠測", reason=f"止まった（交わる所の相対速度 {abs(slope):.2f} m/s < {MIN_REL_SPEED} m/s）"); return res
    for a, b in blocked:
        if a < tc < b:
            res.update(status="欠測", reason=f"隠れて読めない（交わり {tc:.2f} s が 1 秒より長く見えない区間 {a:.2f}〜{b:.2f} s の中・R17-01）"); return res
    res.update(status="ok", t_cpa_s=f"{tc:.3f}", x_cross_m=f"{xc:.3f}", rel_speed_mps=f"{abs(slope):.3f}", dt_est_s=f"{READ_UNCERT_S / abs(slope):.3f}")
    return res


# ---------------------------------------------------------------- まとめ
def pts_and_blocks(rows: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[dict]]:
    """読みの行 → (前後の読みに 2 次式を当てて埋めた (t, x) の点列, 埋められなかった隠れた区間 [(前の読み t, 後の読み t)], 埋めの記録)（変更票 #69 r31 ②・2026-10-23）。
    隠れた印が不完全なら ValueError（R72-05・黙って隠れなしにしない）。埋められなかった区間（3 s より長い・片側に読みが無い）に交わりがあれば欠測。"""
    raw = _pts(rows)
    iv = hidden_intervals(rows)
    if not iv:
        return raw, [], []
    filled, blocked, log = fill_hidden(raw, iv)
    return strip_fill_flag(filled, 2), blocked, log


def range_times(pts: list[tuple[float, float]], lim: float = RANGE_M) -> tuple[str, str, str]:
    """折れ線 (t, x) が |x| ≤ lim に入っている最初と最後の時刻（r31 ①: 検出フレーム率の分母の窓）。読みの範囲の外へは延ばさない。
    戻り = (t_in, t_out, 注)。入っている時刻が無ければ空。"""
    if len(pts) < 2:
        return "", "", "読みが足りない"
    t_in = t_out = None
    for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
        inside0, inside1 = abs(x0) <= lim, abs(x1) <= lim
        if inside0 and t_in is None:
            t_in = t0
        if not inside0 and inside1 and t_in is None:
            # 線分の途中で入る: |x| = lim の時刻
            target = lim if x1 < x0 else -lim
            if x1 != x0:
                t_in = t0 + (target - x0) * (t1 - t0) / (x1 - x0)
        if inside0 or inside1:
            if inside1:
                t_out = t1
            elif inside0:
                target = lim if x1 > x0 else -lim
                t_out = t0 + (target - x0) * (t1 - t0) / (x1 - x0) if x1 != x0 else t1
    if t_in is None:
        return "", "", f"読みの範囲が ±{lim:g} の外"
    note = ""
    if abs(pts[0][1]) <= lim:
        note = "最初の読みが既に中（入りは最初の読みの時刻）"
    if abs(pts[-1][1]) <= lim:
        note = (note + "・" if note else "") + "最後の読みがまだ中（出は最後の読みの時刻）"
    return f"{t_in:.3f}", f"{t_out:.3f}", note


def _pts(rows: list[dict]) -> list[tuple[float, float]]:
    pts = []
    for r in rows:
        if (r.get("hidden") or "").strip():
            continue                                                   # 隠れた印の行（x は空）は位置の読みではない（B4）
        t = r.get("t")
        if t in (None, "", "nan"):
            raise ValueError(f"t が無い行（step19h の reads は t 必須）: {dict(r)}")
        pts.append((float(t), float(r["x"])))
    pts.sort()
    return pts


def build(reads: list[dict], sessions: list[dict], events: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """戻り = (when の行, 変換済み reads の行, 止める理由の一覧)"""
    if reads and ("converted" in reads[0] or any((r.get("converted") or "") in (CONVERTED_MARK, FRONT_MARK, BELL_MARK, HORN_MARK) and (r.get("converted") or "") for r in reads)):   # 列があるか、どれか 1 行でも印があれば止める（AR12-04）
        return [], [], ["入力の reads は変換済み（converted の列か midpoint の印がある）。生の読み（step19h の reads_*.csv）を渡す（2 回目の変換は止める）"]
    lengths = session_lengths(sessions); tidx = take_index(events)
    by_take = defaultdict(lambda: defaultdict(list)); mic_by_take = defaultdict(list)
    for r in reads:
        take = (r.get("take") or "").strip(); c = (r.get("cls") or "").strip().lower()
        if c == MIC_CLS:
            mic_by_take[take].append(r)
        else:
            by_take[take][(c, str(r.get("track") or "1"))].append(r)      # track が空なら 1（step19d の event_id の既定と同じ・AR12-12）
    stops, when_rows, mid_rows = [], [], []
    shift_of = {}
    for take in sorted(set(by_take) | set(mic_by_take)):
        st = _stem(take); meta = tidx.get(st)
        sid = meta["session_id"] if meta else ""
        state = meta["state"] if meta else ""
        mic_rows = mic_by_take.get(take, [])
        mic_x0 = float(meta["mic_x0"]) if meta else 0.0                     # B2: 静止の本人の x（歩く比較の静止の本・R97-01）
        mic_pts, mic_blocked, mic_log, mic_hidden_err = None, [], [], ""
        if state == "歩行" or mic_rows:
            try:
                mic_pts, mic_blocked, mic_log = pts_and_blocks(mic_rows)       # r31 ②: 本人（足元）の隠れた区間も 2 次式で埋める・埋められない区間は欠測の区間
            except ValueError as e:
                mic_pts, mic_hidden_err = _pts(mic_rows), str(e)
        for (c, trk), rows in sorted(by_take[take].items()):
            cname = DIST_CLS_NAMES.get(c)
            if cname is None:
                # 距離クラス以外は「いつ」を作らない。ベルだけ「どこ」を前輪からベルの位置へずらす（#69 r6・2026-10-20 B2: 前輪 − σ × N cm）
                if c == "bike_bell":
                    if not meta:
                        stops.append(f"{take} bike_bell: events.csv に原本が無い（session が分からずベルの位置を引けない）"); continue
                    bell = lengths.get(sid, {}).get("bell_cm") if "_error" not in lengths.get(sid, {}) else None
                    if bell is None:
                        stops.append(f"{take} ({sid}) bike_bell: session.csv の備考に「ベルの位置: 前輪の軸から後ろへ N cm」が無い → 決まった値に置き換えない・止める（10/15 の C は 10/22 に測った同じ自転車の値を書く）"); continue
                    p_ = _pts(rows)
                    if len(p_) < 2 or abs(p_[-1][1] - p_[0][1]) < MIN_TRAVEL_M:
                        stops.append(f"{take} ({sid}) bike_bell: 走る向きが決まらない（読み {len(p_)} 点）→ ベルの位置をずらせない"); continue
                    sg = 1 if p_[-1][1] > p_[0][1] else -1
                    shift_of[(take, c, trk)] = (-sg * bell / 100.0, bell / 100.0, f"{sid} 備考「ベルの位置: 前輪の軸から後ろへ {bell:g} cm」", BELL_MARK)
                elif c == "horn":
                    # #69 r49（10/27）: 走りながらのクラクション E3（動画の読みがある horn = 動く本だけ。止めた E2 は step19m の読みで、ここには来ない）。
                    # 「どこ」= 前輪からホーンの位置へ前に N cm（備考「ホーンの位置: 前輪の軸から前へ N cm」・無ければ止める＝決まった値に置き換えない）
                    if not meta:
                        stops.append(f"{take} horn: events.csv に原本が無い（session が分からずホーンの位置を引けない）"); continue
                    hcm = lengths.get(sid, {}).get("horn_cm") if "_error" not in lengths.get(sid, {}) else None
                    if hcm is None:
                        stops.append(f"{take} ({sid}) horn: session.csv の備考に「ホーンの位置: 前輪の軸から前へ N cm」が無い → 決まった値に置き換えない・止める（当日に巻尺で測って機材のメモに書いた値を台帳の備考へ）"); continue
                    p_ = _pts(rows)
                    if len(p_) < 2 or abs(p_[-1][1] - p_[0][1]) < MIN_TRAVEL_M:
                        stops.append(f"{take} ({sid}) horn: 走る向きが決まらない（読み {len(p_)} 点）→ ホーンの位置をずらせない"); continue
                    sg = 1 if p_[-1][1] > p_[0][1] else -1
                    shift_of[(take, c, trk)] = (sg * hcm / 100.0, hcm / 100.0, f"{sid} 備考「ホーンの位置: 前輪の軸から前へ {hcm:g} cm」", HORN_MARK)
                continue
            need = NEED_LENGTH.get(cname)
            if need is None:
                stops.append(f"{take} {c}: 長さの決まりが無いクラス（実録の台本に無い）"); continue
            if not meta:
                stops.append(f"{take}: events.csv に原本が無い（session が分からず長さを引けない）"); continue
            if "_error" in lengths.get(sid, {}):
                stops.append(f"{take} ({sid}) {c}: {lengths[sid]['_error']} → 備考を直す・決まった値に置き換えない"); continue
            L = lengths.get(sid, {}).get(need)
            if L is None:
                stops.append(f"{take} ({sid}) {c}: session.csv の備考に{'自転車' if need == 'bike' else '車'}の長さ（軸の間）が無い → 決まった値に置き換えない・止める"); continue
            cm, src, model = L
            lo_cm, hi_cm = RANGE_CM[need]
            if not (math.isfinite(cm) and cm > 0):
                stops.append(f"{take} ({sid}) {c}: 長さが 0 以下か数値でない: {src}"); continue
            if not (lo_cm <= cm <= hi_cm):
                stops.append(f"{take} ({sid}) {c}: 長さ {cm:g} cm が範囲外（{lo_cm:g}〜{hi_cm:g} cm）: {src}"); continue
            L_mid = cm / 100.0 / 2.0
            try:
                own_pts, own_blocks, own_log = pts_and_blocks(rows); hidden_err = mic_hidden_err    # r31 ②: 前輪の隠れた区間を 2 次式で埋める
            except ValueError as e:
                own_pts, own_blocks, own_log, hidden_err = _pts(rows), [], [], str(e)
            blocked = own_blocks + mic_blocked
            res = when_for_take(own_pts, mic_pts, L_mid, mic_x0, blocked)
            if hidden_err:
                res.update(status="欠測", reason=hidden_err)                        # R72-05: 印が不完全な本は欠測（黙って隠れなしにしない）
            res["mic_x0_m"] = f"{mic_x0:g}"; res["n_hidden"] = len(own_blocks)
            res["n_hidden_filled"] = sum(1 for g in own_log + mic_log if g["filled"]); res["n_hidden_blocked"] = len(own_blocks) + len(mic_blocked)
            if cname == "car":                                                   # B3（R17-06）: 前の車軸（L_mid = 0）で別に解いた感度
                rf = when_for_take(own_pts, mic_pts, 0.0, mic_x0, blocked)
                res["t_cpa_front_s"] = rf["t_cpa_s"]; res["status_front"] = rf["status"]; res["reason_front"] = rf["reason"]
                res["dt_front_s"] = (f"{float(rf['t_cpa_s']) - float(res['t_cpa_s']):+.3f}" if rf["status"] == "ok" and res["status"] == "ok" else "")
            when_rows.append(dict(take=take, session_id=sid, cls=c, track=trk, L_mid_m=f"{L_mid:.3f}", L_source=f"{sid} 備考「{src}」" + (f"（車種 {model}）" if model else ""),
                                  mic=("歩行（本人の読み）" if mic_pts is not None else "静止（x = 0）"), **res))
            if res["sigma"] != "":
                shift_of[(take, c, trk)] = (-res["sigma"] * L_mid, L_mid, f"{sid} 備考「{src}」" + (f"（車種 {model}）" if model else ""), CONVERTED_MARK)
    for r in reads:                                               # 変換済み reads（距離クラスの行だけ x をずらす・ほかは同じ）
        take = (r.get("take") or "").strip(); c = (r.get("cls") or "").strip().lower(); trk = str(r.get("track") or "1")
        s = shift_of.get((take, c, trk))
        row = dict(r)
        if s is not None and not (r.get("hidden") or "").strip():
            row["x"] = f"{float(r['x']) + s[0]:.4f}"; row["converted"] = s[3]; row["shift_m"] = f"{s[0]:+.3f}"; row["L_mid_m"] = f"{s[1]:.3f}"; row["L_source"] = s[2]
        else:
            row["converted"] = ""; row["shift_m"] = ""; row["L_mid_m"] = ""; row["L_source"] = ""
        mid_rows.append(row)
    return when_rows, mid_rows, stops


def to_markdown(when_rows: list[dict], stops: list[str], reader: str) -> str:
    L = [f"# 「いつ」「どこ」の計算（reads_{reader}・step19l・変更票 #59）", "",
         "- 「いつ」= 車軸の中点（前輪の読み − σ·L_mid）が肩の線（静止 x = 0・歩行は本人の位置）を越えた時刻。読んだ目印の範囲の中だけ。欠測は理由つきで残す（成績で除外を決めない）。",
         f"- 不確かさの見込み dt_est = 読み ±{READ_UNCERT_S} s（仮定・未測定）÷ 交わる所の相対速度。上限ではない。", ""]
    ok = [r for r in when_rows if r["status"] == "ok"]; miss = [r for r in when_rows if r["status"] != "ok"]
    L.append(f"## できた本 {len(ok)}・欠測 {len(miss)}・止めた入力 {len(stops)}")
    L.append("")
    fr = [r for r in when_rows if r.get("status_front")]
    if fr:
        chg = [r for r in fr if r["status"] != r["status_front"]]
        dts = [float(r["dt_front_s"]) for r in fr if r.get("dt_front_s")]
        L.append(f"- 前の車軸の感度（R17-06・車 {len(fr)} 本・別名 when_where_*_front.csv／reads_front_*.csv）: 主（真ん中）と結果の状態が変わる本 {len(chg)}"
                 + (f"・t_cpa の差（前 − 真ん中）{min(dts):+.3f}〜{max(dts):+.3f} s" if dts else "") + "。ベルには真ん中の補正をかけない")
        L.append("")
    for r in when_rows:
        if r["status"] == "ok":
            L.append(f"- {r['take']} {r['cls']} track {r['track']}: t_cpa {r['t_cpa_s']} s（σ {r['sigma']:+d}・L_mid {r['L_mid_m']} m・相対速度 {r['rel_speed_mps']} m/s・見込み ±{r['dt_est_s']} s・目印 {r['n_marks']} 点・{r['mic']}）")
    for r in miss:
        L.append(f"- **欠測** {r['take']} {r['cls']} track {r['track']}: {r['reason']}（目印 {r['n_marks']} 点・{r['mic']}）")
    for s in stops:
        L.append(f"- **止めた**: {s}")
    return "\n".join(L) + "\n"


def write_outputs(out_dir: Path, reader: str, when_rows, mid_rows, stops) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_when = out_dir / f"when_where_{reader}.csv"; p_mid = out_dir / f"reads_mid_{reader}.csv"
    with open(p_when, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows({k: r.get(k, "") for k in OUT_COLS} for r in when_rows)
    if mid_rows:
        cols = list(mid_rows[0].keys())
        with open(p_mid, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(mid_rows)
        # B3（R17-06）: 感度用 = 車だけ前輪の読みのまま（印 front）・ほかは主と同じ。主と別の名前で保存
        front_rows = []
        for r in mid_rows:
            q = dict(r)
            if r.get("converted") == CONVERTED_MARK and DIST_CLS_NAMES.get((r.get("cls") or "").strip().lower()) == "car" and r.get("shift_m"):
                q["x"] = f"{float(r['x']) - float(r['shift_m']):.4f}"; q["converted"] = FRONT_MARK; q["shift_m"] = "+0.000"
            front_rows.append(q)
        with open(out_dir / f"reads_front_{reader}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(front_rows)
    if when_rows:
        front_when = []
        for r in when_rows:
            q = dict(r)
            if r.get("status_front"):
                q["t_cpa_s"], q["status"], q["reason"], q["L_mid_m"] = r["t_cpa_front_s"], r["status_front"], r["reason_front"], "0.000"
            front_when.append(q)
        with open(out_dir / f"when_where_{reader}_front.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows({k: r.get(k, "") for k in OUT_COLS} for r in front_when)
    (out_dir / f"when_where_{reader}.md").write_text(to_markdown(when_rows, stops, reader), encoding="utf-8")
    return p_when, p_mid


# ---------------------------------------------------------------- 自己検査
def selftest(tmp: Path) -> int:
    import json
    tmp.mkdir(parents=True, exist_ok=True); ok = []
    sessions = [dict(session_id="S1", 備考="テープ／自転車の長さ: 軸の間 104 cm／車の長さ: 車種 ノート・軸の間 263 cm／ベルの位置: 前輪の軸から後ろへ 30 cm"), dict(session_id="S2", 備考="無し")]
    events = [dict(session_id="S1", take_id="S1/t01", 原本="251015_001.WAV", event_id="1", **{"class": "car", "状態": "静止"}),
              dict(session_id="S1", take_id="S1/t02", 原本="251015_002.WAV", event_id="1", **{"class": "bicycle", "状態": "歩行"}),
              dict(session_id="S1", take_id="S1/t03", 原本="251015_003.WAV", event_id="1", **{"class": "car", "状態": "静止"}),
              dict(session_id="S1", take_id="S1/t04", 原本="251015_004.WAV", event_id="1", **{"class": "bicycle", "状態": "静止"}),
              dict(session_id="S2", take_id="S2/t01", 原本="251015_010.WAV", event_id="1", **{"class": "car", "状態": "静止"}),
              dict(session_id="S1", take_id="S1/t05", 原本="251015_005.WAV", event_id="1", **{"class": "bike_bell", "状態": "静止"})]
    def rd(take, cls, pts, trk="1"):
        return [dict(take=take, frame="", cls=cls, track=trk, x=f"{x}", y="2", z="", t=f"{t}") for t, x in pts]
    # 1 静止・一定速 4 m/s・前輪が x=−10 → +10（σ=+1）。中点 = 前輪 − 1.315 m → 前輪が x = +1.315 に来る時刻 = 10 + 11.315/4
    r1 = rd("251015_001.WAV", "car", [(10 + (x + 10) / 4.0, x) for x in range(-10, 11, 2)])
    # 2 歩行の自転車（後ろから 3 m/s・前輪 x: −10 → +10）・本人 1.2 m/s で +x に歩く（x_m = 1.2 (t − 10)）
    r2 = rd("251015_002.WAV", "bicycle", [(10 + (x + 10) / 3.0, x) for x in range(-10, 11, 2)]) + rd("251015_002.WAV", "mic", [(10 + k * 2.0, 1.2 * k * 2.0) for k in range(0, 5)], trk="0")
    # 3 交わらない（x が −10 → −2 で終わる）
    r3 = rd("251015_003.WAV", "car", [(10 + (x + 10) / 4.0, x) for x in range(-10, -1, 2)])
    # 4 何度も交わる（行って戻る）
    r4 = rd("251015_004.WAV", "bicycle", [(10, -4), (12, 2), (14, -3), (16, 4)])
    # 5 長さが無い session
    r5 = rd("251015_010.WAV", "car", [(10, -6), (13, 6)])
    # 6 ベルは対象外（ずらさない）
    r6 = rd("251015_005.WAV", "bike_bell", [(10, -6), (13, 6)])
    reads = r1 + r2 + r3 + r4 + r6
    when, mid, stops = build(reads, sessions, events)
    W = {(r["take"], r["cls"]): r for r in when}
    t1 = 10 + (1.315 + 10) / 4.0
    w1 = W[("251015_001.WAV", "car")]
    ok.append((f"静止・一定速: t_cpa = {w1['t_cpa_s']}（理論 {t1:.3f}）・σ +1・L_mid 1.315・見込み ±{w1['dt_est_s']} s（= 0.1/4）", w1["status"] == "ok" and abs(float(w1["t_cpa_s"]) - t1) < 6e-4 and w1["sigma"] == 1 and w1["L_mid_m"] == "1.315" and abs(float(w1["dt_est_s"]) - 0.025) < 1e-6))
    # 2 の理論: 中点 x = −10 + 3(t−10) − 0.52 = 1.2 (t−10) → (t−10) = 10.52/1.8
    t2 = 10 + 10.52 / 1.8; w2 = W[("251015_002.WAV", "bicycle")]
    ok.append((f"歩行の自転車（本人が動く）: t_cpa = {w2['t_cpa_s']}（理論 {t2:.3f}）・相対速度 1.8", w2["status"] == "ok" and abs(float(w2["t_cpa_s"]) - t2) < 6e-4 and abs(float(w2["rel_speed_mps"]) - 1.8) < 1e-9 and w2["mic"].startswith("歩行")))
    w3 = W[("251015_003.WAV", "car")]
    ok.append((f"交わらない本は欠測: {w3['reason'][:20]}…", w3["status"] == "欠測" and w3["reason"].startswith("交わらない")))
    w4 = W[("251015_004.WAV", "bicycle")]
    ok.append((f"何度も交わる本は欠測: {w4['reason'][:24]}…", w4["status"] == "欠測" and w4["reason"].startswith("何度も交わる")))
    ok.append(("ベル（警告音）は「いつ」を作らず、「どこ」はベルの位置へ（前輪 − σ × 30 cm・#69 r6・印 bell_offset）",
               ("251015_005.WAV", "bike_bell") not in W and all(r["converted"] == "bell_offset" and r["shift_m"] == "-0.300" for r in mid if r["take"] == "251015_005.WAV")))
    # #69 r49（10/27）: 走りながらのクラクション E3 = 動画の読みのある horn → 「いつ」は作らず、「どこ」は前輪から前へ（備考「ホーンの位置: 前輪の軸から前へ 85 cm」）。
    # 前→後（x が +10 → −10・σ = −1）なら −0.85 m。備考に無ければ止める
    s_h = [dict(session_id="E1", 備考="車の長さ: 車種 ノート・軸の間 263 cm／ホーンの位置: 前輪の軸から前へ 85 cm"), dict(session_id="E2s", 備考="車の長さ: 車種 ノート・軸の間 263 cm")]
    e_h = [dict(session_id="E1", take_id="E1/t01", 原本="261029_020.WAV", event_id="1", **{"class": "horn", "状態": "静止", "台本ID": "E3-03"}),
           dict(session_id="E2s", take_id="E2s/t01", 原本="261029_021.WAV", event_id="1", **{"class": "horn", "状態": "静止", "台本ID": "E3-01"})]
    w_h, m_h, s_hs = build(rd("261029_020.WAV", "horn", [(10, 10), (14, -6)]) + rd("261029_021.WAV", "horn", [(10, -10), (14, 6)]), s_h, e_h)
    ok.append(("走りながらのクラクション E3: 「いつ」は作らない・「どこ」は前輪から前へ（前→後なら −0.85 m・印 horn_offset）・備考に「ホーンの位置」が無い本は止める（#69 r49）",
               not w_h and all(r["converted"] == "horn_offset" and r["shift_m"] == "-0.850" for r in m_h if r["take"] == "261029_020.WAV")
               and any("261029_021.WAV" in s and "ホーンの位置" in s for s in s_hs)))
    # B2（R97-01）: 歩く比較の静止の本（車の組）で本人が x = +2 に立つ → 中点が x = 2 を越える時刻 = 10 + (1.315 + 10 + 2)/4
    ev_x2 = [dict(events[0], **{"台本ID": "W09", "歩く人の静止x_m": "2"})]
    w_x2 = build(r1, sessions, ev_x2)[0][0]
    t_x2 = 10 + (1.315 + 12) / 4.0
    ok.append((f"静止の本人の x = +2（歩く比較の車の組）: t_cpa = {w_x2['t_cpa_s']}（理論 {t_x2:.3f}）・mic_x0_m 2", w_x2["status"] == "ok" and abs(float(w_x2["t_cpa_s"]) - t_x2) < 6e-4 and w_x2["mic_x0_m"] == "2"))
    ev_x0 = [dict(events[0], **{"台本ID": "W09", "三脚の向き": "180"})]                     # 空なら台本と向きから −2 を補う
    w_x0 = build(r1, sessions, ev_x0)[0][0]
    ok.append(("静止の x が空なら台本（車の組）と向き 180 から −2 を補う", w_x0["mic_x0_m"] == "-2" and abs(float(w_x0["t_cpa_s"]) - (10 + (1.315 + 8) / 4.0)) < 6e-4))
    # B3（R17-06）: 前の車軸（L_mid = 0）の感度 = 前輪が x = 0 を越える時刻 12.5・差 −0.329 s
    ok.append((f"前の車軸の感度: t_cpa_front = {w1['t_cpa_front_s']}（理論 12.500）・差 {w1['dt_front_s']} s", w1["status_front"] == "ok" and abs(float(w1["t_cpa_front_s"]) - 12.5) < 6e-4 and abs(float(w1["dt_front_s"]) + 1.315 / 4.0) < 6e-4))
    ok.append(("自転車には前の車軸の感度を出さない（車だけ）", w2.get("status_front", "") == ""))
    # 変更票 #69 r31 ②（2026-10-23）: 隠れた区間は前後の読みに 2 次式を当てて埋める（上限 3 s・両側に読みがある区間だけ）。旧 B4 R17-01（1 秒の線形）は置き換え
    hid_long = [dict(take="251015_001.WAV", frame="", cls="car", track="1", x="", y="2", z="", t="12.3", hidden="start"),
                dict(take="251015_001.WAV", frame="", cls="car", track="1", x="", y="2", z="", t="13.7", hidden="end")]
    r1_gap = [r for r in r1 if not (12.3 <= float(r["t"]) <= 13.7)]                                     # 隠れている間の読みは無い（実際の読みと同じ形）
    w_h = build(r1_gap + hid_long, sessions, events)[0][0]
    ok.append((f"1.4 s 隠れた区間（読み 12.0〜14.0 s）に交わり 12.829 → 2 次式で埋めて ok（t_cpa {w_h['t_cpa_s']}・埋め 1・欠測の区間 0）",
               w_h["status"] == "ok" and abs(float(w_h["t_cpa_s"]) - t1) < 2e-3 and w_h["n_hidden_filled"] == 1 and w_h["n_hidden_blocked"] == 0 and w_h["n_hidden"] == 0))
    hid_short = [dict(hid_long[0], t="12.5"), dict(hid_long[1], t="13.2")]
    w_hs = build(r1 + hid_short, sessions, events)[0][0]
    ok.append(("0.7 s の隠れも 2 次式で埋める（ok・埋め 1・n_hidden 0）", w_hs["status"] == "ok" and w_hs["n_hidden"] == 0 and w_hs["n_hidden_filled"] == 1))
    hid_3s = [dict(hid_long[0], t="11.2"), dict(hid_long[1], t="14.4")]                                  # 3.2 s 隠れる（上限 3 s を超える）→ 埋めない → 交わりが中なら欠測
    r1_gap3 = [r for r in r1 if not (11.2 <= float(r["t"]) <= 14.4)]
    w_h3 = build(r1_gap3 + hid_3s, sessions, events)[0][0]
    ok.append((f"3.2 s 隠れた区間（上限 3 s 超）は埋めず、交わりが中なら欠測（{w_h3['reason'][:14]}…・埋め 0・欠測の区間 1）",
               w_h3["status"] == "欠測" and w_h3["reason"].startswith("隠れて読めない") and w_h3["n_hidden_blocked"] == 1 and w_h3["n_hidden_filled"] == 0))
    hid_edge = [dict(hid_long[0], t="14.2"), dict(hid_long[1], t="15.5")]                                # 後ろに読みが無い（最後の読み 15.0 のあとまで隠れる）→ 外へは延ばさない
    w_he = build([r for r in r1 if float(r["t"]) < 14.2] + hid_edge, sessions, events)[0][0]
    ok.append(("片側に読みが無い隠れは埋めない（外へ延ばさない・交わりは読みの中なので ok・欠測の区間 1）", w_he["status"] == "ok" and w_he["n_hidden_blocked"] == 1 and w_he["n_hidden_filled"] == 0))
    # r31 ①: 中点が ±10 に入っている時刻。r1 = 前輪 −10 → +10（4 m/s・t 10〜15）・中点 = 前輪 − 1.315 → 入り = 前輪 −8.685 の時刻 10.329・出 = 最後の読み 15.0（まだ中）
    ok.append((f"±10 の窓: t_in10 {w1['t_in10_s']}（理論 10.329）・t_out10 {w1['t_out10_s']}（最後の読み 15.000・注あり）",
               abs(float(w1["t_in10_s"]) - (10 + 1.315 / 4.0)) < 2e-3 and abs(float(w1["t_out10_s"]) - 15.0) < 1e-6 and "まだ中" in w1["range_note"]))
    # r34: 歩く人は最初の読み（∓10・動き始め）より前はそこに立っていた → 自転車がその前に本人の位置を越える本でも「いつ」が出る
    walk_hold = rd("251015_002.WAV", "bicycle", [(10 + (x + 10) / 3.0, x) for x in range(-10, 11, 2)]) + rd("251015_002.WAV", "mic", [(14 + k * 2.0, -10 + 1.2 * k * 2.0) for k in range(0, 4)], trk="0")
    w_wh = build(walk_hold, sessions, events)[0][0]
    t_wh = 10 + 0.52 / 3.0                                                                                # 中点 −10 + 3(t−10) − 0.52 = −10（本人はまだ −10 に立っている）
    ok.append((f"歩く人の静止の扱い（r34）: 本人が動き出す前に中点が本人の位置（−10）を越える → ok・t_cpa {w_wh['t_cpa_s']}（理論 {t_wh:.3f}）",
               w_wh["status"] == "ok" and abs(float(w_wh["t_cpa_s"]) - t_wh) < 2e-3))
    ok.append(("mic の行はずらさない", all(r["converted"] == "" and r["shift_m"] == "" for r in mid if r["cls"] == "mic")))
    car_mid = [r for r in mid if r["take"] == "251015_001.WAV"]
    ok.append((f"変換済み reads: 車の x が −1.315 ずれ・印 midpoint・長さと出典の列（例 {car_mid[0]['x']}）", all(r["converted"] == "midpoint" and r["shift_m"] == "-1.315" for r in car_mid) and abs(float(car_mid[0]["x"]) - (-11.315)) < 1e-9 and "備考" in car_mid[0]["L_source"]))
    ok.append(("長さの無い session の本は止める（決まった値に置き換えない）", not stops))   # まだ r5 を入れていない → 止める理由なし
    when5, mid5, stops5 = build(reads + r5, sessions, events)
    ok.append((f"長さの無い session: 止める理由が出る（{stops5[0][:40]}…）・ほかの本はできる", len(stops5) == 1 and "長さ" in stops5[0] and len(when5) == len(when)))
    bad_len = [dict(session_id="S1", 備考="自転車の長さ: 軸の間 40 cm／車の長さ: 軸の間 263 cm")]
    _, _, stops6 = build(r2, bad_len, events)
    ok.append((f"範囲外の長さ（自転車 40 cm）は止める（{stops6[0][:30]}…）", len(stops6) == 1 and "範囲外" in stops6[0]))
    _, _, stops7 = build(mid, sessions, events)
    ok.append(("変換済みの reads を渡すと止める（2 回目の変換はしない）", len(stops7) == 1 and "変換済み" in stops7[0]))
    mid_mic_first = sorted(mid, key=lambda r: (r["cls"] != "mic"))                       # AR12-04: 先頭が mic の行でも止める
    _, _, stops7b = build(mid_mic_first, sessions, events)
    ok.append(("先頭が mic の変換済み reads でも止める（AR12-04）", len(stops7b) == 1 and "変換済み" in stops7b[0]))
    # AR12-10: 車種に「・」「／」・全角・誤記
    L10 = lengths_from_note("テープ／自転車の長さ: 軸の間 １０４ cm／車の長さ: 車種 トヨタ・プリウス／PHV・軸の間 ２６３ cm")
    ok.append((f"長さの読み取り: 車種に「・」「／」・全角の数字でも読める（NFKC で「／」は「/」に・{L10.get('car')}）", L10.get("bike", (0,))[0] == 104.0 and L10.get("car", (0,))[0] == 263.0 and L10["car"][2] == "トヨタ・プリウス/PHV"))
    bad10 = [dict(session_id="S1", 備考="車の長さ: 車種 ノート・軸の間 26.3.0 cm")]
    _, _, stops10 = build(rd("251015_001.WAV", "car", [(10, -6), (13, 6)]), bad10, events)
    ok.append((f"長さの誤記（26.3.0）は落ちずに止める一覧へ（{stops10[0][:30] if stops10 else '—'}…）", len(stops10) == 1 and "読めない" in stops10[0]))
    # AR12-11: 触れて戻る（越えない）は交わりでない・向きが逆は欠測・重複読みは欠測
    touch = rd("251015_001.WAV", "car", [(10, -6), (12, 1.315), (14, -6)])                 # 中点が x=0 に触れて戻る
    w11 = build(touch, sessions, events)[0][0]
    ok.append((f"触れて戻る本は欠測（{w11['reason'][:12]}…）", w11["status"] == "欠測"))
    dup = rd("251015_001.WAV", "car", [(10, -6), (11, -0.5), (11, 3.5), (13, 6)])
    w11b = build(dup, sessions, events)[0][0]
    ok.append(("同じ時刻の重複読みは欠測", w11b["status"] == "欠測" and "重複" in w11b["reason"]))
    # AR12-12: track が空なら 1
    blank = [dict(r, track="") for r in rd("251015_001.WAV", "car", [(10, -6), (13, 6)])]
    ok.append(("track が空なら 1（step19d の event_id の既定と同じ）", build(blank, sessions, events)[0][0]["track"] == "1"))
    short = rd("251015_001.WAV", "car", [(10, -6)])
    w8 = build(short, sessions, events)[0][0]
    ok.append(("目印 1 点は欠測（読みが足りない）", w8["status"] == "欠測" and "読みが足りない" in w8["reason"]))
    stopped = rd("251015_001.WAV", "car", [(10, -6), (12, 1.315), (16, 1.315), (18, 6)])   # 交わる所で止まっている（前輪 +1.315 = 中点 x=0 に 4 s 居る）
    w9 = build(stopped, sessions, events)[0][0]
    ok.append((f"交わる所で止まった本は欠測（{w9['reason'][:16]}…）", w9["status"] == "欠測" and w9["reason"].startswith("止まった")))
    walk_short = rd("251015_002.WAV", "bicycle", [(10, -6), (13, 6)]) + rd("251015_002.WAV", "mic", [(10, 0)], trk="0")
    w10 = build(walk_short, sessions, events)[0][0]
    ok.append(("歩行の本で本人の読みが 1 点なら欠測", w10["status"] == "欠測" and "本人" in w10["reason"]))
    p_when, p_mid = write_outputs(tmp, "T", when, mid, stops)
    back = read_csv(p_when)
    ok.append((f"書いた CSV が読める（{len(back)} 行・列 {len(OUT_COLS)}）", len(back) == len(when) and list(back[0].keys()) == OUT_COLS))
    fr_when = read_csv(tmp / "when_where_T_front.csv"); fr_reads = read_csv(tmp / "reads_front_T.csv")
    ok.append(("前の車軸の別名の出力: when_where_T_front.csv（車の t_cpa が前輪の値）・reads_front_T.csv（車は印 front・x は元の読み）",
               [r["t_cpa_s"] for r in fr_when if r["take"] == "251015_001.WAV"] == ["12.500"]
               and all(r["converted"] == "front" and abs(float(r["x"]) - float(q["x"]) + float(q["shift_m"])) < 1e-6 for r, q in zip([r for r in fr_reads if r["take"] == "251015_001.WAV"], car_mid))))
    # step19f がそのまま読める（t・x・y・z・cls・track の列は保持）
    import subprocess
    p_car = tmp / "reads_mid_car_only.csv"
    with open(p_car, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mid[0].keys())); w.writeheader(); w.writerows(r for r in mid if r["cls"] == "car")
    r19 = subprocess.run([sys.executable, str(Path(__file__).with_name("step19f_video_to_gt.py")), str(p_car), str(tmp / "gt"), "--mic-z", "1.6"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok.append((f"変換済み reads（距離クラスの行）を step19f がそのまま読める（rc {r19.returncode}）{'' if r19.returncode == 0 else r19.stderr[-300:]}", r19.returncode == 0 and (tmp / "gt").exists()))
    # Astra R142-03（10/27）: field_gate_prep の reads_A_gate.csv を --reader なしで渡しても出力は reads_mid_A.csv（A_gate にならない）。
    # 古い reads_mid_A.csv を先に置いておき、新しい中身で上書きされることも確かめる（後の命令が古いファイルを読まない）
    g = tmp / "gate"; g.mkdir(exist_ok=True)
    for p_, rows_ in ((g / "reads_A_gate.csv", r1), (g / "session.csv", sessions[:1]), (g / "events.csv", events[:1])):
        with open(p_, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys())); w.writeheader(); w.writerows(rows_)
    (g / "out").mkdir(exist_ok=True); (g / "out" / "reads_mid_A.csv").write_text("古い\n", encoding="utf-8")
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        rc_g = main(["--reads", str(g / "reads_A_gate.csv"), "--session", str(g / "session.csv"), "--events", str(g / "events.csv"), "--out", str(g / "out")])
    new_mid = read_csv(g / "out" / "reads_mid_A.csv") if (g / "out" / "reads_mid_A.csv").read_text(encoding="utf-8") != "古い\n" else []
    ok.append(("R142-03: reads_A_gate.csv を --reader なしで渡すと reads_mid_A.csv・when_where_A.csv（A_gate のファイルは作らない・古い reads_mid_A.csv は新しい中身に）・名前の規則 reads_B_gate → B・reads_A → A",
               rc_g == 0 and len(new_mid) == len(r1) and all(r["converted"] == "midpoint" for r in new_mid) and (g / "out" / "when_where_A.csv").is_file()
               and not list((g / "out").glob("*A_gate*")) and default_reader("x/reads_B_gate.csv") == "B" and default_reader("reads_A.csv") == "A"))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# ---------------------------------------------------------------- main
def default_reader(reads_p) -> str:
    """--reader が無いときの読む人の名前: reads_<人>.csv の <人>。field_gate_prep の reads_<人>_gate.csv も <人>（Astra R142-03・10/27:
    旧は A_gate になり reads_mid_A_gate.csv を書くので、後の命令が古い reads_mid_A.csv を読んでいた）。"""
    s = Path(str(reads_p)).stem
    s = s[len("reads_"):] if s.startswith("reads_") else s
    s = s[:-len("_gate")] if s.endswith("_gate") else s
    return s or "A"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    def opt(k, d=None):
        return argv[argv.index(k) + 1] if k in argv else d
    if "--selftest" in argv:
        return selftest(Path(opt("--selftest")))
    reads_p, ses_p, ev_p, out = opt("--reads"), opt("--session"), opt("--events"), opt("--out")
    if not (reads_p and ses_p and ev_p and out):
        print(__doc__); return 2
    reader = opt("--reader", default_reader(reads_p))
    reads, sessions, events = read_csv(Path(reads_p)), read_csv(Path(ses_p)), read_csv(Path(ev_p))
    when, mid, stops = build(reads, sessions, events)
    if stops:
        for s in stops:
            print("STOP", s)
        print(f"止める入力が {len(stops)} 件 → 出力を書かない（長さを記録シートの備考に入れる・生の reads を渡す）")
        return 2
    p_when, p_mid = write_outputs(Path(out), reader, when, mid, stops)
    n_ok = sum(r["status"] == "ok" for r in when); n_miss = len(when) - n_ok
    print(f"できた {n_ok} 本・欠測 {n_miss} 本 → {p_when}（いつ）・{p_mid}（どこ・中点にずらした reads）・{p_when.with_suffix('.md')}")
    for r in when:
        if r["status"] != "ok":
            print(f"  欠測 {r['take']} {r['cls']} track {r['track']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
