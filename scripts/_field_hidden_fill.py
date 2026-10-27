# -*- coding: utf-8 -*-
"""実録の読み（step19h の reads）の共通の部品（変更票 #69 r31 ②・r34・2026-10-23 Fable）。step19k と step19l が同じ関数を使う。

1. 隠れた区間の埋め方（r31 ②・B4 R17-01 の置き換え）:
   前輪（歩く人は足元）が隠れた区間（hidden = start〜end）は、前後の読みに 2 次式（等加速度）を最小二乗で当てて埋める。
   - 上限 3 秒（隠れ始めから見え始めまで）。それより長い区間は欠測（埋めない・区間の中の正解は作らない・交わりが中なら欠測）。
   - 前と後ろの両方に読みがある区間だけ埋める（外へは延ばさない）。
   - 当てる読み = 隠れ始めの前 FIT_SPAN_S 秒以内の読みと、見え始めの後 FIT_SPAN_S 秒以内の読み（各側 1 点以上・合わせて 3 点以上で 2 次・2 点なら 1 次）。
   - 埋めた点は前後の読みの間に FILL_STEP_S ごとに置く（filled=True の印）。埋めた区間の数と本数を呼ぶ側が報告する。
2. 歩く人の行の前後（r34）: 最初の読み（∓10・動き始めたコマ）より前はその位置に、最後の読み（±10・止まったコマ）より後ろはその位置に立っていた
   として扱う（hold_ends）。
  python scripts/_field_hidden_fill.py --selftest
"""
from __future__ import annotations

import sys

HIDE_MAX_S = 3.0          # r31 ②: 隠れた区間の上限（これより長い区間は欠測）
FIT_SPAN_S = 3.0          # 当てる読みを取る幅（隠れ始めの前・見え始めの後）
FILL_STEP_S = 0.1         # 埋めた点の間隔


def hidden_intervals(rows: list[dict]) -> list[tuple[float, float]]:
    """step19h の隠れた印（hidden = start／end・x は空）→ [(隠れ始め t, 見え始め t)]。印が不完全なら ValueError（R72-05・黙って隠れなしにしない）。"""
    hid = sorted((float(r["t"]), (r.get("hidden") or "").strip()) for r in rows if (r.get("hidden") or "").strip())
    out, t_hide = [], None
    for t, kind in hid:
        if kind == "start":
            if t_hide is not None:
                raise ValueError(f"隠れの印が不完全: 隠れ始め {t_hide:.2f} s のあと見え始めが無いまま次の隠れ始め {t:.2f} s（R72-05）")
            t_hide = t
        elif kind == "end":
            if t_hide is None:
                raise ValueError(f"隠れの印が不完全: 隠れ始めの無い見え始め {t:.2f} s（R72-05）")
            out.append((t_hide, t)); t_hide = None
        else:
            raise ValueError(f"隠れの印の値が start／end でない: {kind!r}")
    if t_hide is not None:
        raise ValueError(f"隠れの印が不完全: 隠れ始め {t_hide:.2f} s のあと見え始めが無い（押し忘れ・R72-05）")
    return out


def _lstsq_poly(ts: list[float], xs: list[float], deg: int) -> list[float]:
    """最小二乗の多項式（deg 2 か 1）。numpy 無しでも動く小さな正規方程式（t は中心化して条件を良くする）。戻り = 係数 [c0, c1, c2]（x = c0 + c1 τ + c2 τ²・τ = t − t0）。"""
    t0 = sum(ts) / len(ts)
    tau = [t - t0 for t in ts]
    n = deg + 1
    A = [[sum(tt ** (i + j) for tt in tau) for j in range(n)] for i in range(n)]
    b = [sum(x * (tt ** i) for tt, x in zip(tau, xs)) for i in range(n)]
    # ガウスの消去（n ≤ 3）
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12:
            raise ValueError("最小二乗が解けない（同じ時刻の読み）")
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    coef = [M[i][n] / M[i][i] for i in range(n)]
    return [t0] + coef


def _poly_at(p: list[float], t: float) -> float:
    t0, c = p[0], p[1:]
    tau = t - t0
    return sum(ci * tau ** i for i, ci in enumerate(c))


def fill_hidden(pts: list[tuple], intervals: list[tuple[float, float]], max_s: float = HIDE_MAX_S, span_s: float = FIT_SPAN_S,
                step_s: float = FILL_STEP_S) -> tuple[list[tuple], list[tuple[float, float]], list[dict]]:
    """pts = 読みの点 [(t, x, ...)] 昇順（x は道に沿った位置・残りの列は前後の読みから線形に補う）。intervals = hidden_intervals の出力。
    戻り = (埋めた点を足した点列（埋めた点は末尾に True の印）, 埋められなかった区間 [(前の読み t, 後の読み t)]（欠測の区間）, 記録 [dict])。
    埋められなかった区間 = 3 秒より長い・前か後ろに読みが無い・当てる読みが足りない。"""
    pts = sorted(pts, key=lambda p: p[0])
    out = list(pts); blocked, log = [], []
    times = [p[0] for p in pts]
    for t_hide, t_show in intervals:
        prev = [p for p in pts if p[0] <= t_hide]; nxt = [p for p in pts if p[0] >= t_show]
        rec = dict(t_hide=t_hide, t_show=t_show, dur_s=round(t_show - t_hide, 3), filled=False, reason="", n_fit=0, deg=0, n_points=0)
        if not prev or not nxt:
            a = prev[-1][0] if prev else t_hide; b = nxt[0][0] if nxt else t_show
            rec["reason"] = "前か後ろに読みが無い（外へは延ばさない）"; blocked.append((a, b)); log.append(rec); continue
        a, b = prev[-1][0], nxt[0][0]
        if t_show - t_hide > max_s + 1e-9:
            rec["reason"] = f"隠れた時間 {t_show - t_hide:.2f} s が上限 {max_s:g} s を超える"; blocked.append((a, b)); log.append(rec); continue
        fit = [p for p in prev if p[0] >= t_hide - span_s] + [p for p in nxt if p[0] <= t_show + span_s]
        ts = [p[0] for p in fit]; xs = [p[1] for p in fit]
        if len(set(ts)) < 2:
            rec["reason"] = "当てる読みが足りない"; blocked.append((a, b)); log.append(rec); continue
        deg = 2 if len(set(ts)) >= 3 else 1
        try:
            poly = _lstsq_poly(ts, xs, deg)
        except ValueError as e:
            rec["reason"] = str(e); blocked.append((a, b)); log.append(rec); continue
        n_new = 0
        existing = {round(p[0], 4) for p in pts}
        t = a + step_s
        while t < b - 1e-9:
            if round(t, 4) in existing:                               # 既にある読みの時刻には置かない（同じ時刻の読みを 2 つにしない）
                t += step_s; continue
            x = _poly_at(poly, t)
            rest = []
            for k in range(2, len(prev[-1])):                       # 残りの列（y・z など）は前後の読みから線形に
                v0, v1 = prev[-1][k], nxt[0][k]
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0 is not None and v1 is not None:
                    rest.append(v0 + (v1 - v0) * (t - a) / (b - a) if b != a else v0)
                else:
                    rest.append(v0)
            out.append((round(t, 4), x, *rest, True)); n_new += 1
            t += step_s
        rec.update(filled=True, n_fit=len(fit), deg=deg, n_points=n_new); log.append(rec)
    out.sort(key=lambda p: p[0])
    return out, blocked, log


def strip_fill_flag(pts: list[tuple], width: int) -> list[tuple]:
    """埋めた点の印（末尾の True）を落として幅 width の組に揃える。"""
    return [tuple(p[:width]) for p in pts]


def hold_ends(pts: list[tuple], t_lo: float, t_hi: float) -> list[tuple]:
    """r34: 最初の読みより前（t_lo まで）は最初の位置に、最後の読みより後ろ（t_hi まで）は最後の位置に立っていた点として足す。pts は (t, x, ...)。"""
    if not pts:
        return pts
    pts = sorted(pts, key=lambda p: p[0])
    out = list(pts)
    if t_lo < pts[0][0]:
        out.insert(0, (t_lo, *pts[0][1:]))
    if t_hi > pts[-1][0]:
        out.append((t_hi, *pts[-1][1:]))
    return out


def _selftest() -> int:
    ok = []
    # 等加速度の軌道 x = 2 t + 0.1 t² を 5〜7 s で隠す → 2 次式で埋めると 6 s の値が真値に近い
    truth = lambda t: 2.0 * t + 0.1 * t * t
    pts = [(t, truth(t), 1.0) for t in (2.0, 3.0, 4.0, 4.8, 7.2, 8.0, 9.0, 10.0)]
    rows = [{"t": "5.0", "hidden": "start"}, {"t": "7.0", "hidden": "end"}]
    iv = hidden_intervals(rows)
    ok.append(("hidden_intervals: start〜end を 1 区間に", iv == [(5.0, 7.0)]))
    filled, blocked, log = fill_hidden(pts, iv)
    mid = [p for p in filled if len(p) == 4 and abs(p[0] - 6.0) < 1e-6]
    ok.append(("2 次式の埋め: 6.0 s の値が真値 ±0.05", bool(mid) and abs(mid[0][1] - truth(6.0)) < 0.05 and not blocked and log[0]["deg"] == 2 and log[0]["filled"]))
    ok.append(("埋めた点は 4.8〜7.2 の間に 0.1 s ごと（23 点）・y は線形", log[0]["n_points"] == 23 and all(abs(p[2] - 1.0) < 1e-9 for p in filled if len(p) == 4)))
    f2, b2, l2 = fill_hidden(pts, [(5.0, 8.5)])
    ok.append(("3 秒より長い区間は埋めない → 欠測の区間（前の読み 4.8〜後の読み 9.0）", not l2[0]["filled"] and b2 == [(4.8, 9.0)]))
    f3, b3, l3 = fill_hidden([(2.0, 4.0, 1.0), (3.0, 6.0, 1.0)], [(3.5, 4.5)])
    ok.append(("後ろに読みが無い区間は埋めない（外へ延ばさない）", not l3[0]["filled"] and b3 == [(3.0, 4.5)]))
    f4, b4, l4 = fill_hidden([(2.0, 4.0, 1.0), (5.0, 10.0, 1.0)], [(3.0, 4.0)])
    ok.append(("読みが 2 点だけなら 1 次式で埋める", l4[0]["filled"] and l4[0]["deg"] == 1 and abs([p for p in f4 if len(p) == 4 and abs(p[0] - 3.5) < 1e-6][0][1] - 7.0) < 1e-6))
    try:
        hidden_intervals([{"t": "5.0", "hidden": "start"}]); ok.append(("印が不完全なら止まる", False))
    except ValueError:
        ok.append(("印が不完全なら止まる", True))
    h = hold_ends([(12.0, -10.0, 5.0), (20.0, 10.0, 5.0)], 9.0, 25.0)
    ok.append(("hold_ends: 前は −10・後ろは +10 に立っていた点を足す", h[0] == (9.0, -10.0, 5.0) and h[-1] == (25.0, 10.0, 5.0) and len(h) == 4))
    ok.append(("strip_fill_flag: 印を落として幅をそろえる", strip_fill_flag(filled, 3)[0] == pts[0] and all(len(p) == 3 for p in strip_fill_flag(filled, 3))))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 2)
