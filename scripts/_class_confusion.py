# -*- coding: utf-8 -*-
"""種類の見方の参考の表（2026-10-27 Opus・本人 12:35「(a) ok (b) ok」・Fable 経由）。主の指標（F・向き・距離・届いた）は変えない。
統一採点器（_score_unified.py）・知覚層（_perception_eval.py）・デモの採点（demo_field_unused_score.py）が同じ形で呼ぶ部品。

(a) 取り違えの表（混同行列・フレーム単位）: 正解の種類 × 出した種類の 9×9 に「なし」の行と列を足した 10×10。1 枠（0.1 s）ごとに、
    正解の種類の集合 G と出した種類の集合 P（どちらも 1 枠に複数ありうる）から:
      - 対角 (c, c): c ∈ G ∩ P（正解の種類を出した）
      - (c, d)・d ≠ c: c ∈ G を出さず、代わりに正解に無い d ∈ P − G を出した（取り違え・d が複数なら全部の組を 1 ずつ）
      - (c, なし): c ∈ G を出さず、正解に無い種類も何も出さなかった（見逃し）
      - (なし, d): 正解に無い d ∈ P − G を出したが、出し損ねた正解（G − P）が無い（誤報）。その枠に正解がまったく無い場合は別に数える（nogt）
    採点するフレーム（暖機のあと）と、除外区間を引いた枠だけ。
(b) 本ごとの種類の正答率: 事象（本）の窓の中で発火した枠数が一番多い種類を 1 つ選び、正解の種類と一致すれば正解・発火が無ければ不正解（未検出）。
    同じ枠数なら先に出た種類（それも同じなら番号の小さい方）。向きと距離は見ない。「種類当てだけ・向きと距離を含まない・本数が少ない」の注記を付ける。
    その窓の正解にある別の種類（例: ベルの本で実際に走っている自転車）は多数決に入れない＝取り違えに数えない（本人 10/27 14:13・二重で検出して合っていれば ok）。
md の行は表を使わず「正解の種類ごとに 1 行」（md では表を使わない・本人 9/26）。
試験: scripts/test_class_confusion.py
"""
from __future__ import annotations

from collections import defaultdict

NCLS = 9
NONE = NCLS                     # 「なし」の行・列の番号
NAMES = {0: "サイレン", 1: "クラクション", 2: "バック音", 3: "ベル", 4: "車", 5: "踏切", 6: "キック", 7: "バイク", 8: "自転車", NONE: "なし"}
NOTE_B = "種類当てだけ（向きと距離を含まない）・本数が少ない・参考"


def new_matrix() -> dict:
    return {"m": [[0] * (NCLS + 1) for _ in range(NCLS + 1)], "nogt": [0] * NCLS, "n_frames": 0}


def add_frame(M: dict, G: set, P: set) -> None:
    """1 枠を足す（G = 正解の種類の集合・P = 出した種類の集合）。"""
    m = M["m"]
    M["n_frames"] += 1
    hit = G & P
    miss = G - P
    extra = P - G
    for c in hit:
        m[c][c] += 1
    for c in miss:
        if extra:
            for d in extra:
                m[c][d] += 1
        else:
            m[c][NONE] += 1
    if not miss:
        for d in extra:
            m[NONE][d] += 1
            if not G:
                M["nogt"][d] += 1


def add_clip(M: dict, gt: dict, pred: dict, frames) -> None:
    """gt・pred = {frame: set(種類)}・frames = 採点する枠（暖機のあと・除外区間を引いたもの）。"""
    for k in frames:
        add_frame(M, set(gt.get(k, ())), set(pred.get(k, ())))


def merge(a: dict, b: dict) -> dict:
    out = new_matrix()
    for i in range(NCLS + 1):
        for j in range(NCLS + 1):
            out["m"][i][j] = a["m"][i][j] + b["m"][i][j]
    out["nogt"] = [x + y for x, y in zip(a["nogt"], b["nogt"])]
    out["n_frames"] = a["n_frames"] + b["n_frames"]
    return out


def take_class(fires: dict, window, ignore=()) -> int | None:
    """事象の窓の中で発火した枠数が一番多い種類（fires = {frame: set(種類)}・window = 枠の並び）。発火が無ければ None。
    ignore = その窓の正解にある、目的とは別の種類（2026-10-27 本人 14:13「その本の正解にある別の種類は取り違えに数えない・二重で検出して合っていれば ok」）:
    数えずに外す（例: ベルの本で実際に走っている自転車）。目的の種類か、正解に無い種類だけで多数決する。"""
    ignore = set(ignore)
    cnt, first = defaultdict(int), {}
    for k in sorted(window):
        for c in fires.get(k, ()):
            if c in ignore:
                continue
            cnt[c] += 1
            first.setdefault(c, k)
    if not cnt:
        return None
    return min(cnt, key=lambda c: (-cnt[c], first[c], c))


def take_accuracy(events: list) -> dict:
    """events = [{"gt": 正解の種類, "pred": take_class の結果, ...}] → 全体と正解の種類ごとの 正解数／本数・未検出の数。"""
    by = defaultdict(lambda: {"n": 0, "correct": 0, "undetected": 0, "as": defaultdict(int)})
    for e in events:
        g = by[e["gt"]]
        g["n"] += 1
        if e["pred"] is None:
            g["undetected"] += 1
        else:
            g["as"][e["pred"]] += 1
            g["correct"] += int(e["pred"] == e["gt"])
    n = sum(v["n"] for v in by.values())
    c = sum(v["correct"] for v in by.values())
    return {"n": n, "correct": c, "acc": (c / n if n else None), "note": NOTE_B,
            "per_class": {int(k): {"n": v["n"], "correct": v["correct"], "undetected": v["undetected"], "as": {int(d): x for d, x in v["as"].items()}}
                          for k, v in sorted(by.items())}}


DIST_CLS = (4, 6, 7, 8)          # 距離のクラス（車・キック・バイク・自転車）。ほかは警告音の窓（鳴り始めから）
FPS = 10


def read_gt(path) -> tuple:
    """正解の csv（frame,class,track,az,el,dist）→ ({frame: set(種類)}, {(種類, track): {frame: dist}})。"""
    sets, src = defaultdict(set), defaultdict(dict)
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return sets, src
    with f:
        for line in f:
            g = line.strip().split(",")
            if len(g) < 5 or not g[0].strip().lstrip("-").isdigit():
                continue
            k, c, t = int(g[0]), int(g[1]), int(g[2])
            sets[k].add(c)
            d = float(g[5]) if len(g) >= 6 and g[5] not in ("", "nan") else float("nan")
            src[(c, t)][k] = d
    return sets, src


def pred_sets(frames: dict) -> dict:
    """予測 {frame: [(種類, …), …]}（統一採点器・知覚層・デモの読み方の形）→ {frame: set(種類)}。"""
    return {k: {r[0] for r in rows} for k, rows in frames.items()}


def windows_from_gt(src: dict) -> list:
    """合成の本の事象の窓（§6 の成功の窓と同じ形・台帳が無いとき）: 距離のクラス = [最初の正解の枠 − 10, 最接近の枠 + 10]・
    それ以外 = [最初の正解の枠 − 10, 最初の正解の枠 + 40]（鳴り始め −1 s 〜 +4 s）。戻り = [(正解の種類, 枠の範囲)]。"""
    out = []
    for (c, t), fr in sorted(src.items()):
        if not fr:
            continue
        f0 = min(fr)
        if c in DIST_CLS:
            fin = {k: d for k, d in fr.items() if d == d}
            cpa = min(fin, key=lambda k: (fin[k], k)) if fin else max(fr)
            out.append((c, range(f0 - 10, cpa + 11)))
        else:
            out.append((c, range(f0 - 10, f0 + 41)))
    return out


def windows_from_ledger(rows: list, src: dict) -> list:
    """統一採点器の --events の台帳の行（除外の行は外す）→ [(正解の種類, 枠の範囲)]。警告音 = [採点上の鳴り始め − 1, +4] s・
    距離のクラス = [最初の正解の枠（無ければ最接近）− 10, 物理の最接近 + 10]。"""
    out = []
    for r in rows:
        if r.get("excluded"):
            continue
        c, t = int(r["cls"]), int(r["track"])
        if r.get("kind") == "warn":
            on = int(round(float(r["score_onset_s"]) * FPS))
            out.append((c, range(on - 10, on + 41)))
        else:
            cpa = min(int(float(r["phys_cpa_s"]) * FPS), 99)
            fr = src.get((c, t), {})
            f0 = min(fr) if fr else cpa
            out.append((c, range(f0 - 10, cpa + 11)))
    return out


def score_set(clips, gt_dir, pred: dict, minframe: int = 40, ledger: dict | None = None) -> tuple:
    """本の一覧について (取り違えの表, 本ごとの種類の正答率) を返す。採点する枠 = minframe〜99。ledger = {clip: 台帳の行}（無ければ正解から窓を作る）。"""
    from pathlib import Path as _P
    M, ev = new_matrix(), []
    frames = range(minframe, 100)
    for c in clips:
        sets, src = read_gt(_P(gt_dir) / f"{c}.csv")
        ps = pred_sets(pred.get(c, {}))
        add_clip(M, sets, ps, frames)
        wins = windows_from_ledger(ledger.get(c, []), src) if ledger is not None else windows_from_gt(src)
        for gi, win in wins:
            w = [k for k in win if minframe <= k <= 99]
            if not w:
                continue
            other = {x for k in w for x in sets.get(k, ()) if x != gi}          # その窓の正解にある別の種類は取り違えに数えない
            ev.append({"clip_id": c, "gt": gi, "pred": take_class(ps, w, ignore=other), "n_window_frames": len(w)})
    return M, take_accuracy(ev)


def write_outputs(path, per_label: dict) -> None:
    """--confusion <json>: {ラベル: (表, 正答率)} を json と、同じ名前の .md（表を使わない行）に書く。採点器の主の出力（md・json）には触れない。"""
    import json as _json
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    js = {lb: {"confusion": to_json(M), "take_accuracy": R} for lb, (M, R) in per_label.items()}
    p.write_text(_json.dumps(js, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = ["# 種類の見方（参考・主の指標は変えない・本人 10/27 12:35 (a)(b)）", ""]
    for lb, (M, R) in per_label.items():
        lines += [f"## {lb}", ""] + md_lines(M) + [""] + md_lines_take(R) + [""]
    p.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_json(M: dict) -> dict:
    return {"labels": [NAMES[i] for i in range(NCLS + 1)], "matrix": M["m"], "nogt_by_pred": M["nogt"], "n_frames": M["n_frames"],
            "rule": "行 = 正解の種類（最後は なし）・列 = 出した種類（最後は なし）・フレーム単位（暖機のあと・除外区間を引く）・参考（主の指標は変えない）"}


def md_lines(M: dict, title: str = "取り違えの表（フレーム・参考）") -> list:
    """正解の種類ごとに 1 行（表は使わない）。0 の組は書かない。"""
    m = M["m"]
    out = [f"### {title}", "", f"- 採点した枠 {M['n_frames']:,}（暖機のあと・除外区間を引く）。行 = 正解の種類、並び = 出した種類と枠数。主の指標は変えない。"]
    for i in range(NCLS + 1):
        row = m[i]
        tot = sum(row)
        if not tot:
            continue
        parts = [f"{NAMES[j]} {row[j]:,}" for j in sorted(range(NCLS + 1), key=lambda j: (j != i, -row[j], j)) if row[j]]   # 当たり（対角）を先に
        head = f"正解 {NAMES[i]}（{tot:,} 枠）" if i != NONE else f"正解 なし・または正解は当たっていて余分に出した（{tot:,} 枠）"
        out.append(f"- {head}: " + "・".join(parts))
    if any(M["nogt"]):
        out.append("- うち正解がまったく無い枠で出した: " + "・".join(f"{NAMES[d]} {x:,}" for d, x in enumerate(M["nogt"]) if x))
    return out


def md_lines_take(R: dict, title: str = "本ごとの種類の正答率（参考）") -> list:
    out = [f"### {title}", "", f"- {R['correct']}/{R['n']}" + (f"（{100 * R['acc']:.1f}%）" if R["acc"] is not None else "") + f"・{R['note']}"]
    for k, v in R["per_class"].items():
        alt = "・".join(f"{NAMES[d]} {x}" for d, x in sorted(v["as"].items(), key=lambda t: (-t[1], t[0])) if d != k)
        out.append(f"- {NAMES[k]}: {v['correct']}/{v['n']}" + (f"・未検出 {v['undetected']}" if v["undetected"] else "") + (f"・取り違え {alt}" if alt else ""))
    return out
