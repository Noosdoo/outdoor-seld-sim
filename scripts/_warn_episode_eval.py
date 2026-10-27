# -*- coding: utf-8 -*-
"""警告クラス（siren/horn/backup_beep/bike_bell/crossing）の通知を、正解のエピソードに時間と方位で帰属して採点する（2026-09-08・監査 F2 の修正）。

旧 `_v18_crossing_eval.py` は「暖機（frame<40）後の発火だけ」を数えたので、持続音を最初から正しく検出して暖機中に
通知済み（hold で以後は再通知しない）だと「発火なし」になり、4 秒遅れた検出のほうが有利だった。ここでは:

  エピソード   = 正解の (class, track) ごとの連続区間（2 フレーム以内の切れ目は同じエピソード）。方位はフレームごとの正解
  発火         = 通知層 v4.3＋hold の警告発火（`_score_unified.fires_of`）。**frame 0 から**数える（暖機中の先行通知も到達）
  到達         = エピソードの [開始−tol, 終了+tol] 内に、その時点の正解方位から ±az_tol 以内の発火が 1 つ以上ある
  遅延         = 最初に帰属した発火の frame − エピソード開始（負なら 0）。暖機中に始まるエピソードも同じ規則
  イベント外発火 = どのエピソードにも帰属しない発火（時間外、または方位違い）。列車群（--train-scenario）は正解を無視して全部イベント外
  群           = train（列車の通過音・scenario 一致）/ 有（そのクラスの正解エピソードがある）/ 無（無い）

使い方:
  python scripts/_warn_episode_eval.py <出力md> --cls 5 --plan <plan> [--clips <manifest>] [--split fold2] --meta <GTディレクトリ>
      [--minframe 40] [--tol-frames 5] [--az-tol 30] [--train-scenario v12train] <ラベル>=<val_all_causal.csv> ...
  --minframe は「暖機中開始」の区切りの表示にだけ使う（発火の数え落としには使わない）。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("score_unified", ROOT / "scripts/_score_unified.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
GROUPS = ("train", "有", "無")
GAP = 2


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def gt_episodes(meta: Path, clip: str, cls: int):
    """[(start, end, {frame: az})] を返す（正解 csv: frame,cls,track,az,el,dist）"""
    f = meta / f"{clip}.csv"
    if not f.is_file():
        raise FileNotFoundError(f"missing ground truth: {f}")
    by_track = defaultdict(dict)
    with f.open(encoding="utf-8") as handle:
        for line in handle:
            g = line.strip().split(",")
            if len(g) >= 4 and int(g[1]) == cls:
                by_track[int(g[2])][int(g[0])] = float(g[3])
    eps = []
    for trk, fr in by_track.items():
        ks = sorted(fr)
        start = prev = ks[0]
        for k in ks[1:]:
            if k - prev > GAP + 1:
                eps.append((start, prev, {j: fr[j] for j in fr if start <= j <= prev}))
                start = k
            prev = k
        eps.append((start, prev, {j: fr[j] for j in fr if start <= j <= prev}))
    return sorted(eps, key=lambda ep: (ep[0], ep[1]))


def fires_for(frames, cls: int):
    """[(frame, az)] その警告クラスの発火（frame 0 から）"""
    if not frames:
        return []
    return sorted((k, az) for k, az, tier, d, c in S.fires_of(frames) if c == cls)


def attribute(fires, episodes, tol: int, az_tol: float):
    """各発火を最も近いエピソードに帰属。戻り: per_episode 最初の帰属 frame（None なら未到達）, false 発火の一覧 [(k, az, 理由)]"""
    first = [None] * len(episodes)
    false = []
    for k, az in fires:
        best = None
        time_ok = False
        for i, (s, e, azs) in enumerate(episodes):
            if k < s - tol or k > e + tol:
                continue
            time_ok = True
            j = min(azs, key=lambda x: abs(x - k))          # その時点に最も近い正解フレームの方位
            if dang(az, azs[j]) <= az_tol:
                d = abs(k - s)
                if best is None or d < best[0]:
                    best = (d, i)
        if best is None:
            false.append((k, az, "方位違い" if time_ok else "時間外"))
        else:
            i = best[1]
            if first[i] is None or k < first[i]:
                first[i] = k
    return first, false


def evaluate(pred_path: Path, clips, meta: Path, cls: int, group_of, minframe: int, tol: int, az_tol: float):
    pred = S.load_pred_rows(pred_path, horiz=False)
    acc = {g: dict(n=0, ep=0, reached=0, delays=[], warm=0, warm_reached=0, false=0, false_clips=0, fires=0, fire_clips=0) for g in GROUPS}
    for c in clips:
        g = group_of(c)
        a = acc[g]
        a["n"] += 1
        fires = fires_for(pred.get(c, {}), cls)
        a["fires"] += len(fires)
        a["fire_clips"] += 1 if fires else 0
        if g == "train":
            a["false"] += len(fires)
            a["false_clips"] += 1 if fires else 0
            continue
        eps = gt_episodes(meta, c, cls)
        first, false = attribute(fires, eps, tol, az_tol)
        a["ep"] += len(eps)
        for (s, e, _), k in zip(eps, first):
            warm = s < minframe
            a["warm"] += 1 if warm else 0
            if k is not None:
                a["reached"] += 1
                a["warm_reached"] += 1 if warm else 0
                a["delays"].append(max(0, k - s))
        a["false"] += len(false)
        a["false_clips"] += 1 if false else 0
    return acc


def pct(num, den):
    return f"{100.0 * num / den:.1f}%" if den else "-"


def fmt_row(label, g, a):
    d = np.array(a["delays"], dtype=float)
    med = f"{np.median(d) / 10:.1f}" if len(d) else "-"
    le1 = pct(int(np.sum(d <= 10)), len(d)) if len(d) else "-"
    # 監査 R7（2026-09-09）: 率は 3 桁で示し、整数の分子/分母を併記する（表示値どうしの比較で 4/336 と 5/336 が同じ 0.01 に見えた）
    fpc = f"{a['false'] / max(a['n'], 1):.3f} ({a['false']}/{a['n']})"
    fic = f"{a['fires'] / max(a['n'], 1):.3f} ({a['fires']}/{a['n']})"
    if g == "train":
        return f"| {label} | {g} | {a['n']} | - | - | - | - | - | {fpc} | {pct(a['false_clips'], a['n'])} | {fic} |"
    return (f"| {label} | {g} | {a['n']} | {a['ep']} | {pct(a['reached'], a['ep'])} | {med} | {le1} | "
            f"{a['warm_reached']}/{a['warm']} | {fpc} | {pct(a['false_clips'], a['n'])} | {fic} |")


HEADER = ("| モデル | 群 | 本 | エピソード | 到達率 | 遅延中央値[s] | 遅延≤1s | 暖機中開始(到達/数) | イベント外発火/本 | 外発火あり本 | 発火/本 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def main() -> int:
    argv = list(sys.argv[1:])
    cls = int(arg(argv, "--cls", "5"))
    plan = ROOT / arg(argv, "--plan")
    clips_f = arg(argv, "--clips", "")
    split = arg(argv, "--split", "fold2")
    meta = ROOT / arg(argv, "--meta")
    minframe = int(arg(argv, "--minframe", "40"))
    tol = int(arg(argv, "--tol-frames", "5"))
    az_tol = float(arg(argv, "--az-tol", "30"))
    train_scen = arg(argv, "--train-scenario", "v12train")
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]
    rows = list(csv.DictReader(open(plan, encoding="utf-8")))
    scen = {r["clip_id"]: (r.get("scenario") or "v11core") for r in rows}
    if clips_f:
        clips = [r["clip_id"] for r in csv.DictReader(open(ROOT / clips_f, encoding="utf-8"))]
    else:
        clips = [r["clip_id"] for r in rows if r.get("split") == split]
    if not clips or len(set(clips)) != len(clips) or not items:
        raise ValueError("nonempty unique clip manifest and predictions required")
    has_ep = {c: bool(gt_episodes(meta, c, cls)) for c in clips}

    def group_of(c):
        if scen.get(c, "v11core") == train_scen:
            return "train"
        return "有" if has_ep[c] else "無"

    ng = defaultdict(int)
    for c in clips:
        ng[group_of(c)] += 1
    lines = [f"# 警告クラス {cls} の到達と誤発火（エピソード帰属・時間 ±{tol} frame・方位 ±{az_tol:g} 度・発火は frame 0 から）", "",
             f"- クリップ: {len(clips)} 本（train {ng['train']} / 有 {ng['有']} / 無 {ng['無']}）・GT = {meta.relative_to(ROOT)}・plan = {plan.relative_to(ROOT)}",
             f"- train = scenario が {train_scen}（正解を無視して発火は全部イベント外）。有 = そのクラスの正解エピソードがある。無 = 無い",
             f"- 暖機中開始 = エピソードの開始が frame<{minframe}。到達の数え方は暖機の内外で同じ（先行通知も到達）", "", *HEADER]
    missing = []
    for label, path in items:
        p = ROOT / path
        if not p.exists():
            lines.append(f"| {label} | （未着: {p.name}） |")
            missing.append(label)
            continue
        acc = evaluate(p, clips, meta, cls, group_of, minframe, tol, az_tol)
        for g in GROUPS:
            if acc[g]["n"]:
                lines.append(fmt_row(label, g, acc[g]))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if missing:
        print(f"未着: {missing}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
