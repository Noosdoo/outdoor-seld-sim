# -*- coding: utf-8 -*-
"""場面単位で対応付けた差の 95% 区間（bootstrap）— 統一採点器の指標について、2 つの予測（A → B）の差がゆらぎか見るための道具（2026-09-09）。

なぜ場面単位か: fold2 の val は同じ場面（同一 seed = 同じ音源・軌跡）を 2 つの高さで描いた対（v16/v18: mix N と N+10000、v19: mix N と N+1000）を含む。
対を壊して本ごとに再抽出すると区間が狭すぎる。ここでは「場面」を再抽出の単位にし、同じ再抽出で A と B の両方を採点して差を取る（対応付き）。

指標（`_score_unified.aggregate` と同じ定義）: 強到達 strong・至近到達 crit・安全抑制 safe・至近捕捉(全GT) cap_all・距離誤差 dist_err（対応ペアの相対誤差の中央値・%）。
使い方:
  python scripts/_paired_ci.py <出力md> --plan <plan> [--clips <manifest>] [--split fold2] --meta <GTディレクトリ>
      [--minframe 40] [--n-boot 1000] [--seed 0] <ラベル>=<csvA>,<csvB> ...
  差は B − A。区間は percentile 2.5〜97.5。「0 を含まない」= その向きの差がゆらぎでは説明しにくい（種は 1 つなので学習のゆらぎは含まない）。
"""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("score_unified", ROOT / "scripts/_score_unified.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
METRICS = ("strong", "crit", "safe", "cap_all", "dist_err")
JP = {"strong": "強到達", "crit": "至近到達", "safe": "安全抑制", "cap_all": "至近捕捉(全GT)", "dist_err": "距離誤差"}


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


import sys as _sys
_sys.path.insert(0, str(ROOT / "src"))
from outdoor_seld import clip_ids as _C  # noqa: E402  （2026-09-22 改名。旧名を渡しても新名の鍵を返す）


def scene_key(clip: str) -> str:
    """同じ場面（seed）の 2 高さを 1 つにまとめる鍵（新名で返す）。v16/v18: 番号 N と N+10000。v19: 20001 + k·2000 + i と +1000。"""
    try:
        split, source, n = _C.parse(clip)
    except ValueError:
        return clip
    if n >= 20001:
        base = 20001 + ((n - 20001) // 2000) * 2000
        n = base + (n - base) % 1000
    elif n > 9000:
        n = n - 10000
    return _C.make(split, source, n)


def per_clip_arrays(pred, meta: Path, clips, minframe: int):
    """クリップごとの数（bootstrap で足し合わせる）と相対誤差のリスト"""
    st = [S.clip_stats(pred, meta, c, minframe) for c in clips]
    A = {k: np.array([x[k] for x in st], dtype=float) for k in ("n_strong", "n_close_gt", "n_close_cap")}
    A["crit_hit"] = np.array([x["crit"][0] for x in st], dtype=float)
    A["crit_tot"] = np.array([x["crit"][1] for x in st], dtype=float)
    A["safe_miss"] = np.array([x["safe"][0] for x in st], dtype=float)
    A["safe_tot"] = np.array([x["safe"][1] for x in st], dtype=float)
    A["rel"] = [np.array(x["rel"], dtype=float) for x in st]
    return A


def metrics_of(A, idx, counts):
    """idx（場面の再抽出で選ばれたクリップ番号）と counts（重複回数）から指標を計算"""
    w = counts.astype(float)
    def T(k):
        return float(np.dot(A[k][idx], w))
    n_crit = T("crit_tot")
    out = {"strong": 100 * T("n_strong") / max(n_crit, 1), "crit": 100 * T("crit_hit") / max(n_crit, 1),
           "safe": 100 * T("safe_miss") / max(T("safe_tot"), 1), "cap_all": 100 * T("n_close_cap") / max(T("n_close_gt"), 1)}
    rel = np.concatenate([np.repeat(A["rel"][i], int(c)) for i, c in zip(idx, counts) if len(A["rel"][i])]) if len(idx) else np.array([])
    out["dist_err"] = float(100 * np.median(rel)) if rel.size else float("nan")
    return out


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    clips_f = arg(argv, "--clips", "")
    split = _C.split_to_new(arg(argv, "--split", "val"))      # 2026-09-22 改名（旧 fold2 = val）
    meta = ROOT / arg(argv, "--meta")
    minframe = int(arg(argv, "--minframe", "40"))
    n_boot = int(arg(argv, "--n-boot", "1000"))
    seed = int(arg(argv, "--seed", "0"))
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]
    rows = list(csv.DictReader(open(plan, encoding="utf-8")))
    if clips_f:
        clips = [_C.normalize(r["clip_id"]) for r in csv.DictReader(open(ROOT / clips_f, encoding="utf-8"))]   # --clips の ID も新名に正規化（Opus 5.5 監査 9/23 F01）
    else:
        clips = [_C.normalize(r["clip_id"]) for r in rows if _C.split_to_new(r.get("split", "")) == split]
    groups = {}
    for i, c in enumerate(clips):
        groups.setdefault(scene_key(c), []).append(i)
    glist = list(groups.values())
    rng = np.random.default_rng(seed)
    cache = {}

    def arrays(path):
        p = str(ROOT / path)
        if p not in cache:
            cache[p] = per_clip_arrays(S.load_pred_rows(Path(p), horiz=False), meta, clips, minframe)
        return cache[p]

    lines = [f"# 場面単位の対応付き差の 95% 区間（bootstrap {n_boot} 回・seed {seed}）", "",
             f"- クリップ {len(clips)} 本 = 場面 {len(glist)}（同じ seed の 2 高さを 1 場面）・GT = {meta.relative_to(ROOT)}・plan = {plan.relative_to(ROOT)}・frame≥{minframe}",
             "- 差 = B − A。区間は percentile 2.5〜97.5。種は 1 つなので学習のゆらぎは含まない（データの抽出のゆらぎだけ）", "",
             "| 比較 | 指標 | A | B | 差 | 95% 区間 | 0 を含む |", "| --- | --- | --- | --- | --- | --- | --- |"]
    all_idx = np.arange(len(clips))
    ones = np.ones(len(clips))
    for label, spec in items:
        pa, pb = spec.split(",")
        A, B = arrays(pa), arrays(pb)
        ma, mb = metrics_of(A, all_idx, ones), metrics_of(B, all_idx, ones)
        diffs = {m: [] for m in METRICS}
        for _ in range(n_boot):
            pick = rng.integers(0, len(glist), len(glist))
            sel = np.concatenate([glist[g] for g in pick])
            idx, counts = np.unique(sel, return_counts=True)
            xa, xb = metrics_of(A, idx, counts), metrics_of(B, idx, counts)
            for m in METRICS:
                diffs[m].append(xb[m] - xa[m])
        for m in METRICS:
            d = np.array(diffs[m])
            lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
            unit = "%" if m != "dist_err" else "pt"
            lines.append(f"| {label} | {JP[m]} | {ma[m]:.1f} | {mb[m]:.1f} | {mb[m] - ma[m]:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {'含む' if lo <= 0 <= hi else '**含まない**'} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
