# -*- coding: utf-8 -*-
"""度合い（矢印の長さ）と、方位のずれ・当たり外れの関係（交差 val・v18 e149・frame≥40・距離クラス）。2026-09-17・講義ノート_矢印の長さと閾値 §4 の表を作る。
使い方: python scripts/_norm_vs_error.py [本数=600]"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\satos\research\outdoor_seld_e2e")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("ds", ROOT / "scripts" / "_detect_threshold_sweep.py")
D = importlib.util.module_from_spec(spec); sys.modules["ds"] = D; spec.loader.exec_module(D)
PE = D.PE
sdir = ROOT / "out/detect_scores_2026-09-16/v2/scores_v18ft_e149_v12causal/scores"
meta = ROOT / "out/dataset_outdoor_siren_v12/metadata_dist_h"
files = sorted(sdir.glob("*.csv"))[: int(sys.argv[1]) if len(sys.argv) > 1 else 600]
BINS = [(0.05, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0), (1.0, 9.9)]
stat = {b: dict(n=0, matched=0, in20=0, errs=[], rel=[]) for b in BINS}
gt_total = 0; gt_by_bin = defaultdict(int); gt_miss = 0
raw_n = 0; raw_gt = defaultdict(int)


def b_of(v):
    for b in BINS:
        if b[0] < v <= b[1]:
            return b
    return None


for f in files:
    fs = D.load_scores(f)
    dec = D.decode(fs, 0.05)
    gt = PE.load_gt_all(meta, f.stem)
    for k in sorted(set(gt) | set(dec)):
        if k < 40:
            continue
        gts_all = gt.get(k, [])
        preds_all = dec.get(k, [])
        for cls in PE.DIST_CLS:
            gts = [g for g in gts_all if g[0] == cls]
            ps = [p for p in preds_all if p[0] == cls]
            preds = [(c, az, el, d) for c, az, el, d, sc in ps]
            scs = [sc for *_, sc in ps]
            pairs = PE.match_frame(gts, preds) if gts and preds else []
            used = set()
            gt_total += len(gts)
            for gi, pi, err in pairs:
                used.add(pi)
                b = b_of(scs[pi])
                if b is None:
                    continue
                gt_by_bin[b] += 1
                st = stat[b]; st["n"] += 1; st["matched"] += 1; st["in20"] += int(err <= 20); st["errs"].append(err)
                gd = gts[gi][4]; pd = preds[pi][3]
                if np.isfinite(gd) and np.isfinite(pd) and gd > 0:
                    st["rel"].append(abs(pd - gd) / gd)
            for pi, sc in enumerate(scs):
                if pi in used:
                    continue
                b = b_of(sc)
                if b is not None:
                    stat[b]["n"] += 1
            gt_miss += len(gts) - len(pairs)

print(f"交差 val {len(files)} 本・frame≥40・距離クラス・統合後の予測を長さ 0.05 超で全部出し、正解と 1 対 1 で対応（方位の門なし）")
print(f"正解フレーム {gt_total:,}（対応なし = 見逃し {gt_miss:,} = {100*gt_miss/max(gt_total,1):.1f}%）")
print("| 長さの帯 | 予測数 | 正解と対応した割合 | 対応のうち 20° 以内 | 方位誤差 中央 / 90% 点 | 相対距離誤差 中央 | 正解フレームのうちこの帯で当たった割合 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for b in BINS:
    st = stat[b]
    if st["n"] == 0:
        continue
    e = np.array(st["errs"]) if st["errs"] else np.array([np.nan])
    r = np.array(st["rel"]) if st["rel"] else np.array([np.nan])
    lab = f"{b[0]:.2f}〜{b[1]:.2f}" if b[1] < 9 else f">{b[0]:.2f}"
    print(f"| {lab} | {st['n']:,} | {100*st['matched']/st['n']:.1f}% | {100*st['in20']/max(st['matched'],1):.1f}% | {np.nanmedian(e):.1f}° / {np.nanpercentile(e,90):.0f}° | {100*np.nanmedian(r):.1f}% | {100*gt_by_bin[b]/max(gt_total,1):.1f}% |")
