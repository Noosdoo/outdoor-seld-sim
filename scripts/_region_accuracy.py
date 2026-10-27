# -*- coding: utf-8 -*-
"""本人の例（左 2 m・後ろ 2 m = 方位 135°・水平距離 2.83 m）の近くにいる車で、位置の精度を出す（交差 val・v18 e149・入り 0.7/保持 0.3）"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\satos\research\outdoor_seld_e2e")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("h2", ROOT / "scripts" / "_hysteresis_grid2.py")
H2 = importlib.util.module_from_spec(spec); sys.modules["h2"] = H2; spec.loader.exec_module(H2)
D, PE = H2.D, H2.PE
sdir = ROOT / "out/detect_scores_2026-09-16/v2/scores_v18ft_e149_v12causal/scores"
meta = ROOT / "out/dataset_outdoor_siren_v12/metadata_dist_h"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
files = sorted(sdir.glob("*.csv"))[:N]

REGIONS = {
    "本人の例の近く（後左 110〜160°・2〜4 m）": lambda az, d: 110 <= az <= 160 and 2.0 <= d <= 4.0,
    "同じ距離帯 全方位（2〜4 m）": lambda az, d: 2.0 <= d <= 4.0,
    "至近（≤1.5 m）": lambda az, d: d <= 1.5,
    "3.2〜10 m": lambda az, d: 3.2 < d <= 10.0,
}


def xy(az, d):
    a = math.radians(az); return d * math.cos(a), d * math.sin(a)


res = {}
for name in REGIONS:
    res[name] = {"単独": dict(n=0, hit=0, in20=0, az=[], el=[], dd=[], rel=[], xy=[]), "多重": dict(n=0, hit=0, in20=0, az=[], el=[], dd=[], rel=[], xy=[])}

for f in files:
    fs = D.load_scores(f)
    dec = H2.decode_general(fs, 0.7, 0.3, 20.0)
    frames = {k: [(c, az, el, d) for c, az, el, d, _ in v] for k, v in dec.items()}
    gt = PE.load_gt_all(meta, f.stem)
    ntr = len({(g[0], g[1]) for k in gt for g in gt[k] if g[0] in PE.DIST_CLS})
    kind = "多重" if ntr >= 2 else "単独"
    for k in sorted(set(gt) | set(frames)):
        if k < 40:
            continue
        gts_all = gt.get(k, []); preds_all = frames.get(k, [])
        for cls in PE.DIST_CLS:
            gts = [g for g in gts_all if g[0] == cls]
            preds = [p for p in preds_all if p[0] == cls]
            pairs = PE.match_frame(gts, preds) if gts and preds else []
            matched = {gi: (pi, err) for gi, pi, err in pairs}
            for gi, g in enumerate(gts):
                _, _, gaz, gel, gd = g
                for name, fn in REGIONS.items():
                    if not fn(gaz, gd):
                        continue
                    r = res[name][kind]; r["n"] += 1
                    if gi in matched:
                        pi, err = matched[gi]; _, paz, pel, pd = preds[pi]
                        r["hit"] += 1; r["in20"] += int(err <= 20); r["az"].append(err); r["el"].append(abs(pel - gel))
                        if np.isfinite(pd) and np.isfinite(gd):
                            r["dd"].append(abs(pd - gd)); r["rel"].append(abs(pd - gd) / max(gd, 0.1))
                            gx, gy = xy(gaz, gd); px, py = xy(paz, pd); r["xy"].append(math.hypot(px - gx, py - gy))

print(f"交差 val {len(files)} 本・v18 e149・入り 0.7／保持 0.3・frame≥40・距離クラス（car/kick/bike）・1 対 1 対応")
print("| 領域 | 台数 | 正解フレーム | 検出率 | 20° 以内 | 方位誤差 中央/RMSE/90% | 仰角誤差 中央 | 距離誤差 中央/RMSE | 相対 中央 | 位置誤差 中央/RMSE/90% |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for name in REGIONS:
    for kind in ("単独", "多重"):
        r = res[name][kind]
        if r["n"] == 0:
            continue
        az = np.array(r["az"]); el = np.array(r["el"]); dd = np.array(r["dd"]); rel = np.array(r["rel"]); xyv = np.array(r["xy"])
        rmse = lambda v: math.sqrt(float(np.mean(v ** 2))) if len(v) else float("nan")
        print(f"| {name} | {kind} | {r['n']:,} | {100*r['hit']/r['n']:.1f}% | {100*r['in20']/r['n']:.1f}% | {np.median(az):.1f}° / {rmse(az):.1f}° / {np.percentile(az,90):.0f}° | {np.median(el):.1f}° | {np.median(dd):.2f} / {rmse(dd):.2f} m | {100*np.median(rel):.1f}% | {np.median(xyv):.2f} / {rmse(xyv):.2f} / {np.percentile(xyv,90):.2f} m |")
