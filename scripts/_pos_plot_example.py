# -*- coding: utf-8 -*-
"""正解の位置とモデルの位置のずれを絵にする（交差 val・v18 e149・入り 0.7/保持 0.3・上から見た図）"""
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\satos\research\outdoor_seld_e2e")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("h2", ROOT / "scripts" / "_hysteresis_grid2.py")
H2 = importlib.util.module_from_spec(spec); sys.modules["h2"] = H2; spec.loader.exec_module(H2)
D, PE = H2.D, H2.PE
sdir = ROOT / "out/detect_scores_2026-09-16/v2/scores_v18ft_e149_v12causal/scores"
meta = ROOT / "out/dataset_outdoor_siren_v12/metadata_dist_h"
OUT = ROOT / "out/perception_pos_plot_2026-09-17"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]


def xy(az, d):
    a = math.radians(az); return d * math.cos(a), d * math.sin(a)


files = sorted(sdir.glob("*.csv"))
# 1) 例の近く（後左 2〜4 m）の対応ペアを集める（正解→予測の矢印）
pairs_single, pairs_multi = [], []
tracks_single, tracks_multi = [], []   # 軌跡の例
for f in files[:1800]:
    fs = D.load_scores(f)
    dec = H2.decode_general(fs, 0.7, 0.3, 20.0)
    frames = {k: [(c, az, el, d) for c, az, el, d, _ in v] for k, v in dec.items()}
    gt = PE.load_gt_all(meta, f.stem)
    ntr = len({(g[0], g[1]) for k in gt for g in gt[k] if g[0] in PE.DIST_CLS})
    single = ntr == 1
    traj = {}   # (cls,trk) -> [(k, gx, gy, px, py)]
    for k in sorted(set(gt) | set(frames)):
        if k < 40:
            continue
        gts_all = gt.get(k, []); preds_all = frames.get(k, [])
        for cls in PE.DIST_CLS:
            gts = [g for g in gts_all if g[0] == cls]; preds = [p for p in preds_all if p[0] == cls]
            pairs = PE.match_frame(gts, preds) if gts and preds else []
            matched = {gi: pi for gi, pi, _ in pairs}
            for gi, g in enumerate(gts):
                _, trk, gaz, gel, gd = g
                gx, gy = xy(gaz, gd)
                if gi in matched:
                    _, paz, pel, pd = preds[matched[gi]]
                    if not (np.isfinite(pd) and np.isfinite(gd)):
                        continue
                    px, py = xy(paz, pd)
                    if 110 <= gaz <= 160 and 2.0 <= gd <= 4.0:
                        (pairs_single if single else pairs_multi).append((gx, gy, px, py))
                    traj.setdefault((cls, trk), []).append((k, gx, gy, px, py))
                else:
                    traj.setdefault((cls, trk), []).append((k, gx, gy, np.nan, np.nan))
    # 軌跡の例: 例の近く（後左）を通る車を含むクリップを数本
    for key, seq in traj.items():
        if any(110 <= math.degrees(math.atan2(gy, gx)) <= 160 and 2.0 <= math.hypot(gx, gy) <= 4.0 for _, gx, gy, _, _ in seq) and len(seq) >= 30:
            (tracks_single if single else tracks_multi).append((f.stem, key, seq))
    if len(tracks_single) >= 40 and len(tracks_multi) >= 40 and len(pairs_multi) > 500:
        pass

print(f"pairs single {len(pairs_single)} multi {len(pairs_multi)}; tracks single {len(tracks_single)} multi {len(tracks_multi)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# (a) 正解→予測の矢印（例の近く・1 台）
for ax, pairs, title in ((axes[0], pairs_single[:400], "(a) 車 1 台のとき: 正解 → モデルの位置（後左 2〜4 m・400 フレーム）"),
                         (axes[1], pairs_multi[:400], "(b) 車 2 台以上のとき: 同じ領域・400 フレーム")):
    for gx, gy, px, py in pairs:
        ax.plot([gx, px], [gy, py], color="#c44", alpha=0.25, lw=0.8)
    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=6, c="k", label="正解の位置")
    ax.scatter([p[2] for p in pairs], [p[3] for p in pairs], s=6, c="#c44", label="モデルの位置")
    ax.plot(0, 0, marker="^", ms=12, c="tab:blue"); ax.text(0.2, -0.3, "マイク（前 = 右向き）", fontsize=9)
    ax.plot(-2, 2, marker="*", ms=14, c="tab:green"); ax.text(-1.9, 2.2, "本人の例 (−2, 2)", fontsize=9, color="tab:green")
    for r in (1.5, 3.2):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", lw=0.8, color="gray"))
    ax.set_xlim(-6, 2); ax.set_ylim(-1, 6); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x 前 [m]"); ax.set_ylabel("y 左 [m]"); ax.set_title(title, fontsize=10); ax.legend(loc="lower left", fontsize=8)
# (c) 軌跡の例（1 台 3 本・多重 2 本）
ax = axes[2]
cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
shown = 0
for stem, key, seq in tracks_single[:3] + tracks_multi[:2]:
    c = cols[shown % len(cols)]
    gx = [s[1] for s in seq]; gy = [s[2] for s in seq]; px = [s[3] for s in seq]; py = [s[4] for s in seq]
    ax.plot(gx, gy, color=c, lw=2, alpha=0.9, label=f"{'1台' if shown < 3 else '2台以上'} 正解 ({stem[-7:]})")
    ax.plot(px, py, color=c, lw=1, ls="--", marker=".", ms=3, alpha=0.9)
    shown += 1
ax.plot(0, 0, marker="^", ms=12, c="tab:blue"); ax.plot(-2, 2, marker="*", ms=14, c="tab:green")
for r in (1.5, 3.2):
    ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", lw=0.8, color="gray"))
ax.set_xlim(-12, 12); ax.set_ylim(-8, 12); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x 前 [m]"); ax.set_ylabel("y 左 [m]"); ax.set_title("(c) 軌跡の例: 実線 = 正解、破線 = モデル（0.1 秒ごと）", fontsize=10); ax.legend(fontsize=7, loc="upper right")
fig.suptitle("正解の位置とモデルの位置のずれ（合成の交差 val・v18 e149・入り 0.7／保持 0.3・上から見た図）", fontsize=12)
fig.tight_layout()
out = OUT / "位置のずれ_v18_cross_後左2-4m.png"
fig.savefig(out, dpi=130)
print("->", out)
# 誤差の分布（1 台 / 多重）
fig2, ax2 = plt.subplots(figsize=(7, 4))
for pairs, lab, c in ((pairs_single, "車 1 台", "k"), (pairs_multi, "車 2 台以上", "#c44")):
    e = [math.hypot(px - gx, py - gy) for gx, gy, px, py in pairs]
    ax2.hist(e, bins=np.arange(0, 4.05, 0.1), histtype="step", lw=2, color=c, label=f"{lab}（中央 {np.median(e):.2f} m・90% {np.percentile(e, 90):.2f} m・n={len(e)}）")
ax2.set_xlabel("位置のずれ [m]（正解とモデルの平面距離）"); ax2.set_ylabel("フレーム数"); ax2.set_title("後左 2〜4 m にいた車の位置のずれの分布", fontsize=10); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig2.tight_layout(); out2 = OUT / "位置のずれ_分布_後左2-4m.png"; fig2.savefig(out2, dpi=130); print("->", out2)
