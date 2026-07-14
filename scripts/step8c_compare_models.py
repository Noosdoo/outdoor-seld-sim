# -*- coding: utf-8 -*-
"""step8の summary_mc.json 2つ（例: v5モデル vs v6モデルを同一valで解剖した結果）を
クラス別に横並び比較する図を作る。

実行例:
  python scripts/step8c_compare_models.py \
      --a out/figures_v5run2_on_v6val/summary_mc.json --label-a "v5 model (train 60)" \
      --b out/figures_v6_analysis/summary_mc.json     --label-b "v6 model (train 240)" \
      --out out/figures_v6_analysis/compare_v5_v6.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# dataviz標準パレット（2系列 blue/aqua は検証済み。aquaは低コントラストのため値ラベル併記）
C_A = "#2a78d6"
C_B = "#1baf7a"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"


def load(path):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = s["classes"]
    dir_rate = [100 * pc["dir_err_gt20"] / max(pc["n_gt_frames"], 1)
                for pc in s["per_class"]]
    le = [pc["le_same_class_median_deg"] for pc in s["per_class"]]
    return classes, dir_rate, le


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="model A")
    ap.add_argument("--label-b", default="model B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    classes, dir_a, le_a = load(args.a)
    classes_b, dir_b, le_b = load(args.b)
    assert classes == classes_b, "class lists differ"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor(C_SURFACE)
    xs = np.arange(len(classes))
    w = 0.36

    panels = [
        (axes[0], dir_a, dir_b, "dir_err rate (% of GT frames)",
         "direction errors >20 deg"),
        (axes[1], le_a, le_b, "median LE (deg)", "localization error (matched frames)"),
    ]
    for ax, va, vb, ylab, title in panels:
        ax.set_facecolor(C_SURFACE)
        ba = ax.bar(xs - w / 2, va, w, color=C_A, label=args.label_a)
        bb = ax.bar(xs + w / 2, vb, w, color=C_B, label=args.label_b)
        for bars in (ba, bb):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                        f"{r.get_height():.1f}", ha="center", va="bottom",
                        fontsize=8.5, color=C_INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(classes)
        ax.set_ylabel(ylab, color=C_INK)
        ax.set_title(title, color=C_INK, fontsize=11)
        ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(C_MUTED)
        ax.tick_params(colors=C_MUTED)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("same 80 val clips (v6 val, unseen by both models)", color=C_INK)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
