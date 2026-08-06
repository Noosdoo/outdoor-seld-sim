# -*- coding: utf-8 -*-
"""v12評価拡充セット（heavy600/far300/kick300/bike300）のセット別採点。

- heavy600(fold10): 車recall・至近≤5m距離誤差・tier選別
- far300(fold11): クラス別の検知開始距離（初検知フレームのGT距離）
- kick300(fold12)/bike300(fold13): 基本(mix≤150)/苦手(>150)別recall
使い方: python scripts/_v12_eval_score.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "out" / "dataset_outdoor_siren_v12_eval"
PRED = ROOT / "out" / "predictions_v12_eval" / "eval_all.csv"
OUT = ROOT / "out" / "v12_eval_score"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLS_JA = {0: "サイレン", 4: "車", 6: "キックボード", 7: "バイク"}


def load_pred():
    pred = defaultdict(lambda: defaultdict(dict))
    for line in open(PRED):
        p = line.strip().split(",")
        if len(p) >= 7:
            pred[p[0]][int(p[1])][int(p[2])] = (float(p[4]), float(p[5]),
                                                float(p[6]))
    return pred


def gt_of(clip):
    """metadata_dist(6列) → {frame: {cls: dist}} と 可聴mask。"""
    gt = defaultdict(dict)
    for line in open(DS / "metadata_dist" / f"{clip}.csv"):
        q = line.strip().split(",")
        if len(q) == 6:
            gt[int(q[0])][int(q[1])] = float(q[5])
    mask = {}
    with open(DS / "masks" / f"{clip}.csv") as f:
        next(f)
        for line in f:
            q = line.strip().split(",")
            mask[(int(q[0]), int(q[1]))] = float(q[2])
    return gt, mask


def recall_and_dist(pred, clips, cls):
    tp = n = 0
    near_err = []
    for clip in clips:
        gt, mask = gt_of(clip)
        for k, evs in gt.items():
            if cls not in evs or mask.get((k, cls), 99.0) < 0.0:
                continue
            n += 1
            pk = pred[clip].get(k, {})
            if cls in pk:
                tp += 1
                if evs[cls] <= 5.0:
                    near_err.append(abs(pk[cls][2] - evs[cls]))
    return tp, n, near_err


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = load_pred()
    all_clips = sorted({p.stem for p in (DS / "metadata").glob("*.csv")})
    by_fold = defaultdict(list)
    for c in all_clips:
        by_fold[c.split("_")[0]].append(c)
    L = ["# v12評価拡充セット採点（v12モデル）", ""]

    # heavy600
    tp, n, near = recall_and_dist(pred, by_fold["fold10"], 4)
    L += ["## heavy600（単独大型車・safe幾何）",
          f"- 車recall(可聴): {tp/n:.2%} (n={n:,})",
          f"- 至近≤5m 距離誤差: MAE {np.mean(near):.2f}m / 中央 {np.median(near):.2f}m "
          f"(n={len(near):,})", ""]

    # far300: 検知開始距離
    L += ["## far300（100〜300mスポーン: 検知開始距離）", ""]
    for cls in (0, 4):
        first_d, none = [], 0
        for clip in by_fold["fold11"]:
            gt, _ = gt_of(clip)
            frames = sorted(k for k, e in gt.items() if cls in e)
            if not frames:
                continue
            hit = next((k for k in frames if cls in pred[clip].get(k, {})), None)
            if hit is None:
                none += 1
            else:
                first_d.append(gt[hit][cls])
        if first_d:
            L.append(f"- {CLS_JA[cls]}: 初検知距離 中央 **{np.median(first_d):.0f}m** / "
                     f"p90 {np.percentile(first_d, 90):.0f}m / 最大 {max(first_d):.0f}m "
                     f"(検知clip {len(first_d)} / 未検知 {none})")
    L.append("")

    # kick/bike 基本 vs 苦手
    for fold, cls, name in (("fold12", 6, "kick300"), ("fold13", 7, "bike300")):
        basic = [c for c in by_fold[fold] if int(c.split("mix")[1]) <= 150]
        hard = [c for c in by_fold[fold] if int(c.split("mix")[1]) > 150]
        tb, nb, _ = recall_and_dist(pred, basic, cls)
        th, nh, _ = recall_and_dist(pred, hard, cls)
        L += [f"## {name}",
              f"- 基本: {tb/nb:.2%} (n={nb:,}) / 苦手(騒音58-65dB): {th/nh:.2%} (n={nh:,})",
              ""]

    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
