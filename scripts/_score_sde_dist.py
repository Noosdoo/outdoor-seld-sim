# -*- coding: utf-8 -*-
"""SDE(距離推定)の初採点: val 1,200本で「音だけでどこまで距離が当たるか」。

入力:
  - 予測: out/predictions_v11sde/val_all.csv（clip,frame,class,az,el,dist_m）
  - 正解: out/dataset_outdoor_siren_v11/metadata_dist/*.csv（6列）
対応付け: (clip,frame,class)ごとにGTと予測を方位最近傍でペアリング（同数でなくても
  ペアできた分だけ採点。ペア率も報告）
指標:
  - 距離MAE/中央絶対誤差(全体・クラス別・GT距離帯別)
  - 車の3段仕分け一致率(役割③の本題): GT tier(≤1.5/≤3.0/>3.2) vs 推定距離のtier
  - 至近側(GT≤5m)の性能を別掲(役割③が使う領域)
出力: out/step12_notify_v11sde/dist_score.md
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 引数で予測CSVと出力先を指定可（既定=run1のまま。新しいランは新フォルダに出す=上書き禁止運用）
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "out" / "predictions_v11sde" / "val_all.csv"
META = ROOT / "out" / "dataset_outdoor_siren_v11" / "metadata_dist"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "out" / "step12_notify_v11sde"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLS_JP = {0: "サイレン", 1: "クラクション", 2: "バック音", 3: "ベル",
          4: "車", 5: "踏切"}
CAR = 4


def tier(d: float) -> str:
    if d <= 1.5:
        return "重大"
    if d <= 3.0:
        return "注意"
    return "安全"


def circ_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 予測読み込み: pred[clip][(frame,class)] = [(az,dist),...]
    # 7列 [clip,frame,class,track,az,el,dist]（監査対応v2の新形式）と
    # 旧6列 [clip,frame,class,az,el,dist] の両対応
    pred = defaultdict(lambda: defaultdict(list))
    for line in open(PRED, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            pred[p[0]][(int(p[1]), int(p[2]))].append(
                (float(p[4]), float(p[6])))
        elif len(p) == 6:
            pred[p[0]][(int(p[1]), int(p[2]))].append(
                (float(p[3]), float(p[5])))

    pairs = []          # (class, gt_dist, pred_dist)
    n_gt = n_paired = 0
    for clip in sorted(pred.keys()):
        gt = defaultdict(list)
        mp = META / f"{clip}.csv"
        if not mp.exists():
            continue
        for line in open(mp, encoding="utf-8"):
            g = line.strip().split(",")
            if len(g) == 6:
                gt[(int(g[0]), int(g[1]))].append((float(g[3]), float(g[5])))
        for key, gts in gt.items():
            n_gt += len(gts)
            preds = list(pred[clip].get(key, []))
            # 方位最近傍のgreedyペアリング
            for gaz, gdist in sorted(gts, key=lambda x: x[0]):
                if not preds:
                    break
                j = int(np.argmin([circ_diff(gaz, paz) for paz, _ in preds]))
                paz, pdist = preds.pop(j)
                pairs.append((key[1], gdist, pdist))
                n_paired += 1

    arr = np.array([(c, g, p) for c, g, p in pairs])
    cls, gd, pd_ = arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
    err = np.abs(pd_ - gd)

    R = ["# SDE距離推定の採点（val 1,200本）", "",
         f"- GTフレーム対 {n_gt:,} / ペア成立 {n_paired:,} "
         f"({100*n_paired/max(n_gt,1):.1f}%)",
         f"- **距離MAE {err.mean():.2f}m / 中央絶対誤差 {np.median(err):.2f}m**"
         f"（全クラス、GT距離実測レンジ {gd.min():.2f}〜{gd.max():.2f}m）",
         f"- 予測距離レンジ {pd_.min():.2f}〜{pd_.max():.2f}m / 上限0.05m内への"
         f"張り付き {100*(pd_ >= pd_.max()-0.05).mean():.1f}%"
         "（張り付きが大きい場合、上限超のGT帯のMAEは学習性能でなく出力構造の"
         "上限の帰結=第8回監査SDE-1。旧線形+tanh版は10mが構造上限だった）",
         "", "## クラス別",
         "| クラス | n | MAE | 中央 |", "| --- | --- | --- | --- |"]
    for ci in range(6):
        m = cls == ci
        if m.sum():
            R.append(f"| {CLS_JP[ci]} | {m.sum():,} | {err[m].mean():.2f}m "
                     f"| {np.median(err[m]):.2f}m |")

    R += ["", "## GT距離帯別（車のみ）",
          "| GT距離 | n | MAE | 中央 |", "| --- | --- | --- | --- |"]
    car = cls == CAR
    for lo, hi in [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 10.0),
                   (10.0, 20.0), (20.0, 99.0)]:
        m = car & (gd > lo) & (gd <= hi)
        if m.sum():
            R.append(f"| {lo:g}〜{hi:g}m | {m.sum():,} | {err[m].mean():.2f}m "
                     f"| {np.median(err[m]):.2f}m |")

    # 3段仕分け(役割③の本題)。帰無基準線を併記（監査対応v2:
    # 「安全」が支配的なため全体一致率は帰無と差が出にくい→tier別再現率を主に読む）
    gt_t = np.array([tier(v) for v in gd[car]])
    pd_t = np.array([tier(v) for v in pd_[car]])
    agree = (gt_t == pd_t).mean()
    null_agree = max((gt_t == t).mean() for t in ("重大", "注意", "安全"))
    R += ["", "## 車の3段仕分け（役割③）",
          f"- 全車フレーム一致率: {100*agree:.1f}% "
          f"（帰無=常に最頻tier: {100*null_agree:.1f}% → 上積み"
          f"+{100*(agree-null_agree):.1f}pt。**主指標はtier別再現率**）"]
    for t in ("重大", "注意", "安全"):
        m = gt_t == t
        if m.sum():
            recall = (pd_t[m] == t).mean()
            R.append(f"- GT={t} ({m.sum():,}fr): 正しく{t}と判定 "
                     f"{100*recall:.1f}%")
    near = car & (gd <= 5.0)
    const_near = np.abs(np.median(gd[near]) - gd[near])  # 最良定数の帰無
    R += ["", f"## 至近側（GT≤5m、役割③の作動域）: n={near.sum():,} / "
          f"MAE {err[near].mean():.2f}m / 中央 {np.median(err[near]):.2f}m "
          f"（帰無=最良定数予測のMAE {const_near.mean():.2f}m）"]

    (OUT / "dist_score.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
