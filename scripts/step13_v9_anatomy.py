# -*- coding: utf-8 -*-
"""Step 13: v9 run1 の本解剖。

出すもの（すべて out/figures_v9_analysis/ と out/v9_anatomy_2026-07-17.md へ）:
  1. 車FA（ラベル外の車予測）464フレームの内訳 — 境界にじみ / 閾値下検出 / 深い閾値下
  2. **検出限界カーブ**（主要成果物）: フレームA特性SNR別の検出率、6クラス
  3. 可聴ゲートの感度確認（±5dB）: ゲートを−5/0/+5dBに振ったときの再現率
  4. 静止 vs 歩行の層別（自己移動のコスト）: クラス別 再現率・方向誤差
  5. 車の危険層別（critical/caution/safe）の検出率・方向誤差

入力: out/predictions_v9/val_all.csv・データセットの metadata/masks/scene.json。
"""
from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V91 = "--v91" in sys.argv
_SUF = "_1" if V91 else ""
DS = ROOT / "out" / f"dataset_outdoor_siren_v9{_SUF}"
PRED = ROOT / "out" / f"predictions_v9{_SUF}" / "val_all.csv"
FIG = ROOT / "out" / f"figures_v9{_SUF}_analysis"
MD = ROOT / "out" / f"v9{_SUF}_anatomy_2026-07-17.md"

CLASSES = ["サイレン", "クラクション", "バック警告音", "自転車ベル", "車の走行音", "踏切警報音"]
CAR = 4
# Okabe-Ito（CVD対応、検証済み。曲線は右端に直接ラベル=二次符号化）
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
# 両端は開区間（<-10 / >=10）。サイレン等はSNR30dB超が大半なので上端は十分広く取る
SNR_EDGES = [-90.0, -10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0, 90.0]


def load_all():
    pred = defaultdict(lambda: defaultdict(set))
    pred_az = defaultdict(dict)
    for line in open(PRED):
        p = line.strip().split(",")
        if len(p) >= 5:
            pred[p[0]][int(p[1])].add(int(p[2]))
            pred_az[p[0]][(int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]))
    data = {}
    for clip in sorted(pred):
        gt = defaultdict(dict)
        for line in open(DS / "metadata" / f"{clip}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                gt[int(q[0])][int(q[1])] = (float(q[3]), float(q[4]))
        mask = {}
        with open(DS / "masks" / f"{clip}.csv") as f:
            next(f)
            for line in f:
                q = line.strip().split(",")
                mask[(int(q[0]), int(q[1]))] = float(q[2])
        scene = json.loads((DS / "work" / clip / "scene.json").read_text())
        data[clip] = (pred[clip], pred_az[clip], gt, mask, scene)
    return data


def ang_err(a1, e1, a2, e2):
    a1, e1, a2, e2 = map(np.radians, (a1, e1, a2, e2))
    cos = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_all()
    md = ["# v9 run1 本解剖（2026-07-17、val 240本）", ""]

    # ---- 1. 車FAの内訳 --------------------------------------------------
    fa = {"boundary": 0, "sub_0to-5": 0, "deep_sub": 0, "audible_unlabeled": 0}
    n_fa = 0
    for clip, (pr, _, gt, mask, scene) in data.items():
        car_frames = {k for k, evs in gt.items() if CAR in evs}
        for k in range(100):
            if CAR in pr.get(k, set()) and k not in car_frames:
                n_fa += 1
                if any((k + d) in car_frames for d in (-2, -1, 1, 2)):
                    fa["boundary"] += 1
                else:
                    snr = mask.get((k, CAR), -99.0)
                    if snr >= 0.0:
                        fa["audible_unlabeled"] += 1   # 穴埋め規約の外側等
                    elif snr >= -5.0:
                        fa["sub_0to-5"] += 1
                    else:
                        fa["deep_sub"] += 1
    md += ["## 1. 車FA（ラベル外の車予測）の内訳",
           f"- 総数 {n_fa} フレーム（240本×100フレーム中）",
           f"  - ラベル境界±0.2s のにじみ: {fa['boundary']} ({fa['boundary']/n_fa:.0%})",
           f"  - 閾値下検出（SNR −5〜0dB、耳より鋭い）: {fa['sub_0to-5']} "
           f"({fa['sub_0to-5']/n_fa:.0%})",
           f"  - 深い閾値下（SNR <−5dB）: {fa['deep_sub']} ({fa['deep_sub']/n_fa:.0%})",
           f"  - 可聴なのに未ラベル（穴埋め規約外）: {fa['audible_unlabeled']} "
           f"({fa['audible_unlabeled']/n_fa:.0%})", ""]

    # ---- 2. 検出限界カーブ（クラス別、フレームSNRビン） -------------------
    bins = {c: defaultdict(lambda: [0, 0]) for c in range(6)}  # cls -> bin -> [det, tot]
    for clip, (pr, _, gt, mask, scene) in data.items():
        for src in scene["sources"]:
            c = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
                 "car_drive": 4, "crossing": 5}[src["class"]]
            k0 = int(src["t_on"] * 10)
            k1 = int(min(src["t_off"], scene.get("cpa_rel_time_s", 10.0)
                         if c == CAR else src["t_off"]) * 10)
            for k in range(k0, max(k1, k0 + 1)):
                snr = mask.get((k, c))
                if snr is None:
                    continue
                b = np.digitize(snr, SNR_EDGES) - 1
                bins[c][b][1] += 1
                if c in pr.get(k, set()):
                    bins[c][b][0] += 1
    md += ["## 2. 検出限界カーブ（フレームA特性SNR別の検出率。車は接近フェーズのみ）",
           "", "| SNR帯[dB] | " + " | ".join(CLASSES) + " |",
           "| --- |" + " --- |" * 6]
    curve_rows = []
    for b in range(len(SNR_EDGES) - 1):
        if b == 0:
            lab = f"<{SNR_EDGES[1]:.0f}"
        elif b == len(SNR_EDGES) - 2:
            lab = f">={SNR_EDGES[-2]:.0f}"
        else:
            lab = f"{SNR_EDGES[b]:.0f}〜{SNR_EDGES[b+1]:.0f}"
        cells = []
        for c in range(6):
            det, tot = bins[c][b]
            cells.append(f"{det/tot:.0%} (n={tot})" if tot >= 20 else
                         (f"n={tot}" if tot else "—"))
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
        curve_rows.append((lab, [bins[c][b] for c in range(6)]))
    md.append("")

    # 図（一次データはこの表とCSV。図は同内容の折れ線）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    centers = [(SNR_EDGES[i] + SNR_EDGES[i + 1]) / 2 for i in range(len(SNR_EDGES) - 1)]
    centers[0], centers[-1] = -12.5, 12.5   # 開区間の代表点は端に寄せる
    ends = []
    for c in range(6):
        xs, ys = [], []
        for b in range(len(SNR_EDGES) - 1):
            det, tot = bins[c][b]
            if tot >= 20:
                xs.append(centers[b])
                ys.append(det / tot)
        ax.plot(xs, ys, "-o", lw=2, ms=5, color=COLORS[c])
        if xs:
            ends.append((ys[-1], xs[-1], c))
    # 右端の直接ラベルは重なるので縦に均す（下から順に最低間隔0.05を確保）
    ends.sort()
    y_prev = -1.0
    for y_end, x_end, c in ends:
        y_lab = max(y_end, y_prev + 0.05)
        y_prev = y_lab
        ax.annotate(CLASSES[c], (x_end, y_end), xytext=(14.0, y_lab),
                    textcoords="data", color=COLORS[c], fontsize=10,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=COLORS[c],
                                    lw=0.8, alpha=0.6))
    ax.axvline(0.0, color="0.55", ls=":", lw=1.5)
    ax.annotate("背景雑音 (0 dB)", (0, 0.05), xytext=(6, 0),
                textcoords="offset points", color="0.4", fontsize=9)
    ax.set_xlabel("対象音と背景騒音の音量差［dB］")
    ax.set_ylabel("検出率")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-16, 19)
    ax.grid(alpha=0.25)
    ax.set_title("v9 run1: detection rate vs frame SNR (val 240 clips)")
    plt.tight_layout()
    plt.savefig(FIG / "detection_limit_curves.png", dpi=150)
    plt.close()

    # ---- 3. 可聴ゲートの感度（±5dB） ------------------------------------
    md += ["## 3. 可聴ゲート感度（発音窓内のフレーム検出率、SNRゲートを振る）",
           "", "| クラス | SNR≥−5dB | SNR≥0dB | SNR≥+5dB |", "| --- | --- | --- | --- |"]
    for c in range(6):
        cells = []
        for gate in (-5.0, 0.0, 5.0):
            det = tot = 0
            for b in range(len(SNR_EDGES) - 1):
                if SNR_EDGES[b] >= gate:
                    det += bins[c][b][0]
                    tot += bins[c][b][1]
            cells.append(f"{det/tot:.1%}" if tot else "—")
        md.append(f"| {CLASSES[c]} | " + " | ".join(cells) + " |")
    md.append("")

    # ---- 4. 静止 vs 歩行（自己移動のコスト） ------------------------------
    strat = {m: {c: {"tp": 0, "n": 0, "errs": []} for c in range(6)}
             for m in ("static", "walk")}
    for clip, (pr, paz, gt, mask, scene) in data.items():
        mo = scene["row"]["motion"]
        for k, evs in gt.items():
            for c, (gaz, gel) in evs.items():
                if mask.get((k, c), 99.0) < 0.0:
                    continue
                strat[mo][c]["n"] += 1
                if c in pr.get(k, set()):
                    strat[mo][c]["tp"] += 1
                    a, e = paz[(k, c)]
                    strat[mo][c]["errs"].append(ang_err(gaz, gel, a, e))
    md += ["## 4. 静止 vs 歩行（可聴フレームの再現率 / 方向誤差中央値）",
           "", "| クラス | 静止 | 歩行 | 差 |", "| --- | --- | --- | --- |"]
    for c in range(6):
        row = []
        recs = {}
        for m in ("static", "walk"):
            s = strat[m][c]
            rec = s["tp"] / s["n"] if s["n"] else float("nan")
            recs[m] = rec
            le = np.median(s["errs"]) if s["errs"] else float("nan")
            row.append(f"{rec:.1%} / {le:.1f}°")
        d = (recs["walk"] - recs["static"]) * 100
        md.append(f"| {CLASSES[c]} | {row[0]} | {row[1]} | {d:+.1f}pt |")
    md.append("")

    # ---- 5. 車の危険層別 --------------------------------------------------
    md += ["## 5. 車の危険層別（可聴フレームの再現率 / 方向誤差中央値）", ""]
    tier_s = {t: {"tp": 0, "n": 0, "errs": []} for t in ("critical", "caution", "safe")}
    for clip, (pr, paz, gt, mask, scene) in data.items():
        t = scene["row"]["danger_tier"]
        for k, evs in gt.items():
            if CAR not in evs:
                continue
            tier_s[t]["n"] += 1
            if CAR in pr.get(k, set()):
                tier_s[t]["tp"] += 1
                gaz, gel = evs[CAR]
                a, e = paz[(k, CAR)]
                tier_s[t]["errs"].append(ang_err(gaz, gel, a, e))
    md += ["| 危険層 | 再現率 | 方向誤差 |", "| --- | --- | --- |"]
    for t in ("critical", "caution", "safe"):
        s = tier_s[t]
        md.append(f"| {t} | {s['tp']/s['n']:.1%} (n={s['n']}) | "
                  f"{np.median(s['errs']):.1f}° |")

    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nfigure ->", FIG / "detection_limit_curves.png")


if __name__ == "__main__":
    main()
