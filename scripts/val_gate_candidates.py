"""可聴ゲートの線の候補ごとに、クラスごとのラベルがどれだけ戻るかを数える（2026-10-25 Opus・決める材料・おすすめは付けない）。
val 12,420 本の masks（音源ごとのフレームの広帯域 A 特性 SN 比）で、線を t dB にしたら「鳴っているフレームのうち何 % にラベルが付くか」と
「ラベルが 1 フレームも無い本の割合」を出す。今のラベル（metadata）との違いは、線の外の決まり（踏切など）を含まないので、t = 0 の行と今のラベルの行を並べて読む。
（帯域ごとの判定は masks では出せない＝音源ごとの音〔stem〕が要る・Astra P25-01。ここでは広帯域の線だけ。）
**Astra HB25-03（10/26）**: これは単純な SN 比の線での数で、今のラベル生成（発音区間・D14・no_label・gap fill）を全部は再現しない＝「線を変えたときの正式な教師ラベルの数」ではない。踏切（class 5）の行は列車を含む本も入っている（列車は意図してラベルなし）ので、踏切の線の根拠に使わない。
使い方: python scripts/val_gate_candidates.py --masks <置き場> --meta <置き場> [--lines 0,-3,-5,-10,-15]
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

NAMES = ["サイレン", "クラクション", "バック音", "ベル", "車", "踏切", "キック", "バイク", "自転車"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lines", default="0,-3,-5,-10,-15")
    a = ap.parse_args()
    lines = [float(x) for x in a.lines.split(",")]
    present = Counter()
    kept_now = Counter()
    kept_at = defaultdict(Counter)        # 線 → クラス → フレーム数
    clips_present = Counter()
    zero_now = Counter()
    zero_at = defaultdict(Counter)
    for f in sorted(x for x in os.listdir(a.masks) if x.startswith("val_syn_") and x.endswith(".csv")):
        best = {}
        for r in csv.DictReader(open(os.path.join(a.masks, f), encoding="utf-8")):
            k = (int(r["frame"]), int(r["class"]))
            best[k] = max(best.get(k, -1e9), float(r["snr_a_db"]))
        lab = set()
        mp = os.path.join(a.meta, f)
        if os.path.exists(mp):
            for l in open(mp, encoding="utf-8"):
                p = l.strip().split(",")
                if len(p) >= 2 and p[0].isdigit():
                    lab.add((int(p[0]), int(p[1])))
        cls_frames = defaultdict(list)
        for (k, c), s in best.items():
            if s > -100:
                cls_frames[c].append((k, s))
        for c, fr in cls_frames.items():
            clips_present[c] += 1
            present[c] += len(fr)
            n_now = sum(1 for k, s in fr if (k, c) in lab)
            kept_now[c] += n_now
            zero_now[c] += n_now == 0
            for t in lines:
                n = sum(1 for k, s in fr if s >= t)
                kept_at[t][c] += n
                zero_at[t][c] += n == 0
    hdr = "クラス: 今のラベル | " + " | ".join(f"線 {t:+g} dB" for t in lines) + "（鳴っているフレームのうちラベルが付く割合・ラベルが 1 つも無い本の割合）"
    print(hdr)
    for c in range(9):
        if not present[c]:
            continue
        cells = [f"{100 * kept_now[c] / present[c]:.0f}%・{100 * zero_now[c] / clips_present[c]:.0f}%"]
        for t in lines:
            cells.append(f"{100 * kept_at[t][c] / present[c]:.0f}%・{100 * zero_at[t][c] / clips_present[c]:.0f}%")
        print(f"  {NAMES[c]}: " + " | ".join(cells))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
