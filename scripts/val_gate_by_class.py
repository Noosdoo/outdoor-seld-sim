"""合成の検証データ（val・12,420 本）で、可聴ゲート（フレームの A 特性 SN 比 ≥ 0 dB だけラベルを残す）がクラスごとにどれだけラベルを消しているかを数える
（2026-10-25 Opus・本人 21:39「自転車だけじゃない可能性あるよね？全クラスで確かめて」）。評価データ（eval）と学内の実録には触らない。

入力: サーバの dataset_outdoor_siren_v2final_v29/masks/val_syn_*.csv（frame,class,snr_a_db＝音源ごとのフレームの A 特性 SN 比）と
      metadata/val_syn_*.csv（ゲートのあとのラベル）を手元に写したもの。
数えるもの（クラスごと）:
  - 鳴っているフレーム（snr > −100 dB）のうち、ラベルが残った割合（snr ≥ 0）／消えた中で −10〜0 dB（聞こえる可能性がある所）／−20〜−10／−20 未満
  - 消えた −10〜0 dB とラベルが残ったフレームの比（前提の敵対的レビューの自転車の 3.55 倍と同じ物差し）
  - 鳴っている本のうち、ラベルが 1 フレームも残らなかった本の割合
  - ラベル（metadata）と snr ≥ 0 のフレームの一致（確かめ）
使い方: python scripts/val_gate_by_class.py --masks <置き場> --meta <置き場>
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "out" / "dataset_outdoor_siren_v2final" / "plan" / "assignment_v2final.csv"
# Astra HB25-03（10/26）: 列車の通過音も内部では class 5。v12train は D14 でラベルを消し、組み合わせの列車は no_label＝意図してラベルを付けない。
# 踏切（警報機）として読むのは列車を含まない本だけ（列車ありは「9」として別に出す）。

NAMES = ["サイレン", "クラクション", "バック音", "ベル", "車", "踏切", "キック", "バイク", "自転車"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    fr = defaultdict(Counter)        # クラス → 区分ごとのフレーム数
    clips_present = Counter()
    clips_zero = Counter()
    mismatch = 0
    dup = 0
    files = sorted(f for f in os.listdir(a.masks) if f.startswith("val_syn_") and f.endswith(".csv"))
    train = set()
    for r in csv.DictReader(open(PLAN, encoding="utf-8-sig")):
        if r["clip_id"].startswith("val_syn_") and (r["scenario"] == "v12train" or "train" in (r.get("mix") or "")):
            train.add(r["clip_id"])
    for f in files:
        rows = list(csv.DictReader(open(os.path.join(a.masks, f), encoding="utf-8")))
        best = {}
        for r in rows:
            k = (int(r["frame"]), int(r["class"]))
            s = float(r["snr_a_db"])
            if k in best:
                dup += 1
            best[k] = max(best.get(k, -1e9), s)
        lab = set()
        mp = os.path.join(a.meta, f)
        if os.path.exists(mp):
            for l in open(mp, encoding="utf-8"):
                p = l.strip().split(",")
                if len(p) >= 2 and p[0].isdigit():
                    lab.add((int(p[0]), int(p[1])))
        has_train = f[:-4] in train
        per_cls_present = Counter()
        per_cls_kept = Counter()
        for (k, c0), s in best.items():
            c = 9 if (c0 == 5 and has_train) else c0
            if s <= -100:
                continue
            per_cls_present[c] += 1
            if (k, c) in lab:                      # 残った = 実際のラベル（metadata）にある
                fr[c]["kept"] += 1
                per_cls_kept[c] += 1
                if s < 0:
                    fr[c]["kept_but_neg"] += 1      # SN 比 < 0 でもラベルがある（短い途切れの埋めなど）
            elif s >= 0:
                fr[c]["pos_but_unlab"] += 1        # SN 比 ≥ 0 なのにラベルが無い
            elif s >= -10:
                fr[c]["m10_0"] += 1
            elif s >= -20:
                fr[c]["m20_10"] += 1
            else:
                fr[c]["lt20"] += 1
        kept_set = {k for k, s in best.items() if s >= 0}
        mismatch += len(kept_set ^ lab)
        for c in per_cls_present:
            clips_present[c] += 1
            clips_zero[c] += per_cls_kept[c] == 0
    print(f"val {len(files)} 本・ラベルと snr ≥ 0 の食い違い {mismatch} フレーム・同じフレームに同じクラスが 2 つ以上 {dup} 行（大きい方を使用）")
    print("（名前の訂正・Astra HB25-03: 「音源のエネルギーが数値的にある〔SN 比 > −100 dB〕クラス・フレームのうち、今のラベルがある割合」。−100 dB は数値の無音の除外で、聞こえる線でも発音区間でもない）")
    print("クラス: 鳴っているフレーム → 残った ／ 消えた −10〜0 dB ／ −20〜−10 ／ −20 未満 ・ 消えた −10〜0 ÷ 残った ・ ラベルが全部消えた本")
    NAMES9 = NAMES + ["列車を含む本の class 5（列車は意図してラベルなし・踏切の数に入れない）"]
    for c in range(10):
        t = fr[c]
        n = t["kept"] + t["pos_but_unlab"] + t["m10_0"] + t["m20_10"] + t["lt20"]
        if not n:
            continue
        pct = lambda x: 100 * x / n
        ratio = t["m10_0"] / t["kept"] if t["kept"] else float("inf")
        print(f"  {NAMES9[c]}: {n:,} → 残った {pct(t['kept']):.0f}% ／ −10〜0 {pct(t['m10_0']):.0f}% ／ −20〜−10 {pct(t['m20_10']):.0f}% ／ −20 未満 {pct(t['lt20']):.0f}%"
              f" ・ {ratio:.2f} 倍 ・ 全部消えた本 {clips_zero[c]}/{clips_present[c]}（{100 * clips_zero[c] / clips_present[c]:.0f}%）"
              f" ・（確かめ: SN 比 < 0 でラベルあり {t['kept_but_neg']:,}・SN 比 ≥ 0 でラベルなし {t['pos_but_unlab']:,}）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
