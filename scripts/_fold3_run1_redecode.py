# -*- coding: utf-8 -*-
"""fold3のrun1予測をrun1当時の活性で再復号する（Sol再監査・条件5）。

run1 ckptは全チャンネルtanh時代の学習物。HEADコード（xyzのみtanh）で推論した
fold3_run1.csvの距離列は生値d_HEAD=10*y になっているため、当時の復号
d_old = 10*tanh(y) = 10*tanh(d_HEAD/10) に変換する。
※注意（Solも承知の近似）: 同クラス複数トラックのマージ行は「復号後の平均」なので
  tanhの非線形により厳密には逆変換できない。単独トラック行は厳密。
使い方: python scripts/_fold3_run1_redecode.py
出力: out/predictions_v11sde_fold3/fold3_run1_redecoded.csv
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "out" / "predictions_v11sde_fold3" / "fold3_run1.csv"
DST = ROOT / "out" / "predictions_v11sde_fold3" / "fold3_run1_redecoded.csv"

n = 0
with open(SRC, encoding="utf-8") as f, open(DST, "w", encoding="utf-8") as w:
    for line in f:
        p = line.strip().split(",")
        if len(p) >= 7:
            d = max(float(p[6]), 0.0)
            p[6] = f"{10.0 * math.tanh(d / 10.0):.2f}"
            n += 1
        w.write(",".join(p) + "\n")
print(f"redecoded {n:,} rows -> {DST}")
