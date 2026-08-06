# -*- coding: utf-8 -*-
"""v12通知層v3.2採点（サーバー実行用ラッパ）。

- 対象: v12モデルのval予測のうち core 1,200本（v11と同一の通知意味論で比較可能な部分。
  キック/バイク/列車クリップの通知規則は別途設計＝今回は対象外と明記）
- DSをv12に差し替え、予測をcoreに絞ったCSVを作ってから step12_notify_v3.main() を呼ぶ
使い方: PYTHONPATH=scripts:src python scripts/_v12_notify_server.py <pred> <outdir>
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

pred_in = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

# core（fold2 mix0001-1200）だけに絞った予測CSVを作る
core = outdir / "val_core.csv"
pat = re.compile(r"fold2_room1_mix(\d{4})")
with open(core, "w") as w:
    for line in open(pred_in):
        m = pat.match(line)
        if m and 1 <= int(m.group(1)) <= 1200:
            w.write(line)

spec = importlib.util.spec_from_file_location(
    "nv3", ROOT / "scripts" / "step12_notify_v3.py")
nv3 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], str(core), str(outdir)]
spec.loader.exec_module(nv3)
nv3.DS = ROOT / "out" / "dataset_outdoor_siren_v12"
nv3.PRED = core
nv3.OUT = outdir
nv3.main()
