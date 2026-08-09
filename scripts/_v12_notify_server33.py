# -*- coding: utf-8 -*-
"""v12 core通知のv3.3採点（サーバー実行用ラッパ・Sol再監査対応）。

- v3.3 (step12_notify_v33.py) を v12 core（fold2 mix0001-1200）に適用。
  manifest基準なので予測ゼロのcoreクリップも分母に入る（Sol条件2）。
使い方: PYTHONPATH=scripts:src python scripts/_v12_notify_server33.py <pred> <outdir>
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

# core（fold2 mix0001-1200）だけに絞った予測CSV
core = outdir / "val_core.csv"
pat = re.compile(r"fold2_room1_mix(\d{4})")
with open(core, "w") as w:
    for line in open(pred_in):
        m = pat.match(line)
        if m and 1 <= int(m.group(1)) <= 1200:
            w.write(line)


def _core_only(clip: str) -> bool:
    m = pat.match(clip)
    return bool(m and 1 <= int(m.group(1)) <= 1200)


spec = importlib.util.spec_from_file_location(
    "nv33", ROOT / "scripts" / "step12_notify_v33.py")
nv33 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], str(core), str(outdir)]
spec.loader.exec_module(nv33)
nv33.DS = ROOT / "out" / "dataset_outdoor_siren_v12"
nv33.PRED = core
nv33.OUT = outdir
nv33.MANIFEST_FILTER = _core_only
nv33.main()
