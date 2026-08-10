# -*- coding: utf-8 -*-
"""v12確定評価セットのcore通知採点（サーバー実行用・v3.3/v3.4両方＋B2個票）。

- conf core（fold20 mix0001-1200）にv3.3(25°)とv3.4(60°)を適用（manifest基準）
- B2個票（B2_DUMP=1）を有効化して強未達の重大車を全数分類
使い方: PYTHONPATH=scripts:src python scripts/_v12_conf_notify_server.py <pred> <outdir25> <outdir60>
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

pred_in = Path(sys.argv[1])
out25 = Path(sys.argv[2])
out60 = Path(sys.argv[3])

pat = re.compile(r"fold20_room1_mix(\d{4})")


def _core_only(clip: str) -> bool:
    m = pat.match(clip)
    return bool(m and 1 <= int(m.group(1)) <= 1200)


for outdir, link in ((out25, "25"), (out60, "60")):
    outdir.mkdir(parents=True, exist_ok=True)
    core = outdir / "conf_core.csv"
    with open(core, "w") as w:
        for line in open(pred_in):
            m = pat.match(line)
            if m and 1 <= int(m.group(1)) <= 1200:
                w.write(line)
    os.environ["NOTIFY_LINK_DEG"] = link
    os.environ["B2_DUMP"] = "1"
    spec = importlib.util.spec_from_file_location(
        f"nv33_{link}", ROOT / "scripts" / "step12_notify_v33.py")
    nv = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0], str(core), str(outdir)]
    spec.loader.exec_module(nv)
    nv.DS = ROOT / "out" / "dataset_outdoor_siren_v12_conf"
    nv.PRED = core
    nv.OUT = outdir
    nv.MANIFEST_FILTER = _core_only
    nv.main()
    print(f"[conf notify] link={link} -> {outdir}", flush=True)
