# -*- coding: utf-8 -*-
"""v11評価専用3,246本のSDE採点（run3）: 距離＋通知層v3.2をevalのGTで実行。

使い方: python scripts/_eval_sde_score.py
入力: out/predictions_v11sde_run3_eval/eval_all.csv（回収済み）
GT:   out/dataset_outdoor_siren_v11_eval/{metadata_dist,work,foa,masks}
出力: out/step12_notify_v11sde_run3_eval/（新規）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRED = ROOT / "out" / "predictions_v11sde_run3_eval" / "eval_all.csv"
DSE = ROOT / "out" / "dataset_outdoor_siren_v11_eval"
OUT = ROOT / "out" / "step12_notify_v11sde_run3_eval"
OUT.mkdir(parents=True, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    return spec, m


# 距離採点（META差替）
spec, sd = _load("sd", ROOT / "scripts" / "_score_sde_dist.py")
sys.argv = [sys.argv[0], str(PRED), str(OUT)]
spec.loader.exec_module(sd)
sd.PRED = PRED
sd.META = DSE / "metadata_dist"
sd.OUT = OUT
sd.main()

# 通知層v3.2（DS差替）
spec2, nv3 = _load("nv3", ROOT / "scripts" / "step12_notify_v3.py")
sys.argv = [sys.argv[0], str(PRED), str(OUT)]
spec2.loader.exec_module(nv3)
nv3.DS = DSE
nv3.PRED = PRED
nv3.OUT = OUT
nv3.main()
print("done ->", OUT)
