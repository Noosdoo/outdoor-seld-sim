# -*- coding: utf-8 -*-
"""ablation arm の距離ラベル生成（2026-08-15、確認run用に新規作成）。

`step19_dist_labels_v12.py` は DS/OUT をモジュール定数で持ち、基準データセットを
指す。armのデータに対して走らせるには差し替えが必要（_ablate_smoke.py と同じ手口）。
基準の metadata_dist を上書きしないための専用経路。

安全装置: ABLATE が空なら起動を拒否し、出力先が基準と別物であることを assert する。

使い方:
  ABLATE=no_1r python scripts/_run_v12_abl_labels.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ARM = os.environ.get("ABLATE", "").strip()
if not ARM:
    raise SystemExit("ABLATE が空です。本ドライバは ablation arm 専用です。")

BASE_DS = ROOT / "out" / "dataset_outdoor_siren_v12"
DS = ROOT / "out" / f"dataset_outdoor_siren_v12_abl_{ARM}"
assert DS.resolve() != BASE_DS.resolve(), "出力先が基準データセットと同一です"
assert (DS / "foa").is_dir(), f"armのfoaがありません: {DS}"

spec = importlib.util.spec_from_file_location(
    "s19v12", ROOT / "scripts" / "step19_dist_labels_v12.py")
s19 = importlib.util.module_from_spec(spec)
argv, sys.argv = sys.argv, [sys.argv[0]]      # main()内のargparse対策
try:
    spec.loader.exec_module(s19)
    s19.DS = DS
    s19.OUT = DS / "metadata_dist"
    assert s19.OUT.resolve() != (BASE_DS / "metadata_dist").resolve()
    print(f"arm={ARM} 距離ラベル -> {s19.OUT}", flush=True)
    s19.main()
finally:
    sys.argv = argv
print(f"ABL_LABELS_DONE arm={ARM}")
