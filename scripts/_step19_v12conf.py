# -*- coding: utf-8 -*-
"""v12確定評価セットの距離付き6列ラベル生成（step19_v12のDSをconf側に差し替え）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step11_v12_conf_render  # noqa: E402, F401  (m9.DSをconf側に切替)

spec = importlib.util.spec_from_file_location(
    "s19", ROOT / "scripts" / "step19_dist_labels_v12.py")
s19 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(s19)
s19.DS = ROOT / "out" / "dataset_outdoor_siren_v12_conf"
s19.OUT = s19.DS / "metadata_dist"
s19.main()
