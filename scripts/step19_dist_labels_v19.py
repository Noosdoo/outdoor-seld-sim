# -*- coding: utf-8 -*-
"""v19（新 4 場面）の距離付きラベル（水平距離・mic_z 対応）= step19_dist_labels_v16 と同一。出力先だけ out/dataset_outdoor_siren_v19/metadata_dist。
使い方: PYTHONPATH=scripts:src python scripts/step19_dist_labels_v19.py [--limit N]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import step19_dist_labels_v16 as s16  # noqa: E402

DS = ROOT / "out/dataset_outdoor_siren_v19"
s16.DS = DS
s16.s19.DS = DS
s16.s19.OUT = DS / "metadata_dist"
s16.m9.DS = DS
s16.m9.WORK = DS / "work"

if __name__ == "__main__":
    assert s16.DS.name == "dataset_outdoor_siren_v19"
    s16.s19.main()
