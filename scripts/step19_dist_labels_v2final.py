# -*- coding: utf-8 -*-
"""v2final の距離付きラベル（水平距離・mic_z 対応）= step19_dist_labels_v16 と同一。出力先だけ out/dataset_outdoor_siren_v2final/metadata_dist。

使い方: PYTHONPATH=scripts:src python scripts/step19_dist_labels_v2final.py [--limit N]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step19_dist_labels_v16 as s16  # noqa: E402
import step11_v22_render  # noqa: E402,F401  bicycle（番号 8）を CLASS_IDX_V12 に登録（2026-10-08 変更票 #48）。無いと bicycle の本で KeyError

DS = ROOT / "out/dataset_outdoor_siren_v2final"
if "--ds" in sys.argv:                                   # 2026-10-12: 新しい出力先（描き直し）にも使う。名前は dataset_outdoor_siren_v2final で始まるものだけ
    _i = sys.argv.index("--ds")
    DS = Path(sys.argv[_i + 1]).resolve()
    del sys.argv[_i:_i + 2]                                # s19.main() の引数（--limit）の邪魔をしない
    assert DS.name.startswith("dataset_outdoor_siren_v2final"), DS
s16.DS = DS
s16.s19.DS = DS
s16.s19.OUT = DS / "metadata_dist"
s16.m9.DS = DS
s16.m9.WORK = DS / "work"

if __name__ == "__main__":
    assert s16.DS.name.startswith("dataset_outdoor_siren_v2final"), s16.DS
    s16.s19.main()
