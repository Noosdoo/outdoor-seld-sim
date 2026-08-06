# -*- coding: utf-8 -*-
"""v12 SDE学習の起動ラッパ = 空メタ許容パッチ + sde_patch + train。

- 空メタ許容: 0バイトmetadata CSVを {99: []} として扱う（Colab採用版と同一ロジック、
  検証= scripts/_test_v11_empty_label_patch.py）。学習時のval GT読み経路
  (data/components/data.py) も同じ0バイト判定でラップする
- sde_patch: 距離軸・損失・デコーダ（metadata_dist側の0バイト許容は sde_patch 内蔵）
使い方: _train_sde.py と同じ（experiment等をhydra引数で）
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import utils.data_utilities as du  # noqa: E402

_load, _read = du.load_output_format_file, pd.read_csv


def _is_empty_file(path):
    try:
        return os.path.getsize(path) == 0
    except (TypeError, OSError, ValueError):
        return False


def load_ok(path, *a, **kw):
    if _is_empty_file(path):
        return {99: []}
    return _load(path, *a, **kw)


def read_ok(path, *a, **kw):
    if _is_empty_file(path):
        return pd.DataFrame([[99, 0, 0, 0, 0]])
    return _read(path, *a, **kw)


du.load_output_format_file = load_ok
pd.read_csv = read_ok
try:  # from-import済みの参照も差し替え（val GT経路）
    import data.components.data as dcd
    if hasattr(dcd, "load_output_format_file"):
        dcd.load_output_format_file = load_ok
except Exception:
    pass

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
