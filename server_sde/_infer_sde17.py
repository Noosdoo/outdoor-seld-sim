# -*- coding: utf-8 -*-
"""v17 SDE推論ラッパ = v12 版 ＋ height_patch（装着高さの入力・HEIGHT_COND=1 必須）。"""
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
try:
    import data.components.data as dcd
    if hasattr(dcd, "load_output_format_file"):
        dcd.load_output_format_file = load_ok
except Exception:
    pass

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

import height_patch  # noqa: E402

height_patch.apply()

from infer import main  # noqa: E402

if __name__ == "__main__":
    main()
