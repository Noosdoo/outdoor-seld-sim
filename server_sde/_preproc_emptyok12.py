# -*- coding: utf-8 -*-
"""空メタ許容preproc（v12用。Colab採用版 _preproc_emptyok.py と同一ロジック）。

0バイトのmetadata CSVを {99: []}（イベント0件・num_frames=100）として扱う。
根拠= scripts/_test_v11_empty_label_patch.py（偽イベント混入なしを回帰込みで実証済み）。
使い方: PSELDNetsルートで src/preproc.py と同じ引数を渡す:
  python _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
      wav_format=.flac mode=extract_data paths.dataset_dir=$PWD/datasets_v12
"""
import os
import runpy
import sys

sys.path.insert(0, "src")
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
        return {99: []}              # イベント0件（ゼロ行列になる）
    return _load(path, *a, **kw)


def read_ok(path, *a, **kw):
    if _is_empty_file(path):
        return pd.DataFrame([[99, 0, 0, 0, 0]])   # num_frames=100算出専用の番兵
    return _read(path, *a, **kw)


du.load_output_format_file = load_ok
pd.read_csv = read_ok

runpy.run_path("src/preproc.py", run_name="__main__")
