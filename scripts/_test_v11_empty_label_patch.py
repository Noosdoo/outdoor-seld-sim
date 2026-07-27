# -*- coding: utf-8 -*-
"""v11空ラベル対策の実証テスト（最終採用版=0バイト判定ラップの通し確認）。

確認事項:
 1) 素のPSELDNetsは空CSVで停止する（pd.read_csv / load_output_format_file の両方
    = preproc経路と学習時val GT経路(data/components/data.py:98)が使う読み口）
 2) 採用版パッチ（0バイト判定。ノートブックの _preproc_emptyok.py / _train_emptyok.py と
    同一ロジック）で、空クリップが num_frames=100・イベント0件になり、
    mACCDOA(ADPIT)経路の先頭 list(keys())[-1] も通ること
 3) 【回帰】例外catch方式の合成事故——pd.read_csvを先に全域ラップすると
    load_output_format_file が番兵行を実イベント化する——が、0バイト判定版では
    起きないこと（frame99クラス0の偽イベント混入なし）

実行: PSELDNets/.venv の python で。パスは本ファイル位置から解決（絶対パスなし）。
終了コード: 全PASS=0 / 失敗=assertで非0。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PS = HERE.parents[2] / "PSELDNet" / "PSELDNets"
sys.path.insert(0, str(PS / "src"))

from utils.data_utilities import load_output_format_file  # noqa: E402

tmp = Path(tempfile.mkdtemp())
empty = tmp / "fold1_room1_mix0001.csv"
empty.write_text("", encoding="utf-8")
normal = tmp / "fold1_room1_mix0002.csv"
normal.write_text("10,4,0,45,0\n11,4,0,44,0\n96,0,0,-30,-2\n", encoding="utf-8")

# --- 1) 素の挙動: 空CSVで両読み口とも停止すること ---
for fn, name in ((lambda: pd.read_csv(empty, header=None, sep=','), "pd.read_csv"),
                 (lambda: load_output_format_file(empty), "load_output_format_file")):
    try:
        fn()
        raise AssertionError(f"{name}: 空CSVで例外が出ない（上流仕様が変わった疑い）")
    except pd.errors.EmptyDataError:
        print(f"1) {name}: EmptyDataError を確認")

# --- 2) 採用版パッチ（0バイト判定、ラッパと同一ロジック）の通し確認 ---
_load, _read = load_output_format_file, pd.read_csv


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


for f, want_frames, want_ev in ((empty, 100, 0), (normal, 97, 3)):
    df = read_ok(f, header=None, sep=',').values
    num_frames = df[-1, 0] + 1
    meta = load_ok(f)
    se = np.zeros((num_frames, 6))
    for fr, evs in meta.items():
        if fr < num_frames:
            for ev in evs:
                se[fr, ev[0]] = 1
    assert num_frames == want_frames and int(se.sum()) == want_ev, \
        (f.name, num_frames, se.sum())
    nb = list(meta.keys())[-1]   # ADPIT経路の先頭（空dictならIndexErrorになる箇所）
    assert nb >= 96, nb
    print(f"2) {f.name}: num_frames={num_frames} イベント={int(se.sum())} "
          f"ADPIT nb={nb} OK")

# --- 3) 回帰: 例外catch方式の合成事故が採用版では起きない ---
pd.read_csv = read_ok            # read側を先に全域ラップ（事故の前提条件を再現）
try:
    poisoned = _load(empty)      # 例外catch方式相当: 番兵行が実イベント化する
    n_ev = sum(len(v) for v in poisoned.values())
    assert n_ev > 0, "事故が再現しない（load_output_format_fileの実装が変わった疑い）"
    print(f"3) 参考: 素のloadは番兵行を{n_ev}イベントとして返す（旧・例外catch方式の事故）")
    ours = load_ok(empty)        # 採用版は0バイト判定が先に効くので汚染されない
    assert ours == {99: []}, ours
    print("3) 採用版 load_ok: read側ラップ下でも {99: []}（偽イベント混入なし）OK")
finally:
    pd.read_csv = _read

print("\nEMPTY-LABEL PATCH TEST: ALL PASS")
