# -*- coding: utf-8 -*-
"""T6実データ版: 実際の DatasetMultiACCDOA.__getitem__ を通す距離注入検証。

Sol第8回再監査・条件3への対応。旧T6（test_sde_units2.py）はCSV→h5照合のみで、
実行時の距離注入経路（segments_listのi0/i1計算・datasetパス解決・padding）を
通していなかった。本テストは学習と同一のhydra設定・同一のDatasetクラスで
__getitem__ を呼び、返却ラベル第5軸をh5から独立に再構成した期待値と全照合する。
ケース: 先頭セグメント・中間・末尾（短尾+padding）・空ラベルクリップ。

使い方: cd ~/research/PSELDNet/PSELDNets && cp <repo>/server_sde/test_sde_units3.py . \
        && .venv/bin/python test_sde_units3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

import h5py  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from utils.config import get_dataset  # noqa: E402
import data.data as ddata  # noqa: E402

ROOT = Path.cwd()
with initialize_config_dir(config_dir=str(ROOT / "configs"),
                           version_base="1.3"):
    cfg = compose(config_name="train.yaml",
                  overrides=["experiment=outdoor_siren_v11"])

dataset = get_dataset(dataset_name="outdoor_siren_v11", cfg=cfg)
ds = ddata.DatasetMultiACCDOA(cfg, dataset, "outdoor_siren_v11",
                              ["fold1_room1"], "train")
N = len(ds.segments_list)
assert N > 10, f"segments_listが小さすぎる: {N}"
idxs = {"先頭": 0, "中間": N // 2, "末尾": N - 1}

# 空ラベルクリップ（metadata_dist 0バイト）のセグメントを1本追加
empty = None
for p in sorted((ROOT / "datasets/outdoor_siren_v11/metadata_dist")
                .glob("fold1_*.csv")):
    if p.stat().st_size == 0:
        empty = p.stem
        break
if empty is not None:
    for i, seg in enumerate(ds.segments_list):
        if Path(seg[0]).stem == empty:
            idxs["空ラベル"] = i
            break

h5path = sorted(ROOT.glob("_hdf5/label/adpit_dist/*/outdoor_siren_v11.h5"))
assert h5path, "adpit_dist h5が無い"
hf = h5py.File(h5path[0], "r")

ppp = ds.points_per_predictions
# 【Sol検証第2R】pad>0ケース: v11は全クリップ10秒でpaddingが自然発生しないため、
# 実在クリップの前方93フレームだけを切り出す合成セグメントを追加して
# 実__getitem__のpadding分岐を通す（index行の列構造 [path, begin, end,
# pad_before, pad_after] は本物と同一）。
seg0 = list(ds.segments_list[0])
ds.segments_list.append(
    [seg0[0], 0, int(93 * ppp), 0, int(7 * ppp)])
idxs["合成短尾(pad>0)"] = len(ds.segments_list) - 1
for name, idx in idxs.items():
    seg = ds.segments_list[idx]
    fn = Path(seg[0]).stem
    i0, i1 = int(seg[1] // ppp), int(seg[2] // ppp)
    dist = hf[f"{fn}/adpit/dist"][i0:i1, ...].astype(np.float32)

    sample = ds[idx]                      # ← 実__getitem__（パッチ済み）
    lab = sample["adpit_label"]
    assert lab.ndim == 4 and lab.shape[2] == 5, \
        f"{name}: ラベルが5軸でない {lab.shape}"
    se = lab[:dist.shape[0], :, 0, :]
    expect = se * sde_patch.dist_encode(dist)
    got = lab[:, :, 4, :]
    assert np.array_equal(got[:dist.shape[0]], expect.astype(got.dtype)), \
        f"{name}({fn}): 距離軸がh5再構成と不一致"
    assert not got[dist.shape[0]:].any(), f"{name}({fn}): padding部が非0"
    tag = f"h5 {i0}:{i1} dist>0 {int((dist > 0).sum()):,}"
    print(f"T6実データ getitem [{name}] {fn}: PASS ({tag}, "
          f"pad {lab.shape[0]-dist.shape[0]}fr)")

hf.close()
assert "空ラベル" in idxs, "空ラベルクリップのセグメントが見つからなかった(要調査)"
print("T6実データ版: ALL PASS (実DatasetMultiACCDOA.__getitem__経由)")
