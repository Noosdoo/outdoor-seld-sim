# -*- coding: utf-8 -*-
"""SDEパッチの等価性テスト（サーバのPSELDNets環境で実行）。

T1: 新損失のxyz成分 == 既存ADPIT損失（距離行を挿入しただけの証明）
T2: 新ラベル抽出のse/azi/ele == 既存adpit h5（スロット規則の等価証明、200本）
T3: ckptヘッド拡張の形状・コピー・ゼロ初期化

使い方: cd ~/PSELDNets && python test_sde_units.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

# --- T3(依存なし)を先に -------------------------------------------------
from _make_sde_init_ckpt import expand_head  # noqa: E402

C = 6
w = torch.randn(C * 9, 32, 4, 3)
w_new, c_inf = expand_head(w)
assert c_inf == C and w_new.shape[0] == C * 12
for t in range(3):
    for a in range(3):
        src = w[(t * 3 + a) * C:(t * 3 + a + 1) * C]
        dst = w_new[(t * 4 + a) * C:(t * 4 + a + 1) * C]
        assert torch.equal(src, dst)
    assert torch.all(w_new[(t * 4 + 3) * C:(t * 4 + 4) * C] == 0)
print("T3 ckptヘッド拡張: PASS")

# --- 既存クラスを先に確保してからパッチ適用 ------------------------------
from loss.multi_accdoa import Losses as OrigLosses  # noqa: E402

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()
import loss.multi_accdoa as loss_mod  # noqa: E402

# --- T1: 損失の等価性 -----------------------------------------------------
torch.manual_seed(0)
B, T, NC = 2, 10, 6
tgt4 = torch.zeros(B, T, 6, 4, NC)
tgt4[:, :, :, 0, :] = (torch.rand(B, T, 6, NC) > 0.7).float()  # act
tgt4[:, :, :, 1:, :] = torch.randn(B, T, 6, 3, NC)             # xyz
out9 = torch.randn(B, T, 9 * NC)

# 5軸ターゲット(dist行=乱数)と12チャンネル出力(距離行挿入)を構成
tgt5 = torch.zeros(B, T, 6, 5, NC)
tgt5[:, :, :, :4, :] = tgt4
tgt5[:, :, :, 4, :] = torch.rand(B, T, 6, NC) * 4.0
o9 = out9.reshape(B, T, 9, NC)
o12 = torch.zeros(B, T, 12, NC)
for t in range(3):
    o12[:, :, t * 4:t * 4 + 3, :] = o9[:, :, t * 3:t * 3 + 3, :]
    o12[:, :, t * 4 + 3, :] = torch.randn(B, T, NC)  # 距離出力は任意
out12 = o12.reshape(B, T, 12 * NC)

orig = OrigLosses(None, "multi_accdoa")(
    {"multi_accdoa": out9}, {"adpit_label": tgt4})
new = loss_mod.Losses(None, "multi_accdoa")(
    {"multi_accdoa": out12}, {"adpit_label": tgt5})
diff = float(abs(orig["loss_adpit"] - new["loss_adpit"]))
assert diff < 1e-6, f"xyz損失が既存と不一致: diff={diff}"
assert new["loss_other"] > 0
print(f"T1 損失xyz等価: PASS (diff={diff:.2e}, dist項={float(new['loss_other']):.4f})")

# --- T2: ラベル抽出の等価性(既存adpit h5と照合、200本) --------------------
import h5py  # noqa: E402
from _preproc_sde import build_adpit_dist  # noqa: E402
import utils.data_utilities as du  # noqa: E402

ADPIT_H5 = sorted(Path("./").glob("_hdf5/label/adpit/*/outdoor_siren_v11.h5"))
META = Path("datasets/outdoor_siren_v11/metadata")
if ADPIT_H5 and META.is_dir():
    hf = h5py.File(ADPIT_H5[0], "r")
    n_ok = 0
    for meta_file in sorted(META.glob("*.csv"))[:200]:
        fn = meta_file.stem
        desc = du.load_output_format_file(meta_file)   # 5列(dist無し)でもOK
        se, azi, ele, _ = build_adpit_dist(desc, 6)
        g = hf[fn]["adpit"]
        assert np.array_equal(se, g["se"][:len(se)]), fn
        assert np.array_equal(azi, g["azi"][:len(azi)]), fn
        assert np.array_equal(ele, g["ele"][:len(ele)]), fn
        n_ok += 1
    hf.close()
    print(f"T2 抽出等価(既存h5と照合): PASS ({n_ok}本)")
else:
    print("T2 SKIP: 既存adpit h5またはmetadataが見つからない(パスを確認)")

print("ALL UNIT TESTS DONE")
