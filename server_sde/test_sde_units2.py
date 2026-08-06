# -*- coding: utf-8 -*-
"""SDEパッチの追加等価性テスト T4-T6（第8回監査の指摘対応）。

T1-T3（test_sde_units.py）は「損失・ラベル・初期重み」のt=0等価しか
証明していない、という監査指摘を受けた最小の追加網:
  T4: 実ckptでのヘッドforward等価 — run2 ep094(9C)と拡張ckpt(12C)を同一入力で
      畳み込み、xyzチャンネル完全一致・距離チャンネル恒等0を確認
      （T3の「重み検査」を「挙動検査」に格上げ。エンコーダはコード非接触のため対象外）
  T5: デコードチェーン等価 — 乱数logitsで 上流(get_multi_accdoa_labels→
      dcase_format→polar) と SDE版の[class,az,el]出力が完全一致
      （run1で実際に非等価事故があった経路。回帰網）
  T6: adpit_dist h5のdist値をCSVから独立再構築して全照合＋フレーム数一致
      （T2の「dist値はh5と未照合」穴を閉じる）

使い方: cd ~/research/PSELDNet/PSELDNets && cp <repo>/server_sde/test_sde_units2.py . \
        && .venv/bin/python test_sde_units2.py
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

# --- 既存(上流)関数をパッチ前に確保 --------------------------------------
from utils.data_utilities import (  # noqa: E402
    get_multi_accdoa_labels as orig_gml,
    multi_accdoa_to_dcase_format as orig_mtd,
    convert_output_format_cartesian_to_polar as orig_ctp,
)

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()
import utils.data_utilities as du  # noqa: E402

HOME = os.path.expanduser("~")

# --- T4: 実ckptヘッドのforward等価 ---------------------------------------
run2 = sorted(glob.glob(
    f"{HOME}/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_run2/"
    "checkpoints/epoch_*.ckpt"))
init = f"{HOME}/PSELDNets_logs/sde_init_from_run2ep94.ckpt"
assert run2 and os.path.exists(init), "T4前提ckptが無い(SKIPは許さずFAIL)"


def head_wb(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("net.", "").replace("_orig_mod.", ""): v
          for k, v in sd.items()}
    return sd["tscam_conv.weight"], sd["tscam_conv.bias"]


w9, b9 = head_wb(run2[-1])
w12, b12 = head_wb(init)
C = w9.shape[0] // 9
assert w12.shape[0] == C * 12, (w9.shape, w12.shape)
torch.manual_seed(0)
x = torch.randn(2, w9.shape[1], w9.shape[2], 31)
y9 = F.conv2d(x, w9, b9, padding=(0, 1))     # [2, 9C, 1, T']
y12 = F.conv2d(x, w12, b12, padding=(0, 1))  # [2, 12C, 1, T']
worst = 0.0
for t in range(3):
    for a in range(3):
        s9 = y9[:, (t * 3 + a) * C:(t * 3 + a + 1) * C]
        s12 = y12[:, (t * 4 + a) * C:(t * 4 + a + 1) * C]
        worst = max(worst, float((s9 - s12).abs().max()))
    d = y12[:, (t * 4 + 3) * C:(t * 4 + 4) * C]
    assert float(d.abs().max()) == 0.0, "距離チャンネルが非0(ゼロ初期化破れ)"
assert worst == 0.0, f"xyzチャンネル不一致: {worst}"
# tanhは時間集約(線形)の後段で両者ともxyz行にのみ掛かる(sde_patch forward)ため、
# 畳み込み一致⇒集約後のtanh出力も一致。
print(f"T4 実ckptヘッドforward等価: PASS (xyz diff={worst:.1e}, dist≡0, "
      f"src={Path(run2[-1]).name})")

# --- T5: デコードチェーン等価 ---------------------------------------------
torch.manual_seed(1)
NC, T = 6, 60
acc9 = torch.randn(1, T, 9 * NC) * 0.9        # 多数がsed閾値0.5を跨ぐ強度
o9 = acc9.reshape(1, T, 9, NC)
o12 = torch.zeros(1, T, 12, NC)
for t in range(3):
    o12[:, :, t * 4:t * 4 + 3] = o9[:, :, t * 3:t * 3 + 3]
    o12[:, :, t * 4 + 3] = torch.rand(1, T, NC) * 3.0   # 距離行(任意)
acc12 = o12.reshape(1, T, -1)

sed9, doa9 = orig_gml(acc9, NC)
sed12, doa12 = du.get_multi_accdoa_labels(acc12, NC)
assert torch.equal(sed9, sed12), "sed判定が不一致"
d9 = orig_mtd(sed9[:, 0].numpy().astype(float), doa9[:, 0].numpy(),
              nb_classes=NC)
d12 = du.multi_accdoa_to_dcase_format(sed12[:, 0].numpy().astype(float),
                                      doa12[:, 0].numpy(), nb_classes=NC)
p9 = orig_ctp(d9)
p12 = du.convert_output_format_cartesian_to_polar(d12)
assert set(p9.keys()) == set(p12.keys()), "フレーム集合が不一致"
n_rows = 0
for f in p9:
    r9 = sorted([float(v) for v in row[:3]] for row in p9[f])
    r12 = sorted([float(v) for v in row[:3]] for row in p12[f])
    assert len(r9) == len(r12), f"frame {f}: 行数不一致 {len(r9)} vs {len(r12)}"
    for a, b in zip(r9, r12):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9, (f, a, b)
    for row in p12[f]:
        assert len(row) == 4 and row[3] >= 0.0, "dist列欠落/負値"
    n_rows += len(r9)
assert n_rows > 50, f"T5の検定力不足(行数{n_rows})"
print(f"T5 デコードチェーン等価: PASS ({len(p9)}フレーム/{n_rows}行、"
      f"[class,az,el]完全一致・dist同乗)")

# 往復: 6列書出→自前ローダ読戻しで az/el/dist が保存されること
tmp = Path("./_t5_roundtrip.csv")
du.write_output_format_file(tmp, p12)
back = du.load_output_format_file(tmp)
for f in p12:
    rows_w = sorted((int(r[0]), int(r[1]), int(r[2]), round(float(r[3]), 2))
                    for r in p12[f])
    rows_r = sorted((int(r[0]), int(r[1]), int(r[2]), round(float(r[3]), 2))
                    for r in back[f])
    assert rows_w == rows_r, (f, rows_w[:2], rows_r[:2])
tmp.unlink()
print("T5b 6列CSV往復: PASS")

# --- T6: adpit_dist h5のdist値をCSVから全照合 ------------------------------
import h5py  # noqa: E402
from _preproc_sde import build_adpit_dist  # noqa: E402

H5 = sorted(Path("./").glob("_hdf5/label/adpit_dist/*/outdoor_siren_v11.h5"))
META = Path("datasets/outdoor_siren_v11/metadata_dist")
assert H5 and META.is_dir(), "T6前提が無い(adpit_dist h5 / metadata_dist)"
hf = h5py.File(H5[0], "r")
n_ok = n_dist = 0
for meta_file in sorted(META.glob("*.csv"))[:200]:
    fn = meta_file.stem
    desc = du.load_output_format_file(meta_file)
    se, azi, ele, dist = build_adpit_dist(desc, 6)
    g = hf[fn]["adpit"]
    h_dist = g["dist"][...]
    assert h_dist.shape[0] == len(dist), \
        f"{fn}: フレーム数不一致 h5={h_dist.shape[0]} csv={len(dist)}"
    assert np.array_equal(h_dist, dist), f"{fn}: dist値がh5と不一致"
    assert h_dist.shape[0] == g["se"][...].shape[0], f"{fn}: se/dist長不一致"
    n_dist += int((dist > 0).sum())
    n_ok += 1
hf.close()
assert n_dist > 0
print(f"T6 dist値h5全照合+フレーム数一致: PASS ({n_ok}本, dist>0要素 {n_dist:,})")

print("ALL UNIT TESTS (T4-T6) DONE")
