# -*- coding: utf-8 -*-
"""v17: 装着高さ（マイク高さ mic_z）をモデルの入力に与えるモンキーパッチ（2026-09-05・本人「高さ入力で行きましょう」）。

sde_patch.apply_train_patches()（と causal_patch.apply()）の**後**に apply() を呼ぶ。既定 OFF（HEIGHT_COND=1 で有効）。
設計 = md/design/v17候補学習_装着高さの入力_事前宣言_2026-09-05.md §2

  1. データセット: クリップ名 → mic_z を表（HEIGHT_TABLE: plan の csv、clip_id と mic_z 列）から引き、
     正規化 h = (z − 1.75) / 0.35 を sample["mic_z"] に入れる。学習時だけ N(0, HEIGHT_JITTER) の雑音を足す。
     表に無いクリップ（v12 val など）は HEIGHT_DEFAULT（1.5 m）。HEIGHT_OFFSET は感度評価用の一律ずらし
  2. Lightning の step: batch の mic_z を net._mic_z に置く（forward の引数を変えずに渡す）
  3. モデル: HTSAT_SDE を継承した HTSAT_SDE_H。h → MLP(1→64→2×特徴次元) → FiLM（特徴 × (1+γ) + β）を
     エンコーダ出力（[B, C, F', T']）に掛けてから tscam_conv へ。最終層はゼロ初期化＝初期状態は無条件モデルと同一。
     pretrained_path の ckpt に h_mlp があれば読む（HTSAT.load_ckpts は __init__ 時点で h_mlp が無いので自前で読む）
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("HEIGHT_COND", "0") == "1"
TABLE = os.environ.get("HEIGHT_TABLE", "")
DEFAULT = float(os.environ.get("HEIGHT_DEFAULT", "1.5"))
JITTER = float(os.environ.get("HEIGHT_JITTER", "0.05"))
OFFSET = float(os.environ.get("HEIGHT_OFFSET", "0.0"))
Z0, ZS = 1.75, 0.35                     # 学習範囲 1.4〜2.1 m の中心と半幅
HIDDEN = 64
_table = None
SEED = int(os.environ.get("HEIGHT_SEED", "20260917"))
_rng = np.random.default_rng(SEED)   # 旧: module-global（DataLoader の各ワーカーが複製して同じ列を引く・レビュー F9）。学習のゆらぎには使わない


def _jitter_rng(idx: int) -> np.random.Generator:
    """依頼 3-2（敵対レビュー F9・2026-09-14）: 高さゆらぎの乱数を item ごとに引く。
    num_workers=8 の各ワーカーは fork/spawn で module-global の _rng を複製するので、同じ乱数列が 8 本並んでいた。
    causal_patch の終端フレームの修正（2026-08-19）と同じく、torch.initial_seed()（DataLoader がワーカーごと・エポックごとに
    振り直す）と HEIGHT_SEED と idx を混ぜ、worker / epoch / item のすべてで変える。周辺分布は N(0, JITTER) のまま。"""
    import torch
    w = torch.utils.data.get_worker_info()
    base = int(torch.initial_seed()) % (2 ** 32) if w else SEED
    return np.random.default_rng((base + SEED + int(idx) * 2654435761) % (2 ** 32))


def norm(z: float) -> float:
    return (float(z) - Z0) / ZS


def load_table() -> dict:
    global _table
    if _table is None:
        _table = {}
        if TABLE:
            with open(TABLE, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("mic_z"):
                        _table[r["clip_id"]] = float(r["mic_z"])
        print(f"[height] table={TABLE or '(none)'} entries={len(_table)} default={DEFAULT} jitter={JITTER} offset={OFFSET}", flush=True)
    return _table


def lookup(clip: str) -> float:
    return load_table().get(clip, DEFAULT)


def apply():
    assert ENABLED, "HEIGHT_COND=1 のときだけ呼ぶ"
    import torch
    import torch.nn as nn
    import data.data as ddata
    import models.multi_accdoa as ma_mod
    import models.accdoa as accdoa_mod
    import models.model_module as top_mm

    load_table()

    # ---- 1) データセット ----
    _orig_getitem = ddata.DatasetMultiACCDOA.__getitem__

    def _getitem_h(self, idx):
        sample = _orig_getitem(self, idx)
        fn = Path(self.segments_list[idx][0]).stem
        z = lookup(fn)
        if getattr(self, "dataset_type", "") == "train" and JITTER > 0:
            z += float(_jitter_rng(idx).normal(0.0, JITTER))   # 3-2: ワーカー別・エポック別・item 別の種
        sample["mic_z"] = np.float32(norm(z + OFFSET))
        return sample

    ddata.DatasetMultiACCDOA.__getitem__ = _getitem_h

    # ---- 2) Lightning step: batch の mic_z を net に置く ----
    MM = top_mm.SELDModelModule

    def _wrap(orig):
        def step(self, batch_sample, batch_idx):
            z = batch_sample.get("mic_z") if isinstance(batch_sample, dict) else None
            net = getattr(self.net, "_orig_mod", self.net)
            net._mic_z = None if z is None else torch.as_tensor(z).to(self.device)
            return orig(self, batch_sample, batch_idx)
        return step

    for name in ("training_step", "validation_step", "test_step"):
        setattr(MM, name, _wrap(getattr(MM, name)))

    # ---- 3) モデル ----
    Base = ma_mod.HTSAT                      # sde_patch 適用後は HTSAT_SDE
    XYZ_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    interpolate = accdoa_mod.interpolate

    class HTSAT_SDE_H(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nf = self.encoder.num_features
            self.h_mlp = nn.Sequential(nn.Linear(1, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2 * nf))
            nn.init.zeros_(self.h_mlp[2].weight)
            nn.init.zeros_(self.h_mlp[2].bias)
            self._mic_z = None
            self._n_missing = 0
            pre = kwargs.get("pretrained_path") or (args[4] if len(args) > 4 else None)
            if pre and os.path.exists(str(pre)):
                sd = torch.load(str(pre), map_location="cpu", weights_only=False)
                sd = sd.get("state_dict", sd)
                sd = {k.replace("net.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
                keys = [k for k in sd if k.startswith("h_mlp.")]
                if keys:
                    self.h_mlp.load_state_dict({k[len("h_mlp."):]: sd[k] for k in keys})
                    print(f"[height] h_mlp loaded from {pre} ({len(keys)} tensors)", flush=True)
                else:
                    print(f"[height] h_mlp not in {pre} -> zero-init (= 無条件モデルと同一の出発点)", flush=True)
            print(f"[height] FiLM on encoder features (C={nf}, hidden={HIDDEN})", flush=True)

        def forward(self, x):
            B, C, T, F = x.shape
            if self.output_frames is None:
                self.output_frames = int(T // self.pred_res)
            assert self.output_frames == self.tgt_output_frames, "v17 は 10 秒入力のみ"
            x = x.transpose(1, 3)
            for nch in range(x.shape[-1]):
                x[..., [nch]] = self.scalar[nch](x[..., [nch]])
            x = x.transpose(1, 3)
            x = self.encoder(x)                       # [B, C, F', T']
            nf = x.shape[1]
            z = self._mic_z
            if z is None:
                self._n_missing += 1
                if self._n_missing <= 3:
                    print("[height] WARNING: mic_z が渡されていない → 1.75 m 相当 (h=0) で推論", flush=True)
                z = torch.zeros(B, device=x.device)
            z = torch.as_tensor(z, device=x.device).to(x.dtype).reshape(-1)
            if z.shape[0] != B:                       # AugMix 等でバッチが複製されたとき
                z = z.repeat((B + z.shape[0] - 1) // z.shape[0])[:B]
            gb = self.h_mlp(z.view(-1, 1))
            g, b = gb[:, :nf], gb[:, nf:]
            x = x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
            x = self.tscam_conv(x)
            x = torch.flatten(x, 2)
            x = x.permute(0, 2, 1).contiguous()
            x = self.fc(x)
            x = interpolate(x, ratio=self.encoder.time_res, method="bilinear")
            x = x[:, :self.output_frames * self.pred_res]
            x = x.reshape(B, self.output_frames, self.pred_res, -1).mean(dim=2)
            x = self.final_act(x)                     # Identity（sde_patch）
            B2, T2, _ = x.shape
            o = x.reshape(B2, T2, 12, self.num_classes).clone()
            o[:, :, XYZ_IDX] = torch.tanh(o[:, :, XYZ_IDX])
            return {"multi_accdoa": o.reshape(B2, T2, -1)}

    ma_mod.HTSAT = HTSAT_SDE_H
    print(f"[height] patches applied (table entries={len(load_table())}, jitter={JITTER}, offset={OFFSET})", flush=True)
