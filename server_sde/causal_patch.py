# -*- coding: utf-8 -*-
"""因果窓での学習パッチ（2026-08-19）— 学習時から未来を見せない。

## なぜ

推論を因果（各時刻までの音だけ）にすると、至近車への距離推定が systematically
遠めに出て、至近到達が 85.0%→50.2% に落ちた（設計文書 §6-sexies）。
分離能はほぼ保たれている（AUC 0.996→0.987）ので情報はあり、
検証データで作った分位マッピングで 62.8% までは戻せた（§6-septies）。
残る差の原因は**学習と推論の食い違い**である。学習時はどの出力フレームも
前後両方の音を見られるのに、推論時の最終フレームには未来が無い。

そこで**学習時の入力を推論時とまったく同じ形にする**:

  各クリップから終端フレーム k をランダムに選び、
  「先頭から k フレームぶんの音を右詰めし、残りをゼロ埋めした10秒窓」を作る。
  ラベルも同じように右詰めし、**最後の1フレームだけ**を教師にする。

こうすると、教師に使う唯一のフレームは未来を一切見ていない。推論時と同一条件である。

## 代償（正直に）

1窓あたり教師フレームが1つだけなので、勾配の信号は通常学習の 1/100 になる。
そのぶん REPEATS で1クリップを複数回サンプルして補う（**既定2**。ft1/ft2 の実運用値。
2026-09-03 に docstring と既定値を実運用に揃えた＝第11回監査 論点7。REPEAT=8 は ④感度測定で
「減衰後に安定・強到達+2.9pt」を確認済みで、v2 の束の設計で採否を決める）。
学習済み ckpt からの微調整前提であり、ゼロから学習する用途ではない。

## 使い方

  import causal_patch; causal_patch.apply()

環境変数:
  CAUSAL_KMIN   窓の終端フレームの下限（既定40。先頭は実音が少なくノイズになる）
  CAUSAL_REPEAT 1クリップを1エポックで何回サンプルするか（既定2）
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.utils.data

KMIN = int(os.environ.get("CAUSAL_KMIN", "40"))
REPEAT = int(os.environ.get("CAUSAL_REPEAT", "2"))   # 実運用の既定（sbatch と同値）
NFRAME = 100                      # 10秒 / 0.1秒
SEED = int(os.environ.get("CAUSAL_SEED", "0"))


def _draw_k(idx):
    """終端フレーム k を引く。

    【2026-08-19 監査①の修正】以前はモジュール変数の乱数を使っていたが、
    num_workers=8 だと8つのワーカーが同じ種を複製するため、
    各クリップが毎エポックほぼ同じ k を見ていた可能性があった。
    torch.initial_seed() は DataLoader が**ワーカーごと・エポックごと**に
    振り直すので、これに idx を混ぜれば worker/epoch/item のすべてで変わる。
    """
    # numpy整数(idx)と巨大なPython整数(torch.initial_seed)を直接足すと
    # numpyがint64へ変換しようとして溢れる。先に縮めて全部Python intにする。
    w = torch.utils.data.get_worker_info()
    base = int(torch.initial_seed()) % (2 ** 32) if w else SEED
    seed = (base + SEED + int(idx) * 2654435761) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return int(rng.integers(KMIN, NFRAME + 1))


def _right_align(x, lab, k):
    """音とラベルを「先頭kフレームぶんを右詰め・残りゼロ」に作り替える。"""
    n = x.shape[-1]
    hop = n // NFRAME
    xn = np.zeros_like(x)
    xn[..., n - k * hop:] = x[..., :k * hop]
    ln = np.zeros_like(lab)
    ln[NFRAME - k:] = lab[:k]
    return xn, ln


def apply():
    import data.data as dd
    import loss.multi_accdoa as loss_mod

    # ---- 1) データセット: 右詰め窓にする + 1クリップをREPEAT回サンプルする ----
    ds_cls = dd.DatasetMultiACCDOA
    _init, _get = ds_cls.__init__, ds_cls.__getitem__

    def init_causal(self, *a, **kw):
        _init(self, *a, **kw)
        if getattr(self, "dataset_type", "") == "train" and REPEAT > 1:
            self.segments_list = list(self.segments_list) * REPEAT

    def get_causal(self, idx):
        s = _get(self, idx)
        if getattr(self, "dataset_type", "") != "train":
            return s                      # 検証・評価は通常どおり（比較の土俵を保つ）
        lab = s["adpit_label"]
        if lab.shape[0] != NFRAME:
            return s
        k = _draw_k(idx)
        s["data"], s["adpit_label"] = _right_align(s["data"], lab, k)
        return s

    ds_cls.__init__, ds_cls.__getitem__ = init_causal, get_causal

    # ---- 2) 損失: 最後の1フレームだけを使う ----
    _Losses = loss_mod.Losses

    class LossesCausal(_Losses):
        def __call__(self, output, target, epoch_it=0):
            o = {k: (v[:, -1:] if torch.is_tensor(v) and v.dim() >= 2 else v)
                 for k, v in output.items()}
            t = {k: (v[:, -1:] if torch.is_tensor(v) and v.dim() >= 2 else v)
                 for k, v in target.items()}
            return super().__call__(o, t, epoch_it)

    loss_mod.Losses = LossesCausal
    print(f"[causal] 学習を因果窓に変更: KMIN={KMIN} REPEAT={REPEAT} "
          f"（教師は最後の1フレームのみ）", flush=True)
