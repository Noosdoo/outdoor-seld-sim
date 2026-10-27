"""PC（CPU）で server_sde/_causal_infer_v17.py をそのまま動かす包み（2026-10-25 Opus・本人 21:11「ダウンロードしていいよ」）。
サーバの台本は書き換えない。**切り分け・手応えの確かめ専用**（凍結の「同じ GPU 種」と違うので、確定評価や実録の評価には使わない）。

- GPU の呼び出し（Tensor.cuda・Module.cuda・LightningModule の cuda・device="cuda"）を CPU に向け、torch.compile は使わない（計算の中身は同じ・最適化だけ外す）。
- 各本の生の出力（最終フレームの multi-ACCDOA）を LOCAL_RAW_DIR/<clip>.npy に残す（閾値を変えた後処理を、同じ関数で作り直すため・scripts/local_cpu_post_from_raw.py）。
- 前提（PC の準備・10/25 に済み）:
  - PC の PSELDNet/PSELDNets/.venv（Python 3.13・torch 2.13.0+cpu）に hydra-core 1.3.2・omegaconf 2.3.0・hydra-colorlog 1.2.0・lightning・torchmetrics・
    librosa 0.11.0・rich・PyYAML（PyPI）と torchaudio 2.11.0+cpu（PyTorch の CPU 版の置き場・--no-deps・2.13 は無い。使うのは Spectrogram・MelScale・AmplitudeToDB だけ）。
  - PC の PSELDNets には experiment の設定（outdoor_siren_v12 など）が無い → サーバの PSELDNets/configs を写した作業用の置き場を PS に渡す
    （scripts/local_cpu_infer.sh が作る・src は PC の写し＝サーバの 9269 の来歴とコード 48 ファイルの中身が一致（改行を正規化して・37 本は CRLF の差・HB25-08））。
使い方: scripts/local_cpu_infer.sh から呼ぶ（環境変数はサーバの server_sde/demo_bike_check_infer.sbatch と同じ）。
"""
import os
import runpy
import sys

import numpy as np
import soundfile as sf
import torch

torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
from lightning.fabric.utilities import device_dtype_mixin as _ddm  # noqa: E402  LightningModule は自前の cuda() を持つ

_ddm._DeviceDtypeModuleMixin.cuda = lambda self, *a, **k: self
_full = torch.full


def _full_cpu(*a, **k):
    if str(k.get("device", "")).startswith("cuda"):
        k["device"] = "cpu"
    return _full(*a, **k)


torch.full = _full_cpu
torch.compile = lambda m, *a, **k: m

RAW = os.environ["LOCAL_RAW_DIR"]
os.makedirs(RAW, exist_ok=True)
_cur = {"name": None}
_read = sf.read


def _read_track(path, *a, **k):
    _cur["name"] = os.path.basename(str(path))
    return _read(path, *a, **k)


sf.read = _read_track
sys.path.insert(0, os.path.join(os.environ["PS"], "src"))
sys.path.insert(0, os.path.join(os.environ["PS"]))
sys.path.insert(0, os.path.join(os.environ["E2E"], "server_sde"))   # 台本と同じ順（HERE が先）＝同じ sde_patch を読む
import sde_patch  # noqa: E402

_apply = sde_patch.apply_train_patches


def _apply_then_hook(*a, **k):
    """sde_patch が du.get_multi_accdoa_labels を差し替えたあとで、生の出力を残す包みをかぶせる（先にかぶせると上書きされる・10/25 の 1 回目の失敗）。"""
    r = _apply(*a, **k)
    import utils.data_utilities as du
    g = du.get_multi_accdoa_labels

    def _g_dump(causal, ncls, th):
        n = _cur["name"]
        if n and n.endswith(".flac"):
            np.save(os.path.join(RAW, n[:-5] + ".npy"), causal.detach().cpu().numpy())
        return g(causal, ncls, th)

    du.get_multi_accdoa_labels = _g_dump
    return r


sde_patch.apply_train_patches = _apply_then_hook
print(f"[cpu_wrap] torch {torch.__version__} threads {torch.get_num_threads()} raw → {RAW}", flush=True)
runpy.run_path(os.environ["CAUSAL_SCRIPT"], run_name="__main__")
