# -*- coding: utf-8 -*-
"""SDE(距離推定)付き学習の起動ラッパ。PSELDNetsルートに置いて実行する。

使い方(サーバ):
    python _train_sde.py experiment=<run2と同じ実験名> \
        <run2と同じオーバーライド> [ckptは _make_sde_init_ckpt.py の出力を指定]
環境変数: SDE_W_DIST=5.0 (距離損失の重み)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

from train import main  # noqa: E402  (src/train.py の hydra main)

if __name__ == "__main__":
    main()
