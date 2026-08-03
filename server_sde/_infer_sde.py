# -*- coding: utf-8 -*-
"""SDE(距離推定)付き推論の起動ラッパ。PSELDNetsルートに置いて実行する。

予測CSVは各行 [frame, class, az, el, dist_m] の5列（パッチ済みwriter）。
使い方(サーバ):
    python _infer_sde.py experiment=outdoor_siren_v11_valinfer \
        ckpt_path=<sde run1 epoch_094> ほかrun2の推論と同じオーバーライド
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

from infer import main  # noqa: E402  (src/infer.py の hydra main)

if __name__ == "__main__":
    main()
