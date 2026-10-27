# -*- coding: utf-8 -*-
"""相手 B（変更票 #73）: 公式ベースラインの学習をそのまま走らせる（train_seldnet.main）。params だけこの研究用に差し替える（monkeypatch）。
環境変数: B_NB_EPOCHS（既定 = 公式 250）・B_QUICK=1（公式の quick_test = 2 epoch）・B_BATCH（既定 = 公式 128）・B_JOB（出力名の札）
  本体 §3-6（2026-10-26）: B_INPUT_GAIN_DB・B_DIST_ENCODING（_baseline_B_common の 15・16・特徴の置き場の印と照合）・
  B_LOSS_BREAKDOWN=1（1 epoch の損失の内訳 → $B_OUT_DIR/loss_breakdown_<job>.json・B_LOSS_BREAKDOWN_OUT で上書き・common の 17）
出力: $B_MODEL_DIR/custom_<job>_dev_split0_multiaccdoa_foa_model.h5（公式の命名）・val の予測 csv は $B_OUT_DIR
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _baseline_B_common as C  # noqa: E402

C.setup_path()
P = C.get_params(nb_epochs=(int(os.environ["B_NB_EPOCHS"]) if os.environ.get("B_NB_EPOCHS") else None),
                 quick=os.environ.get("B_QUICK") == "1",
                 batch_size=(int(os.environ["B_BATCH"]) if os.environ.get("B_BATCH") else None))
import parameters  # noqa: E402

parameters.get_params = lambda argv="1": dict(P)          # 公式 main が読む params をこの研究用に
import train_seldnet  # noqa: E402

C.patch_compute_results()       # 正解の無い本（負例）で公式の評価の初期化が止まらないように包む（docstring の 7・試走 8598・公式ファイルは変えない）
job = os.environ.get("B_JOB", "v29")
C.check_feat_marker(C.FEAT)     # 本体 §3-6: 特徴の置き場の印（倍率・距離の符号化）と環境が同じか（違えば止める・印の無い旧の置き場 = 0 dB・linear）
C.patch_dist_decode()           # 本体 §3-6: log1p のときだけ公式の復号を m に戻す（val の指標）
LOSS_BD = None
if os.environ.get("B_LOSS_BREAKDOWN") == "1":
    LOSS_BD = Path(os.environ.get("B_LOSS_BREAKDOWN_OUT", str(C.OUT_DIR / f"loss_breakdown_{job}.json")))
    C.patch_loss_breakdown(LOSS_BD, P)
    print(f"[B-train] 損失の内訳を記録 → {LOSS_BD}（学習の値は変えない）", flush=True)
print(f"[B-train] nb_epochs={P['nb_epochs']} quick={P['quick_test']} batch={P['batch_size']} classes={P['unique_classes']} dataset_dir={P['dataset_dir']} feat_label_dir={P['feat_label_dir']} "
      f"gain={C.INPUT_GAIN_DB:g} dB dist={C.DIST_ENCODING}", flush=True)
t0 = time.time()
train_seldnet.main(["", "custom", job])
print(f"[B-train] done {time.time() - t0:.0f}s", flush=True)
print("B_TRAIN_DONE")
