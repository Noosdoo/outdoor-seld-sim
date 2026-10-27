#!/bin/bash
#SBATCH -J bBtrain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 72:00:00
#SBATCH -o bBtrain-%j.out
# 相手 B（変更票 #73）の学習: 公式ベースライン（train_seldnet.main・公式の既定の設定・白紙）を v29 の学習 split（fold3・49,682 本）で学習する。
#   準備（baseline_B_prepare.sbatch）が済んでいること（B_prepare_record.json の check.ok）。公式の学習に途中再開は無い → low で割り込まれると最初から。
#   使い方:
#     1 epoch の計測: B_NB_EPOCHS=1 B_JOB=timing sbatch -t 6:00:00 server_sde/baseline_B_train.sbatch → ログの [B-train] done の秒数から本番の見通し → #73 に epoch 数を固定
#     本番:           B_NB_EPOCHS=<#73 で固定した数> B_JOB=v29 sbatch server_sde/baseline_B_train.sbatch
#   出力: $B_MODEL_DIR/custom_<B_JOB>_dev_split0_multiaccdoa_foa_model.h5（公式の命名・val で最良の epoch の重み）。sha を記録。
# 本体 §3-6（2026-10-26）: 置き場は環境で替えられる（未設定なら下の v29 のまま）。v30 は B_DATASET_DIR=…/B_dataset2024_v30・B_FEAT_DIR=…/B_feat_v30c<_g倍率>・
#   B_MODEL_DIR=…/B_models_v30・B_OUT_DIR=…/B_out_v30 と B_INPUT_GAIN_DB=<決めた倍率>（特徴の置き場の印と照合）。距離を対数にするなら B_DIST_ENCODING=log1p（別の B_FEAT_DIR）。
#   (a) 1 epoch の損失の内訳: B_LOSS_BREAKDOWN=1 B_NB_EPOCHS=1 B_JOB=lossbd ＋ 上の v30 の置き場 → $B_OUT_DIR/loss_breakdown_lossbd.json（比率・支配 = 比率 > 0.5）
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
export E2E
export B_DATASET_DIR=${B_DATASET_DIR:-$HOME/research/_external/B_dataset2024_v29}
export B_FEAT_DIR=${B_FEAT_DIR:-$HOME/research/_external/B_feat_v29c}     # c = top_db なしの特徴（Astra RI73-02）。旧 B_feat_v29 の重み（v29・timing）は使わない
export B_MODEL_DIR=${B_MODEL_DIR:-$HOME/research/_external/B_models_v29}
export B_OUT_DIR=${B_OUT_DIR:-$HOME/research/_external/B_out_v29}
echo "B: dataset=$B_DATASET_DIR feat=$B_FEAT_DIR gain=${B_INPUT_GAIN_DB:-未設定} dist=${B_DIST_ENCODING:-linear} loss_breakdown=${B_LOSS_BREAKDOWN:-0}"
export B_JOB=${B_JOB:-v29}
[ -n "${B_NB_EPOCHS:-}" ] || { echo "B_NB_EPOCHS（epoch 数・#73 で固定した値、計測なら 1）を渡す"; exit 3; }
REC=$B_DATASET_DIR/B_prepare_record.json
[ -f "$REC" ] || { echo "準備の記録が無い（$REC）→ baseline_B_prepare.sbatch を先に"; exit 3; }
$PY -c "import json,sys;r=json.load(open('$REC'));c=r['steps']['check'];print('prepare check:',c);sys.exit(0 if c.get('ok') and r['limit_train']==0 and r['limit_val']==0 else 3)" || { echo "準備の検査が通っていない・または試走の制限つき → 止める"; exit 3; }
M=$B_MODEL_DIR/custom_${B_JOB}_dev_split0_multiaccdoa_foa_model.h5
[ -e "$M" ] && { echo "同じ札の重みが既にある（$M）。札（B_JOB）を変えるか、消すなら本人に聞く"; exit 4; }
cd ~/research/_external/DCASE2024_seld_baseline && echo "official commit $(git rev-parse --short HEAD)" && [ -z "$(git status --porcelain)" ] || echo "注意: 公式コードに変更がある"; cd $E2E
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
$PY $E2E/server_sde/_baseline_B_train.py
ls -la $B_MODEL_DIR | cut -c1-120
sha256sum "$M" $E2E/server_sde/_baseline_B_common.py $E2E/server_sde/_baseline_B_train.py "$REC" | cut -c1-16,65-
[ "${B_LOSS_BREAKDOWN:-0}" = 1 ] && cat "${B_LOSS_BREAKDOWN_OUT:-$B_OUT_DIR/loss_breakdown_${B_JOB}.json}"
echo BASELINE_B_TRAIN_DONE
