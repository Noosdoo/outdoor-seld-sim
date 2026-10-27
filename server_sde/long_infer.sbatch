#!/bin/bash
#SBATCH -J longinfer
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o longinfer-%j.out
# 長尺セット v1（60 秒 × 400 本）を採用モデル（ft2 e099）の因果推論に掛ける（10 秒の右詰め窓を 0.1 秒ずつ）。
# 入力: ~/research/outdoor_seld_e2e/out/dataset_outdoor_long_v1/foa/long_*.flac（step11_long_render.py）
# 出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_long_v1_causal/val_all_causal.csv（frame 0..599）
#
#   sbatch server_sde/long_infer.sbatch            （pro_6000: -p pro_6000 --gres=gpu:pro_6000:1）
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_FOA=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_long_v1/foa
export CAUSAL_PREFIX=${CAUSAL_PREFIX:-long_}
export CAUSAL_CKPT=${CAUSAL_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_causal_ft2/checkpoints/epoch_099.ckpt}
export CAUSAL_OUT=${CAUSAL_OUT:-infer_long_v1_causal}
export CAUSAL_BATCH=${CAUSAL_BATCH:-50}
ls $CAUSAL_FOA | wc -l
ls -l $CAUSAL_CKPT
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_long.py

guard_verify || exit 1
echo LONG_INFER_DONE
