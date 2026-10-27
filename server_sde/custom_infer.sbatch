#!/bin/bash
#SBATCH -J custinfer
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH -o custinfer-%j.out
# デモ v2 の自作場面（オラクル）を本物の検出層に通す（2026-09-03 本人「最初から正解を渡していないですよね？」への対応）。
# 入力: ~/research/outdoor_seld_e2e/out/joycon_demo_v2/model_infer/foa/custom_*.flac（v3 ツール --save-foa の 4ch 24kHz 10s）
# モデル: 採用の因果ft ft2 epoch_099（fold32 の「本物のモデル出力」A〜D と同じ）
# 出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_demo_custom_causal/val_all_causal.csv
#
#   sbatch server_sde/custom_infer.sbatch      （pro_6000 に出すなら -p pro_6000 --gres=gpu:pro_6000:1）
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
export CAUSAL_FOA=$HOME/research/outdoor_seld_e2e/out/joycon_demo_v2/model_infer/foa
export CAUSAL_PREFIX=custom_
export CAUSAL_CKPT=${CAUSAL_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_causal_ft2/checkpoints/epoch_099.ckpt}
export CAUSAL_OUT=infer_demo_custom_causal
ls $CAUSAL_FOA | head -3
ls -l $CAUSAL_CKPT
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py

guard_verify || exit 1
echo CUSTOM_INFER_DONE
