#!/bin/bash
#SBATCH -J v12causal
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 06:00:00
#SBATCH -o v12causal-%j.out
# v12(w3) の因果推論（未来を見ない）— 通知v4.1のリアルタイム検証用。
#
# 読むのは FOA音声 と ckpt だけ。書くのは新しい experiment_name のディレクトリのみ。
# 既存の推論結果・_hdf5・データセットには触れない。
# QoS=low で中断されても、1クリップごとに保存しているので再投入で続きから走る。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS

source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

export PS
export CAUSAL_CKPT=${CAUSAL_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt}
export CAUSAL_OUT=${CAUSAL_OUT:-infer_v12_w3_causal_val}
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_PREFIX=${CAUSAL_PREFIX:-fold2_}
export CAUSAL_FOA=${CAUSAL_FOA:-}
export CAUSAL_LIMIT=${CAUSAL_LIMIT:-0}

ls -l $CAUSAL_CKPT
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py

guard_verify
echo CAUSAL_SBATCH_DONE
