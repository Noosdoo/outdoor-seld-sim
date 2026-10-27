#!/bin/bash
#SBATCH -J hpclast
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o hpclast-%j.out
# ④ハイパラ感度 Stage C の補助: 因果ft arm の last.ckpt（=最終エポック）を val 1,800 の因果推論に掛ける。
# hp_causal.sbatch は epoch_*.ckpt（EVERY毎）だけ推論するので、最終エポック（例: e150 の e149）が抜ける。
# 出力= infer_hp_<arm>_e<LAST>_causal/（LAST は既定 149。他の arm なら --export で指定）
#
#   sbatch --export=ALL,ARM=clr1e-4e150,LAST=149 server_sde/hp_causal_infer_last.sbatch
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

: "${ARM:?ARM を指定してください}"
LAST=${LAST:-149}
EXP=outdoor_siren_v12_hp_${ARM}
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export PS
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_PREFIX=fold2_
ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
ls -l $ck
export CAUSAL_CKPT=$ck
export CAUSAL_OUT=infer_hp_${ARM}_e${LAST}_causal
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py

guard_verify || exit 1
echo HP_CAUSAL_LAST_DONE_$ARM
