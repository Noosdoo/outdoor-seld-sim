#!/bin/bash
#SBATCH -J v15bxinf
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o v15bxinf-%j.out
# v15b（高さ 2.0〜2.1 m 固定で学習）の因果ft ckpt を、v15 の fold2 val（高さ 1.4〜2.1 m）に当てる。
# 目的: 「1.4〜2.1 全体」での頑健性を v15 / v15c と同じ土俵で比べる（本人 2026-09-04 12:42「1.4〜2.1 の評価はないの？」）。
#   sbatch --dependency=afterok:<v15b_causal job> server_sde/v15b_infer_on_v15val.sbatch
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_DSDIR=$PS/datasets_v15
export CAUSAL_PREFIX=fold2_
EXP=outdoor_siren_v15b_causal_ft
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  export CAUSAL_CKPT=$ck
  export CAUSAL_OUT=infer_v15bft_${e/epoch_/e}_v15causal
  .venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py
done
guard_verify || exit 1
echo V15B_XINF_DONE
