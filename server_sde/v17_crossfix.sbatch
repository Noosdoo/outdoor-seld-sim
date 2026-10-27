#!/bin/bash
#SBATCH -J v17xfix
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o v17xfix-%j.out
# v17 の交差 val（v12 fold2・1.5 m）推論のやり直し（2026-09-05 22:20）。
# 不具合: v12 val の clip 名（fold2_room1_mixNNNN）が v16 plan と同じなので、HEIGHT_TABLE から v16 の高さ（1.4〜2.1 m）が誤って引かれていた。
# → 表を外して全クリップ 1.5 m で推論し直す（出力名 *_v12causal_fix）。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
cp $HOME/research/outdoor_seld_e2e/server_sde/height_patch.py $PS/height_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v17.py $PS/_causal_infer_v17.py
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export HEIGHT_COND=1
export HEIGHT_TABLE=""
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_PREFIX=fold2_
EXP=outdoor_siren_v17_causal_ft
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  export CAUSAL_CKPT=$ck
  export CAUSAL_OUT=infer_v17ft_${e/epoch_/e}_v12causal_fix
  .venv/bin/python $PS/_causal_infer_v17.py
done
guard_verify || exit 1
echo V17_CROSSFIX_DONE
