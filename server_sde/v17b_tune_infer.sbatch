#!/bin/bash
#SBATCH -J v17btune
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o v17btune-%j.out
# 距離の補正表の通し稽古（v2 束 §6.1 判断 1・本人「作る」2026-09-07 07:05）: v17b 因果 ft e139 を
#   fold32（v43tune・作成用 1,800 本）と fold30（v42tune・確認用 1,800 本）で因果推論（高さ入力は表なし・1.5 m）。
# 出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_v17b_e139_fold32 / _fold30 の val_all_causal.csv
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_causal_infer_v17.py $PS/_causal_infer_v17.py
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export CAUSAL_DSDIR=$PS/datasets_v12
export HEIGHT_COND=1
export HEIGHT_TABLE=""
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
export CAUSAL_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v17b_causal_ft/checkpoints/epoch_139.ckpt
ls -l $CAUSAL_CKPT
for T in "fold32 v43tune" "fold30 v42tune"; do
  set -- $T
  export CAUSAL_FOA=$E2E/out/dataset_outdoor_siren_$2/foa
  export CAUSAL_PREFIX=${1}_
  export CAUSAL_OUT=infer_v17b_e139_$1
  unset CAUSAL_GLOB
  ls $CAUSAL_FOA | wc -l
  .venv/bin/python $PS/_causal_infer_v17.py
done
guard_verify || exit 1
echo V17B_TUNE_DONE
