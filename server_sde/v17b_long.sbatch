#!/bin/bash
#SBATCH -J v17blong
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o v17blong-%j.out
# 長尺セット v1（60 秒 × 400 本・fold40/41・マイク 1.5 m）を v17b 因果 ft（e139・採用候補）で因果推論（2026-09-07）。
# v1（ft2 e099・infer_long_v1_causal）と同じ条件で、鳴らしすぎ（安全な車への誤った中・強・同じ車への再発火）を v2 土台で測るため。
# 装着高さ入力は表なし・既定 1.5 m（HEIGHT_TABLE=""）。出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_long_v17b_e139/val_all_causal.csv
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_FOA=$E2E/out/dataset_outdoor_long_v1/foa
export CAUSAL_PREFIX=long_
export CAUSAL_BATCH=${CAUSAL_BATCH:-50}
export HEIGHT_COND=1
export HEIGHT_TABLE=""
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
for E in ${EPOCHS:-139}; do
  export CAUSAL_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v17b_causal_ft/checkpoints/epoch_${E}.ckpt
  export CAUSAL_OUT=infer_long_v17b_e${E}
  ls -l $CAUSAL_CKPT; ls $CAUSAL_FOA | wc -l
  .venv/bin/python $E2E/server_sde/_causal_infer_long17.py
done
guard_verify || exit 1
echo V17B_LONG_DONE
