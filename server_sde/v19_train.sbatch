#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v19train-%j.out
# v19 基準学習 = v17b/v18 レシピそのまま（w3・100ep・高さ入力 FiLM・ゆらぎ 0.10）。データだけ datasets_v19_$SCENE（土台 v18 ＋ 新場面）。
# 投入例: sbatch -J v19fasttrain --export=ALL,SCENE=fast --dependency=afterok:<prep job> server_sde/v19_train.sbatch
# 宣言 = md/design/v19_新4場面_事前宣言_2026-09-08.md
set -ex
SCENE=${SCENE:?SCENE=fast|static|turn|platoon}
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
PS=~/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
ROOTDIR=$PS/datasets_v19_$SCENE
HDF5DIR=$PS/_hdf5_v19_$SCENE
EXP=outdoor_siren_v19${SCENE}_sde_w3
[ -d "$ROOTDIR" ] || { echo "datasets_v19_$SCENE が無い（先に v19_prep.sbatch）"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
cd $PS
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/_train_sde17.py $PS/_train_sde17.py
sha256sum $PS/sde_patch.py $PS/height_patch.py $PS/_train_sde17.py $E2E/server_sde/causal_patch.py | tee v19_${SCENE}_train_sha256_job${SLURM_JOB_ID}.txt   # 依頼 2-4: 実行されるパッチの sha256 をログに
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export HEIGHT_COND=1
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v19_$SCENE/plan/assignment_v19_${SCENE}_all.csv
export HEIGHT_JITTER=0.10
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} scene=$SCENE"
.venv/bin/python _train_sde17.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT \
  $RESUME
ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
guard_verify || exit 1
echo V19_TRAIN_DONE_$SCENE
