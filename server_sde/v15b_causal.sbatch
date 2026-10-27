#!/bin/bash
#SBATCH -J v15bcausal
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v15bcausal-%j.out
# v15b 因果ft: **ft2 と同一レシピ**（起点=v15b w3 ep099 / KMIN40 / REPEAT2 / lr3e-4 StepLR60 / 100ep /
# 20ep毎に全保存）でデータだけ v15b。保存ckpt全部を因果推論:
#   v12 fold2（旧文法・雨なし）→ infer_v15bft_e<NNN>_v12causal  ← ft2 と直接比較
#   v15b fold2（新文法・雨20%）→ infer_v15bft_e<NNN>_selfcausal ← plan列で層別
# requeue 時は last.ckpt から再開。基準には触れない（guard）。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

ROOTDIR=$PS/datasets_v15b
HDF5DIR=$PS/_hdf5_v15b
EXP=outdoor_siren_v15b_causal_ft
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v15b_sde_w3/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v15b_sde_w3/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python ~/research/outdoor_seld_e2e/server_sde/hp_make_configs.py clr1e-4e150
grep -n "lr\|max_epochs\|step_size" $CFG

cp $HOME/research/outdoor_seld_e2e/server_sde/causal_patch.py $PS/causal_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_causal12.py $PS/_train_causal12.py

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python $PS/_train_causal12.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $RESUME

export PS
export CAUSAL_PREFIX=fold2_
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  export CAUSAL_CKPT=$ck
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_v15bft_${e/epoch_/e}_v12causal
  .venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v15bft_${e/epoch_/e}_selfcausal
  .venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py
done

guard_verify || exit 1
echo V15B_CAUSAL_DONE
