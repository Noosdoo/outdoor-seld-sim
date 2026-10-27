#!/bin/bash
#SBATCH -J v17bcausal
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v17bcausal-%j.out
# v17b 因果ft（ゆらぎ 0.10 m）= v16 と同一（起点=v17 w3 ep099 / KMIN40 / REPEAT2 / v2 案 lr1e-4・150ep・StepLR90 / 20ep毎に全保存）＋ 装着高さの入力。
# 保存 ckpt 全部を因果推論:
#   v12 fold2（1.5 m・表に無いので既定 1.5 m）→ infer_v17bft_e<NNN>_v12causal
#   v16 fold2（3,600 本。mix≤9000 が v15 val）    → infer_v17bft_e<NNN>_selfcausal
# 感度評価（宣言 §3）: e099/e119/e139/e149 で高さ入力を ±0.20 m ずらし、v15 val（fold2 の 4 桁 mix＝1,800 本）だけ推論
#   → infer_v17bft_e<NNN>_selfcausal_hp20 / _hm20
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

ROOTDIR=$PS/datasets_v16
HDF5DIR=$PS/_hdf5_v16
EXP=outdoor_siren_v17b_causal_ft
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v17b_sde_w3/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v17b_sde_w3/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export HEIGHT_COND=1
export HEIGHT_TABLE=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
export HEIGHT_JITTER=0.10
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python ~/research/outdoor_seld_e2e/server_sde/hp_make_configs.py clr1e-4e150
grep -n "lr\|max_epochs\|step_size" $CFG

cp $HOME/research/outdoor_seld_e2e/server_sde/height_patch.py $PS/height_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/causal_patch.py $PS/causal_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_causal17.py $PS/_train_causal17.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v17.py $PS/_causal_infer_v17.py

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $RESUME

export PS
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=fold2_
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_v17bft_${e/epoch_/e}_v12causal
  unset CAUSAL_GLOB
  export HEIGHT_TABLE=""      # v12 val は clip 名が v16 と同じなので表を外す（全クリップ 1.5 m）。v17 では表が誤って引かれていた
  .venv/bin/python $PS/_causal_infer_v17.py
  export HEIGHT_TABLE=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v17bft_${e/epoch_/e}_selfcausal
  .venv/bin/python $PS/_causal_infer_v17.py
done

# 感度評価: ±0.20 m（v15 val 1,800 本 = fold2 の 4 桁 mix）
export CAUSAL_DSDIR=$ROOTDIR
export CAUSAL_GLOB="fold2_room1_mix[01][0-9][0-9][0-9].flac"
for e in 099 119 139 149; do
  ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_${e}.ckpt
  [ "$e" = "149" ] && [ ! -f "$ck" ] && ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
  [ -f "$ck" ] || continue
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0.20;  export CAUSAL_OUT=infer_v17bft_e${e}_selfcausal_hp20; .venv/bin/python $PS/_causal_infer_v17.py
  export HEIGHT_OFFSET=-0.20; export CAUSAL_OUT=infer_v17bft_e${e}_selfcausal_hm20; .venv/bin/python $PS/_causal_infer_v17.py
  export HEIGHT_OFFSET=0.10;  export CAUSAL_OUT=infer_v17bft_e${e}_selfcausal_hp10; .venv/bin/python $PS/_causal_infer_v17.py
  export HEIGHT_OFFSET=-0.10; export CAUSAL_OUT=infer_v17bft_e${e}_selfcausal_hm10; .venv/bin/python $PS/_causal_infer_v17.py
done
export HEIGHT_OFFSET=0

guard_verify || exit 1
echo V17B_CAUSAL_DONE
