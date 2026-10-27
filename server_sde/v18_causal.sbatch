#!/bin/bash
#SBATCH -J v18causal
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v18causal-%j.out
# v18 因果 ft = v17b と同一（起点 = v18 基準 ep099 / KMIN40 / REPEAT2 / lr1e-4・150ep・StepLR90 / 20ep 毎に全保存 / 高さ入力・ゆらぎ 0.10）。
# 保存 ckpt 全部を因果推論: v12 fold2（1.5 m・表なし）→ infer_v18ft_e<NNN>_v12causal、v18(=v16) fold2 → infer_v18ft_e<NNN>_selfcausal
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v18
HDF5DIR=$PS/_hdf5_v18
EXP=outdoor_siren_v18_causal_ft
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v18_sde_w3/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v18_sde_w3/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export HEIGHT_COND=1
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v18/plan/assignment_v18.csv
export HEIGHT_JITTER=0.10
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py clr1e-4e150
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR     # ジョブ専用ディレクトリ（4037 の混入の再発防止・監査 A7: import 先まで専用化）
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
INFER=$JOBDIR/_causal_infer_v17.py
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
set -o pipefail
# 監査 F5（2026-09-08 修正・ジョブ 4037 は修正前の写しで実行中）: epoch 名は ckpt の中身から読む。last = epoch_149 の重複は飛ばす
ck_epoch() { .venv/bin/python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']))" "$1"; }
done_epochs=""
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  ep=$(ck_epoch $ck); e=$(printf 'e%03d' $ep)
  case " $done_epochs " in *" $e "*) echo "skip $ck (epoch $ep は済み)"; continue;; esac
  done_epochs="$done_epochs $e"
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_v18ft_${e}_v12causal
  unset CAUSAL_GLOB
  export HEIGHT_TABLE=""
  .venv/bin/python $INFER
  export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v18/plan/assignment_v18.csv
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v18ft_${e}_selfcausal
  .venv/bin/python $INFER
done
for e in e099 e119 e139 e149; do
  for s in v12causal selfcausal; do
    [ -f $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v18ft_${e}_${s}/val_all_causal.done.json ] || { echo "推論が欠けている: ${e}_${s}"; exit 5; }
  done
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo V18_CAUSAL_DONE
