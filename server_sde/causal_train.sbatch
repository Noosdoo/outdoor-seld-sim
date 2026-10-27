#!/bin/bash
#SBATCH -J v12causaltr
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH -o v12causaltr-%j.out
# v12(w3)を「因果窓」で微調整する。学習時から未来を見せない。
#
# 書き込みは新しい experiment_name のディレクトリのみ。
# 基準run(outdoor_siren_v12_sde_w3)・基準キャッシュ・基準データセットには一切書かない。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS

source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
EXP=${EXP:-outdoor_siren_v12_causal_ft}
export SDE_W_DIST=3.0          # w3と同一条件で微調整する
export SDE_DIST_MODE=log
export CAUSAL_KMIN=${CAUSAL_KMIN:-40}
export CAUSAL_REPEAT=${CAUSAL_REPEAT:-8}
ls -l $CK

# ラッパは src/ を相対で見るので PSELDNets 直下に置いて実行する
cp $HOME/research/outdoor_seld_e2e/server_sde/causal_patch.py $PS/causal_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_causal12.py $PS/_train_causal12.py

.venv/bin/python $PS/_train_causal12.py \
  experiment=outdoor_siren_v12 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 \
  ${EXTRA:-}

guard_verify
echo CAUSAL_TRAIN_DONE
