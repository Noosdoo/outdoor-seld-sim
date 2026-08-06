#!/bin/bash
#SBATCH -J v12train
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v12train-%j.out
# v12学習: run3同一レシピ（warm start=run3の8クラス拡張ckpt・W_DIST=2.0・log距離）
set -ex
PS=~/research/PSELDNet/PSELDNets
cd $PS

# config複製（dataset名差し替え。paths.dataset_dirはCLIで独立8クラスルートを指定）
dcfg=configs/data/outdoor_siren_v12.yaml
[ -f $dcfg ] || sed "s/outdoor_siren_v11/outdoor_siren_v12/g" configs/data/outdoor_siren_v11.yaml > $dcfg
ecfg=configs/experiment/outdoor_siren_v12.yaml
[ -f $ecfg ] || sed -e "s#/data: outdoor_siren_v11.yaml#/data: outdoor_siren_v12.yaml#" \
    -e "s/task_name: outdoor_siren_v11/task_name: outdoor_siren_v12/" \
    configs/experiment/outdoor_siren_v11.yaml > $ecfg

# 8クラスwarm-start ckpt（run3 epoch_084 → class次元6→8拡張・新クラス行ゼロ）
INIT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
[ -f $INIT ] || .venv/bin/python _make_v12_init_ckpt.py \
  $HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
  $INIT

export SDE_INIT_CKPT=$INIT
export SDE_W_DIST=2.0
.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=outdoor_siren_v12_sde_run1 \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 \
  model.kwargs.pretrained_path=$INIT
echo V12_TRAIN_DONE
