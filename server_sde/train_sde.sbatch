#!/bin/bash
#SBATCH -J v11sde1
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v11sde1-%j.out
# 役割③: SDE(距離推定)付きファインチューニング run1
# ウォームスタート: run2 ep094の全重み(ヘッドはxyzコピー+距離行0)
set -x
cd ~/research/PSELDNet/PSELDNets
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
export SDE_W_DIST=5.0
.venv/bin/python _train_sde.py \
  experiment=outdoor_siren_v11 \
  experiment_name=outdoor_siren_v11_sde_run1 \
  paths.log_dir=$HOME/PSELDNets_logs \
  model.kwargs.pretrained_path=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
