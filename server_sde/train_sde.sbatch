#!/bin/bash
#SBATCH -J v11sde1
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v11sde1-%j.out
# 役割③: SDE(距離推定)付きファインチューニング run1
# ウォームスタート: run2 ep094の全重み(ヘッドはxyzコピー+距離行0)
# 【注意・第8回監査2026-08-06】これはrun1(W=5.0・線形距離・tanh天井時代)の起動記録。
# HEADのsde_patch既定はlog/W=2.0(run3仕様)のため、今このままsbatchを叩くと
# W=5.0×logという「どのrunとも違う混成構成」になる。run3の正=train_sde_run3.sbatch。
set -x
cd ~/research/PSELDNet/PSELDNets
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
export SDE_W_DIST=5.0
.venv/bin/python _train_sde.py \
  experiment=outdoor_siren_v11 \
  experiment_name=outdoor_siren_v11_sde_run1 \
  paths.log_dir=$HOME/PSELDNets_logs \
  model.kwargs.pretrained_path=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
