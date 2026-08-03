#!/bin/bash
#SBATCH -J sdeinfval
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o sdeinfval-%j.out
# SDE run1 (epoch_094) で val 1,200本を距離付き推論
set -x
cd ~/research/PSELDNet/PSELDNets
.venv/bin/python _infer_sde.py \
  experiment=outdoor_siren_v11_valinfer \
  mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run1/checkpoints/epoch_094.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v11_sde_run1_val \
  paths.log_dir=$HOME/PSELDNets_logs
