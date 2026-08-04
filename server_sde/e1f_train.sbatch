#!/bin/bash
#SBATCH -J e1ftrain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o e1ftrain-%j.out
# E1-full: run3と同一レシピで v12heavy を学習（差分はデータのみ）
set -ex
cd ~/research/PSELDNet/PSELDNets

# config複製（dataset名のみ差し替え）
dcfg=configs/data/outdoor_siren_v12heavy.yaml
[ -f "$dcfg" ] || sed "s/outdoor_siren_v11/outdoor_siren_v12heavy/g" \
    configs/data/outdoor_siren_v11.yaml > "$dcfg"
ecfg=configs/experiment/outdoor_siren_v12heavy.yaml
[ -f "$ecfg" ] || sed "s#/data: outdoor_siren_v11.yaml#/data: outdoor_siren_v12heavy.yaml#" \
    configs/experiment/outdoor_siren_v11.yaml > "$ecfg"

export SDE_INIT_CKPT=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
export SDE_W_DIST=2.0
.venv/bin/python _train_sde.py \
  experiment=outdoor_siren_v12heavy \
  experiment_name=outdoor_siren_v12heavy_sde_run1 \
  paths.log_dir=$HOME/PSELDNets_logs \
  model.kwargs.pretrained_path=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
echo E1F_TRAIN_DONE
