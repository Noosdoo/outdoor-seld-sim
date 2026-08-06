#!/bin/bash
#SBATCH -J v11sdew0
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v11sdew0-%j.out
# 【第8回監査・問1対応】W_DIST=0の対照run＋追加テストT4-T6。
# 「距離勾配が共有エンコーダのSELD性能を劣化させない」ことの唯一の直接証拠:
# run3と同一のwarm start・スケジュール・データで、距離損失の重みだけ0にする。
# 比較: この対照runのbest val SELD vs run3の0.070（差が小さければ干渉は無害）。
set -ex
cd ~/research/PSELDNet/PSELDNets

# ① 追加テストT4-T6（数分・失敗したら学習は走らせない）
cp ~/research/outdoor_seld_e2e/server_sde/test_sde_units2.py .
.venv/bin/python test_sde_units2.py

# ② 対照run（W_DIST=0・他はrun3と完全同一）
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
export SDE_W_DIST=0.0
export SDE_DIST_MODE=log
.venv/bin/python _train_sde.py \
  experiment=outdoor_siren_v11 \
  experiment_name=outdoor_siren_v11_sde_w0 \
  paths.log_dir=$HOME/PSELDNets_logs \
  model.kwargs.pretrained_path=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
echo W0_CONTROL_DONE
