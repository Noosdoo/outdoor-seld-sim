#!/bin/bash
#SBATCH -J v11sde3
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v11sde3-%j.out
# 【記録用・第8回監査の指摘2対応】run3の実行構成の正本（再現可能化）。
# run3 = run2 ep094 warm start・SDE_W_DIST=2.0・SDE_DIST_MODE=log・100ep。
# 従来の train_sde.sbatch は run1（W=5.0・線形距離・tanh天井あり）の起動記録の
# まま残す（履歴保存）。run3投入時は手元編集で実行されリポジトリ未反映だった
# ため、このファイルを正として追加した（実行済みジョブの再現以外に再実行不要）。
set -x
cd ~/research/PSELDNet/PSELDNets
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
export SDE_W_DIST=2.0
export SDE_DIST_MODE=log
.venv/bin/python _train_sde.py \
  experiment=outdoor_siren_v11 \
  experiment_name=outdoor_siren_v11_sde_run3 \
  paths.log_dir=$HOME/PSELDNets_logs \
  model.kwargs.pretrained_path=$HOME/PSELDNets_logs/sde_init_from_run2ep94.ckpt
