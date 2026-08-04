#!/bin/bash
#SBATCH -J e0inf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o e0inf-%j.out
# E0: SDE run3 (epoch_084) で HPF3条件のval推論。素の条件は既存run3結果を再利用
set -ex
cd ~/research/PSELDNet/PSELDNets

# valinfer実験configを複製し、dataset名だけ差し替え（train_sde2/3と同じsed方式）
for fc in 50 100 200; do
  cfg=configs/experiment/outdoor_siren_v11_valinfer_hpf${fc}.yaml
  [ -f "$cfg" ] || sed "s/outdoor_siren_v11/outdoor_siren_v11_hpf${fc}/g" \
      configs/experiment/outdoor_siren_v11_valinfer.yaml > "$cfg"
  .venv/bin/python _infer_sde.py \
    experiment=outdoor_siren_v11_valinfer_hpf${fc} \
    mode=test \
    ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_v11_sde_run3_val_hpf${fc} \
    paths.log_dir=$HOME/PSELDNets_logs
done
echo E0_INFER_DONE
