#!/bin/bash
#SBATCH -J v12cont
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v12cont-%j.out
# 【G4/G5僅差対応・レバー2】v12 run1のbest ckptからW_DIST=2.0のまま追加学習。
# 狙い: G5（val macro SELD 0.086 vs 目標0.085）を学習量で回復
# （v11でrun3が0.075→0.070に改善したのと同じ「warm start+追加100ep」処方）。
# 学習後にそのままval 1,800本を推論・連結までやる。
set -ex
PS=~/research/PSELDNet/PSELDNets
cd $PS
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_run1/checkpoints/epoch_094.ckpt
export SDE_W_DIST=2.0
export SDE_DIST_MODE=log
.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=outdoor_siren_v12_sde_cont \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_cont/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_cont_val \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_cont_val/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "val_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out, sum(1 for _ in open(out)))
EOF
echo V12_CONT_DONE
