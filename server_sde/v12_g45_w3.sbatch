#!/bin/bash
#SBATCH -J v12w3
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o v12w3-%j.out
# 【G4/G5僅差対応・レバー1】v12をW_DIST=3.0で再学習（他はrun1と完全同一）。
# 狙い: G4後半（重大tier再現 66.5% vs 目標68%）を距離損失の重み増で回復。
# 副作用監視: G5（val macro SELD ≤0.085、run1=0.086）が悪化しないか。
# 学習後にそのままval 1,800本を推論・連結までやる（ジョブ1本で完結）。
set -ex
PS=~/research/PSELDNet/PSELDNets
cd $PS
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=outdoor_siren_v12_sde_w3 \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_w3_val \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_val/submissions <<'EOF'
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
echo V12_W3_DONE
