#!/bin/bash
#SBATCH -J v12inf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o v12inf-%j.out
# v12モデルで v12-val 全1,800本（core1200+ext600）を距離つき推論 + 連結
set -ex
PS=~/research/PSELDNet/PSELDNets
cd $PS
RUN=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_run1
ck=$(ls $RUN/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"

.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12 mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_run1_val \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_run1_val/submissions <<'EOF'
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
print("wrote", out)
EOF
echo V12_INFER_DONE
