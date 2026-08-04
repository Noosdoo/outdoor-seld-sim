#!/bin/bash
#SBATCH -J e1finf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o e1finf-%j.out
# E1-full: 新モデルで plain-val と heavy-val の両方を推論 + 連結
set -ex
cd ~/research/PSELDNet/PSELDNets
LOGROOT=$HOME/PSELDNets_logs/outdoor_siren_v11/runs
RUN=$HOME/PSELDNets_logs/outdoor_siren_v12heavy/runs/outdoor_siren_v12heavy_sde_run1
ck=$(ls $RUN/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"

for ev in plain heavy; do
  if [ "$ev" = "plain" ]; then exp=outdoor_siren_v11_valinfer; else exp=outdoor_siren_v11_valinfer_e1aheavy; fi
  .venv/bin/python _infer_sde.py \
    experiment=$exp \
    mode=test \
    ckpt_path=$ck \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_v12heavy_run1_val_${ev} \
    paths.log_dir=$HOME/PSELDNets_logs

  python3 - "$LOGROOT/infer_v12heavy_run1_val_${ev}/submissions" <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "val_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(f"{clip},{line}\n")
print("wrote", out)
EOF
done
echo E1F_INFER_DONE
