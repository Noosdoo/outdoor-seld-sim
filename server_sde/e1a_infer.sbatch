#!/bin/bash
#SBATCH -J e1ainf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o e1ainf-%j.out
# E1a: SDE run3 (epoch_084) で大型車低周波デルタ版valを推論 + val_all.csv連結
set -ex
cd ~/research/PSELDNet/PSELDNets
LOGROOT=$HOME/PSELDNets_logs/outdoor_siren_v11/runs
NAME=e1aheavy

dcfg=configs/data/outdoor_siren_v11_valinfer_${NAME}.yaml
[ -f "$dcfg" ] || sed "s/outdoor_siren_v11/outdoor_siren_v11_${NAME}/g" \
    configs/data/outdoor_siren_v11_valinfer.yaml > "$dcfg"
ecfg=configs/experiment/outdoor_siren_v11_valinfer_${NAME}.yaml
[ -f "$ecfg" ] || sed "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v11_valinfer_${NAME}.yaml#" \
    configs/experiment/outdoor_siren_v11_valinfer.yaml > "$ecfg"

.venv/bin/python _infer_sde.py \
  experiment=outdoor_siren_v11_valinfer_${NAME} \
  mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v11_sde_run3_val_${NAME} \
  paths.log_dir=$HOME/PSELDNets_logs

python3 - "$LOGROOT/infer_v11_sde_run3_val_${NAME}/submissions" <<'EOF'
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
echo E1A_INFER_DONE
