#!/bin/bash
#SBATCH -J e0inf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o e0inf-%j.out
# E0: SDE run3 (epoch_084) で HPF3条件のval推論 + val_all.csv連結
set -ex
cd ~/research/PSELDNet/PSELDNets
LOGROOT=$HOME/PSELDNets_logs/outdoor_siren_v11/runs

for fc in 50 100 200; do
  # data yaml / experiment yaml をsedで複製（dataset名とdata参照だけ差し替え）
  dcfg=configs/data/outdoor_siren_v11_valinfer_hpf${fc}.yaml
  [ -f "$dcfg" ] || sed "s/outdoor_siren_v11/outdoor_siren_v11_hpf${fc}/g" \
      configs/data/outdoor_siren_v11_valinfer.yaml > "$dcfg"
  ecfg=configs/experiment/outdoor_siren_v11_valinfer_hpf${fc}.yaml
  [ -f "$ecfg" ] || sed "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v11_valinfer_hpf${fc}.yaml#" \
      configs/experiment/outdoor_siren_v11_valinfer.yaml > "$ecfg"

  .venv/bin/python _infer_sde.py \
    experiment=outdoor_siren_v11_valinfer_hpf${fc} \
    mode=test \
    ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_v11_sde_run3_val_hpf${fc} \
    paths.log_dir=$HOME/PSELDNets_logs

  # 連結: 各クリップCSVの行頭にクリップ名列を足して1本化（run3回収と同形式）
  python3 - "$LOGROOT/infer_v11_sde_run3_val_hpf${fc}/submissions" <<'EOF'
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
echo E0_INFER_DONE
