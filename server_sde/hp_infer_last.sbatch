#!/bin/bash
#SBATCH -J hplast
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o hplast-%j.out
# ④ハイパラ感度 Stage B の補助: 各arm の last.ckpt で val 1,800 を通常推論する。
# hp_train.sbatch は w3 と同じ「best-val-SELD の epoch_*.ckpt」で推論するが、
# val SELD が平坦な区間ではこの選択が距離指標を左右しうる（B-short で疑い）。
# best と last の両方を採点して「ckpt選択の効果」を分離するための追加推論。
#
#   sbatch --export=ALL,ARM=short server_sde/hp_infer_last.sbatch
set -ex
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

: "${ARM:?ARM を指定してください}"
PS=~/research/PSELDNet/PSELDNets
EXP=outdoor_siren_v12_hp_${ARM}
cd $PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export SDE_INIT_CKPT=""

ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
ls -l $ck
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_hp_${ARM}_last \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_hp_${ARM}_last/submissions <<'EOF'
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

guard_verify || exit 1
echo HP_LAST_DONE_$ARM
