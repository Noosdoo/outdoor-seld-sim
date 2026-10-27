#!/bin/bash
#SBATCH -J v13cross
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o v13cross-%j.out
# 交差推論: **旧モデル（v12 w3 / ft2）を v13 fold2 val に当てる**（新データの難しさを旧モデルで測る）。
#   w3  ep099 → 通常推論 → infer_v12w3_v13val/val_all.csv
#   ft2 ep099 → 因果推論 → infer_ft2_e099_v13causal/val_all_causal.csv
# 読むだけ（v13ルート・基準ckpt）。書くのは新しい experiment_name のみ。guardで基準無変更を保証。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v13
HDF5DIR=$PS/_hdf5_v13
[ -d "$ROOTDIR" ] || { echo "datasets_v13 が無い"; exit 1; }
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export SDE_INIT_CKPT=""

W3=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
FT2=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_causal_ft2/checkpoints/epoch_099.ckpt
ls -l $W3 $FT2

.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$W3 model.kwargs.pretrained_path=null \
  experiment_name=infer_v12w3_v13val \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12w3_v13val/submissions <<'EOF'
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

export PS
export CAUSAL_CKPT=$FT2
export CAUSAL_DSDIR=$ROOTDIR
export CAUSAL_PREFIX=fold2_
export CAUSAL_OUT=infer_ft2_e099_v13causal
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py

guard_verify || exit 1
echo V13_CROSS_DONE
