#!/bin/bash
#SBATCH -J v16train
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v16train-%j.out
# v16 基準学習（データ 2 倍＝各場面を 2 高さ・14,400 本。QoS normal＝割り込まれない）: **w3 と同一レシピ**（init=v12_init_from_run3ep84 / W_DIST=3 / log / batch8 / lr3e-4 /
# StepLR60 / 100ep）でデータだけ v16 に替える → best ckpt で val 推論を2つ:
#   (a) v16 fold2（自分の世界・雨20%・新文法）   → infer_v16_w3_selfval
#   (b) v12 fold2（基準の世界・旧文法・雨なし）   → infer_v16_w3_v12val  ← データ変更の効果はこちらで測る
# requeue 時は last.ckpt から再開。基準には触れない（guard）。
set -ex
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
ROOTDIR=$PS/datasets_v16
HDF5DIR=$PS/_hdf5_v16
EXP=outdoor_siren_v16_sde_w3
[ -d "$ROOTDIR" ] || { echo "datasets_v16 が無い（先に v16_gen.sbatch）"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
cd $PS
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT \
  $RESUME

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
echo "[paths] selfval: dataset_dir=$ROOTDIR hdf5_dir=$HDF5DIR"
echo "[paths] v12val : dataset_dir=$PS/datasets_v12 hdf5_dir=$PS/_hdf5 (read-only)"

for pair in "selfval:$ROOTDIR:$HDF5DIR" "v12val:$PS/datasets_v12:$PS/_hdf5"; do
  IFS=: read tag dsd h5d <<< "$pair"
  .venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
    ckpt_path=$ck \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_v16_w3_${tag} \
    paths.log_dir=$HOME/PSELDNets_logs \
    paths.dataset_dir=$dsd paths.hdf5_dir=$h5d
  python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v16_w3_${tag}/submissions <<'EOF'
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
done

guard_verify || exit 1
echo V15_TRAIN_DONE
