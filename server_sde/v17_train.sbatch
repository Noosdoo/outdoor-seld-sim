#!/bin/bash
#SBATCH -J v17train
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v17train-%j.out
# v17 基準学習 = v16 と同一データ（datasets_v16・18,000 本）・同一レシピ（w3: init=v12_init_from_run3ep84 / W_DIST=3 / log /
# batch8 / lr3e-4 / StepLR60 / 100ep）＋ **装着高さの入力**（height_patch・HEIGHT_COND=1・学習時ゆらぎ 0.05 m）。
# 宣言= md/design/v17候補学習_装着高さの入力_事前宣言_2026-09-05.md。best ckpt で val 推論を 2 つ（selfval: v16 fold2 / v12val: 1.5 m）。
# requeue 時は last.ckpt から再開。基準には触れない（guard）。
set -ex
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
ROOTDIR=$PS/datasets_v16
HDF5DIR=$PS/_hdf5_v16
EXP=outdoor_siren_v17_sde_w3
[ -d "$ROOTDIR" ] || { echo "datasets_v16 が無い（先に v16_gen.sbatch）"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
cd $PS
cp $HOME/research/outdoor_seld_e2e/server_sde/height_patch.py $PS/height_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_sde17.py $PS/_train_sde17.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_infer_sde17.py $PS/_infer_sde17.py
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export HEIGHT_COND=1
export HEIGHT_TABLE=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
export HEIGHT_JITTER=0.05
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python _train_sde17.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT \
  $RESUME

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
.venv/bin/python - "$ck" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
keys = [k for k in sd if "h_mlp" in k]
print("[height] h_mlp keys in ckpt:", len(keys), "| last-layer |w| mean =", float(sd[[k for k in keys if k.endswith("2.weight")][0]].abs().mean()))
assert keys, "h_mlp が ckpt に無い"
EOF

export HEIGHT_JITTER=0            # 推論はゆらぎ無し
for pair in "selfval:$ROOTDIR:$HDF5DIR" "v12val:$PS/datasets_v12:$PS/_hdf5"; do
  IFS=: read tag dsd h5d <<< "$pair"
  .venv/bin/python _infer_sde17.py experiment=outdoor_siren_v12_valinfer mode=test \
    ckpt_path=$ck \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_v17_w3_${tag} \
    paths.log_dir=$HOME/PSELDNets_logs \
    paths.dataset_dir=$dsd paths.hdf5_dir=$h5d
  python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v17_w3_${tag}/submissions <<'EOF'
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
echo V17_TRAIN_DONE
