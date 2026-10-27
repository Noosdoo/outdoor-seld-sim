#!/bin/bash
#SBATCH -J v17smoke
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 00:40:00
#SBATCH -o v17smoke-%j.out
# v17 の煙試験（投入前の正しさ確認・宣言 §2）:
#  (1) 高さ入力 ON で v16 の因果ft ckpt（h_mlp 無し＝ゼロ初期化）を推論 → 既存の v16 予測（infer_v16ft_e139_selfcausal）と**完全一致**するはず
#      （FiLM のゼロ初期化＝無条件モデルと同一、の確認）
#  (2) 高さ入力 ON で基準学習を数バッチだけ回し、ckpt に h_mlp が保存され、最終層の重みがゼロから動くことを確認
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
cp $HOME/research/outdoor_seld_e2e/server_sde/height_patch.py $PS/height_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_sde17.py $PS/_train_sde17.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v17.py $PS/_causal_infer_v17.py
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export HEIGHT_COND=1
export HEIGHT_TABLE=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0

# (1) ゼロ初期化の同一性
export CAUSAL_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v16_causal_ft/checkpoints/epoch_139.ckpt
export CAUSAL_DSDIR=$PS/datasets_v16
export CAUSAL_PREFIX=fold2_
export CAUSAL_LIMIT=3
export CAUSAL_OUT=smoke_v17_zeroinit_v16e139
rm -rf $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT
.venv/bin/python $PS/_causal_infer_v17.py
same=0; diffn=0
for f in $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT/submissions/*.csv; do
  b=$(basename $f)
  if cmp -s $f $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v16ft_e139_selfcausal/submissions/$b; then same=$((same+1)); else diffn=$((diffn+1)); fi
done
echo "SMOKE1 identical=$same different=$diffn"
[ $diffn -eq 0 ] || { echo "SMOKE1_FAILED: ゼロ初期化で v16 と一致しない"; exit 1; }

# (1b) 高さを +0.2 ずらしても（ゼロ初期化なら）一致するはず
export HEIGHT_OFFSET=0.2; export CAUSAL_OUT=smoke_v17_zeroinit_off20
rm -rf $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT
.venv/bin/python $PS/_causal_infer_v17.py
d2=0; for f in $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT/submissions/*.csv; do b=$(basename $f); cmp -s $f $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v16ft_e139_selfcausal/submissions/$b || d2=$((d2+1)); done
echo "SMOKE1b different=$d2"; [ $d2 -eq 0 ] || { echo "SMOKE1b_FAILED"; exit 1; }
export HEIGHT_OFFSET=0

# (2) 学習の煙試験（数バッチ・別 experiment_name・ログは smoke 用）
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export HEIGHT_JITTER=0.05
EXP=smoke_v17_train
rm -rf $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP
.venv/bin/python _train_sde17.py \
  experiment=outdoor_siren_v12 experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs paths.dataset_dir=$PS/datasets_v16 paths.hdf5_dir=$PS/_hdf5_v16 \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT \
  trainer.max_epochs=1 ++trainer.limit_train_batches=30 ++trainer.limit_val_batches=10 trainer.check_val_every_n_epoch=1
ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/*.ckpt | head -1)
.venv/bin/python - "$ck" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
keys = [k for k in sd if "h_mlp" in k]
w = sd[[k for k in keys if k.endswith("2.weight")][0]]
print("SMOKE2 h_mlp keys:", len(keys), "| last-layer |w| mean =", float(w.abs().mean()))
assert keys and float(w.abs().mean()) > 0, "h_mlp が学習で動いていない"
EOF
guard_verify || exit 1
echo V17_SMOKE_DONE
