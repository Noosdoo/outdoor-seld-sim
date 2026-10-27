#!/bin/bash
# 【開発用】確定評価 v2 の基準学習を、仮の数字の版（PROVISIONAL）の学習データで先に 1 回通す（2026-09-15 夜・本人「寝る前に何か回したい」）。
# 中身は v2final_train.sbatch（本番）と同じレシピ。違いは 3 つだけ:
#   1. params が fixed でなくても走る代わりに、PROVISIONAL.txt が無ければ止まる（確定版にはこの台本を使わない）
#   2. 実験名を outdoor_siren_v2final_dev_sde_w3 にして本番（outdoor_siren_v2final_sde_w3）と混ざらないようにする
#   3. 目的 = 11 月の手順の通し試験と、束のレシピの知覚層・通知層の数字を先に見ること。この結果は確定評価に使わない
# 呼び出し: v2final_dev_all.sbatch から bash で実行（単体 sbatch では使わない）
set -ex
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
DATA=${V2FINAL_DATA:-v2final}                     # 2026-10-12: 学習データの置き場の札（v2final＝9/22 の束・v22＝10/8 の 9 クラス・v26＝音の直しの描き直し）。既定 v2final
ROOTDIR=$PS/datasets_$DATA
# 2026-10-01 (c) 種の追加・対照（本人「(b)〜(c) 今やっちゃいましょう」）: DEV_HDF5DIR（既定 _hdf5_v2final・対照は _hdf5_v2final_noD21）、
#   DEV_SEED（hydra の seed・既定は設定ファイルの 2024）、DEV_HEIGHT_SEED（height_patch の HEIGHT_SEED・既定 20260917）。指定しなければ従来と同じ
HDF5DIR=${DEV_HDF5DIR:-$PS/_hdf5_$DATA}
MARK=$ROOTDIR/outdoor_siren_v12/PROVISIONAL.txt           # 束に付く仮の版の印（AR12-05: 本番は印があれば止める・開発用は無ければ止める）
SEED_ARG=""; [ -n "${DEV_SEED:-}" ] && SEED_ARG="seed=$DEV_SEED"
[ -n "${DEV_HEIGHT_SEED:-}" ] && export HEIGHT_SEED=$DEV_HEIGHT_SEED
[ "$DATA" = v2final ] || [ -n "${DEV_TAG+x}" ] || { echo "V2FINAL_DATA=$DATA なのに DEV_TAG が無い（古い last.ckpt から再開しうる・AR12-06・SV26-04: 既定値を入れる前に検査）"; exit 3; }
DEV_TAG=${DEV_TAG:-dev}
TAGD=""; [ "$DATA" = v2final ] || TAGD="${DATA}_"     # SV26-04: 実験名に学習データの札（v2final の既存の名前はそのまま）
EXP=outdoor_siren_v2final_${TAGD}${DEV_TAG}_sde_w3          # 2026-09-23: DEV_TAG で 2 回目以降を別名に（dev = 9/16 の一周）
[ -d "$HDF5DIR" ] || { echo "HDF5 ルートが無い: $HDF5DIR"; exit 1; }
PARAMS=$E2E/md/design/v2final_params.json
[ -d "$ROOTDIR" ] || { echo "$ROOTDIR が無い（先に生成の台本 v2final_gen.sbatch／v26_gen.sbatch）"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
$PS/.venv/bin/python $E2E/server_sde/_train_entrance_guard.py "$ROOTDIR" "$HDF5DIR" || { echo "学習の入口の門で止めた（確定評価の本・実録の置き場が学習データに混じっている）"; exit 2; }   # 2026-10-12 引き継ぎ §3-8
[ -f "$MARK" ] || { echo "PROVISIONAL.txt が無い — この開発用台本は仮の版にだけ使う（確定版は v2final_train.sbatch）"; exit 3; }
echo "DEV_RUN_ON_PROVISIONAL: $(tr '\n' ' ' < $MARK)"
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
cd $PS
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/_train_sde17.py $PS/_train_sde17.py
sha256sum $PS/height_patch.py $PS/_train_sde17.py $HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt $PARAMS $MARK | tee $E2E/v2final_${DEV_TAG}_train_sha256_job${SLURM_JOB_ID}.txt
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export HEIGHT_COND=1
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export HEIGHT_JITTER=$JIT
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} jitter=$JIT exp=$EXP hdf5=$HDF5DIR seed=${DEV_SEED:-default} height_seed=${HEIGHT_SEED:-default} train_rows=$(wc -l < $HDF5DIR/data/24000fs/wav/dev/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv)"

.venv/bin/python _train_sde17.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT \
  $SEED_ARG $RESUME

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
.venv/bin/python - "$ck" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
keys = [k for k in sd if "h_mlp" in k]
assert keys, "h_mlp が ckpt に無い"
print("[height] h_mlp keys:", len(keys))
EOF
guard_verify || exit 1
echo V2FINAL_DEV_TRAIN_DONE
