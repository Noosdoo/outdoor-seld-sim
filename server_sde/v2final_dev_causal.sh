#!/bin/bash
# 【開発用】確定評価 v2 の因果 ft ＋ 保存 ckpt 全部の推論を、仮の数字の版（PROVISIONAL）で先に通す（2026-09-15 夜）。
# 中身は v2final_causal.sbatch（本番）と同じ（起点 = 開発用の基準学習 ep099 / KMIN40 / REPEAT2 / lr1e-4・150ep / 20ep 毎に全保存 / 高さ入力）。
# 違い: params の fixed 検査なし（PROVISIONAL.txt を要求）・実験名と推論名に dev を入れる（infer_v2finaldevft_e<NNN>_{v12causal,selfcausal}）・
#       推論の入口と来歴モジュールはジョブ専用ディレクトリに写す（v20 と同じ）。fold20（確定評価）は推論しない。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
DATA=${V2FINAL_DATA:-v2final}                     # 2026-10-12: 学習データの置き場の札（v2final＝9/22 の束・v22＝10/8 の 9 クラス・v26＝音の直しの描き直し）。既定 v2final
ROOTDIR=$PS/datasets_$DATA
# 2026-10-01 (c): DEV_HDF5DIR・DEV_SEED・DEV_CAUSAL_SEED・DEV_HEIGHT_SEED で対照（noD21）と種を変える。指定しなければ従来と同じ
HDF5DIR=${DEV_HDF5DIR:-$PS/_hdf5_$DATA}
MARK=$ROOTDIR/outdoor_siren_v12/PROVISIONAL.txt           # 束に付く仮の版の印（AR12-05: 本番は印があれば止める・開発用は無ければ止める）
SEED_ARG=""; [ -n "${DEV_SEED:-}" ] && SEED_ARG="seed=$DEV_SEED"
[ -n "${DEV_HEIGHT_SEED:-}" ] && export HEIGHT_SEED=$DEV_HEIGHT_SEED
[ -d "$HDF5DIR" ] || { echo "HDF5 ルートが無い: $HDF5DIR"; exit 1; }
[ "$DATA" = v2final ] || [ -n "${DEV_TAG+x}" ] || { echo "V2FINAL_DATA=$DATA なのに DEV_TAG が無い（AR12-06・SV26-04: 既定値を入れる前に検査）"; exit 3; }
DEV_TAG=${DEV_TAG:-dev}                               # 2026-09-23: DEV_TAG で 2 回目以降を別名に（dev = 9/16 の一周）
TAGD=""; [ "$DATA" = v2final ] || TAGD="${DATA}_"     # SV26-04: 実験名・起点に学習データの札（v2final の既存の名前はそのまま）
EXP=outdoor_siren_v2final_${TAGD}${DEV_TAG}_causal_ft
BASE=outdoor_siren_v2final_${TAGD}${DEV_TAG}_sde_w3
PARAMS=$E2E/md/design/v2final_params.json
[ -f "$MARK" ] || { echo "PROVISIONAL.txt が無い — 開発用台本は仮の版にだけ使う"; exit 3; }
$PS/.venv/bin/python $E2E/server_sde/_train_entrance_guard.py "$ROOTDIR" "$HDF5DIR" || { echo "学習の入口の門で止めた（確定評価の本・実録の置き場が学習データに混じっている）"; exit 2; }   # AR12-03・SV26-03: ROOTDIR／HDF5DIR を決めたあとに呼ぶ
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=${DEV_CAUSAL_SEED:-0}
export HEIGHT_COND=1
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export HEIGHT_JITTER=$JIT
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py clr1e-4e150
grep -n "lr\|max_epochs\|step_size" $CFG
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
INFER=$JOBDIR/_causal_infer_v17.py
# 検出のヒステリシス（変更票 #34・2026-09-16 本人「①採用」）: 入り 0.7／保持 0.3。両設定で共通（scripts/_notify_modes.py と同じ値）
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
sha256sum $CK $PS/height_patch.py $PS/causal_patch.py $PS/_train_causal17.py $INFER $JOBDIR/_prediction_integrity.py $CFG $PARAMS $MARK | tee $E2E/v2final_dev_causal_sha256_job${SLURM_JOB_ID}.txt

CKDIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints
LAST=$CKDIR/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} exp=$EXP hdf5=$HDF5DIR seed=${DEV_SEED:-default} causal_seed=$CAUSAL_SEED height_seed=${HEIGHT_SEED:-default}"

if [ ! -f "$CKDIR/epoch_149.ckpt" ]; then
.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $SEED_ARG $RESUME
fi

export PS
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=val_syn_          # 2026-09-22 改名（旧 fold2_）
# Astra A6（2026-09-16）: epoch_149.ckpt と last.ckpt を同じ出力名に入れない。last.ckpt は epoch_149.ckpt が無いときだけ e149 として使う
CKS=$(ls $CKDIR/epoch_*.ckpt | grep -v -- "-v"); [ -f "$CKDIR/epoch_149.ckpt" ] || CKS="$CKS $CKDIR/last.ckpt"
for ck in $CKS; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_v2final${DEV_TAG}ft_${e/epoch_/e}_v12causal
  unset CAUSAL_GLOB
  export HEIGHT_TABLE=""      # v12 val は全クリップ 1.5 m（表を外す。v17 の教訓）
  .venv/bin/python $INFER
  export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v2final${DEV_TAG}ft_${e/epoch_/e}_selfcausal
  .venv/bin/python $INFER
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo V2FINAL_DEV_CAUSAL_DONE
