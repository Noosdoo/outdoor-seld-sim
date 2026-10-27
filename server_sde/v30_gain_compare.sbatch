#!/bin/bash
#SBATCH -J v30gain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v30gain-%j.out
# 入力の倍率の小さな比べ（本体 §3-2・§9-1 の 4・§9-4 の 2・変更票 #75 ②・2026-10-26）。1 ジョブ = 1 つの倍率 × 1 つの種。
#   1. 基準学習 20 epoch（v2final_train.sbatch・データ v30・SDE_DIST_MASK=1・SDE_INPUT_GAIN_DB=$V30_GAIN・実験名 _gain<g>・trainer.max_epochs=20）
#   2. epoch_019 の因果推論を v30 の val 全部に（入り 0.5・ヒステリシス無し〔保持 = 入り〕・HYST_START_FRAME=0・高さの表あり・出力 v30gain_g<g><種>_e019_val）
#   決め方は scripts/v30_gain_decide.py（F = フレーム 40〜99 のクラス平均 F・g1／g2・2 seed・本人の裁量なし）。結果は out/v30_gain_compare/（本文には載せない）。
# 使い方（発効のあと・Fable の OK のあと）:
#   seed 1: for g in 0 20 30 40; do sbatch --export=ALL,V30_GAIN=$g server_sde/v30_gain_compare.sbatch; done
#   seed 2: V2FINAL_SEED_TAG=s1 で g1・g2 だけ（decision.json の seed 1 の結果が出てから）
#   乾式: V30_DRY=1 bash server_sde/v30_gain_compare.sbatch（命令を印字するだけ）
set -euxo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
G=${V30_GAIN:?V30_GAIN（0／20／30／40）を渡す}
case "$G" in 0|20|30|40) ;; *) echo "V30_GAIN は 0／20／30／40（$G）"; exit 3 ;; esac
SEED_TAG=${V2FINAL_SEED_TAG:-}
case "$SEED_TAG" in ""|s1) ;; *) echo "V2FINAL_SEED_TAG は空（seed 1 = 主）か s1（seed 2）"; exit 3 ;; esac
SEED_SUF=${SEED_TAG:+_$SEED_TAG}
export V2FINAL_DATA=v30
export SDE_INPUT_GAIN_DB=$G
export SDE_DIST_MASK=1
export V2FINAL_EXP_SUFFIX=_gain${G}
export V2FINAL_EXTRA_OVR="trainer.max_epochs=20"
EXP=outdoor_siren_v2final_v30_sde_w3${SEED_SUF}_gain${G}
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_019.ckpt
OUT=v30gain_g${G}${SEED_SUF}_e019_val
RUN() { if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
# 1. 基準学習 20 epoch（epoch_019.ckpt があれば飛ばす＝再開のため）
if [ ! -f "$CK" ]; then
  RUN bash $E2E/server_sde/v2final_train.sbatch
fi
if [ ! -f "$CK" ] && [ "${V30_DRY:-0}" != 1 ]; then         # 保存の名前が epoch_019 でないとき: epoch が 19 の last.ckpt だけを使う（v2final_causal.sbatch の e149 と同じ守り）
  LASTC=$(dirname $CK)/last.ckpt
  CKE=$($PS/.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$LASTC" 2>/dev/null || echo none)
  [ "$CKE" = "19" ] || { echo "epoch_019.ckpt が無く、last.ckpt の epoch も 19 でない（$CKE）→ 止める"; exit 3; }
  CK=$LASTC
fi
# 2. epoch_019 の因果推論（val 全部・入り 0.5・ヒステリシス無し）
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; RUN mkdir -p $JOBDIR
for f in _causal_infer_v17.py _prediction_integrity.py hysteresis_patch.py gain_patch.py; do RUN cp $E2E/server_sde/$f $JOBDIR/$f; done
export PS HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_OFFSET=0 HEIGHT_DEFAULT=1.5
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export CAUSAL_PREFIX=val_syn_ CAUSAL_LIMIT=0 CAUSAL_SED_TH=0.5 CAUSAL_SED_STAY=0.5 HYST_START_FRAME=0
export CAUSAL_CKPT=$CK CAUSAL_DSDIR=$PS/datasets_v30 CAUSAL_OUT=$OUT
export CAUSAL_EXPECTED_CLIPS=$(($(wc -l < $E2E/out/dataset_outdoor_siren_v2final/plan/manifest_val_syn_v2final.csv) - 1))
cd $PS
RUN .venv/bin/python $JOBDIR/_causal_infer_v17.py
RUN rm -rf "$JOBDIR"
echo "V30_GAIN_COMPARE_DONE gain=$G seed=${SEED_TAG:-main} ckpt=$CK out=$OUT"
