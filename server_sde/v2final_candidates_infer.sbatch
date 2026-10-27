#!/bin/bash
#SBATCH -J v2fcand
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v2fcand-%j.out
# 因果 ft の学習が終わっている run の候補 8 個（epoch_019〜139 ＋ last = e149）に、選択器 2.1 が使う推論（val 12,420 本・_selfcausal）だけを回す（2026-10-22 Fable）。
#   きっかけ: job 7972（主 seed・150 epoch 完走）が v2final_causal.sbatch の古い推論の段（旧 v12 val・8 クラス）で落ち、_selfcausal が出なかった。学習はやり直さない。
#   条件は v2final_causal.sbatch の推論の段と同じ（同じ env・同じ _causal_infer_v17.py・名前の決まり infer_v2finalft<_seed><_abl_tag>_e<NNN>_selfcausal〔RN20-02〕）。
#   既に完了印（val_all_causal.done.json）がある候補は飛ばす（再開できる・割り込まれても続きから）。
# 使い方: [V2FINAL_SEED_TAG=s1|s2] [V2FINAL_ABL_TAG=scratch|noheight] [V2FINAL_DATA=v29] sbatch server_sde/v2final_candidates_infer.sbatch
#   CAND_DRYRUN=1 = ckpt の確認まで（推論しない）
# 2026-10-26 本体 §3-5（Fable 16:5x「norm は候補 8 個の推論のときに取る」）: V2FINAL_DATA=v30 のときだけ、DUMP_NORMS=1（norms と norms_manifest.json・選んだ候補の run を
#   そのまま scripts/v30_select_thresholds.py の --run に渡す）・HYST_START_FRAME=40（明示）・入力の倍率 SDE_INPUT_GAIN_DB の明示が必須（決めた倍率・0 でも渡す）。
#   v29 では 3 つとも外す（旧の run と同じ identity）。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
DATA=${V2FINAL_DATA:-v29}
ROOTDIR=$PS/datasets_$DATA
EXP=outdoor_siren_v2final_${DATA}_causal_ft
SEED_TAG=${V2FINAL_SEED_TAG:-}; SEED_SUF=""; SEED_CAUSAL=0
case "$SEED_TAG" in
  "") ;;
  s1) SEED_SUF=_s1; SEED_CAUSAL=1 ;;
  s2) SEED_SUF=_s2; SEED_CAUSAL=2 ;;
  *) echo "V2FINAL_SEED_TAG は s1 か s2（$SEED_TAG）"; exit 3 ;;
esac
EXP=${EXP}${SEED_SUF}
ABL_TAG=${V2FINAL_ABL_TAG:-}; ABL_SUF=""; ABL_HEIGHT=1
case "$ABL_TAG" in
  "") ;;
  scratch) ABL_SUF=_abl_scratch ;;
  noheight) ABL_SUF=_abl_noheight; ABL_HEIGHT=0 ;;
  *) echo "V2FINAL_ABL_TAG は scratch か noheight（$ABL_TAG）"; exit 3 ;;
esac
EXP=${EXP}${ABL_SUF}
PLAN_DS=$E2E/out/dataset_outdoor_siren_v2final/plan               # v29（旧の道）の plan
if [ "$DATA" = v30 ]; then
  [ -n "${SDE_INPUT_GAIN_DB:-}" ] || { echo "v30 は入力の倍率 SDE_INPUT_GAIN_DB を明示する（決めた倍率・0 でも）"; exit 3; }
  export SDE_INPUT_GAIN_DB DUMP_NORMS=1 HYST_START_FRAME=40
  # Fable 17:0x: v30 は v30 の置き場の plan を見る。描き直しは同じ plan を写して始めるので v29 の plan と sha256 が同じはず（違えば描き直しの事故 → 止める）
  PLAN30=$E2E/out/${V2FINAL_DS_NAME:-dataset_outdoor_siren_v2final_v30}/plan
  for f in assignment_v2final.csv manifest_val_syn_v2final.csv; do
    [ -f "$PLAN30/$f" ] || { echo "v30 の plan が無い: $PLAN30/$f"; exit 3; }
    [ "$(sha256sum < "$PLAN30/$f" | cut -d' ' -f1)" = "$(sha256sum < "$PLAN_DS/$f" | cut -d' ' -f1)" ] || { echo "v30 の plan $f が v29 の plan と違う（描き直しの事故？）→ 止める"; exit 3; }
  done
  PLAN_DS=$PLAN30
else
  unset SDE_INPUT_GAIN_DB DUMP_NORMS HYST_START_FRAME
fi
PARAMS=$E2E/md/design/v2final_params.json
$PS/.venv/bin/python -c "import json,sys;P=json.load(open('$PARAMS'));sys.exit(0 if P.get('status')=='fixed' else 3)" || { echo "params が fixed でない"; exit 3; }
[ -d "$ROOTDIR/outdoor_siren_v12" ] || { echo "学習データの置き場が無い（$ROOTDIR）"; exit 3; }
CKD=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints
[ -d "$CKD" ] || { echo "run の ckpt の置き場が無い（$CKD）。因果 ft が終わってから"; exit 3; }
# 候補 8 個: epoch_019〜139 の 7 個 ＋ epoch_149（無ければ last.ckpt を epoch 149 と確かめて使う・Astra A6）
CKS=$(ls $CKD/epoch_*.ckpt | grep -v -- "-v"); [ -f "$CKD/epoch_149.ckpt" ] || CKS="$CKS $CKD/last.ckpt"
NCK=$(echo $CKS | wc -w); [ "$NCK" = 8 ] || { echo "候補が 8 個でない（$NCK 個）: $CKS"; exit 3; }
for ck in $CKS; do
  if [ "$(basename $ck)" = "last.ckpt" ]; then
    CKE=$(.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$ck")
    [ "$CKE" = "149" ] || { echo "last.ckpt の epoch が 149 でない（$CKE）→ e149 として使えない。止める"; exit 3; }
  fi
done
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
# 推論の import 先はジョブ専用 _job<ID>/（監査 A3/R1 の決まり・job 8965 の教訓: $PS 直下の _prediction_integrity.py は 9/9 の古い写しで落ちる）
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
sha256sum $CKS $PS/height_patch.py $PS/causal_patch.py $JOBDIR/_causal_infer_v17.py $JOBDIR/_prediction_integrity.py $JOBDIR/hysteresis_patch.py $PARAMS | tee $E2E/v2final_candidates_sha256_job${SLURM_JOB_ID:-dry}.txt
if [ "$DATA" = v30 ]; then                       # §3-2・§3-5: 倍率の入口もジョブの置き場に（sha を追記）
  cp $E2E/server_sde/gain_patch.py $JOBDIR/gain_patch.py
  sha256sum $JOBDIR/gain_patch.py | tee -a $E2E/v2final_candidates_sha256_job${SLURM_JOB_ID:-dry}.txt
fi
# 推論の条件 = v2final_causal.sbatch の推論の段と同じ
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=$SEED_CAUSAL   # #71: 主 seed は 0・s1 は 1・s2 は 2
export HEIGHT_COND=$ABL_HEIGHT    # #74: noheight は 0
export HEIGHT_JITTER=0
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
export CAUSAL_PREFIX=val_syn_
export CAUSAL_LIMIT=0
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
unset CAUSAL_GLOB CAUSAL_FOA CAUSAL_MANIFEST
OUTPFX=infer_v2finalft${SEED_SUF}${ABL_SUF}
N_VAL=$(($(wc -l < $PLAN_DS/manifest_val_syn_v2final.csv) - 1))   # 選択器の名簿と同じ本数（v29 = 12,420・v30 は v30 の置き場の plan〔v29 と sha が同じこと確認済み〕）
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
echo "EXP=$EXP OUTPFX=$OUTPFX N_VAL=$N_VAL HEIGHT_COND=$HEIGHT_COND SEED=$CAUSAL_SEED 候補=$NCK DATA=$DATA GAIN=${SDE_INPUT_GAIN_DB:-未設定} DUMP_NORMS=${DUMP_NORMS:-0} HYST_START=${HYST_START_FRAME:-未設定}"
[ "${CAND_DRYRUN:-0}" = 1 ] && { echo "V2FINAL_CANDIDATES_DRYRUN_OK"; exit 0; }
for ck in $CKS; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  OUT=${OUTPFX}_${e/epoch_/e}_selfcausal
  if [ -f "$RUNS/$OUT/val_all_causal.done.json" ]; then echo "済み: $OUT（完了印あり・飛ばす）"; continue; fi
  export CAUSAL_CKPT=$ck
  export HEIGHT_TABLE=$PLAN_DS/assignment_v2final.csv
  [ "$ABL_HEIGHT" = 1 ] || export HEIGHT_TABLE=""
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=$OUT
  export CAUSAL_EXPECTED_CLIPS=$N_VAL
  .venv/bin/python $JOBDIR/_causal_infer_v17.py
  [ -f "$RUNS/$OUT/val_all_causal.done.json" ] || { echo "$OUT の完了印が無い"; exit 5; }
  if [ "$DATA" = v30 ]; then [ -f "$RUNS/$OUT/norms_manifest.json" ] || { echo "$OUT の norms_manifest.json が無い"; exit 5; }; fi
done
guard_verify || exit 1
rm -rf "$JOBDIR"
echo V2FINAL_CANDIDATES_DONE $OUTPFX
