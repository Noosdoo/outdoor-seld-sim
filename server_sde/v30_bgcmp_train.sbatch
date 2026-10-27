#!/bin/bash
#SBATCH -J v30bgt
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v30bgt-%j.out
# 背景の形の小さな比べ（決め直し 5・仕様 §9-7 の 3）の学習と推論。V30BG_VARIANT = off／i／ii で 1 本ずつ（v30_bgcmp_gen.sbatch で置き場ができてから）。
#   1. 基準学習 20 epoch（v2final_train.sbatch・データ = $PS/datasets_v30bg_<x>〔部分集合 12,000 本 ＋ val (ii) 12,420 本〕・倍率 0 を明示・SDE_DIST_MASK=1・
#      実験名 outdoor_siren_v2final_v30bg_<x>_sde_w3_bgcmp_<x>・trainer.max_epochs=20）
#   2. epoch_019 の因果推論を val 全部に（入り 0.5・ヒステリシス無し〔保持 = 入り〕・HYST_START_FRAME=0・高さの表あり・出力 v30bgcmp_<x>_e019_val）
#   3. 同じ重みで家の 13 本（out/_demo_bike_check_149・149 の約束・入り 0.5・保持 0.5・出力 v30bgcmp_<x>_e019_home13）＝ **参考**（選択には使わない）
#   決め方は scripts/v30_bgcmp_decide.py（主 = val のクラス平均 F・フレーム 40〜99）。投入は Fable の OK のあと:
#     for x in off i ii; do sbatch --export=ALL,V30BG_VARIANT=$x server_sde/v30_bgcmp_train.sbatch; done
#   乾式: V30BG_DRY=1（命令を印字するだけ）。区画は空いている所（low 可・requeue は last.ckpt から）
set -euxo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
VAR=${V30BG_VARIANT:?V30BG_VARIANT（off／i／ii）を渡す}
case "$VAR" in off|i|ii) ;; *) echo "V30BG_VARIANT は off／i／ii（$VAR）"; exit 3 ;; esac
RUN() { if [ "${V30BG_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
export V2FINAL_DATA=v30bg_$VAR
export SDE_INPUT_GAIN_DB=0                                  # 倍率 0 を明示（来歴に 0 が残る・§3-2）
export SDE_DIST_MASK=1
export V2FINAL_EXP_SUFFIX=_bgcmp_$VAR
export V2FINAL_EXTRA_OVR="trainer.max_epochs=20"
EXP=outdoor_siren_v2final_v30bg_${VAR}_sde_w3_bgcmp_${VAR}
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CK=$RUNS/$EXP/checkpoints/epoch_019.ckpt
[ -d "$PS/datasets_v30bg_$VAR/outdoor_siren_v12" ] || [ "${V30BG_DRY:-0}" = 1 ] || { echo "置き場 $PS/datasets_v30bg_$VAR が無い → 先に v30_bgcmp_gen.sbatch"; exit 3; }
# 1. 学習（epoch_019.ckpt があれば飛ばす＝再開のため）
if [ ! -f "$CK" ]; then
  RUN bash $E2E/server_sde/v2final_train.sbatch
fi
if [ ! -f "$CK" ] && [ "${V30BG_DRY:-0}" != 1 ]; then          # 保存の名前が epoch_019 でないとき: epoch が 19 の last.ckpt だけを使う
  LASTC=$(dirname $CK)/last.ckpt
  CKE=$($PS/.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$LASTC" 2>/dev/null || echo none)
  [ "$CKE" = "19" ] || { echo "epoch_019.ckpt が無く、last.ckpt の epoch も 19 でない（$CKE）→ 止める"; exit 3; }
  CK=$LASTC
fi
# 2. val の因果推論（入り 0.5・ヒステリシス無し）
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; RUN mkdir -p $JOBDIR
for f in _causal_infer_v17.py _prediction_integrity.py hysteresis_patch.py gain_patch.py; do RUN cp $E2E/server_sde/$f $JOBDIR/$f; done
RUN cp $E2E/server_sde/height_patch.py $PS/height_patch.py
RUN cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
export PS HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_OFFSET=0 HEIGHT_DEFAULT=1.5
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final_v30bg_val/plan/assignment_v2final.csv   # plan は全置き場で同じ（v2final の plan の写し）
export CAUSAL_PREFIX=val_syn_ CAUSAL_LIMIT=0 CAUSAL_SED_TH=0.5 CAUSAL_SED_STAY=0.5 HYST_START_FRAME=0
unset CAUSAL_GLOB CAUSAL_FOA CAUSAL_MANIFEST CAUSAL_THRESHOLDS_JSON CAUSAL_THRESHOLDS_MODE DUMP_NORMS
export CAUSAL_CKPT=$CK CAUSAL_DSDIR=$PS/datasets_v30bg_$VAR CAUSAL_OUT=v30bgcmp_${VAR}_e019_val
export CAUSAL_EXPECTED_CLIPS=12420
cd $PS
if [ -f "$RUNS/$CAUSAL_OUT/val_all_causal.done.json" ]; then echo "済み: $CAUSAL_OUT"; else RUN .venv/bin/python $JOBDIR/_causal_infer_v17.py; fi
# 3. 家の 13 本（参考・選択には使わない・149 の約束の置き場）
if [ -e "$RUNS/v30bgcmp_${VAR}_e019_home13" ]; then echo "済み: v30bgcmp_${VAR}_e019_home13"; else
  RUN env CAUSAL_CKPT=$CK OUTTAG=v30bgcmp_${VAR}_e019_home13 DEMO_DIR=$E2E/out/_demo_bike_check_149 CAUSAL_SED_TH=0.5 CAUSAL_SED_STAY=0.5 \
      bash $E2E/server_sde/demo_bike_check_infer.sbatch
fi
RUN rm -rf "$JOBDIR"
echo "V30BG_TRAIN_DONE variant=$VAR ckpt=$CK val=$RUNS/v30bgcmp_${VAR}_e019_val home13=$RUNS/v30bgcmp_${VAR}_e019_home13"
