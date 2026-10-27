#!/usr/bin/env bash
# PC（CPU）で今の主のモデルの因果推論を回す（切り分け・手応えの確かめ専用・確定評価や実録の評価には使わない・2026-10-25 Opus）。
# 条件はサーバの server_sde/demo_bike_check_infer.sbatch と同じ（提案法: 高さあり・入り 0.7／保持 0.3・ASSOC 20・SDE 3.0 log・KMIN 40）。
# 使い方: bash scripts/local_cpu_infer.sh <FOA の置き場> <高さの表> <本数> <出力名> [入り] [保持]
#   作業の置き場 LOCAL_DIR（既定 out/_local_cpu_infer・git の外）に、ckpt/（CKPT が無ければサーバから写す）・PS/（サーバの configs ＋ PC の src）・
#   ds/（cls_indices_train.tsv）・runs/<出力名>/val_all_causal.csv・raw_<出力名>/<clip>.npy を作る。
# 前提: PC の PSELDNet/PSELDNets/.venv に部品が入っていること（scripts/local_cpu_infer_wrap.py の説明）。
set -e
E2E=$(cd "$(dirname "$0")/.." && pwd)
L=${LOCAL_DIR:-$E2E/out/_local_cpu_infer}
SRC_PS=$(cd "$E2E/../PSELDNet/PSELDNets" && pwd)
VENV_PY="$SRC_PS/.venv/Scripts/python.exe"
CKPT_REMOTE=${CKPT_REMOTE:-PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_causal_ft/checkpoints/epoch_139.ckpt}
CKPT=${CKPT:-$L/ckpt/$(basename "$(dirname "$(dirname "$CKPT_REMOTE")")")_$(basename "$CKPT_REMOTE")}
mkdir -p "$L/ckpt" "$L/ds/outdoor_siren_v12" "$L/runs"
[ -f "$CKPT" ] || timeout 900 scp -q "is-server:$CKPT_REMOTE" "$CKPT"
[ -f "$L/ds/cls_indices_train.tsv" ] || timeout 60 scp -q is-server:research/PSELDNet/PSELDNets/datasets_v27/cls_indices_train.tsv "$L/ds/cls_indices_train.tsv"
if [ ! -d "$L/PS/configs" ]; then
  mkdir -p "$L/PS"
  timeout 300 scp -q -r is-server:research/PSELDNet/PSELDNets/configs "$L/PS/configs"
  cp -r "$SRC_PS/src" "$L/PS/src"
  find "$L/PS/src" -name "__pycache__" -type d -prune -exec rm -rf {} +
fi
export PS="$L/PS" E2E PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export CAUSAL_SCRIPT="$E2E/server_sde/_causal_infer_v17.py"
export CAUSAL_CKPT="$CKPT" CAUSAL_OUT="$4" CAUSAL_RUNS_DIR="$L/runs"
export HEIGHT_COND=1 HEIGHT_TABLE="$2" HEIGHT_DEFAULT=1.5 HEIGHT_JITTER=0 HEIGHT_OFFSET=0
export CAUSAL_SED_TH="${5:-0.7}" CAUSAL_SED_STAY="${6:-0.3}" CAUSAL_SED_ASSOC=20 CAUSAL_DSDIR="$L/ds"
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0 CAUSAL_DATASET=outdoor_siren_v12
export CAUSAL_FOA="$1" CAUSAL_PREFIX=eval_field_ CAUSAL_LIMIT=0 CAUSAL_EXPECTED_CLIPS="$3"
export LOCAL_RAW_DIR="$L/raw_$4"
unset CAUSAL_GLOB
cd "$PS"
"$VENV_PY" "$E2E/scripts/local_cpu_infer_wrap.py"
echo LOCAL_INFER_DONE "$4"
