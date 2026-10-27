#!/bin/bash
#SBATCH -J bBsmoke
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH -t 2:00:00
#SBATCH -o bBsmoke-%j.out
# 相手 B（公式ベースライン）の試走: v29 の train 先頭 200 本・val 先頭 50 本で、層 → 特徴 → 標準化（train だけで fit）→ ラベル → 単位の往復検査 → 公式の学習を quick_test（2 epoch）で回す。
#   置き場は _external の試走用（B_DATASET_DIR 等を smoke に）。本番の置き場には触れない。使い方: sbatch --qos=low server_sde/baseline_B_smoke.sbatch
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
export E2E
export B_DATASET_DIR=$HOME/research/_external/B_smoke2024/dataset2024     # 名前に 2024（公式が分割を名前で決める・train fold3／val fold4）
export B_FEAT_DIR=$HOME/research/_external/B_smoke2024/feat
export B_MODEL_DIR=$HOME/research/_external/B_smoke2024/models
export B_OUT_DIR=$HOME/research/_external/B_smoke2024/out
mkdir -p $B_DATASET_DIR $B_FEAT_DIR $B_MODEL_DIR $B_OUT_DIR
[ -d ~/research/_external/DCASE2024_seld_baseline ] || { echo "公式コードが無い"; exit 3; }
cd ~/research/_external/DCASE2024_seld_baseline && echo "official commit $(git rev-parse --short HEAD)"; cd $E2E
$PY $E2E/server_sde/_baseline_B_prepare.py --step all --limit-train 200 --limit-val 50 --jobs 32
export B_QUICK=1 B_JOB=smoke
$PY $E2E/server_sde/_baseline_B_train.py
ls -la $B_MODEL_DIR | cut -c1-100
echo "--- val の予測（公式の出力）"; ls $B_OUT_DIR | tail -n 3; f=$(find $B_OUT_DIR -type f -name "*.csv" | sort | tail -n 1); echo "$f"; head -n 3 "$f" || true
# --- この研究と同じ条件の因果推論（_baseline_B_infer.py）を val の先頭 5 本で試す → 7 列 csv・来歴
export B_MODEL=$(ls $B_MODEL_DIR/*_model.h5 | head -n 1)
export CAUSAL_FOA=$PS/datasets_v29/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
export CAUSAL_GLOB="val_syn_*.flac"
export CAUSAL_LIMIT=5
export CAUSAL_SED_TH=0.5 CAUSAL_SED_STAY=0.5 CAUSAL_SED_ASSOC=20
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_smoke
export CAUSAL_OUT=infer_baselineB_smoke5_$SLURM_JOB_ID
$PY $E2E/server_sde/_baseline_B_infer.py
$PY $E2E/server_sde/_prediction_integrity.py "$CAUSAL_RUNS_DIR/$CAUSAL_OUT" 5
echo "--- 行数と頭"; wc -l < $CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.csv; head -n 5 $CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.csv
echo BASELINE_B_SMOKE_DONE
