#!/bin/bash
#SBATCH -J bAsmoke
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -t 0:40:00
#SBATCH -o bAsmoke-%j.out
# 相手 A（PANNs ＋ 強度ベクトル・変更票 #73）の推論の台本の試走: v29 の val の先頭 20 本・出力は _baseline_smoke（追跡外）。
#   使い方: BASELINE_GAIN_DB=<dB> sbatch --qos=low server_sde/baseline_A_smoke.sbatch
#   初回: panns_inference と torchlibrosa を venv に入れず _vendor に置く（--target・--no-deps）。PANNs の重みは Zenodo から取る（CC BY 4.0）
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
export E2E
VENDOR=$E2E/server_sde/_vendor
[ -d "$VENDOR/panns_inference" ] || uv pip install --quiet --python "$PY" --target "$VENDOR" --no-deps panns-inference torchlibrosa   # venv に pip は無い・uv を使う（venv は触らない）
# panns_inference の __init__ が matplotlib.pyplot を読む（描画の補助だけ）。venv に matplotlib は無く、入れると numpy の版を乱すので、_vendor に空の代用を置く（venv は触らない）
[ -f "$VENDOR/matplotlib/pyplot.py" ] || { mkdir -p "$VENDOR/matplotlib"; : > "$VENDOR/matplotlib/__init__.py"; : > "$VENDOR/matplotlib/pyplot.py"; }
export PANNS_VENDOR=$VENDOR
ls "$VENDOR" | grep -iE "panns|torchlibrosa" | tr "\n" " "; echo   # 入れた版の記録（dist-info の名前に版が入る）
W=$E2E/out/_baselines
mkdir -p $W
export PANNS_CKPT=$W/Cnn14_DecisionLevelMax_mAP=0.385.pth
[ -f "$PANNS_CKPT" ] || curl -sL -o "$PANNS_CKPT" "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
sha256sum "$PANNS_CKPT" | cut -c1-16
$PY -c "import sys; sys.path.insert(0, '$VENDOR'); import panns_inference, torchlibrosa, scipy; print('panns_inference', panns_inference.__version__ if hasattr(panns_inference, '__version__') else 'ok', 'scipy', scipy.__version__)"
export BASELINE_CLASSMAP=$E2E/md/design/baselines_class_map_v1.json
export BASELINE_GAIN_DB=${BASELINE_GAIN_DB:-0}
export CAUSAL_FOA=$PS/datasets_v29/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
export CAUSAL_GLOB="val_syn_*.flac"
export CAUSAL_LIMIT=20
export CAUSAL_EXPECTED_CLIPS=0
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_smoke
export CAUSAL_OUT=infer_baselineA_smoke20_g${BASELINE_GAIN_DB}_$SLURM_JOB_ID
$PY $E2E/server_sde/_baseline_A_panns_iv_infer.py
R=$CAUSAL_RUNS_DIR/$CAUSAL_OUT
$PY $E2E/server_sde/_prediction_integrity.py "$R" 20
echo "--- 行数と頭"; wc -l < $R/val_all_causal.csv; head -n 5 $R/val_all_causal.csv
echo "--- クラス別の行数"; cut -d, -f3 $R/val_all_causal.csv | sort | uniq -c
echo BASELINE_A_SMOKE_DONE
