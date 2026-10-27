#!/bin/bash
#SBATCH -J calibinfer
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 3:00:00
#SBATCH -o calibinfer-%j.out
# 補正表（変更票 #72・凍結 r14）: 選んだ ckpt で calib の 4,000 本（calib_fit＋calib_check・tune_syn_）と長尺 v1（400 本・60 s）を因果推論する。
#   - 使い方: V2FINAL_CKPT=<選んだ ckpt の絶対パス> sbatch server_sde/calib_infer.sbatch   （選択器 2.1 で ckpt を 1 つ選んだあと）
#   - GPU は pro_6000（凍結 §13 (c)・候補の推論と同じ GPU 種）。確定評価セット（eval v3）には触れない（eval_syn_ は読まない）。
#   - 出力: $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_calib_<TAG>/（4,000 本）と infer_long_v1_<TAG>/（400 本）。どちらも run_meta.json・val_all_causal.csv・val_all_causal.done.json（来歴つき）
#   - そのあと手元で: bash server_sde/_handoff_scripts_20261020/run_calib_decide.sh <TAG> <選択器の JSON>
# Astra RN20（2026-10-20）の直し: CAUSAL_* を初期化（RN20-06）・CONTRACT の全体 sha と全ファイルの照合（RN20-06）・本数の契約 4,000／400（RN20-06）・
#   長尺は _causal_infer_long18.py（入り 0.7・保持 0.3 のヒステリシス = 確定評価と同じ・来歴つき・RN20-01／04）・推論のあと _prediction_integrity で完了を照合
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done      # RN20-06: 外の環境を引き継がない
source $E2E/server_sde/_baseline_guard.sh
guard_record
[ -n "${V2FINAL_CKPT:-}" ] && [ -f "$V2FINAL_CKPT" ] || { echo "V2FINAL_CKPT（選んだ ckpt）が無い"; exit 3; }
SETS=$E2E/md/design/calib_sets_v29.json
CALIB=$E2E/out/dataset_outdoor_siren_v2final_calib
[ -f "$CALIB/CONTRACT_calib.json" ] || { echo "補正表の組が無い（calib_gen.sbatch）"; exit 3; }
N_CAL=$($PY -c "import json;print(json.load(open('$SETS'))['n_clips'])")
WANT=$($PY -c "import json;print(json.load(open('$SETS'))['contract_sha256'])")
[ "$(sha256sum $CALIB/CONTRACT_calib.json | cut -c1-64)" = "$WANT" ] || { echo "CONTRACT_calib.json の sha が calib_sets_v29.json の記録と違う"; exit 3; }
# CONTRACT の全ファイル（16,005）の sha が今のファイルと一致するか（抜き取りでなく全部）
$PY - "$CALIB" <<'EOF' || { echo "補正表の組の照合に失敗（組が変わっている）。中止"; exit 3; }
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1]); c = json.loads((out / "CONTRACT_calib.json").read_text(encoding="utf-8"))
bad = [k for k, v in c["files"].items() if not (out / k).is_file() or hashlib.sha256((out / k).read_bytes()).hexdigest() != v]
print(f"[calib contract] files={len(c['files'])} bad={len(bad)}"); sys.exit(1 if bad else 0)
EOF
[ "$(ls $CALIB/foa/tune_syn_*.flac | wc -l)" = "$N_CAL" ] || { echo "calib の foa が $N_CAL 本でない"; exit 3; }
LONG=$E2E/out/dataset_outdoor_long_v1
N_LONG=$($PY -c "import json;print(json.load(open('$SETS'))['long_set_n_clips'])")
FR_LONG=$($PY -c "import json;print(json.load(open('$SETS'))['long_set_frames'])")
[ -d "$LONG/foa" ] && [ "$(ls $LONG/foa/long_*.flac | wc -l)" = "$N_LONG" ] || { echo "長尺 v1 が $N_LONG 本でない"; exit 3; }
TAG=$(basename $(dirname $(dirname $V2FINAL_CKPT)))_$(basename $V2FINAL_CKPT .ckpt)
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
for d in "$RUNS/infer_calib_$TAG" "$RUNS/infer_long_v1_$TAG"; do [ -e "$d" ] && { echo "既に推論の置き場がある（黙って再開しない）: $d"; exit 4; }; done
sha256sum $V2FINAL_CKPT $CALIB/plan/assignment_v2final.csv $CALIB/CONTRACT_calib.json $SETS server_sde/_causal_infer_v17.py server_sde/_causal_infer_long18.py \
  server_sde/_prediction_integrity.py server_sde/hysteresis_patch.py server_sde/height_patch.py | tee $E2E/calib_infer_sha256_job${SLURM_JOB_ID}.txt
cd $PS
export PS E2E
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export CAUSAL_LIMIT=0
export CAUSAL_DATASET=outdoor_siren_v12
export HEIGHT_COND=1
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
export HEIGHT_DEFAULT=1.5
export CAUSAL_CKPT=$V2FINAL_CKPT
# --- calib（4,000 本・tune_syn_・高さは calib の plan の mic_z）---
export CAUSAL_DSDIR=$PS/datasets_v29
export CAUSAL_FOA=$CALIB/foa
export CAUSAL_PREFIX=tune_syn_
export CAUSAL_GLOB="tune_syn_*.flac"
export CAUSAL_EXPECTED_CLIPS=$N_CAL
export HEIGHT_TABLE=$CALIB/plan/assignment_v2final.csv
export CAUSAL_OUT=infer_calib_$TAG
.venv/bin/python $E2E/server_sde/_causal_infer_v17.py
$PY $E2E/server_sde/_prediction_integrity.py "$RUNS/$CAUSAL_OUT" $N_CAL || { echo "calib の推論の完了の照合に失敗"; exit 5; }
# --- 長尺 v1（400 本・60 s・高さ 1.5 m 固定＝旧セットの条件・検出の設定は確定評価と同じ 0.7／0.3）---
export CAUSAL_FOA=$LONG/foa
export CAUSAL_PREFIX=long_
export CAUSAL_GLOB="long_*.flac"
export CAUSAL_EXPECTED_CLIPS=$N_LONG
export CAUSAL_FRAMES=$FR_LONG
export HEIGHT_TABLE=""
export CAUSAL_OUT=infer_long_v1_$TAG
export CAUSAL_BATCH=50
.venv/bin/python $E2E/server_sde/_causal_infer_long18.py
$PY $E2E/server_sde/_prediction_integrity.py "$RUNS/$CAUSAL_OUT" $N_LONG || { echo "長尺の推論の完了の照合に失敗"; exit 5; }
guard_verify || exit 1
echo CALIB_INFER_DONE
