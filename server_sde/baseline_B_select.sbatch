#!/bin/bash
#SBATCH -J bBselect
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -t 6:00:00
#SBATCH -o bBselect-%j.out
# 相手 B の検出の閾値の選択（変更票 #73・D と同じ手順 R73-05・ゲインは無し）: val の選択用（clip%12==0・manifest_val_select.csv・1,033 本）と確認用（%12==6・manifest_val_check.csv・1,038 本）を
#   学習した重み（B_MODEL）で因果推論し、ヒステリシス前の norm を残す（BASELINE_DUMP_NORMS=1）。閾値の掃引は手元で
#   scripts/_baseline_select_gain.py --target B --gains 0 --runs 0=<選択用の run dir> → md/design/baselines_B_selection.json → 凍結 JSON に B を足す。
#   出力: $HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_select/infer_baselineB_select_g0／infer_baselineB_check_g0（来歴つき・norms/）
#   使い方: B_MODEL=<学習した .h5> sbatch --qos=low server_sde/baseline_B_select.sbatch   （eval v3・実録には触れない・val だけ）
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
export E2E
export B_DATASET_DIR=$HOME/research/_external/B_dataset2024_v29
export B_FEAT_DIR=$HOME/research/_external/B_feat_v29c     # c = top_db なしの特徴（Astra RI73-02）
export B_MODEL=${B_MODEL:-$HOME/research/_external/B_models_v29/custom_v29c_dev_split0_multiaccdoa_foa_model.h5}
[ -f "$B_MODEL" ] || { echo "B の重みが無い（$B_MODEL）"; exit 3; }
export B_OFFICIAL_COMMIT=$(cd ~/research/_external/DCASE2024_seld_baseline && git rev-parse --short HEAD)
export CAUSAL_FOA=$PS/datasets_v29/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
export CAUSAL_LIMIT=0
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export BASELINE_DUMP_NORMS=1
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_select
sha256sum $B_MODEL $E2E/server_sde/_baseline_B_infer.py $E2E/server_sde/_baseline_B_common.py $E2E/server_sde/_baseline_merge.py | cut -c1-16,65-
for sub in select check; do
  export CAUSAL_MANIFEST=$E2E/out/dataset_outdoor_siren_v2final/plan/manifest_val_$sub.csv
  N_SUB=$(($(wc -l < $CAUSAL_MANIFEST) - 1))
  export CAUSAL_EXPECTED_CLIPS=$N_SUB
  export CAUSAL_OUT=infer_baselineB_${sub}_g0
  $PY $E2E/server_sde/_baseline_B_infer.py
  $PY $E2E/server_sde/_prediction_integrity.py "$CAUSAL_RUNS_DIR/$CAUSAL_OUT" $N_SUB
  echo "$sub: norms $(ls $CAUSAL_RUNS_DIR/$CAUSAL_OUT/norms | wc -l)・rows $(wc -l < $CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.csv)"
done
echo BASELINE_B_SELECT_DONE
