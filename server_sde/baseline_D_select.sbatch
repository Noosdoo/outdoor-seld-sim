#!/bin/bash
#SBATCH -J bDselect
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -t 6:00:00
#SBATCH -o bDselect-%j.out
# 相手 D のゲインと閾値の選択（変更票 #73・Q4・Astra R73-05）: val の選択用の部分集合（clip 番号 %12==0・名簿 manifest_val_select.csv）を
# ゲイン {0,20,30,40,50,60,70} dB で推論し、ヒステリシス前の norm を残す（BASELINE_DUMP_NORMS=1）。閾値の掃引は手元で scripts/_baseline_select_gain.py。
#   出力: $HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_select/infer_baselineD_<部分集合>_g<ゲイン>/（来歴つき・norms/）
#   使い方: sbatch --qos=low server_sde/baseline_D_select.sbatch   （eval v3・実録には触れない・val だけ）
#   確認用（%12==6・manifest_val_check.csv）を選んだゲインだけで: D_SUBSET=check D_GAINS=40 sbatch --qos=low -t 1:30:00 server_sde/baseline_D_select.sbatch
#     → 手元で scripts/_baseline_select_gain.py ... --check-runs 40=<infer_baselineD_check_g40> で check_at_selected を埋める（選択には使わない・併記だけ）
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $PS
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
export PS E2E
DS170=$PS/datasets_pretrain170
mkdir -p $DS170
cp $E2E/out/_baselines/cls_indices_train_pseldnets170.tsv $DS170/cls_indices_train.tsv
export CAUSAL_CKPT=$PS/ckpts/mACCDOA-HTSAT-0.567.ckpt
export CAUSAL_DSDIR=$DS170
export CAUSAL_DATASET=synth
export BASELINE_EXPERIMENT=synth_maccdoa
export BASELINE_CLASSMAP=$E2E/md/design/baselines_class_map_v2.json
export CAUSAL_FOA=$PS/datasets_v29/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
SUBSET=${D_SUBSET:-select}            # select（%12==0・選ぶ）／check（%12==6・確認・併記だけ）
case "$SUBSET" in select|check) ;; *) echo "D_SUBSET は select か check（$SUBSET）"; exit 3 ;; esac
GAINS=${D_GAINS:-0 20 30 40 50 60 70}
export CAUSAL_MANIFEST=$E2E/out/dataset_outdoor_siren_v2final/plan/manifest_val_$SUBSET.csv
export CAUSAL_LIMIT=0
N_SEL=$(($(wc -l < $CAUSAL_MANIFEST) - 1))
export CAUSAL_EXPECTED_CLIPS=$N_SEL
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export BASELINE_DUMP_NORMS=1
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_select
echo "subset $SUBSET（$N_SEL 本）・gains: $GAINS"
sha256sum $CAUSAL_CKPT $E2E/server_sde/_baseline_D_pretrain170_infer.py $E2E/server_sde/_baseline_merge.py $BASELINE_CLASSMAP $DS170/cls_indices_train.tsv $CAUSAL_MANIFEST | cut -c1-16,65-
for g in $GAINS; do
  export BASELINE_GAIN_DB=$g
  export CAUSAL_OUT=infer_baselineD_${SUBSET}_g$g
  $PY $E2E/server_sde/_baseline_D_pretrain170_infer.py
  $PY $E2E/server_sde/_prediction_integrity.py "$CAUSAL_RUNS_DIR/$CAUSAL_OUT" $N_SEL
  echo "gain $g: norms $(ls $CAUSAL_RUNS_DIR/$CAUSAL_OUT/norms | wc -l)・rows $(wc -l < $CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.csv)"
done
echo BASELINE_D_SELECT_DONE
