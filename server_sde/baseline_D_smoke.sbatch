#!/bin/bash
#SBATCH -J bDsmoke
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -t 0:40:00
#SBATCH -o bDsmoke-%j.out
# 相手 D（事前学習だけの PSELDNets・変更票 #73）の推論の台本の試走: v29 の val の先頭 20 本だけ・出力は _baseline_smoke（追跡外・本番の置き場に入れない）。
#   使い方: sbatch [--qos=<低優先の名>] server_sde/baseline_D_smoke.sbatch
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
[ "$(grep -c . $DS170/cls_indices_train.tsv)" = 170 ] || { echo "170 行でない"; exit 3; }
export CAUSAL_CKPT=$PS/ckpts/mACCDOA-HTSAT-0.567.ckpt
export CAUSAL_DSDIR=$DS170
export CAUSAL_DATASET=synth
export BASELINE_EXPERIMENT=synth_maccdoa
export BASELINE_CLASSMAP=$E2E/md/design/baselines_class_map_v2.json   # v2 = 車は Car と Car passing by だけ（Astra R73-04）
export CAUSAL_FOA=$PS/datasets_v29/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
export CAUSAL_GLOB="val_syn_*.flac"
export CAUSAL_LIMIT=20
export CAUSAL_EXPECTED_CLIPS=0
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_smoke
export CAUSAL_OUT=infer_baselineD_smoke20_$SLURM_JOB_ID
sha256sum $CAUSAL_CKPT $E2E/server_sde/_baseline_D_pretrain170_infer.py $BASELINE_CLASSMAP $DS170/cls_indices_train.tsv | cut -c1-16,65-
$PY $E2E/server_sde/_baseline_D_pretrain170_infer.py
R=$CAUSAL_RUNS_DIR/$CAUSAL_OUT
$PY $E2E/server_sde/_prediction_integrity.py "$R" 20
echo "--- 行数と頭"; wc -l < $R/val_all_causal.csv; head -n 5 $R/val_all_causal.csv
echo "--- クラス別の行数"; cut -d, -f3 $R/val_all_causal.csv | sort | uniq -c
echo BASELINE_D_SMOKE_DONE
