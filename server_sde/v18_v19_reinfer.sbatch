#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -t 40:00:00
#SBATCH -o verified-infer-%j.out
# Re-inference only: no training, no legacy deletion/certification, no automatic submission.
# Example: sbatch --export=ALL,MODEL=v17b server_sde/v18_v19_reinfer.sbatch
# MODEL=v17b|v18|v19fast|v19static|v19turn|v19platoon.
# MODEL=v18 also computes the four comparisons, once every bundle is complete.
# Fetch afterwards with VERIFIED_RUNS_DIR matching CAUSAL_RUNS_DIR below.
set -euxo pipefail
MODEL=${MODEL:?MODEL required}
export E2E=$HOME/research/outdoor_seld_e2e
export PS=$HOME/research/PSELDNet/PSELDNets
cd "$PS"
export CAUSAL_RUNS_DIR=${CAUSAL_RUNS_DIR:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2}
export SDE_DIST_MODE=log SDE_W_DIST=3.0
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_OFFSET=0 HEIGHT_DEFAULT=1.5 CAUSAL_PREFIX=fold2_
unset CAUSAL_GLOB CAUSAL_LIMIT CAUSAL_FOA
case "$MODEL" in
  v17b) ROOTDIR=$PS/datasets_v16; TABLE=$E2E/out/dataset_outdoor_siren_v16/plan/assignment_v16.csv ;;
  v18) ROOTDIR=$PS/datasets_v18; TABLE=$E2E/out/dataset_outdoor_siren_v18/plan/assignment_v18.csv ;;
  v19fast|v19static|v19turn|v19platoon)
    SCENE=${MODEL#v19}; ROOTDIR=$PS/datasets_v19_$SCENE
    TABLE=$E2E/out/dataset_outdoor_siren_v19_$SCENE/plan/assignment_v19_${SCENE}_all.csv ;;
  *) echo "invalid MODEL: $MODEL"; exit 2 ;;
esac
CHECKPOINTS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_${MODEL}_causal_ft/checkpoints
# 推論台本と来歴モジュールはジョブごとの写しに固定する（共有パスの差し替えが実行中のジョブに混入した 2026-09-08 14:11 の再発防止）
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p "$JOBDIR"       # ジョブ専用ディレクトリ（監査 A7: 推論の import 先まで専用化・2026-09-09）
INFER=$JOBDIR/_causal_infer_v17.py
INTEG=$JOBDIR/_prediction_integrity.py
cp "$E2E/server_sde/_causal_infer_v17.py" "$INFER"
cp "$E2E/server_sde/_prediction_integrity.py" "$INTEG"
run_infer() {   # 完了済み（done.json）なら推論を飛ばして検証だけ。出力先が別ジョブの排他ロックで使用中なら 10 分おきに最大 12 回待つ
  local expected=$1 n=0 log=$E2E/out/_infer_retry_${SLURM_JOB_ID}.log
  export CAUSAL_EXPECTED_CLIPS=$expected
  if [ ! -f "$CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.done.json" ]; then
    while :; do
      if .venv/bin/python "$INFER" 2>&1 | tee "$log"; then break; fi
      if grep -q "run is in use" "$log" && [ $n -lt 12 ]; then n=$((n + 1)); echo "出力先が使用中 → 600 s 待って再試行 ($n)"; sleep 600; continue; fi
      return 1
    done
  fi
  .venv/bin/python "$INTEG" "$CAUSAL_RUNS_DIR/$CAUSAL_OUT" "$expected"
}
# BASELINE_SCENES=none で v19 の束への比較推論を行わない（各場面の因果台本が自分で行う）
[ "${BASELINE_SCENES:-}" = "none" ] && BASELINE_SCENES=" "
for e in 099 119 139 149; do
  export CAUSAL_CKPT=$CHECKPOINTS/epoch_$e.ckpt
  if [ "$e" = 149 ] && [ ! -f "$CAUSAL_CKPT" ]; then export CAUSAL_CKPT=$CHECKPOINTS/last.ckpt; fi
  # The inference identity also verifies the actual checkpoint epoch against CAUSAL_OUT.
  [ -f "$CAUSAL_CKPT" ] || { echo "missing checkpoint: $CAUSAL_CKPT"; exit 3; }
  export HEIGHT_TABLE="" CAUSAL_DSDIR=$PS/datasets_v12 CAUSAL_OUT=infer_${MODEL}ft_e${e}_v12causal
  run_infer 1800
  export HEIGHT_TABLE=$TABLE CAUSAL_DSDIR=$ROOTDIR CAUSAL_OUT=infer_${MODEL}ft_e${e}_selfcausal
  count=3600; [[ "$MODEL" != v19* ]] || count=3900
  run_infer "$count"
  if [ "$MODEL" = v18 ]; then
    for scene in ${BASELINE_SCENES:-fast static turn platoon}; do
      [[ "$scene" =~ ^(fast|static|turn|platoon)$ ]] || exit 2
      export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v19_$scene/plan/assignment_v19_${scene}_all.csv
      export CAUSAL_DSDIR=$PS/datasets_v19_$scene CAUSAL_OUT=infer_v18ft_e${e}_v19${scene}causal
      run_infer 3900
    done
  fi
done
echo VERIFIED_REINFERENCE_DONE_$MODEL
