#!/bin/bash
#SBATCH -J detscores
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o detscores-%j.out
# 検出層の閾値掃引の材料（2026-09-15 深夜・本人「質疑は検出層の閾値の話だった」）:
#   v18 e149 と v17b e149 を、既存 val（v18 fold2 3,600 本）と交差 val（v12 fold2 1,800 本）で推論し直し、検出の度合い（ACCDOA ベクトルの長さ）を全部保存する。
#   推論の中身は v18_causal.sbatch の推論部と同じ（KMIN40・REPEAT2・高さ入力）。出力は runs/scores_dev/ の下（verified_v2 には混ぜない・確定評価には使わない）。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export HEIGHT_COND=1
export HEIGHT_JITTER=0
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
export CAUSAL_PREFIX=val_syn_          # 2026-09-22 改名（旧 fold2_）
export CAUSAL_SCORES=1
export CAUSAL_SCORE_MIN=0.05
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/scores_dev2   # Astra A4/A5 の可逆版（9 列＋manifest）。旧 scores_dev は残す
mkdir -p $CAUSAL_RUNS_DIR
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_scores.py $JOBDIR/_causal_infer_scores.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
INFER=$JOBDIR/_causal_infer_scores.py
sha256sum $INFER $JOBDIR/_prediction_integrity.py | tee $E2E/detect_scores_sha256_job${SLURM_JOB_ID}.txt
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) $(date)"
for NAME in v18 v17b; do
  CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_${NAME}_causal_ft/checkpoints/last.ckpt
  ls -l $CK
  export CAUSAL_CKPT=$CK
  # 交差 val = v12 fold2（全クリップ 1.5 m・表なし）
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=scores_${NAME}ft_e149_v12causal
  export HEIGHT_TABLE=""
  unset CAUSAL_GLOB
  .venv/bin/python $INFER
  # 既存 val = v18 fold2 3,600 本（高さ表あり）
  export CAUSAL_DSDIR=$PS/datasets_v18
  export CAUSAL_OUT=scores_${NAME}ft_e149_selfcausal
  export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v18/plan/assignment_v18.csv
  .venv/bin/python $INFER
done
for d in $CAUSAL_RUNS_DIR/scores_*; do echo "$d: $(ls $d/scores | wc -l) clips, $(du -sh $d/scores | cut -f1)"; done
cd $CAUSAL_RUNS_DIR && tar czf $E2E/out/detect_scores_$(date +%Y%m%d).tgz scores_*/scores scores_*/val_all_causal.csv scores_*/run_meta.json 2>/dev/null || true
ls -la $E2E/out/detect_scores_*.tgz
echo DETECT_SCORES_DONE
