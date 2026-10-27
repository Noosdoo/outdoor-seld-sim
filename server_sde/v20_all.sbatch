#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 40:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v20all-%j.out
# v20 = 合成学習（v18 のラベル ＋ 採る場面を全部足した 1 本）を 1 ジョブで通す: 束と前処理 → 基準学習 → 因果 ft と推論。2026-09-09。
# 生成は不要（v19 の音声・ラベルをそのまま束ねる）。段階印 out/v20_stage_<NAME>_<prep|train|causal>.done で再開時に飛ばす。
# 投入例: sbatch -J v20a --export=ALL,NAME=v20a,SCENES="fast turn platoon" server_sde/v20_all.sbatch
#         sbatch -J v20b --export=ALL,NAME=v20b,SCENES="fast turn platoon static" server_sde/v20_all.sbatch
set -x
NAME=${NAME:?NAME=v20a|v20b}
SCENES=${SCENES:?SCENES="fast turn platoon"}
export NAME SCENES
cd ~/research/outdoor_seld_e2e
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) name=$NAME scenes=[$SCENES] restart=${SLURM_RESTART_COUNT:-0} $(date)"
stage() {
  local name=$1
  local file=$2
  local mark=out/v20_stage_${NAME}_${name}.done
  if [ -f "$mark" ]; then echo "skip $name (done)"; return 0; fi
  bash $file; local rc=$?
  if [ $rc -ne 0 ]; then echo "STAGE_FAILED $name rc=$rc"; return $rc; fi
  touch "$mark"
}
stage prep   server_sde/v20_prep.sbatch   || exit 1
stage train  server_sde/v20_train.sbatch  || exit 1
stage causal server_sde/v20_causal.sbatch || exit 1
echo V20_ALL_DONE_$NAME
