#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 40:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v19all-%j.out
# v19 の 1 場面を 1 ジョブで通す（生成 → 束と前処理 → 基準学習 → 因果 ft と推論）。2026-09-08。
# QoS normal は同時 1 本・投入 8 本までなので、場面ごとの連鎖（4 ジョブ）を 4 つ並べられない → low（横取りされたら requeue・学習は last.ckpt から再開・生成は済んだ分を飛ばす）で 1 ジョブに畳む。
# 投入例: sbatch -J v19fast --export=ALL,SCENE=fast server_sde/v19_all.sbatch
set -x
SCENE=${SCENE:?SCENE=fast|static|turn|platoon}
cd ~/research/outdoor_seld_e2e
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) scene=$SCENE restart=${SLURM_RESTART_COUNT:-0} $(date)"
stage() {   # stage <名前> <sbatch ファイル> : 済み印があれば飛ばす
  local name=$1
  local file=$2
  local mark=out/v19_stage_${SCENE}_${name}.done   # 1 行の local に書くと $name が空のまま展開される（4044/4045 で踏んだ）
  if [ -f "$mark" ]; then echo "skip $name (done)"; return 0; fi
  bash $file; local rc=$?
  if [ $rc -ne 0 ]; then echo "STAGE_FAILED $name rc=$rc"; return $rc; fi
  touch "$mark"
}
stage gen    server_sde/v19_gen.sbatch    || exit 1
stage prep   server_sde/v19_prep.sbatch   || exit 1
stage train  server_sde/v19_train.sbatch  || exit 1
stage causal server_sde/v19_causal.sbatch || exit 1
echo V19_ALL_DONE_$SCENE
