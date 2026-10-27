#!/bin/bash
#SBATCH -J v2fdev
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 40:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v2fdev-%j.out
# 【開発用】決定版の学習データ（仮の数字の版・PROVISIONAL）で、基準学習 → 因果 ft → 保存 ckpt 全部の推論 を 1 ジョブで通す（2026-09-15 夜）。
# 目的 = 11 月の本番手順の通し試験と、束のレシピの知覚層・通知層の数字を先に見ること。**確定評価には使わない**（本番は v2final_train.sbatch → v2final_causal.sbatch を 10 月末の確定後に）。
# 段階印 out/v2final_dev_stage_<train|causal>.done で再開時に飛ばす（低優先で横取りされても requeue で続きから）。
# 投入: sbatch server_sde/v2final_dev_all.sbatch
set -x
cd ~/research/outdoor_seld_e2e
# DEV_TAG（2026-09-23）: 2 回目以降の開発用の一周は別の札で走らせ、9/16 の一周（dev・D1 なし・発進なしの束）の ckpt・推論・段階印を壊さない。
#   投入例: sbatch --export=ALL,DEV_TAG=dev2 server_sde/v2final_dev_all.sbatch   （既定 dev = 従来と同じ名前）
export DEV_TAG=${DEV_TAG:-dev}
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) restart=${SLURM_RESTART_COUNT:-0} tag=$DEV_TAG $(date)"
stage() {
  local name=$1
  local file=$2
  local mark=out/v2final_${DEV_TAG}_stage_${name}.done
  if [ -f "$mark" ]; then echo "skip $name (done)"; return 0; fi
  bash $file; local rc=$?
  if [ $rc -ne 0 ]; then echo "STAGE_FAILED $name rc=$rc"; return $rc; fi
  touch "$mark"
}
stage train  server_sde/v2final_dev_train.sh  || exit 1
stage causal server_sde/v2final_dev_causal.sh || exit 1
echo V2FINAL_DEV_ALL_DONE
