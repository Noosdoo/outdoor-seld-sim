#!/bin/bash
#SBATCH -J v2fdev
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 40:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v2fdev-%j.out
# 【開発用】v2final_dev_all.sbatch の改訂 r2（2026-10-02）: 因果 ft の段を v2final_dev_causal_r2.sh（再開の穴を塞いだ版）で走らせる。
# 走行中のジョブが読んでいる v2final_dev_causal.sh を書き換えないために別名にした。基準学習の段は同じ v2final_dev_train.sh。
# 投入例: sbatch --export=ALL,DEV_TAG=dev2s1,DEV_HDF5DIR=...,DEV_SEED=2025,DEV_CAUSAL_SEED=1,DEV_HEIGHT_SEED=20260918 server_sde/v2final_dev_all_r2.sbatch
set -x
cd ~/research/outdoor_seld_e2e
export DEV_TAG=${DEV_TAG:-dev}
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) restart=${SLURM_RESTART_COUNT:-0} tag=$DEV_TAG hdf5=${DEV_HDF5DIR:-default} seed=${DEV_SEED:-default} $(date)"
# r2-3（Astra 監査 10/7 V06）: 因果 ft の段は完了印があっても必ず台本を通す。台本の側（v2final_dev_causal_r2.sh）が
#   学習は last.ckpt の epoch で、推論は done.json の ckpt sha・csv sha・本数で「済み」を確かめてから飛ばすので、
#   完了印だけで中身の検査を迂回しない。基準学習の段は従来どおり完了印で飛ばす（学習の再開は台本側が last.ckpt で行う）。
stage() {
  local name=$1
  local file=$2
  local always=${3:-0}
  local mark=out/v2final_${DEV_TAG}_stage_${name}.done
  if [ -f "$mark" ] && [ "$always" != "1" ]; then echo "skip $name (done)"; return 0; fi
  [ -f "$mark" ] && echo "verify $name (done mark exists; re-running the script, which checks each artifact before skipping)"
  bash $file; local rc=$?
  if [ $rc -ne 0 ]; then echo "STAGE_FAILED $name rc=$rc"; return $rc; fi
  touch "$mark"
}
stage train  server_sde/v2final_dev_train.sh       || exit 1
stage causal server_sde/v2final_dev_causal_r2.sh 1 || exit 1
echo V2FINAL_DEV_ALL_DONE
