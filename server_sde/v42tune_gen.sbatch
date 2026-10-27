#!/bin/bash
#SBATCH -J v42tunegen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o v42tunegen-%j.out
# v4.2チューニング専用val 1,800本: 生成(12シャード並列) → 距離GT(metadata_dist)。
# 書き込みは out/dataset_outdoor_siren_v42tune/ のみ。基準3データセットはguardで保証。
set -x
cd ~/research/outdoor_seld_e2e
PY=~/research/PSELDNet/PSELDNets/.venv/bin/python
export PYTHONPATH=scripts:src

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

# plan（決定論・全既存planとのseed排他をassert）
$PY scripts_archive/step10_v42tune_plan.py

N=1800; SH=12
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/_run_v42tune_gen.py --rows $lo-$hi > v42tune_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v42tune/foa | wc -l

# 距離GT（az/el整数一致の検証ゲート込み）
$PY scripts_archive/_step19_dist_labels_v42tune.py
ls out/dataset_outdoor_siren_v42tune/metadata_dist | wc -l

guard_verify
echo V42TUNE_GEN_DONE
