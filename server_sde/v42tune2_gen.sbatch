#!/bin/bash
#SBATCH -J v42tune2gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o v42tune2gen-%j.out
# fold31（v4.2チューニングval第2版）1,800本: 生成(12シャード) → 距離GT。
# 書き込みは out/dataset_outdoor_siren_v42tune2/ のみ。基準はguardで保証。
set -x
cd ~/research/outdoor_seld_e2e
PY=~/research/PSELDNet/PSELDNets/.venv/bin/python
export PYTHONPATH=scripts:src

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

$PY scripts_archive/step10_v42tune2_plan.py

N=1800; SH=12
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/_run_v42tune2_gen.py --rows $lo-$hi > v42tune2_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v42tune2/foa | wc -l

$PY scripts_archive/_step19_dist_labels_v42tune2.py
ls out/dataset_outdoor_siren_v42tune2/metadata_dist | wc -l

guard_verify
echo V42TUNE2_GEN_DONE
