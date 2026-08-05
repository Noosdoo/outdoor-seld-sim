#!/bin/bash
#SBATCH -J v12gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -c 26
#SBATCH -o v12gen-%j.out
# v12全量生成: 10,200行を12シャード並列（CPUレンダ、GPU未使用だがノード確保のため要求）
set -x
cd ~/research/outdoor_seld_e2e
PY=~/research/PSELDNet/PSELDNets/.venv/bin/python
export PYTHONPATH=scripts:src
N=10200; SH=12
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v12_gen.py --rows $lo-$hi > v12gen_shard_$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
ls out/dataset_outdoor_siren_v12/foa | wc -l
echo V12_GEN_DONE
