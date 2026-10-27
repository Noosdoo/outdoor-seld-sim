#!/bin/bash
#SBATCH -J v43tunegen
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o v43tunegen-%j.out
# fold32（v4.3チューニングval）1,800本: 生成(12シャード) → 距離GT → ft2因果推論 → 連結。
# 宣言= md/design/通知v4.3_検討の事前宣言_2026-09-03.md §1 段階2。
# 書き込みは out/dataset_outdoor_siren_v43tune/ と infer_v43tune_ft2_causal/ のみ。基準はguardで保証。
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

$PY scripts_archive/step10_v43tune_plan.py | tail -2
N=1800; SH=12
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/_run_v43tune_gen.py --rows $lo-$hi > v43tune_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v43tune/foa | wc -l
$PY scripts_archive/_step19_dist_labels_v43tune.py | tail -1
ls out/dataset_outdoor_siren_v43tune/metadata_dist | wc -l

# ft2（確定評価73.0%のモデル）の因果推論を当てる
cd $PS
export PS
export CAUSAL_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_causal_ft2/checkpoints/epoch_099.ckpt
export CAUSAL_OUT=infer_v43tune_ft2_causal
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_PREFIX=fold32_
export CAUSAL_FOA=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v43tune/foa
export CAUSAL_LIMIT=0
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py

guard_verify || exit 1
echo V43TUNE_DONE
