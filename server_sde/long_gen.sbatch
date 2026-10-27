#!/bin/bash
#SBATCH -J longgen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o longgen-%j.out
# 長尺セット v1（60 秒 × 400 本）: 生成（24 シャード並列・再開可）→ 採用モデル ft2 e099 の因果推論（任意長版）。
# 設計= md/design/長尺セット_設計と事前宣言_2026-09-03.md
# 書き込み先は out/dataset_outdoor_long_v1/ と PSELDNets_logs/.../infer_long_v1_causal/ のみ（基準は guard で無変更を保証）。
#
#   sbatch server_sde/long_gen.sbatch            （a100 なら -p a100 --gres=gpu:a100:1、SH=12 が目安）
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
export PYTHONIOENCODING=utf-8

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

# 計画表はローカルで作ったものを scp 済み（決定論なので再生成しても同一）。無ければ作る
[ -f out/dataset_outdoor_long_v1/plan/assignment_long_v1.csv ] || $PY scripts_archive/step10_long_plan.py | tail -3

N=400; SH=${SH:-24}
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/step11_long_render.py --rows $lo-$hi > long_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_long_v1/foa | wc -l
ls out/dataset_outdoor_long_v1/metadata_dist | wc -l

# 因果推論（10 秒右詰め窓・0.1 秒ずつ・600 フレーム）
cd $PS
export PS
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_FOA=$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_long_v1/foa
export CAUSAL_PREFIX=long_
export CAUSAL_CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_causal_ft2/checkpoints/epoch_099.ckpt
export CAUSAL_OUT=infer_long_v1_causal
export CAUSAL_BATCH=50
.venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_long.py

guard_verify || exit 1
echo LONG_GEN_DONE
