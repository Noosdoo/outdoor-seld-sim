#!/bin/bash
# v19 の「normal 枠を盾にする」当番制（2026-09-08・本人「やって」）。サーバで実行: bash server_sde/_v19_rotate_normal.sh
# 規則:
#   - QoS normal は同時 1 本（running）。pending の normal は 1 本まで常に控えさせておく（running が終わった瞬間に始まる）
#   - 自分の normal の pending が無ければ、順番 platoon → static → fast のうち「未完了で、low が pending か、fast が a100_3g にいる」場面を
#     1 つだけ low を取り消して normal（a100・gpu:a100:1）で出し直す。満杯 GPU で走っている low は触らない（turn など）
#   - 生成・前処理は段階印で飛び、学習は last.ckpt から再開。失うのは最大 1 epoch
set -u
cd ~/research/outdoor_seld_e2e
ME=$USER
echo "== $(date '+%F %T') rotate check"
squeue -u $ME -h -o '%i %j %T %M %P %q %R' | sort -n
pend_normal=$(squeue -u $ME -h -t PD -q normal -o '%i' | wc -l)
if [ "$pend_normal" -ge 1 ]; then
  echo "normal の pending が既に $pend_normal 本 → 何もしない"; exit 0
fi
for s in platoon static fast; do
  [ -f out/v19_stage_${s}_causal.done ] && { echo "$s: 完了済み"; continue; }
  # この場面の normal が running なら次へ（重複させない）
  if squeue -u $ME -h -n v19$s -q normal -o '%i' | grep -q .; then echo "$s: normal で実行中"; continue; fi
  line=$(squeue -u $ME -h -n v19$s -q low -o '%i %T %P' | head -1)
  jid=$(echo "$line" | awk '{print $1}'); st=$(echo "$line" | awk '{print $2}'); part=$(echo "$line" | awk '{print $3}')
  if [ -z "$jid" ]; then
    echo "$s: low のジョブが無い（消えた？）→ normal で投入"
  elif [ "$st" = "RUNNING" ] && [ "$part" != "a100_3g" ]; then
    echo "$s: 満杯 GPU（$part）で実行中 → 触らない"; continue
  else
    echo "$s: low $jid ($st on $part) を取り消して normal へ"
    scancel $jid; sleep 2
  fi
  nj=$(sbatch --parsable -J v19$s -p a100 --gres=gpu:a100:1 --qos=normal --export=ALL,SCENE=$s server_sde/v19_all.sbatch) && echo "$s: normal job=$nj"
  break
done
sleep 2
squeue -u $ME -h -o '%i %j %T %M %P %q %R' | sort -n
