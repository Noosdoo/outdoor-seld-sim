#!/bin/bash
# v16 の学習ジョブが pro_6000 の GPU 待ちで 10 分以上止まったら a100（空きがあれば）に振り替える。
# 本人 2026-09-04 23:09「早いサーバ空いてたら移して」。pro_6000 が最速（31 s/ep）なので基本はそこで待ち、
# 待ちが長引いたときだけ a100（52 s/ep）へ。サーバ側で nohup 実行:
#   nohup bash server_sde/_v16_failover.sh <train job> <causal job> [train.sbatch] [causal.sbatch] > /dev/null 2>&1 &
# 段階ごと（train → causal）に同じ規則を適用する。振り替えたら job 番号を v16_failover.log に書く。
cd ~/research/outdoor_seld_e2e
LOG=v16_failover.log
TRAIN=${1:-3968}; CAUSAL=${2:-3969}
TRAIN_SB=${3:-server_sde/v16_train.sbatch}; CAUSAL_SB=${4:-server_sde/v16_causal.sbatch}   # 2026-09-05: v17 でも使えるよう sbatch を引数に
echo "$(date) start train=$TRAIN causal=$CAUSAL" >> $LOG

a100_free() {  # 空いている a100（フル GPU）の枚数
  local used
  used=$(sinfo -h -p a100 -O gresused | grep -o "gpu:a100:[0-9]" | head -1 | cut -d: -f3)
  echo $(( 3 - ${used:-3} ))
}

watch_stage() {  # $1=job id, $2=sbatch file, $3=次段の job id（あれば。振り替え時に依存を貼り直す）, $4=次段 sbatch
  local job=$1 sb=$2 nxt=$3 nxt_sb=$4 waited=0 st state reason
  while true; do
    st=$(squeue -h -j $job -o "%T %r" 2>/dev/null)
    if [ -z "$st" ]; then echo "$(date) $job finished/gone: $(sacct -j $job -n -o State 2>/dev/null | head -1)" >> $LOG; return 0; fi
    state=${st%% *}; reason=${st#* }
    if [ "$state" = "RUNNING" ]; then echo "$(date) $job RUNNING ($sb) -> wait for end" >> $LOG; waited=0
    elif [ "$state" = "PENDING" ] && [ "$reason" != "Dependency" ] && [ "$reason" != "DependencyNeverSatisfied" ]; then waited=$((waited + 1))
    else waited=0; fi
    if [ $waited -ge 2 ] && [ "$(a100_free)" -ge 1 ]; then
      local a100_sb=${sb%.sbatch}_a100.sbatch
      sed -e 's/^#SBATCH -p pro_6000/#SBATCH -p a100/' -e 's/gpu:pro_6000:1/gpu:a100:1/' $sb > $a100_sb
      local j2 j3
      j2=$(sbatch --parsable $a100_sb) || { echo "$(date) sbatch failed" >> $LOG; sleep 300; continue; }
      if [ -n "$nxt" ]; then
        scancel $nxt
        j3=$(sbatch --parsable --dependency=afterok:$j2 $nxt_sb)
        echo "$(date) moved $job -> a100 $j2 ($a100_sb); next stage $nxt -> $j3 (pro_6000, afterok:$j2)" >> $LOG
        NEXT_ID=$j3
      else
        echo "$(date) moved $job -> a100 $j2 ($a100_sb)" >> $LOG
      fi
      scancel $job
      job=$j2; waited=0
    fi
    if [ "$state" = "PENDING" ] && [ "$reason" = "DependencyNeverSatisfied" ]; then echo "$(date) $job dependency never satisfied" >> $LOG; return 1; fi
    sleep 300
  done
}

NEXT_ID=$CAUSAL
watch_stage $TRAIN $TRAIN_SB $CAUSAL $CAUSAL_SB || exit 1
watch_stage $NEXT_ID $CAUSAL_SB "" ""
echo "$(date) all stages done" >> $LOG
