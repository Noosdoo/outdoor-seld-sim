#!/bin/bash
# 空き GPU 取り（本人 2026-09-08 15:07「俺が言わなくても定期的に確認して、空いてたらすぐとってください」）。
# ログインノードで常駐:  setsid nohup bash server_sde/_gpu_grabber.sh > out/_gpu_grabber.log 2>&1 &
# 2 分おきに a100（満杯 3 枚）と pro_6000（4 枚）の空きを数え、自分の PENDING ジョブ（v19* / reinfer_*）を空きのある区画へ出し直す。
#  - 動いているジョブには触らない。MIG（3g/1g）は使わない（3g は 3.4 倍遅い・1g は 7 倍遅い）
#  - 出し直しは取消 → 同じ引数で sbatch（v19_all は段階印と last.ckpt で続きから、reinfer は done.json で済みを飛ばす）
#  - その区画に他人の pending があれば空きは自分に来ないので、その区画は空き 0 とみなす
#  - 行動は out/_gpu_grabber_actions.txt に残す（監査用）
INTERVAL=${INTERVAL:-120}
cd ~/research/outdoor_seld_e2e
mkdir -p out
LOG=out/_gpu_grabber_actions.txt
echo "$(date '+%F %T') grabber start pid=$$ interval=${INTERVAL}s" | tee -a $LOG

free_gpus() {   # free_gpus <node> <gres 名> <枚数>。AllocTRES の行だけを見る（CfgTRES にも同じ語があり、先に一致すると空き 0 に誤る: 19:35 に発見）
  local a
  a=$(scontrol show node "$1" 2>/dev/null | grep -oE "AllocTRES=[^ ]*" | grep -oE "gres/gpu:$2=[0-9]+" | head -1 | cut -d= -f2)
  echo $(( $3 - ${a:-0} ))
}

submit_cmd() {  # submit_cmd <ジョブ名> <区画>
  local name=$1 part=$2 gres
  case $part in a100) gres=gpu:a100:1 ;; pro_6000) gres=gpu:pro_6000:1 ;; *) return ;; esac
  case $name in
    v19fast|v19static|v19turn|v19platoon)
      echo "sbatch --parsable -J $name -p $part --gres=$gres --qos=low --export=ALL,SCENE=${name#v19} server_sde/v19_all.sbatch" ;;
    reinfer_v18)
      echo "sbatch --parsable -J $name -p $part --gres=$gres --qos=low --export=ALL,MODEL=v18,BASELINE_SCENES=none server_sde/v18_v19_reinfer.sbatch" ;;
    reinfer_v17b)
      echo "sbatch --parsable -J $name -p $part --gres=$gres --qos=low --export=ALL,MODEL=v17b server_sde/v18_v19_reinfer.sbatch" ;;
    reinfer_v19fast|reinfer_v19static|reinfer_v19turn|reinfer_v19platoon)
      echo "sbatch --parsable -J $name -p $part --gres=$gres --qos=low --export=ALL,MODEL=${name#reinfer_} server_sde/v18_v19_reinfer.sbatch" ;;
    *) echo "" ;;
  esac
}

while :; do
  fa=$(free_gpus isws-dnnserver02 a100 3)
  fp=$(free_gpus isws-dnnserver03 pro_6000 4)
  others_a=$(squeue -h -t PD -p a100 -o "%u" | grep -vc "^$USER$")
  others_p=$(squeue -h -t PD -p pro_6000 -o "%u" | grep -vc "^$USER$")
  [ "$others_a" -gt 0 ] && fa=0
  [ "$others_p" -gt 0 ] && fp=0
  if [ "$fa" -gt 0 ] || [ "$fp" -gt 0 ]; then
    squeue -h -u "$USER" -t PD -o "%i %j %P" | sort -n | while read -r jid name part; do
      target=""
      if [ "$part" != a100 ] && [ "$fa" -gt 0 ]; then target=a100; fa=$((fa - 1))
      elif [ "$part" != pro_6000 ] && [ "$fp" -gt 0 ] && [[ "$name" != *_inf ]]; then target=pro_6000; fp=$((fp - 1)); fi   # E2: 補助推論（*_inf）は pro_6000 に移さない（GPU 種の混在を避ける）
      [ -n "$target" ] || continue
      cmd=$(submit_cmd "$name" "$target")
      [ -n "$cmd" ] || continue
      if scancel "$jid"; then
        sleep 2
        nj=$($cmd) && echo "$(date '+%F %T') moved $name $jid($part) -> $target job=$nj" | tee -a $LOG
      fi
    done
  fi
  sleep "$INTERVAL"
done
