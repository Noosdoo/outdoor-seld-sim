#!/bin/bash
# 保守の予約（maint）と主 seed の学習 job 7018 を 30 分ごとに見張る（最長 24 時間・読むだけ）。7018 が走り出すか消えたら止まる。
LOG="/c/Users/satos/AppData/Local/Temp/claude/c--Users-satos-research/cec807fb-e518-4328-9066-69d11c9ff819/scratchpad/watch_7018.log"
: > "$LOG"
for i in $(seq 1 48); do
  out=$(timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes is-server 'r=$(scontrol show res maint 2>/dev/null | grep -oE "StartTime=[^ ]+|EndTime=[^ ]+|State=[A-Z]+" | tr "\n" " "); j=$(squeue -j 7018 -h -o "%t %M %r" 2>/dev/null); echo "res[$r] 7018[$j]"' 2>&1)
  echo "$(date +%m/%d %H:%M) $out" >> "$LOG"
  case "$out" in
    *"7018[R "*|*"7018[]"*|*"res[] 7018"*)
      echo "== 変化あり $(date +%m/%d %H:%M) ==" >> "$LOG"
      timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes is-server 'squeue -u $USER -o "%i %j %t %M %r"; sacct -j 7018 -X -n -o JobID,State,Elapsed 2>/dev/null; scontrol show res 2>/dev/null | head -3' >> "$LOG" 2>&1
      exit 0;;
  esac
  sleep 1800
done
echo "見張りの上限（24 h）に達した" >> "$LOG"
