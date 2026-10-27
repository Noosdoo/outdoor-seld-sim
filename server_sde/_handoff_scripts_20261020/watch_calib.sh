#!/bin/bash
# calib_gen の job を 5 分ごとに見張り、終わったらログの末尾を出す（最長 3 時間・読むだけ）。引数 = job id
JID="$1"
LOG="/c/Users/satos/AppData/Local/Temp/claude/c--Users-satos-research/cec807fb-e518-4328-9066-69d11c9ff819/scratchpad/watch_calib.log"
: > "$LOG"
for i in $(seq 1 36); do
  st=$(timeout 40 ssh -o ConnectTimeout=8 -o BatchMode=yes is-server "squeue -j $JID -h -o '%t %M'" 2>/dev/null)
  echo "$(date +%H:%M) [$st] $(timeout 40 ssh -o ConnectTimeout=8 -o BatchMode=yes is-server 'ls ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v2final_calib/foa 2>/dev/null | wc -l' 2>/dev/null) 本" >> "$LOG"
  if [ -z "$st" ]; then
    echo "== job $JID がキューから消えた $(date +%H:%M) ==" >> "$LOG"
    timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && sacct -j $JID -X -n -o JobID,State,Elapsed,ExitCode; grep -nE '中止|失敗|NG|判定|CALIB_GEN_DONE|CONTRACT_calib' calibgen-$JID.out | tail -8 | cut -c1-160; tail -3 calibgen-$JID.out | cut -c1-160" >> "$LOG" 2>&1
    exit 0
  fi
  sleep 300
done
echo "見張りの上限（3 h）に達した" >> "$LOG"
