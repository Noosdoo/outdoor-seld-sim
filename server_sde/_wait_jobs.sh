#!/bin/bash
# サーバのジョブ完了を待つ見張り（2026-09-06）。ssh の失敗と「キューが空」を区別する（5:11 の接続断で見張りが誤って完了扱いにした教訓）。
# 使い方: bash server_sde/_wait_jobs.sh "<job名をカンマ区切り>" <間隔秒> [<ログ>]
#   例: bash server_sde/_wait_jobs.sh "v17btrain,v17bcausal" 900 out/v17b_poll_log.txt
# 終了コード: 0 = キューが空になった（完了）、2 = ssh が 6 回続けて失敗（判定不能）
NAMES=$1; INTERVAL=${2:-600}; LOG=${3:-/dev/stdout}
fails=0
while true; do
  out=$(ssh -o ConnectTimeout=30 -o BatchMode=yes is-server "squeue -u \$USER -h -n $NAMES -o '%i:%T' 2>/dev/null | tr '\n' ' '"; echo "rc=$?")
  rc=${out##*rc=}; st=${out%rc=*}
  if [ "$rc" != "0" ]; then
    fails=$((fails + 1)); echo "$(date +%H:%M) ssh失敗 ($fails/6) — キューが空とは扱わない" >> $LOG
    [ $fails -ge 6 ] && { echo "$(date +%H:%M) ssh が続けて失敗。判定不能で終了" >> $LOG; exit 2; }
    sleep 120; continue
  fi
  fails=0
  echo "$(date +%H:%M) $st" >> $LOG
  [ -z "${st// /}" ] && { echo "$(date +%H:%M) キューが空（完了）" >> $LOG; exit 0; }
  sleep $INTERVAL
done
