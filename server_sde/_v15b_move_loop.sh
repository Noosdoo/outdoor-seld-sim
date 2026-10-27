#!/bin/bash
# 3 分ごとに _v15b_move.sh を呼ぶ。移動したか、ジョブが無くなったら終了。nohup で login ノードに常駐。
cd ~/research/outdoor_seld_e2e
for i in $(seq 1 200); do
  out=$(bash server_sde/_v15b_move.sh)
  echo "$(date +%H:%M) $out"
  case "$out" in MOVED*|NOOP_no_job*|NOOP_already_on_*|NOOP_progress_*) exit 0;; esac
  sleep 180
done
