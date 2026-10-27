#!/bin/bash
# v20 の因果推論が始まったら補助ジョブ（v20_infer_helper.sbatch）を空いている GPU に出す（ログインノード常駐・2026-09-09 本人「並列で」「pro_6000 が空けばそっち」）。
#   setsid nohup bash server_sde/_v20_parallel_launcher.sh > out/_v20_launcher.log 2>&1 < /dev/null &
# 判定: 本体ジョブ（v20all-<job>.out）に CAUSAL_OUT= が出た = 因果学習が終わって推論に入った。補助は 1 本体につき 1 回だけ（印 out/_v20_helper_<NAME>.submitted）。
# 区画: a100 に空きがあれば a100、無ければ pro_6000 に空きがあれば pro_6000、どちらも無ければ a100 で待機（空き GPU 取りが空いた方へ動かす）。
cd ~/research/outdoor_seld_e2e
TASKS_DEFAULT="v18:e149 v18:e139 v18:e119 v18:e099 self:e149 v12:e149"
free_gpus() { local a; a=$(scontrol show node "$1" 2>/dev/null | grep -oE "AllocTRES=[^ ]*" | grep -oE "gres/gpu:$2=[0-9]+" | head -1 | cut -d= -f2); echo $(( $3 - ${a:-0} )); }
echo "$(date '+%F %T') launcher start pid=$$"
while :; do
  left=0
  for NAME in v20a v20b; do
    mark=out/_v20_helper_${NAME}.submitted
    [ -f $mark ] && continue
    left=$((left + 1))
    jid=$(squeue -h -u $USER -n $NAME -o '%i' | head -1)
    [ -n "$jid" ] || continue
    f=v20all-$jid.out
    [ -f "$f" ] && grep -q 'CAUSAL_OUT=' "$f" || continue
    fa=$(free_gpus isws-dnnserver02 a100 3); fp=$(free_gpus isws-dnnserver03 pro_6000 4)
    others_a=$(squeue -h -t PD -p a100 -o "%u" | grep -vc "^$USER$"); others_p=$(squeue -h -t PD -p pro_6000 -o "%u" | grep -vc "^$USER$")
    # 敵対レビュー E2/A（2026-09-09）: 推論の出力は GPU 種（A100 / PRO 6000）で違う。同じ表の中で混ぜないため補助は a100 だけに出す（空いていなければ次回）
    if [ "$fa" -gt 0 ] && [ "$others_a" -eq 0 ]; then part=a100; gres=gpu:a100:1
    else echo "$(date '+%F %T') $NAME: a100 に空きなし（pro_6000 は使わない）→ 次回"; part=""; fi
    [ -n "$part" ] && hj=$(sbatch --parsable -J ${NAME}_inf -p $part --gres=$gres --qos=low --export=ALL,NAME=$NAME,TASKS="$TASKS_DEFAULT" server_sde/v20_infer_helper.sbatch) && \
      { echo "$(date '+%F %T') helper for $NAME -> $part job=$hj (main=$jid)" | tee -a out/_gpu_grabber_actions.txt; touch $mark; }
  done
  [ $left -eq 0 ] && { echo "$(date '+%F %T') launcher done"; break; }
  sleep 300
done
