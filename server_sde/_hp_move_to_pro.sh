#!/bin/bash
# ④追試の因果ftジョブ（hpc_*）を pro_6000 が空き次第そちらへ移す（冪等・監視から定期実行）。
# 本人「早いサーバ空いてたら順次移動してね」（2026-09-03 10:07）。
#
#   bash ~/research/outdoor_seld_e2e/server_sde/_hp_move_to_pro.sh   → 1行出力
#
# 方針: 空きGPU 1枚につき1本移す。待機中を優先、無ければ実行中（last.ckpt=5ep毎から再開・因果推論は
# クリップ単位で再開するので損失は最大5ep分）。pro_6000 上のジョブは触らない。
set -u
cd ~/research/outdoor_seld_e2e
used=$(sinfo -h -N -p pro_6000 -O gresused | grep -o "pro_6000:[0-9]*" | head -1 | cut -d: -f2)
free=$(( 4 - ${used:-4} ))
[ "$free" -ge 1 ] || { echo "NOOP_pro_busy(used=$used)"; exit 0; }
PRO="-p pro_6000 --gres=gpu:pro_6000:1"
moved=""
for st in PD R; do
  while read -r jid name state part; do
    [ -z "${jid:-}" ] && continue
    [ "$part" = "pro_6000" ] && continue
    [ "$free" -ge 1 ] || break
    arm=${name#hpc_}
    scancel "$jid"; sleep 5
    new=$(sbatch --parsable --export=ALL,ARM=$arm -J hpc_$arm $PRO server_sde/hp_causal.sbatch)
    moved="$moved $arm:$jid->$new($st)"
    free=$((free-1))
  done < <(squeue -h -u "$USER" -t "$st" -o "%i %j %t %P" | grep " hpc_")
done
[ -n "$moved" ] && echo "MOVED$moved" || echo "NOOP_nothing_movable"
