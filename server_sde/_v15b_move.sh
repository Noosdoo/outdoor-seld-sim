#!/bin/bash
# v15b 因果ft（MIG 3g.40gb で実行中）を、速い GPU が空き次第そちらへ移す（本人「順次早いサーバ空いたら移して」2026-09-04 19:16）。
# 優先: pro_6000（31 s/ep）> a100 丸ごと（52 s/ep）> そのまま（3g ≈ 90 s/ep）。last.ckpt（20 ep 毎）から再開するので損失は最大 20 ep。
# 進捗が epoch_099 以上なら移さない（残りが短く、再開の損失の方が大きい）。移したら v15 val 推論（xinf）も付け直す。
#   bash server_sde/_v15b_move.sh   → 1 行出力（MOVED / NOOP_*）
set -u
cd ~/research/outdoor_seld_e2e
CK=~/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v15b_causal_ft/checkpoints
cur=$(squeue -h -u "$USER" -o "%i %j %P %t" | grep " v15bcausal " | head -1)
[ -n "$cur" ] || { echo "NOOP_no_job"; exit 0; }
read -r jid _name part state <<< "$cur"
[ "$part" = "a100_3g" ] || { echo "NOOP_already_on_$part"; exit 0; }
last_ep=$(ls $CK/epoch_*.ckpt 2>/dev/null | sed 's/.*epoch_//; s/.ckpt//' | sort -n | tail -1)
[ "${last_ep:-0}" -lt 99 ] || { echo "NOOP_progress_${last_ep}_ge_99"; exit 0; }
used_pro=$(sinfo -h -N -p pro_6000 -O gresused | grep -o "pro_6000:[0-9]*" | head -1 | cut -d: -f2)
used_a=$(sinfo -h -N -p a100 -O gresused | grep -o "gpu:a100:[0-9]*" | head -1 | cut -d: -f3)
if [ $(( 4 - ${used_pro:-4} )) -ge 1 ]; then P="-p pro_6000 --gres=gpu:pro_6000:1"; tag=pro_6000
elif [ $(( 3 - ${used_a:-3} )) -ge 1 ]; then P="-p a100 --gres=gpu:a100:1"; tag=a100
else echo "NOOP_busy(pro=$used_pro/4 a100=$used_a/3 ep=${last_ep:-0})"; exit 0; fi
xinf=$(squeue -h -u "$USER" -o "%i %j" | grep " v15bxinf" | awk '{print $1}')
scancel $jid $xinf; sleep 8
new=$(sbatch --parsable $P server_sde/v15b_causal.sbatch)
newx=$(sbatch --parsable $P --dependency=afterok:$new server_sde/v15b_infer_on_v15val.sbatch)
echo "MOVED v15bcausal $jid->$new on $tag (resume from epoch_${last_ep:-none}), xinf $xinf->$newx"
