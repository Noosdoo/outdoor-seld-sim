#!/bin/bash
# アブレーション（#74）の学習 → 因果 ft を「pro_6000 と a100 の空いたほう」で（本人 2026-10-21 09:49「両候補で」）。
#   同じ札の古い列（v2ftrain_abl_<札>・v2fcausal_abl_<札>・pro_6000 だけ・待ち）を取り消し、包み v2final_abl_either.sbatch を
#   partition ごとに 1 本ずつ（学習 2 本・因果 ft 2 本・因果 ft は afterany で学習 2 本のあと）投入する。先に走ったほうが鍵を取り、相方を取り消す。
# 使い方: bash server_sde/_handoff_scripts_20261020/submit_abl_either.sh scratch   （scratch か noheight）
set -u
TAG="${1:?scratch か noheight}"
case "$TAG" in scratch|noheight) ;; *) echo "scratch か noheight"; exit 1;; esac
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && set -u
R=~/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_sde_w3_abl_$TAG
[ ! -e \"\$R\" ] || { echo \"run が既にある（\$R）→ 学習が始まっている。投入しない\"; exit 4; }
for J in \$(squeue -u \$USER -h -n v2ftrain_abl_${TAG}_pro,v2ftrain_abl_${TAG}_a100,v2fcausal_abl_${TAG}_pro,v2fcausal_abl_${TAG}_a100 -o %i); do echo \"既に両候補の列がある（\$J）→ 投入しない\"; exit 4; done
for J in \$(squeue -u \$USER -h -n v2ftrain_abl_${TAG},v2fcausal_abl_${TAG} -o '%i %t' | awk '\$2==\"PD\"{print \$1}'); do scancel \$J && echo \"古い列 \$J（待ち）を取り消し\"; done
sleep 2
NQ=\$(squeue -u \$USER -h -q low -o %i | wc -l); [ \$((NQ + 4)) -le 8 ] || { echo \"low の投入上限 8 に当たる（今 \$NQ 本 ＋ 4）→ 先に枠を空ける（s1 の列など）\"; exit 6; }
TP=\$(EITHER_TAG=$TAG EITHER_STAGE=train sbatch --parsable -p pro_6000 --gres=gpu:pro_6000:1 --qos=low --requeue -J v2ftrain_abl_${TAG}_pro server_sde/v2final_abl_either.sbatch) || exit 5
TA=\$(EITHER_TAG=$TAG EITHER_STAGE=train sbatch --parsable -p a100 --gres=gpu:a100:1 --qos=low --requeue -J v2ftrain_abl_${TAG}_a100 server_sde/v2final_abl_either.sbatch) || exit 5
CP=\$(EITHER_TAG=$TAG EITHER_STAGE=causal sbatch --parsable -p pro_6000 --gres=gpu:pro_6000:1 --qos=low --requeue --dependency=afterany:\$TP:\$TA -J v2fcausal_abl_${TAG}_pro server_sde/v2final_abl_either.sbatch) || exit 5
CA=\$(EITHER_TAG=$TAG EITHER_STAGE=causal sbatch --parsable -p a100 --gres=gpu:a100:1 --qos=low --requeue --dependency=afterany:\$TP:\$TA -J v2fcausal_abl_${TAG}_a100 server_sde/v2final_abl_either.sbatch) || exit 5
echo \"$TAG: 学習 pro \$TP / a100 \$TA → 因果 ft pro \$CP / a100 \$CA（afterany）\"
squeue -u \$USER -h -o '%i %t %P %q %j %r' | sort -n" 2>&1 | sed -E 's/s[0-9]{7}/<user>/g'
