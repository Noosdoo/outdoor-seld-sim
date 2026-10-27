#!/bin/bash
# アブレーション（変更票 #74・本人 2026-10-21 01:00「1〜4 で」）の学習し直し 2 本: scratch（事前学習なし）／noheight（高さの入力なし）。
# 主 seed・低優先（QoS low・割り込まれたら last.ckpt から再開）で基準学習 → 因果 ft を afterok で連結。同じ札のジョブが列にあれば二重投入しない。
# 使い方: bash server_sde/_handoff_scripts_20261020/submit_abl.sh scratch   （scratch か noheight）
set -u
TAG="${1:?scratch か noheight}"
case "$TAG" in scratch|noheight) ;; *) echo "scratch か noheight"; exit 1;; esac
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && R=~/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_sde_w3_abl_$TAG; [ ! -e \"\$R\" ] || { echo \"run の置き場が既にある（黙って再開しない）: \$R\"; exit 1; }; Q=\$(squeue -u \$USER -h -n v2ftrain_abl_$TAG,v2fcausal_abl_$TAG -o '%i %j %t'); [ -z \"\$Q\" ] || { echo \"同じ札のジョブが列にある: \$Q\"; exit 1; }; grep -q 'V2FINAL_ABL_TAG' server_sde/v2final_train.sbatch && grep -q 'V2FINAL_ABL_TAG' server_sde/v2final_causal.sbatch || { echo 'サーバの台本が #74 の版でない（配備してから）'; exit 1; }; T=\$(V2FINAL_ABL_TAG=$TAG sbatch --parsable --qos=low -J v2ftrain_abl_$TAG -t 36:00:00 server_sde/v2final_train.sbatch) || exit 2; echo \"train abl_$TAG job \$T\"; C=\$(V2FINAL_ABL_TAG=$TAG sbatch --parsable --qos=low -J v2fcausal_abl_$TAG --dependency=afterok:\$T -t 48:00:00 server_sde/v2final_causal.sbatch) || exit 2; echo \"causal abl_$TAG job \$C (afterok:\$T)\"; squeue -u \$USER -o '%i %j %t %l %q %r'"
