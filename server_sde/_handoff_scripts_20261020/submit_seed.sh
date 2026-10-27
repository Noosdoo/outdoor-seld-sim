#!/bin/bash
# 追加の種（#71）の学習 → 因果 ft を列へ（主 seed の因果 ft 7019 が走り始めてから・本人に一言は言う）。
# 使い方: bash server_sde/_handoff_scripts_20261020/submit_seed.sh s1   （s1 か s2）
# Astra RN20（2026-10-20）: ジョブ名に種の札を付け（v2ftrain_s1／v2fcausal_s1）、同じ札のジョブが列にあれば二重投入しない。
#   因果 ft の推論の出力名は infer_v2finalft_s1_e<NNN>_selfcausal（v2final_causal.sbatch の SEED_SUF・RN20-02）。
set -u
TAG="${1:?s1 か s2}"
case "$TAG" in s1|s2) ;; *) echo "s1 か s2"; exit 1;; esac
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && R=~/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_sde_w3_$TAG; [ ! -e \"\$R\" ] || { echo \"run の置き場が既にある（黙って再開しない）: \$R\"; exit 1; }; Q=\$(squeue -u \$USER -h -n v2ftrain_$TAG,v2fcausal_$TAG -o '%i %j %t'); [ -z \"\$Q\" ] || { echo \"同じ札のジョブが列にある（二重投入しない）: \$Q\"; exit 1; }; grep -q 'OUTPFX=infer_v2finalft\${SEED_SUF}' server_sde/v2final_causal.sbatch || { echo 'サーバの v2final_causal.sbatch が RN20-02 の版でない（配備してから）'; exit 1; }; T=\$(V2FINAL_SEED_TAG=$TAG sbatch --parsable -J v2ftrain_$TAG -t 36:00:00 server_sde/v2final_train.sbatch) || exit 2; echo \"train $TAG job \$T\"; C=\$(V2FINAL_SEED_TAG=$TAG sbatch --parsable -J v2fcausal_$TAG --dependency=afterok:\$T -t 48:00:00 server_sde/v2final_causal.sbatch) || exit 2; echo \"causal $TAG job \$C (afterok:\$T)\"; squeue -u \$USER -o '%i %j %t %l %r'"
