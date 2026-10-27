#!/bin/bash
# 主 seed の学習 → 因果 ft を連結して列へ（本人 2026-10-19 21:47 了承・#71・検査 A〜I が通ってから実行する）。
# -t はコマンド行で上書き（ヘッダの 6 h／12 h では足りない・台本も台帳も変えない）。保守の予約（10/20 13:30〜）に掛かるので、始まるのは保守明け（自動）。
cd ~/research/outdoor_seld_e2e || exit 1
test -z "$V2FINAL_SEED_TAG" || { echo "主 seed の投入なのに V2FINAL_SEED_TAG がある"; exit 1; }
[ -e ~/research/PSELDNet/PSELDNets/datasets_v29/outdoor_siren_v12 ] || { echo "datasets_v29 が無い"; exit 1; }
ls ~/research/PSELDNet/PSELDNets/_hdf5_v29/label/adpit_dist/dev/*.h5 > /dev/null || { echo "HDF5 が無い"; exit 1; }
R=~/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_sde_w3
[ ! -e "$R" ] || { echo "run の置き場が既にある（黙って再開しない）: $R"; ls $R; exit 1; }
T=$(sbatch --parsable -t 36:00:00 server_sde/v2final_train.sbatch) || exit 2
echo "train job $T"
C=$(sbatch --parsable --dependency=afterok:$T -t 48:00:00 server_sde/v2final_causal.sbatch) || exit 2
echo "causal job $C (afterok:$T)"
squeue -u $USER -o "%i %j %t %l %S %r"
