#!/bin/bash
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
# 前処理ジョブ（v11 + v10評価セット）。Colabノートブック セル8後半の移植。冪等。
# 投入例:
#   sbatch -J v11prep -p a100_1g --gres=gpu:1g.10gb:1 --qos=low \
#       ~/research/outdoor_seld_e2e/server/preproc_v11.sbatch
set -euo pipefail

cd "$HOME/research/PSELDNet/PSELDNets"
source .venv/bin/activate

echo "ジョブID: $SLURM_JOB_ID / ノード: $(hostname)"

for ds in outdoor_siren_v11 outdoor_siren_v10; do
    idx="_hdf5/data/24000fs/wav/dev/${ds}_10sChunklen_10sHoplen_train.csv"
    if [[ -f "$idx" ]]; then
        echo "${ds}: 前処理済み"
    else
        python _preproc_emptyok.py "dataset=${ds}"
    fi
    python - "$ds" "$idx" <<'PY'
import sys
import pandas as pd
print(sys.argv[1], 'index rows:', len(pd.read_csv(sys.argv[2], header=None)))
PY
done
echo "=== preproc 完了 ==="
