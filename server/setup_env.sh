#!/bin/bash
# IS計算サーバーのログインノードで実行: bash setup_env.sh
# PSELDNets 学習用 venv を uv で構築する（Colabノートブック セル4 の移植）
# torch は 2.8.0+cu128（cu132 索引には torchaudio の x86 wheel が無いため。
# cu128 は A100 sm_80 / PRO 6000 sm_120 両対応）。
# lightning 等は v11 Colab ノートブックのピン留めと同一。
set -euo pipefail

REPO="$HOME/research/PSELDNet/PSELDNets"
cd "$REPO"

command -v uv >/dev/null || { echo "ERROR: uv が見つかりません（ログインノードで実行していますか）"; exit 1; }

if [[ ! -d .venv ]]; then
    uv venv .venv --python 3.12
fi
source .venv/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"

uv pip install \
    lightning==2.2.1 \
    hydra-core==1.3.2 \
    hydra-colorlog==1.2.0 \
    hydra-joblib-launcher==1.2.0 \
    torchmetrics==1.3.1 \
    omegaconf==2.3.0 \
    "pandas<3" \
    h5py scipy rich tqdm librosa soundfile tensorboard

python - <<'PY'
import torch, torchaudio, lightning, torchmetrics, librosa, numpy, pandas
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("lightning", lightning.__version__, "| torchmetrics", torchmetrics.__version__)
print("numpy", numpy.__version__, "| pandas", pandas.__version__, "| librosa", librosa.__version__)
PY
echo "=== setup_env.sh 完了 ==="
