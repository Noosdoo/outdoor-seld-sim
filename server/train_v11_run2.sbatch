#!/bin/bash
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
# v11 学習 run2（IS計算サーバー）。Colabノートブック セル10の移植。
# --requeue はクラスタ既定の中断時自動再実行を明示したもの（Sol監査#6）。
# ハイパーパラメータは run1（Colab T4）と同一（batch_size=8, 100ep, lr3e-4）＝環境間再現性の検証を兼ねる。
# low QoS で中断→自動再実行されても last.ckpt から自動再開する。
# 投入例（run2実績= pro_6000。a100なら -p a100 --gres=gpu:a100:1）:
#   sbatch -J v11run2 -p pro_6000 --gres=gpu:pro_6000:1 --qos=low \
#       ~/research/outdoor_seld_e2e/server/train_v11_run2.sbatch
set -euo pipefail

cd "$HOME/research/PSELDNet/PSELDNets"
source .venv/bin/activate

DATASET=outdoor_siren_v11
EXP_NAME=outdoor_siren_v11_run2          # ★データや条件を変えるときは必ず変更
LOGDIR="$HOME/PSELDNets_logs"

LAST="$LOGDIR/$DATASET/runs/$EXP_NAME/checkpoints/last.ckpt"
RESUME=""
if [[ -f "$LAST" ]]; then
    RESUME="ckpt_path=$LAST"
fi
echo "ジョブID: $SLURM_JOB_ID / ノード: $(hostname) / GPU: $(nvidia-smi -L | head -1)"
echo "resume: ${RESUME:-（新規run）}"

python _train_emptyok.py "experiment=${DATASET}" \
    "experiment_name=${EXP_NAME}" \
    "paths.log_dir=${LOGDIR}" \
    ${RESUME}
echo "=== 学習完了 ==="
