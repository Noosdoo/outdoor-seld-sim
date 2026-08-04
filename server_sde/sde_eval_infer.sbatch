#!/bin/bash
#SBATCH -J sdeevinf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o sdeevinf-%j.out
# 評価専用3,246本(v11_eval)に SDE run3 (epoch_084) の距離付き推論 + 連結
set -ex
cd ~/research/PSELDNet/PSELDNets
LOGROOT=$HOME/PSELDNets_logs/outdoor_siren_v11/runs

# 注: 評価データセットのサーバー名は outdoor_siren_v11e（_evalは8/3の失敗残骸=index0バイト）
# task_name=outdoor_siren_v11e のため出力は PSELDNets_logs/outdoor_siren_v11e/runs/ 配下
.venv/bin/python _infer_sde.py \
  experiment=outdoor_siren_v11e \
  mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v11_sde_run3_eval \
  paths.log_dir=$HOME/PSELDNets_logs

python3 - "$HOME/PSELDNets_logs/outdoor_siren_v11e/runs/infer_v11_sde_run3_eval/submissions" <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "eval_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(f"{clip},{line}\n")
print("wrote", out)
EOF
echo SDE_EVAL_INFER_DONE
