#!/bin/bash
#SBATCH -J w3adopt
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o w3adopt-%j.out
# v12本番候補をw3（W_DIST=3.0・G1-G4達成）へ切替えるための再採点素材づくり:
# ① w3ckptで評価拡充セット1,500本(fold10-13)を推論→連結
# ② w3のval予測でcore通知層v3.2採点（音声依存のためサーバ実行）
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
cd $PS
export SDE_DIST_MODE=log

ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_evalinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_w3_eval \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_eval/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "eval_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out, sum(1 for _ in open(out)))
EOF

# ② core通知（v3.2凍結ルール・計上穴修正版）
cd $E2E
PYTHONPATH=scripts:src $PS/.venv/bin/python scripts/_v12_notify_server.py \
  $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_val/val_all.csv \
  $E2E/out/v12_notify_w3
echo W3_ADOPT_DONE
