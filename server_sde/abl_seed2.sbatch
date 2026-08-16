#!/bin/bash
#SBATCH -J ablseed2
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o ablseed2-%j.out
# 【ラン揺れフロア】基準(v12 w3)と**完全に同じ条件・seedだけ変えて**もう1本学習する。
# ablationの arm 間差が「本物の差」か「ラン間のブレ」かを判定する物差しになる（§9.3）。
# これが無いと、どのarmの差も解釈できない。
#
# 使い方: sbatch server_sde/abl_seed2.sbatch
#
# 【基準を触らない設計】
#  - 学習データは**基準をそのまま読む**（datasets_v12 / _hdf5）。**書き込みは一切しない**
#    （前処理は再実行しない＝キャッシュを作り直さない）
#  - 出力先の実験名は outdoor_siren_v12_sde_w3_seed2 で、基準run（..._sde_w3）とは別物
#  - 冒頭と末尾でガードを回し、基準3データセット・基準ルート・基準キャッシュ・基準run・
#    初期重みが1バイトでも変わっていればジョブを失敗させる
set -ex

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
EXP=outdoor_siren_v12_sde_w3_seed2
SEED="${ABL_SEED:-20260816}"          # 既定 seed（train.yaml の既定 2024 とは別の値）
[ "$EXP" != "outdoor_siren_v12_sde_w3" ] || { echo "基準runと同名"; exit 1; }

cd $PS
# ---- 初期重み・損失重み・距離表現・epoch数は基準(job315)と完全同一。seedだけ変える ----
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log

.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  seed=$SEED \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5 \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT

ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
echo "using ckpt: $ck"

# 基準と同じフル物理valで推論（armのフル物理valと同じ土俵）
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_${EXP}_fullval \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_${EXP}_fullval/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "val_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out, sum(1 for _ in open(out)))
EOF

guard_verify || exit 1
echo ABL_SEED2_DONE
