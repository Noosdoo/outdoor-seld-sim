#!/bin/bash
#SBATCH -J bBprep
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 24:00:00
#SBATCH -o bBprep-%j.out
# 相手 B（変更票 #73）の本番の準備: v29 の学習 split 全部（train 49,682 本 = fold3）と val（12,420 本 = fold4）を公式の層に並べ、特徴 → 標準化（train だけで fit）→ ラベル → 単位の往復検査。
#   置き場（名前に 2024 が要る・公式が分割を名前で決める）: B_dataset2024_v29（層・シンボリックリンク）・B_feat_v29（特徴 約 56 GB ×2〔標準化前後〕・ラベル）。
#   GPU は使わない（割り当てられた CPU で特徴抽出・並列数は nproc）。GPU の最小の切れ端（1g.10gb）で枠を取る。使い方: sbatch --qos=low server_sde/baseline_B_prepare.sbatch
#   注意: -c（CPU 数）は指定できない（このクラスタは "CPU, memory and node layout cannot be specified" で拒否・04:2x）
# 本体 §3-6（2026-10-26）: 置き場と元の束は環境で替えられる（未設定なら v29 のまま）。v30: B_SRC_BUNDLE=$E2E/out/dataset_outdoor_siren_v2final_v30・
#   B_DATASET_DIR=…/B_dataset2024_v30・B_FEAT_DIR=…/B_feat_v30c<_g倍率>・B_INPUT_GAIN_DB=<決めた倍率>。距離を対数にする置き場は
#   B_DIST_ENCODING=log1p B_LINK_FEAT_FROM=<同じ倍率の linear の B_FEAT_DIR>（特徴は借りてラベルだけ作る）。
#   空き容量が 150 GB 未満なら始めない（途中で止まると特徴が中途半端に残る）。再実行は既にある特徴を飛ばす（_baseline_B_prepare.py step_feat）。
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $E2E
export E2E
export B_SRC_BUNDLE=${B_SRC_BUNDLE:-$E2E/out/dataset_outdoor_siren_v2final_v29}
export B_DATASET_DIR=${B_DATASET_DIR:-$HOME/research/_external/B_dataset2024_v29}
export B_FEAT_DIR=${B_FEAT_DIR:-$HOME/research/_external/B_feat_v29c}     # c = top_db なしの特徴（Astra RI73-02・2026-10-21）。旧 B_feat_v29（top_db=80）は使わない
export B_MODEL_DIR=${B_MODEL_DIR:-$HOME/research/_external/B_models_v29}
export B_OUT_DIR=${B_OUT_DIR:-$HOME/research/_external/B_out_v29}
echo "B: src=$B_SRC_BUNDLE dataset=$B_DATASET_DIR feat=$B_FEAT_DIR gain=${B_INPUT_GAIN_DB:-未設定} dist=${B_DIST_ENCODING:-linear} link=${B_LINK_FEAT_FROM:-なし}"
mkdir -p $B_DATASET_DIR $B_FEAT_DIR $B_MODEL_DIR $B_OUT_DIR
[ -d ~/research/_external/DCASE2024_seld_baseline ] || { echo "公式コードが無い"; exit 3; }
cd ~/research/_external/DCASE2024_seld_baseline && echo "official commit $(git rev-parse --short HEAD)" && git status --porcelain | head -n 3; cd $E2E
AVAIL_GB=$(df -BG --output=avail $HOME | tail -n 1 | tr -dc '0-9')
echo "空き $AVAIL_GB GB"
[ "$AVAIL_GB" -ge 150 ] || { echo "空き容量が 150 GB 未満 → 始めない"; exit 3; }
N_TRAIN=$($PY -c "import csv;print(sum(1 for r in csv.DictReader(open('$B_SRC_BUNDLE/plan/assignment_v2final.csv',encoding='utf-8')) if r['split']=='train'))")
N_VAL=$($PY -c "import csv;print(sum(1 for r in csv.DictReader(open('$B_SRC_BUNDLE/plan/assignment_v2final.csv',encoding='utf-8')) if r['split']=='val'))")
echo "plan: train $N_TRAIN・val $N_VAL（#73: train 49,682・val 12,420 のはず）"
[ "$N_TRAIN" = 49682 ] && [ "$N_VAL" = 12420 ] || { echo "本数が #73 と違う → 止める"; exit 3; }
NCPU=$(nproc); echo "CPU $NCPU 本（SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?}）"
$PY $E2E/server_sde/_baseline_B_prepare.py --step all --jobs ${B_JOBS:-$NCPU} ${B_LINK_FEAT_FROM:+--link-feat-from "$B_LINK_FEAT_FROM"}
echo "--- 置き場の大きさ"; du -sh $B_FEAT_DIR/* 2>/dev/null | tail -n 6
sha256sum $B_DATASET_DIR/B_prepare_record.json $E2E/server_sde/_baseline_B_common.py $E2E/server_sde/_baseline_B_prepare.py | cut -c1-16,65-
echo BASELINE_B_PREPARE_DONE
