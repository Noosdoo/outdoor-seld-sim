#!/bin/bash
#SBATCH -J fdemo
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH -o fdemo-%j.out
# 家のデモ（2026-10-07・3 本・調整用）を、開発用の採用候補の重み（dev2cont5_D21 epoch_029・D21 腕）で因果推論する（本人「実録で運用できているか確かめたい」）。
# 入力: $E2E/out/field_demo/clips/eval_field_clip*.flac（step19b の 4ch 24 kHz 10 s）と高さ表 out/field_demo/height_table.csv（mic_z 1.5）
# 環境は開発用の自分の val の推論（v2final_dev_causal_r2.sh・d21_control_ft.sbatch）と同じ: HEIGHT_COND=1・SDE_W_DIST=3.0・log・KMIN 40。
# 出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_field_demo_D21e029_causal_h0703/val_all_causal.csv（7 列・h0703 = 入り 0.7／保持 0.3。旧 infer_field_demo_D21e029_causal は 0.5/0.5）
#   sbatch server_sde/field_demo_infer.sbatch   （CAUSAL_CKPT・OUTTAG で差し替え可）
set -ex
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
export PS E2E PYTHONIOENCODING=utf-8
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export CAUSAL_SED_TH=${CAUSAL_SED_TH:-0.7} CAUSAL_SED_STAY=${CAUSAL_SED_STAY:-0.3}   # 本番・開発比較と同じ入り 0.7／保持 0.3（2026-10-08 Astra D08-04。最初の実行 5733 は既定 0.5/0.5 だった）
export HEIGHT_COND=1 HEIGHT_TABLE=$E2E/out/field_demo/height_table.csv HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_FOA=$E2E/out/field_demo/clips
export CAUSAL_PREFIX=eval_field_
export CAUSAL_CKPT=${CAUSAL_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_dev2cont5_D21_e30/checkpoints/epoch_029.ckpt}
export CAUSAL_OUT=${OUTTAG:-infer_field_demo_D21e029_causal_h0703}
unset CAUSAL_GLOB CAUSAL_LIMIT
ls $CAUSAL_FOA | head; ls -l $CAUSAL_CKPT; cat $HEIGHT_TABLE
.venv/bin/python $E2E/server_sde/_causal_infer_v17.py
guard_verify || exit 1
ls -l $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT/
echo FIELD_DEMO_INFER_DONE
