#!/bin/bash
#SBATCH -J dbike
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 00:30:00
#SBATCH -o dbike-%j.out
# 家で録った 10/7 の自転車（デモ 3 本・走行音の測定 10 本＝調整用・評価には使わない〔10/7 の決まり ⑥〕）を、今の主のモデルで因果推論する
# 手応えの確かめ（2026-10-25 Opus・本人 20:25「測定の 10 本も足しましょう・学内の実録には絶対に触れない」）。学内の実録（raw/・評価用）には触らない。
# 入力: $D/clips/eval_field_clip*.flac（scripts/demo_bike_check_prepare.py の出力を scp したもの・13 本）と $D/height_table.csv（mic_z 1.5）
# 条件は確定評価の提案法と同じ（server_sde/_v2final_members.sh の member_v17: 高さあり・入り 0.7／保持 0.3・ASSOC 20・datasets_v27・
#   SDE_W_DIST 3.0・log・KMIN 40・REPEAT 2・SEED 0）。重み = v29 の因果 ft の epoch_139（選択器 2.1 の選んだもの・sha256 88fbc9fd…）。
# 出力: ~/PSELDNets_logs/outdoor_siren_v12/runs/$OUTTAG/val_all_causal.csv（7 列）。採点は手元の scripts/demo_bike_check_score.py。
#   sbatch server_sde/demo_bike_check_infer.sbatch   （CAUSAL_CKPT・OUTTAG・DEMO_DIR で差し替え可＝学習し直したあとに同じ 13 本で比べる）
set -ex
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
D=${DEMO_DIR:-$E2E/out/_demo_bike_check}
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
export PS E2E PYTHONIOENCODING=utf-8
export CAUSAL_CKPT=${CAUSAL_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_causal_ft/checkpoints/epoch_139.ckpt}
export CAUSAL_OUT=${OUTTAG:-infer_demo_bike_check_v29e139}
export HEIGHT_COND=1 HEIGHT_TABLE=$D/height_table.csv HEIGHT_DEFAULT=1.5 HEIGHT_JITTER=0 HEIGHT_OFFSET=0
export CAUSAL_SED_TH=${CAUSAL_SED_TH:-0.7} CAUSAL_SED_STAY=${CAUSAL_SED_STAY:-0.3} CAUSAL_SED_ASSOC=20 CAUSAL_DSDIR=$PS/datasets_v27   # 閾値は切り分けのときだけ下げる（既定 = 提案法）
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0 CAUSAL_DATASET=outdoor_siren_v12
export CAUSAL_FOA=$D/clips CAUSAL_PREFIX=${DEMO_PREFIX:-eval_field_} CAUSAL_LIMIT=0 CAUSAL_EXPECTED_CLIPS=${N_CLIPS:-13}   # 切り分け 3（合成の自転車＋家の背景）は N_CLIPS=60・DEMO_DIR を差し替え
#   2026-10-26: 開発用の学内の本（台帳で使わないの本・scripts/demo_field_unused_prepare.py）は DEMO_PREFIX=tune_field_（評価の eval_field_ と別の名前）。未設定なら今のまま eval_field_
unset CAUSAL_GLOB
[ ! -e $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT ] || { echo "出力の置き場がもうある（$CAUSAL_OUT）: 名前を変える"; exit 4; }
sha256sum $CAUSAL_CKPT | cut -c1-16
ls $CAUSAL_FOA; cat $HEIGHT_TABLE
.venv/bin/python $E2E/server_sde/_causal_infer_v17.py
guard_verify || exit 1
ls -l $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$CAUSAL_OUT/
echo DEMO_BIKE_CHECK_DONE
