#!/bin/bash
#SBATCH -J v2fconf
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH -o v2fconf-%j.out
# 確定評価 v2（fold20 core 1,200 本・1 回だけ・介入なし）: 選んだ因果 ft ckpt（V2FINAL_CKPT）で fold20 を因果推論 → conf_all.csv → v1 と同じ採点器
# （_v12_conf_notify_server.py: v3.3/v3.4 ＋ B2 個票）で v1 と直接比較。fold20 は v1 の確定評価セット（outdoor_siren_v12_conf・マイク 1.5 m）を
# そのまま使う（生成しない）。装着高さ入力は表なし・既定 1.5 m。停止規則: 本ジョブは 1 回だけ。結果が悪くても差し替えない。
# 使い方: V2FINAL_CKPT=<選んだ ckpt の絶対パス> sbatch server_sde/v2final_conf.sbatch
# 乾式: V2FINAL_DRYRUN=1 bash server_sde/v2final_conf.sbatch → 入力の契約（conf の 1,800 本）を確かめて推論の前で止まる（GPU 不要・fold20 の本推論は使わない）
#
# 2026-09-14（Fable 依頼 2-9・独立監査 2026-09-13 R1）: 音声の入力先 CAUSAL_FOA を渡しておらず、推論スクリプトの既定
# outdoor_siren_v12/foa/fold20_*.flac（サーバ実測 0 本）を読んでいた。fold20 の音声は outdoor_siren_v12_conf/foa（1,800 本）にある。
# 入力の契約（FOA・PREFIX・GLOB・LIMIT・EXPECTED・RUNS_DIR）をこの台本で明示し、継承した環境変数が混ざらないよう先に全部消す。
set -exo pipefail
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
DRY=${V2FINAL_DRYRUN:-0}
cd $E2E
# --- 継承した CAUSAL_* を全部消す（--export=ALL や対話シェルの残りが入力先を変えないように）
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
source $E2E/server_sde/_baseline_guard.sh
guard_record
PARAMS=$E2E/md/design/v2final_params.json
OUTNAME=infer_v2final_conf
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2      # 来歴 schema 2 の置き場（v18 以降と同じ）
LEGACY_RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CONF=$PS/datasets_v12/outdoor_siren_v12_conf
[ -d "$CONF/foa" ] || { echo "fold20 の確定評価セットが無い（$CONF/foa。v12_conf_all.sbatch で作られたもの）"; exit 1; }
if [ "$DRY" != 1 ]; then
  $PY -c "import json,sys;P=json.load(open('$PARAMS'));sys.exit(0 if P.get('status')=='fixed' else 3)" || { echo "params が fixed でない"; exit 3; }
  [ -n "$V2FINAL_CKPT" ] && [ -f "$V2FINAL_CKPT" ] || { echo "V2FINAL_CKPT（選んだ因果 ft の ckpt）を渡す"; exit 3; }
  for d in "$RUNS/$OUTNAME" "$LEGACY_RUNS/$OUTNAME"; do
    [ -e "$d" ] && { echo "既に確定評価を 1 回実行済み（$d）。停止規則により再実行しない"; exit 4; }
  done
  sha256sum $V2FINAL_CKPT $PARAMS server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/height_patch.py server_sde/causal_patch.py \
    server_sde/hysteresis_patch.py scripts/_v12_conf_notify_server.py scripts/_gt_as_pred.py out/predictions_v12_conf/conf_all.csv | tee v2final_conf_sha256_job${SLURM_JOB_ID:-dry}.txt
  git rev-parse HEAD | tee -a v2final_conf_sha256_job${SLURM_JOB_ID:-dry}.txt
fi

cd $PS
# ジョブ専用ディレクトリに推論の入口と来歴モジュールをそろえる（監査 A7: 共有パスの写しを使わない）
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p $JOBDIR
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
# 検出のヒステリシス（変更票 #34・2026-09-16 本人「①採用」）: 入り 0.7／保持 0.3。確定評価も同じ値（両設定で共通）
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export PS E2E
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export HEIGHT_COND=1
export HEIGHT_TABLE=""          # fold20 は全クリップ 1.5 m
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
# --- 入力の契約（ここが確定評価の入力を決める。推論スクリプトの既定に頼らない）
export CAUSAL_CKPT=${V2FINAL_CKPT:-/dev/null}
export CAUSAL_DATASET=outdoor_siren_v12
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_FOA=$CONF/foa                 # fold20 の音声はここ（outdoor_siren_v12/foa には fold20_ が 0 本）
export CAUSAL_PREFIX=eval_syn_             # 2026-09-22 改名（旧 fold20_）
export CAUSAL_GLOB="eval_syn_*.flac"
export CAUSAL_LIMIT=0                       # 全部（試走用の間引きは使わない）
export CAUSAL_EXPECTED_CLIPS=1800           # core 1,200 ＋ 拡張 600。推論スクリプトも同じ数で止まる
export CAUSAL_RUNS_DIR=$RUNS
export CAUSAL_OUT=$OUTNAME
# --- 事前点検: 推論スクリプトと同じ規則で対象 clip を解決し、1,800 本でなければ止まる
$PY - <<'EOF'
import os, sys
from pathlib import Path
foa = Path(os.environ["CAUSAL_FOA"]); glob = os.environ["CAUSAL_GLOB"]; limit = int(os.environ["CAUSAL_LIMIT"]); expected = int(os.environ["CAUSAL_EXPECTED_CLIPS"])
clips = sorted(p.name for p in foa.glob(glob))
if limit: clips = clips[:limit]
print(f"[preflight] foa={foa} glob={glob} limit={limit} n={len(clips)} first={clips[:1]} last={clips[-1:]}")
bad = [c for c in clips if not c.startswith(os.environ["CAUSAL_PREFIX"])]
if len(clips) != expected or bad:
    print(f"[preflight] NG: {len(clips)} 本（期待 {expected}）・prefix 違い {len(bad)} 本"); sys.exit(3)
print("[preflight] OK")
EOF
if [ "$DRY" = 1 ]; then echo "V2FINAL_CONF_DRYRUN_OK（推論は行わない）"; rm -rf "$JOBDIR"; exit 0; fi

.venv/bin/python $JOBDIR/_causal_infer_v17.py
# 来歴の照合（本数・行数・sha256・identity）を通った連結だけを採点に渡す。conf_all.csv は照合済みの連結の写し
.venv/bin/python $JOBDIR/_prediction_integrity.py "$RUNS/$OUTNAME" 1800
[ -f "$RUNS/$OUTNAME/val_all_causal.done.json" ] || { echo "完了印が無い"; exit 5; }
cp "$RUNS/$OUTNAME/val_all_causal.csv" "$RUNS/$OUTNAME/conf_all.csv"
echo "wrote $RUNS/$OUTNAME/conf_all.csv $(wc -l < "$RUNS/$OUTNAME/conf_all.csv")"
cd $E2E
PYTHONPATH=scripts:src $PY scripts/_v12_conf_notify_server.py \
  "$RUNS/$OUTNAME/conf_all.csv" \
  $E2E/out/v2final_conf_notify_v33 $E2E/out/v2final_conf_notify_v34
# 2026-09-28（変更票 #41・凍結文書 改訂 r5・監査 Astra 9/28 R02）: 上の v3.3/v3.4 は v1 との直接比較用。事前登録の採点器（統一採点器 = v4.3 ＋ 警告音 v9c・
# 方位帰属・1 対 1・全 GT 分母）と旧 hold の併記、併記枠（知覚層 _perception_eval・閾値なし _notify_threshold_sweep）を fold20 core 1,200 本にも掛ける。
# 出力は out/v2final_conf_unified/ に保存し、sha256 を第 2 段階の実行証跡（凍結文書 §11）に写す。検査 = scripts/test_v2final_conf_scoring.py（開発入力で同じ 4 本の命令が通る）
CP=out/dataset_outdoor_siren_v12_conf/plan/assignment_v12conf.csv     # split=eval・core = clip 1〜1200
CG=out/dataset_outdoor_siren_v12_conf/metadata_dist
CU=out/v2final_conf_unified; mkdir -p $CU
CPRED="$RUNS/$OUTNAME/conf_all.csv"
# 2026-10-02（変更票 #45・凍結文書 改訂 r6）: モデル × 規則の 4 セル。(v1, 新規則) = v1 の fold20 予測（8 月の確定評価の元・旧名 fold20_room1_mix）を同じ統一採点器に通す。
# 使う前に、core に絞った行が 8 月の採点入力 out/v12_conf_notify_v33/conf_core.csv と 1 行も違わないことを確かめる（違えば止める = 別の予測を v1 と呼ばない）。
V1PRED=out/predictions_v12_conf/conf_all.csv
V1CORE=out/v12_conf_notify_v33/conf_core.csv
[ -f "$V1PRED" ] && [ -f "$V1CORE" ] || { echo "v1 の fold20 予測か 8 月の採点入力が無い（$V1PRED / $V1CORE）"; exit 5; }
grep -E "^(fold20_room1_mix|eval_syn_clip)0*([1-9][0-9]{0,2}|1[01][0-9]{2}|1200)," "$V1PRED" | cmp -s - "$V1CORE" || { echo "v1 の予測を core に絞った行が 8 月の conf_core.csv と一致しない。中止"; exit 5; }
echo "v1 の予測 = 8 月の採点入力と一致（$(wc -l < $V1CORE) 行）"
# オラクル入力（正解をそのまま予測の形に）= モデルに依らない規則の上限の行。統一採点器の 2 規則だけに掛ける（知覚層・閾値なしは自明に満点なので掛けない）
PYTHONPATH=scripts:src $PY scripts/_gt_as_pred.py --plan $CP --split eval --clip-max 1200 --meta $CG --out $CU/gt_as_pred_core.csv || { echo "オラクル入力の生成に失敗"; exit 5; }
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_core_v9c.md --strict --raw --plan $CP --split eval --clip-max 1200 --meta $CG v1=$V1PRED v2final=$CPRED oracle=$CU/gt_as_pred_core.csv > $CU/_log_unified_v9c.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_core_hold.md --warn-rule hold --strict --raw --plan $CP --split eval --clip-max 1200 --meta $CG v1=$V1PRED v2final=$CPRED oracle=$CU/gt_as_pred_core.csv > $CU/_log_unified_hold.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_perception_eval.py $CU/perception_core.md --plan $CP --split eval --clip-max 1200 --meta $CG --json $CU/perception_core.json --strict v1=$V1PRED v2final=$CPRED > $CU/_log_perception.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_notify_threshold_sweep.py $CU/notify_sweep_core.md --plan $CP --split eval --clip-max 1200 --meta $CG --csv $CU/notify_sweep_core.csv --strict v1=$V1PRED v2final=$CPRED > $CU/_log_sweep.txt 2>&1
sha256sum $CU/*.md $CU/*.json $CU/*.csv "$CPRED" "$V1PRED" "$V1CORE" scripts/_gt_as_pred.py scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_threshold_sweep.py | tee $CU/sha256_job${SLURM_JOB_ID:-dry}.txt
guard_verify || exit 1
echo V2FINAL_CONF_DONE
