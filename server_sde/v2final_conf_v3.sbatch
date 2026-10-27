#!/bin/bash
#SBATCH -J v2fconf3
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 04:00:00
#SBATCH -o v2fconf3-%j.out
# 確定評価 v2・評価セット eval v3（凍結 r11・変更票 #65・2026-10-13 本人「確定評価も最新版に合わせて作り直す」「基本使うものは全て最新版に」）。
#   評価セット = out/dataset_outdoor_siren_v2final_eval_v3（6,000 本・封印 SEALED.txt・CONTRACT_eval_v3.json）。開くのはこのジョブの 1 回だけ（停止規則: 再実行しない・結果が悪くても差し替えない）。
#   v2 = 選んだ因果 ft の ckpt（V2FINAL_CKPT）・装着高さ入力あり（高さの表 = 評価セットの plan の mic_z・U(1.4, 2.1)）・9 クラス（datasets_v27）。
#   v1 = 8 月の確定評価のモデル（outdoor_siren_v12_sde_w3 epoch_099・8 クラス・高さ入力なし → 1.5 m 固定・datasets_v12）を**同じ最新の推論（因果・ヒステリシス 0.7／0.3）と同じ規則**で測り直す（4 セルの (v1, 新規則)・自転車の列は空）。
#   採点 = 統一採点器（v4.3 ＋ 警告音 v9c・旧 hold 併記）・知覚層・閾値なし。plan と GT は eval v3 のもの（split=eval・6,000 本）。旧 core 1,200 本（fold20）は測らない（本人 15:13）。
#   v3.3/v3.4（_v12_conf_notify_server.py）は core 1〜1200 固定の 8 月の互換採点なので eval v3 には掛けない。
# 使い方: V2FINAL_CKPT=<選んだ ckpt の絶対パス> V2FINAL_SELECT_JSON=<選択器 2.1 の JSON> V2FINAL_DECIDE_JSON=<補正表の採否の JSON> [DIST_TABLE=<表・採用のときだけ>] sbatch server_sde/v2final_conf_v3.sbatch
# 乾式: V2FINAL_DRYRUN=1 bash server_sde/v2final_conf_v3.sbatch（同じ環境変数）→ 実行の契約と封印と入力の契約（6,000 本）を確かめて推論の前で止まる（GPU 不要・予約しない）
# 変更票 #73（比べる相手 B・D）・#74（アブレーション 1〜4）・2026-10-21: 同じ 1 回のジョブで相手とアブレーションも推論し、同じ採点器で採点する（server_sde/_v2final_members.sh の関数）。
#   追加の入力: V2FINAL_BASELINES_JSON（既定 md/design/baselines_selected.json・相手の凍結・中の相手を全部回す）
#              V2FINAL_BASE_LAST_CKPT（既定 runs/outdoor_siren_v2final_v29_sde_w3/checkpoints/last.ckpt・アブレーション 3 = 因果の微調整なし・epoch 99 を確かめる）
#              V2FINAL_ABL_SCRATCH_CKPT／V2FINAL_ABL_SCRATCH_SELECT_JSON・V2FINAL_ABL_NOHEIGHT_CKPT／V2FINAL_ABL_NOHEIGHT_SELECT_JSON（アブレーション 1・2 = 学習し直しの選んだ ckpt と選択器の JSON）
#              V2FINAL_MISSING_OK="名前=理由;名前=理由"（B・D・scratch・noheight・nocausalft・nohyst のうち間に合わなかったものの理由。理由なしで欠けると止まる = 「作れなかった事実を書く」の約束）
#   出力名: ${OUT2}_abl_nocausalft／_abl_nohyst／_abl_scratch／_abl_noheight／_baselineD／_baselineB。採点は v2final と並べた別ファイル（unified_eval_v3_v9c_abl.md・_baselines.md・perception_eval_v3_abl.md・_baselines.md・
#   相手 D と共通の 6 クラスだけに絞った perception_eval_v3_commonD.md〔R73-04〕）。D は距離を出さないので知らせの表には入れない（#73・Q3）。補正表は v2final だけに掛ける（R72-01）= v2final_notable の行で表なしも並べる。
# Astra RN20-03（2026-10-20）: 封印を開く前に、選択器の結果・ckpt の sha・補正表の採否と表の sha を 1 つの契約として照合し（_conf_exec_contract.py）、
#   開封を $RUNS/_conf_v3_opened の mkdir で原子的に 1 回だけ予約する（2 本同時・完了後の再実行は予約で止まる。予約の記録は失敗しても消さない）
set -exo pipefail
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
DRY=${V2FINAL_DRYRUN:-0}
cd $E2E
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
source $E2E/server_sde/_baseline_guard.sh
guard_record
source $E2E/server_sde/_v2final_members.sh     # #73／#74: 相手とアブレーションの推論の関数（member_v17・member_D・member_B・member_absent）
PARAMS=$E2E/md/design/v2final_params.json
EVAL=$E2E/out/dataset_outdoor_siren_v2final_eval_v3
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2
OUT2=infer_v2final_conf_v3            # v2 の出力名
OUT1=infer_v1_conf_v3                 # v1 の出力名
V1CKPT=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
N=$($PY -c "import json;P=json.load(open('$PARAMS'));print(P['EVAL_V3']['n_clips'])")
# ---- 実行の契約（RN20-03）: 封印に触る前に照合する（乾式でも）。選択器の JSON・採否の JSON・ckpt・表（採用のときだけ）
SEL=${V2FINAL_SELECT_JSON:-}; DEC=${V2FINAL_DECIDE_JSON:-}
[ -n "$SEL" ] && [ -f "$SEL" ] && [ -n "$DEC" ] && [ -f "$DEC" ] || { echo "V2FINAL_SELECT_JSON（選択器 2.1 の JSON）と V2FINAL_DECIDE_JSON（補正表の採否の JSON）を渡す"; exit 3; }
[ -n "${V2FINAL_CKPT:-}" ] && [ -f "$V2FINAL_CKPT" ] || { echo "V2FINAL_CKPT（選んだ因果 ft の ckpt）を渡す"; exit 3; }
RES=$RUNS/_conf_v3_opened
# ---- #73／#74 の入力: 相手の凍結 JSON・アブレーションの ckpt（3 = 基準学習の last.ckpt〔epoch 99〕・4 = 主の ckpt・1／2 = 選んだ ckpt ＋ 選択器の JSON）・欠けの理由
BJ=${V2FINAL_BASELINES_JSON:-$E2E/md/design/baselines_selected.json}
[ -f "$BJ" ] || { echo "相手の凍結 JSON が無い（$BJ）"; exit 3; }
NAMES=$($PY server_sde/_baselines_env.py "$BJ" --list) || { echo "相手の凍結 JSON が読めない"; exit 3; }
# Astra RI73-04: 環境変数への変換と必要ファイルの存在を、契約の照合・予約より前に相手ごとに確かめる（乾式でも）
for nm in $NAMES; do
  ENVS=$($PY server_sde/_baselines_env.py "$BJ" --env "$nm") || { echo "相手 $nm の設定が変換できない: $ENVS"; exit 3; }
  ( eval "$ENVS"
    for f in "${CAUSAL_CKPT:-}" "${BASELINE_CLASSMAP:-}" "${BASELINE_TSV170:-}" "${B_MODEL:-}"; do [ -z "$f" ] || [ -f "$f" ] || { echo "相手 $nm の入力が無い: $f"; exit 3; }; done
    [ -z "${B_FEAT_DIR:-}" ] || [ -f "$B_FEAT_DIR/foa_wts" ] || { echo "相手 $nm の標準化の重みが無い: $B_FEAT_DIR/foa_wts"; exit 3; }
    # Astra RF73-02: ライブラリの版を、推論に使う python で読んで凍結と照合（予約より前・乾式でも）
    $PY -c "import os,sys,torch,numpy; w=(os.environ['BASELINE_TORCH_VERSION'],os.environ['BASELINE_NUMPY_VERSION']); h=(torch.__version__,numpy.__version__); print('版', w, h); sys.exit(0 if w==h else 3)" || { echo "相手 $nm: torch／numpy の版が凍結と違う"; exit 3; }
    echo "相手 $nm: 設定と入力 OK（入り ${CAUSAL_SED_TH}・保持 ${CAUSAL_SED_STAY}・対応幅 ${CAUSAL_SED_ASSOC}・ゲイン ${BASELINE_GAIN_DB:-0}）" ) || exit 3
done
BASE_LAST=${V2FINAL_BASE_LAST_CKPT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_sde_w3/checkpoints/last.ckpt}
ABL_ARGS=(--ablation nohyst "$V2FINAL_CKPT")
if [ -f "$BASE_LAST" ]; then
  CKE=$($PY -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$BASE_LAST")
  [ "$CKE" = "99" ] || { echo "アブレーション 3（因果の微調整なし）の last.ckpt の epoch が 99 でない（$CKE）→ 止める"; exit 3; }
  ABL_ARGS+=(--ablation nocausalft "$BASE_LAST")
else
  echo "注意: 基準学習の last.ckpt が無い（$BASE_LAST）→ アブレーション 3 は V2FINAL_MISSING_OK の理由が要る"; BASE_LAST=""
fi
SCR=${V2FINAL_ABL_SCRATCH_CKPT:-}; SCRJ=${V2FINAL_ABL_SCRATCH_SELECT_JSON:-}; NOH=${V2FINAL_ABL_NOHEIGHT_CKPT:-}; NOHJ=${V2FINAL_ABL_NOHEIGHT_SELECT_JSON:-}
[ -z "$SCR" ] || ABL_ARGS+=(--ablation scratch "$SCR" "$SCRJ")
[ -z "$NOH" ] || ABL_ARGS+=(--ablation noheight "$NOH" "$NOHJ")
MISS_ARGS=()
if [ -n "${V2FINAL_MISSING_OK:-}" ]; then
  IFS=';' read -ra _MK <<< "$V2FINAL_MISSING_OK"
  for kv in "${_MK[@]}"; do [ -n "$kv" ] && MISS_ARGS+=(--missing-ok "${kv%%=*}" "${kv#*=}"); done
fi
CONTRACT_ARGS=(--select-json "$SEL" --decide-json "$DEC" --ckpt "$V2FINAL_CKPT" --dist-table "${DIST_TABLE:-}" --eval-contract "$EVAL/CONTRACT_eval_v3.json"
  --baselines-json "$BJ" "${ABL_ARGS[@]}" --require B,D,scratch,noheight,nocausalft,nohyst "${MISS_ARGS[@]}"
  --code server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/hysteresis_patch.py server_sde/height_patch.py scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_threshold_sweep.py scripts/_gt_as_pred.py
         scripts/_notify_detect.py scripts/_filter_classes.py server_sde/_v2final_members.sh server_sde/_baselines_env.py server_sde/_baseline_D_pretrain170_infer.py server_sde/_baseline_merge.py server_sde/_baseline_B_infer.py server_sde/_baseline_B_common.py)
$PY server_sde/_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --dry || { echo "実行の契約が通らない（開封しない）"; exit 3; }
[ -e "$RES" ] && { echo "既に開封を予約した記録がある（$RES）。停止規則により再実行しない"; exit 4; }
[ -f "$EVAL/SEALED.txt" ] && [ -f "$EVAL/CONTRACT_eval_v3.json" ] || { echo "評価セットが封印されていない（$EVAL/SEALED.txt）。eval_v3_gen.sbatch を先に"; exit 1; }
[ "$(ls $EVAL/foa | wc -l)" = "$N" ] || { echo "評価セットの本数が N（$N）と違う"; exit 1; }
# 封印の照合: CONTRACT の sha256 と今のファイルが一致するか（抜き取りでなく全部）
$PY - "$EVAL" <<'EOF' || { echo "封印の照合に失敗（評価セットが変わっている）。中止"; exit 1; }
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1]); c = json.loads((out / "CONTRACT_eval_v3.json").read_text(encoding="utf-8"))
bad = [k for k, v in c["files"].items() if not (out / k).is_file() or hashlib.sha256((out / k).read_bytes()).hexdigest() != v]
print(f"[seal] files={len(c['files'])} bad={len(bad)}"); sys.exit(1 if bad else 0)
EOF
if [ "$DRY" != 1 ]; then
  $PY -c "import json,sys;P=json.load(open('$PARAMS'));sys.exit(0 if P.get('status')=='fixed' else 3)" || { echo "params が fixed でない"; exit 3; }
  [ -f "$V1CKPT" ] || { echo "v1 の ckpt が無い（$V1CKPT）"; exit 3; }
  for d in "$RUNS/$OUT2" "$RUNS/$OUT1"; do [ -e "$d" ] && { echo "既に確定評価を 1 回実行済み（$d）。停止規則により再実行しない"; exit 4; }; done
  member_absent $(member_outs "$OUT2" "$NAMES") || exit 4     # #73／#74 の出力（_abl_*・_baseline*）も同じ停止規則
  # ---- 開封の予約（RN20-03）: mkdir で原子的に 1 回だけ。契約の記録 EXEC_CONTRACT.json（ckpt・選択・採否・表・評価セットの契約・コードの sha）を中に残す
  mkdir -p "$RUNS"
  $PY server_sde/_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --reserve-dir "$RES" --out "$RES/EXEC_CONTRACT.json" --job "${SLURM_JOB_ID:-}" || { echo "開封の予約ができない（既に予約の記録がある・または契約が通らない）"; exit 4; }   # Astra RI73-01: 乾式と同じ配列展開（配列を展開せずに渡すと最初の要素だけになる）
  sha256sum $V2FINAL_CKPT $V1CKPT $PARAMS server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/height_patch.py server_sde/causal_patch.py \
    server_sde/hysteresis_patch.py scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_threshold_sweep.py scripts/_gt_as_pred.py server_sde/_conf_exec_contract.py \
    scripts/_notify_detect.py scripts/_filter_classes.py server_sde/_v2final_members.sh server_sde/_baselines_env.py server_sde/_baseline_D_pretrain170_infer.py server_sde/_baseline_merge.py \
    server_sde/_baseline_B_infer.py server_sde/_baseline_B_common.py "$BJ" ${BASE_LAST:-} ${SCR:-} ${SCRJ:-} ${NOH:-} ${NOHJ:-} \
    $EVAL/plan/assignment_v2final.csv $EVAL/CONTRACT_eval_v3.json $SEL $DEC ${DIST_TABLE:-} "$RES/EXEC_CONTRACT.json" | tee v2final_conf_v3_sha256_job${SLURM_JOB_ID:-dry}.txt
  git rev-parse HEAD | tee -a v2final_conf_v3_sha256_job${SLURM_JOB_ID:-dry}.txt
fi
cd $PS
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p $JOBDIR
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
export PS E2E
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export CAUSAL_DATASET=outdoor_siren_v12
export CAUSAL_FOA=$EVAL/foa
export CAUSAL_PREFIX=eval_syn_
export CAUSAL_GLOB="eval_syn_*.flac"
export CAUSAL_LIMIT=0
export CAUSAL_EXPECTED_CLIPS=$N
export CAUSAL_RUNS_DIR=$RUNS
# 事前点検: 対象 clip が N 本・全部 eval_syn_
$PY - <<'EOF'
import os, sys
from pathlib import Path
foa = Path(os.environ["CAUSAL_FOA"]); clips = sorted(p.name for p in foa.glob(os.environ["CAUSAL_GLOB"])); expected = int(os.environ["CAUSAL_EXPECTED_CLIPS"])
bad = [c for c in clips if not c.startswith(os.environ["CAUSAL_PREFIX"])]
print(f"[preflight] n={len(clips)} expected={expected} bad_prefix={len(bad)}")
sys.exit(3 if len(clips) != expected or bad else 0)
EOF
if [ "$DRY" = 1 ]; then echo "V2FINAL_CONF_V3_DRYRUN_OK（推論は行わない）"; rm -rf "$JOBDIR"; exit 0; fi
# ---- v2: 9 クラス（datasets_v27）・高さ入力 = 評価セットの plan の mic_z
export HEIGHT_COND=1
export HEIGHT_TABLE=$EVAL/plan/assignment_v2final.csv
export HEIGHT_DEFAULT=1.5
export HEIGHT_JITTER=0
export HEIGHT_OFFSET=0
export CAUSAL_CKPT=$V2FINAL_CKPT
export CAUSAL_DSDIR=$PS/datasets_v27
export CAUSAL_OUT=$OUT2
.venv/bin/python $JOBDIR/_causal_infer_v17.py
.venv/bin/python $JOBDIR/_prediction_integrity.py "$RUNS/$OUT2" $N
[ -f "$RUNS/$OUT2/val_all_causal.done.json" ] || { echo "v2 の完了印が無い"; exit 5; }
cp "$RUNS/$OUT2/val_all_causal.csv" "$RUNS/$OUT2/conf_all.csv"
# ---- v1: 8 クラス（datasets_v12）・高さ入力なし（1.5 m 固定 = v1 の学習時の前提）
export HEIGHT_COND=0
export HEIGHT_TABLE=""
export CAUSAL_CKPT=$V1CKPT
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_OUT=$OUT1
.venv/bin/python $JOBDIR/_causal_infer_v17.py
.venv/bin/python $JOBDIR/_prediction_integrity.py "$RUNS/$OUT1" $N
[ -f "$RUNS/$OUT1/val_all_causal.done.json" ] || { echo "v1 の完了印が無い"; exit 5; }
cp "$RUNS/$OUT1/val_all_causal.csv" "$RUNS/$OUT1/conf_all.csv"
# ---- #74 アブレーション（推論だけ: 3 = 基準学習の last.ckpt・4 = 1 閾値〔保持 = 入り 0.7〕／学習し直し: 1 = scratch・2 = noheight〔高さ入力なし〕）と #73 相手（D・B）。
#   _v2final_members.sh の関数（サブシェル・v2 と同じ推論の条件・来歴・完了印・conf_all.csv）。欠けたものは実行の契約で理由を記録済み（--missing-ok）
export JOBDIR PY RUNS N
HT=$EVAL/plan/assignment_v2final.csv
[ -z "$BASE_LAST" ] || member_v17 "${OUT2}_abl_nocausalft" "$BASE_LAST" 1 "$HT" 0.3 || exit 5
member_v17 "${OUT2}_abl_nohyst" "$V2FINAL_CKPT" 1 "$HT" 0.7 || exit 5
[ -z "$SCR" ] || member_v17 "${OUT2}_abl_scratch" "$SCR" 1 "$HT" 0.3 || exit 5
[ -z "$NOH" ] || member_v17 "${OUT2}_abl_noheight" "$NOH" 0 "" 0.3 || exit 5
for nm in $NAMES; do
  case "$nm" in
    D) member_D "${OUT2}_baselineD" "$BJ" || exit 5 ;;
    B) member_B "${OUT2}_baselineB" "$BJ" || exit 5 ;;
    *) echo "知らない相手: $nm"; exit 5 ;;
  esac
done
# ---- 採点（統一採点器・併記枠）。plan と GT は eval v3
cd $E2E
CP=$EVAL/plan/assignment_v2final.csv
CG=$EVAL/metadata_dist
CU=out/v2final_conf_v3_unified; mkdir -p $CU
V2PRED="$RUNS/$OUT2/conf_all.csv"; V1PRED="$RUNS/$OUT1/conf_all.csv"
# A10（2026-10-20・#72）: 距離の補正表（凍結 §5）。DIST_TABLE=<json> を渡したときだけ統一採点器に --dist-table を足す。--dist-table-for v2final で **v2final の入力だけ**に掛ける（v1 とオラクルは表なし・Astra R72-01）。知覚層と閾値掃引は表なしのまま（表題に書く）。
#   表の有無と採否の一致・表の sha は、推論の前に実行の契約（_conf_exec_contract.py・RN20-03）で照合済み。ここは掛け方だけ
DT_ARG=""; if [ -n "${DIST_TABLE:-}" ]; then [ -f "$DIST_TABLE" ] || { echo "DIST_TABLE が無い: $DIST_TABLE"; exit 5; }; DT_ARG="--dist-table $DIST_TABLE --dist-table-for v2final"; echo "距離の補正表: $DIST_TABLE ($(sha256sum $DIST_TABLE | cut -c1-16))"; fi
PYTHONPATH=scripts:src $PY scripts/_gt_as_pred.py --plan $CP --split eval --clip-max $N --meta $CG --out $CU/gt_as_pred_eval_v3.csv || { echo "オラクル入力の生成に失敗"; exit 5; }
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_eval_v3_v9c.md --strict --raw $DT_ARG --plan $CP --split eval --clip-max $N --meta $CG v1=$V1PRED v2final=$V2PRED oracle=$CU/gt_as_pred_eval_v3.csv > $CU/_log_unified_v9c.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_eval_v3_hold.md --warn-rule hold --strict --raw $DT_ARG --plan $CP --split eval --clip-max $N --meta $CG v1=$V1PRED v2final=$V2PRED oracle=$CU/gt_as_pred_eval_v3.csv > $CU/_log_unified_hold.txt 2>&1
# 変更票 #73（Q2・本人 10/20 23:34）: 知らせの規則だけを「検出したら知らせる（距離を見ない）」に替えた採点（知覚は v2final の同じ予測・警告音の規則は同じ・比較用の 1 本）
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_eval_v3_detect.md --notify-rule detect --strict --raw $DT_ARG --plan $CP --split eval --clip-max $N --meta $CG v2final=$V2PRED > $CU/_log_unified_detect.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_perception_eval.py $CU/perception_eval_v3.md --plan $CP --split eval --clip-max $N --meta $CG --json $CU/perception_eval_v3.json --strict v1=$V1PRED v2final=$V2PRED > $CU/_log_perception.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_notify_threshold_sweep.py $CU/notify_sweep_eval_v3.md --plan $CP --split eval --clip-max $N --meta $CG --csv $CU/notify_sweep_eval_v3.csv --strict v1=$V1PRED v2final=$V2PRED > $CU/_log_sweep.txt 2>&1
# ---- #74 アブレーションの採点（v2final と並べる・補正表は v2final だけ〔R72-01〕→ 表ありのときは v2final_notable で表なしの v2final も並べる）
ABL_ITEMS=""; for t in nocausalft nohyst scratch noheight; do p="$RUNS/${OUT2}_abl_$t/conf_all.csv"; [ -f "$p" ] && ABL_ITEMS="$ABL_ITEMS abl_$t=$p"; done
NT_ITEM=""; [ -z "$DT_ARG" ] || NT_ITEM="v2final_notable=$V2PRED"
PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_eval_v3_v9c_abl.md --strict --raw $DT_ARG --plan $CP --split eval --clip-max $N --meta $CG v2final=$V2PRED $NT_ITEM $ABL_ITEMS > $CU/_log_unified_abl.txt 2>&1
PYTHONPATH=scripts:src $PY scripts/_perception_eval.py $CU/perception_eval_v3_abl.md --plan $CP --split eval --clip-max $N --meta $CG --json $CU/perception_eval_v3_abl.json --strict v2final=$V2PRED $ABL_ITEMS > $CU/_log_perception_abl.txt 2>&1
# ---- #73 相手の採点: 知覚の表 = 全部の相手（B・D）・知らせの表 = 距離を出す相手だけ（B。D は入れない〔Q3〕）。相手は同じ予測 csv の形なので採点器は同じ
BL_PERC=""; BL_NOTIFY=""
for nm in $NAMES; do p="$RUNS/${OUT2}_baseline$nm/conf_all.csv"; [ -f "$p" ] || continue; BL_PERC="$BL_PERC baseline$nm=$p"; [ "$nm" = D ] || BL_NOTIFY="$BL_NOTIFY baseline$nm=$p"; done
[ -z "$BL_PERC" ] || PYTHONPATH=scripts:src $PY scripts/_perception_eval.py $CU/perception_eval_v3_baselines.md --plan $CP --split eval --clip-max $N --meta $CG --json $CU/perception_eval_v3_baselines.json --strict v2final=$V2PRED $BL_PERC > $CU/_log_perception_baselines.txt 2>&1
[ -z "$BL_NOTIFY" ] || PYTHONPATH=scripts:src $PY scripts/_score_unified.py $CU/unified_eval_v3_v9c_baselines.md --strict --raw $DT_ARG --plan $CP --split eval --clip-max $N --meta $CG v2final=$V2PRED $NT_ITEM $BL_NOTIFY > $CU/_log_unified_baselines.txt 2>&1
# ---- R73-04: 相手 D と共通の 6 クラス {0,1,3,4,7,8} だけに正解と予測を絞った写しで知覚を採点し直す（提案法も同じクラスで集計・採点器は変えない）
if [ -f "$RUNS/${OUT2}_baselineD/conf_all.csv" ]; then
  PYTHONPATH=scripts:src $PY scripts/_filter_classes.py --classes 0,1,3,4,7,8 --meta-in $CG --meta-out $CU/meta_commonD --pred-in $V2PRED --pred-out $CU/pred_v2final_commonD.csv > $CU/_log_filter_commonD.txt 2>&1 || { echo "共通クラスの写し（v2final）に失敗"; exit 5; }
  PYTHONPATH=scripts:src $PY scripts/_filter_classes.py --classes 0,1,3,4,7,8 --meta-in $CG --meta-out $CU/meta_commonD --pred-in "$RUNS/${OUT2}_baselineD/conf_all.csv" --pred-out $CU/pred_baselineD_commonD.csv >> $CU/_log_filter_commonD.txt 2>&1 || { echo "共通クラスの写し（D）に失敗"; exit 5; }
  PYTHONPATH=scripts:src $PY scripts/_perception_eval.py $CU/perception_eval_v3_commonD.md --plan $CP --split eval --clip-max $N --meta $CU/meta_commonD --json $CU/perception_eval_v3_commonD.json --strict v2final=$CU/pred_v2final_commonD.csv baselineD=$CU/pred_baselineD_commonD.csv > $CU/_log_perception_commonD.txt 2>&1
fi
ALLPRED=""; for d in $(member_outs "$OUT2" "$NAMES"); do [ -f "$RUNS/$d/conf_all.csv" ] && ALLPRED="$ALLPRED $RUNS/$d/conf_all.csv"; done
sha256sum $CU/*.md $CU/*.json $CU/*.csv "$V1PRED" $ALLPRED scripts/_gt_as_pred.py scripts/_score_unified.py scripts/_notify_detect.py scripts/_perception_eval.py scripts/_notify_threshold_sweep.py scripts/_filter_classes.py | tee $CU/sha256_job${SLURM_JOB_ID:-dry}.txt
guard_verify || exit 1
echo V2FINAL_CONF_V3_DONE
