#!/bin/bash
#SBATCH -J fieldinf
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH -o fieldinf-%j.out
# 実録の推論（変更票 #73 比べる相手・#74 アブレーション・2026-10-21 Fable）: 切り出した実録の本（step19b の eval_field_clip*.flac・4ch 24 kHz 10 s・原本 raw/ はサーバに置かない）を、
#   確定評価と同じ顔ぶれ（提案法・アブレーション 1〜4・相手 D・B）で、確定評価と同じ条件（_v2final_members.sh の関数・因果・ヒステリシス 0.7/0.3・高さの表）で推論する。
#   採点（step20・step19k・知覚層）は手元で行う（正解の台帳 raw/ledger は手元にしかない）。予測は $RUNS/<OUT>/conf_all.csv（来歴つき）を scp で手元へ。
#   推論と採点は録り終えて最後の学習が終わってから 1 回（引き継ぎ §B）。同じ FIELD_TAG で 2 回目は予約（mkdir）で止まる。
# 使い方: FIELD_TAG=<例 f20261029> FIELD_FOA=<切り出した本の置き場> FIELD_HEIGHT_TABLE=<clip_id,mic_z の csv> FIELD_N=<本数> \
#         V2FINAL_CKPT=… V2FINAL_SELECT_JSON=… V2FINAL_DECIDE_JSON=… [DIST_TABLE=…] [V2FINAL_BASELINES_JSON=…] [V2FINAL_BASE_LAST_CKPT=…] \
#         [V2FINAL_ABL_SCRATCH_CKPT=… V2FINAL_ABL_SCRATCH_SELECT_JSON=… V2FINAL_ABL_NOHEIGHT_CKPT=… V2FINAL_ABL_NOHEIGHT_SELECT_JSON=…] [V2FINAL_MISSING_OK="名前=理由;…"] \
#         sbatch server_sde/field_eval_infer.sbatch
# 乾式: FIELD_DRYRUN=1 bash server_sde/field_eval_infer.sbatch（同じ環境変数）→ 契約と入力の点検だけ（GPU 不要・予約しない）
#   実行の契約は確定評価と同じ _conf_exec_contract.py（選択器・採否・表・相手の凍結・アブレーション・欠けの理由）。評価セットの契約（eval v3）は無い。記録は $RUNS/_field_<TAG>_opened/FIELD_EXEC_CONTRACT.json
set -exo pipefail
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
DRY=${FIELD_DRYRUN:-0}
cd $E2E
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
source $E2E/server_sde/_baseline_guard.sh
guard_record
source $E2E/server_sde/_v2final_members.sh
TAG=${FIELD_TAG:-}; FOA=${FIELD_FOA:-}; HT=${FIELD_HEIGHT_TABLE:-}; N=${FIELD_N:-}
[ -n "$TAG" ] && [ -n "$FOA" ] && [ -d "$FOA" ] && [ -n "$HT" ] && [ -f "$HT" ] && [ -n "$N" ] || { echo "FIELD_TAG・FIELD_FOA（置き場）・FIELD_HEIGHT_TABLE（csv）・FIELD_N（本数）を渡す"; exit 3; }
case "$TAG" in *[!A-Za-z0-9_]*) echo "FIELD_TAG は英数字と _ だけ（$TAG）"; exit 3 ;; esac
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/field_v3
OUT2=infer_field_${TAG}
RES=$RUNS/_field_${TAG}_opened
SEL=${V2FINAL_SELECT_JSON:-}; DEC=${V2FINAL_DECIDE_JSON:-}
[ -n "$SEL" ] && [ -f "$SEL" ] && [ -n "$DEC" ] && [ -f "$DEC" ] || { echo "V2FINAL_SELECT_JSON と V2FINAL_DECIDE_JSON を渡す（確定評価と同じ）"; exit 3; }
[ -n "${V2FINAL_CKPT:-}" ] && [ -f "$V2FINAL_CKPT" ] || { echo "V2FINAL_CKPT を渡す"; exit 3; }
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
  [ "$CKE" = "99" ] || { echo "アブレーション 3 の last.ckpt の epoch が 99 でない（$CKE）→ 止める"; exit 3; }
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
CONTRACT_ARGS=(--select-json "$SEL" --decide-json "$DEC" --ckpt "$V2FINAL_CKPT" --dist-table "${DIST_TABLE:-}"
  --baselines-json "$BJ" "${ABL_ARGS[@]}" --require B,D,scratch,noheight,nocausalft,nohyst "${MISS_ARGS[@]}"
  --code server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/hysteresis_patch.py server_sde/height_patch.py
         server_sde/_v2final_members.sh server_sde/_baselines_env.py server_sde/_baseline_D_pretrain170_infer.py server_sde/_baseline_merge.py server_sde/_baseline_B_infer.py server_sde/_baseline_B_common.py)
$PY server_sde/_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --dry || { echo "実行の契約が通らない（推論しない）"; exit 3; }
[ -e "$RES" ] && { echo "既にこの札（$TAG）で推論を予約した記録がある（$RES）。停止規則により再実行しない"; exit 4; }
# 入力の点検: 本数 N・全部 eval_field_・高さの表が全部の本を持つ（mic_z は有限の正）
export CAUSAL_FOA=$FOA CAUSAL_PREFIX=eval_field_ CAUSAL_GLOB="eval_field_*.flac"
$PY - "$FOA" "$HT" "$N" <<'EOF'
import csv, math, sys
from pathlib import Path
import soundfile as sf
foa, ht, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
clips = sorted(p.stem for p in foa.glob("eval_field_*.flac"))
other = [p.name for p in foa.glob("*.flac") if not p.name.startswith("eval_field_")]
with open(ht, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
z = {r["clip_id"]: r.get("mic_z", "") for r in rows}
bad = [c for c in clips if c not in z or not z[c] or not math.isfinite(float(z[c])) or float(z[c]) <= 0]
# Astra RI73（確認できた点）: 名前だけでなく音声のヘッダーも点検（切り出した 10 s・4 ch・24 kHz = 240,000 サンプル。原本の長回しではない）
badhdr = []
for c in clips:
    i = sf.info(str(foa / f"{c}.flac"))
    if not (i.channels == 4 and i.samplerate == 24000 and i.frames == 240000):
        badhdr.append((c, i.channels, i.samplerate, i.frames))
print(f"[preflight] n={len(clips)} expected={n} other_prefix={len(other)} height_rows={len(rows)} height_missing={len(bad)} bad_header={len(badhdr)} {badhdr[:3]}")
sys.exit(3 if len(clips) != n or other or bad or badhdr else 0)
EOF
if [ "$DRY" = 1 ]; then echo "FIELD_EVAL_INFER_DRYRUN_OK（推論は行わない）"; exit 0; fi
mkdir -p "$RUNS"
member_absent $(member_outs "$OUT2" "$NAMES") || exit 4
$PY server_sde/_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --reserve-dir "$RES" --out "$RES/FIELD_EXEC_CONTRACT.json" --job "${SLURM_JOB_ID:-}" || { echo "予約ができない"; exit 4; }
sha256sum $V2FINAL_CKPT server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/height_patch.py server_sde/causal_patch.py server_sde/hysteresis_patch.py server_sde/_conf_exec_contract.py \
  server_sde/_v2final_members.sh server_sde/_baselines_env.py server_sde/_baseline_D_pretrain170_infer.py server_sde/_baseline_merge.py server_sde/_baseline_B_infer.py server_sde/_baseline_B_common.py \
  "$BJ" ${BASE_LAST:-} ${SCR:-} ${SCRJ:-} ${NOH:-} ${NOHJ:-} "$HT" $SEL $DEC ${DIST_TABLE:-} "$RES/FIELD_EXEC_CONTRACT.json" | tee field_eval_infer_${TAG}_sha256_job${SLURM_JOB_ID:-dry}.txt
git rev-parse HEAD | tee -a field_eval_infer_${TAG}_sha256_job${SLURM_JOB_ID:-dry}.txt
cd $PS
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p $JOBDIR
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
export PS E2E JOBDIR PY RUNS N
# 顔ぶれ: 提案法 → 3 nocausalft → 4 nohyst → 1 scratch → 2 noheight → 相手（D・B）。条件は確定評価と同じ（_v2final_members.sh）
member_v17 "$OUT2" "$V2FINAL_CKPT" 1 "$HT" 0.3 || exit 5
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
cd $E2E
ALLPRED=""; for d in $(member_outs "$OUT2" "$NAMES"); do [ -f "$RUNS/$d/conf_all.csv" ] && ALLPRED="$ALLPRED $RUNS/$d/conf_all.csv"; done
sha256sum $ALLPRED | tee -a field_eval_infer_${TAG}_sha256_job${SLURM_JOB_ID:-dry}.txt
guard_verify || exit 1
rm -rf "$JOBDIR"
echo "FIELD_EVAL_INFER_DONE $OUT2（次: 手元で scp して step20／step19k で採点）"
