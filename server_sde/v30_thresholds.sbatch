#!/bin/bash
#SBATCH -J v30thr
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v30thr-%j.out
# 検出のしきい値（本体 §3-5・§9-1 の 5・R30-05・2026-10-26）: 選んだ 1 つの ckpt の因果推論を v30 の val 全部に DUMP_NORMS=1 で回し（norm を間引かず全部）、
#   scripts/v30_select_thresholds.py でクラスごとの入りを選ぶ（選択用 clip%12==0・確認用 %12==6・候補 0.1〜0.7・入りだけの F・フレーム 40〜99・保持 = 入り × 3/7）。
#   推論の条件 = 候補の推論（v2final_candidates_infer.sbatch）と同じ env ＋ 決めた倍率（SDE_INPUT_GAIN_DB を明示）＋ HYST_START_FRAME=40（明示）。表は使わない（norm は表に依らない）。
#   出力の表はサーバの out/v30_thresholds/ に書く → Opus が手元に取って md/design/thresholds_v30.json（主）／thresholds_v30_<札>.json としてコミットする（Fable の確認のあと）。
#   **予備（Fable 16:5x）**: 本番は候補 8 個の推論（v2final_candidates_infer.sbatch・v30 は DUMP_NORMS=1）が norms を取るので、選んだ候補の run を
#   直接 scripts/v30_select_thresholds.py --run に渡す（もう 1 回の推論はしない）。この台本は norms が無いときだけ使う。
# 使い方（発効のあと・Fable の OK のあと・§4 の順で選択器 2.2 のあと）:
#   主:         V30_GAIN=<決めた倍率> V30_SELECT_JSON=<選択器 2.2 の select.json> sbatch server_sde/v30_thresholds.sbatch
#   scratch／noheight: 上に V2FINAL_ABL_TAG=scratch|noheight（それぞれの select.json）
#   因果 ft なし: V30_TH_LABEL=nocausalft V30_TH_CKPT=<基準学習の last.ckpt>（select.json なし）
#   乾式: V30_TH_DRY=1（ckpt の確かめと命令の印字だけ）
set -euxo pipefail
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
G=${V30_GAIN:?V30_GAIN（決めた入力の倍率・dB）を渡す（明示が来歴の約束・0 でも渡す）}
DATA=${V2FINAL_DATA:-v30}
DSN=${V2FINAL_DS_NAME:-dataset_outdoor_siren_v2final_v30}
DS=$E2E/out/$DSN
ABL_TAG=${V2FINAL_ABL_TAG:-}; ABL_SUF=""; ABL_HEIGHT=1
case "$ABL_TAG" in
  "") ;;
  scratch) ABL_SUF=_abl_scratch ;;
  noheight) ABL_SUF=_abl_noheight; ABL_HEIGHT=0 ;;
  *) echo "V2FINAL_ABL_TAG は scratch か noheight（$ABL_TAG）"; exit 3 ;;
esac
LABEL=${V30_TH_LABEL:-${ABL_TAG:-main}}
case "$LABEL" in main|scratch|noheight|nocausalft) ;; *) echo "V30_TH_LABEL は main／scratch／noheight／nocausalft（$LABEL）"; exit 3 ;; esac
GT=${V30_TH_GT:-$DS/metadata_dist}
MAN=${V30_TH_MANIFEST:-$DS/plan/manifest_val_syn_v2final.csv}
SELL=${V30_TH_SELECT_LIST:-$DS/plan/manifest_val_select.csv}
CHKL=${V30_TH_CHECK_LIST:-$DS/plan/manifest_val_check.csv}
for f in "$GT" "$MAN" "$SELL" "$CHKL" "$DS/plan/assignment_v2final.csv" "$PS/datasets_$DATA/outdoor_siren_v12"; do [ -e "$f" ] || { echo "無い: $f"; exit 3; }; done
# ckpt: 選択器 2.2 の select.json の selected の候補（sha も照合）か、nocausalft は V30_TH_CKPT
if [ "$LABEL" = nocausalft ]; then
  CK=${V30_TH_CKPT:?nocausalft は V30_TH_CKPT（基準学習の last.ckpt）を渡す}
  EP=base
else
  SJ=${V30_SELECT_JSON:?V30_SELECT_JSON（選択器 2.2 の select.json）を渡す}
  read -r CK EP CKSHA < <($PS/.venv/bin/python - "$SJ" <<'EOF'
import json, os, sys
S = json.load(open(sys.argv[1], encoding="utf-8"))
if not str(S.get("spec_version", "")).startswith("v2_ckpt_selector/2.2"):
    sys.exit(f"選択器の版が 2.2 でない: {str(S.get('spec_version'))[:40]}")
if S.get("not_for_registration") or S.get("selected") is None:
    sys.exit("選択器が選んでいないか『登録には使えない』")
m = S["candidates"][str(S["selected"])]
print(os.path.expanduser(m["ckpt"]), f"e{int(S['selected']):03d}", m["ckpt_sha256"])
EOF
)
  CKSHA=${CKSHA%$'\r'}
  [ "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CKSHA" ] || { echo "ckpt の sha が select.json と違う: $CK"; exit 3; }
fi
[ -f "$CK" ] || { echo "ckpt が無い: $CK"; exit 3; }
OUT=infer_v30th_${LABEL}_${EP}_norms
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
TBL=$E2E/out/v30_thresholds/thresholds_v30$([ "$LABEL" = main ] || echo "_$LABEL").json
RUN() { if [ "${V30_TH_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
echo "LABEL=$LABEL CK=$CK EP=$EP OUT=$OUT GAIN=$G TBL=$TBL"
[ ! -f "$TBL" ] || { echo "表が既にある: $TBL（作り直すなら消す理由を PROGRESS に書いてから）"; exit 3; }
# 1. 因果推論（val 全部・DUMP_NORMS=1・完了印があれば飛ばす）
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; RUN mkdir -p $JOBDIR
for f in _causal_infer_v17.py _prediction_integrity.py hysteresis_patch.py gain_patch.py; do RUN cp $E2E/server_sde/$f $JOBDIR/$f; done
RUN cp $E2E/server_sde/height_patch.py $PS/height_patch.py
RUN cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
export PS SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=$ABL_HEIGHT HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export HEIGHT_TABLE=$DS/plan/assignment_v2final.csv; [ "$ABL_HEIGHT" = 1 ] || export HEIGHT_TABLE=""
export CAUSAL_PREFIX=val_syn_ CAUSAL_LIMIT=0 CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3 HYST_START_FRAME=40 DUMP_NORMS=1
export SDE_INPUT_GAIN_DB=$G
unset CAUSAL_GLOB CAUSAL_FOA CAUSAL_MANIFEST CAUSAL_THRESHOLDS_JSON CAUSAL_THRESHOLDS_MODE
export CAUSAL_CKPT=$CK CAUSAL_DSDIR=$PS/datasets_$DATA CAUSAL_OUT=$OUT
export CAUSAL_EXPECTED_CLIPS=$(($(wc -l < $MAN) - 1))
cd $PS
if [ -f "$RUNS/$OUT/val_all_causal.done.json" ] && [ -f "$RUNS/$OUT/norms_manifest.json" ]; then
  echo "済み: $OUT（完了印と norms_manifest あり・推論は飛ばす）"
else
  RUN .venv/bin/python $JOBDIR/_causal_infer_v17.py
fi
# 2. しきい値を選ぶ（選択用・確認用が plan の 2 つの名簿と同じ本かも確かめる）
RUN mkdir -p $E2E/out/v30_thresholds
RUN .venv/bin/python $E2E/scripts/v30_select_thresholds.py --gt "$GT" --manifest "$MAN" --run "$RUNS/$OUT" --label "$LABEL" --out "$TBL" --expect-lists "$SELL" "$CHKL"
RUN sha256sum "$TBL" "$RUNS/$OUT/norms_manifest.json" "$CK"
RUN rm -rf "$JOBDIR"
echo "V30_THRESHOLDS_DONE label=$LABEL table=$TBL"
