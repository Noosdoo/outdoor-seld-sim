#!/bin/bash
# 推論する「顔ぶれ」の共通の関数（変更票 #73 比べる相手・#74 アブレーション・2026-10-21 Fable）。
#   確定評価（v2final_conf_v3.sbatch・eval v3）と実録の推論（field_eval_infer.sbatch）が source して同じ関数で回す = 合成と実録で推論の条件を 1 か所に。
# 呼ぶ前に設定する変数:
#   PS E2E PY RUNS N JOBDIR（_causal_infer_v17.py・_prediction_integrity.py・hysteresis_patch.py の写し。height_patch.py・causal_patch.py は $PS に写してある）
#   CAUSAL_FOA CAUSAL_PREFIX CAUSAL_GLOB（推論する本の置き場・名前の頭・glob）
#   MEMBER_DRY=1 なら実行せず、命令と主な環境変数を印字するだけ（検査用）
# 各関数はサブシェルで走る（環境変数を外に漏らさない）。
# 出力: $RUNS/<OUT>/{submissions/, val_all_causal.csv, run_meta.json, val_all_causal.done.json, conf_all.csv}（来歴は _prediction_integrity）
# 失敗したら 0 以外で返す（呼ぶ側は set -e か || exit で止める）。

member_common_env() {
  export CAUSAL_LIMIT=0 CAUSAL_EXPECTED_CLIPS="$N" CAUSAL_RUNS_DIR="$RUNS" CAUSAL_SED_ASSOC=20
  unset CAUSAL_MANIFEST BASELINE_DUMP_NORMS BASELINE_DEBUG
}

member_run() {   # member_run <説明> <命令...>
  local what="$1"; shift
  if [ "${MEMBER_DRY:-0}" = 1 ]; then
    echo "[member-dry] $what: $*"
    echo "[member-dry]   OUT=${CAUSAL_OUT:-} CKPT=${CAUSAL_CKPT:-${B_MODEL:-}} HEIGHT_COND=${HEIGHT_COND:-} HEIGHT_TABLE=${HEIGHT_TABLE:-} TH=${CAUSAL_SED_TH:-} STAY=${CAUSAL_SED_STAY:-} ASSOC=${CAUSAL_SED_ASSOC:-} GAIN=${BASELINE_GAIN_DB:-} CLASSMAP=${BASELINE_CLASSMAP:-} DSDIR=${CAUSAL_DSDIR:-} DATASET=${CAUSAL_DATASET:-} FEAT=${B_FEAT_DIR:-} EXPECTED=${CAUSAL_EXPECTED_CLIPS:-} FOA=${CAUSAL_FOA:-} PREFIX=${CAUSAL_PREFIX:-}"
    return 0
  fi
  echo "[member] $what"
  "$@"
}

member_finish() {   # member_finish <OUT>: 来歴の照合（本数 N）・完了印・conf_all.csv
  local out="$1"
  if [ "${MEMBER_DRY:-0}" = 1 ]; then echo "[member-dry] finish $out (N=$N)"; return 0; fi
  "$PY" "$JOBDIR/_prediction_integrity.py" "$RUNS/$out" "$N" || return 5
  [ -f "$RUNS/$out/val_all_causal.done.json" ] || { echo "$out の完了印が無い"; return 5; }
  cp "$RUNS/$out/val_all_causal.csv" "$RUNS/$out/conf_all.csv"
}

member_absent() {   # member_absent <OUT>...: どれか既にあれば 4（停止規則: 再実行しない）
  local d
  for d in "$@"; do
    [ -e "$RUNS/$d" ] && { echo "既に推論の出力がある（$RUNS/$d）。停止規則により再実行しない"; return 4; }
  done
  return 0
}

# この研究の形の推論（v17）: member_v17 <OUT> <CKPT> <HEIGHT_COND 0|1> <HEIGHT_TABLE（HEIGHT_COND=0 なら ""）> <SED_STAY> [DSDIR]
#   提案法 = HEIGHT_COND 1・表あり・保持 0.3。アブレーション: noheight = 0・""・0.3／nohyst = 1・表・0.7（保持 = 入り = 1 閾値）／nocausalft・scratch = 提案法と同じ条件で ckpt だけ違う
member_v17() {
  local out="$1" ckpt="$2" hc="$3" ht="$4" stay="$5" dsdir="${6:-$PS/datasets_v27}"
  ( set -e
    member_common_env
    export CAUSAL_CKPT="$ckpt" CAUSAL_OUT="$out" HEIGHT_COND="$hc" HEIGHT_TABLE="$ht" HEIGHT_DEFAULT=1.5 HEIGHT_JITTER=0 HEIGHT_OFFSET=0
    export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY="$stay" CAUSAL_DSDIR="$dsdir"
    export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0 CAUSAL_DATASET=outdoor_siren_v12
    [ "$hc" = 1 ] && [ -z "$ht" ] && { echo "HEIGHT_COND=1 なのに高さの表が無い（$out）"; exit 3; }
    [ "$hc" = 0 ] && [ -n "$ht" ] && { echo "HEIGHT_COND=0 なのに高さの表がある（$out）"; exit 3; }
    [ "${MEMBER_DRY:-0}" = 1 ] || [ -f "$ckpt" ] || { echo "ckpt が無い（$ckpt）"; exit 3; }
    cd "$PS"
    member_run "v17 $out" "$PY" "$JOBDIR/_causal_infer_v17.py"
    member_finish "$out"
  )
}

# 相手 D（事前学習だけの PSELDNets・170 クラス → 9）: member_D <OUT> <baselines_selected.json>
#   ゲイン・閾値・対応表・重み・170 行の tsv は凍結 JSON から（_baselines_env.py）。高さ入力なし・距離 nan
member_D() {
  local out="$1" bj="$2"
  ( set -e
    member_common_env
    local envs
    envs=$("$PY" "$E2E/server_sde/_baselines_env.py" "$bj" --env D) || { echo "$envs"; exit 3; }
    eval "$envs"
    [ -n "${BASELINE_TSV170:-}" ] || { echo "D の 170 行の tsv が凍結 JSON に無い"; exit 3; }
    local ds170="$PS/datasets_pretrain170"
    if [ "${MEMBER_DRY:-0}" != 1 ]; then
      mkdir -p "$ds170"
      cp "$BASELINE_TSV170" "$ds170/cls_indices_train.tsv"
      [ "$(wc -l < "$ds170/cls_indices_train.tsv")" = 170 ] || { echo "170 行の tsv でない"; exit 3; }
    fi
    export CAUSAL_DSDIR="$ds170" CAUSAL_OUT="$out" HEIGHT_COND=0 HEIGHT_TABLE=""
    cd "$PS"
    member_run "D $out" "$PY" "$E2E/server_sde/_baseline_D_pretrain170_infer.py"
    member_finish "$out"
  )
}

# 相手 B（DCASE 2024 公式ベースラインを v29 で学習した重み）: member_B <OUT> <baselines_selected.json>
#   重み（.h5）・標準化の重み（B_FEAT_DIR の wts）・閾値は凍結 JSON から。ゲイン無し。距離 m
member_B() {
  local out="$1" bj="$2"
  ( set -e
    member_common_env
    local envs
    envs=$("$PY" "$E2E/server_sde/_baselines_env.py" "$bj" --env B) || { echo "$envs"; exit 3; }
    eval "$envs"
    export CAUSAL_OUT="$out"
    [ "${MEMBER_DRY:-0}" = 1 ] || [ -f "$B_MODEL" ] || { echo "B の重みが無い（$B_MODEL）"; exit 3; }
    cd "$E2E"
    member_run "B $out" "$PY" "$E2E/server_sde/_baseline_B_infer.py"
    member_finish "$out"
  )
}

# 顔ぶれを一気に回す: member_all <OUTPFX> <HEIGHT_TABLE> <V2_CKPT> <BASE_LAST_CKPT|""> <SCRATCH_CKPT|""> <NOHEIGHT_CKPT|""> <baselines_selected.json> <相手の名前（空白区切り）>
#   出力名: <OUTPFX>（提案法）・<OUTPFX>_abl_nocausalft・_abl_nohyst・_abl_scratch・_abl_noheight・<OUTPFX>_baselineD・_baselineB
#   空 "" を渡したアブレーションは飛ばす（契約で理由を記録してあること = 台本側で照合済み）
member_all() {
  local pfx="$1" ht="$2" v2="$3" last="$4" scr="$5" noh="$6" bj="$7" names="$8"
  member_v17 "$pfx" "$v2" 1 "$ht" 0.3 || return $?
  [ -z "$last" ] || member_v17 "${pfx}_abl_nocausalft" "$last" 1 "$ht" 0.3 || return $?
  member_v17 "${pfx}_abl_nohyst" "$v2" 1 "$ht" 0.7 || return $?
  [ -z "$scr" ] || member_v17 "${pfx}_abl_scratch" "$scr" 1 "$ht" 0.3 || return $?
  [ -z "$noh" ] || member_v17 "${pfx}_abl_noheight" "$noh" 0 "" 0.3 || return $?
  local nm
  for nm in $names; do
    case "$nm" in
      D) member_D "${pfx}_baselineD" "$bj" || return $? ;;
      B) member_B "${pfx}_baselineB" "$bj" || return $? ;;
      *) echo "知らない相手: $nm"; return 3 ;;
    esac
  done
}

# member_all が作る出力名の一覧（停止規則の検査・採点の入力に使う）: member_outs <OUTPFX> <相手の名前（空白区切り）> → 空白区切り
member_outs() {
  local pfx="$1" names="$2" nm s
  s="$pfx ${pfx}_abl_nocausalft ${pfx}_abl_nohyst ${pfx}_abl_scratch ${pfx}_abl_noheight"
  for nm in $names; do s="$s ${pfx}_baseline$nm"; done
  echo "$s"
}
