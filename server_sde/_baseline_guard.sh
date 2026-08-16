#!/bin/bash
# 基準データセット・基準runの「無変更」を機械で保証するガード（2026-08-15）。
#
#   source _baseline_guard.sh; guard_record   # ジョブ冒頭で指紋を取る
#   ...本処理...
#   guard_verify                              # 末尾で照合。1バイトでも違えば非ゼロ終了
#
# 対象は ablation で絶対に書き換えてはいけないもの:
#   - 基準データセット3種（v12 / v12_conf / v12_eval）
#   - 基準 datasets ルート（前処理h5がここへ書かれる事故の検知）
#   - 基準run（outdoor_siren_v12_sde_w3）と初期重みckpt
#
# 指紋は (size, mtime, path) の sha256。内容書き換え・追加・削除・touch を全部拾う。
set -u

GUARD_PATHS=(
  "$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v12"
  "$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v12_conf"
  "$HOME/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v12_eval"
  "$HOME/research/PSELDNet/PSELDNets/datasets_v12"
  "$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3"
  "$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt"
  # 基準の前処理キャッシュ。**データセット名でキー付けされる**ため、armの前処理が
  # 同名で上書きしうる（2026-08-16に実際に上書きされた）。armは paths.hdf5_dir を
  # 分けること。ここに入れておけば同じ事故を検知できる。
  "$HOME/research/PSELDNet/PSELDNets/_hdf5"
)
GUARD_FILE="${GUARD_FILE:-/tmp/abl_guard_${SLURM_JOB_ID:-manual}.txt}"

_guard_fp() {
  local d="$1"
  if [ -d "$d" ]; then
    find "$d" \( -type f -o -type l \) -printf '%s %T@ %p\n' | sort | sha256sum | cut -d' ' -f1
  elif [ -e "$d" ]; then
    stat -c '%s %Y %n' "$d" | sha256sum | cut -d' ' -f1
  else
    echo "MISSING"
  fi
}

guard_record() {
  : > "$GUARD_FILE"
  for d in "${GUARD_PATHS[@]}"; do
    echo "$(_guard_fp "$d")  $d" >> "$GUARD_FILE"
  done
  echo "[guard] 基準の指紋を記録: $GUARD_FILE"
  cat "$GUARD_FILE"
}

guard_verify() {
  local bad=0
  echo "[guard] 基準の無変更を照合中..."
  while read -r fp d; do
    local now
    now="$(_guard_fp "$d")"
    if [ "$now" != "$fp" ]; then
      echo "[guard] !!! 変更を検出: $d"
      echo "[guard]     before=$fp"
      echo "[guard]     after =$now"
      bad=1
    fi
  done < "$GUARD_FILE"
  if [ "$bad" -ne 0 ]; then
    echo "[guard] ABORT: 基準が書き換わりました。生成物を破棄して原因を調べてください。"
    return 1
  fi
  echo "[guard] OK: 基準は1バイトも変わっていません"
  return 0
}
