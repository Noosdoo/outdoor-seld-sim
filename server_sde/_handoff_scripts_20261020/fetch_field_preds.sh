#!/bin/bash
# 実録の推論の出力（field_eval_infer.sbatch・顔ぶれ全員）をサーバから手元へ写し、来歴を照合する（2026-10-21・#73/#74）。
# 使い方: bash server_sde/_handoff_scripts_20261020/fetch_field_preds.sh <TAG> <N 本数> [手元の置き場（既定 out/_field_preds/<TAG>）]
#   写すもの: ~/PSELDNets_logs/outdoor_siren_v12/runs/field_v3/infer_field_<TAG>*/{val_all_causal.csv, val_all_causal.done.json, run_meta.json}・_field_<TAG>_opened/FIELD_EXEC_CONTRACT.json
#   照合: _prediction_integrity.py <csv> <N> --download --done --meta（1 本でも通らなければ 0 以外）。次: scripts/field_score_members.py
set -euo pipefail
cd /c/Users/satos/research/outdoor_seld_e2e
TAG="${1:?TAG}"; N="${2:?N}"; DST="${3:-out/_field_preds/$TAG}"
case "$TAG" in *[!A-Za-z0-9_]*) echo "TAG は英数字と _ だけ（$TAG）"; exit 1;; esac
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
R="PSELDNets_logs/outdoor_siren_v12/runs/field_v3"
mkdir -p "$DST"
DIRS=$(ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/$R && ls -d infer_field_${TAG} infer_field_${TAG}_* 2>/dev/null | tr '\n' ' '")
[ -n "$DIRS" ] || { echo "サーバに infer_field_${TAG}* が無い"; exit 2; }
echo "写す: $DIRS"
fail=0
for d in $DIRS; do
  mkdir -p "$DST/$d"
  for f in val_all_causal.csv val_all_causal.done.json run_meta.json; do
    scp -o ConnectTimeout=10 -q "is-server:$R/$d/$f" "$DST/$d/$f" || { echo "写せない: $d/$f"; fail=1; }
  done
  if [ -f "$DST/$d/val_all_causal.csv" ]; then
    PYTHONUTF8=1 $PY server_sde/_prediction_integrity.py "$DST/$d/val_all_causal.csv" "$N" --download --done "$DST/$d/val_all_causal.done.json" --meta "$DST/$d/run_meta.json" \
      | sed -E 's/s[0-9]{7}/<user>/g' | tail -n 1 || { echo "来歴の照合に失敗: $d"; fail=1; }
  fi
done
scp -o ConnectTimeout=10 -q "is-server:$R/_field_${TAG}_opened/FIELD_EXEC_CONTRACT.json" "$DST/FIELD_EXEC_CONTRACT.json" || { echo "契約の記録が写せない（予約が無い？）"; fail=1; }
grep -lE 's[0-9]{7}' "$DST"/*/run_meta.json "$DST"/FIELD_EXEC_CONTRACT.json 2>/dev/null | while read -r f; do echo "注意: ユーザー名を含む（手元の照合用・git に入れない）: $f"; done
echo "-> $DST（次: python scripts/field_score_members.py --tag $TAG --pred-root $DST --n $N ...）"
exit $fail
