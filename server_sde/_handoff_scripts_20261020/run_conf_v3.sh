#!/bin/bash
# 確定評価（1 回きり）の入力をサーバへ写し、実行の契約の乾式（RN20-03）を通してから、--submit のときだけ投入する。
# 使い方: bash server_sde/_handoff_scripts_20261020/run_conf_v3.sh <selected_ckpt.json> <select.json> <decide.json> [dist_table.json] [--submit]
#   - dist_table.json は decide.json の adopt が true のときだけ渡す（契約がどちらの取り違えも拒否する）
#   - --submit は本人の合図のあとだけ（eval v3 を開く 1 回きり）。無ければ乾式で止まる
#   - #74（r15・2026-10-22）: 学習し直したアブレーションの ckpt は環境変数で渡す（選択器 2.1 の出力・札の照合つき）:
#       ABL_SCRATCH_SELCK=<selected_ckpt.json> ABL_SCRATCH_SEL=<select.json> ABL_NOHEIGHT_SELCK=... ABL_NOHEIGHT_SEL=... bash run_conf_v3.sh ...
#     欠ける顔ぶれは MISSING_OK='名前=理由;名前=理由' で理由を渡す（契約が --missing-ok に記録・黙って落とさない）
set -euo pipefail
cd /c/Users/satos/research/outdoor_seld_e2e
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONUTF8=1
SELCK="${1:?selected_ckpt.json}"; SEL="${2:?select.json}"; DEC="${3:?decide.json}"
TABLE=""; SUBMIT=0
for a in "${@:4}"; do case "$a" in --submit) SUBMIT=1;; *) TABLE="$a";; esac; done
for f in "$SELCK" "$SEL" "$DEC" ${TABLE:+"$TABLE"}; do [ -f "$f" ] || { echo "無い: $f"; exit 1; }; done
ckpt_of() {   # ckpt_of <selected_ckpt.json> <期待する label>
  $PY -c "import json,sys;J=json.load(open(sys.argv[1]));lab=J.get('label');(sys.exit('札が違う: ' + str(lab) + ' != ' + sys.argv[2]) if lab != sys.argv[2] else print(J['ckpt']))" "$1" "$2"
}
CK=$(ckpt_of "$SELCK" v2final)       # ~/PSELDNets_logs/...（ユーザー名は伏せてある）
STAMP=$(date +%Y%m%d_%H%M%S)
RDIR="out/v2final_conf_v3_inputs/$STAMP"
R_ABS="\$HOME/research/outdoor_seld_e2e/$RDIR"
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "mkdir -p ~/research/outdoor_seld_e2e/$RDIR"
scp -o ConnectTimeout=10 -q "$SEL" "is-server:research/outdoor_seld_e2e/$RDIR/select.json"
scp -o ConnectTimeout=10 -q "$DEC" "is-server:research/outdoor_seld_e2e/$RDIR/decide.json"
[ -z "$TABLE" ] || scp -o ConnectTimeout=10 -q "$TABLE" "is-server:research/outdoor_seld_e2e/$RDIR/dist_table.json"
ENV="V2FINAL_CKPT=$CK V2FINAL_SELECT_JSON=$R_ABS/select.json V2FINAL_DECIDE_JSON=$R_ABS/decide.json"
[ -z "$TABLE" ] || ENV="$ENV DIST_TABLE=$R_ABS/dist_table.json"
# アブレーション（学習し直した 2 本）: 選択器の出力を札で照合してから渡す
for TAG in scratch noheight; do
  U=$(echo "$TAG" | tr a-z A-Z)
  eval "ASELCK=\${ABL_${U}_SELCK:-}"; eval "ASEL=\${ABL_${U}_SEL:-}"
  [ -n "$ASELCK" ] || [ -n "$ASEL" ] || continue
  [ -f "$ASELCK" ] && [ -f "$ASEL" ] || { echo "無い: $ASELCK / $ASEL（$TAG は selected_ckpt.json と select.json の両方）"; exit 1; }
  ACK=$(ckpt_of "$ASELCK" "v2final_abl_$TAG")
  scp -o ConnectTimeout=10 -q "$ASEL" "is-server:research/outdoor_seld_e2e/$RDIR/select_$TAG.json"
  ENV="$ENV V2FINAL_ABL_${U}_CKPT=$ACK V2FINAL_ABL_${U}_SELECT_JSON=$R_ABS/select_$TAG.json"
  echo "アブレーション $TAG: $ACK"
done
[ -z "${MISSING_OK:-}" ] || ENV="$ENV V2FINAL_MISSING_OK='$MISSING_OK'"
echo "乾式（契約と封印の照合・推論なし）..."
# 2026-10-22 10:2x: 遠隔側の pipeline に pipefail（grep／sed が台本の exit 3 を隠し「乾式 OK」と出ていた）。乾式の最後の行 DRYRUN_OK が無ければ通らない扱い
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && set -o pipefail && $ENV V2FINAL_DRYRUN=1 bash server_sde/v2final_conf_v3.sbatch 2>&1 | grep -E 'OK ckpt|\[seal\]|\[preflight\]|DRYRUN_OK|NG |exit|アブレーション' | sed -E 's/s[0-9]{7}/<user>/g' | tee /tmp/conf_v3_dry.txt && grep -q DRYRUN_OK /tmp/conf_v3_dry.txt" || { echo "乾式が通らない。投入しない"; exit 3; }
if [ "$SUBMIT" != 1 ]; then echo "乾式 OK。投入は本人の合図のあと: 同じ引数（と同じ環境変数）に --submit を足す"; exit 0; fi
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && J=\$($ENV sbatch --parsable server_sde/v2final_conf_v3.sbatch) && echo \"conf_v3 job \$J\" && squeue -j \$J -o '%i %j %t %l %r'" | sed -E 's/s[0-9]{7}/<user>/g'
