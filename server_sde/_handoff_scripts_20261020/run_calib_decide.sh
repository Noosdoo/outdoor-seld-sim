#!/bin/bash
# 補正表（#72・凍結 r14）: calib_infer.sbatch の出力（calib 4,000 本・長尺 v1 400 本の因果推論）を手元に写し、来歴と ckpt を照合し、表を作り、
# 条件 (1)(2) を表あり・なしで比べ、採否を丸め前の数で判定する（decide.json = 確定評価の実行の契約が読む）。
# 使い方: bash server_sde/_handoff_scripts_20261020/run_calib_decide.sh <TAG> <selected_ckpt.json（run_selector_v29.sh の出力）>
#   TAG = calib_infer.sbatch が付けた run 名の末尾（例 outdoor_siren_v2final_v29_causal_ft_epoch_099）
# Astra RN20（2026-10-20）の直し:
#   RN20-04 サーバの run_meta.json を val_all_causal.run_meta.json に写し、短尺（4,000）も長尺（400）も _prediction_integrity --download で照合
#   RN20-05 表は全体の名簿（calib の plan 4,000 本）で予測を読み、fit の名簿（2,000 本）に絞って作る（--manifest-all）
#   RN20-06 ckpt の sha（2 つの run_meta・selected_ckpt.json）・検出の設定（0.7／0.3／20°）・長尺の frames（600）を照合してから進む。decide には表・ckpt・予測の sha を渡す
#   RN20-07 set -euo pipefail・出力の置き場は新規（既にあれば止まる）
# 検査用の上書き（本番は触らない）: CALIB_SRC_DIR（手元から cp・空 = scp）・CALIB_N・LONG_N・CALIB_GT・CALIB_PLANDIR・CALIB_OUT_ROOT・CALIB_PRED_ROOT
set -euo pipefail
cd /c/Users/satos/research/outdoor_seld_e2e
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONUTF8=1 PYTHONPATH=scripts:src
TAG="${1:?TAG（run 名の末尾）が要る}"
SELCK="${2:?selected_ckpt.json（run_selector_v29.sh の出力）が要る}"
[ -f "$SELCK" ] || { echo "selected_ckpt.json が無い: $SELCK"; exit 1; }
SRC="${CALIB_SRC_DIR:-}"
RUNS='~/PSELDNets_logs/outdoor_siren_v12/runs'
N_CAL="${CALIB_N:-4000}"; N_LONG="${LONG_N:-400}"
GT="${CALIB_GT:-out/_v29_calib_gt/metadata_dist}"
PLANDIR="${CALIB_PLANDIR:-out/dataset_outdoor_siren_v2final_calib/plan}"
O="${CALIB_OUT_ROOT:-out/v2final_calib_decide}/$TAG"
D="${CALIB_PRED_ROOT:-out/_v29_calib_pred}/$TAG"
[ ! -e "$O" ] || { echo "出力の置き場が既にある（古い結果と混ぜない）: $O"; exit 1; }
[ ! -e "$D" ] || { echo "写しの置き場が既にある（途中で止まった写しを使わない）: $D"; exit 1; }
mkdir -p "$O" "$D"
fetch() {   # <run 名> <サーバのファイル名> <手元のファイル>
  if [ -n "$SRC" ]; then cp "$SRC/$1/$2" "$3"; else scp -o ConnectTimeout=10 -q "is-server:$RUNS/$1/$2" "$3"; fi
}
for pair in "infer_calib_$TAG:$N_CAL" "infer_long_v1_$TAG:$N_LONG"; do
  name="${pair%%:*}"; n="${pair##*:}"; mkdir -p "$D/$name"
  fetch "$name" val_all_causal.csv "$D/$name/val_all_causal.csv"
  fetch "$name" val_all_causal.done.json "$D/$name/val_all_causal.done.json"
  fetch "$name" run_meta.json "$D/$name/val_all_causal.run_meta.json"
  $PY server_sde/_prediction_integrity.py "$D/$name/val_all_causal.csv" "$n" --download --done "$D/$name/val_all_causal.done.json" --meta "$D/$name/val_all_causal.run_meta.json"
  echo "$name: $(wc -l < "$D/$name/val_all_causal.csv") 行・来歴 OK（$n 本）"
done
CAL="$D/infer_calib_$TAG/val_all_causal.csv"; LONGP="$D/infer_long_v1_$TAG/val_all_causal.csv"
# ckpt・検出の設定・長尺の frames の照合（RN20-06）→ inputs_check.json
$PY - "$D/infer_calib_$TAG/val_all_causal.run_meta.json" "$D/infer_long_v1_$TAG/val_all_causal.run_meta.json" "$SELCK" "$O/inputs_check.json" "$N_LONG" <<'EOF'
import json, sys
from pathlib import Path
mc, ml, sk = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:4])
bad = []
if mc["ckpt_sha256"] != ml["ckpt_sha256"]: bad.append(f"calib と長尺の ckpt が違う（{mc['ckpt_sha256'][:12]} ≠ {ml['ckpt_sha256'][:12]}）")
if mc["ckpt_sha256"] != sk["ckpt_sha256"]: bad.append(f"推論の ckpt が選択器の選んだ ckpt と違う（{mc['ckpt_sha256'][:12]} ≠ {sk['ckpt_sha256'][:12]}）")
for tag, m in (("calib", mc), ("long", ml)):
    s = m.get("settings", {})
    if not (float(s.get("sed_threshold", -1)) == 0.7 and float(s.get("sed_stay", -1)) == 0.3 and float(s.get("sed_assoc", -1)) == 20.0 and s.get("hysteresis") is True):
        bad.append(f"{tag}: 検出の設定が登録の値（入り 0.7・保持 0.3・対応幅 20°）でない: {{k: s.get(k) for k in ('sed_threshold','sed_stay','sed_assoc','hysteresis')}}")
if int(ml.get("frames", 100)) != 600: bad.append(f"long: frames が 600 でない（{ml.get('frames')}）")
if not all(c.startswith("tune_syn_") for c in mc["clips"]): bad.append("calib: tune_syn_ 以外の本が混じっている")
if not all(c.startswith("long_") for c in ml["clips"]): bad.append("long: long_ 以外の本が混じっている")
if int(sys.argv[5]) != len(ml["clips"]): bad.append(f"long: 本数 {len(ml['clips'])} ≠ {sys.argv[5]}")
rec = {"ok": not bad, "problems": bad, "ckpt_sha256": mc["ckpt_sha256"], "ckpt_epoch": mc.get("ckpt_epoch"), "selected_epoch": sk.get("selected_epoch"),
       "calib_settings": {k: mc["settings"].get(k) for k in ("sed_threshold", "sed_stay", "sed_assoc", "hysteresis")}, "long_frames": ml.get("frames"),
       "n_calib": len(mc["clips"]), "n_long": len(ml["clips"])}
Path(sys.argv[4]).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(("OK " if not bad else "NG ") + json.dumps(rec, ensure_ascii=False)[:400])
sys.exit(0 if not bad else 6)
EOF
CKSHA=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['ckpt_sha256'])" "$SELCK")
# 1) 表を作る（fit の名簿・全体の名簿で読む・選択器と同じ対応）
$PY scripts/_dist_calib_v2.py fit --pred "$CAL" --gt "$GT" --manifest "$PLANDIR/manifest_calib_fit.csv" --manifest-all "$PLANDIR/assignment_v2final.csv" --ckpt-sha "$CKSHA" --out "$O/dist_table.json"
# 2) 条件 (1): calib_check を統一採点器で 表なし／あり（表は v2final の入力だけ）
$PY scripts/_score_unified.py "$O/unified_check_off.md" --strict --raw --plan "$PLANDIR/plan_calib_check.csv" --split tune --meta "$GT" v2final="$CAL" > "$O/_log_off.txt" 2>&1 || { echo "統一採点器（表なし）が失敗"; tail -5 "$O/_log_off.txt"; exit 4; }
$PY scripts/_score_unified.py "$O/unified_check_on.md" --strict --raw --dist-table "$O/dist_table.json" --dist-table-for v2final --plan "$PLANDIR/plan_calib_check.csv" --split tune --meta "$GT" v2final="$CAL" > "$O/_log_on.txt" 2>&1 || { echo "統一採点器（表あり）が失敗"; tail -5 "$O/_log_on.txt"; exit 4; }
# 3) 条件 (2): 長尺 v1 を fold40・41 別々に 表なし／あり（中/分 と 幹線歩道の安全な車への強）
for fold in fold40 fold41; do
  $PY scripts/_long_score_v2.py --pred "$LONGP" --split $fold --json "$O/long_${fold}_off.json" > "$O/long_${fold}_off.md"
  $PY scripts/_long_score_v2.py --pred "$LONGP" --split $fold --dist-table "$O/dist_table.json" --json "$O/long_${fold}_on.json" > "$O/long_${fold}_on.md"
  $PY scripts/_long_breakdown_v2.py --pred "$LONGP" --split $fold --scene arterial_walk --json "$O/break_${fold}_off.json" > "$O/break_${fold}_off.md"
  $PY scripts/_long_breakdown_v2.py --pred "$LONGP" --split $fold --scene arterial_walk --dist-table "$O/dist_table.json" --json "$O/break_${fold}_on.json" > "$O/break_${fold}_on.md"
done
# 4) 採否（丸め前の分子・分母・分母 0 は判定不能＝表なし・fold40 と 41 は両方・入力の一致と sha の記録）
$PY scripts/_dist_calib_v2.py decide --unified-off "$O/unified_check_off.md" --unified-on "$O/unified_check_on.md" \
  --long-off40 "$O/long_fold40_off.json" --long-on40 "$O/long_fold40_on.json" --long-off41 "$O/long_fold41_off.json" --long-on41 "$O/long_fold41_on.json" \
  --break-off40 "$O/break_fold40_off.json" --break-on40 "$O/break_fold40_on.json" --break-off41 "$O/break_fold41_off.json" --break-on41 "$O/break_fold41_on.json" \
  --label v2final --table "$O/dist_table.json" --ckpt-sha "$CKSHA" --pred-calib "$CAL" --pred-long "$LONGP" --out "$O/decide.json"
echo "-> $O/decide.json（採否・各条件の分子・分母・入力の sha）。採否を凍結 §11 に書く。次: run_conf_v3.sh <selected_ckpt.json> <select.json> $O/decide.json [$O/dist_table.json（採用のときだけ）]"
