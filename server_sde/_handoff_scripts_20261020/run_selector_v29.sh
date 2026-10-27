#!/bin/bash
# 因果 ft が終わったあと: 候補 8 個の予測（val 12,420 本・selfcausal）をサーバから手元へ写し、来歴を照合し、選択器 2.1 を回す（凍結 §4・r11〜r14）。
# 使い方: bash server_sde/_handoff_scripts_20261020/run_selector_v29.sh [s1|s2]   （空 = 主 seed）
#   run 名は v2final_causal.sbatch の決まり（infer_v2finalft<SEED_SUF>_e<NNN>_selfcausal・RN20-02）から作る。手で名前を渡さない。
# Astra RN20（2026-10-20）の直し:
#   RN20-04 サーバの run_meta.json を手元の val_all_causal.run_meta.json に写す（選択器は <csv>.run_meta.json を見る）・写した 3 点を _prediction_integrity --download で照合
#   RN20-07 set -euo pipefail・選択器の失敗はそのまま失敗・出力は実行ごとの新しい置き場（古い結果を指さない）・selected と not_for_registration を確かめてから selected_ckpt.json を書く
# 検査用の上書き（本番は触らない）: SEL_SRC_DIR（手元のディレクトリから cp・空 = scp）・SEL_MANIFEST・SEL_GT・SEL_HEIGHT_TABLE・SEL_EXPECTED・SEL_OUT_ROOT・SEL_CAND_ROOT
set -euo pipefail
cd /c/Users/satos/research/outdoor_seld_e2e
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONUTF8=1
SEED="${1:-}"; ABL="${2:-}"
case "$SEED" in ""|s1|s2) ;; *) echo "種の札は 空・s1・s2（$SEED）"; exit 1;; esac
case "$ABL" in ""|scratch|noheight) ;; *) echo "アブレーションの札は 空・scratch・noheight（$ABL）"; exit 1;; esac
SUF=""; [ -z "$SEED" ] || SUF="_$SEED"
[ -z "$ABL" ] || SUF="${SUF}_abl_$ABL"                       # #74: 名前の決まりは v2final_causal.sbatch の OUTPFX と同じ（infer_v2finalft<_seed><_abl_tag>）
PREFIX="infer_v2finalft${SUF}"; LABEL="v2final${SUF}"
HT_ARGS=(--height-table "${SEL_HEIGHT_TABLE:-out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv}")
[ "$ABL" != "noheight" ] || HT_ARGS=(--height-none)           # noheight は高さの入力なしの候補（run_meta の height.on=false）を受ける
MANIFEST="${SEL_MANIFEST:-out/dataset_outdoor_siren_v2final/plan/manifest_val_syn_v2final.csv}"
GT="${SEL_GT:-out/_v29_val_gt/metadata_dist}"
EXPECTED="${SEL_EXPECTED:-$(( $(wc -l < "$MANIFEST") - 1 ))}"
SRC="${SEL_SRC_DIR:-}"
RUNS='~/PSELDNets_logs/outdoor_siren_v12/runs'
STAMP=$(date +%Y%m%d_%H%M%S)
O="${SEL_OUT_ROOT:-out/v2final_select_v29}/${LABEL}_$STAMP"
DST="${SEL_CAND_ROOT:-out/_v29_candidates}/${LABEL}_$STAMP"
mkdir -p "$O" "$DST"
fetch() {   # <run 名> <サーバのファイル名> <手元のファイル>
  if [ -n "$SRC" ]; then cp "$SRC/$1/$2" "$3"; else scp -o ConnectTimeout=10 -q "is-server:$RUNS/$1/$2" "$3"; fi
}
for e in 019 039 059 079 099 119 139 149; do
  R="${PREFIX}_e${e}_selfcausal"; mkdir -p "$DST/e$e"
  fetch "$R" val_all_causal.csv "$DST/e$e/val_all_causal.csv"
  fetch "$R" val_all_causal.done.json "$DST/e$e/val_all_causal.done.json"
  fetch "$R" run_meta.json "$DST/e$e/val_all_causal.run_meta.json"
  $PY server_sde/_prediction_integrity.py "$DST/e$e/val_all_causal.csv" "$EXPECTED" --download --done "$DST/e$e/val_all_causal.done.json" --meta "$DST/e$e/val_all_causal.run_meta.json"
  echo "e$e: $(wc -l < "$DST/e$e/val_all_causal.csv") 行・来歴 OK"
done
$PY scripts/_v2_ckpt_selector.py --pattern "$DST/e{epoch:03d}/val_all_causal.csv" --manifest "$MANIFEST" --gt "$GT" \
  --expected "$EXPECTED" "${HT_ARGS[@]}" --label "$LABEL" --out "$O/select.json"
$PY - "$O/select.json" "$DST" "$O/selected_ckpt.json" <<'EOF'
import json, re, sys
from pathlib import Path
sel_p, dst, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = json.loads(sel_p.read_text(encoding="utf-8"))
if S.get("not_for_registration") is not False or type(S.get("selected")) is not int:
    print(f"NG 選択器の結果が登録に使えない／選べていない（not_for_registration={S.get('not_for_registration')}・selected={S.get('selected')!r}）"); sys.exit(7)
e = S["selected"]
m = json.loads((dst / f"e{e:03d}" / "val_all_causal.run_meta.json").read_text(encoding="utf-8"))
if S["candidates"][str(e)]["ckpt_sha256"] != m["ckpt_sha256"]:
    print("NG 選んだ候補の ckpt の sha が run_meta と違う"); sys.exit(7)
redact = lambda s: re.sub(r"/home/[^/]+/", "~/", str(s))        # サーバのユーザー名を残さない
rec = {"label": S["label"], "selected_epoch": e, "ckpt": redact(m["ckpt"]), "ckpt_sha256": m["ckpt_sha256"], "ckpt_epoch": m["ckpt_epoch"],
       "select_json": str(sel_p), "spec_version": S["spec_version"], "n_clips": S["n_clips"], "order": S["order"]}
out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for p in (sel_p, sel_p.with_suffix(".md")):                      # 選択器の出力にも候補の ckpt のパスが入る → 同じく伏せる（sha は変わらない）
    t = p.read_text(encoding="utf-8"); p.write_text(redact(t), encoding="utf-8")
print(f"OK 選んだ epoch e{e:03d}・ckpt {rec['ckpt']}（sha {m['ckpt_sha256'][:12]}）・順位 {' > '.join(f'e{x:03d}' for x in S['order'])}")
EOF
echo "-> $O/select.json（同名 .md）・$O/selected_ckpt.json（次: calib_infer.sbatch に V2FINAL_CKPT=<ckpt の絶対パス>・run_calib_decide.sh に selected_ckpt.json）"
