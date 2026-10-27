#!/bin/bash
#SBATCH -J v30evalv4
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o v30evalv4-%j.out
# 確定評価セット eval v4（仕様 §4 の 1・§3-4・変更票 #75・Fable 10/27 07:3x の 3）: eval v3 と**同じ plan（6,000 本・同じ seed）を v30 の描画規則で描き直す**
#   （仕様「v3 → v4 は同じ seed の描画規則の変更と記録し、旧封印は保持」）。eval_v3_gen.sbatch（job 6992）の写しで、違うのは描画の入口（V2FINAL_RENDER=v30）・置き場
#   （out/dataset_outdoor_siren_v2final_v30_eval_v4）・plan は eval v3 の置き場の plan/ を写すだけ（音・ラベル・契約には触らない・写した plan が元と同じことを cmp で確かめる）・
#   検査 A〜I・契約の名前 CONTRACT_eval_v4.json と SEALED.txt・**契約を書いた直後に server_sde/_eval_seal.py write（封印の記録 eval_v4_seal.json）**。
#   seal.json は手元に取って md/design/eval_v4_seal.json としてコミットする（Opus）。**封印のあと eval v4 は開けない**（本人 10/26 14:15・開封は本人の別の合図）。
#   学習の置き場には何も書かない（前処理は作らない）。乾式: V30_DRY=1。投入は Fable の OK のあと: sbatch server_sde/v30_eval_v4_gen.sbatch（区画は -p で替えてよい）
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
RUN() { if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
unset V30_BG_SHAPE V30_BG_SLOPE_RANGE V30_BG_KNEE_RANGE V30_GATE_BAND_DB V2FINAL_ROWS_LIST V2FINAL_PROVISIONAL
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30_eval_v4 V2FINAL_RENDER=v30
OUT=out/$V2FINAL_DS_NAME
SRC=out/dataset_outdoor_siren_v2final_eval_v3/plan
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF' || { echo "params が fixed でないか EVAL_V3（本数の鍵）が無い。中止"; exit 3; }
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
print("params status =", P.get("status"), "EVAL_V3 n =", (P.get("EVAL_V3") or {}).get("n_clips"))
sys.exit(0 if P.get("status") == "fixed" and (P.get("EVAL_V3") or {}).get("n_clips") else 3)
EOF
for f in $(tr -d '\r' < server_sde/v30_render_files.txt) $SRC/assignment_v2final.csv; do [ -f "$f" ] || { echo "無い: $f（配備してから）"; exit 3; }; done
sha256sum --quiet -c server_sde/v30_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v30_expected_sha256.txt) 本）"
$PY -c "import step11_v30_render as v;import sys;sys.exit(0 if all(v.FEATURES.values()) and v.GATE_BAND_DB == 0 and v.BG_SLOPE_RANGE == (-6.5, -2.5) else 3)" \
  || { echo "v30 の既定が (ii)・帯域ゲート 0・全部入 でない。中止"; exit 3; }
[ ! -f "$OUT/SEALED.txt" ] || { echo "既に封印済み（$OUT/SEALED.txt）。作り直すなら新しい名前で"; exit 1; }
[ ! -e "$OUT/eval_v4_seal.json" ] || { echo "封印の記録が既にある（$OUT/eval_v4_seal.json）。作り直すなら新しい名前で"; exit 1; }
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && REUSE=1
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ] && [ "${REUSE:-0}" != 1 ]; then echo "出力先 $OUT に既に本がある（続きから回すなら REUSE=1）"; exit 1; fi
RUN mkdir -p $OUT/plan
for f in assignment_v2final.csv assignment_eval_v3.csv manifest_eval_v3_strata.csv manifest_v26_bellbike.csv; do
  [ -f "$SRC/$f" ] || continue
  if [ -f "$OUT/plan/$f" ]; then cmp -s "$OUT/plan/$f" "$SRC/$f" || { echo "$OUT/plan/$f が eval v3 の plan と違う → 止める"; exit 3; }; else RUN cp "$SRC/$f" "$OUT/plan/$f"; fi
done
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['EVAL_V3']['n_clips'])")
[ "$(($(wc -l < $SRC/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
grep -c "^eval_syn_clip" $SRC/assignment_v2final.csv | grep -qx "$N" || { echo "plan の番号が全部 eval_syn_ でない"; exit 3; }
echo "eval v3 の plan の sha: $(sha256sum $SRC/assignment_v2final.csv | cut -c1-16)（v4 はこの plan をそのまま使う）N=$N OUT=$OUT job=${SLURM_JOB_ID:-dry}"
RUN $PY scripts/_run_v2final_gen.py --list
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=${V30_SHARDS:-48}
pids=()
for i in $(seq 0 $((SH - 1))); do
  if [ "${V30_DRY:-0}" = 1 ]; then [ $i -lt 2 ] && echo "[dry] $PY scripts/_run_v2final_gen.py --shard $i/$SH（…$SH 本）"; continue; fi
  $PY scripts/_run_v2final_gen.py --shard $i/$SH > v30evalv4_shard_${i}_job${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] 距離ラベル → D14（と --check）→ 検査 A〜I → 本数（work も）→ CONTRACT_eval_v4.json・SEALED.txt → _eval_seal.py write → eval_v4_seal.json"; echo V30_EVAL_V4_DRY_DONE; exit 0; fi
ls $OUT | grep -c "^GEN_VERSIONS_" && { echo "版の記録が途中で変わった。封印しない。中止"; exit 7; }
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルに失敗。中止"; exit 5; }
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜I）に NG。封印しない。中止"; exit 7; }
for d in foa metadata metadata_dist masks; do n=$(ls $OUT/$d | wc -l); [ "$n" = "$N" ] || { echo "$d が $n 本（N=$N）。中止"; exit 7; }; done
[ "$(ls $OUT/work | wc -l)" = "$N" ] || { echo "work が N でない。中止"; exit 7; }
$PY - "$OUT" "$N" <<'EOF'
import hashlib, json, sys, datetime
from pathlib import Path
out = Path(sys.argv[1]); n = int(sys.argv[2])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files = {}
for d in ("foa", "metadata", "metadata_dist", "masks"):
    for p in sorted((out / d).iterdir()):
        files[f"{d}/{p.name}"] = sha(p)
files["plan/assignment_v2final.csv"] = sha(out / "plan/assignment_v2final.csv")
files["GEN_VERSIONS.txt"] = sha(out / "GEN_VERSIONS.txt")
c = {"name": out.name, "n_clips": n, "sealed_at": datetime.datetime.now().isoformat(timespec="seconds"), "n_files": len(files),
     "rule": "確定評価の 1 回まで開かない（本人の別の合図・当面は開けない＝本人 10/26 14:15）。metadata を読む採点・予測の閲覧もその 1 回だけ。学習・検証・調整に使わない（学習の入口の門が eval_ を弾く）。eval v3 と同じ plan・同じ seed を v30 の描画規則で描いた（v3 の封印は保持）",
     "files": files}
(out / "CONTRACT_eval_v4.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "SEALED.txt").write_text(f"SEALED {c['sealed_at']} n={n} contract_sha256={hashlib.sha256(json.dumps(c, ensure_ascii=False, indent=1).encode()).hexdigest()}\n", encoding="utf-8")
print("SEALED", n, len(files))
EOF
$PY server_sde/_eval_seal.py write --ds $OUT --out $OUT/eval_v4_seal.json || { echo "封印の記録（seal.json）を書けなかった（規約の混在など）。中止"; exit 9; }
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt $OUT/CONTRACT_eval_v4.json $OUT/eval_v4_seal.json | tee v30evalv4_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo V30_EVAL_V4_DONE
