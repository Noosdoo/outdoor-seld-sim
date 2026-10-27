#!/bin/bash
#SBATCH -J v30calib
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=normal
#SBATCH -t 6:00:00
#SBATCH --requeue
#SBATCH -o v30calib-%j.out
# 学習し直し v30 の距離の補正表の組（calib_fit 2,000 ＋ calib_check 2,000 = 4,000 本・仕様 §4 の 1・Fable 10/27 07:3x の 3）。calib_gen.sbatch（v29・job 7033）の写しで、
#   違うのは描画の入口（V2FINAL_RENDER=v30・既定 (ii)・帯域ゲート 0）・置き場（out/dataset_outdoor_siren_v2final_v30_calib）・plan は v29 の補正表の組の plan をそのまま写す
#   （同じ seed の描画規則の変更・plan の sha は md/design/calib_sets_v29.json の plan_sha256 と同じこと）・検査 A〜I・版の台帳は v30 と共通。
#   学習の置き場には何も書かない（前処理は作らない）。成果物の sha の一覧 CONTRACT_calib.json（封印ではない・来歴）。学習の入口の門は tune_syn_ を弾く。
#   乾式: V30_DRY=1。投入は Fable の OK のあと: sbatch server_sde/v30_calib_gen.sbatch（区画は -p で替えてよい）
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
RUN() { if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
unset V30_BG_SHAPE V30_BG_SLOPE_RANGE V30_BG_KNEE_RANGE V30_GATE_BAND_DB V2FINAL_ROWS_LIST V2FINAL_PROVISIONAL
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30_calib V2FINAL_RENDER=v30
OUT=out/$V2FINAL_DS_NAME
SRC=out/dataset_outdoor_siren_v2final_calib/plan
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF' || { echo "params が fixed でないか calib の設定（plan_sha256）が無い。中止"; exit 3; }
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8")); C = json.load(open("md/design/calib_sets_v29.json", encoding="utf-8"))
print("params status =", P.get("status"), "calib n_clips =", C.get("n_clips"), "plan_sha256 =", C.get("plan_sha256", "")[:16])
sys.exit(0 if P.get("status") == "fixed" and C.get("n_clips") and C.get("plan_sha256") else 3)
EOF
for f in $(tr -d '\r' < server_sde/v30_render_files.txt) $SRC/assignment_v2final.csv; do [ -f "$f" ] || { echo "無い: $f（配備してから）"; exit 3; }; done
sha256sum --quiet -c server_sde/v30_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v30_expected_sha256.txt) 本）"
$PY -c "import step11_v30_render as v;import sys;sys.exit(0 if all(v.FEATURES.values()) and v.GATE_BAND_DB == 0 and v.BG_SLOPE_RANGE == (-6.5, -2.5) else 3)" \
  || { echo "v30 の既定が (ii)・帯域ゲート 0・全部入 でない。中止"; exit 3; }
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && REUSE=1
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ] && [ "${REUSE:-0}" != 1 ]; then echo "出力先 $OUT に既に本がある（続きから回すなら REUSE=1）"; exit 1; fi
RUN mkdir -p $OUT/plan
for f in assignment_v2final.csv manifest_calib_strata.csv manifest_v26_bellbike.csv; do
  [ -f "$SRC/$f" ] || continue
  if [ -f "$OUT/plan/$f" ]; then cmp -s "$OUT/plan/$f" "$SRC/$f" || { echo "$OUT/plan/$f が元の plan と違う → 止める"; exit 3; }; else RUN cp "$SRC/$f" "$OUT/plan/$f"; fi
done
N=$($PY -c "import json;print(json.load(open('md/design/calib_sets_v29.json'))['n_clips'])")
[ "$(($(wc -l < $SRC/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
grep -c "^tune_syn_clip" $SRC/assignment_v2final.csv | grep -qx "$N" || { echo "plan の番号が全部 tune_syn_ でない"; exit 3; }
echo "$($PY -c "import json;print(json.load(open('md/design/calib_sets_v29.json'))['plan_sha256'])")  $SRC/assignment_v2final.csv" | sha256sum --quiet -c - || { echo "plan の sha が calib_sets_v29.json と違う"; exit 3; }
echo "N=$N OUT=$OUT job=${SLURM_JOB_ID:-dry}"
RUN $PY scripts/_run_v2final_gen.py --list
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=${V30_SHARDS:-12}
pids=()
for i in $(seq 0 $((SH - 1))); do
  if [ "${V30_DRY:-0}" = 1 ]; then [ $i -lt 2 ] && echo "[dry] $PY scripts/_run_v2final_gen.py --shard $i/$SH（…$SH 本）"; continue; fi
  $PY scripts/_run_v2final_gen.py --shard $i/$SH > v30calib_shard_${i}_job${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] 距離ラベル → D14（と --check）→ 検査 A〜I → 本数 → CONTRACT_calib.json"; echo V30_CALIB_DRY_DONE; exit 0; fi
ls $OUT | grep -c "^GEN_VERSIONS_" && { echo "版の記録が途中で変わった。中止"; exit 7; }
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルに失敗。中止"; exit 5; }
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜I）に NG。中止"; exit 7; }
for d in foa metadata metadata_dist masks; do n=$(ls $OUT/$d | wc -l); [ "$n" = "$N" ] || { echo "本数が違う: $d $n ≠ $N"; exit 7; }; done
$PY - "$OUT" "$N" <<'EOF'
import hashlib, json, sys, datetime
from pathlib import Path
out = Path(sys.argv[1]); files = {}
for d in ("plan", "foa", "metadata", "metadata_dist", "masks"):
    for p in sorted((out / d).glob("*")):
        if p.is_file():
            files[f"{d}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
files["GEN_VERSIONS.txt"] = hashlib.sha256((out / "GEN_VERSIONS.txt").read_bytes()).hexdigest()
c = dict(name=out.name, n_clips=int(sys.argv[2]), made_at=datetime.datetime.now().isoformat(timespec="seconds"), n_files=len(files),
         rule="v30 の補正表の組（変更票 #72・#75）。表は calib_fit（00001〜02000）で作り、条件 (1) は calib_check（02001〜04000）で見る。学習・検証・確定評価には使わない（入口の門が tune_syn_ を弾く）", files=files)
(out / "CONTRACT_calib.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
print("CONTRACT_calib.json", len(files))
EOF
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt $OUT/CONTRACT_calib.json | tee v30calib_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo V30_CALIB_DONE
