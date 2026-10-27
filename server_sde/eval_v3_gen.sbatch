#!/bin/bash
#SBATCH -J evalv3gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH -o evalv3gen-%j.out
# 確定評価セット eval v3（本人決定 2026-10-13 15:06〜15:13・変更票 #65・凍結 r11）: plan = out/dataset_outdoor_siren_v2final_eval_v3/plan/assignment_v2final.csv（6,000 本・step10_eval_v3_plan.py）。
#   - 描画の連なりは学習（v27）と同一（_run_v2final_gen.py・V2FINAL_DS_NAME に _eval_ を含むと総数は params の EVAL_V3.n_clips）。
#   - 学習の置き場（$PS/datasets_*・$PS/_hdf5_*）には何も書かない。前処理（HDF5）は作らない（確定評価の推論は FOA と metadata を直接読む）。
#   - 終わったら 距離ラベル → D14（v12train の crossing 行を落とす）→ _v26_regen_check.py（A〜H）→ **封印 = CONTRACT_eval_v3.json**（plan・GEN_VERSIONS・foa／metadata／metadata_dist／masks の全ファイルの sha256）と SEALED.txt。
#   - 封印のあとは確定評価の 1 回（v2final_conf_v3.sbatch・v1 と v2）まで誰も開かない（metadata を読む採点も、予測を見るのも、その 1 回だけ）。
#   - 台帳 = server_sde/eval_v3_expected_sha256.txt（配備のときサーバ側で作る）。投入は本人の許可のあと: sbatch server_sde/eval_v3_gen.sbatch（区画は -p で替えてよい）
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_eval_v3
OUT=out/$V2FINAL_DS_NAME
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF'
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
print("params status =", P.get("status"), "EVAL_V3 =", (P.get("EVAL_V3") or {}).get("n_clips"))
sys.exit(0 if P.get("status") == "fixed" and (P.get("EVAL_V3") or {}).get("adopt") else 3)
EOF
[ $? -eq 0 ] || { echo "params が fixed でないか EVAL_V3 が無い。中止"; exit 3; }
sha256sum --quiet -c server_sde/eval_v3_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/eval_v3_expected_sha256.txt) 本）"
[ -f "$OUT/plan/assignment_v2final.csv" ] || { echo "評価セットの plan が無い（$OUT/plan/assignment_v2final.csv・配備する）"; exit 3; }
[ ! -f "$OUT/SEALED.txt" ] || { echo "既に封印済み（$OUT/SEALED.txt）。作り直すなら新しい名前で"; exit 1; }
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ]; then echo "出力先 $OUT に既に本がある。続きから回すなら REUSE=1"; [ "${REUSE:-0}" = 1 ] || exit 1; fi
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && REUSE=1
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['EVAL_V3']['n_clips'])")
[ "$(($(wc -l < $OUT/plan/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
grep -c "^eval_syn_clip" $OUT/plan/assignment_v2final.csv | grep -qx "$N" || { echo "plan の番号が全部 eval_syn_ でない"; exit 3; }
$PY scripts/_run_v2final_gen.py --list | tee eval_v3_gen_list_job${SLURM_JOB_ID}.log
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=12; STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v2final_gen.py --rows $lo-$hi > eval_v3_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls $OUT/foa | wc -l
cat $OUT/GEN_VERSIONS.txt
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルの生成に失敗。中止"; exit 5; }
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 のラベル変換に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜H）に NG。封印しない。中止"; exit 7; }
# 本数の照合（foa・metadata・metadata_dist・masks・work/scene.json が全部 N）
for d in foa metadata metadata_dist masks; do n=$(ls $OUT/$d | wc -l); [ "$n" = "$N" ] || { echo "$d が $n 本（N=$N）。中止"; exit 7; }; done
[ "$(ls $OUT/work | wc -l)" = "$N" ] || { echo "work が N でない。中止"; exit 7; }
# 封印: 全ファイルの sha256 を CONTRACT に
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
     "rule": "確定評価の 1 回（v2final_conf_v3.sbatch・v1 と v2・介入なし）まで開かない。metadata を読む採点・予測の閲覧もその 1 回だけ。学習・検証・調整に使わない（学習の入口の門が eval_ を弾く）",
     "files": files}
(out / "CONTRACT_eval_v3.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "SEALED.txt").write_text(f"SEALED {c['sealed_at']} n={n} contract_sha256={hashlib.sha256(json.dumps(c, ensure_ascii=False, indent=1).encode()).hexdigest()}\n", encoding="utf-8")
print("SEALED", n, len(files))
EOF
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt $OUT/CONTRACT_eval_v3.json | tee eval_v3_gen_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo EVAL_V3_GEN_DONE
