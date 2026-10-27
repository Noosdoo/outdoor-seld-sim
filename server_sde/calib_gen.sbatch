#!/bin/bash
#SBATCH -J calibgen
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=normal
#SBATCH -t 3:00:00
#SBATCH --requeue
#SBATCH -o calibgen-%j.out
# 距離の補正表の組（calib_fit 2,000 ＋ calib_check 2,000 = 4,000 本・変更票 #72・本人 2026-10-20 03:34 (c)）を v29 の生成器で描く。
#   - plan = out/dataset_outdoor_siren_v2final_calib/plan/assignment_v2final.csv（step10_calib_sets_plan.py・番号 tune_syn_clip00001〜04000・split = tune）。
#   - 描画の連なりは学習（v29）と同一（_run_v2final_gen.py・V2FINAL_DS_NAME が _calib で終わると総数は md/design/calib_sets_v29.json の n_clips）。
#   - 区画は --shard i/n（行を 1 つおきに配る・2026-10-20・尻尾が出ない）。学習の置き場（$PS/datasets_*・$PS/_hdf5_*）には何も書かない。前処理（HDF5）は作らない。
#   - 終わったら 距離ラベル → D14 → _v26_regen_check.py（A〜H）→ 成果物の sha の一覧 CONTRACT_calib.json（封印ではない・来歴）。学習の入口の門は tune_syn_ を弾く。
#   - 区画は a100_3g（CPU 12・GPU は使わない）: 保守明けの学習（pro_6000）の邪魔をしないため。投入は本人の許可のあと: sbatch server_sde/calib_gen.sbatch
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_calib
OUT=out/$V2FINAL_DS_NAME
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF'
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8")); C = json.load(open("md/design/calib_sets_v29.json", encoding="utf-8"))
print("params status =", P.get("status"), "calib n_clips =", C.get("n_clips"), "plan_sha256 =", C.get("plan_sha256", "")[:16])
sys.exit(0 if P.get("status") == "fixed" and C.get("n_clips") and C.get("plan_sha256") else 3)
EOF
[ $? -eq 0 ] || { echo "params が fixed でないか calib の設定（plan_sha256）が無い。中止"; exit 3; }
sha256sum --quiet -c server_sde/calib_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/calib_expected_sha256.txt) 本）"
[ -f "$OUT/plan/assignment_v2final.csv" ] || { echo "plan が無い（$OUT/plan/assignment_v2final.csv・配備する）"; exit 3; }
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ]; then echo "出力先 $OUT に既に本がある。続きから回すなら REUSE=1"; [ "${REUSE:-0}" = 1 ] || exit 1; fi
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && REUSE=1
N=$($PY -c "import json;print(json.load(open('md/design/calib_sets_v29.json'))['n_clips'])")
[ "$(($(wc -l < $OUT/plan/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
grep -c "^tune_syn_clip" $OUT/plan/assignment_v2final.csv | grep -qx "$N" || { echo "plan の番号が全部 tune_syn_ でない"; exit 3; }
echo "$($PY -c "import json;print(json.load(open('md/design/calib_sets_v29.json'))['plan_sha256'])")  $OUT/plan/assignment_v2final.csv" | sha256sum --quiet -c - || { echo "plan の sha が calib_sets_v29.json と違う"; exit 3; }
$PY scripts/_run_v2final_gen.py --list | tee calib_gen_list_job${SLURM_JOB_ID}.log
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=6
pids=()
for i in $(seq 0 $((SH - 1))); do
  $PY scripts/_run_v2final_gen.py --shard $i/$SH > calib_shard_$i.log 2>&1 &
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
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜H）に NG。中止"; exit 7; }
for d in foa metadata metadata_dist masks; do n=$(ls $OUT/$d | wc -l); [ "$n" = "$N" ] || { echo "本数が違う: $d $n ≠ $N"; exit 7; }; done
$PY - "$OUT" <<'EOF'
import hashlib, json, sys, datetime
from pathlib import Path
out = Path(sys.argv[1]); files = {}
for d in ("plan", "foa", "metadata", "metadata_dist", "masks"):
    for p in sorted((out / d).glob("*")):
        if p.is_file():
            files[f"{d}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
files["GEN_VERSIONS.txt"] = hashlib.sha256((out / "GEN_VERSIONS.txt").read_bytes()).hexdigest()
c = dict(name=out.name, n_clips=4000, made_at=datetime.datetime.now().isoformat(timespec="seconds"), n_files=len(files),
         rule="補正表の組（変更票 #72）。表は calib_fit（00001〜02000）で作り、条件 (1) は calib_check（02001〜04000）で見る。学習・検証・確定評価には使わない（入口の門が tune_syn_ を弾く）", files=files)
(out / "CONTRACT_calib.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
print("CONTRACT_calib.json", len(files))
EOF
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt | tee calib_gen_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo CALIB_GEN_DONE
