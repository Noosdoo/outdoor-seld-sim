#!/bin/bash
#SBATCH -J sensvar
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH -o sensvar-%j.out
# 感度の変種（〔仮〕の値を学習し直さず推論だけで測る・2026-10-21 Fable・本人 13:57「それでお願いします」・23:39「お願いします」）。
#   描く（定数の上書き = SENS_OVERRIDES.json）→ 距離ラベル → D14 変換＋検査 → 描き直しの検査 A〜I → 固定した ckpt で推論（提案法と同じ条件）→ 採点（統一採点器・知覚層）。
#   基準（上書き無し）と変種を**同じ ckpt・同じ plan（場面と種）**で回し、手元の scripts/sens_variant.py compare で差を見る。確定評価が 1 回出たあとに使う。
# 使い方: SENS_TAG=<tag> V2FINAL_CKPT=<確定評価と同じ ckpt> [DIST_TABLE=<採用した補正表>] [SENS_SHARDS=8] sbatch server_sde/sens_variant.sbatch
#   先に手元で作って配備: out/dataset_outdoor_siren_v2final_eval_sens_<tag>/{plan/assignment_v2final.csv, SENS_PLAN.json, SENS_OVERRIDES.json}（scripts/sens_variant.py plan／overrides）
#   触らない置き場: 封印の eval v3・学習の束（dataset_outdoor_siren_v2final）・補正表の組（_calib）。出力は out/<置き場>/ と runs/sens_v29/infer_sens_<tag> だけ。
#   SENS_DRYRUN=1 = 点検と --list だけ（描かない・推論しない）。SENS_E2E／SENS_PS／SENS_PY は手元の検査（scripts/test_sens_variant.py）用の置き換え。
set -exo pipefail
PS=${SENS_PS:-$HOME/research/PSELDNet/PSELDNets}
E2E=${SENS_E2E:-$HOME/research/outdoor_seld_e2e}
PY=${SENS_PY:-$PS/.venv/bin/python}
DRY=${SENS_DRYRUN:-0}
cd "$E2E"
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
TAG=${SENS_TAG:-}
[ -n "$TAG" ] || { echo "SENS_TAG を渡す"; exit 3; }
case "$TAG" in *[!A-Za-z0-9_]*) echo "SENS_TAG は英数字と _ だけ（$TAG）"; exit 3 ;; esac
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_eval_sens_$TAG
OUT=$E2E/out/$V2FINAL_DS_NAME
CP=$OUT/plan/assignment_v2final.csv
[ -f "$CP" ] && [ -f "$OUT/SENS_OVERRIDES.json" ] && [ -f "$OUT/SENS_PLAN.json" ] || { echo "plan・SENS_OVERRIDES.json・SENS_PLAN.json を $OUT に配備してから"; exit 3; }
[ ! -f "$OUT/SEALED.txt" ] || { echo "この置き場は封印されている（$OUT）。感度の束には使わない"; exit 3; }
[ -n "${V2FINAL_CKPT:-}" ] && [ -f "$V2FINAL_CKPT" ] || { echo "V2FINAL_CKPT（確定評価と同じ ckpt）を渡す"; exit 3; }
[ -z "${DIST_TABLE:-}" ] || [ -f "$DIST_TABLE" ] || { echo "DIST_TABLE が無い: $DIST_TABLE"; exit 3; }
N=$(($(wc -l < "$CP") - 1))
[ "$N" -ge 1 ] || { echo "plan が空"; exit 3; }
grep -c "^eval_syn_clip9" "$CP" | grep -qx "$N" || { echo "plan の番号が全部 eval_syn_clip9NNNN でない（封印の eval v3 の番号と混ざる）"; exit 3; }
source "$E2E/server_sde/_baseline_guard.sh"
guard_record
source "$E2E/server_sde/_v2final_members.sh"
RUNS=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/sens_v29
OUT2=infer_sens_${TAG}
export PYTHONPATH=scripts:src
# 上書きの記録を当てて plan を読めるか（無い定数・種類違い・記録の『元の値』とコードの差 → ここで止まる）
"$PY" scripts/_sens_render.py --list | tee "sens_${TAG}_list_job${SLURM_JOB_ID:-dry}.log"
if [ "$DRY" = 1 ]; then echo "SENS_VARIANT_DRYRUN_OK（描かない・推論しない）"; exit 0; fi
# ---- 描く（foa が既に N 本なら飛ばす = 再開。行ごとに決定的・already_done も行ごと）
if [ "$(ls "$OUT/foa" 2>/dev/null | wc -l)" != "$N" ]; then
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  SH=${SENS_SHARDS:-8}; pids=()
  for i in $(seq 0 $((SH - 1))); do
    "$PY" scripts/_sens_render.py --shard $i/$SH > "sens_${TAG}_shard_$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  [ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
fi
[ "$(ls "$OUT/foa" | wc -l)" = "$N" ] || { echo "foa が N 本（$N）でない"; exit 1; }
cat "$OUT/GEN_VERSIONS.txt" | head -n 3
"$PY" scripts/step19_dist_labels_v2final.py --ds "$OUT" | tail -2 || { echo "距離ラベルの生成に失敗。中止"; exit 5; }
"$PY" scripts/step10_v2final_relabel.py --ds "$OUT" || { echo "D14 のラベル変換に失敗。中止"; exit 4; }
"$PY" scripts/step10_v2final_relabel.py --ds "$OUT" --check || { echo "D14 の検査に失敗。中止"; exit 4; }
# 描き直しの検査 A〜I。小さな束なので I の「車の種別の割合」だけは NG でも進む（記録に残す）。ほかの NG は止める
set +e; "$PY" scripts/_v26_regen_check.py --ds "$OUT" --no-audio --n-listen 0 > "sens_${TAG}_check_job${SLURM_JOB_ID:-dry}.log" 2>&1; RC=$?; set -e
if [ $RC -ne 0 ]; then
  OTHER=$(grep -E "^NG " "sens_${TAG}_check_job${SLURM_JOB_ID:-dry}.log" | grep -vE "^NG I: 車の種別の割合" || true)
  [ -z "$OTHER" ] || { echo "描き直しの検査に NG（割合以外）。中止:"; echo "$OTHER"; exit 7; }
  echo "注意: 検査 I の割合だけ NG（小さな束・記録に残す）"
fi
for d in foa metadata metadata_dist masks; do n=$(ls "$OUT/$d" | wc -l); [ "$n" = "$N" ] || { echo "$d が $n 本（N=$N）。中止"; exit 7; }; done
# ---- 推論（提案法と同じ条件: HEIGHT_COND 1・高さの表 = plan の mic_z・保持 0.3・固定した ckpt）
cd "$PS"
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p "$JOBDIR"
cp "$E2E/server_sde/height_patch.py" "$PS/height_patch.py"
cp "$E2E/server_sde/causal_patch.py" "$PS/causal_patch.py"
cp "$E2E/server_sde/_causal_infer_v17.py" "$JOBDIR/_causal_infer_v17.py"
cp "$E2E/server_sde/_prediction_integrity.py" "$JOBDIR/_prediction_integrity.py"
cp "$E2E/server_sde/hysteresis_patch.py" "$JOBDIR/hysteresis_patch.py"
export PS E2E JOBDIR PY RUNS N
export CAUSAL_FOA=$OUT/foa CAUSAL_PREFIX=eval_syn_ CAUSAL_GLOB="eval_syn_*.flac"
mkdir -p "$RUNS"
member_absent "$OUT2" || exit 4
member_v17 "$OUT2" "$V2FINAL_CKPT" 1 "$CP" 0.3 || exit 5
cd "$E2E"
# ---- 採点（確定評価と同じ採点器・同じ掛け方。補正表は渡したときだけ・v2final の入力だけに掛ける）
CG=$OUT/metadata_dist; CU=$OUT/unified; mkdir -p "$CU"
PRED=$RUNS/$OUT2/conf_all.csv
DT_ARG=""; [ -z "${DIST_TABLE:-}" ] || DT_ARG="--dist-table $DIST_TABLE --dist-table-for v2final"
"$PY" scripts/_score_unified.py "$CU/unified_sens.md" --strict --raw $DT_ARG --plan "$CP" --split eval --meta "$CG" v2final="$PRED" > "$CU/_log_unified.txt" 2>&1 || { echo "統一採点器に失敗"; tail -n 5 "$CU/_log_unified.txt"; exit 6; }
"$PY" scripts/_perception_eval.py "$CU/perception_sens.md" --plan "$CP" --split eval --meta "$CG" --json "$CU/perception_sens.json" --strict v2final="$PRED" > "$CU/_log_perception.txt" 2>&1 || { echo "知覚層の採点に失敗"; tail -n 5 "$CU/_log_perception.txt"; exit 6; }
cp "$PRED" "$CU/pred_v2final.csv"
# ---- 来歴（手元の compare が「同じ ckpt・同じ plan」を照合する）
"$PY" - "$OUT" "$TAG" "$N" "$V2FINAL_CKPT" "${DIST_TABLE:-}" "$RUNS/$OUT2" "${SLURM_JOB_ID:-}" <<'EOF'
import hashlib, json, sys, datetime, subprocess
from pathlib import Path
out, tag, n, ckpt, dt, run, job = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6]), sys.argv[7]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
import torch, numpy
M = {"tag": tag, "n": n, "finished_at": datetime.datetime.now().isoformat(timespec="seconds"), "job": job,
     "ckpt": str(ckpt), "ckpt_sha256": sha(ckpt), "dist_table": dt or None, "dist_table_sha256": sha(dt) if dt else None,
     "plan_sha256": sha(out / "plan/assignment_v2final.csv"), "overrides_sha256": sha(out / "SENS_OVERRIDES.json"), "gen_versions_sha256": sha(out / "GEN_VERSIONS.txt"),
     "overrides": json.loads((out / "SENS_OVERRIDES.json").read_text(encoding="utf-8")).get("overrides", {}),
     "pred_sha256": sha(run / "conf_all.csv"), "runs_dir": str(run), "torch": torch.__version__, "numpy": numpy.__version__,
     "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
     "infer": {"HEIGHT_COND": 1, "SED_TH": 0.7, "SED_STAY": 0.3, "SED_ASSOC": 20, "DSDIR": "datasets_v27"}}
(out / "SENS_META.json").write_text(json.dumps(M, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("SENS_META", tag, n, M["ckpt_sha256"][:12])
EOF
sha256sum "$V2FINAL_CKPT" "$CP" "$OUT/SENS_OVERRIDES.json" "$OUT/GEN_VERSIONS.txt" "$OUT/SENS_META.json" "$PRED" "$CU"/*.md "$CU"/*.json ${DIST_TABLE:-} \
  scripts/_sens_render.py scripts/_sens_overrides.py scripts/_run_v2final_gen.py server_sde/_causal_infer_v17.py server_sde/_prediction_integrity.py server_sde/height_patch.py \
  server_sde/causal_patch.py server_sde/hysteresis_patch.py server_sde/_v2final_members.sh scripts/_score_unified.py scripts/_perception_eval.py | tee "sens_${TAG}_sha256_job${SLURM_JOB_ID:-dry}.txt"
git rev-parse HEAD | tee -a "sens_${TAG}_sha256_job${SLURM_JOB_ID:-dry}.txt"
guard_verify || exit 1
rm -rf "$JOBDIR"
echo "SENS_VARIANT_DONE $TAG（次: 手元で python scripts/sens_variant.py fetch --tag $TAG → compare --base <基準の tag> --var $TAG）"
