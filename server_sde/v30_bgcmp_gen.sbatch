#!/bin/bash
#SBATCH -J v30bgg
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o v30bgg-%j.out
# 背景の形の小さな比べ（決め直し 5・本人 10/26 20:03・仕様 §9-7）の描画。V30BG_VARIANT で 1 つずつ:
#   val = 比べ用の置き場 out/dataset_outdoor_siren_v2final_v30bg_val に val 12,420 本を (ii)（背景の形あり・傾き −6.5〜−2.5）で描く
#         （Fable 20:3x: 本番の置き場 _v30 は勝ち筋が決まってから train・val・補正表の組・eval v4 を全部その幅で描く＝来歴を 1 つに）
#   off／i／ii = out/dataset_outdoor_siren_v2final_v30bg_<x> に train の部分集合 12,000 本（md/design/v30_bgcmp_本の一覧_2026-10-26.csv）を描き、
#                val（比べ用の置き場 _v30bg_val・(ii)）をファイルごとのリンクで足して、$PS/datasets_v30bg_<x>・$PS/_hdf5_v30bg_<x> を作る（学習は v30_bgcmp_train.sbatch）
#     off = 背景の形なし（FEATURES["bg_shape"] 切）／i = 形あり・幅は既定（折れ点 63〜1,000・傾き −5〜−3）／ii = 形あり・傾き −6.5〜−2.5
#   ほかの v30 の直し（帯域ゲート 0 dB・歩く向き・クラクションの Δ）は全部入（step11_v30_render の既定）。描画は _run_v2final_gen.py の V2FINAL_RENDER=v30・V2FINAL_ROWS_LIST。
# 順（Fable の OK のあと）: val → off・i・ii（val が全部そろってから。部分集合の 3 本は同時でよい）
#   sbatch --export=ALL,V30BG_VARIANT=val server_sde/v30_bgcmp_gen.sbatch
#   for x in off i ii; do sbatch --export=ALL,V30BG_VARIANT=$x server_sde/v30_bgcmp_gen.sbatch; done
#   区画は空いている所（pro_6000 が満杯なら -p a100_3g --gres=gpu:3g.40gb:1・low 可・requeue で続きから）。乾式: V30BG_DRY=1（命令を印字するだけ）
# 版の照合: server_sde/v30bgcmp_expected_sha256.txt（配備のときサーバ側で作る: sha256sum <下の FILES> > server_sde/v30bgcmp_expected_sha256.txt）
set -x
cd ~/research/outdoor_seld_e2e
E2E=$PWD
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
VAR=${V30BG_VARIANT:?V30BG_VARIANT（val／off／i／ii）を渡す}
RUN() { if [ "${V30BG_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
SRC_PLAN_DIR=out/dataset_outdoor_siren_v2final/plan
VAL_DS=out/dataset_outdoor_siren_v2final_v30bg_val
LIST=md/design/v30_bgcmp_本の一覧_2026-10-26.csv
unset V30_BG_SHAPE V30_BG_SLOPE_RANGE V30_BG_KNEE_RANGE V30_GATE_BAND_DB
case "$VAR" in
  val) export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30bg_val V30_BG_SHAPE=1 V30_BG_SLOPE_RANGE=-6.5,-2.5; ROWS=$VAL_DS/plan/manifest_val_syn_v2final.csv; NEXP=12420 ;;
  off) export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30bg_off V30_BG_SHAPE=0; ROWS=$LIST; NEXP=12000 ;;   # 10/27: v30 の既定が (ii) になったので切を明示
  i)   export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30bg_i V30_BG_SHAPE=1 V30_BG_SLOPE_RANGE=-5,-3; ROWS=$LIST; NEXP=12000 ;;   # 10/27: 幅を明示（既定は (ii) に変わった）
  ii)  export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30bg_ii V30_BG_SHAPE=1 V30_BG_SLOPE_RANGE=-6.5,-2.5; ROWS=$LIST; NEXP=12000 ;;
  *) echo "V30BG_VARIANT は val／off／i／ii（$VAR）"; exit 3 ;;
esac
export V2FINAL_RENDER=v30 V2FINAL_ROWS_LIST=$ROWS
OUT=out/$V2FINAL_DS_NAME
echo "VARIANT=$VAR OUT=$OUT ROWS=$ROWS N=$NEXP BG_SHAPE=${V30_BG_SHAPE:-0} SLOPE=${V30_BG_SLOPE_RANGE:-既定} job=${SLURM_JOB_ID:-dry}"
source $E2E/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF' || { echo "params が fixed でない。中止"; exit 3; }
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
sys.exit(0 if P.get("status") == "fixed" else 3)
EOF
FILES="scripts/_run_v2final_gen.py scripts/step11_v30_render.py scripts/step11_v29_render.py scripts/step11_v13_render.py scripts/step11_v9_render.py scripts/step11_v12_render.py \
  src/outdoor_seld/band_gate.py src/outdoor_seld/noise_v30.py src/outdoor_seld/calibration.py src/outdoor_seld/geometry.py src/outdoor_seld/labels.py \
  scripts/step19_dist_labels_v2final.py scripts/step19_dist_labels_v16.py scripts/step10_v2final_relabel.py $LIST md/design/v2final_params.json server_sde/_train_entrance_guard.py"
for f in $FILES; do [ -f "$f" ] || { echo "無い: $f（配備してから）"; exit 3; }; done
if [ "${V30BG_DRY:-0}" != 1 ]; then
  sha256sum --quiet -c server_sde/v30bgcmp_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
fi
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && REUSE=1
PFX=$( [ "$VAR" = val ] && echo val_syn || echo train_syn )
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | grep -c "^$PFX")" -gt 0 ] && [ "${REUSE:-0}" != 1 ]; then
  echo "出力先 $OUT に既に $PFX の本がある（続きから回すなら REUSE=1）"; exit 1
fi
RUN mkdir -p $OUT/plan
for f in assignment_v2final.csv manifest_val_syn_v2final.csv manifest_val_select.csv manifest_val_check.csv; do
  if [ -f "$OUT/plan/$f" ]; then
    cmp -s "$OUT/plan/$f" "$SRC_PLAN_DIR/$f" || { echo "$OUT/plan/$f が元の plan と違う → 止める"; exit 3; }
  else RUN cp "$SRC_PLAN_DIR/$f" "$OUT/plan/$f"; fi
done
# params の写しをシャードを起こす前に 1 回だけ作る（job 9342 の教訓・2026-10-26 21:5x: 48 シャードが同時に _run_v2final_gen.write_gen_versions の
#   snap.write_bytes を通り、書きかけを読んだ 2 シャード〔19・43〕が「写しの中身が今の params と違う」で止まった。描画のコードは変えずに〔版の sha を
#   val と off／i／ii でそろえる〕、ここで一時ファイル → 置き換えで先に作る。名前と中身は write_gen_versions と同じ〔params_<sha の頭 16>.json〕）
if [ "${V30BG_DRY:-0}" != 1 ]; then
  $PY - "$OUT" <<'EOF' || { echo "params の写しを作れない。中止"; exit 3; }
import hashlib, os, sys
from pathlib import Path
pb = Path("md/design/v2final_params.json").read_bytes(); sha = hashlib.sha256(pb).hexdigest()
d = Path(sys.argv[1]) / "params_snapshots"; d.mkdir(parents=True, exist_ok=True)
p = d / f"params_{sha[:16]}.json"
if not p.is_file():
    t = d / f".params_{os.getpid()}.tmp"; t.write_bytes(pb); os.replace(t, p)
assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, f"{p} の中身が今の params と違う"
print("params の写し", p)
EOF
fi
# 描く（行を 1 つおきに配る --shard i/n・BLAS のスレッドは 2 本）。続きから回すとき（REUSE=1）は V30BG_SHARDS を 48 と割り切れない数（例 47）にすると、
#   止まったシャードの行が全シャードに散る。ログはジョブ番号つき（前のジョブのログを上書きしない）
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=${V30BG_SHARDS:-48}
pids=()
for i in $(seq 0 $((SH - 1))); do
  if [ "${V30BG_DRY:-0}" = 1 ]; then echo "[dry] $PY scripts/_run_v2final_gen.py --shard $i/$SH"; continue; fi
  $PY scripts/_run_v2final_gen.py --shard $i/$SH > v30bg_${VAR}_shard_${i}_job${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
[ "${V30BG_DRY:-0}" = 1 ] && { echo V30BG_GEN_DRY_DONE; exit 0; }
n=$(ls $OUT/foa | grep -c "^$PFX"); [ "$n" = "$NEXP" ] || { echo "描けた本 $n ≠ $NEXP。中止"; exit 7; }
cat $OUT/GEN_VERSIONS.txt | head -5
set -o pipefail
# 距離ラベル → D14（この置き場の本だけ・val のリンクを張る前）
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルに失敗。中止"; exit 5; }
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
for d in foa metadata metadata_dist masks; do m=$(ls $OUT/$d | grep -c "^$PFX"); [ "$m" = "$NEXP" ] || { echo "$d が $m 本（$NEXP）。中止"; exit 7; }; done
if [ "$VAR" = val ]; then
  echo "V30BG_VAL_DONE $OUT（val $NEXP 本・(ii)）"; guard_verify || exit 1; exit 0
fi
# val（比べ用の置き場 _v30bg_val・(ii)）をファイルごとのリンクで足す（全部そろっていなければ止める）
for d in foa metadata metadata_dist masks; do
  v=$(ls $VAL_DS/$d 2>/dev/null | grep -c "^val_syn"); [ "$v" = 12420 ] || { echo "val の $d が $v 本（12,420 ではない）→ 先に V30BG_VARIANT=val"; exit 3; }
done
for d in foa metadata metadata_dist masks; do
  for f in $VAL_DS/$d/val_syn_*; do ln -sfn "$E2E/$f" "$OUT/$d/$(basename $f)"; done
done
for w in $VAL_DS/work/val_syn_*; do ln -sfn "$E2E/$w" "$OUT/work/$(basename $w)"; done
ROOTDIR=$PS/datasets_v30bg_$VAR
HDF5DIR=$PS/_hdf5_v30bg_$VAR
mkdir -p $ROOTDIR $HDF5DIR
if [ ! -f $ROOTDIR/cls_indices_train.tsv ]; then
  cp $PS/datasets_v29/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
fi
[ "$(wc -l < $ROOTDIR/cls_indices_train.tsv)" = 9 ] || { echo "クラス一覧が 9 行でない"; exit 1; }
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s $E2E/$OUT $ROOTDIR/outdoor_siren_v12
$PY server_sde/_train_entrance_guard.py $ROOTDIR $HDF5DIR --allow-empty || { echo "学習の入口の門で止めた"; exit 2; }
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 1 に失敗。中止"; exit 6; }
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 2 に失敗。中止"; exit 6; }
cd $E2E
$PY - "$HDF5DIR" <<'EOF' || { echo "ラベル h5 のクラス数が 9 でない。中止"; exit 7; }
import glob, sys, h5py
for p in glob.glob(f"{sys.argv[1]}/label/adpit_dist/dev/*.h5"):
    with h5py.File(p) as hf:
        k = next(iter(hf)); shp = hf[f"{k}/adpit/se"].shape
        print(p, k, shp); assert shp[-1] == 9, shp
EOF
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt $LIST | tee v30bg_${VAR}_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo "V30BG_GEN_DONE $VAR $OUT train $NEXP + val 12420（リンク）"
