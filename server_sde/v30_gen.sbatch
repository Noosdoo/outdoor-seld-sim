#!/bin/bash
#SBATCH -J v30gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 20:00:00
#SBATCH --requeue
#SBATCH -o v30gen-%j.out
# 学習し直し v30 の本番の描き直し（仕様 §4 の 1・変更票 #75・Fable 10/27 07:3x の 3）: 新しい置き場 out/dataset_outdoor_siren_v2final_v30 に plan の全部 62,102 本
#   （train 49,682 ＋ val 12,420）を v30 の描画（V2FINAL_RENDER=v30 = step11_v30_render.generate_clip・既定 = 歩く向き・クラクションの Δ・帯域ゲート 0 dB・
#   背景の形 (ii)〔10/27 の決め直し 5 の結果〕）で描く。v29 の台本（v29_gen.sbatch）の写しで、違うのは描画の入口・置き場の名前・shard 48（行を 1 つおきに配る）・
#   検査 A〜I（_v26_regen_check.py）を前処理の前に・val の中身が比べの val（_v30bg_val）と同じことの検査（_v30_val_match_check.py）・版の台帳。
#   - 既存の束（_v2final・_v26・_v27・_v29・_v30bg_*・$PS/datasets_*・_hdf5_*）には書かない。出力先 = out/dataset_outdoor_siren_v2final_v30・$PS/datasets_v30・$PS/_hdf5_v30。
#   - 版の照合: server_sde/v30_expected_sha256.txt（配備のときサーバ側で作る: sha256sum $(tr -d '\r' < server_sde/v30_render_files.txt) > server_sde/v30_expected_sha256.txt）。
#   - 乾式: V30_DRY=1（照合と入口の確かめだけして、重い命令は印字するだけ）。区画は空いている所（-p a100 --gres=gpu:a100:1 など・CPU の数で速さが決まる）。
# 投入は Fable の OK のあと: sbatch server_sde/v30_gen.sbatch
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
RUN() { if [ "${V30_DRY:-0}" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }
unset V30_BG_SHAPE V30_BG_SLOPE_RANGE V30_BG_KNEE_RANGE V30_GATE_BAND_DB V2FINAL_ROWS_LIST V2FINAL_PROVISIONAL
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30 V2FINAL_RENDER=v30
OUT=out/$V2FINAL_DS_NAME
SRC=out/dataset_outdoor_siren_v2final/plan
BG_VAL=out/dataset_outdoor_siren_v2final_v30bg_val
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY - <<'EOF' || { echo "params が fixed でない。中止"; exit 3; }
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
print("params status =", P.get("status"))
sys.exit(0 if P.get("status") == "fixed" else 3)
EOF
for f in $(tr -d '\r' < server_sde/v30_render_files.txt) $SRC/assignment_v2final.csv $SRC/manifest_val_syn_v2final.csv; do [ -f "$f" ] || { echo "無い: $f（配備してから）"; exit 3; }; done
sha256sum --quiet -c server_sde/v30_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v30_expected_sha256.txt) 本）"
$PY -c "import step11_v30_render as v;import json;print('v30 の既定', json.dumps(v.FEATURES, sort_keys=True), v.GATE_BAND_DB, v.BG_KNEE_RANGE, v.BG_SLOPE_RANGE)"
$PY -c "import step11_v30_render as v;import sys;sys.exit(0 if all(v.FEATURES.values()) and v.GATE_BAND_DB == 0 and v.BG_SLOPE_RANGE == (-6.5, -2.5) and v.BG_KNEE_RANGE == (63.0, 1000.0) else 3)" \
  || { echo "v30 の既定が (ii)・帯域ゲート 0・全部入 でない。中止"; exit 3; }
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && { echo "requeue（${SLURM_RESTART_COUNT} 回目）→ 続きから（REUSE=1）"; REUSE=1; }
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ] && [ "${REUSE:-0}" != 1 ]; then echo "出力先 $OUT に既に本がある（続きから回すなら REUSE=1）"; exit 1; fi
RUN mkdir -p $OUT/plan
for f in assignment_v2final.csv manifest_v26_bellbike.csv manifest_v27_mix.csv manifest_v28_topup.csv manifest_v29_d19.csv manifest_val_syn_v2final.csv manifest_val_select.csv manifest_val_check.csv; do
  [ -f "$SRC/$f" ] || continue
  if [ -f "$OUT/plan/$f" ]; then cmp -s "$OUT/plan/$f" "$SRC/$f" || { echo "$OUT/plan/$f が元の plan と違う → 止める"; exit 3; }; else RUN cp "$SRC/$f" "$OUT/plan/$f"; fi
done
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['N_ROWS_TOTAL_EXPECTED_V29'])")
[ "$(($(wc -l < $SRC/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
NVAL=$(($(wc -l < $SRC/manifest_val_syn_v2final.csv) - 1))
echo "N=$N（val $NVAL）OUT=$OUT job=${SLURM_JOB_ID:-dry}"
RUN $PY scripts/_run_v2final_gen.py --list
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=${V30_SHARDS:-48}
pids=()
for i in $(seq 0 $((SH - 1))); do
  if [ "${V30_DRY:-0}" = 1 ]; then [ $i -lt 2 ] && echo "[dry] $PY scripts/_run_v2final_gen.py --shard $i/$SH（…$SH 本）"; continue; fi
  $PY scripts/_run_v2final_gen.py --shard $i/$SH > v30_shard_${i}_job${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
if [ "${V30_DRY:-0}" = 1 ]; then
  for c in "$PY scripts/step19_dist_labels_v2final.py --ds $OUT" "$PY scripts/step10_v2final_relabel.py --ds $OUT（と --check）" "$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0（検査 A〜I）" \
           "$PY server_sde/_v30_val_match_check.py --a $OUT --b $BG_VAL --manifest $SRC/manifest_val_syn_v2final.csv" "前処理 _preproc_emptyok12・_preproc_sde → $PS/datasets_v30・$PS/_hdf5_v30・完了の検査・9 クラス"; do echo "[dry] $c"; done
  echo V30_GEN_DRY_DONE; exit 0
fi
n=$(ls $OUT/foa | wc -l); [ "$n" = "$N" ] || { echo "描けた本 $n ≠ $N。中止"; exit 7; }
head -5 $OUT/GEN_VERSIONS.txt
ls $OUT | grep -c "^GEN_VERSIONS_" && { echo "版の記録が途中で変わった（GEN_VERSIONS_<日時>.txt がある）。中止"; exit 7; }
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルに失敗。中止"; exit 5; }
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
for d in foa metadata metadata_dist masks; do m=$(ls $OUT/$d | wc -l); [ "$m" = "$N" ] || { echo "$d が $m 本（$N）。中止"; exit 7; }; done
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜I）に NG。中止"; exit 7; }
$PY server_sde/_v30_val_match_check.py --a $OUT --b $BG_VAL --manifest $SRC/manifest_val_syn_v2final.csv --out v30_val_match_job${SLURM_JOB_ID}.json \
  || { echo "val の中身が比べの val（_v30bg_val）と違う、か版の記録に許さない違い。中止（v30_val_match_job${SLURM_JOB_ID}.json）"; exit 8; }
# 9 クラスの学習の置き場（新しい名前）
ROOTDIR=$PS/datasets_v30
HDF5DIR=$PS/_hdf5_v30
for x in datasets_v12 datasets_v2final datasets_v22 datasets_v26 datasets_v27 datasets_v29; do [ "$ROOTDIR" != "$PS/$x" ] || { echo "既存ルートと同一"; exit 1; }; done
mkdir -p $ROOTDIR $HDF5DIR
[ -f $ROOTDIR/cls_indices_train.tsv ] || cp $PS/datasets_v29/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ "$(wc -l < $ROOTDIR/cls_indices_train.tsv)" = 9 ] || { echo "クラス一覧が 9 行でない"; exit 1; }
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/$OUT $ROOTDIR/outdoor_siren_v12
$PY server_sde/_train_entrance_guard.py $ROOTDIR $HDF5DIR --allow-empty || { echo "学習の入口の門で止めた"; exit 2; }
cd $PS
T_PRE=$(date +%s)
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 1（特徴）に失敗。中止"; exit 6; }
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 2（距離ラベル）に失敗。中止"; exit 6; }
cd ~/research/outdoor_seld_e2e
$PY server_sde/_v2final_gen_complete_check.py --n "$N" --hdf5 "$HDF5DIR" --t-pre "$T_PRE" --ds $OUT --dataset-type dev \
  || { echo "完了の検査に失敗（本数・成果物の欠け・HDF5 の更新や中身）。中止"; exit 7; }
$PY - "$HDF5DIR" <<'EOF' || { echo "ラベル h5 のクラス数が 9 でない。中止"; exit 7; }
import glob, sys, h5py
for p in glob.glob(f"{sys.argv[1]}/label/adpit_dist/dev/*.h5"):
    with h5py.File(p) as hf:
        k = next(iter(hf)); shp = hf[f"{k}/adpit/se"].shape
        print(p, k, shp); assert shp[-1] == 9, shp
EOF
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt | tee v30_gen_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo V30_GEN_DONE
