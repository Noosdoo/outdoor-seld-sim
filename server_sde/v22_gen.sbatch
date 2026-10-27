#!/bin/bash
#SBATCH -J v22gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o v22gen-%j.out
# v22（2026-10-08・変更票 #48 D22 bicycle 4,500＋500・#49 D23 ピン型ベル 774）の開発用の生成。中身は v2final_gen.sbatch と同じ流れで、違いは 4 つ:
#   1. plan = step10_v2final_plan（土台＋v19 の 25,500 行）のあとに step10_v22_plan で 5,774 行を末尾に足す（総行数 N_ROWS_TOTAL_EXPECTED_V22 = 31,274）
#   2. 描画は新しい行だけ（_run_v2final_gen は出来ている本を飛ばす）。既存 25,500 本の音・ラベルは変えない
#   3. 9 クラスの置き場 $PS/datasets_v22（cls_indices_train.tsv に 8 行目 bicycle を足す）と $PS/_hdf5_v22。8 クラスの置き場（datasets_v2final・_hdf5_v2final）は触らない
#   4. 仮の印（V2FINAL_PROVISIONAL=1）で走る開発用。本番の 11 月の生成は params を fixed にしてから v2final_gen.sbatch 系で通し直す
# 投入: sbatch --export=ALL,V2FINAL_PROVISIONAL=1 server_sde/v22_gen.sbatch
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
[ "${V2FINAL_PROVISIONAL:-0}" = 1 ] || { echo "開発用の台本。V2FINAL_PROVISIONAL=1 で投入する"; exit 3; }
for f in scripts/step10_v2final_plan.py scripts/step10_v22_plan.py scripts/step11_v22_render.py src/outdoor_seld/bicycle_v2.py src/outdoor_seld/bike_bell_v2.py \
         scripts/_run_v2final_gen.py scripts/step19_dist_labels_v2final.py scripts/step10_v2final_relabel.py server_sde/_v2final_gen_complete_check.py; do
  [ -f "$f" ] || { echo "無い: $f（手元から配備してから投入する）"; exit 3; }
done
sha256sum --quiet -c server_sde/v2final_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v2final_expected_sha256.txt) 本）"
# plan: 土台＋v19（25,500）を作って検査 → v22 の 5,774 行を足して検査
$PY scripts/step10_v2final_plan.py --check-v16 --check > v22_plan_check_job${SLURM_JOB_ID}.log 2>&1 || { tail -20 v22_plan_check_job${SLURM_JOB_ID}.log; echo "土台の plan の検査に失敗。中止"; exit 3; }
cp out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv out/dataset_outdoor_siren_v2final/plan/assignment_v2final_pre_v22.csv
$PY scripts/step10_v22_plan.py >> v22_plan_check_job${SLURM_JOB_ID}.log 2>&1 || { tail -20 v22_plan_check_job${SLURM_JOB_ID}.log; echo "v22 の plan の検査に失敗。中止"; exit 3; }
tail -6 v22_plan_check_job${SLURM_JOB_ID}.log
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['N_ROWS_TOTAL_EXPECTED_V22'])"); SH=30
[ "$(($(wc -l < out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N_ROWS_TOTAL_EXPECTED_V22 と違う"; exit 3; }
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v2final_gen.py --rows $lo-$hi > v22_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v2final/foa | wc -l
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py | tail -2 || { echo "距離ラベルの生成に失敗。中止"; exit 5; }
ls out/dataset_outdoor_siren_v2final/metadata_dist | wc -l
$PY scripts/step10_v2final_relabel.py --ds out/dataset_outdoor_siren_v2final || { echo "D14 のラベル変換に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds out/dataset_outdoor_siren_v2final --check || { echo "D14 の検査に失敗。中止"; exit 4; }
# 9 クラスの置き場
ROOTDIR=$PS/datasets_v22
HDF5DIR=$PS/_hdf5_v22
[ "$ROOTDIR" != "$PS/datasets_v12" ] && [ "$ROOTDIR" != "$PS/datasets_v2final" ] || { echo "既存ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
if [ ! -f $ROOTDIR/cls_indices_train.tsv ]; then
  cp $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
  printf '8\t/m/x_bicycle\tBicycle\t0\t0.0\n' >> $ROOTDIR/cls_indices_train.tsv
fi
[ "$(wc -l < $ROOTDIR/cls_indices_train.tsv)" = 9 ] || { echo "クラス一覧が 9 行でない"; exit 1; }
cat $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v2final $ROOTDIR/outdoor_siren_v12
cd $PS
T_PRE=$(date +%s)
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 1（特徴）に失敗。中止"; exit 6; }
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 2（距離ラベル）に失敗。中止"; exit 6; }
ls -la $HDF5DIR/label/*/dev/ || true
cd ~/research/outdoor_seld_e2e
$PY server_sde/_v2final_gen_complete_check.py --n "$N" --hdf5 "$HDF5DIR" --t-pre "$T_PRE" --ds out/dataset_outdoor_siren_v2final --dataset-type dev \
  || { echo "完了の検査に失敗（本数・成果物の欠け・HDF5 の更新や中身）。中止"; exit 7; }
# 9 クラスのラベルになっているか（h5 の最後の軸が 9）
$PY - "$HDF5DIR" <<'EOF' || { echo "ラベル h5 のクラス数が 9 でない。中止"; exit 7; }
import glob, sys, h5py
root = sys.argv[1]
for p in glob.glob(f"{root}/label/adpit_dist/dev/*.h5"):
    with h5py.File(p) as hf:
        k = next(iter(hf)); shp = hf[f"{k}/adpit/se"].shape
        print(p, k, shp); assert shp[-1] == 9, shp
EOF
echo "PROVISIONAL v22 $(date)" > out/dataset_outdoor_siren_v2final/PROVISIONAL.txt
guard_verify || exit 1
echo V22_GEN_DONE
