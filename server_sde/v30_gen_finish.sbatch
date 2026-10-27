#!/bin/bash
#SBATCH -J v30fin
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 06:00:00
#SBATCH -o v30fin-%j.out
# 本番の描き直し v30（job 9388）の続き（2026-10-27 Opus）。9388 は 62,102 本を描き終え、距離ラベル・D14・検査 A〜I まで通ったあと、
#   val の一致の検査（_v30_val_match_check.py）で止まった（終了コード 8）。違いは val 12,266 本の scene.json の stats.gen_seconds（描くのにかかった秒数）だけで、
#   foa・metadata・metadata_dist・masks は全部同じだった → 検査を直した（描いた秒数を外して比べる・試験 V6・V7）。この台本は止まった所から先だけを回す:
#   版の照合 → 本数・D14 の検査・検査 A〜I（読むだけ）→ val の一致（直した検査）→ 9 クラスの学習の置き場 → 前処理（特徴・距離ラベル）→ 完了の検査 → 9 クラスの確かめ。
#   描き直し・距離ラベル・D14 の書き換えはしない（9388 の出力をそのまま使う）。出力先 = $PS/datasets_v30・$PS/_hdf5_v30（9388 は作る前に止まった）。
#   版の照合は直した検査と入口の門を配備したあとの台帳（server_sde/v30_expected_sha256.txt・9388 のときの台帳は v30_expected_sha256_job9388.txt に残す）。
# 投入: sbatch server_sde/v30_gen_finish.sbatch（9390 が始まってから＝9390 の版の照合のあとに台帳を作り直す）
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30 V2FINAL_RENDER=v30
OUT=out/$V2FINAL_DS_NAME
SRC=out/dataset_outdoor_siren_v2final/plan
BG_VAL=out/dataset_outdoor_siren_v2final_v30bg_val
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
for f in $(tr -d '\r' < server_sde/v30_render_files.txt) $SRC/assignment_v2final.csv $SRC/manifest_val_syn_v2final.csv; do [ -f "$f" ] || { echo "無い: $f（配備してから）"; exit 3; }; done
sha256sum --quiet -c server_sde/v30_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v30_expected_sha256.txt) 本）"
grep -q "SCENE_IGNORE" server_sde/_v30_val_match_check.py || { echo "val の一致の検査が直した版でない。中止"; exit 3; }
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['N_ROWS_TOTAL_EXPECTED_V29'])")
set -o pipefail
for d in foa metadata metadata_dist masks; do m=$(ls $OUT/$d | wc -l); [ "$m" = "$N" ] || { echo "$d が $m 本（$N）。中止"; exit 7; }; done
ls $OUT | grep -c "^GEN_VERSIONS_" && { echo "版の記録が途中で変わった（GEN_VERSIONS_<日時>.txt がある）。中止"; exit 7; }
head -5 $OUT/GEN_VERSIONS.txt
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
$PY scripts/_v26_regen_check.py --ds $OUT --no-audio --n-listen 0 || { echo "描き直しの検査（A〜I）に NG。中止"; exit 7; }
$PY server_sde/_v30_val_match_check.py --a $OUT --b $BG_VAL --manifest $SRC/manifest_val_syn_v2final.csv --out v30_val_match_job${SLURM_JOB_ID}.json \
  || { echo "val の中身が比べの val（_v30bg_val）と違う、か版の記録に許さない違い。中止（v30_val_match_job${SLURM_JOB_ID}.json）"; exit 8; }
# ここから下は v30_gen.sbatch の val の一致のあとと同じ（9 クラスの学習の置き場・前処理・完了の検査・9 クラスの確かめ）
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
