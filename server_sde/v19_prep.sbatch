#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 02:00:00
#SBATCH -o v19prep-%j.out
# v19 の学習用データ束（土台 v18 ＋ 新場面 1 つ）を組んで前処理（2026-09-08）。
# 投入例: sbatch -J v19fastprep --export=ALL,SCENE=fast --dependency=afterok:<gen job> server_sde/v19_prep.sbatch
# 書き込み先は out/dataset_outdoor_siren_v19_$SCENE ・ $PS/datasets_v19_$SCENE ・ $PS/_hdf5_v19_$SCENE のみ。
set -exo pipefail    # 監査 F4: `python | tail` の失敗を隠さない
SCENE=${SCENE:?SCENE=fast|static|turn|platoon}
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
[ -d out/dataset_outdoor_siren_v18/metadata_dist ] || { echo "v18 のラベルが無い（v18_prep.sbatch を先に）"; exit 3; }
[ "$(ls out/dataset_outdoor_siren_v18/metadata_dist | wc -l)" = "18000" ] || { echo "v18 のラベルが 18,000 本ない"; exit 3; }
$PY scripts/step10_v19_assemble.py --scene $SCENE | tail -5
DST=out/dataset_outdoor_siren_v19_$SCENE
[ "$(ls $DST/metadata_dist | wc -l)" = "19500" ] || { echo "束の本数が 19,500 でない"; exit 1; }
ROOTDIR=$PS/datasets_v19_$SCENE
HDF5DIR=$PS/_hdf5_v19_$SCENE
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/$DST $ROOTDIR/outdoor_siren_v12
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true
guard_verify || exit 1
echo V19_PREP_DONE_$SCENE
