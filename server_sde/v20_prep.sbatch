#!/bin/bash
# v20 の束（土台 v18 ＋ SCENES の新場面）を組んで前処理（2026-09-09）。v20_all.sbatch から NAME と SCENES を受け取る。
# 書き込み先は out/dataset_outdoor_siren_$NAME ・ $PS/datasets_$NAME ・ $PS/_hdf5_$NAME のみ。
set -exo pipefail
NAME=${NAME:?}; SCENES=${SCENES:?}
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
[ "$(ls out/dataset_outdoor_siren_v18/metadata_dist | wc -l)" = "18000" ] || { echo "v18 のラベルが 18,000 本ない"; exit 3; }
n_sc=0; for s in $SCENES; do n_sc=$((n_sc + 1)); [ "$(ls out/dataset_outdoor_siren_v19/metadata_dist | grep -c .)" -ge 1500 ] || exit 3; done
SCENE_CSV=$(echo $SCENES | tr ' ' ',')
$PY scripts/step10_v19_assemble.py --scene $SCENE_CSV --name $NAME --dst out/dataset_outdoor_siren_$NAME | tail -5
DST=out/dataset_outdoor_siren_$NAME
EXPECT=$((18000 + 1500 * n_sc))
[ "$(ls $DST/metadata_dist | wc -l)" = "$EXPECT" ] || { echo "束の本数が $EXPECT でない"; exit 1; }
ROOTDIR=$PS/datasets_$NAME
HDF5DIR=$PS/_hdf5_$NAME
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
echo V20_PREP_DONE_$NAME
