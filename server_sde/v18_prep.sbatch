#!/bin/bash
#SBATCH -J v18prep
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 02:00:00
#SBATCH -o v18prep-%j.out
# v18 = v16 の音声そのまま・ラベルだけ変更（列車の通過音 → 背景・D14 案 A・2026-09-08）。生成は無し。
# out/dataset_outdoor_siren_v18（foa/work/masks はリンク、metadata/metadata_dist は列車クリップの crossing 行を落として複写）→ datasets_v18 → 前処理。
# 書き込み先は out/dataset_outdoor_siren_v18 ・ $PS/datasets_v18 ・ $PS/_hdf5_v18 のみ。v16 と基準は不変（guard）。
set -exo pipefail    # 監査 F4: `python | tail` の失敗を隠さない（2026-09-08・ジョブ 4035 は修正前に完了済み）
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
$PY scripts/step10_v18_relabel.py --src out/dataset_outdoor_siren_v16 --dst out/dataset_outdoor_siren_v18 | tail -6
ls out/dataset_outdoor_siren_v18/metadata_dist | wc -l
ROOTDIR=$PS/datasets_v18
HDF5DIR=$PS/_hdf5_v18
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v18 $ROOTDIR/outdoor_siren_v12
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true
guard_verify || exit 1
echo V18_PREP_DONE
