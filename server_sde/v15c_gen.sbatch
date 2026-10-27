#!/bin/bash
#SBATCH -J v15cgen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o v15cgen-%j.out
# v13（合成データ修正の束）: 全量生成 9,000本（12シャード） → 距離GT → datasets_v15c ルート → 前処理。
# 設計= md/design/v13設計書_合成データ修正の束_2026-09-02.md
#
# 書き込み先は out/dataset_outdoor_siren_v15/ ・ $PS/datasets_v15c ・ $PS/_hdf5_v15c のみ。
# 基準（v12データ・datasets_v12・_hdf5・w3run）は guard で無変更を保証。
# ルート内のリンク名を outdoor_siren_v12 にするのは ablation arm と同じ流儀
# （hydra の dataset=outdoor_siren_v12 設定を一切変えず paths.dataset_dir だけで差し替えるため）。
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record


# v15c: 音は v15 と同一（再描画しない）。foa/metadata/work をリンクし、ラベルだけ 3D 距離で作る
[ -d out/dataset_outdoor_siren_v15/foa ] || { echo "v15 が無い"; exit 1; }
mkdir -p out/dataset_outdoor_siren_v15c/plan
for sub in foa metadata work; do [ -e out/dataset_outdoor_siren_v15c/$sub ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v15/$sub out/dataset_outdoor_siren_v15c/$sub; done
cp -n out/dataset_outdoor_siren_v15/plan/assignment_v15.csv out/dataset_outdoor_siren_v15c/plan/assignment_v15.csv
$PY scripts_archive/step19_dist_labels_v15c.py | tail -2
ls out/dataset_outdoor_siren_v15c/metadata_dist | wc -l

# datasets_v15c ルート（クラス辞書は基準と同一内容をコピー。リンク名は outdoor_siren_v12）
ROOTDIR=$PS/datasets_v15c
HDF5DIR=$PS/_hdf5_v15c
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v15c $ROOTDIR/outdoor_siren_v12

# 空メタ許容ラベル抽出 ＋ adpit_dist（8クラス）。キャッシュは _hdf5_v15c に分離
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true

guard_verify || exit 1
echo V15C_GEN_DONE
