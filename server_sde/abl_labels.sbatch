#!/bin/bash
#SBATCH -J abllab
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -c 16
#SBATCH -o abllab-%j.out
# ablation arm のラベルチェーン。v12_labels.sbatch のarm版。
#
# 使い方: sbatch --export=ALL,ABLATE=no_1r server_sde/abl_labels.sbatch
#
# 【設計】armごとに **独立した datasets ルート** を作り、その中のシンボリックリンク名だけ
# 基準と同じ outdoor_siren_v12 にする。こうすると hydra の dataset=outdoor_siren_v12 を
# 一切変更せず、paths.dataset_dir の差し替えだけでarmを学習・推論できる。
# 基準の datasets_v12/ には触らない（前処理h5が基準ルートへ書かれる事故を防ぐ）。
set -eux

: "${ABLATE:?ABLATE を指定してください}"
case "$ABLATE" in
  no_1r|no_doppler|no_ground|no_airabs) ;;
  *) echo "未知のarm: $ABLATE"; exit 1 ;;
esac

PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
ARMDS=$E2E/out/dataset_outdoor_siren_v12_abl_${ABLATE}
ROOTDIR=$PS/datasets_v12_abl_${ABLATE}
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
[ -d "$ARMDS/foa" ] || { echo "armのfoaが無い: $ARMDS"; exit 1; }
echo "arm=$ABLATE データ=$ARMDS ルート=$ROOTDIR （基準 $PS/datasets_v12 には触れない）"

# ① arm専用ルート（クラス辞書は基準と同一内容をコピー）
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s $ARMDS $ROOTDIR/outdoor_siren_v12

# ② 距離GT（arm用ドライバ。基準のmetadata_distは触らない）
cd $E2E
PYTHONPATH=scripts:src ABLATE=$ABLATE $PY scripts/_run_v12_abl_labels.py

# ③ 空メタ許容ラベル抽出（index + accdoa + track + adpit、8クラス）
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR

# ④ adpit_dist（SDE距離ラベル、8クラス）
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR

echo ABL_LABELS_DONE_$ABLATE
