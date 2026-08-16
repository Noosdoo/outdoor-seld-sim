#!/bin/bash
#SBATCH -J ablcache
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o ablcache-%j.out
# 前処理キャッシュの分離と基準キャッシュの復元（2026-08-16、確認runで発覚した事故の後始末）。
#
# 【何が起きたか】
# 前処理(_preproc_emptyok12 / _preproc_sde)の出力先 `paths.hdf5_dir` は既定 ./_hdf5 で、
# ファイル名は**データセット名だけ**で決まる（outdoor_siren_v12.h5）。
# armのルートを分けても、シンボリックリンク名を基準と同じ outdoor_siren_v12 にしていたため、
# armの前処理が**基準のキャッシュを上書き**した。その結果、学習も推論も同じ（armの）
# キャッシュを読み、自条件valとフル物理valの予測がバイト同一になった。
#
# 【この後始末で行うこと】
#  ① arm用キャッシュを $PS/_hdf5_abl_<arm>/ に作り直す
#  ② 基準キャッシュ $PS/_hdf5/ を基準データから作り直して復元する
# ②は基準の**生データには一切書き込まない**（読むだけ）。step19（metadata_dist生成）は
# 走らせない＝基準データセットのファイルは触らない。
#
# 使い方: sbatch --export=ALL,ABLATE=no_1r server_sde/abl_cache_fix.sbatch
set -eux

: "${ABLATE:?ABLATE を指定してください}"
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
ARMROOT=$PS/datasets_v12_abl_${ABLATE}
ARMH5=$PS/_hdf5_abl_${ABLATE}
[ -d "$ARMROOT/outdoor_siren_v12" ] || { echo "arm用ルートが無い: $ARMROOT"; exit 1; }

cd $PS

# ① arm用キャッシュ（arm専用ディレクトリへ）
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ARMROOT paths.hdf5_dir=$ARMH5
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ARMROOT paths.hdf5_dir=$ARMH5

# ② 基準キャッシュの復元（基準データを読み、既定の _hdf5 へ書き戻す）
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5

echo '--- 分離できたか（別物なら成功）---'
md5sum $ARMH5/label/adpit_dist/dev/outdoor_siren_v12.h5 \
       $PS/_hdf5/label/adpit_dist/dev/outdoor_siren_v12.h5
echo ABL_CACHE_FIX_DONE_$ABLATE
