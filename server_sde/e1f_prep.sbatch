#!/bin/bash
#SBATCH -J e1fprep
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -o e1fprep-%j.out
# E1-full: v12heavy データセット組立（変更flacはアップロード済み前提）+ index + ラベルsymlink
set -ex
cd ~/research/PSELDNet/PSELDNets
SRC=datasets/outdoor_siren_v11
DST=datasets/outdoor_siren_v12heavy
test -d $DST/foa   # アップロード済み確認（変更fold1クリップのみ入っている）

# foa以外を本体へsymlink
for item in $SRC/*; do
  base=$(basename "$item")
  [ "$base" = "foa" ] && continue
  [ -e "$DST/$base" ] || ln -s "$(readlink -f $item)" "$DST/$base"
done
# 未変更のfoa（fold1の非大型クリップ+fold2全部+fold3全部）を個別symlink
for f in $SRC/foa/*.flac; do
  base=$(basename "$f")
  [ -e "$DST/foa/$base" ] || ln -s "$(readlink -f $f)" "$DST/foa/$base"
done
ls $DST/foa | wc -l          # 7,200になるはず

# index抽出のみ（E0で確立した型）
.venv/bin/python _e0_preproc_dataonly.py \
  dataset=outdoor_siren_v12heavy dataset_type=dev \
  wav_format=.flac mode=extract_data num_workers=8

# ラベルh5はv11本体をsymlink（GT不変）
for t in _hdf5/label/*/dev; do
  src="$t/outdoor_siren_v11.h5"
  [ -f "$src" ] && ln -sfn "$(basename $src)" "$t/outdoor_siren_v12heavy.h5" || true
done
echo E1F_PREP_DONE
