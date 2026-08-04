#!/bin/bash
#SBATCH -J e1aprep
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -o e1aprep-%j.out
# E1a: 派生データセット組立（foaはローカルレンダ済みをscp済み前提）+ index抽出 + ラベルsymlink
set -ex
cd ~/research/PSELDNet/PSELDNets
SRC=datasets/outdoor_siren_v11
DST=datasets/outdoor_siren_v11_e1aheavy

# foa 以外を本体へsymlink（E0と同じ構成。foaはアップロード済みディレクトリ）
test -d $DST/foa
for item in $SRC/*; do
  base=$(basename "$item")
  [ "$base" = "foa" ] && continue
  [ -e "$DST/$base" ] || ln -s "$(readlink -f $item)" "$DST/$base"
done

# index抽出のみ（ラベル抽出なし。E0の_e0_preproc_dataonly.pyを流用）
.venv/bin/python _e0_preproc_dataonly.py \
  dataset=outdoor_siren_v11_e1aheavy dataset_type=dev \
  wav_format=.flac mode=extract_data num_workers=8

# ラベルh5はv11本体をsymlink（デルタ加算でGT不変のため）
for t in _hdf5/label/*/dev; do
  src="$t/outdoor_siren_v11.h5"
  [ -f "$src" ] && ln -sfn "$(basename $src)" "$t/outdoor_siren_v11_e1aheavy.h5" || true
done
ls -la _hdf5/label/*/dev/ | grep e1aheavy || true
echo E1A_PREP_DONE
