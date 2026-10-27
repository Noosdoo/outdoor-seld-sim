#!/bin/bash
# D21 なし（24,000 本）の学習用 HDF5 ルートを作る（2026-10-01・対照実験 (b)(c) 用）。
# PSELDNets は学習クリップを <hdf5_dir>/data/24000fs/wav/dev/<ds>_10sChunklen_10sHoplen_train.csv から読む（src/data/components/data.py:69）。
# そこから kickfar（val_syn/train_syn の clip 30001〜31750）の行を外した csv を置き、ラベル h5（clip 名で引く）は本物へのシンボリックリンクにする。
# _hdf5_v2final 自体には書き込まない。何度走らせても同じ結果（冪等）。
set -e
PS=$HOME/research/PSELDNet/PSELDNets
SRC=$PS/_hdf5_v2final
DST=$PS/_hdf5_v2final_noD21
D=data/24000fs/wav/dev
mkdir -p $DST/$D
[ -e $DST/label ] || ln -s $SRC/label $DST/label
grep -vE "clip3[01][0-9]{3}\.flac" $SRC/$D/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv > $DST/$D/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv
cp -f $SRC/$D/outdoor_siren_v12_10sChunklen_10sHoplen_test.csv $DST/$D/outdoor_siren_v12_10sChunklen_10sHoplen_test.csv
n_tr=$(wc -l < $DST/$D/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv)
n_te=$(wc -l < $DST/$D/outdoor_siren_v12_10sChunklen_10sHoplen_test.csv)
n_kf=$(grep -cE "clip3[01][0-9]{3}\.flac" $DST/$D/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv || true)
echo "noD21 hdf5 root: $DST  train.csv=$n_tr 行（kickfar $n_kf 行）  test.csv=$n_te 行  label -> $(readlink $DST/label)"
[ "$n_tr" -eq 24000 ] || { echo "train.csv の行数が 24,000 でない"; exit 1; }
[ "$n_kf" -eq 0 ] || { echo "train.csv に kickfar が残っている"; exit 1; }
[ "$n_te" -eq 25500 ] || { echo "test.csv の行数が 25,500 でない"; exit 1; }
echo NOD21_HDF5_OK
