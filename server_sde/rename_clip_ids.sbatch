#!/bin/bash
#SBATCH -J renameids
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 04:00:00
#SBATCH -o renameids-%j.out
# 識別子の改名（2026-09-22・本人決定・手順書 = md/design/識別子の改名_手順書_2026-09-22.md §5）。1 回だけ実行する。
#   fold1_room1_mix0001 → train_syn_clip00001 / fold2 → val_syn / fold20 → eval_syn（5 桁）。内容は 1 バイトも変えない。
#   対象 = dataset_outdoor_siren_v12（fold3 は改名せず dataset_outdoor_siren_v11_fold3_archive へ）・v12_conf・v2final・out/_eval_aux_v12。
#   v12_eval（fold10〜13）と旧い試作・退避フォルダは触らない。HDF5 は _hdf5_v2final だけ作り直す（_hdf5（v12 基準の前処理キャッシュ）は
#   v2 の台本が使わないので触らない。索引の中の名前は旧名のまま残る＝履歴）。
#   PSELDNets の data 設定（train/valid/test の接頭辞）と evaluate.py は投入前に配備済みであることを sha で確かめる。
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
[ "$(squeue -u $USER -h | wc -l)" = 1 ] || { echo "他に自分のジョブが走っている。中止"; exit 3; }
# 配備の照合（手元のコミット版と同じか）
sha256sum --quiet -c server_sde/v2final_expected_sha256.txt || { echo "スクリプトの版が手元と違う。配備してから投入する。中止"; exit 3; }
sha256sum --quiet -c server_sde/rename_expected_sha256.txt || { echo "改名に使うファイル（migrate_clip_ids.py・clip_ids.py・PSELDNets の設定と evaluate.py）が手元と違う。中止"; exit 3; }
echo "版の照合 OK"
# data 設定が新名になっているか
grep -q "\[train_syn\]" $PS/configs/data/outdoor_siren_v12.yaml || { echo "PSELDNets の data 設定が旧名のまま。中止"; exit 3; }

# 1) 改名（sha 台帳つき）。既に新名なら 0 件で通る
$PY scripts/migrate_clip_ids.py out/dataset_outdoor_siren_v12 --archive-unmapped out/dataset_outdoor_siren_v11_fold3_archive || exit 4
$PY scripts/migrate_clip_ids.py out/dataset_outdoor_siren_v12_conf || exit 4
$PY scripts/migrate_clip_ids.py out/dataset_outdoor_siren_v2final || exit 4
[ -d out/_eval_aux_v12 ] && { $PY scripts/migrate_clip_ids.py out/_eval_aux_v12 || exit 4; }
# 2) 照合（改名後の全ファイルの sha256 が改名前と一致）
for d in out/dataset_outdoor_siren_v12 out/dataset_outdoor_siren_v12_conf out/dataset_outdoor_siren_v2final out/_eval_aux_v12; do
  [ -d "$d" ] || continue
  $PY scripts/migrate_clip_ids.py "$d" --verify || { echo "照合に失敗: $d"; exit 5; }
done
ls out/dataset_outdoor_siren_v12/foa | head -2; ls out/dataset_outdoor_siren_v12_conf/foa | head -2; ls out/dataset_outdoor_siren_v2final/foa | head -2
ls out/dataset_outdoor_siren_v11_fold3_archive/foa | wc -l
# 3) リンクの確認（置き場の名前は変えていないので不変のはず）
ls -la $PS/datasets_v12/outdoor_siren_v12 $PS/datasets_v2final/outdoor_siren_v12
# 4) HDF5 の作り直し（v2final だけ。旧版は _pre_rename に退避）
ROOTDIR=$PS/datasets_v2final
HDF5DIR=$PS/_hdf5_v2final
[ -d "$HDF5DIR" ] && mv "$HDF5DIR" "${HDF5DIR}_pre_rename"
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true
head -2 $HDF5DIR/data/24000fs/wav/dev/*train.csv | cut -c1-140
grep -c "train_syn_clip" $HDF5DIR/data/24000fs/wav/dev/*train.csv
grep -c "fold" $HDF5DIR/data/24000fs/wav/dev/*.csv || true
cd ~/research/outdoor_seld_e2e
echo V2_RENAME_DONE
