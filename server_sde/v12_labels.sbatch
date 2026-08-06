#!/bin/bash
#SBATCH -J v12lab
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -c 16
#SBATCH -o v12lab-%j.out
# v12ラベルチェーン: datasets_v12ルート組立 → 空メタ許容ラベル抽出 → 距離GT → adpit_dist
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python

# ① datasets_v12 独立ルート（8クラス辞書=v11系と分離、地雷対策）
mkdir -p $PS/datasets_v12
cat > $PS/datasets_v12/cls_indices_train.tsv <<'EOF'
0	/m/03kmc9	Siren	0	0.0
1	/m/x_horn	Horn	0	0.0
2	/m/x_backup_beep	BackupBeep	0	0.0
3	/m/x_bike_bell	BikeBell	0	0.0
4	/m/x_car_drive	CarDrive	0	0.0
5	/m/x_crossing	Crossing	0	0.0
6	/m/x_kickboard	Kickboard	0	0.0
7	/m/x_motorcycle	Motorcycle	0	0.0
EOF
[ -e $PS/datasets_v12/outdoor_siren_v12 ] || \
  ln -s $E2E/out/dataset_outdoor_siren_v12 $PS/datasets_v12/outdoor_siren_v12
# 基準線推論用（6クラス辞書の旧ルートからも同じデータを見えるように）
[ -e $PS/datasets/outdoor_siren_v12 ] || \
  ln -s $E2E/out/dataset_outdoor_siren_v12 $PS/datasets/outdoor_siren_v12

# ② 距離GT（step19移植版。az/el全行一致の検証ゲート込み）
cd $E2E
PYTHONPATH=scripts:src $PY scripts/step19_dist_labels_v12.py

# ③ 空メタ許容ラベル抽出（index + accdoa + track + adpit、8クラス）
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$PS/datasets_v12

# ④ adpit_dist（SDE距離ラベル、8クラス）
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$PS/datasets_v12

ls -la $PS/_hdf5/label/*/dev/ | grep v12 || true
echo V12_LABELS_DONE
