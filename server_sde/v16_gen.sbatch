#!/bin/bash
#SBATCH -J v16gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o v16gen-%j.out
# v16（高さ増強: v15 の各場面を 2 つの高さで描く）: 全量生成 18,000本（24シャード） → 距離GT（水平） → datasets_v16 ルート → 前処理。
# 設計= md/design/v16候補学習_高さ増強_事前宣言_2026-09-05.md
#
# 書き込み先は out/dataset_outdoor_siren_v16/ ・ $PS/datasets_v16 ・ $PS/_hdf5_v16 のみ。
# 基準（v12データ・datasets_v12・_hdf5・w3run）は guard で無変更を保証。v15 以前のデータにも書かない。
# ルート内のリンク名を outdoor_siren_v12 にするのは ablation arm と同じ流儀
# （hydra の dataset=outdoor_siren_v12 設定を一切変えず paths.dataset_dir だけで差し替えるため）。
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

$PY scripts/step10_v14_plan.py | tail -2
$PY scripts/step10_v15_plan.py | tail -3
$PY scripts/step10_v16_plan.py | tail -5

N=18000; SH=24
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/_run_v16_gen.py --rows $lo-$hi > v16_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v16/foa | wc -l

# 距離GT（水平距離・プリロール対応版）
$PY scripts/step19_dist_labels_v16.py | tail -2
ls out/dataset_outdoor_siren_v16/metadata_dist | wc -l

# datasets_v16 ルート（クラス辞書は基準と同一内容をコピー。リンク名は outdoor_siren_v12）
ROOTDIR=$PS/datasets_v16
HDF5DIR=$PS/_hdf5_v16
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16 $ROOTDIR/outdoor_siren_v12

# 空メタ許容ラベル抽出 ＋ adpit_dist（8クラス）。キャッシュは _hdf5_v16 に分離
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true

guard_verify || exit 1
echo V16_GEN_DONE
