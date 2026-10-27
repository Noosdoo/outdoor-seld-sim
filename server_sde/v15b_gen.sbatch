#!/bin/bash
#SBATCH -J v15bgen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -o v15bgen-%j.out
# v13（合成データ修正の束）: 全量生成 9,000本（12シャード） → 距離GT → datasets_v15b ルート → 前処理。
# 設計= md/design/v13設計書_合成データ修正の束_2026-09-02.md
#
# 書き込み先は out/dataset_outdoor_siren_v15b/ ・ $PS/datasets_v15b ・ $PS/_hdf5_v15b のみ。
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

$PY scripts/step10_v14_plan.py | tail -2
$PY scripts_archive/step10_v15b_plan.py | tail -3

N=9000; SH=24
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts_archive/_run_v15b_gen.py --rows $lo-$hi > v15b_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v15b/foa | wc -l

# 距離GT（プリロール対応版）
$PY scripts_archive/step19_dist_labels_v15b.py | tail -2
ls out/dataset_outdoor_siren_v15b/metadata_dist | wc -l

# datasets_v15b ルート（クラス辞書は基準と同一内容をコピー。リンク名は outdoor_siren_v12）
ROOTDIR=$PS/datasets_v15b
HDF5DIR=$PS/_hdf5_v15b
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v15b $ROOTDIR/outdoor_siren_v12

# 空メタ許容ラベル抽出 ＋ adpit_dist（8クラス）。キャッシュは _hdf5_v15b に分離
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR
ls -la $HDF5DIR/label/*/dev/ || true

guard_verify || exit 1
echo V15B_GEN_DONE
