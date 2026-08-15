#!/bin/bash
#SBATCH -J ablgen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -c 26
#SBATCH -o ablgen-%j.out
# ablation arm の全量生成（10,200行を12シャード並列）。v12_gen.sbatch のarm版。
#
# 使い方: sbatch --export=ALL,ABLATE=no_1r server_sde/abl_gen.sbatch
#
# 【絶対条件】基準データセット out/dataset_outdoor_siren_v12 には一切書き込まない。
# 出力は out/dataset_outdoor_siren_v12_abl_<arm>/ のみ。ドライバ側にも同じassertがある。
set -eux

: "${ABLATE:?ABLATE を指定してください（例 --export=ALL,ABLATE=no_1r）}"
case "$ABLATE" in
  no_1r|no_doppler|no_ground|no_airabs) ;;
  *) echo "未知のarm: $ABLATE"; exit 1 ;;
esac

E2E=~/research/outdoor_seld_e2e
OUT=$E2E/out/dataset_outdoor_siren_v12_abl_${ABLATE}
BASE=$E2E/out/dataset_outdoor_siren_v12
[ "$OUT" != "$BASE" ] || { echo "出力先が基準と同一"; exit 1; }
echo "arm=$ABLATE 書き込み先=$OUT （基準 $BASE には触れない）"

cd $E2E
PY=~/research/PSELDNet/PSELDNets/.venv/bin/python
export PYTHONPATH=scripts:src
N=10200; SH=12
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v12_abl_gen.py --rows $lo-$hi > ablgen_${ABLATE}_shard_$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done

n=$(ls $OUT/foa | wc -l)
echo "arm=$ABLATE foa=$n"
[ "$n" -eq 10200 ] || { echo "生成数が10200でない: $n"; exit 1; }
echo ABL_GEN_DONE_$ABLATE
