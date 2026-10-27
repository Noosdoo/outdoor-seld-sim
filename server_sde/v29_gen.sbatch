#!/bin/bash
#SBATCH -J v29gen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 17:00:00
#SBATCH --requeue
#SBATCH -o v29gen-%j.out
# v29（変更票 #70・2026-10-19・凍結 r12）の **全部の描き直し = 新しい出力先で 62,102 本（60,502 ＋ D19 1,600）**。
#   v27_gen.sbatch（job 6347）の写しで、名前・出力先・総数の鍵・配備の対象だけ v29。描画の連なりに step11_v29_render（高さはクラス別・圏は横距離・警告音の車 1.5〜15 m・
#   重い車 10%・歩く速さ 1.0〜1.6・隊列の下限 1.0・D10 は cs_lat_m・D19）を足し、plan は step10_v29_plan の新しい plan（雨の dba・cs_lat_m・D19 の行）。
#   - 既存の束（out/dataset_outdoor_siren_v2final・_v26・_v27・$PS/datasets_v2final／v22／v26／v27・_hdf5_*）は消さない・書かない。
#   - 出力先 = out/dataset_outdoor_siren_v2final_v29・$PS/datasets_v29・$PS/_hdf5_v29。
#   - params が fixed なら本番（印なし）。draft のまま回すときは V2FINAL_PROVISIONAL=1 が要る（印 PROVISIONAL.txt が束に残る・本番の学習には使えない）。
#     **発効条件（凍結 r12）: 実装の検査 → params fixed（本人）→ Astra の実装後の監査 → 本人の了承 → まず小さい標本で全経路（--smoke）→ 全量。**
#   - 版の照合: server_sde/v29_expected_sha256.txt（配備のときにサーバ側で作る・LF）。手元と違えば止まる。
#   - 区画: 既定 pro_6000（空きが無ければ sbatch -p a100_3g --gres=gpu:3g.40gb:1 で投入・v27 は 50,574 本に 11.3 h）。
# 投入（本人の許可のあと）: sbatch server_sde/v29_gen.sbatch            （fixed のとき）
#                           sbatch --export=ALL,V2FINAL_PROVISIONAL=1 server_sde/v29_gen.sbatch   （draft のとき・開発用）
#   場所取り（本人 2026-10-19 13:28「サーバの場所だけ取れませんか」）: sbatch --export=ALL,WAIT_FOR_GO=1,WAIT_MAX_H=4 server_sde/v29_gen.sbatch
#     → 資源を取ったまま server_sde/GO_v29 ができるまで 60 秒ごとに待つ（上限 WAIT_MAX_H 時間・超えたら何もせず終わる＝資源を返す）。GO の前に配備と params fixed を済ませる。
#     GO の出し方（本人の了承のあと）: ssh is-server 'touch ~/research/outdoor_seld_e2e/server_sde/GO_v29'。待っている間も -t（17 h・10/19 19:1x 投入で 10/20 13:30 の保守の予約に掛からない長さ）の中で、描き直しは GO のあと。
set -x
cd ~/research/outdoor_seld_e2e
if [ "${WAIT_FOR_GO:-0}" = 1 ]; then
  GO=server_sde/GO_v29; MAXS=$(( ${WAIT_MAX_H:-4} * 3600 )); W=0
  echo "WAIT_FOR_GO: $GO を待つ（上限 ${WAIT_MAX_H:-4} h・job $SLURM_JOB_ID・$(date)）"
  while [ ! -f "$GO" ]; do
    sleep 60; W=$((W + 60))
    [ $W -ge $MAXS ] && { echo "GO が来なかった（${WAIT_MAX_H:-4} h）。何もせず終わる・資源を返す"; exit 0; }
  done
  echo "GO を受けた（$(date)・待ち $((W / 60)) 分）: $(cat $GO 2>/dev/null)"; rm -f "$GO"
fi
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
export V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v29
OUT=out/$V2FINAL_DS_NAME
SRC_PLAN=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
# 発効チェック: params が fixed か、仮の印つきか
$PY - <<'EOF'
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
print("params status =", P.get("status"), "declared_at =", P.get("declared_at"))
sys.exit(0 if P.get("status") == "fixed" else 3)
EOF
rc=$?
if [ "${V2FINAL_PROVISIONAL:-0}" = 1 ]; then
  echo "PROVISIONAL: 仮の数字で描き直す（印が残る・本番の学習には使わない）"
elif [ $rc -ne 0 ]; then
  echo "params が fixed でない。本番の描き直しは fixed にしてから。開発用なら V2FINAL_PROVISIONAL=1 で投入する"; exit 3
fi
for f in scripts/_run_v2final_gen.py scripts/step11_v9_render.py scripts/step11_v12_render.py scripts/step11_v13_render.py scripts/step11_v14_render.py scripts/step11_v16_render.py \
         scripts/step11_v19_render.py scripts/step11_v21_render.py scripts/step11_v22_render.py scripts/step11_v23_render.py scripts/step11_v24_render.py scripts/step11_v25_render.py scripts/step11_v26_render.py \
         scripts/step11_v27_render.py scripts/step11_v29_render.py scripts/step10_v27_plan.py scripts/step10_v28_plan.py scripts/step10_v29_plan.py \
         out/dataset_outdoor_siren_v2final/plan/manifest_v27_mix.csv out/dataset_outdoor_siren_v2final/plan/manifest_v28_topup.csv out/dataset_outdoor_siren_v2final/plan/manifest_v29_d19.csv \
         src/outdoor_seld/alert_sounds.py src/outdoor_seld/alert_sounds_v23.py src/outdoor_seld/bike_bell_v2.py src/outdoor_seld/bicycle_v2.py src/outdoor_seld/calibration.py src/outdoor_seld/engine.py \
         src/outdoor_seld/engine_ev.py src/outdoor_seld/engine_heavy.py src/outdoor_seld/engine_heavy_v25.py src/outdoor_seld/fastsim.py src/outdoor_seld/level_v24.py src/outdoor_seld/motorcycle.py \
         src/outdoor_seld/siren.py src/outdoor_seld/siren_v26.py src/outdoor_seld/avas_v26.py src/outdoor_seld/noise.py src/outdoor_seld/geometry.py src/outdoor_seld/labels.py src/outdoor_seld/foa.py \
         md/design/level_match_v24.json md/design/v2final_params.json scripts/step19_dist_labels_v2final.py scripts/step10_v2final_relabel.py server_sde/_v2final_gen_complete_check.py \
         server_sde/_train_entrance_guard.py $SRC_PLAN; do
  [ -f "$f" ] || { echo "無い: $f（手元から配備してから投入する）"; exit 3; }
done
sha256sum --quiet -c server_sde/v29_expected_sha256.txt || { echo "スクリプトの版が台帳と違う。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v29_expected_sha256.txt) 本）"
# 既存の束に書かないことの確認（出力先は新しい名前・既に中身があれば止める＝描き直しは 1 回）
[ "$OUT" != "out/dataset_outdoor_siren_v2final" ] || { echo "出力先が既存の束と同じ"; exit 1; }
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && { echo "requeue（${SLURM_RESTART_COUNT} 回目）→ 続きから（REUSE=1）"; REUSE=1; }   # AR12-17
if [ -d "$OUT/foa" ] && [ "$(ls $OUT/foa | wc -l)" -gt 0 ]; then echo "出力先 $OUT に既に本がある。描き直しは新しい名前で 1 回だけ（続きから回すなら REUSE=1）"; [ "${REUSE:-0}" = 1 ] || exit 1; fi
mkdir -p $OUT/plan
cp $SRC_PLAN $OUT/plan/assignment_v2final.csv
cp out/dataset_outdoor_siren_v2final/plan/manifest_v26_bellbike.csv $OUT/plan/ 2>/dev/null || true
cp out/dataset_outdoor_siren_v2final/plan/manifest_v27_mix.csv $OUT/plan/ 2>/dev/null || true
cp out/dataset_outdoor_siren_v2final/plan/manifest_v28_topup.csv $OUT/plan/ 2>/dev/null || true
cp out/dataset_outdoor_siren_v2final/plan/manifest_v29_d19.csv $OUT/plan/ 2>/dev/null || true
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P['N_ROWS_TOTAL_EXPECTED_V29'])")
[ "$(($(wc -l < $OUT/plan/assignment_v2final.csv) - 1))" = "$N" ] || { echo "plan の行数が N（$N）と違う"; exit 3; }
$PY scripts/_run_v2final_gen.py --list | tee v29_gen_list_job${SLURM_JOB_ID}.log
# 全部描く（30 シャード）。BLAS／OpenMP のスレッドは 1 プロセス 2 本に絞る（2026-10-13 job 6305: 既定だと 1 プロセスが 200 本以上のスレッドで空回りし、最後の 2 区画が 2.5 分で 0 本になった。新しいプロセスなら 1 本 3 秒）
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SH=30; STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v2final_gen.py --rows $lo-$hi > v29_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls $OUT/foa | wc -l
cat $OUT/GEN_VERSIONS.txt
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py --ds $OUT | tail -2 || { echo "距離ラベルの生成に失敗。中止"; exit 5; }
ls $OUT/metadata_dist | wc -l
$PY scripts/step10_v2final_relabel.py --ds $OUT || { echo "D14 のラベル変換に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds $OUT --check || { echo "D14 の検査に失敗。中止"; exit 4; }
# 9 クラスの置き場（新しい名前）
ROOTDIR=$PS/datasets_v29
HDF5DIR=$PS/_hdf5_v29
for x in datasets_v12 datasets_v2final datasets_v22 datasets_v26 datasets_v27; do [ "$ROOTDIR" != "$PS/$x" ] || { echo "既存ルートと同一"; exit 1; }; done
mkdir -p $ROOTDIR
if [ ! -f $ROOTDIR/cls_indices_train.tsv ]; then
  cp $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
  printf '8\t/m/x_bicycle\tBicycle\t0\t0.0\n' >> $ROOTDIR/cls_indices_train.tsv
fi
[ "$(wc -l < $ROOTDIR/cls_indices_train.tsv)" = 9 ] || { echo "クラス一覧が 9 行でない"; exit 1; }
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/$OUT $ROOTDIR/outdoor_siren_v12
# 学習の入口の門（確定評価の本・実録の置き場が混じっていないこと）を、前処理の前にも掛ける（AR12-01: HDF5 の置き場はまだ無いので先に作る・--allow-empty は「無い HDF5」も許す）
mkdir -p $HDF5DIR
$PY server_sde/_train_entrance_guard.py $ROOTDIR $HDF5DIR --allow-empty || { echo "学習の入口の門で止めた"; exit 2; }
cd $PS
T_PRE=$(date +%s)
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 1（特徴）に失敗。中止"; exit 6; }
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 2（距離ラベル）に失敗。中止"; exit 6; }
ls -la $HDF5DIR/label/*/dev/ || true
cd ~/research/outdoor_seld_e2e
$PY server_sde/_v2final_gen_complete_check.py --n "$N" --hdf5 "$HDF5DIR" --t-pre "$T_PRE" --ds $OUT --dataset-type dev \
  || { echo "完了の検査に失敗（本数・成果物の欠け・HDF5 の更新や中身）。中止"; exit 7; }
$PY - "$HDF5DIR" <<'EOF' || { echo "ラベル h5 のクラス数が 9 でない。中止"; exit 7; }
import glob, sys, h5py
root = sys.argv[1]
for p in glob.glob(f"{root}/label/adpit_dist/dev/*.h5"):
    with h5py.File(p) as hf:
        k = next(iter(hf)); shp = hf[f"{k}/adpit/se"].shape
        print(p, k, shp); assert shp[-1] == 9, shp
EOF
if [ "${V2FINAL_PROVISIONAL:-0}" = 1 ]; then echo "PROVISIONAL v29 $(date)" > $OUT/PROVISIONAL.txt; fi
sha256sum $OUT/plan/assignment_v2final.csv $OUT/GEN_VERSIONS.txt | tee v29_gen_sha256_job${SLURM_JOB_ID}.txt
guard_verify || exit 1
echo V27_GEN_DONE
