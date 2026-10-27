#!/bin/bash
#SBATCH -J v2fgen
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o v2fgen-%j.out
# 確定評価 v2 の最終学習データ生成（2026-09-07 準備・11 月に 1 回だけ実行）: plan（v2final_params.json のつまみ）→ 全量生成 24,000 本（30 シャード・
# 2026-09-21 改訂 r2 で発進 1,500 本を追加）→ 距離GT（水平）→ D14 → datasets_v2final ルート → 前処理。実行計画 = md/design/確定評価v2_最終学習_実行計画_2026-09-07.md
# 2026-09-22: 使うスクリプトが手元と同じ版かを server_sde/v2final_expected_sha256.txt と照合してから生成する（9/14 の仮の版は step11_v13_render.py が
# D1 実装前の古い版のままで、D1 の効かない束を作った → dataset_outdoor_siren_v2final_prov0914_noD1 に退避）
# 書き込み先は out/dataset_outdoor_siren_v2final/ ・ $PS/datasets_v2final ・ $PS/_hdf5_v2final のみ。基準（v12）は guard で無変更を保証。
set -x
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
# 発効チェック: params が fixed で、git の HEAD に入っていること（証跡）
$PY - <<'EOF'
import json, sys
P = json.load(open("md/design/v2final_params.json", encoding="utf-8"))
print("params status =", P.get("status"), "declared_at =", P.get("declared_at"))
sys.exit(0 if P.get("status") == "fixed" else 3)
EOF
rc=$?
if [ "${V2FINAL_PROVISIONAL:-0}" = 1 ]; then
  echo "PROVISIONAL: 仮の数字で生成する（2026-09-14 本人指示）。本番の学習に使う前に、実測値を入れて確定し、この台本をもう一度通す（設定が変わった行だけ作り直される）"
else
  [ $rc -eq 0 ] || { echo "設定がまだ確定していない。中止（仮の数字で作るなら V2FINAL_PROVISIONAL=1）"; exit 3; }
  git status --porcelain md/design/v2final_params.json | grep -q . && { echo "設定ファイルが未コミット。中止"; exit 3; }
fi
# 必要なスクリプトがサーバにそろっているか先に確かめる（2026-09-14: 距離ラベルのスクリプトが未配備のまま走り、ラベル 0 本で前処理が失敗した）
for f in scripts/step10_v2final_plan.py scripts/step10_v19_plan.py scripts/_run_v2final_gen.py scripts/step19_dist_labels_v2final.py scripts/step19_dist_labels_v16.py \
         scripts/step10_v2final_relabel.py scripts/step11_v21_render.py scripts/step11_v19_render.py scripts/step11_v16_render.py scripts/step11_v14_render.py scripts/step11_v13_render.py \
         server_sde/_v2final_gen_complete_check.py; do
  [ -f "$f" ] || { echo "無い: $f（手元から配備してから投入する）"; exit 3; }
done
sha256sum md/design/v2final_params.json scripts/step10_v2final_plan.py scripts/step10_v19_plan.py scripts/_run_v2final_gen.py scripts/step19_dist_labels_v2final.py \
  scripts/step10_v2final_relabel.py scripts/step11_v21_render.py scripts/step11_v19_render.py scripts/step11_v16_render.py scripts/step11_v14_render.py scripts/step11_v13_render.py \
  server_sde/_v2final_gen_complete_check.py | tee v2final_sha256_job${SLURM_JOB_ID}.txt
# 意図した版か（手元でコミットした版の sha256 との照合・2026-09-22）。1 本でも違えば配備してから投入し直す
sha256sum --quiet -c server_sde/v2final_expected_sha256.txt || { echo "スクリプトの版が手元と違う（server_sde/v2final_expected_sha256.txt と不一致）。配備してから投入する。中止"; exit 3; }
echo "版の照合 OK（$(wc -l < server_sde/v2final_expected_sha256.txt) 本）"

# plan（土台 18,000 ＋ 新場面 6,000 = 24,000・設計凍結 2026-09-14 §2・改訂 r2 2026-09-21）を作り、構造と軌道の検査に落ちたら生成しない
$PY scripts/step10_v2final_plan.py --check-v16 --check > v2final_plan_check_job${SLURM_JOB_ID}.log 2>&1 || { tail -20 v2final_plan_check_job${SLURM_JOB_ID}.log; echo "plan の検査に失敗。中止"; exit 3; }
tail -8 v2final_plan_check_job${SLURM_JOB_ID}.log
N=$($PY -c "import json;P=json.load(open('md/design/v2final_params.json'));print(P.get('N_ROWS_TOTAL_EXPECTED', P['N_ROWS_EXPECTED']))"); SH=30
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  $PY scripts/_run_v2final_gen.py --rows $lo-$hi > v2final_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; exit 1; }
ls out/dataset_outdoor_siren_v2final/foa | wc -l

# 距離GT（水平距離・プリロール対応版）。監査 Astra 9/28 R05: 各工程の終了値を伝える（| tail で終了値が消えないよう pipefail）
set -o pipefail
$PY scripts/step19_dist_labels_v2final.py | tail -2 || { echo "距離ラベルの生成に失敗。中止"; exit 5; }
ls out/dataset_outdoor_siren_v2final/metadata_dist | wc -l
# D14（設計凍結 §2・2026-09-14）: 列車クリップ（scenario=v12train）の crossing 行を metadata/ と metadata_dist/ から落とす。元は *_orig/ に残し、記録 relabel_D14.json
$PY scripts/step10_v2final_relabel.py --ds out/dataset_outdoor_siren_v2final || { echo "D14 のラベル変換に失敗。中止"; exit 4; }
$PY scripts/step10_v2final_relabel.py --ds out/dataset_outdoor_siren_v2final --check || { echo "D14 の検査に失敗。中止"; exit 4; }

# datasets_v2final ルート（リンク名は outdoor_siren_v12 = hydra の dataset 設定を変えないため）
ROOTDIR=$PS/datasets_v2final
HDF5DIR=$PS/_hdf5_v2final
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
mkdir -p $ROOTDIR
cp -n $PS/datasets_v12/cls_indices_train.tsv $ROOTDIR/cls_indices_train.tsv
[ -e $ROOTDIR/outdoor_siren_v12 ] || ln -s ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v2final $ROOTDIR/outdoor_siren_v12

cd $PS
T_PRE=$(date +%s)
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12 dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 1（特徴）に失敗。中止"; exit 6; }
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12 \
    dataset_type=dev paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR || { echo "前処理 2（距離ラベル）に失敗。中止"; exit 6; }
ls -la $HDF5DIR/label/*/dev/ || true
cd ~/research/outdoor_seld_e2e
# 完了の検査（監査 R05 → 再監査 10/1 RA01 で強化）: plan の全 ID に音声・ラベル・距離ラベル・masks がそろい（空ファイルは欠け）、
# HDF5 ルートの *.h5 が全部この job で更新され、ラベル h5（label/adpit・label/adpit_dist）が存在して h5py で開け、plan の全 clip を持つことを確かめてから完了印を出す。
# 検査 = scripts/test_v2final_gen_complete_check.py（unrelated.h5 だけ・片方だけ更新・clip 欠け・空の出力の反例で NG になる）
$PY server_sde/_v2final_gen_complete_check.py --n "$N" --hdf5 "$HDF5DIR" --t-pre "$T_PRE" --ds out/dataset_outdoor_siren_v2final --dataset-type dev \
  || { echo "完了の検査に失敗（本数・成果物の欠け・HDF5 の更新や中身）。中止"; exit 7; }

guard_verify || exit 1
echo V2FINAL_GEN_DONE
