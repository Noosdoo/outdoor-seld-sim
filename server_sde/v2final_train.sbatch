#!/bin/bash
#SBATCH -J v2ftrain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v2ftrain-%j.out
# 確定評価 v2 の基準学習 = v17b レシピそのまま（w3: init=v22_init9_from_run3ep84〔9 クラス用・凍結 §3 r8・2026-10-19 に 8 クラス用 v12_init から直した＝勉強チャット 10/17 23:1x の指摘〕 / W_DIST=3 / log / batch8 / lr3e-4 / StepLR60 / 100ep
# ＋ 装着高さの入力 HEIGHT_COND=1・学習ゆらぎ 0.10 m）。データだけ datasets_v2final。11 月に 1 回だけ。requeue 時は last.ckpt から再開。
set -ex
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
DATA=${V2FINAL_DATA:-v29}                     # 2026-10-12: 学習データの置き場の札（v2final＝9/22 の束・v22＝10/8 の 9 クラス・v26＝音の直しの描き直し・v27＝組み合わせの網羅＋遠いバイク #64）。既定 v27（2026-10-13）
ROOTDIR=$PS/datasets_$DATA
HDF5DIR=$PS/_hdf5_$DATA
MARK=$ROOTDIR/outdoor_siren_v12/PROVISIONAL.txt           # 束に付く仮の版の印（AR12-05: 本番は印があれば止める・開発用は無ければ止める）
[ ! -f "$MARK" ] || { echo "束 $ROOTDIR に仮の版の印 $MARK がある → 本番の学習はしない（確定した設定で描き直してから）"; exit 3; }
EXP=outdoor_siren_v2final_${DATA}_sde_w3            # AR12-06: 実験名に学習データの札（v26 → outdoor_siren_v2final_v26_sde_w3）。古い last.ckpt から黙って再開しない
# #71（2026-10-19）: 追加の種（s1／s2）。未設定 = 主 seed（台本の動きは変わらない）。札があるときだけ run 名・seed・HEIGHT_SEED を上書き
SEED_TAG=${V2FINAL_SEED_TAG:-}; SEED_OVR=""
case "$SEED_TAG" in
  "") ;;
  s1) EXP=${EXP}_s1; SEED_OVR="seed=20261001"; export HEIGHT_SEED=20261001 ;;
  s2) EXP=${EXP}_s2; SEED_OVR="seed=20261002"; export HEIGHT_SEED=20261002 ;;
  *) echo "V2FINAL_SEED_TAG は s1 か s2（$SEED_TAG）"; exit 3 ;;
esac
# #74（2026-10-21・本人 01:00「1〜4 で」）: アブレーション。V2FINAL_ABL_TAG = scratch（事前学習なし = pretrained_path null・SDE の頭も初期化のまま）／noheight（高さの入力なし = HEIGHT_COND=0）。
#   未設定 = 主の学習（台本の動きは変わらない）。run 名に _abl_<tag> を付ける（主の run に混ぜない）
ABL_TAG=${V2FINAL_ABL_TAG:-}; ABL_SUF=""; ABL_HEIGHT=1; ABL_INIT="$HOME/PSELDNets_logs/v22_init9_from_run3ep84.ckpt"
case "$ABL_TAG" in
  "") ;;
  scratch) ABL_SUF=_abl_scratch; EXP=${EXP}_abl_scratch; ABL_INIT="" ;;
  noheight) ABL_SUF=_abl_noheight; EXP=${EXP}_abl_noheight; ABL_HEIGHT=0 ;;
  *) echo "V2FINAL_ABL_TAG は scratch か noheight（$ABL_TAG）"; exit 3 ;;
esac
# 2026-10-26 本体 §3-2（変更票 #75）: 小さな比べ（v30_gain_compare.sbatch）と v30 の鎖のための入口。どれも未設定なら台本の動きは変わらない。
#   V2FINAL_EXP_SUFFIX = 実験名の後ろの札（例 _gain20・英数字と _ だけ）／V2FINAL_EXTRA_OVR = hydra の追加の設定（例 trainer.max_epochs=20）／
#   SDE_INPUT_GAIN_DB（入力の倍率・gain_patch）と SDE_DIST_MASK（距離の損失のマスク）は呼ぶ側の環境をそのまま通す（明示されていれば 0 でも run に input_gain.json）
EXP_SUF=${V2FINAL_EXP_SUFFIX:-}
[[ "$EXP_SUF" =~ ^[A-Za-z0-9_]*$ ]] || { echo "V2FINAL_EXP_SUFFIX は英数字と _ だけ（$EXP_SUF）"; exit 3; }
EXP=${EXP}${EXP_SUF}
EXTRA_OVR=${V2FINAL_EXTRA_OVR:-}
PARAMS=$E2E/md/design/v2final_params.json
[ -d "$ROOTDIR" ] || { echo "$ROOTDIR が無い（先に生成の台本 v2final_gen.sbatch／v26_gen.sbatch）"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }
$PS/.venv/bin/python $E2E/server_sde/_train_entrance_guard.py "$ROOTDIR" "$HDF5DIR" || { echo "学習の入口の門で止めた（確定評価の本・実録の置き場が学習データに混じっている）"; exit 2; }   # 2026-10-12 引き継ぎ §3-8
$PS/.venv/bin/python -c "import json,sys;P=json.load(open('$PARAMS'));sys.exit(0 if P.get('status')=='fixed' else 3)" || { echo "params が fixed でない"; exit 3; }
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
cd $PS
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/_train_sde17.py $PS/_train_sde17.py
sha256sum $PS/height_patch.py $PS/_train_sde17.py $HOME/PSELDNets_logs/v22_init9_from_run3ep84.ckpt $PARAMS | tee $E2E/v2final_train_sha256_job${SLURM_JOB_ID}.txt
if [ -n "${SDE_INPUT_GAIN_DB:-}" ]; then
  cp $E2E/server_sde/gain_patch.py $PS/gain_patch.py          # 本体 §3-2: 倍率の入口（明示されていれば 0 でも写す・未設定なら写さない＝旧の記録のまま）
  sha256sum $PS/gain_patch.py | tee -a $E2E/v2final_train_sha256_job${SLURM_JOB_ID}.txt
fi
if [ -n "$ABL_INIT" ]; then export SDE_INIT_CKPT=$ABL_INIT; INIT_ARG="model.kwargs.pretrained_path=$SDE_INIT_CKPT"; else unset SDE_INIT_CKPT; INIT_ARG="model.kwargs.pretrained_path=null"; echo "[abl] 事前学習なし（白紙から・SDE の頭も初期化のまま）"; fi
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export HEIGHT_COND=$ABL_HEIGHT
[ "$ABL_HEIGHT" = 1 ] || echo "[abl] 高さの入力なし（HEIGHT_COND=0）"
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export HEIGHT_JITTER=$JIT
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} jitter=$JIT"
if [ -n "${SDE_INPUT_GAIN_DB:-}" ]; then   # 本体 §3-2: run の記録（明示なら 0 でも・ckpt にも gain_patch が sde_input_gain_db を書く）
  RUND=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP; mkdir -p $RUND
  echo "{\"input_gain_db\": ${SDE_INPUT_GAIN_DB}, \"sde_dist_mask\": \"${SDE_DIST_MASK:-0}\", \"job\": \"${SLURM_JOB_ID}\"}" > $RUND/input_gain.json
fi

.venv/bin/python _train_sde17.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  $INIT_ARG \
  $SEED_OVR \
  $EXTRA_OVR \
  $RESUME

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
if [ "$HEIGHT_COND" = 1 ]; then
.venv/bin/python - "$ck" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
keys = [k for k in sd if "h_mlp" in k]
assert keys, "h_mlp が ckpt に無い"
print("[height] h_mlp keys:", len(keys))
EOF
fi
guard_verify || exit 1
echo V2FINAL_TRAIN_DONE
