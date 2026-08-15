#!/bin/bash
#SBATCH -J abltrain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -o abltrain-%j.out
# ablation arm の学習＋val推論＋連結。**v12_g45_w3.sbatch（job315＝v12本番w3）の複製**で、
# 変えるのは①データセットルート ②実験名 の2つだけ。
# 初期重み・W_DIST=3.0・DIST_MODE=log・epoch数は基準と完全同一に保つ
# （「タスクは同じで世界の物理だけ変える」= ablationの原則。ここを揃えないと比較が壊れる）。
#
# 使い方:
#   確認run（既定・基準ルートに一切触れない）:
#     sbatch --export=ALL,ABLATE=no_1r server_sde/abl_train.sbatch
#   本番（フル物理valの転移評価まで行う。基準ルートを**読む**）:
#     sbatch --export=ALL,ABLATE=no_1r,ABL_FULLVAL=1 server_sde/abl_train.sbatch
#
# ABL_FULLVAL=0（既定）では自条件valのみを推論し、基準の datasets_v12 には
# 読み書きとも一切アクセスしない。確認runで基準に触れる理由がないため既定を0にした。
#
# 前提: server_sde/_train_sde12.py と _infer_sde12.py がサーバ側 PSELDNets に同期済みであること
# （sbatchは ~/research/PSELDNet/PSELDNets へ cd してそこの _train_sde12.py を呼ぶ）。
set -ex

# 基準の無変更を機械で保証する（冒頭で指紋を取り、末尾で照合して違えば失敗させる）
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

: "${ABLATE:?ABLATE を指定してください}"
case "$ABLATE" in
  no_1r|no_doppler|no_ground|no_airabs) ;;
  *) echo "未知のarm: $ABLATE"; exit 1 ;;
esac

PS=~/research/PSELDNet/PSELDNets
ROOTDIR=$PS/datasets_v12_abl_${ABLATE}
EXP=outdoor_siren_v12_abl_${ABLATE}
[ -d "$ROOTDIR" ] || { echo "arm用ルートが無い（先に abl_labels.sbatch を回す）: $ROOTDIR"; exit 1; }
[ "$ROOTDIR" != "$PS/datasets_v12" ] || { echo "基準ルートと同一"; exit 1; }

cd $PS
# ---- 以下、初期重み・損失重み・距離表現は v12_g45_w3.sbatch と同一 ----
export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log

.venv/bin/python _train_sde12.py \
  experiment=outdoor_siren_v12 \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR \
  model.kwargs.pretrained_path=$SDE_INIT_CKPT

ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"

# 自条件val（そのarmの世界で学習し、その世界で採点＝「そもそも学習できるか」）
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_${EXP}_selfval \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR

# フル物理val（基準の世界で採点＝転移ギャップ。**主要評価はこちら**）
# 確認runでは既定でスキップする。ここだけが基準ルートに触れる工程のため。
TAGS="selfval"
if [ "${ABL_FULLVAL:-0}" = "1" ]; then
  .venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
    ckpt_path=$ck \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_${EXP}_fullval \
    paths.log_dir=$HOME/PSELDNets_logs \
    paths.dataset_dir=$PS/datasets_v12
  TAGS="selfval fullval"
else
  echo "[abl] ABL_FULLVAL=0 のためフル物理val推論はスキップ（基準ルートに触れない）"
fi

for tag in $TAGS; do
  python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_${EXP}_${tag}/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "val_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out, sum(1 for _ in open(out)))
EOF
done

guard_verify || exit 1
echo ABL_TRAIN_DONE_$ABLATE
