#!/bin/bash
#SBATCH -J ablinf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o ablinf-%j.out
# ablation arm の**推論のみ**を回す（学習が時間制限で切れた場合の受け皿）。
#
# 使い方:
#   sbatch --export=ALL,ABLATE=no_1r server_sde/abl_infer.sbatch
#   sbatch --export=ALL,ABLATE=no_1r,ABL_CKPT=last.ckpt,ABL_FULLVAL=1 ...
#
# ABL_CKPT: 使うチェックポイント名（既定 last.ckpt）。学習が途中で切れたときは
#           last.ckpt が最終エポックの重みになる。
# ABL_FULLVAL: 1でフル物理val（基準ルートを読む）も推論。確認runでは既定0。
set -ex

: "${ABLATE:?ABLATE を指定してください}"
CK_NAME="${ABL_CKPT:-last.ckpt}"

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

PS=~/research/PSELDNet/PSELDNets
ROOTDIR=$PS/datasets_v12_abl_${ABLATE}
EXP=outdoor_siren_v12_abl_${ABLATE}
HDF5DIR=$PS/_hdf5_abl_${ABLATE}
ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/$CK_NAME
[ -f "$ck" ] || { echo "ckptが無い: $ck"; exit 1; }
echo "using ckpt: $ck"
# 【2026-08-17 監査対応F】どのデータを読んだかをログに残す（従来は間接証拠しか無かった）
echo "[paths] 自条件val  : dataset_dir=$ROOTDIR  hdf5_dir=$HDF5DIR"
echo "[paths] フル物理val: dataset_dir=$PS/datasets_v12  hdf5_dir=$PS/_hdf5"
ls -l $ROOTDIR/outdoor_siren_v12 $PS/datasets_v12/outdoor_siren_v12

cd $PS
# 自条件val（その世界で学習し、その世界で採点）
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_${EXP}_selfval \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR

TAGS="selfval"
if [ "${ABL_FULLVAL:-0}" = "1" ]; then
  .venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
    ckpt_path=$ck \
    model.kwargs.pretrained_path=null \
    experiment_name=infer_${EXP}_fullval \
    paths.log_dir=$HOME/PSELDNets_logs \
    paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5
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
echo ABL_INFER_DONE_$ABLATE
