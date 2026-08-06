#!/bin/bash
#SBATCH -J v12bl
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o v12bl-%j.out
# v12基準線: run3(6クラスSDE)を v12-val core 1,200本 で推論する。
# 8クラス辞書と衝突しないよう「別名データセット outdoor_siren_v12bl」
# （foa=v12 fold2 coreのsymlink・ラベルh5=v11本体を流用=形状6クラス）を6クラスルートに組む。
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
cd $PS

# ① 別名データセット組立（fold2 core 1,200本のみ）
mkdir -p datasets/outdoor_siren_v12bl/foa
for i in $(seq -w 1 1200); do
  f=$E2E/out/dataset_outdoor_siren_v12/foa/fold2_room1_mix0$i.flac
  ln -sfn $f datasets/outdoor_siren_v12bl/foa/fold2_room1_mix0$i.flac
done
for item in datasets/outdoor_siren_v11/*; do
  base=$(basename $item); [ "$base" = "foa" ] && continue
  [ -e datasets/outdoor_siren_v12bl/$base ] || ln -s $(readlink -f $item) datasets/outdoor_siren_v12bl/$base
done
# ラベルh5（6クラス形状。GT実体はローカル採点でv12正を使う）
for t in _hdf5/label/*/dev; do
  src=$t/outdoor_siren_v11.h5
  [ -f $src ] && ln -sfn $(basename $src) $t/outdoor_siren_v12bl.h5 || true
done

# ② index抽出（データのみ）
.venv/bin/python _e0_preproc_dataonly.py dataset=outdoor_siren_v12bl \
  dataset_type=dev wav_format=.flac mode=extract_data num_workers=8

# ③ config複製 → run3で推論 → 連結
dcfg=configs/data/outdoor_siren_v12bl.yaml
sed "s/outdoor_siren_v11/outdoor_siren_v12bl/g" configs/data/outdoor_siren_v11_valinfer.yaml > $dcfg
ecfg=configs/experiment/outdoor_siren_v12bl.yaml
sed -e "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v12bl.yaml#" \
    configs/experiment/outdoor_siren_v11_valinfer.yaml > $ecfg

.venv/bin/python _infer_sde.py experiment=outdoor_siren_v12bl mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_run3_on_v12val \
  paths.log_dir=$HOME/PSELDNets_logs

python3 - $HOME/PSELDNets_logs/outdoor_siren_v11/runs/infer_run3_on_v12val/submissions <<'EOF'
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
print("wrote", out)
EOF
echo V12_BASELINE_DONE
