#!/bin/bash
#SBATCH -J e0prep
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -o e0prep-%j.out
# E0: HPF派生データセット生成(fold2のみ) + extract_data 前処理 ×3
set -ex
cd ~/research/PSELDNet/PSELDNets

# ① HPFレンダ（元データ不変更・派生3フォルダ新規作成）
.venv/bin/python _e0_hpf_render.py --src datasets/outdoor_siren_v11 --cutoffs 50 100 200

# ② 各派生を extract_data（dataset名=フォルダ名。Synthesisクラスの汎用マッピングで解決）
for fc in 50 100 200; do
  .venv/bin/python src/preproc.py \
    dataset=outdoor_siren_v11_hpf${fc} dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8
done
echo E0_PREP_DONE
