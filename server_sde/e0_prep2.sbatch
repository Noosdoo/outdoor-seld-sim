#!/bin/bash
#SBATCH -J e0prep2
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -o e0prep2-%j.out
# E0前処理リトライ: index抽出のみ + ラベルh5はv11本体をsymlink（HPFでもGT不変のため）
set -ex
cd ~/research/PSELDNet/PSELDNets

# クラッシュで書きかけになった部分h5を撤去（hpf専用の新規ファイルのみ。本体は触らない）
rm -f _hdf5/label/accdoa/dev/outdoor_siren_v11_hpf50.h5

# HPFレンダは前回完了済み（3条件×1,200本、検品PASS）。未了なら再実行しても既存はスキップされる
.venv/bin/python _e0_hpf_render.py --src datasets/outdoor_siren_v11 --cutoffs 50 100 200

for fc in 50 100 200; do
  # ① index抽出のみ（ラベル抽出なし）
  .venv/bin/python _e0_preproc_dataonly.py \
    dataset=outdoor_siren_v11_hpf${fc} dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8
  # ② ラベルh5をv11本体へsymlink（存在する種別すべて）
  for t in _hdf5/label/*/dev; do
    src="$t/outdoor_siren_v11.h5"
    [ -f "$src" ] && ln -sfn "$(basename $src)" "$t/outdoor_siren_v11_hpf${fc}.h5" || true
    # basenameでの相対リンク: 同一ディレクトリ内なのでこれで解決する
  done
done
ls -la _hdf5/label/*/dev/ | grep hpf || true
echo E0_PREP2_DONE
