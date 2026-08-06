#!/bin/bash
#SBATCH -J f3confirm
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o f3confirm-%j.out
# 【第8回監査・問2対応】fold3凍結追試: 通知ルールv3.2を凍結コミット後、
# fold3(1,200本)にrun1/run3の両ckptで距離付き推論する（採点はローカル）。
# fold3はv11 SDE予測未生成＝未汚染であることを監査で確認済み。
# 注意（監査の履歴的トラップ指摘）: run1は線形距離時代のckptなので復号は
# SDE_DIST_MODE=linear、run3はlog。取り違えると距離が系統的に誤復号される。
set -ex
PS=~/research/PSELDNet/PSELDNets
cd $PS

# fold3_room1_mix0181 はサーバ転送から漏れていた1本（2026-08-06にscp済み）。
# インデックスcsvにも行を冪等に追記（test側のみ。train側は履歴保存のため触らない）
CSV=_hdf5/data/24000fs/wav/dev/outdoor_siren_v11_10sChunklen_10sHoplen_test.csv
grep -q fold3_room1_mix0181 $CSV || \
  echo "datasets/outdoor_siren_v11/foa/fold3_room1_mix0181.flac,0,240000,0,0" >> $CSV

# fold3用config（valinferのfold2→fold3差し替え）
dcfg=configs/data/outdoor_siren_v11_fold3infer.yaml
sed "s/\[fold2_room1\]/[fold3_room1]/" \
    configs/data/outdoor_siren_v11_valinfer.yaml > $dcfg
ecfg=configs/experiment/outdoor_siren_v11_fold3infer.yaml
sed "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v11_fold3infer.yaml#" \
    configs/experiment/outdoor_siren_v11_valinfer.yaml > $ecfg

# ① run1 ckpt（線形時代）→ linear復号
export SDE_DIST_MODE=linear
.venv/bin/python _infer_sde.py experiment=outdoor_siren_v11_fold3infer mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run1/checkpoints/epoch_094.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v11_sde_run1_fold3 \
  paths.log_dir=$HOME/PSELDNets_logs

# ② run3 ckpt → log復号
export SDE_DIST_MODE=log
.venv/bin/python _infer_sde.py experiment=outdoor_siren_v11_fold3infer mode=test \
  ckpt_path=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/epoch_084.ckpt \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_v11_sde_run3_fold3 \
  paths.log_dir=$HOME/PSELDNets_logs

# 連結
for nm in infer_v11_sde_run1_fold3 infer_v11_sde_run3_fold3; do
python3 - $HOME/PSELDNets_logs/outdoor_siren_v11/runs/$nm/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "fold3_all.csv")
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
echo FOLD3_CONFIRM_DONE
