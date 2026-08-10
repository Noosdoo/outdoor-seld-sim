#!/bin/bash
#SBATCH -J v12conf
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -c 14
#SBATCH -o v12conf-%j.out
# v12確定評価セット（事前登録= md/design/v12確定評価セット_事前登録_2026-08-10.md）
# 一気通貫・介入なし: 生成(6並列)→距離GT→ラベル→w3推論→連結→core通知v3.3/v3.4+B2個票
# 停止規則により本ジョブは1回だけ実行する。
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
cd $E2E
export PYTHONPATH=scripts:src

# 証跡: 実行ファイルのsha256（scp運用の未追跡対策）
sha256sum scripts/step10_v12_conf_plan.py scripts/step11_v12_conf_render.py \
  scripts/_run_v12_conf_gen.py scripts/_step19_v12conf.py \
  scripts/step12_notify_v33.py scripts/_v12_conf_notify_server.py \
  scripts/step11_v12_render.py | tee conf_sha256_job${SLURM_JOB_ID}.txt

# ① 生成（6シャード並列）
pids=()
for i in 0 1 2 3 4 5; do
  lo=$((i * 300)); hi=$((lo + 299))
  $PY scripts/_run_v12_conf_gen.py --rows $lo-$hi > v12conf_shard_$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
ls out/dataset_outdoor_siren_v12_conf/foa | wc -l

# ② 距離GT（全行一致検証つき）
$PY scripts/_step19_v12conf.py

# ③ datasets_v12ルートへ接続＋ラベル抽出（空メタ許容）＋adpit_dist
[ -e $PS/datasets_v12/outdoor_siren_v12_conf ] || \
  ln -s $E2E/out/dataset_outdoor_siren_v12_conf $PS/datasets_v12/outdoor_siren_v12_conf
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12_conf dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 paths.dataset_dir=$PS/datasets_v12
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12_conf \
    dataset_type=dev paths.dataset_dir=$PS/datasets_v12

# ④ config複製 → w3で推論 → 連結
dcfg=configs/data/outdoor_siren_v12_confinfer.yaml
sed -e "s/outdoor_siren_v11/outdoor_siren_v12_conf/g" \
    -e "s/\[fold2_room1\]/[fold20_room1]/" \
    configs/data/outdoor_siren_v11_valinfer.yaml > $dcfg
ecfg=configs/experiment/outdoor_siren_v12_confinfer.yaml
sed -e "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v12_confinfer.yaml#" \
    -e "s/task_name: outdoor_siren_v11/task_name: outdoor_siren_v12/" \
    configs/experiment/outdoor_siren_v11_valinfer.yaml > $ecfg

export SDE_DIST_MODE=log
ck=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
$PY _infer_sde12.py experiment=outdoor_siren_v12_confinfer mode=test \
  ckpt_path=$ck model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_w3_conf \
  paths.log_dir=$HOME/PSELDNets_logs paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_conf/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "conf_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out, sum(1 for _ in open(out)))
EOF

# ⑤ core通知 v3.3/v3.4（B2個票つき）
cd $E2E
PYTHONPATH=scripts:src $PY scripts/_v12_conf_notify_server.py \
  $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_conf/conf_all.csv \
  $E2E/out/v12_conf_notify_v33 $E2E/out/v12_conf_notify_v34
echo V12_CONF_ALL_DONE
