#!/bin/bash
#SBATCH -J v12ev
#SBATCH -p pro_6000
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -c 14
#SBATCH -o v12ev-%j.out
# v12評価専用1,500本: 生成(6並列) → 距離GT → ラベル抽出 → v12モデルで推論 → 連結
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
PY=$PS/.venv/bin/python
cd $E2E
export PYTHONPATH=scripts:src

# ① 生成（6シャード並列）
pids=()
for i in 0 1 2 3 4 5; do
  lo=$((i * 250)); hi=$((lo + 249))
  $PY scripts/_run_v12_eval_gen.py --rows $lo-$hi > v12ev_shard_$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
ls out/dataset_outdoor_siren_v12_eval/foa | wc -l

# ② 距離GT（全行一致検証つき）
$PY scripts/_step19_v12eval.py

# ③ datasets_v12ルートへ接続＋ラベル抽出（空メタ許容）＋adpit_dist
[ -e $PS/datasets_v12/outdoor_siren_v12_eval ] || \
  ln -s $E2E/out/dataset_outdoor_siren_v12_eval $PS/datasets_v12/outdoor_siren_v12_eval
cd $PS
$PY _preproc_emptyok12.py dataset=outdoor_siren_v12_eval dataset_type=dev \
    wav_format=.flac mode=extract_data num_workers=8 paths.dataset_dir=$PS/datasets_v12
$PY _preproc_sde.py mode=extract_data dataset=outdoor_siren_v12_eval \
    dataset_type=dev paths.dataset_dir=$PS/datasets_v12

# ④ config複製 → v12モデルで推論 → 連結
dcfg=configs/data/outdoor_siren_v12_evalinfer.yaml
sed -e "s/outdoor_siren_v11/outdoor_siren_v12_eval/g" \
    -e "s/\[fold2_room1\]/[fold10_room1, fold11_room1, fold12_room1, fold13_room1]/" \
    configs/data/outdoor_siren_v11_valinfer.yaml > $dcfg
ecfg=configs/experiment/outdoor_siren_v12_evalinfer.yaml
sed -e "s#/data: outdoor_siren_v11_valinfer.yaml#/data: outdoor_siren_v12_evalinfer.yaml#" \
    -e "s/task_name: outdoor_siren_v11/task_name: outdoor_siren_v12/" \
    configs/experiment/outdoor_siren_v11_valinfer.yaml > $ecfg

RUN=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_run1
ck=$(ls $RUN/checkpoints/epoch_*.ckpt | head -1)
$PY _infer_sde12.py experiment=outdoor_siren_v12_evalinfer mode=test \
  ckpt_path=$ck model.kwargs.pretrained_path=null \
  experiment_name=infer_v12_sde_run1_eval \
  paths.log_dir=$HOME/PSELDNets_logs paths.dataset_dir=$PS/datasets_v12

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_run1_eval/submissions <<'EOF'
import sys, glob, os
sub = sys.argv[1]
out = os.path.join(os.path.dirname(sub), "eval_all.csv")
with open(out, "w") as w:
    for f in sorted(glob.glob(os.path.join(sub, "*.csv"))):
        clip = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            line = line.strip()
            if line:
                w.write(clip + "," + line + "\n")
print("wrote", out)
EOF
echo V12_EVAL_ALL_DONE
