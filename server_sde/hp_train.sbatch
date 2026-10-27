#!/bin/bash
#SBATCH -J hptrain
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o hptrain-%j.out
# ④ハイパラ感度 Stage B: 基準学習(w3レシピ)から1因子だけ変えて学習→best ckptでval推論→連結。
# 宣言= md/design/再学習④_ハイパラ感度_方針宣言_2026-09-02.md §2
#
#   sbatch --export=ALL,ARM=lr1e-4 server_sde/hp_train.sbatch
#   ARM ∈ lr1e-4 | cos | 2step | short | bs32 | scratch
#
# 基準(datasets_v12 / _hdf5 / w3run / init ckpt)は読むだけ。guardで冒頭・末尾に指紋照合。
# low QoSで中断→requeue されたら last.ckpt から再開する（train_v11_run2.sbatch と同方式）。
set -ex

source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

: "${ARM:?ARM を指定してください}"
case "$ARM" in
  lr1e-4|cos|2step|short|bs32|scratch) ;;
  *) echo "未知のarm: $ARM"; exit 1 ;;
esac

PS=~/research/PSELDNet/PSELDNets
EXP=outdoor_siren_v12_hp_${ARM}
[ "$EXP" != "outdoor_siren_v12_sde_w3" ] || { echo "基準runと同名"; exit 1; }
cd $PS

# 設定yaml（無ければ生成。生成物は .out に残す）
CFG=configs/experiment/${EXP}.yaml
[ -f "$CFG" ] || .venv/bin/python ~/research/outdoor_seld_e2e/server_sde/hp_make_configs.py "$ARM"
cat "$CFG"

# ---- 損失重み・距離表現は w3 と同一。初期重みは scratch だけ公式ckpt（ヘッド乱数初期化） ----
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
if [ "$ARM" = "scratch" ]; then
  export SDE_INIT_CKPT=""
  PRETRAINED=ckpts/mACCDOA-HTSAT-0.567.ckpt
else
  export SDE_INIT_CKPT=$HOME/PSELDNets_logs/v12_init_from_run3ep84.ckpt
  PRETRAINED=$SDE_INIT_CKPT
fi

# hydraの辞書は**マージ**されるため、StepLR用の step_size/gamma が既定(htsat.yaml)から残る。
# 他のスケジューラでは `~key` で明示的に消す（--cfg job で確認済み、2026-09-02）
EXTRA=""
case "$ARM" in
  cos)   EXTRA="~model.lr_scheduler.kwargs.step_size ~model.lr_scheduler.kwargs.gamma" ;;
  2step) EXTRA="~model.lr_scheduler.kwargs.step_size" ;;
esac

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python _train_sde12.py \
  experiment=$EXP \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5 \
  model.kwargs.pretrained_path=$PRETRAINED \
  $EXTRA $RESUME

# best-val ckpt（w3 job315 と同じ選び方: save_top_k=1 の epoch_*.ckpt）で val 1,800 を通常推論
ck=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | head -1)
echo "using ckpt: $ck"
.venv/bin/python _infer_sde12.py experiment=outdoor_siren_v12_valinfer mode=test \
  ckpt_path=$ck \
  model.kwargs.pretrained_path=null \
  experiment_name=infer_hp_${ARM}_val \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5

python3 - $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_hp_${ARM}_val/submissions <<'EOF'
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

guard_verify || exit 1
echo HP_TRAIN_DONE_$ARM
