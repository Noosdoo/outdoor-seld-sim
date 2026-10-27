#!/bin/bash
#SBATCH -J hpcausal
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o hpcausal-%j.out
# ④ハイパラ感度 Stage C: 因果ft(ft2レシピ)から1因子だけ変えて微調整→保存ckpt全部を
# val 1,800 の因果推論にかける→連結。宣言= md/design/再学習④_ハイパラ感度_方針宣言_2026-09-02.md §2
#
#   sbatch --export=ALL,ARM=clr3e-5 server_sde/hp_causal.sbatch
#   ARM ∈ cseed | clr3e-5 | clr1e-4 | crep8 | clr1e-4s2 | clr1e-4r8 | clr1e-4e150（追試・宣言§7）
#
# ft2の実レシピ（job3443）= 起点 w3 ep099 / KMIN=40 / **REPEAT=2** / lr3e-4 StepLR60 / 100ep /
# 20ep毎に全保存。ここから1因子だけ変える。基準には触れない（guard）。
# requeue されたら last.ckpt から再開。因果推論は1クリップ毎に保存されるので続きから走る。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
cd $PS

source $HOME/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record

: "${ARM:?ARM を指定してください}"
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
SEED=2024
EVERY=20
case "$ARM" in
  cseed)   export CAUSAL_SEED=1; SEED=20260902 ;;
  clr3e-5|clr1e-4|clr1e-4e150) ;;
  crep8)   export CAUSAL_REPEAT=8; EVERY=5 ;;
  clr1e-4s2) export CAUSAL_SEED=2; SEED=20260903 ;;                 # 追試: 別seed再現
  clr1e-4r8) export CAUSAL_REPEAT=8; EVERY=5 ;;                     # 追試: lr1e-4 × REPEAT8
  *) echo "未知のarm: $ARM"; exit 1 ;;
esac

EXP=outdoor_siren_v12_hp_${ARM}
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/checkpoints/epoch_099.ckpt
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
ls -l $CK

CFG=configs/experiment/${EXP}.yaml
[ -f "$CFG" ] || .venv/bin/python ~/research/outdoor_seld_e2e/server_sde/hp_make_configs.py "$ARM"
cat "$CFG"

# ラッパは src/ を相対で見るので PSELDNets 直下に置いて実行する（causal_train.sbatch と同じ）
cp $HOME/research/outdoor_seld_e2e/server_sde/causal_patch.py $PS/causal_patch.py
cp $HOME/research/outdoor_seld_e2e/server_sde/_train_causal12.py $PS/_train_causal12.py

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python $PS/_train_causal12.py \
  experiment=$EXP \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  seed=$SEED \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$PS/datasets_v12 paths.hdf5_dir=$PS/_hdf5 \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=$EVERY \
  $RESUME

# 保存ckpt全部（epoch_*.ckpt）を val の因果推論へ。出力= infer_hp_<arm>_e<NNN>_causal/
export PS
export CAUSAL_DSDIR=$PS/datasets_v12
export CAUSAL_PREFIX=fold2_
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt); do
  e=$(basename $ck .ckpt)                       # epoch_019
  export CAUSAL_CKPT=$ck
  export CAUSAL_OUT=infer_hp_${ARM}_${e/epoch_/e}_causal
  .venv/bin/python $HOME/research/outdoor_seld_e2e/server_sde/_causal_infer_v12.py
done

guard_verify || exit 1
echo HP_CAUSAL_DONE_$ARM
