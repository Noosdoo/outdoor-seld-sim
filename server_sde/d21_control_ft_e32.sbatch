#!/bin/bash
#SBATCH -J d21ctrl32
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o d21ctrl32-%j.out
# D21 の対照つき追加学習 (b) の「更新回数を揃えるやり直し」（2026-10-06 本人「やり直しましょう」・Astra 再監査 RF03）。
#   §3e の 2 腕は同じ 30 ep でも学習セグメント数が 20,400 対 19,200 なので更新回数が 153,030 対 144,030（6.25%）違った。
#   ここでは noD21 腕（24,000 本・kickfar なし）を dev2 の因果 ft e149 から lr 1e-5 で **32 ep** 続け、更新回数 153,632（D21 腕 153,030 との差 +0.39%）に揃える。
#   32 ep で止める（max_steps で途中停止すると最後の ckpt の保存が保証されないため、エポック単位で最も近い 32 を取る）。
#   起点・種・lr・コード・HDF5（_hdf5_v2final_noD21）は §3e の noD21 腕と同じ。学習後、last.ckpt（epoch 31・global_step を確認）で自分の val と v12 val を推論。
#   採点は d21_control_e32_score.sbatch（D21 腕は §3e の cont5_D21_e029 をそのまま使う）。
#   注意（job 5606 の失敗から・10/6 20:58）: last.ckpt は検証の終わり（check_val_every_n_epoch・既定 5）に保存されるので、32 ep だと epoch 31 の状態が保存されない
#   （5606 は 32 ep 走り切ったのに last.ckpt が epoch 29 のままで止まった）。ここでは check_val_every_n_epoch=8（epoch 7・15・23・31 で検証）にして最後が保存されるようにする。
#   検証の回数は学習の中身（重み・更新回数）に影響しない。5606 の last.ckpt（epoch 29）から再開して残り 2 ep を走らせる。
# 投入: sbatch --export=ALL,ARM=noD21 server_sde/d21_control_ft_e32.sbatch
set -x
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
ARM=${ARM:?ARM=noD21（D21 も可）を指定}
EPOCHS=${EPOCHS:-32}
LRARM=${LRARM:-clr1e-5e0${EPOCHS}}
CONT=${CONT:-dev2cont5s}
case "$ARM" in D21) HDF5DIR=$PS/_hdf5_v2final ;; noD21) HDF5DIR=$PS/_hdf5_v2final_noD21 ;; *) echo "ARM が不正: $ARM"; exit 2 ;; esac
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v2final
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CK=$R/outdoor_siren_v2final_dev2_causal_ft/checkpoints/last.ckpt        # dev2 e149（sha b020bb…・§3e と同じ起点）
[ -f "$CK" ] || { echo "dev2 の ckpt が無い: $CK"; exit 2; }
PARAMS=$E2E/md/design/v2final_params.json
MARK=$E2E/out/dataset_outdoor_siren_v2final/PROVISIONAL.txt
[ -f "$MARK" ] || { echo "PROVISIONAL.txt が無い — 開発用台本は仮の版にだけ使う"; exit 3; }
[ "$ARM" = "noD21" ] && { bash $E2E/server_sde/_make_hdf5_noD21.sh || exit 2; }
[ -d "$HDF5DIR" ] || { echo "HDF5 ルートが無い: $HDF5DIR"; exit 2; }
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
EXP=outdoor_siren_v2final_${CONT}_${ARM}_e${EPOCHS}
LASTEP=$((EPOCHS - 1))
# RF06: LRARM の札
RUNDIR=$R/$EXP
if [ -d "$RUNDIR" ]; then
  [ -f "$RUNDIR/lrarm.txt" ] || { echo "停止: $RUNDIR に lrarm.txt が無い"; exit 5; }
  [ "$(cat $RUNDIR/lrarm.txt)" = "$LRARM" ] || { echo "停止: $RUNDIR は LRARM=$(cat $RUNDIR/lrarm.txt) の走り"; exit 5; }
else
  mkdir -p "$RUNDIR"; echo "$LRARM" > "$RUNDIR/lrarm.txt"
fi
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv HEIGHT_JITTER=$JIT HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
CFG=configs/experiment/outdoor_siren_v12_hp_${LRARM}.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py $LRARM
grep -n "lr\|max_epochs\|step_size" $CFG
grep -q "max_epochs: $EPOCHS" $CFG || { echo "停止: $CFG の max_epochs が $EPOCHS でない"; exit 2; }
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $E2E/server_sde/_prediction_integrity.py $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
TRAINCSV=$HDF5DIR/data/24000fs/wav/dev/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv
sha256sum $CK $PS/height_patch.py $PS/causal_patch.py $PS/_train_causal17.py $INFER $CFG $PARAMS $MARK $TRAINCSV | tee $E2E/d21ctrl32_${CONT}_${ARM}_sha256
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) arm=$ARM lrarm=$LRARM cont=$CONT epochs=$EPOCHS hdf5=$HDF5DIR train_rows=$(wc -l < $TRAINCSV)" | tee -a $E2E/d21ctrl32_${CONT}_${ARM}_sha256

CKDIR=$R/$EXP/checkpoints
LAST=$CKDIR/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
done_epoch() { [ -f "$LAST" ] && .venv/bin/python - "$LAST" "$LASTEP" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if int(ck.get("epoch", -1)) >= int(sys.argv[2]) else 1)
PY
}
if ! done_epoch; then
.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_${LRARM} \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=10 \
  trainer.check_val_every_n_epoch=${VALEVERY:-8} \
  $RESUME || { echo "学習に失敗"; exit 4; }
fi
done_epoch || { echo "last.ckpt が epoch $LASTEP に達していない"; exit 4; }
# 更新回数の記録（RF03 の確認のため）
.venv/bin/python - "$LAST" <<'PY' | tee -a $E2E/d21ctrl32_${CONT}_${ARM}_sha256
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"last.ckpt epoch={ck.get('epoch')} global_step={ck.get('global_step')}")
PY
sha256sum $LAST | tee -a $E2E/d21ctrl32_${CONT}_${ARM}_sha256

# 推論（last.ckpt = epoch LASTEP）: 自分の val と v12 の val 1,800 本
E3=$(printf "%03d" $LASTEP)
export PS HEIGHT_JITTER=0 CAUSAL_PREFIX=val_syn_ CAUSAL_CKPT=$LAST
export CAUSAL_DSDIR=$PS/datasets_v12 CAUSAL_OUT=infer_v2final${CONT}${ARM}_e${E3}_v12causal HEIGHT_TABLE=""
unset CAUSAL_GLOB
.venv/bin/python $INFER || { echo "v12 val の推論に失敗"; exit 4; }
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv CAUSAL_DSDIR=$ROOTDIR CAUSAL_OUT=infer_v2final${CONT}${ARM}_e${E3}_selfcausal
.venv/bin/python $INFER || { echo "自分の val の推論に失敗"; exit 4; }
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo "D21_CONTROL_FT_E32_DONE arm=$ARM epochs=$EPOCHS $(date)"
