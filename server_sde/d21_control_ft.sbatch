#!/bin/bash
#SBATCH -J d21ctrl
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o d21ctrl-%j.out
# D21 の対照つき追加学習 (b)（2026-10-01 本人「(b)〜(c) 今やっちゃいましょう」・敵対レビュー B09 の案）。
#   dev2 の因果 ft e149（last.ckpt）から、同じ種・同じエポック数（lr 1e-4・30 ep）で続けて学習する。ARM=D21 は 25,500 本（kickfar あり）、
#   ARM=noD21 は 24,000 本（kickfar なし・対照）。2 本の差は D21 のデータの分と、それに伴う更新回数の差（学習セグメント 20,400 対 19,200 →
#   30 ep の global_step 153,030 対 144,030・6.25%・Astra 再監査 10/6 RF03）。「歩数がそろう」と書いていたのは誤り。
#   学習後、最後の ckpt（epoch_029）で自分の val 5,100 本と v12 の val 1,800 本を推論する。採点は d21_control_score.sbatch（両腕の完了後）。
# 投入: sbatch --export=ALL,ARM=D21 server_sde/d21_control_ft.sbatch ; sbatch --export=ALL,ARM=noD21 server_sde/d21_control_ft.sbatch
set -x
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
ARM=${ARM:?ARM=D21 か noD21 を指定}
# 2026-10-01 22:10 やり直し用: LRARM = 学習設定の arm（既定 clr1e-4e030 = 1 回目。やり直しは clr1e-5e030 = e149 時点の lr 1e-5）、CONT = 実験名と推論名の札（既定 dev2cont）
LRARM=${LRARM:-clr1e-4e030}
CONT=${CONT:-dev2cont}
case "$ARM" in D21) HDF5DIR=$PS/_hdf5_v2final ;; noD21) HDF5DIR=$PS/_hdf5_v2final_noD21 ;; *) echo "ARM が不正: $ARM"; exit 2 ;; esac
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v2final
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CK=$R/outdoor_siren_v2final_dev2_causal_ft/checkpoints/last.ckpt        # dev2 e149（sha b020bb…）
[ -f "$CK" ] || { echo "dev2 の ckpt が無い: $CK"; exit 2; }
PARAMS=$E2E/md/design/v2final_params.json
MARK=$E2E/out/dataset_outdoor_siren_v2final/PROVISIONAL.txt
[ -f "$MARK" ] || { echo "PROVISIONAL.txt が無い — 開発用台本は仮の版にだけ使う"; exit 3; }
[ "$ARM" = "noD21" ] && { bash $E2E/server_sde/_make_hdf5_noD21.sh || exit 2; }
[ -d "$HDF5DIR" ] || { echo "HDF5 ルートが無い: $HDF5DIR"; exit 2; }
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
EXP=outdoor_siren_v2final_${CONT}_${ARM}_e30
# RF06（Astra 再監査 10/6）: LRARM と CONT の取り違えを止める。走りのフォルダに lrarm.txt を置き、既存のフォルダは札が一致するときだけ続ける
RUNDIR=$R/$EXP
if [ -d "$RUNDIR" ]; then
  [ -f "$RUNDIR/lrarm.txt" ] || { echo "停止: $RUNDIR に lrarm.txt が無い（10/6 より前の走り）。続けるなら中身を確かめて手で LRARM を書く"; exit 5; }
  [ "$(cat $RUNDIR/lrarm.txt)" = "$LRARM" ] || { echo "停止: $RUNDIR は LRARM=$(cat $RUNDIR/lrarm.txt) の走り。いまの LRARM=$LRARM と違う → CONT を変える"; exit 5; }
else
  mkdir -p "$RUNDIR"; echo "$LRARM" > "$RUNDIR/lrarm.txt"
fi
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv HEIGHT_JITTER=$JIT HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
CFG=configs/experiment/outdoor_siren_v12_hp_${LRARM}.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py $LRARM
grep -n "lr\|max_epochs\|step_size" $CFG
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $E2E/server_sde/_prediction_integrity.py $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
sha256sum $CK $PS/height_patch.py $PS/causal_patch.py $PS/_train_causal17.py $INFER $CFG $PARAMS $MARK $HDF5DIR/data/24000fs/wav/dev/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv | tee $E2E/d21ctrl_${CONT}_${ARM}_sha256_job${SLURM_JOB_ID}.txt
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) arm=$ARM lrarm=$LRARM cont=$CONT hdf5=$HDF5DIR train_rows=$(wc -l < $HDF5DIR/data/24000fs/wav/dev/outdoor_siren_v12_10sChunklen_10sHoplen_train.csv) restart=${SLURM_RESTART_COUNT:-0}"

CKDIR=$R/$EXP/checkpoints
LAST=$CKDIR/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
if [ ! -f "$CKDIR/epoch_029.ckpt" ]; then
.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_${LRARM} \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=10 \
  $RESUME || { echo "学習に失敗"; exit 4; }
fi
[ -f "$CKDIR/epoch_029.ckpt" ] || { echo "epoch_029.ckpt が無い"; exit 4; }

# 推論（最後の ckpt だけ）: 自分の val 5,100 本（kickfar 込み）と v12 の val 1,800 本
export PS HEIGHT_JITTER=0 CAUSAL_PREFIX=val_syn_ CAUSAL_CKPT=$CKDIR/epoch_029.ckpt
export CAUSAL_DSDIR=$PS/datasets_v12 CAUSAL_OUT=infer_v2final${CONT}${ARM}_e029_v12causal HEIGHT_TABLE=""
unset CAUSAL_GLOB
.venv/bin/python $INFER || { echo "v12 val の推論に失敗"; exit 4; }
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv CAUSAL_DSDIR=$ROOTDIR CAUSAL_OUT=infer_v2final${CONT}${ARM}_e029_selfcausal
.venv/bin/python $INFER || { echo "自分の val の推論に失敗"; exit 4; }
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo "D21_CONTROL_FT_DONE arm=$ARM $(date)"
