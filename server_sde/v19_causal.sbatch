#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v19causal-%j.out
# v19 因果 ft = v17b/v18 と同一（起点 = v19 基準 ep099 / KMIN40 / REPEAT2 / lr1e-4・150ep / 20ep 毎に全保存 / 高さ入力・ゆらぎ 0.10）。
# 投入例: sbatch -J v19fastcausal --export=ALL,SCENE=fast --dependency=afterok:<train job> server_sde/v19_causal.sbatch
# 推論: 保存 ckpt 全部を v12 fold2（1.5 m・表なし）→ infer_v19${SCENE}ft_e<NNN>_v12causal、
#       束の fold2（v18 fold2 3,600 ＋ 新場面 fold2 300）→ infer_v19${SCENE}ft_e<NNN>_selfcausal。
#       比較用に v18 の因果 ft ckpt（e139 と last）も束の fold2 で推論 → infer_v18ft_e<NNN>_v19${SCENE}causal
set -ex
SCENE=${SCENE:?SCENE=fast|static|turn|platoon}
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v19_$SCENE
HDF5DIR=$PS/_hdf5_v19_$SCENE
EXP=outdoor_siren_v19${SCENE}_causal_ft
BASE=outdoor_siren_v19${SCENE}_sde_w3
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
TABLE=$E2E/out/dataset_outdoor_siren_v19_$SCENE/plan/assignment_v19_${SCENE}_all.csv
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=0
export HEIGHT_COND=1
export HEIGHT_TABLE=$TABLE
export HEIGHT_JITTER=0.10
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py clr1e-4e150
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
# 推論台本はジョブごとの名前で固定する（共有の $PS/_causal_infer_v17.py は別ジョブの段階開始で差し替わり、
# 実行中の推論に混入した: 2026-09-08 14:11 → 4037 の e139 が新版で落ちた）
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR          # ジョブ専用ディレクトリ（監査 A7: import 先まで専用化・2026-09-09）
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
INFER=$JOBDIR/_causal_infer_v17.py
LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} scene=$SCENE"
.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $RESUME
export PS
export E2E
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=fold2_
unset CAUSAL_GLOB
set -o pipefail
# 監査対応（Astra の来歴設計・2026-09-08 14:20）: 推論は来歴つきの新しい置き場 verified_v2 に書く（旧置き場の出力は認証しない）
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2
RUNS=$CAUSAL_RUNS_DIR
ck_epoch() { .venv/bin/python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']))" "$1"; }
# 監査 F5: epoch 名は ckpt の中身から読む（last.ckpt を e149 と決め打ちしない）。同じ epoch の重複（last = epoch_149）は飛ばす
done_epochs=""
for ck in $(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/epoch_*.ckpt | grep -v -- "-v") $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt; do
  ep=$(ck_epoch $ck); e=$(printf 'e%03d' $ep)
  case " $done_epochs " in *" $e "*) echo "skip $ck (epoch $ep は済み)"; continue;; esac
  done_epochs="$done_epochs $e"
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_v19${SCENE}ft_${e}_v12causal
  export HEIGHT_TABLE=""
  .venv/bin/python $INFER
  export HEIGHT_TABLE=$TABLE
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v19${SCENE}ft_${e}_selfcausal
  .venv/bin/python $INFER
done
for e in e099 e119 e139 e149; do
  for s in v12causal selfcausal; do
    [ -f $RUNS/infer_v19${SCENE}ft_${e}_${s}/val_all_causal.done.json ] || { echo "自分の推論が欠けている: ${e}_${s}"; exit 5; }
  done
done
# 比較用: v18 の因果 ft（e099/e119/e139/e149・宣言 §3「同じ epoch どうし」）を同じ束の fold2 で推論。
# 監査 F5: v18 の学習完了（V18_CAUSAL_DONE）を待ち、ckpt の epoch を読んで名前と一致させる。欠けていれば完了にしない
# v18 の学習完了は ckpt で判定する（V18_CAUSAL_DONE は 4037 が推論で落ちたため出ない。学習 150 epoch と ckpt は完了済み）
V18=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v18_causal_ft/checkpoints
waited=0
until [ -f $V18/epoch_139.ckpt ] && [ -f $V18/last.ckpt ] && [ "$(ck_epoch $V18/last.ckpt 2>/dev/null)" = "149" ]; do
  [ $waited -ge 43200 ] && { echo "v18 の ckpt（epoch_139 と last=149）を 12 時間待っても来ない → 比較なしでは完了にしない"; exit 5; }
  echo "v18 の ckpt 待ち ($waited s)"; sleep 600; waited=$((waited + 600))
done
infer_retry() {   # 同じ出力先を別ジョブ（reinfer_v18 など）が書いている間は排他ロックで拒否される → 10 分おきに最大 12 回待つ
  local n=0 log=$E2E/out/_infer_retry_${SLURM_JOB_ID}.log
  while :; do
    if [ -f "$RUNS/$CAUSAL_OUT/val_all_causal.done.json" ]; then echo "済み: $CAUSAL_OUT（別ジョブが仕上げた・来歴は取得時に照合）"; return 0; fi   # 監査 R1/A3
    if .venv/bin/python $INFER 2>&1 | tee $log; then return 0; fi
    if grep -q "run is in use" $log && [ $n -lt 12 ]; then n=$((n + 1)); echo "出力先が使用中 → 600 s 待って再試行 ($n)"; sleep 600; continue; fi
    return 1
  done
}
for e in 099 119 139 149; do
  ck=$V18/epoch_$e.ckpt
  [ -f "$ck" ] || { [ "$e" = "149" ] && ck=$V18/last.ckpt; }
  [ -f "$ck" ] || { echo "v18 の ckpt が無い: epoch_$e"; exit 5; }
  ep=$(ck_epoch $ck); [ "$ep" = "$((10#$e))" ] || { echo "v18 ckpt の epoch が違う: $ck → $ep（期待 $e）"; exit 5; }
  export CAUSAL_CKPT=$ck
  export HEIGHT_TABLE=$TABLE
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v18ft_e${e}_v19${SCENE}causal
  if [ -f $RUNS/$CAUSAL_OUT/val_all_causal.done.json ]; then
    echo "比較推論 $CAUSAL_OUT は完了済み（来歴は取得時に照合）"
  else
    infer_retry || { echo "比較推論に失敗: $CAUSAL_OUT"; exit 5; }
  fi
  [ -f $RUNS/infer_v18ft_e${e}_v19${SCENE}causal/val_all_causal.done.json ] || { echo "比較推論が完了していない: e$e"; exit 5; }
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo V19_CAUSAL_DONE_$SCENE
