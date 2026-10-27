#!/bin/bash
# v20 因果 ft = v17b/v18/v19 と同一（起点 = v20 基準 ep099 / KMIN40 / REPEAT2 / lr1e-4・150ep / 20ep 毎に全保存 / 高さ入力・ゆらぎ 0.10）。2026-09-09。
# 推論（来歴つき・verified_v2・ジョブ固有の写し）: 保存 ckpt 全部を v12 fold2 → infer_${NAME}ft_e<NNN>_v12causal、
#   束の fold2（v18 fold2 3,600 ＋ 新場面 fold2 300 × 場面数）→ infer_${NAME}ft_e<NNN>_selfcausal。
#   比較用に v18 の e099/e119/e139/e149 を同じ束で推論 → infer_v18ft_e<NNN>_${NAME}causal。4 epoch そろわなければ完了印を作らない。
set -ex
NAME=${NAME:?}
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_$NAME
HDF5DIR=$PS/_hdf5_$NAME
EXP=outdoor_siren_${NAME}_causal_ft
BASE=outdoor_siren_${NAME}_sde_w3
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
TABLE=$E2E/out/dataset_outdoor_siren_$NAME/plan/assignment_${NAME}_all.csv
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
# ジョブ専用ディレクトリに推論の入口と来歴モジュールをそろえる（監査 A7: import 先まで専用化。A2: 共有パスの差し替え混入の再発防止）
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
INFER=$JOBDIR/_causal_infer_v17.py
CKDIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints
LAST=$CKDIR/last.ckpt
E149=$CKDIR/epoch_149.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} name=$NAME"
ck_epoch() { .venv/bin/python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']))" "$1"; }
# 敵対レビュー E1（2026-09-09）: 学習完了後に横取り→再開すると _train_causal17.py が last.ckpt を書き直し、e149 の来歴（ckpt_sha256）が変わって
# 推論が「run identity changed」で止まる。→ 学習が済んでいれば（epoch_149.ckpt がある）学習は呼ばず、e149 は last.ckpt の固定した写し epoch_149.ckpt で推論する
if [ -f "$E149" ]; then
  echo "学習は済み（$E149 あり）→ 学習を飛ばす"
else
  .venv/bin/python $PS/_train_causal17.py \
    experiment=outdoor_siren_v12_hp_clr1e-4e150 \
    model.kwargs.pretrained_path=$CK \
    experiment_name=$EXP \
    paths.log_dir=$HOME/PSELDNets_logs \
    paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
    callbacks.model_checkpoint.save_top_k=-1 \
    callbacks.model_checkpoint.every_n_epochs=20 \
    $RESUME
  [ -f "$LAST" ] || { echo "last.ckpt が無い"; exit 5; }
  ep=$(ck_epoch $LAST); [ "$ep" = 149 ] || { echo "学習が最後まで行っていない: last.ckpt = epoch $ep"; exit 5; }
  cp $LAST $E149.tmp && mv $E149.tmp $E149     # 原子的に固定（以後 last.ckpt が書き直されても e149 の来歴は変わらない）
fi
export PS
export E2E
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=fold2_
unset CAUSAL_GLOB
set -o pipefail
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2
RUNS=$CAUSAL_RUNS_DIR
EXP_SELF=$(.venv/bin/python -c "import csv,sys; print(sum(1 for r in csv.DictReader(open(sys.argv[1], encoding='utf-8')) if r.get('split') == 'fold2'))" "$TABLE")
[ "$EXP_SELF" -gt 3600 ] || { echo "束の fold2 本数が変: $EXP_SELF"; exit 5; }
infer_retry() {   # 出力先を別ジョブが書いている間は排他ロックで拒否される → 10 分おきに最大 12 回待つ。完了済み（done.json が照合を通る）なら飛ばす
  local n=0 log=$E2E/out/_infer_retry_${SLURM_JOB_ID}.log expected=$1 run=$RUNS/$CAUSAL_OUT
  # 敵対レビュー E1: 途中で止まった run（run_meta.json はあるが done.json が無い）の ckpt の経路が今の ckpt と違えば、退避してから作り直す（横取り再開で last.ckpt → epoch_149.ckpt に変わった場合）
  if [ -f "$run/run_meta.json" ] && [ ! -f "$run/val_all_causal.done.json" ]; then
    if ! .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['ckpt'] == sys.argv[2] else 1)" "$run/run_meta.json" "$CAUSAL_CKPT"; then
      mv "$run" "${run}.stale_${SLURM_JOB_ID}" && echo "途中の run の ckpt 経路が違う → 退避: ${run}.stale_${SLURM_JOB_ID}"
    fi
  fi
  while :; do
    # 監査 R1/A3・Astra V2: 待った後も毎回、完了印を先に見る。印があるだけでは信用せず来歴の照合（本数・行数・sha256）を通ったときだけ飛ばす
    if [ -f "$run/val_all_causal.done.json" ] && .venv/bin/python $JOBDIR/_prediction_integrity.py "$run" "$expected"; then echo "済み（照合済）: $CAUSAL_OUT"; return 0; fi
    if .venv/bin/python $INFER 2>&1 | tee $log; then return 0; fi
    if grep -q "run is in use" $log && [ $n -lt 12 ]; then n=$((n + 1)); echo "出力先が使用中 → 600 s 待って再試行 ($n)"; sleep 600; continue; fi
    return 1
  done
}
done_epochs=""
for ck in $(ls $CKDIR/epoch_*.ckpt | grep -v -- "-v"); do   # E1: last.ckpt は使わない（e149 は固定した epoch_149.ckpt）
  ep=$(ck_epoch $ck); e=$(printf 'e%03d' $ep)
  case " $done_epochs " in *" $e "*) echo "skip $ck (epoch $ep は済み)"; continue;; esac
  case $e in e099|e119|e139|e149) ;; *) echo "skip $ck（宣言で使う 4 epoch 以外は推論しない・2026-09-09 10:50 本人「もっと速く」）"; continue;; esac
  done_epochs="$done_epochs $e"
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  export CAUSAL_DSDIR=$PS/datasets_v12
  export CAUSAL_OUT=infer_${NAME}ft_${e}_v12causal
  export HEIGHT_TABLE=""
  infer_retry 1800 || exit 5
  export HEIGHT_TABLE=$TABLE
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_${NAME}ft_${e}_selfcausal
  infer_retry $EXP_SELF || exit 5
done
for e in e099 e119 e139 e149; do
  for s in v12causal selfcausal; do
    [ -f $RUNS/infer_${NAME}ft_${e}_${s}/val_all_causal.done.json ] || { echo "自分の推論が欠けている: ${e}_${s}"; exit 5; }
  done
done
V18=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v18_causal_ft/checkpoints
for e in 099 119 139 149; do
  ck=$V18/epoch_$e.ckpt
  [ -f "$ck" ] || { [ "$e" = "149" ] && ck=$V18/last.ckpt; }
  [ -f "$ck" ] || { echo "v18 の ckpt が無い: epoch_$e"; exit 5; }
  ep=$(ck_epoch $ck); [ "$ep" = "$((10#$e))" ] || { echo "v18 ckpt の epoch が違う: $ck → $ep（期待 $e）"; exit 5; }
  export CAUSAL_CKPT=$ck
  export HEIGHT_TABLE=$TABLE
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v18ft_e${e}_${NAME}causal
  infer_retry $EXP_SELF || { echo "比較推論に失敗: $CAUSAL_OUT"; exit 5; }
  [ -f $RUNS/$CAUSAL_OUT/val_all_causal.done.json ] || { echo "比較推論が完了していない: e$e"; exit 5; }
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo V20_CAUSAL_DONE_$NAME
