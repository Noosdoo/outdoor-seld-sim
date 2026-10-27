#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v20inf-%j.out
# v20 の因果推論を別 GPU で分担する補助ジョブ（2026-09-09 本人「並列で」）。本体（v20_all.sbatch の causal 段階）と同じ推論スクリプト・同じ置き場
# （verified_v2・来歴つき）に書く。done.json のある run は飛ばし、本体が書いている最中（排他ロック）の run は後回しにする。出力は誰が作っても同一。
# 投入例: sbatch -J v20a_inf --export=ALL,NAME=v20a,TASKS="v18:e149 v18:e139 v18:e119 v18:e099 self:e149 v12:e149" server_sde/v20_infer_helper.sbatch
set -x
set -o pipefail   # Astra V2（2026-09-09）: `python | tee` の成否は python のもの。無いと tee の 0 で失敗が「done」になる
NAME=${NAME:?}; TASKS=${TASKS:?}
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
ROOTDIR=$PS/datasets_$NAME
TABLE=$E2E/out/dataset_outdoor_siren_$NAME/plan/assignment_${NAME}_all.csv
OWN=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_${NAME}_causal_ft/checkpoints
V18=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v18_causal_ft/checkpoints
export PS E2E
export SDE_W_DIST=3.0 SDE_DIST_MODE=log
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export CAUSAL_PREFIX=fold2_
unset CAUSAL_GLOB CAUSAL_LIMIT CAUSAL_FOA
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/verified_v2
RUNS=$CAUSAL_RUNS_DIR
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR          # ジョブ専用ディレクトリ（監査 A7: import 先まで専用化）
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
INFER=$JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) name=$NAME tasks=[$TASKS]"
ck_epoch() { .venv/bin/python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']))" "$1"; }
ckpt_of() {   # ckpt_of <own|v18> <eNNN>
  local dir=$OWN; [ "$1" = v18 ] && dir=$V18
  local n=${2#e}; local ck=$dir/epoch_${n}.ckpt        # 敵対レビュー E3: 実ファイルは epoch_099.ckpt（3 桁）。$((10#$n)) だと epoch_99 になり見つからなかった
  # v18 の e149 だけ last.ckpt に落とす（本体と同じ経路）。自分の e149 は本体が固定する epoch_149.ckpt ができるまで「まだ無い」扱い（E1: last.ckpt は横取り再開で書き直される）
  [ -f "$ck" ] || { [ "$1" = v18 ] && [ "$n" = "149" ] && ck=$dir/last.ckpt; }
  [ -f "$ck" ] && echo "$ck"
}
remaining="$TASKS"
for round in $(seq 1 12); do
  next=""
  for t in $remaining; do
    kind=${t%%:*}; e=${t##*:}
    case $kind in
      v18)  ck=$(ckpt_of v18 $e); out=infer_v18ft_${e}_${NAME}causal; dsdir=$ROOTDIR; table=$TABLE ;;
      self) ck=$(ckpt_of own $e); out=infer_${NAME}ft_${e}_selfcausal; dsdir=$ROOTDIR; table=$TABLE ;;
      v12)  ck=$(ckpt_of own $e); out=infer_${NAME}ft_${e}_v12causal; dsdir=$PS/datasets_v12; table="" ;;
      *) echo "unknown task $t"; continue ;;
    esac
    if [ -f "$RUNS/$out/val_all_causal.done.json" ]; then echo "済み: $out"; continue; fi
    if [ -z "$ck" ]; then echo "ckpt がまだ無い: $t → 後回し"; next="$next $t"; continue; fi
    ep=$(ck_epoch $ck); [ "$ep" = "$((10#${e#e}))" ] || { echo "epoch 不一致 $ck → $ep（期待 $e）"; next="$next $t"; continue; }
    export CAUSAL_CKPT=$ck CAUSAL_OUT=$out CAUSAL_DSDIR=$dsdir HEIGHT_TABLE=$table
    log=$E2E/out/_infer_helper_${SLURM_JOB_ID}.log
    if .venv/bin/python $INFER 2>&1 | tee $log; then echo "done: $out"; else
      if grep -q "run is in use" $log; then echo "本体が書き込み中: $out → 後回し"; next="$next $t"; else echo "FAILED: $out"; next="$next $t"; fi
    fi
  done
  remaining=$(echo $next)
  [ -z "$remaining" ] && break
  echo "round $round 残り: [$remaining] → 300 s 待つ"; sleep 300
done
[ -z "$remaining" ] && echo V20_HELPER_DONE_$NAME || { echo "V20_HELPER_INCOMPLETE_$NAME 残り: [$remaining]"; exit 1; }
