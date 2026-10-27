#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -o v2fabl-%j.out
#SBATCH --open-mode=append
# アブレーション（#74）の学習／因果 ft を「pro_6000 と a100 の空いたほう」で走らせる包み（本人 2026-10-21 09:49「両候補で」）。
#   同じ段（TAG × STAGE）を 2 つの partition に 1 本ずつ投入し、先に走り始めたほうが鍵（mkdir・原子的）を取って本体を実行、相方を取り消す。
#   後から走り始めたほうは鍵が取れないので何もせず 0 で終わる（因果 ft の依存は afterany なので止まらない）。
#   割り込まれて再投入（--requeue）されたときは、鍵の中の job id が自分なので続けて本体を実行（本体の再開の決まりはそのまま）。
#   鍵の持ち主が列に無い（落ちた）ときは鍵を引き継いで本体を実行する。
# 使い方（submit_abl_either.sh が投入する）:
#   EITHER_TAG=scratch|noheight EITHER_STAGE=train|causal sbatch -p pro_6000 --gres=gpu:pro_6000:1 --qos=low --requeue -J v2f<段>_abl_<札>_pro  server_sde/v2final_abl_either.sbatch
#   EITHER_TAG=...              EITHER_STAGE=...          sbatch -p a100     --gres=gpu:a100:1     --qos=low --requeue -J v2f<段>_abl_<札>_a100 server_sde/v2final_abl_either.sbatch
#   本体 = server_sde/v2final_train.sbatch（train）／server_sde/v2final_causal.sbatch（causal）を V2FINAL_ABL_TAG=<札> で exec（中の #SBATCH 行は効かない・ログはこの包みの v2fabl-<id>.out）
# 検査用: EITHER_DRY=1（本体を exec せず判断だけ印字）・EITHER_LOCK_ROOT（鍵の置き場の上書き）
set -u
TAG=${EITHER_TAG:?scratch か noheight}; STAGE=${EITHER_STAGE:?train か causal}
case "$TAG" in scratch|noheight) ;; *) echo "EITHER_TAG は scratch か noheight（$TAG）"; exit 3 ;; esac
case "$STAGE" in train|causal) ;; *) echo "EITHER_STAGE は train か causal（$STAGE）"; exit 3 ;; esac
E2E=$HOME/research/outdoor_seld_e2e
LOCK_ROOT=${EITHER_LOCK_ROOT:-$HOME/PSELDNets_logs/outdoor_siren_v12/runs}
LOCK=$LOCK_ROOT/_either_${TAG}_${STAGE}
ME=${SLURM_JOB_ID:-dry$$}
NAME_PRO=v2f${STAGE}_abl_${TAG}_pro; NAME_A=v2f${STAGE}_abl_${TAG}_a100
mkdir -p "$LOCK_ROOT"
if mkdir "$LOCK" 2>/dev/null; then
  echo "$ME" > "$LOCK/jobid"; echo "鍵を取った（$LOCK・job $ME・partition ${SLURM_JOB_PARTITION:-?}）"
else
  OWNER=$(cat "$LOCK/jobid" 2>/dev/null || echo "")
  OWNER_ST=$([ -n "$OWNER" ] && squeue -j "$OWNER" -h -o %t 2>/dev/null | head -n 1 || echo "")
  if [ "$OWNER" = "$ME" ]; then
    echo "再投入（自分の鍵・job $ME）→ 続ける"
  elif [ -n "$OWNER" ] && { [ "$OWNER_ST" = "R" ] || [ "$OWNER_ST" = "CG" ]; }; then
    echo "相方（job $OWNER・$OWNER_ST）が先に走っている → 何もせず終わる（0）"; exit 0
  else
    # 持ち主が列に無い（落ちた）か、待ち（割り込まれて再投入され GPU 待ち）なら引き継ぐ: 本体は last.ckpt から再開するので partition が変わっても続きから
    echo "鍵の持ち主（$OWNER・状態 ${OWNER_ST:-無し}）は走っていない → 鍵を引き継ぐ（job $ME）"; echo "$ME" > "$LOCK/jobid"
  fi
fi
# 相方（同じ札・同じ段・別の partition の名前）を取り消す（自分以外）
for J in $(squeue -u "$USER" -h -n "$NAME_PRO,$NAME_A" -o %i 2>/dev/null); do
  [ "$J" = "$ME" ] || { scancel "$J" && echo "相方 $J を取り消した"; }
done
case "$STAGE" in
  train)  BODY=$E2E/server_sde/v2final_train.sbatch ;;
  causal) BODY=$E2E/server_sde/v2final_causal.sbatch ;;
esac
[ -f "$BODY" ] || { echo "本体が無い（$BODY）"; exit 3; }
echo "本体: $BODY（V2FINAL_ABL_TAG=$TAG・GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1)）"
if [ "${EITHER_DRY:-0}" = 1 ]; then echo "EITHER_DRY_OK $TAG $STAGE"; exit 0; fi
export V2FINAL_ABL_TAG=$TAG
exec bash "$BODY"
