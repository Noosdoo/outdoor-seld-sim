#!/bin/bash
# 【開発用】因果 ft ＋ 保存 ckpt 全部の推論（v2final_dev_causal.sh の改訂 r2・2026-10-02）。
# r2 で足したこと（低優先で横取りされて再開するときの穴・job 5299 で発生）:
#   1. 学習の完了判定を「epoch_149.ckpt がある」から「last.ckpt の epoch が 149 以上」に変える。every_n_epochs=20 では epoch_149.ckpt は作られず
#      last.ckpt が e149 の実体なので、旧判定では再開のたびに trainer を呼び直して last.ckpt が再保存され（中身は同じでも sha が変わる）、
#      e149 の推論フォルダの来歴（ckpt_sha256）と食い違って `run identity changed` で止まっていた。
#   2. 推論の前に出力フォルダを確かめる: done.json があり ckpt の sha が一致すれば飛ばす（再開を速く）、無い・一致しないなら消して作り直す
#      （途中で切れた推論の残骸・古い ckpt の来歴を引き継がない）。
# 中身はそれ以外 v2final_dev_causal.sh と同じ（起点 = 基準学習の ckpt / KMIN40 / REPEAT2 / lr1e-4・150ep / 20ep 毎に保存 / 高さ入力 / DEV_* の上書き）。
# 起点の ckpt（2026-10-12 注記の訂正・Opus 10/9 00:25 の確認）: 基準学習は検証の成績が最良の回だけを epoch_NNN.ckpt として残す設定なので、
#   下の epoch_099.ckpt は通常は無く、代わりの分岐で「残っている epoch_*.ckpt の最初」＝最良の回から始まる（v22 = epoch_094・dev3 = 089・dev3s1 = 084・dev2s2 = 084）。意図どおり。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
ROOTDIR=$PS/datasets_v22          # v22（2026-10-08・9 クラス）
HDF5DIR=${DEV_HDF5DIR:-$PS/_hdf5_v22}
SEED_ARG=""; [ -n "${DEV_SEED:-}" ] && SEED_ARG="seed=$DEV_SEED"
[ -n "${DEV_HEIGHT_SEED:-}" ] && export HEIGHT_SEED=$DEV_HEIGHT_SEED
[ -d "$HDF5DIR" ] || { echo "HDF5 ルートが無い: $HDF5DIR"; exit 1; }
DEV_TAG=${DEV_TAG:-dev}
EXP=outdoor_siren_v2final_${DEV_TAG}_causal_ft
BASE=outdoor_siren_v2final_${DEV_TAG}_sde_w3
PARAMS=$E2E/md/design/v2final_params.json
MARK=$E2E/out/dataset_outdoor_siren_v2final/PROVISIONAL.txt
[ -f "$MARK" ] || { echo "PROVISIONAL.txt が無い — 開発用台本は仮の版にだけ使う"; exit 3; }
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
CK=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_099.ckpt
[ -f "$CK" ] || CK=$(ls $HOME/PSELDNets_logs/outdoor_siren_v12/runs/$BASE/checkpoints/epoch_*.ckpt | head -1)
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=${DEV_CAUSAL_SEED:-0}
export HEIGHT_COND=1
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export HEIGHT_JITTER=$JIT
export HEIGHT_DEFAULT=1.5
export HEIGHT_OFFSET=0
CFG=configs/experiment/outdoor_siren_v12_hp_clr1e-4e150.yaml
[ -f "$CFG" ] || .venv/bin/python $E2E/server_sde/hp_make_configs.py clr1e-4e150
grep -n "lr\|max_epochs\|step_size" $CFG
cp $E2E/server_sde/height_patch.py $PS/height_patch.py
cp $E2E/server_sde/causal_patch.py $PS/causal_patch.py
cp $E2E/server_sde/_train_causal17.py $PS/_train_causal17.py
JOBDIR=$PS/_job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
INFER=$JOBDIR/_causal_infer_v17.py
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
sha256sum $CK $PS/height_patch.py $PS/causal_patch.py $PS/_train_causal17.py $INFER $JOBDIR/_prediction_integrity.py $CFG $PARAMS $MARK | tee $E2E/v2final_dev_causal_sha256_job${SLURM_JOB_ID}.txt

CKDIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints
LAST=$CKDIR/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
# r2-1: last.ckpt に記録された epoch で完了を判定（epoch_149.ckpt は every_n_epochs=20 では作られない）
DONE_EP=-1
if [ -f "$LAST" ]; then
  DONE_EP=$(.venv/bin/python - "$LAST" <<'EOF'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(d.get("epoch", -1)))
EOF
)
fi
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new} exp=$EXP hdf5=$HDF5DIR seed=${DEV_SEED:-default} causal_seed=$CAUSAL_SEED height_seed=${HEIGHT_SEED:-default} last_epoch=$DONE_EP restart=${SLURM_RESTART_COUNT:-0}"

if [ -f "$CKDIR/epoch_149.ckpt" ] || [ "$DONE_EP" -ge 149 ]; then
  echo "因果 ft は完了済み（last.ckpt の epoch=$DONE_EP）。学習は呼ばない"
else
.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $SEED_ARG $RESUME
fi

export PS
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=val_syn_
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
# r2-3（Astra 再監査 10/6 RF02・旧 r2-2 を置き換え）: 推論フォルダの再利用は done.json の存在だけで決めない。
#   csv がある・done.json の sha256（csv のハッシュ）と一致・ckpt の sha が一致・n_clips が期待の本数と一致 のときだけ飛ばす（0 を返す）。
#   done.json があるのに合わない（別の ckpt の成果物・途中までの csv）→ 消さずに止める（終了値 6・手で確かめてから消す）。
#   done.json が無い（未完）→ 消して作り直す（1 を返す）。
prep_out() {   # $1 = フォルダ名, $2 = ckpt の sha256, $3 = 期待する本数（空なら本数は見ない）
  local d=$R/$1
  if [ -f "$d/val_all_causal.done.json" ]; then
    if $PS/.venv/bin/python - "$d" "$2" "${3:-}" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, ck, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
done = json.loads((d / "val_all_causal.done.json").read_text(encoding="utf-8"))
csv = d / "val_all_causal.csv"
ok = (csv.is_file() and csv.stat().st_size > 0 and done.get("ckpt_sha256") == ck
      and (not n or done.get("n_clips") == int(n))
      and done.get("sha256") == hashlib.sha256(csv.read_bytes()).hexdigest())
print(f"done.json: ckpt一致={done.get('ckpt_sha256') == ck} n_clips={done.get('n_clips')} 期待={n or '-'} csv={'あり' if csv.is_file() else '無い'} sha一致={csv.is_file() and done.get('sha256') == hashlib.sha256(csv.read_bytes()).hexdigest()}")
sys.exit(0 if ok else 1)
PY
    then echo "推論済み（ckpt sha・csv sha・本数が一致）: $1"; return 0; fi
    echo "停止: $1 に done.json があるが ckpt／csv／本数が合わない（別の走りの成果物か壊れた成果物。手で確かめてから消す）"; exit 6
  fi
  if [ -d "$d" ]; then echo "作り直し（未完・done.json なし）: $1"; rm -rf "$d"; fi
  return 1
}
N12=$(( $(wc -l < $E2E/out/_eval_aux_v12/manifest_val_syn.csv) - 1 ))        # 交差 val の本数（1,800）
NSELF=$(ls $ROOTDIR/outdoor_siren_v12/foa/val_syn_*.flac 2>/dev/null | wc -l)   # 自分の val の本数 = 推論が読む foa の val_syn_*（いまの束で 5,100）
[ "${NSELF:-0}" -gt 0 ] || NSELF=""                                           # 数えられなければ本数は見ない
# Astra A6（2026-09-16）: epoch_149.ckpt と last.ckpt を同じ出力名に入れない。last.ckpt は epoch_149.ckpt が無いときだけ e149 として使う
CKS=$(ls $CKDIR/epoch_*.ckpt | grep -v -- "-v"); [ -f "$CKDIR/epoch_149.ckpt" ] || CKS="$CKS $CKDIR/last.ckpt"
for ck in $CKS; do
  e=$(basename $ck .ckpt); [ "$e" = "last" ] && e=epoch_149
  CKSHA=$(sha256sum $ck | cut -c1-64)
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  # v22（9 クラス）: v12 の val 1,800 本には「9 行の cls_indices ＋ v12 の音声へのリンク」の置き場を指す（推論はクラス数を置き場の一覧から取る・2026-10-08）
  V12ROOT=$PS/datasets_v22v12; mkdir -p $V12ROOT; cp -n $ROOTDIR/cls_indices_train.tsv $V12ROOT/cls_indices_train.tsv
  [ "$(wc -l < $V12ROOT/cls_indices_train.tsv)" = 9 ] || { echo "datasets_v22v12 の一覧が 9 行でない"; exit 1; }
  [ -e $V12ROOT/outdoor_siren_v12 ] || ln -s $PS/datasets_v12/outdoor_siren_v12 $V12ROOT/outdoor_siren_v12
  export CAUSAL_DSDIR=$V12ROOT
  export CAUSAL_OUT=infer_v2final${DEV_TAG}ft_${e/epoch_/e}_v12causal
  unset CAUSAL_GLOB
  export HEIGHT_TABLE=""      # v12 val は全クリップ 1.5 m（表を外す。v17 の教訓）
  prep_out $CAUSAL_OUT $CKSHA $N12 || .venv/bin/python $INFER
  export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=infer_v2final${DEV_TAG}ft_${e/epoch_/e}_selfcausal
  prep_out $CAUSAL_OUT $CKSHA $NSELF || .venv/bin/python $INFER
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
echo V2FINAL_DEV_CAUSAL_DONE
