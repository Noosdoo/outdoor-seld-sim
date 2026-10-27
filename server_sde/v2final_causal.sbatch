#!/bin/bash
#SBATCH -J v2fcausal
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --qos=normal
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o v2fcausal-%j.out
# 確定評価 v2 の因果 ft = v17b と同一（起点 = v2final 基準学習 ep099 / KMIN40 / REPEAT2 / lr1e-4・150ep・StepLR90 / 20ep 毎に全保存 / 高さ入力・ゆらぎ 0.10）。
# 保存 ckpt 全部を因果推論（v12 fold2 1.5 m ＝ 表なし、self val ＝ v2final fold2）→ 統一採点器で ckpt を選ぶ（規則 = 至近帯の距離誤差・実行計画 §4）。
# ⚠️ fold20（確定評価）はここでは推論しない。ckpt を 1 つ選んで v2final_conf.sbatch で 1 回だけ。
set -ex
PS=$HOME/research/PSELDNet/PSELDNets
E2E=$HOME/research/outdoor_seld_e2e
cd $PS
source $E2E/server_sde/_baseline_guard.sh
guard_record
DATA=${V2FINAL_DATA:-v29}                     # 2026-10-12: 学習データの置き場の札（v2final＝9/22 の束・v22＝10/8 の 9 クラス・v26＝音の直しの描き直し・v27＝組み合わせの網羅＋遠いバイク #64）。既定 v27（2026-10-13）
ROOTDIR=$PS/datasets_$DATA
HDF5DIR=$PS/_hdf5_$DATA
MARK=$ROOTDIR/outdoor_siren_v12/PROVISIONAL.txt           # 束に付く仮の版の印（AR12-05: 本番は印があれば止める・開発用は無ければ止める）
[ ! -f "$MARK" ] || { echo "束 $ROOTDIR に仮の版の印 $MARK がある → 本番の学習はしない（確定した設定で描き直してから）"; exit 3; }
$PS/.venv/bin/python $E2E/server_sde/_train_entrance_guard.py "$ROOTDIR" "$HDF5DIR" || { echo "学習の入口の門で止めた（確定評価の本・実録の置き場が学習データに混じっている）"; exit 2; }   # AR12-03・SV26-03: ROOTDIR／HDF5DIR を決めたあとに呼ぶ
EXP=outdoor_siren_v2final_${DATA}_causal_ft          # AR12-06: 実験名に学習データの札
# #71（2026-10-19）: 追加の種（s1／s2）。未設定 = 主 seed（台本の動きは変わらない）。札があるときだけ run 名・元 ckpt の run・seed・HEIGHT_SEED・CAUSAL_SEED を上書き
SEED_TAG=${V2FINAL_SEED_TAG:-}; SEED_OVR=""; SEED_SUF=""; SEED_CAUSAL=0
case "$SEED_TAG" in
  "") ;;
  s1) SEED_SUF=_s1; EXP=${EXP}_s1; SEED_OVR="seed=20261001"; export HEIGHT_SEED=20261001; SEED_CAUSAL=1 ;;
  s2) SEED_SUF=_s2; EXP=${EXP}_s2; SEED_OVR="seed=20261002"; export HEIGHT_SEED=20261002; SEED_CAUSAL=2 ;;
  *) echo "V2FINAL_SEED_TAG は s1 か s2（$SEED_TAG）"; exit 3 ;;
esac
# #74（2026-10-21）: アブレーション（scratch／noheight）。基準学習の run 名・因果 ft の run 名・推論の出力名に _abl_<tag>。noheight は HEIGHT_COND=0 で学習と推論
ABL_TAG=${V2FINAL_ABL_TAG:-}; ABL_SUF=""; ABL_HEIGHT=1
case "$ABL_TAG" in
  "") ;;
  scratch) ABL_SUF=_abl_scratch ;;
  noheight) ABL_SUF=_abl_noheight; ABL_HEIGHT=0 ;;
  *) echo "V2FINAL_ABL_TAG は scratch か noheight（$ABL_TAG）"; exit 3 ;;
esac
EXP=${EXP}${ABL_SUF}
PARAMS=$E2E/md/design/v2final_params.json
$PS/.venv/bin/python -c "import json,sys;P=json.load(open('$PARAMS'));sys.exit(0 if P.get('status')=='fixed' else 3)" || { echo "params が fixed でない"; exit 3; }
JIT=$($PS/.venv/bin/python -c "import json;print(json.load(open('$PARAMS'))['HEIGHT_JITTER_TRAIN'])")
CKD0=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_${DATA}_sde_w3${SEED_SUF}${ABL_SUF}/checkpoints
CK=$CKD0/epoch_099.ckpt
# 2026-10-20 22:1x（job 7019 の事故）: 基準学習は epoch_099.ckpt を残さず（残るのは検証最良の epoch_0NN.ckpt と last.ckpt）、旧の fallback（ls epoch_*.ckpt | head -1）が
#   黙って epoch_089 を起点にしていた。起点は凍結どおり最終 epoch 99 = last.ckpt（中の epoch を読んで 99 でなければ止める）。黙って別の epoch を使わない
if [ ! -f "$CK" ]; then
  CK=$CKD0/last.ckpt
  [ -f "$CK" ] || { echo "起点の ckpt が無い（$CKD0 に epoch_099.ckpt も last.ckpt も無い）"; exit 3; }
  CKE=$($PS/.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$CK")
  [ "$CKE" = "99" ] || { echo "last.ckpt の epoch が 99 でない（$CKE）。起点は最終 epoch 99 の決まり → 止める"; exit 3; }
  echo "起点 = last.ckpt（epoch $CKE）"
fi
ls -l $CK
export SDE_W_DIST=3.0
export SDE_DIST_MODE=log
export CAUSAL_KMIN=40
export CAUSAL_REPEAT=2
export CAUSAL_SEED=$SEED_CAUSAL   # #71: 主 seed は 0・s1 は 1・s2 は 2
export HEIGHT_COND=$ABL_HEIGHT    # #74: noheight は 0（学習も推論も高さの入力なし）
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
# 2026-10-22 01:1x（job 8965 の教訓・監査 A3/R1 の決まり = 推論の import 先はジョブ専用 _job<ID>/）: 推論の 3 本（_causal_infer_v17・_prediction_integrity・hysteresis_patch）は
#   $PS 直下でなく $JOBDIR に写して、そこから走らせる。$PS/_prediction_integrity.py は 9/9 の古い写し（8 クラス・logical_code_name 無し）で、ここから import すると落ちる
JOBDIR=$PS/_job${SLURM_JOB_ID:-dry$$}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/_causal_infer_v17.py
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/_prediction_integrity.py
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/hysteresis_patch.py
sha256sum $CK $PS/height_patch.py $PS/causal_patch.py $PS/_train_causal17.py $JOBDIR/_causal_infer_v17.py $JOBDIR/_prediction_integrity.py $JOBDIR/hysteresis_patch.py $CFG $PARAMS | tee $E2E/v2final_causal_sha256_job${SLURM_JOB_ID}.txt

LAST=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints/last.ckpt
RESUME=""
[ -f "$LAST" ] && RESUME="ckpt_path=$LAST"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi -L | head -1) resume=${RESUME:-new}"

.venv/bin/python $PS/_train_causal17.py \
  experiment=outdoor_siren_v12_hp_clr1e-4e150 \
  model.kwargs.pretrained_path=$CK \
  experiment_name=$EXP \
  paths.log_dir=$HOME/PSELDNets_logs \
  paths.dataset_dir=$ROOTDIR paths.hdf5_dir=$HDF5DIR \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  $SEED_OVR \
  $RESUME

export PS
export HEIGHT_JITTER=0
export CAUSAL_PREFIX=val_syn_          # 2026-09-22 改名（旧 fold2_）
# Astra RN20-02（2026-10-20）: 推論の出力名にも種の札（SEED_SUF）を入れる（s1／s2 が主 seed の置き場に入らない）。主 seed は空 = 従来の名前のまま。
#   選択器の台本（run_selector_v29.sh）はこの名前の決まり（infer_v2finalft${SEED_SUF}_e<NNN>_selfcausal）で写す
OUTPFX=infer_v2finalft${SEED_SUF}${ABL_SUF}
export CAUSAL_LIMIT=0
N_VAL=$(($(wc -l < $E2E/out/dataset_outdoor_siren_v2final/plan/manifest_val_syn_v2final.csv) - 1))   # 選択器の名簿と同じ本数（v29 = 12,420）
# 検出のヒステリシス（変更票 #34・2026-09-16 本人「①採用」）: 入り 0.7／保持 0.3。両設定で共通（scripts/_notify_modes.py と同じ値）。run_meta の settings に記録される
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
CKD=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/$EXP/checkpoints
# Astra A6（2026-09-16）: epoch_149.ckpt と last.ckpt を同じ出力名に入れない。last.ckpt は epoch_149.ckpt が無いときだけ e149 として使う
CKS=$(ls $CKD/epoch_*.ckpt | grep -v -- "-v"); [ -f "$CKD/epoch_149.ckpt" ] || CKS="$CKS $CKD/last.ckpt"
# 2026-10-22 00:14（job 7972 の教訓）: 旧 v12 val（8 クラス・1,800 本・datasets_v12）への推論「_v12causal」は外した。9 クラスのモデル（#48 自転車）では
#   出力層の形（108 ≠ 96）が合わず最初の ckpt で落ち、選択器に要る _selfcausal が 1 つも出なかった。選択器 2.1（run_selector_v29.sh）は _selfcausal だけを使う。
#   last.ckpt を e149 として使うときは epoch が 149 であることを確かめる（7019 の起点の事故と同じ型を防ぐ）
for ck in $CKS; do
  e=$(basename $ck .ckpt)
  if [ "$e" = "last" ]; then
    CKE=$(.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch'))" "$ck")
    [ "$CKE" = "149" ] || { echo "last.ckpt の epoch が 149 でない（$CKE）→ e149 として使えない。止める"; exit 3; }
    e=epoch_149
  fi
  export CAUSAL_CKPT=$ck
  export HEIGHT_OFFSET=0
  unset CAUSAL_GLOB
  export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
  [ "$ABL_HEIGHT" = 1 ] || export HEIGHT_TABLE=""      # #74 noheight: 高さの表も渡さない（HEIGHT_COND=0 なので読まれないが明示）
  export CAUSAL_DSDIR=$ROOTDIR
  export CAUSAL_OUT=${OUTPFX}_${e/epoch_/e}_selfcausal
  export CAUSAL_EXPECTED_CLIPS=$N_VAL
  .venv/bin/python $JOBDIR/_causal_infer_v17.py
done
export HEIGHT_OFFSET=0
guard_verify || exit 1
rm -rf "$JOBDIR"
echo V2FINAL_CAUSAL_DONE
