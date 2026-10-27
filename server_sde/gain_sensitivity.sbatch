#!/bin/bash
#SBATCH -J gainsens
#SBATCH -p pro_6000
#SBATCH --gres=gpu:pro_6000:1
#SBATCH -t 04:00:00
#SBATCH -o gainsens-%j.out
# 較正の感度測定（実録後リスト 3b・敵対レビュー C の C06 の裏取り・2026-09-27 本人「サーバつなげときました」）。
# 騒音計 CENTER392 の精度 ±1.4 dB（クラス 2）が装置の出力にどれだけ効くかを、自分の val（4,800 本）の入力音量を ±2 dB
# ずらして dev2 の因果 ft（last.ckpt = e149）で推論し直し、統一採点器と知覚層の評価で 0 dB と比べる。
# 推論のコード・既存の推論結果・データセットには触れない（音は別フォルダに写し、CAUSAL_FOA で差し替える）。
# 投入: sbatch server_sde/gain_sensitivity.sbatch
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
DEV_TAG=dev2
CK=$R/outdoor_siren_v2final_${DEV_TAG}_causal_ft/checkpoints/last.ckpt     # 9/23 の e149 selfcausal と同じ ckpt（run_meta.ckpt）
[ -f "$CK" ] || { echo "ckpt が無い: $CK"; exit 2; }
ROOTDIR=$PS/datasets_v2final
FOA0=$ROOTDIR/outdoor_siren_v12/foa
OUT=$E2E/out/gain_sensitivity_2026-09-27; mkdir -p $OUT
JOBDIR=$OUT/job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
SCALED=$PS/datasets_v2final_gain          # 音の写し（既存のデータセットの外）
echo "job=$SLURM_JOB_ID node=$(hostname) $(date)"
sha256sum $CK $INFER $JOBDIR/_prediction_integrity.py $JOBDIR/hysteresis_patch.py $PS/height_patch.py $PS/sde_patch.py $E2E/server_sde/_gain_scale_foa.py | tee $OUT/sha256_job${SLURM_JOB_ID}.txt

# 1) 音を ±2 dB して写す（CPU・4,800 本 × 2）
for g in -2 +2; do
  tag=$(echo "$g" | sed 's/+/p/; s/-/m/')
  $PY $E2E/server_sde/_gain_scale_foa.py --src $FOA0 --dst $SCALED/g${tag}/foa --gain-db $g --glob 'val_syn_*.flac' --workers 4 || exit 3
done
cp $SCALED/scale_report_*.json $OUT/ 2>/dev/null || true

# 2) 推論（9/23 の selfcausal と同じ設定。v2final_dev_causal.sh の値をそのまま）
cd $PS
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
export CAUSAL_PREFIX=val_syn_ CAUSAL_DSDIR=$ROOTDIR CAUSAL_CKPT=$CK CAUSAL_EXPECTED_CLIPS=4800
unset CAUSAL_GLOB
for tag in gm2 gp2; do
  export CAUSAL_FOA=$SCALED/${tag}/foa
  export CAUSAL_OUT=infer_v2final${DEV_TAG}ft_e149_selfcausal_${tag}
  .venv/bin/python $INFER || exit 4
done
unset CAUSAL_FOA

# 3) 採点（0 dB = 9/23 の e149 selfcausal をそのまま並べる）
cd $E2E
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
S0=$R/infer_v2final${DEV_TAG}ft_e149_selfcausal/val_all_causal.csv
SM=$R/infer_v2final${DEV_TAG}ft_e149_selfcausal_gm2/val_all_causal.csv
SP=$R/infer_v2final${DEV_TAG}ft_e149_selfcausal_gp2/val_all_causal.csv
fail=0
$PY scripts/_score_unified.py $OUT/unified_gain.md --strict --raw --plan $P2 --meta $G2 --split-by scenario g0=$S0 gm2=$SM gp2=$SP > $OUT/_log_unified.txt 2>&1 || { echo FAIL_unified; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_gain.md --plan $P2 --meta $G2 --json $OUT/perception_gain.json --strict g0=$S0 gm2=$SM gp2=$SP > $OUT/_log_perception.txt 2>&1 || { echo FAIL_perception; fail=1; }
$PY scripts/_notify_threshold_sweep.py $OUT/notify_sweep_gain.md --plan $P2 --meta $G2 --csv $OUT/notify_sweep_gain.csv --strict g0=$S0 gm2=$SM gp2=$SP > $OUT/_log_notify.txt 2>&1 || { echo FAIL_notify; fail=1; }
for t in gm2 gp2; do cp $R/infer_v2final${DEV_TAG}ft_e149_selfcausal_${t}/run_meta.json $OUT/run_meta_${t}.json; done
tar czf $OUT/../gain_sensitivity_$(date +%Y%m%d).tgz -C $E2E out/gain_sensitivity_2026-09-27
echo "GAIN_SENS_DONE fail=$fail $(date)"
