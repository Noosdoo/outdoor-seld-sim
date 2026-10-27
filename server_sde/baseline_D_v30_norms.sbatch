#!/bin/bash
#SBATCH -J bDv30
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o bDv30-%j.out
# 相手 D の検出のしきい値を v30 の val で作り直すための norms（本体 §3-5・§4 の 4・Fable 10/26 16:5x「D の倍率 40 は据え置き・D の表は v30 の val の norms で作り直す」）。
#   baseline_D_select.sbatch（v29・ゲインの掃引）の v30 版。ゲインは選び直さない（40 固定・#73 の選択のまま）。主の倍率（SDE_INPUT_GAIN_DB）は掛けない（外して止める）。
#   対象 = v30 の val の 選択用（clip%12==0・manifest_val_select.csv）と確認用（%12==6・manifest_val_check.csv）を合わせた名簿（1 回の推論で 1 つの run に）。
#   出力: $HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_v30/infer_baselineD_v30selchk_g40/（norms/・norms_manifest.json・来歴）
#   そのあと手元で: python scripts/v30_select_thresholds.py --gt <v30 の metadata_dist> --manifest <この run の名簿> --run <この run> --label D \
#                     --classmap md/design/baselines_class_map_v2.json --target D --out md/design/thresholds_v30_D.json
#   開発の比べ（§4b）の val 全部の D の予測は、表ができたあと CAUSAL_THRESHOLDS_JSON を渡して別に回す（この run の 0.7/0.3 の予測は使わない）。
#   eval v4・実録・学習の置き場には触れない（v30 の val_syn_ だけ）。low は中断されうる → --requeue と推論の「未処理だけ」の再開で続きから。
#   使い方: sbatch server_sde/baseline_D_v30_norms.sbatch   （乾式: D_V30_DRY=1 で名簿と sha の印字だけ）
set -exo pipefail
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
cd $PS
for v in $(compgen -e | grep '^CAUSAL_' || true); do unset "$v"; done
unset SDE_INPUT_GAIN_DB                                     # 主の倍率は D に重ねない（本体 §3-2・二重適用の防ぎ）
export PS E2E
DS30=$E2E/out/dataset_outdoor_siren_v2final_v30
DS170=$PS/datasets_pretrain170
mkdir -p $DS170
cp $E2E/out/_baselines/cls_indices_train_pseldnets170.tsv $DS170/cls_indices_train.tsv
export CAUSAL_CKPT=$PS/ckpts/mACCDOA-HTSAT-0.567.ckpt
export CAUSAL_DSDIR=$DS170
export CAUSAL_DATASET=synth
export BASELINE_EXPERIMENT=synth_maccdoa
export BASELINE_CLASSMAP=$E2E/md/design/baselines_class_map_v2.json
export CAUSAL_FOA=$PS/datasets_v30/outdoor_siren_v12/foa
export CAUSAL_PREFIX=val_syn_
export CAUSAL_RUNS_DIR=$HOME/PSELDNets_logs/outdoor_siren_v12/runs/_baseline_v30
mkdir -p $CAUSAL_RUNS_DIR
export CAUSAL_OUT=infer_baselineD_v30selchk_g40
MAN=$CAUSAL_RUNS_DIR/manifest_val_selchk_v30.csv
# 名簿 = 選択用 ＋ 確認用（見出しは 1 行・順は選択用 → 確認用・重複があれば止める）。再投入のときは同じ中身であることを確かめる
TMPM=$(mktemp)
{ head -n 1 $DS30/plan/manifest_val_select.csv; tail -n +2 $DS30/plan/manifest_val_select.csv; tail -n +2 $DS30/plan/manifest_val_check.csv; } > $TMPM
DUP=$(tail -n +2 $TMPM | cut -d, -f1 | sort | uniq -d | wc -l)
[ "$DUP" = 0 ] || { echo "選択用と確認用に重なる本がある（$DUP）→ 止める"; exit 3; }
if [ -f $MAN ]; then cmp -s $TMPM $MAN || { echo "前の名簿 $MAN と中身が違う → 止める"; exit 3; }; else cp $TMPM $MAN; fi
rm -f $TMPM
export CAUSAL_MANIFEST=$MAN
export CAUSAL_LIMIT=0
N=$(($(wc -l < $MAN) - 1))
export CAUSAL_EXPECTED_CLIPS=$N
export BASELINE_GAIN_DB=40
export CAUSAL_SED_TH=0.7
export CAUSAL_SED_STAY=0.3
export CAUSAL_SED_ASSOC=20
export BASELINE_DUMP_NORMS=1
echo "D v30: 名簿 $N 本（選択用 $(($(wc -l < $DS30/plan/manifest_val_select.csv) - 1))・確認用 $(($(wc -l < $DS30/plan/manifest_val_check.csv) - 1))）・ゲイン 40・foa $(ls $CAUSAL_FOA | grep -c '^val_syn_') 本"
sha256sum $CAUSAL_CKPT $E2E/server_sde/_baseline_D_pretrain170_infer.py $E2E/server_sde/_baseline_merge.py $BASELINE_CLASSMAP $DS170/cls_indices_train.tsv \
  $DS30/plan/manifest_val_select.csv $DS30/plan/manifest_val_check.csv $MAN | cut -c1-16,65-
[ "${D_V30_DRY:-0}" = 1 ] && { echo "乾式なのでここで終わる"; exit 0; }
$PY $E2E/server_sde/_baseline_D_pretrain170_infer.py
$PY $E2E/server_sde/_prediction_integrity.py "$CAUSAL_RUNS_DIR/$CAUSAL_OUT" $N
echo "norms $(ls $CAUSAL_RUNS_DIR/$CAUSAL_OUT/norms | wc -l)・rows $(wc -l < $CAUSAL_RUNS_DIR/$CAUSAL_OUT/val_all_causal.csv)"
echo BASELINE_D_V30_NORMS_DONE
