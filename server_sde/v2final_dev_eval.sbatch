#!/bin/bash
#SBATCH -J v2fdeval
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 08:00:00
#SBATCH -o v2fdeval-%j.out
# 【開発用】v2final_dev_all.sbatch（ジョブ 4527）の推論が全部そろったら、サーバ側で採点まで済ませる（2026-09-16 深夜・本人「今やって」→ 手元の PC が切れていても結果ができているように）。
#   1. 統一採点器（自己 val = v2final fold2 4,500 本・場面別 / 交差 val = v12 fold2 1,800 本・v18 e149 を併記）
#   2. 知覚層だけの精度（dev e099・e149 / 交差 val は v18 e149 も）
#   3. 通知層の閾値掃引（dev e149・図なし = サーバに matplotlib が無い）
#   出力 = out/v2final_dev_eval/ と out/v2final_dev_eval_<日付>.tgz（手元へは scp するだけ）。確定評価には使わない。
# 投入: sbatch --dependency=afterok:4527 server_sde/v2final_dev_eval.sbatch
set -x
cd ~/research/outdoor_seld_e2e
PY=$HOME/research/PSELDNet/PSELDNets/.venv/bin/python
export PYTHONIOENCODING=utf-8
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
DEV_TAG=${DEV_TAG:-dev}                                 # 2026-09-23: 採点する一周の札（dev = 9/16、dev2 = 発進・D1 込みの束）。投入: sbatch --export=ALL,DEV_TAG=dev2 …
PFX=infer_v2final${DEV_TAG}ft
OUT=out/v2final_${DEV_TAG}_eval; rm -rf $OUT; mkdir -p $OUT   # Astra A7: 前回の成果物を混ぜない
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
# 交差 val の manifest と水平の正解は基準フォルダ（_baseline_guard.sh が指紋を取る out/dataset_outdoor_siren_v12）の**外**に置く
# （2026-09-16 01:33 に基準フォルダへ置いたら、走行中のジョブ 4527 のガードが「基準が書き換わった」で止まった。学習自体は完了していた）
M12=out/_eval_aux_v12/manifest_val_syn.csv    # 2026-09-22 改名（旧 manifest_fold2_val.csv）
G12=out/_eval_aux_v12/metadata_dist_h
V18C=$R/verified_v2/infer_v18ft_e149_v12causal/val_all_causal.csv
echo "job=$SLURM_JOB_ID node=$(hostname) $(date)"
a=""; c=""; miss=0
for e in e019 e039 e059 e079 e099 e119 e139 e149; do
  for s in selfcausal v12causal; do
    f=$R/${PFX}_${e}_${s}/val_all_causal.csv
    [ -s "$f" ] || { echo "MISSING $f"; miss=1; }
  done
  a="$a ${DEV_TAG}_${e}=$R/${PFX}_${e}_selfcausal/val_all_causal.csv"
  c="$c ${DEV_TAG}_${e}=$R/${PFX}_${e}_v12causal/val_all_causal.csv"
done
[ $miss = 0 ] || { echo "推論が 16 本そろっていない"; exit 1; }
[ -d "$G12" ] && [ -f "$M12" ] || { echo "交差 val の正解か manifest が無い"; exit 1; }
S=$R/${PFX}_e149_selfcausal/val_all_causal.csv
S99=$R/${PFX}_e099_selfcausal/val_all_causal.csv
C=$R/${PFX}_e149_v12causal/val_all_causal.csv
fail=0
$PY scripts/_score_unified.py $OUT/unified_self.md --strict --raw --plan $P2 --meta $G2 --split-by scenario $a > $OUT/_log_unified_self.txt 2>&1 || { echo FAIL_unified_self; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross.md --strict --raw --plan $M12 --meta $G12 $c v18_e149=$V18C > $OUT/_log_unified_cross.txt 2>&1 || { echo FAIL_unified_cross; fail=1; }
# 変更票 #41（2026-09-28）: 警告音の規則の既定は v9c（名札方式）。旧 hold は併記の規約どおり別ファイルに出す（本文の数字は既定の v9c）
$PY scripts/_score_unified.py $OUT/unified_self_hold.md --warn-rule hold --strict --raw --plan $P2 --meta $G2 --split-by scenario $a > $OUT/_log_unified_self_hold.txt 2>&1 || { echo FAIL_unified_self_hold; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross_hold.md --warn-rule hold --strict --raw --plan $M12 --meta $G12 $c v18_e149=$V18C > $OUT/_log_unified_cross_hold.txt 2>&1 || { echo FAIL_unified_cross_hold; fail=1; }
# 札は ${DEV_TAG}_ にそろえる（Opus 5.5 監査 B10・2026-09-24: dev2 の採点なのに知覚層と掃引の札が dev_ だった。ジョブ 4878 の原本の dev_e099/dev_e149 は dev2 の予測）
$PY scripts/_perception_eval.py $OUT/perception_self.md --plan $P2 --meta $G2 --json $OUT/perception_self.json --strict ${DEV_TAG}_e099=$S99 ${DEV_TAG}_e149=$S > $OUT/_log_perception_self.txt 2>&1 || { echo FAIL_perception_self; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_cross.md --plan $M12 --meta $G12 --json $OUT/perception_cross.json --strict ${DEV_TAG}_e149=$C v18_e149=$V18C > $OUT/_log_perception_cross.txt 2>&1 || { echo FAIL_perception_cross; fail=1; }
$PY scripts/_notify_threshold_sweep.py $OUT/notify_sweep_self.md --plan $P2 --meta $G2 --csv $OUT/notify_sweep_self.csv --strict ${DEV_TAG}_e149=$S > $OUT/_log_notify_self.txt 2>&1 || { echo FAIL_notify_self; fail=1; }
$PY scripts/_notify_threshold_sweep.py $OUT/notify_sweep_cross.md --plan $M12 --meta $G12 --csv $OUT/notify_sweep_cross.csv --strict ${DEV_TAG}_e149=$C v18_e149=$V18C > $OUT/_log_notify_cross.txt 2>&1 || { echo FAIL_notify_cross; fail=1; }
ls -la $OUT
tar czf out/v2final_${DEV_TAG}_eval_$(date +%Y%m%d).tgz $OUT
ls -la out/v2final_${DEV_TAG}_eval_*.tgz
[ $fail = 0 ] && { echo V2FINAL_DEV_EVAL_DONE; exit 0; } || { echo V2FINAL_DEV_EVAL_PARTIAL; exit 1; }   # Astra A7: 失敗は非 0 で終える
