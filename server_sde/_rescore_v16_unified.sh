#!/bin/bash
# v16 の因果ft ckpt を統一採点器で採点する（2026-09-05・v16 宣言 §6）。_fetch_v16_results.sh のあとにローカルで実行:
#   bash server_sde/_rescore_v16_unified.sh
# 出力: out/audit_rescore/v16_on_v15val.md（主指標: v16 fold2 のうち mix≤9000 = v15 val・帯ごと）
#       out/audit_rescore/v16_self_all.md（参考: v16 fold2 全 3,600 本）
#       out/audit_rescore/v16_cross_v12h.md（交差 val 1,800 本・水平 GT）
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/satos/research/outdoor_seld_e2e
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
a=""; for f in out/v16/C/infer_v16ft_e*_selfcausal.csv; do e=$(basename $f .csv | sed 's/infer_v16ft_//; s/_selfcausal//'); a="$a v16_${e}=$f"; done
c=""; for f in out/v16/C/infer_v16ft_e*_v12causal.csv; do e=$(basename $f .csv | sed 's/infer_v16ft_//; s/_v12causal//'); c="$c v16_${e}=$f"; done
P16=out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
G16=out/dataset_outdoor_siren_v16/metadata_dist
$PY scripts/_score_unified.py out/audit_rescore/v16_on_v15val.md --plan $P16 --clip-max 9000 --meta $G16 --bands 1.4,1.6,1.85,2.1 $a > out/audit_rescore/_log_v16_on_v15val.txt 2>&1 &
$PY scripts/_score_unified.py out/audit_rescore/v16_self_all.md --plan $P16 --meta $G16 $a > out/audit_rescore/_log_v16_self_all.txt 2>&1 &
$PY scripts/_score_unified.py out/audit_rescore/v16_cross_v12h.md --plan out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv --meta out/dataset_outdoor_siren_v12/metadata_dist_h $c > out/audit_rescore/_log_v16_cross.txt 2>&1 &
wait
# v15 val の GT が v16 の GT（mix≤9000）と同一かの確認（v16 宣言 §4）
n_diff=0; n=0
for f in out/dataset_outdoor_siren_v15/metadata_dist/fold2_room1_mix0*.csv; do
  b=$(basename $f); n=$((n+1))
  cmp -s "$f" "out/dataset_outdoor_siren_v16/metadata_dist/$b" || n_diff=$((n_diff+1))
done
echo "GT同一性: v15 fold2 $n 本のうち v16 と異なる $n_diff 本"
echo "V16_UNIFIED_DONE"
