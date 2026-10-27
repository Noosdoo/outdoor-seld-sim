#!/bin/bash
# v15 候補学習（job 3943→3944→3945）の因果推論結果を取得して採点する（2026-09-04）。ローカルで実行:
#   bash server_sde/_fetch_v15_results.sh
# 取得: infer_v15bft_e<NNN>_selfcausal（v15 fold2 val）/ infer_v15bft_e<NNN>_v12causal（v12 fold2 val）の val_all_causal.csv
# 採点: 自己 val = META v15 metadata_dist（水平距離）、交差 val = v12 metadata_dist_h（水平距離）
cd /c/Users/satos/research/outdoor_seld_e2e
mkdir -p out/v15b/C
for d in $(ssh is-server "ls -d ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_v15bft_e*_*causal 2>/dev/null | grep -v -- '-v'"); do
  n=$(basename $d); scp -q is-server:$d/val_all_causal.csv out/v15b/C/${n}.csv && echo "fetched $n"
done
ls out/v15b/C
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
args_self=""; args_cross=""
for f in out/v15b/C/infer_v15bft_e*_selfcausal.csv; do e=$(basename $f .csv | sed 's/infer_v15bft_//; s/_selfcausal//'); args_self="$args_self v15bft_${e}=$f"; done
for f in out/v15b/C/infer_v15bft_e*_v12causal.csv; do e=$(basename $f .csv | sed 's/infer_v15bft_//; s/_v12causal//'); args_cross="$args_cross v15bft_${e}=$f"; done
# 交差 val は ft2（現行）も同じ水平距離ラベルで再採点して並べる
args_cross="ft2_e079=out/hp_sweep/ref/ft2_e079_val_causal.csv $args_cross"
PYTHONIOENCODING=utf-8 $PY scripts_archive/_hp_score.py out/v15b/score_self.md --meta out/dataset_outdoor_siren_v15b/metadata_dist $args_self
PYTHONIOENCODING=utf-8 $PY scripts_archive/_hp_score.py out/v15b/score_cross_v12h.md --meta out/dataset_outdoor_siren_v12/metadata_dist_h $args_cross
echo "-> out/v15b/score_self.md, out/v15b/score_cross_v12h.md"
