#!/bin/bash
# v16 候補学習（高さ増強）の因果推論結果を取得して採点する（2026-09-05）。ローカルで実行:
#   bash server_sde/_fetch_v16_results.sh
# 取得: infer_v16ft_e<NNN>_selfcausal（v16 fold2 val 3,600 本）/ infer_v16ft_e<NNN>_v12causal（v12 fold2 val）の val_all_causal.csv
# 採点（⚠️ 以下は旧採点器 _hp_score/_band_score の値＝履歴用。判定は server_sde/_rescore_v16_unified.sh（統一採点器）で行う）:
#   score_self_all.md   = v16 fold2 全 3,600 本（GT= v16 metadata_dist・水平距離）
#   score_on_v15val.md  = そのうち mix≤9000 の 1,800 本 = v15 val そのもの（主指標）＋ 高さ 3 帯（副指標）。v15/v15b/v15c と同じ土俵
#   score_cross_v12h.md = v12 fold2 val（水平距離ラベル metadata_dist_h。ft2 も同じラベルで再採点）
cd /c/Users/satos/research/outdoor_seld_e2e
mkdir -p out/v16/C
for d in $(ssh is-server "ls -d ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_v16ft_e*_*causal 2>/dev/null | grep -v -- '-v'"); do
  n=$(basename $d); scp -q is-server:$d/val_all_causal.csv out/v16/C/${n}.csv && echo "fetched $n"
done
ls out/v16/C
# ラベルは tar ストリームで取る（scp -r は 18,000 ファイルの途中で欠けることがあった: 2026-09-05 に 8,936 本で止まった）
if [ "$(ls out/dataset_outdoor_siren_v16/metadata_dist 2>/dev/null | wc -l)" != "18000" ]; then
  mkdir -p out/dataset_outdoor_siren_v16
  ssh is-server "cd ~/research/outdoor_seld_e2e/out/dataset_outdoor_siren_v16 && tar cf - metadata_dist" | tar xf - -C out/dataset_outdoor_siren_v16
fi
ls out/dataset_outdoor_siren_v16/metadata_dist | wc -l
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
args_self=""; args_cross=""
for f in out/v16/C/infer_v16ft_e*_selfcausal.csv; do e=$(basename $f .csv | sed 's/infer_v16ft_//; s/_selfcausal//'); args_self="$args_self v16ft_${e}=$f"; done
for f in out/v16/C/infer_v16ft_e*_v12causal.csv; do e=$(basename $f .csv | sed 's/infer_v16ft_//; s/_v12causal//'); args_cross="$args_cross v16ft_${e}=$f"; done
args_cross="ft2_e079=out/hp_sweep/ref/ft2_e079_val_causal.csv $args_cross"
PYTHONIOENCODING=utf-8 $PY scripts_archive/_hp_score.py out/v16/score_self_all.md --meta out/dataset_outdoor_siren_v16/metadata_dist $args_self
PYTHONIOENCODING=utf-8 $PY scripts_archive/_band_score.py out/v16/score_on_v15val.md --plan out/dataset_outdoor_siren_v16/plan/assignment_v16.csv \
    --meta out/dataset_outdoor_siren_v16/metadata_dist --clip-max 9000 $args_self
PYTHONIOENCODING=utf-8 $PY scripts_archive/_hp_score.py out/v16/score_cross_v12h.md --meta out/dataset_outdoor_siren_v12/metadata_dist_h $args_cross
echo "-> out/v16/score_self_all.md, out/v16/score_on_v15val.md, out/v16/score_cross_v12h.md"
