#!/bin/bash
# v17b（装着高さの入力・ゆらぎ 0.10 m）の因果推論結果を取得して統一採点器で採点する（2026-09-05）。ローカルで実行:
#   bash server_sde/_fetch_v17b_results.sh
# 取得: infer_v17bft_e<NNN>_{selfcausal,v12causal,selfcausal_hp20,selfcausal_hm20}/val_all_causal.csv
# 採点（統一採点器・宣言 §3）:
#   out/audit_rescore/v17b_on_v15val.md   主指標 = v16 fold2 の mix≤9000（= v15 val 1,800 本）＋ 高さ 3 帯。v16 e139 と比較
#   out/audit_rescore/v17b_sens.md        感度 = ±0.20 m ずらした入力（同じ 1,800 本）
#   out/audit_rescore/v17b_self_all.md    参考 = v16 fold2 全 3,600 本
#   out/audit_rescore/v17b_cross_v12h.md  交差 = v12 fold2（1.5 m・入力は既定 1.5 m）
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/satos/research/outdoor_seld_e2e
mkdir -p out/v17b/C
for d in $(ssh is-server "ls -d ~/PSELDNets_logs/outdoor_siren_v12/runs/infer_v17bft_e*causal* 2>/dev/null | grep -v -- '-v'"); do
  n=$(basename $d); [ -f out/v17b/C/${n}.csv ] || { scp -q is-server:$d/val_all_causal.csv out/v17b/C/${n}.csv && echo "fetched $n"; }
done
ls out/v17b/C
[ "$(ls out/dataset_outdoor_siren_v16/metadata_dist 2>/dev/null | wc -l)" = "18000" ] || { echo "v16 ラベルが 18,000 本ない"; exit 1; }
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
P16=out/dataset_outdoor_siren_v16/plan/assignment_v16.csv
G16=out/dataset_outdoor_siren_v16/metadata_dist
a=""; for f in out/v17b/C/infer_v17bft_e*_selfcausal.csv; do e=$(basename $f .csv | sed 's/infer_v17bft_//; s/_selfcausal//'); a="$a v17b_${e}=$f"; done
c=""; for f in out/v17b/C/infer_v17bft_e*_v12causal.csv; do e=$(basename $f .csv | sed 's/infer_v17bft_//; s/_v12causal//'); c="$c v17b_${e}=$f"; done
s=""; for f in out/v17b/C/infer_v17bft_e*_selfcausal_h[pm][12]0.csv; do n=$(basename $f .csv | sed 's/infer_v17bft_//; s/_selfcausal//'); s="$s v17b_${n}=$f"; done
$PY scripts/_score_unified.py out/audit_rescore/v17b_on_v15val.md --plan $P16 --clip-max 9000 --meta $G16 --bands 1.4,1.6,1.85,2.1 v16_e139=out/v16/C/infer_v16ft_e139_selfcausal.csv $a > out/audit_rescore/_log_v17b_on_v15val.txt 2>&1 &
$PY scripts/_score_unified.py out/audit_rescore/v17b_sens.md --plan $P16 --clip-max 9000 --meta $G16 --bands 1.4,1.6,1.85,2.1 $s > out/audit_rescore/_log_v17b_sens.txt 2>&1 &
$PY scripts/_score_unified.py out/audit_rescore/v17b_self_all.md --plan $P16 --meta $G16 $a > out/audit_rescore/_log_v17b_self_all.txt 2>&1 &
$PY scripts/_score_unified.py out/audit_rescore/v17b_cross_v12h.md --plan out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv --meta out/dataset_outdoor_siren_v12/metadata_dist_h $c > out/audit_rescore/_log_v17b_cross.txt 2>&1 &
wait
echo "-> out/audit_rescore/v17b_on_v15val.md, v17b_sens.md, v17b_self_all.md, v17b_cross_v12h.md"
echo V17B_FETCH_SCORE_DONE
