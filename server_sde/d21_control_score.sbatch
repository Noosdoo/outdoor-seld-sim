#!/bin/bash
#SBATCH -J d21ctrlsc
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH -o d21ctrlsc-%j.out
# D21 の対照つき追加学習 (b) の採点（2026-10-01）。d21_control_ft.sbatch の 2 腕（D21・noD21）の完了後に --dependency=afterok で走らせる。
#   並べる 3 行: dev2_e149（起点・学習外）・cont_noD21_e029（対照・学習外・同じ歩数）・cont_D21_e029（学習内・同じ歩数）。
#   名簿 = kickfar val 300 本・v12kick val 450 本・自分の val 5,100 本（scenario 別）・v12 の交差 val 1,800 本。統一採点器（v9c・--strict --raw）＋知覚層。
# 出力 = out/d21_control_2026-10-01/
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
cd $E2E
# CONT = 推論名の札（既定 dev2cont = 1 回目 lr1e-4）、OUTTAG = 出力フォルダの札（既定 d21_control_2026-10-01）
CONT=${CONT:-dev2cont}
OUT=$E2E/out/${OUTTAG:-d21_control_2026-10-01}; mkdir -p $OUT
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
M12=out/_eval_aux_v12/manifest_val_syn.csv
G12=out/_eval_aux_v12/metadata_dist_h
A=$R/infer_v2finaldev2ft_e149_selfcausal/val_all_causal.csv
AK=$R/infer_v2finaldev2ft_e149_selfcausal_kickfar/val_all_causal.csv
A12=$R/infer_v2finaldev2ft_e149_v12causal/val_all_causal.csv
B=$R/infer_v2final${CONT}noD21_e029_selfcausal/val_all_causal.csv
B12=$R/infer_v2final${CONT}noD21_e029_v12causal/val_all_causal.csv
C=$R/infer_v2final${CONT}D21_e029_selfcausal/val_all_causal.csv
C12=$R/infer_v2final${CONT}D21_e029_v12causal/val_all_causal.csv
for f in $A $AK $A12 $B $B12 $C $C12; do [ -f "$f" ] || { echo "無い: $f"; exit 2; }; done
rm -f $OUT/manifest_*.csv $OUT/unified_*.md $OUT/perception_*.md $OUT/perception_*.json $OUT/_log_*.txt
$PY - "$P2" "$OUT" <<'EOF' || { echo "FAIL_manifest"; exit 1; }
import csv, sys
from pathlib import Path
plan, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(plan, encoding="utf-8")))
cols = list(rows[0].keys())
for scen, name, n in (("v19kickfar", "manifest_kickfar_val.csv", 300), ("v12kick", "manifest_v12kick_val.csv", 450)):
    sel = [r for r in rows if r["scenario"] == scen and r["split"] == "val"]
    assert len(sel) == n, f"{name} {len(sel)}"
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sel)
    print(name, len(sel))
EOF
MK=$OUT/manifest_kickfar_val.csv; MV=$OUT/manifest_v12kick_val.csv
fail=0
# kickfar: dev2 e149 は kickfar 専用の推論（job 5283）、続きの 2 腕は自分の val の推論から名簿で抜く
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion dev2_e149=$AK cont_noD21_e029=$B cont_D21_e029=$C > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_kickfar.md --plan $MK --meta $G2 --json $OUT/perception_kickfar.json --strict dev2_e149=$AK cont_noD21_e029=$B cont_D21_e029=$C > $OUT/_log_perception_kickfar.txt 2>&1 || { echo FAIL_perception_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion dev2_e149=$A cont_noD21_e029=$B cont_D21_e029=$C > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_v12kick.md --plan $MV --meta $G2 --json $OUT/perception_v12kick.json --strict dev2_e149=$A cont_noD21_e029=$B cont_D21_e029=$C > $OUT/_log_perception_v12kick.txt 2>&1 || { echo FAIL_perception_v12kick; fail=1; }
# 自分の val 全体（scenario 別）。dev2 e149 の自分の val の予測は 4,800 本で plan の 5,100 本を覆わず --strict で止まる（job 5297）ので、続きの 2 腕だけ並べる
$PY scripts/_score_unified.py $OUT/unified_self.md --strict --raw --plan $P2 --meta $G2 --split-by scenario cont_noD21_e029=$B cont_D21_e029=$C > $OUT/_log_unified_self.txt 2>&1 || { echo FAIL_unified_self; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross.md --strict --raw --plan $M12 --meta $G12 dev2_e149=$A12 cont_noD21_e029=$B12 cont_D21_e029=$C12 > $OUT/_log_unified_cross.txt 2>&1 || { echo FAIL_unified_cross; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_cross.md --plan $M12 --meta $G12 --json $OUT/perception_cross.json --strict dev2_e149=$A12 cont_noD21_e029=$B12 cont_D21_e029=$C12 > $OUT/_log_perception_cross.txt 2>&1 || { echo FAIL_perception_cross; fail=1; }
{ sha256sum $A $AK $A12 $B $B12 $C $C12 scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P2 $MK $MV $M12 || echo LEDGER_SHA_FAILED;
  for d in infer_v2final${CONT}noD21_e029_selfcausal infer_v2final${CONT}D21_e029_selfcausal; do echo "$d: $(cat $R/$d/val_all_causal.done.json | tr -d '\n ')"; done;
  echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)";
  echo "git_head $(git rev-parse HEAD 2>/dev/null)"; } > $OUT/sha256_job${SLURM_JOB_ID}.txt
grep -q LEDGER_SHA_FAILED $OUT/sha256_job${SLURM_JOB_ID}.txt && fail=1
[ $fail -eq 0 ] || { echo "D21_CONTROL_SCORE_FAILED fail=$fail"; exit 1; }
echo "D21_CONTROL_SCORE_DONE $(date)"
