#!/bin/bash
#SBATCH -J d21seeds
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH -o d21seeds-%j.out
# (c) 種を足した 6 本（D21 なし: dev2・dev2s1・dev2s2／D21 あり: dev3・dev3s1・dev3s2・いずれも因果 ft e149）を同じ名簿で並べる（2026-10-02）。
#   kickfar val 300 本（D21 なし側は学習外・dev2 だけは job 5283 の kickfar 専用の推論、dev2s1/s2 は自分の val の推論に kickfar が含まれる）、
#   v12kick val 450 本、自分の val（scenario 別・dev2 は 4,800 本なので外す）、交差 val 1,800 本。統一採点器（v9c・--strict --raw）＋知覚層。
# 投入: 4 本（job 5298〜5301）の完走後に sbatch server_sde/d21_seeds_score.sbatch。出力 = out/d21_seeds_2026-10-02/
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
cd $E2E
OUT=$E2E/out/${OUTTAG:-d21_seeds_2026-10-02}; mkdir -p $OUT
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
M12=out/_eval_aux_v12/manifest_val_syn.csv
G12=out/_eval_aux_v12/metadata_dist_h
S() { echo $R/infer_v2final${1}ft_e149_selfcausal/val_all_causal.csv; }
C() { echo $R/infer_v2final${1}ft_e149_v12causal/val_all_causal.csv; }
AK=$R/infer_v2finaldev2ft_e149_selfcausal_kickfar/val_all_causal.csv
for t in dev2 dev2s1 dev2s2 dev3 dev3s1 dev3s2; do for f in $(S $t) $(C $t); do [ -f "$f" ] || { echo "無い: $f"; exit 2; }; done; done
[ -f "$AK" ] || { echo "無い: $AK"; exit 2; }
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
# 行の並び: D21 なし 3 種 → D21 あり 3 種
KROWS="dev2_e149=$AK dev2s1_e149=$(S dev2s1) dev2s2_e149=$(S dev2s2) dev3_e149=$(S dev3) dev3s1_e149=$(S dev3s1) dev3s2_e149=$(S dev3s2)"
SROWS="dev2_e149=$(S dev2) dev2s1_e149=$(S dev2s1) dev2s2_e149=$(S dev2s2) dev3_e149=$(S dev3) dev3s1_e149=$(S dev3s1) dev3s2_e149=$(S dev3s2)"
SROWS5100="dev2s1_e149=$(S dev2s1) dev2s2_e149=$(S dev2s2) dev3_e149=$(S dev3) dev3s1_e149=$(S dev3s1) dev3s2_e149=$(S dev3s2)"
CROWS="dev2_e149=$(C dev2) dev2s1_e149=$(C dev2s1) dev2s2_e149=$(C dev2s2) dev3_e149=$(C dev3) dev3s1_e149=$(C dev3s1) dev3s2_e149=$(C dev3s2)"
fail=0
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion $KROWS > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_kickfar.md --plan $MK --meta $G2 --json $OUT/perception_kickfar.json --strict $KROWS > $OUT/_log_perception_kickfar.txt 2>&1 || { echo FAIL_perception_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion $SROWS > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_v12kick.md --plan $MV --meta $G2 --json $OUT/perception_v12kick.json --strict $SROWS > $OUT/_log_perception_v12kick.txt 2>&1 || { echo FAIL_perception_v12kick; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_self.md --strict --raw --plan $P2 --meta $G2 --split-by scenario $SROWS5100 > $OUT/_log_unified_self.txt 2>&1 || { echo FAIL_unified_self; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross.md --strict --raw --plan $M12 --meta $G12 $CROWS > $OUT/_log_unified_cross.txt 2>&1 || { echo FAIL_unified_cross; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_cross.md --plan $M12 --meta $G12 --json $OUT/perception_cross.json --strict $CROWS > $OUT/_log_perception_cross.txt 2>&1 || { echo FAIL_perception_cross; fail=1; }
{ for t in dev2 dev2s1 dev2s2 dev3 dev3s1 dev3s2; do sha256sum $(S $t) $(C $t) || echo LEDGER_SHA_FAILED; echo "$t: $(cat $R/infer_v2final${t}ft_e149_selfcausal/val_all_causal.done.json | tr -d '\n ')"; done;
  sha256sum $AK scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P2 $MK $MV $M12 || echo LEDGER_SHA_FAILED;
  for t in dev2s1 dev2s2 dev3s1 dev3s2; do grep -hE "seed=|hdf5=" $E2E/v2fdev-*.out 2>/dev/null | grep "exp=outdoor_siren_v2final_${t}_" | tail -1 | cut -c1-300; done;
  echo "git_head $(git rev-parse HEAD 2>/dev/null)"; } > $OUT/sha256_job${SLURM_JOB_ID}.txt
grep -q LEDGER_SHA_FAILED $OUT/sha256_job${SLURM_JOB_ID}.txt && fail=1
[ $fail -eq 0 ] || { echo "D21_SEEDS_SCORE_FAILED fail=$fail"; exit 1; }
echo "D21_SEEDS_SCORE_DONE $(date)"
