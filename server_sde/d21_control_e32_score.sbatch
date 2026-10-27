#!/bin/bash
#SBATCH -J d21c32sc
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH -o d21c32sc-%j.out
# D21 の対照つき追加学習 (b)・更新回数を揃えたやり直し（2026-10-06・RF03）の採点。d21_control_ft_e32.sbatch（noD21・32 ep）の完了後に --dependency=afterok。
#   並べる 3 行: dev2_e149（起点）・cont5s_noD21_e031（対照・24,000 本・32 ep・153,632 歩）・cont5_D21_e029（§3e の D21 腕・25,500 本・30 ep・153,030 歩）。
#   名簿・採点器は d21_control_score.sbatch と同じ（kickfar 300・v12kick 450・自分の val 5,100・交差 val 1,800・統一採点器 v9c --strict --raw ＋ 知覚層）。
# 出力 = out/d21_control5s_2026-10-06/
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
cd $E2E
OUT=$E2E/out/${OUTTAG:-d21_control5s_2026-10-06}; mkdir -p $OUT
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
M12=out/_eval_aux_v12/manifest_val_syn.csv
G12=out/_eval_aux_v12/metadata_dist_h
A=$R/infer_v2finaldev2ft_e149_selfcausal/val_all_causal.csv
AK=$R/infer_v2finaldev2ft_e149_selfcausal_kickfar/val_all_causal.csv
A12=$R/infer_v2finaldev2ft_e149_v12causal/val_all_causal.csv
BN=${BNAME:-dev2cont5snoD21_e031}; CN=${CNAME:-dev2cont5D21_e029}
B=$R/infer_v2final${BN}_selfcausal/val_all_causal.csv
B12=$R/infer_v2final${BN}_v12causal/val_all_causal.csv
C=$R/infer_v2final${CN}_selfcausal/val_all_causal.csv
C12=$R/infer_v2final${CN}_v12causal/val_all_causal.csv
for f in $A $AK $A12 $B $B12 $C $C12; do [ -f "$f" ] || { echo "無い: $f"; exit 2; }; done
# RF02: 再利用する予測は done.json の csv sha と一致することを確かめる
for d in infer_v2final${BN}_selfcausal infer_v2final${BN}_v12causal infer_v2final${CN}_selfcausal infer_v2final${CN}_v12causal; do
  $PY - "$R/$d" <<'PY' || { echo "予測の来歴が合わない: $d"; exit 2; }
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); done = json.loads((d / "val_all_causal.done.json").read_text(encoding="utf-8"))
sys.exit(0 if done.get("sha256") == hashlib.sha256((d / "val_all_causal.csv").read_bytes()).hexdigest() else 1)
PY
done
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
LB=cont5s_noD21_e031; LC=cont5_D21_e029
fail=0
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion dev2_e149=$AK $LB=$B $LC=$C > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_kickfar.md --plan $MK --meta $G2 --json $OUT/perception_kickfar.json --strict dev2_e149=$AK $LB=$B $LC=$C > $OUT/_log_perception_kickfar.txt 2>&1 || { echo FAIL_perception_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion dev2_e149=$A $LB=$B $LC=$C > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_v12kick.md --plan $MV --meta $G2 --json $OUT/perception_v12kick.json --strict dev2_e149=$A $LB=$B $LC=$C > $OUT/_log_perception_v12kick.txt 2>&1 || { echo FAIL_perception_v12kick; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_self.md --strict --raw --plan $P2 --meta $G2 --split-by scenario $LB=$B $LC=$C > $OUT/_log_unified_self.txt 2>&1 || { echo FAIL_unified_self; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross.md --strict --raw --plan $M12 --meta $G12 dev2_e149=$A12 $LB=$B12 $LC=$C12 > $OUT/_log_unified_cross.txt 2>&1 || { echo FAIL_unified_cross; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_cross.md --plan $M12 --meta $G12 --json $OUT/perception_cross.json --strict dev2_e149=$A12 $LB=$B12 $LC=$C12 > $OUT/_log_perception_cross.txt 2>&1 || { echo FAIL_perception_cross; fail=1; }
{ sha256sum $A $AK $A12 $B $B12 $C $C12 scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P2 $MK $MV $M12 || echo LEDGER_SHA_FAILED;
  for d in infer_v2final${BN}_selfcausal infer_v2final${BN}_v12causal infer_v2final${CN}_selfcausal infer_v2final${CN}_v12causal; do echo "$d: $(cat $R/$d/val_all_causal.done.json | tr -d '\n ')"; done;
  cat $E2E/d21ctrl32_*_sha256 2>/dev/null | grep "global_step" ;
  echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)";
  echo "git_head $(git rev-parse HEAD 2>/dev/null)"; } > $OUT/sha256_job${SLURM_JOB_ID}.txt
grep -q LEDGER_SHA_FAILED $OUT/sha256_job${SLURM_JOB_ID}.txt && fail=1
[ $fail -eq 0 ] || { echo "D21_CONTROL_E32_SCORE_FAILED fail=$fail"; exit 1; }
echo "D21_CONTROL_E32_SCORE_DONE $(date)"
