#!/bin/bash
#SBATCH -J v22score
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 02:00:00
#SBATCH -o v22score-%j.out
# v22（D22 bicycle・D23 ピン型ベル・D24 打音合わせ・9 クラス）の開発一周の採点（2026-10-08）。v22_dev_all.sbatch（5799）の完了後に投入。
#   並べる: 基準 = dev3 の D21 腕 cont5_D21_e029（8 クラス・旧 25,500 本）と、v22 の因果 ft（9 クラス・31,274 本）の最後の ckpt。
#   合格線（レビュー C 10/7・変更票 #48）: (a) dev3 の 3 種（kickfar・v12kick・交差 val）が ±2 pt 以内、(b) ベルの到達が −2 pt 以内、(c) キックと自転車の取り違え 5% 以下。
#   比較は「共通の検証 5,100 本（v22 前の plan の val）」で行い、新しい場面（v19bicycle／v19bellpin／v19bellbike）は別行で出す。
# 出力 = out/v22_dev_score_2026-10-08/
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
cd $E2E
OUT=$E2E/out/${OUTTAG:-v22_dev_score_2026-10-08}; mkdir -p $OUT
P22=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv            # 31,274 行
PPRE=out/dataset_outdoor_siren_v2final/plan/assignment_v2final_pre_v22.csv   # 25,500 行（共通の検証 5,100 本）
G2=out/dataset_outdoor_siren_v2final/metadata_dist
M12=out/_eval_aux_v12/manifest_val_syn.csv
G12=out/_eval_aux_v12/metadata_dist_h
# 基準（dev3 の D21 腕）
BN=dev2cont5D21_e029
B=$R/infer_v2final${BN}_selfcausal/val_all_causal.csv
B12=$R/infer_v2final${BN}_v12causal/val_all_causal.csv
# v22（最後の ckpt の推論。v22_dev_causal.sh は epoch ごとに infer_v2finalv22ft_e<N>_selfcausal を作る）
VE=${VE:-$(ls -d $R/infer_v2finalv22ft_e*_selfcausal 2>/dev/null | sed 's/.*_e\([0-9]*\)_selfcausal/\1/' | sort -n | tail -1)}
[ -n "$VE" ] || { echo "v22 の推論が無い"; exit 2; }
V=$R/infer_v2finalv22ft_e${VE}_selfcausal/val_all_causal.csv
V12=$R/infer_v2finalv22ft_e${VE}_v12causal/val_all_causal.csv
for f in $B $B12 $V $V12; do [ -f "$f" ] || { echo "無い: $f"; exit 2; }; done
for d in infer_v2final${BN}_selfcausal infer_v2final${BN}_v12causal infer_v2finalv22ft_e${VE}_selfcausal infer_v2finalv22ft_e${VE}_v12causal; do
  $PY - "$R/$d" <<'PY' || { echo "予測の来歴が合わない: $d"; exit 2; }
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); done = json.loads((d / "val_all_causal.done.json").read_text(encoding="utf-8"))
sys.exit(0 if done.get("sha256") == hashlib.sha256((d / "val_all_causal.csv").read_bytes()).hexdigest() else 1)
PY
done
rm -f $OUT/manifest_*.csv $OUT/unified_*.md $OUT/perception_*.md $OUT/perception_*.json $OUT/confusion_*.json $OUT/_log_*.txt
$PY - "$P22" "$OUT" <<'EOF' || { echo "FAIL_manifest"; exit 1; }
import csv, sys
from pathlib import Path
plan, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(plan, encoding="utf-8")))
cols = list(rows[0].keys())
for scen, name, n in (("v19kickfar", "manifest_kickfar_val.csv", 300), ("v12kick", "manifest_v12kick_val.csv", 450),
                      ("v19bicycle", "manifest_bicycle_val.csv", 900), ("v19bellpin", "manifest_bellpin_val.csv", 156), ("v19bellbike", "manifest_bellbike_val.csv", 100)):
    sel = [r for r in rows if r["scenario"] == scen and r["split"] == "val"]
    assert len(sel) == n, f"{name} {len(sel)}"
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sel)
    print(name, len(sel))
EOF
MK=$OUT/manifest_kickfar_val.csv; MV=$OUT/manifest_v12kick_val.csv; MB=$OUT/manifest_bicycle_val.csv; MP=$OUT/manifest_bellpin_val.csv; MBB=$OUT/manifest_bellbike_val.csv
LB=dev3_D21_e029; LV=v22_e${VE}
fail=0
# (a) 3 種（基準と同じ名簿・共通の本）
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion $LB=$B $LV=$V > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion $LB=$B $LV=$V > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross.md --strict --raw --plan $M12 --meta $G12 $LB=$B12 $LV=$V12 > $OUT/_log_unified_cross.txt 2>&1 || { echo FAIL_unified_cross; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_cross.md --plan $M12 --meta $G12 --json $OUT/perception_cross.json --strict $LB=$B12 $LV=$V12 > $OUT/_log_perception_cross.txt 2>&1 || { echo FAIL_perception_cross; fail=1; }
# 共通の検証 5,100 本（v22 前の plan の val）で場面別（ベルの到達 (b) はここの bike_bell の行と、v19bellpin／v19bellbike の行で見る）
$PY scripts/_score_unified.py $OUT/unified_common.md --strict --raw --plan $PPRE --meta $G2 --split-by scenario $LB=$B $LV=$V > $OUT/_log_unified_common.txt 2>&1 || { echo FAIL_unified_common; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_common.md --plan $PPRE --meta $G2 --json $OUT/perception_common.json --strict $LB=$B $LV=$V > $OUT/_log_perception_common.txt 2>&1 || { echo FAIL_perception_common; fail=1; }
# 新しい場面（v22 だけ・基準には本が無い）
$PY scripts/_score_unified.py $OUT/unified_bicycle.md --strict --raw --plan $MB --meta $G2 --split-by motion $LV=$V > $OUT/_log_unified_bicycle.txt 2>&1 || { echo FAIL_unified_bicycle; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_bicycle.md --plan $MB --meta $G2 --json $OUT/perception_bicycle.json --strict $LV=$V > $OUT/_log_perception_bicycle.txt 2>&1 || { echo FAIL_perception_bicycle; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_bellpin.md --strict --raw --plan $MP --meta $G2 $LV=$V > $OUT/_log_unified_bellpin.txt 2>&1 || { echo FAIL_unified_bellpin; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_bellbike.md --strict --raw --plan $MBB --meta $G2 $LV=$V > $OUT/_log_unified_bellbike.txt 2>&1 || { echo FAIL_unified_bellbike; fail=1; }
# (c) キックと自転車の取り違え（知覚層・20° 以内・frame ≥ 40）
$PY scripts/_bicycle_kick_confusion.py --pred $V --meta $G2 --plan $P22 --scenario v19bicycle,v12kick,v19kickfar --split val --json $OUT/confusion_v22.json > $OUT/_log_confusion_v22.txt 2>&1 || { echo FAIL_confusion; fail=1; }
$PY scripts/_bicycle_kick_confusion.py --pred $B --meta $G2 --plan $PPRE --scenario v12kick,v19kickfar --split val --json $OUT/confusion_dev3.json > $OUT/_log_confusion_dev3.txt 2>&1 || { echo FAIL_confusion_dev3; fail=1; }
{ sha256sum $B $B12 $V $V12 scripts/_score_unified.py scripts/_perception_eval.py scripts/_bicycle_kick_confusion.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P22 $PPRE $MK $MV $MB $MP $MBB || echo LEDGER_SHA_FAILED;
  for d in infer_v2final${BN}_selfcausal infer_v2final${BN}_v12causal infer_v2finalv22ft_e${VE}_selfcausal infer_v2finalv22ft_e${VE}_v12causal; do echo "$d: $(cat $R/$d/val_all_causal.done.json | tr -d '\n ')"; done;
  echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; echo "git_head $(git rev-parse HEAD 2>/dev/null)"; } > $OUT/sha256_job${SLURM_JOB_ID}.txt
grep -q LEDGER_SHA_FAILED $OUT/sha256_job${SLURM_JOB_ID}.txt && fail=1
[ $fail -eq 0 ] || { echo "V22_DEV_SCORE_FAILED fail=$fail"; exit 1; }
echo "V22_DEV_SCORE_DONE $(date)"
