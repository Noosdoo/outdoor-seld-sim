#!/bin/bash
#SBATCH -J d21epochs
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH -o d21epochs-%j.out
# D21 の効果比較を e149 の 1 点から e099〜e149 の 4 点に広げる（2026-10-01 本人「お願いします」= 敵対レビュー B06・E01・監査 A F1 の「エポックで向きが変わる」への追加実験 (a)）。
#   1) dev2 の epoch_099／119／139 の ckpt で kickfar の val 300 本を推論する（e149 は job 5283 で済み）。dev3 側は job 5092 の自分の val の推論に含まれている。
#   2) kickfar val 300 本と v12kick val 450 本の 2 名簿で、dev2 と dev3 の 4 エポックずつ（8 行）を統一採点器（v9c・--strict --raw）と知覚層で並べる。
# 既存の推論結果・データセットには触れない。出力 = out/d21_effect_epochs_2026-10-01/。投入: sbatch server_sde/d21_effect_epochs.sbatch
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CKDIR=$R/outdoor_siren_v2final_dev2_causal_ft/checkpoints
ROOTDIR=$PS/datasets_v2final
OUT=$E2E/out/d21_effect_epochs_2026-10-01; mkdir -p $OUT
JOBDIR=$OUT/job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $E2E/server_sde/_prediction_integrity.py $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
echo "job=$SLURM_JOB_ID node=$(hostname) $(date)"
LEDGER=$OUT/sha256_job${SLURM_JOB_ID}.txt
sha256sum $CKDIR/epoch_099.ckpt $CKDIR/epoch_119.ckpt $CKDIR/epoch_139.ckpt $CKDIR/last.ckpt $INFER $JOBDIR/_prediction_integrity.py $JOBDIR/hysteresis_patch.py $PS/height_patch.py $PS/sde_patch.py | tee $LEDGER
for E in 099 119 139 149; do
  for f in $R/infer_v2finaldev3ft_e${E}_selfcausal/val_all_causal.csv $R/infer_v2finaldev2ft_e${E}_selfcausal/val_all_causal.csv; do
    [ -f "$f" ] || { echo "無い: $f"; exit 2; }
  done
done
[ -f $R/infer_v2finaldev2ft_e149_selfcausal_kickfar/val_all_causal.csv ] || { echo "job 5283 の kickfar 推論（e149）が無い"; exit 2; }

# 1) dev2 の e099／e119／e139 で kickfar の val 300 本を推論（設定は d21_effect.sbatch と同じ = v2final_dev_causal.sh の selfcausal）
cd $PS
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
export CAUSAL_PREFIX=val_syn_ CAUSAL_DSDIR=$ROOTDIR
export CAUSAL_EXPECTED_CLIPS=300
for E in 099 119 139; do
  CK=$CKDIR/epoch_$E.ckpt
  CK_SHA=$(sha256sum $CK | cut -c1-64); [ ${#CK_SHA} -eq 64 ] || { echo "ckpt の sha が取れない: $CK"; exit 2; }
  export CAUSAL_CKPT=$CK CAUSAL_FOA=$ROOTDIR/outdoor_siren_v12/foa CAUSAL_GLOB='val_syn_clip3[01]*.flac'
  export CAUSAL_OUT=infer_v2finaldev2ft_e${E}_selfcausal_kickfar
  REUSE=0
  if [ -f "$R/$CAUSAL_OUT/val_all_causal.csv" ] && [ -f "$R/$CAUSAL_OUT/val_all_causal.done.json" ]; then
    CSV_SHA=$(sha256sum "$R/$CAUSAL_OUT/val_all_causal.csv" | cut -c1-64)
    .venv/bin/python - "$R/$CAUSAL_OUT/val_all_causal.done.json" "$CK_SHA" "$CSV_SHA" <<'EOF' && REUSE=1
import json, sys
d = json.load(open(sys.argv[1]))
ok = d.get("ckpt_sha256") == sys.argv[2] and int(d.get("n_clips", -1)) == 300 and d.get("sha256") == sys.argv[3]
print("reuse check:", ok, d.get("ckpt_sha256", "")[:12], d.get("n_clips"))
sys.exit(0 if ok else 1)
EOF
  fi
  if [ $REUSE -eq 1 ]; then echo "既存の kickfar 推論を使う: $CAUSAL_OUT"; else
    rm -rf "$R/$CAUSAL_OUT"
    .venv/bin/python $INFER || { echo "dev2 e$E の kickfar 推論に失敗。中止"; exit 4; }
  fi
  [ -f "$R/$CAUSAL_OUT/val_all_causal.csv" ] || { echo "kickfar の予測が無い: $CAUSAL_OUT"; exit 4; }
  unset CAUSAL_FOA CAUSAL_GLOB CAUSAL_CKPT
done

# 2) 名簿 2 つを plan から作る（d21_effect.sbatch と同じ）
cd $E2E
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
rm -f $OUT/manifest_kickfar_val.csv $OUT/manifest_v12kick_val.csv
$PY - "$P2" "$OUT" <<'EOF' || { echo "FAIL_manifest（名簿の生成に失敗）"; exit 1; }
import csv, sys
from pathlib import Path
plan, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(plan, encoding="utf-8")))
cols = list(rows[0].keys())
for scen, name in (("v19kickfar", "manifest_kickfar_val.csv"), ("v12kick", "manifest_v12kick_val.csv")):
    sel = [r for r in rows if r["scenario"] == scen and r["split"] == "val"]
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sel)
    print(name, len(sel))
    assert len(sel) == {"v19kickfar": 300, "v12kick": 450}[scen], f"{name} の本数 {len(sel)}"
EOF
MK=$OUT/manifest_kickfar_val.csv
MV=$OUT/manifest_v12kick_val.csv
[ -f "$MK" ] && [ -f "$MV" ] || { echo "名簿が無い"; exit 1; }

# 3) 8 行（dev2 e099/119/139/149・dev3 e099/119/139/149）を並べる。kickfar: dev2 は kickfar 専用の推論、dev3 は自分の val の推論から名簿で抜く
K2=""; K3=""; S2=""; S3=""
for E in 099 119 139 149; do
  K2="$K2 dev2_e${E}=$R/infer_v2finaldev2ft_e${E}_selfcausal_kickfar/val_all_causal.csv"
  K3="$K3 dev3_e${E}=$R/infer_v2finaldev3ft_e${E}_selfcausal/val_all_causal.csv"
  S2="$S2 dev2_e${E}=$R/infer_v2finaldev2ft_e${E}_selfcausal/val_all_causal.csv"
  S3="$S3 dev3_e${E}=$R/infer_v2finaldev3ft_e${E}_selfcausal/val_all_causal.csv"
done
rm -f $OUT/unified_*.md $OUT/perception_*.md $OUT/perception_*.json $OUT/_log_*.txt
fail=0
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion $K2 $K3 > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_kickfar.md --plan $MK --meta $G2 --json $OUT/perception_kickfar.json --strict $K2 $K3 > $OUT/_log_perception_kickfar.txt 2>&1 || { echo FAIL_perception_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion $S2 $S3 > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_v12kick.md --plan $MV --meta $G2 --json $OUT/perception_v12kick.json --strict $S2 $S3 > $OUT/_log_perception_v12kick.txt 2>&1 || { echo FAIL_perception_v12kick; fail=1; }
for E in 099 119 139; do cp $R/infer_v2finaldev2ft_e${E}_selfcausal_kickfar/run_meta.json $OUT/run_meta_dev2_e${E}_kickfar.json 2>/dev/null || true; done
{ sha256sum scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P2 $MK $MV || echo "LEDGER_SHA_FAILED";
  for E in 099 119 139 149; do sha256sum $R/infer_v2finaldev2ft_e${E}_selfcausal_kickfar/val_all_causal.csv $R/infer_v2finaldev2ft_e${E}_selfcausal/val_all_causal.csv $R/infer_v2finaldev3ft_e${E}_selfcausal/val_all_causal.csv || echo "LEDGER_SHA_FAILED"; done;
  ( cd $G2 && find . -name '*.csv' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | sed "s|-|metadata_dist の中身（全 csv の sha の sha）|" );
  echo "git_head $(git rev-parse HEAD 2>/dev/null)"; echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } >> $LEDGER
grep -q LEDGER_SHA_FAILED $LEDGER && fail=1
[ $fail -eq 0 ] || { echo "D21_EPOCHS_FAILED fail=$fail $(date)"; exit 1; }
tar czf $E2E/out/d21_effect_epochs_$(date +%Y%m%d).tgz -C $E2E out/d21_effect_epochs_2026-10-01
echo "D21_EPOCHS_DONE fail=$fail $(date)"
