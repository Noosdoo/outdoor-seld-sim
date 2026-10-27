#!/bin/bash
#SBATCH -J d21effect
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 03:00:00
#SBATCH -o d21effect-%j.out
# D21（キックの遠い横距離 2.5〜8 m・1,500 本）の効果を、同じ val で dev2（D21 なし・24,000 本で学習）と dev3（D21 込み・25,500 本）の
# 因果 ft e149 を並べて見る（2026-10-01・本人 9/28「学習データの追加はやってください」→ dev3 job 5092 完走後）。
#   1) dev2 の e149（last.ckpt）で、dev3 の val にしか無い kickfar の val 300 本（val_syn_clip30xxx/31xxx）を推論する
#      （dev2 は学習にも val にも kickfar を持たない = 学習外の条件への汎化）。
#   2) 名簿 2 つで新旧を採点: kickfar val 300 本（2.5〜8 m・学習外 vs 学習内）と v12kick val 450 本（0.8〜2.5 m・近いキックが悪くならないか）。
#      統一採点器（v4.3＋警告音 v9c・--strict --raw）と知覚層 _perception_eval。
# 既存の推論結果・データセットには触れない。出力 = out/d21_effect_2026-10-01/。投入: sbatch server_sde/d21_effect.sbatch
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
CK2=$R/outdoor_siren_v2final_dev2_causal_ft/checkpoints/last.ckpt     # dev2 e149（9/23 の selfcausal と同じ ckpt）
[ -f "$CK2" ] || { echo "dev2 の ckpt が無い: $CK2"; exit 2; }
D3=$R/infer_v2finaldev3ft_e149_selfcausal/val_all_causal.csv            # dev3 e149 の自分の val 5,100 本（kickfar 込み）
D2=$R/infer_v2finaldev2ft_e149_selfcausal/val_all_causal.csv            # dev2 e149 の自分の val 4,800 本（kickfar 無し）
[ -f "$D3" ] && [ -f "$D2" ] || { echo "dev2/dev3 の selfcausal の予測が無い"; exit 2; }
ROOTDIR=$PS/datasets_v2final
OUT=$E2E/out/d21_effect_2026-10-01; mkdir -p $OUT
JOBDIR=$OUT/job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $JOBDIR/
cp $E2E/server_sde/_prediction_integrity.py $JOBDIR/
cp $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
echo "job=$SLURM_JOB_ID node=$(hostname) $(date)"
sha256sum $CK2 $INFER $JOBDIR/_prediction_integrity.py $JOBDIR/hysteresis_patch.py $PS/height_patch.py $PS/sde_patch.py $D3 $D2 | tee $OUT/sha256_job${SLURM_JOB_ID}.txt

# 1) dev2 で kickfar の val 300 本を推論（設定は v2final_dev_causal.sh の selfcausal と同じ）
cd $PS
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0
export HEIGHT_TABLE=$E2E/out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
export CAUSAL_PREFIX=val_syn_ CAUSAL_DSDIR=$ROOTDIR CAUSAL_CKPT=$CK2
export CAUSAL_FOA=$ROOTDIR/outdoor_siren_v12/foa
export CAUSAL_GLOB='val_syn_clip3[01]*.flac'       # kickfar = clip 30001〜31750 のうち val（300 本）
export CAUSAL_EXPECTED_CLIPS=300
export CAUSAL_OUT=infer_v2finaldev2ft_e149_selfcausal_kickfar
# 既存の推論を使い回すのは done.json の ckpt_sha256 が CK2 と一致し、n_clips=300、csv の sha が done.json の sha と一致するときだけ（敵対レビュー D・R3、監査 A・F10）
CK2_SHA=$(sha256sum $CK2 | cut -c1-64)
[ ${#CK2_SHA} -eq 64 ] || { echo "ckpt の sha が取れない"; exit 2; }
REUSE=0
if [ -f "$R/$CAUSAL_OUT/val_all_causal.csv" ] && [ -f "$R/$CAUSAL_OUT/val_all_causal.done.json" ]; then
  CSV_SHA=$(sha256sum "$R/$CAUSAL_OUT/val_all_causal.csv" | cut -c1-64)
  .venv/bin/python - "$R/$CAUSAL_OUT/val_all_causal.done.json" "$CK2_SHA" "$CSV_SHA" <<'EOF' && REUSE=1
import json, sys
d = json.load(open(sys.argv[1]))
ok = d.get("ckpt_sha256") == sys.argv[2] and int(d.get("n_clips", -1)) == 300 and d.get("sha256") == sys.argv[3]
print("reuse check:", ok, d.get("ckpt_sha256", "")[:12], d.get("n_clips"), (d.get("sha256") or "")[:12])
sys.exit(0 if ok else 1)
EOF
fi
if [ $REUSE -eq 1 ]; then echo "既存の kickfar 推論を使う（done.json の ckpt sha が CK2 と一致）"; else
  rm -rf "$R/$CAUSAL_OUT"
  .venv/bin/python $INFER || { echo "dev2 の kickfar 推論に失敗。中止"; exit 4; }
fi
unset CAUSAL_FOA CAUSAL_GLOB
D2K=$R/infer_v2finaldev2ft_e149_selfcausal_kickfar/val_all_causal.csv
[ -f "$D2K" ] || { echo "kickfar の予測が無い: $D2K"; exit 4; }

# 2) 名簿 2 つ（kickfar val 300・v12kick val 450）を plan から作る
cd $E2E
P2=out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv
G2=out/dataset_outdoor_siren_v2final/metadata_dist
rm -f $OUT/manifest_kickfar_val.csv $OUT/manifest_v12kick_val.csv      # 前回の名簿を消してから作る（監査 A・F11）
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
# 前回の採点の成果物を消してから走らせる（残っていると 1 段だけ失敗しても成功に見える・敵対レビュー D・R3）
rm -f $OUT/unified_*.md $OUT/perception_*.md $OUT/perception_*.json $OUT/_log_*.txt $OUT/run_meta_dev2_kickfar.json
fail=0
# 3) 採点（kickfar: dev2 は学習外・dev3 は学習内。v12kick: 両方学習内）
$PY scripts/_score_unified.py $OUT/unified_kickfar.md --strict --raw --plan $MK --meta $G2 --split-by motion dev2_e149=$D2K dev3_e149=$D3 > $OUT/_log_unified_kickfar.txt 2>&1 || { echo FAIL_unified_kickfar; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_kickfar.md --plan $MK --meta $G2 --json $OUT/perception_kickfar.json --strict dev2_e149=$D2K dev3_e149=$D3 > $OUT/_log_perception_kickfar.txt 2>&1 || { echo FAIL_perception_kickfar; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_v12kick.md --strict --raw --plan $MV --meta $G2 --split-by motion dev2_e149=$D2 dev3_e149=$D3 > $OUT/_log_unified_v12kick.txt 2>&1 || { echo FAIL_unified_v12kick; fail=1; }
$PY scripts/_perception_eval.py $OUT/perception_v12kick.md --plan $MV --meta $G2 --json $OUT/perception_v12kick.json --strict dev2_e149=$D2 dev3_e149=$D3 > $OUT/_log_perception_v12kick.txt 2>&1 || { echo FAIL_perception_v12kick; fail=1; }
cp $R/infer_v2finaldev2ft_e149_selfcausal_kickfar/run_meta.json $OUT/run_meta_dev2_kickfar.json 2>/dev/null || true
# 採点のコード・plan・名簿・正解フォルダの指紋も台帳に足す（敵対レビュー D・R4）。GPU の種類は nvidia-smi から
{ sha256sum scripts/_score_unified.py scripts/_perception_eval.py scripts/_notify_v43_attrib.py scripts/step12_notify_v9c_multitrack.py $P2 $MK $MV $D2K || echo "LEDGER_SHA_FAILED";
  ( cd $G2 && find . -name '*.csv' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | sed "s|-|metadata_dist の中身（全 csv の sha の sha）|" );
  echo "git_head $(git rev-parse HEAD 2>/dev/null)"; echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } >> $OUT/sha256_job${SLURM_JOB_ID}.txt
grep -q LEDGER_SHA_FAILED $OUT/sha256_job${SLURM_JOB_ID}.txt && fail=1
[ $fail -eq 0 ] || { echo "D21_EFFECT_FAILED fail=$fail $(date)"; exit 1; }
tar czf $E2E/out/d21_effect_$(date +%Y%m%d).tgz -C $E2E out/d21_effect_2026-10-01
echo "D21_EFFECT_DONE fail=$fail $(date)"
