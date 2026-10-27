#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH -t 03:00:00
#SBATCH --open-mode=append
#SBATCH -o v19gen-%j.out
# v19 新場面の生成（2026-09-08）。投入例: sbatch -J v19fastgen --export=ALL,SCENE=fast server_sde/v19_gen.sbatch
# plan（scripts/step10_v19_plan.py・ローカルで作って git に入れる）の当該場面 1,500 本を 24 シャードで描画 → 距離ラベル。
# 書き込み先は out/dataset_outdoor_siren_v19/ だけ。v16/v18 と基準は不変（guard）。
set -euxo pipefail    # 監査 F4: `python | tail` の失敗を隠さない
SCENE=${SCENE:?SCENE=fast|static|turn|platoon}
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
PLAN=out/dataset_outdoor_siren_v19/plan/assignment_v19_${SCENE}.csv
[ -f "$PLAN" ] || { echo "plan が無い: $PLAN（ローカルで step10_v19_plan.py → git push → サーバで git pull か scp）"; exit 3; }
sha256sum $PLAN scripts/step11_v19_render.py scripts/_run_v19_gen.py scripts/step19_dist_labels_v19.py scripts/_v19_realized_ledger.py | tee v19_${SCENE}_sha256_job${SLURM_JOB_ID}.txt
$PY scripts/_run_v19_gen.py --plan $PLAN --list
# 監査 F1: 生成前に plan 全行の軌道を検査（瞬時の反転・時刻の逆行・速度超過があれば止める）
$PY scripts/step11_v19_render.py --check-plan $PLAN | tail -3 || { echo "軌道検査 NG"; exit 1; }
N=$(( $(wc -l < $PLAN) - 1 )); SH=24
STEP=$(( (N + SH - 1) / SH ))
pids=()
for i in $(seq 0 $((SH - 1))); do
  lo=$((i * STEP)); hi=$(( (i + 1) * STEP - 1 ))
  [ $hi -ge $N ] && hi=$((N - 1))
  [ $lo -gt $hi ] && continue
  $PY scripts/_run_v19_gen.py --plan $PLAN --rows $lo-$hi > v19_${SCENE}_shard_$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo SHARD_FAILED; grep -l Traceback v19_${SCENE}_shard_*.log; exit 1; }
n_foa=$(for c in $(tail -n +2 $PLAN | cut -d, -f1); do [ -f out/dataset_outdoor_siren_v19/foa/$c.flac ] && echo 1; done | wc -l)
echo "foa for $SCENE: $n_foa / $N"
[ "$n_foa" = "$N" ] || { echo "生成が足りない"; exit 1; }
# 監査 F7: foa だけでなく metadata・masks・scene.json・foa の長さまで揃っているか（全成果物の検査）
$PY - $PLAN <<'EOF' || { echo "成果物の検査 NG"; exit 1; }
import csv, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import _run_v19_gen as G
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
bad = [r["clip_id"] for r in rows if not G.clip_complete(r["clip_id"], r)]
print(f"成果物の検査: {len(rows) - len(bad)}/{len(rows)} 本そろっている" + (f" / 不完全 {bad[:5]}" if bad else ""))
sys.exit(1 if bad else 0)
EOF
# 距離GT（水平距離・mic_z 対応）: v19 の metadata 全部を対象（他の場面の分も再計算されるが同一値）
$PY scripts/step19_dist_labels_v19.py | tail -2
ls out/dataset_outdoor_siren_v19/metadata_dist | wc -l
# 実現値の台帳（plan の n_car/n_warnings は v19static では仮の 0。実際の構成は scene.json から）
$PY scripts/_v19_realized_ledger.py --scene $SCENE | tail -3
# 場面の点検（物理的にあり得ない値が無いか）
mkdir -p out/audit_rescore
$PY scripts/_data_grammar_audit.py --ds out/dataset_outdoor_siren_v19 --out out/audit_rescore/data_grammar_v19_${SCENE}.md > out/audit_rescore/_log_grammar_v19_${SCENE}.txt 2>&1
guard_verify || exit 1
echo V19_GEN_DONE_$SCENE
