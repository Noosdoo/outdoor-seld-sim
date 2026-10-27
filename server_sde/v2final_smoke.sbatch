#!/bin/bash
#SBATCH -J v2fsmoke
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=low
#SBATCH -t 01:00:00
#SBATCH -o v2fsmoke-%j.out
# 確定評価 v2 の最終学習データの**試し描画**（2026-09-14・新場面 D15/D17/D18 を生成器に組み込んだ確認）。
# 本番の出力先（out/dataset_outdoor_siren_v2final/）には書かず、out/dataset_outdoor_siren_v2final_smoke/ に少数の clip だけ描く。
# params が draft でも動く（本番の生成は v2final_gen.sbatch が status=fixed を要求する）。
# 使い方: sbatch --export=ALL,CLIPS="train_syn_clip20001 train_syn_clip24003"（空白区切り。--export はカンマで変数を区切る） server_sde/v2final_smoke.sbatch
set -exo pipefail
cd ~/research/outdoor_seld_e2e
PS=~/research/PSELDNet/PSELDNets
PY=$PS/.venv/bin/python
export PYTHONPATH=scripts:src
CLIPS=${CLIPS:?CLIPS（カンマ区切りの clip_id）を渡す}
source ~/research/outdoor_seld_e2e/server_sde/_baseline_guard.sh
guard_record
sha256sum md/design/v2final_params.json scripts/step10_v2final_plan.py scripts/_run_v2final_gen.py scripts/step11_v21_render.py scripts/step11_v19_render.py out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv | tee v2fsmoke_sha256_job${SLURM_JOB_ID}.txt
$PY scripts/_run_v2final_gen.py --smoke "$CLIPS"
DS=out/dataset_outdoor_siren_v2final_smoke
echo "foa: $(ls $DS/foa | wc -l) / metadata: $(ls $DS/metadata | wc -l) / masks: $(ls $DS/masks | wc -l) / work: $(ls $DS/work | wc -l)"
# 場面点検（規則違反 0 を期待）と、clip の完全性（v19 の生成ドライバの検査を流用）
$PY scripts/_data_grammar_audit.py --ds $DS --out $DS/grammar_audit.md | tail -5
$PY - <<'EOF'
import sys, json
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import _run_v19_gen as g
from pathlib import Path
ds = Path("out/dataset_outdoor_siren_v2final_smoke")
g.m9.DS = ds
ok = bad = 0
for p in sorted((ds / "work").iterdir()):
    scene = json.loads((p / "scene.json").read_text(encoding="utf-8"))
    r = scene["row"]
    complete = g.clip_complete(p.name)
    print(f"{p.name}: scenario={r['scenario']} rain={r.get('rain')!r} motion={r['motion']} mic_z={r['mic_z']} sources={len(scene['sources'])} complete={complete}")
    ok += complete; bad += (not complete)
print(f"complete {ok} / incomplete {bad}")
sys.exit(0 if bad == 0 else 1)
EOF
guard_verify || exit 1
echo V2FINAL_SMOKE_DONE
