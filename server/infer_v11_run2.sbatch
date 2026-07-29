#!/bin/bash
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
# 推論6セット一括（Colabノートブック セル12の移植）。run2のbest ckptで
# val(v11) + v10評価5種（交差点/プローブ/6シナリオ/交通量/幻覚）を推論し、
# セットごとに予測CSVを1本へ連結して ~/PSELDNets_data/ に保存する。
# ファイル名規約は run1 と同じ: infer_<EXP_NAME>_<tag>_all.csv
# 投入例:
#   sbatch -J v11r2infer -p a100_3g --gres=gpu:3g.40gb:1 --qos=low \
#       ~/research/outdoor_seld_e2e/server/infer_v11_run2.sbatch
set -euo pipefail

cd "$HOME/research/PSELDNet/PSELDNets"
source .venv/bin/activate

EXP_NAME=outdoor_siren_v11_run2
LOGDIR="$HOME/PSELDNets_logs"
OUTDIR="$HOME/PSELDNets_data"

# best ckpt（save_top_k=1構成のためepoch_*.ckptは1本＝それがbest。前提が崩れたら中止=fail-closed、Sol監査2-#2）
BEST=$(ls "$LOGDIR/outdoor_siren_v11/runs/$EXP_NAME/checkpoints/"epoch_*.ckpt | sort | tail -1)
N_CKPT=$(ls "$LOGDIR/outdoor_siren_v11/runs/$EXP_NAME/checkpoints/"epoch_*.ckpt | wc -l)
echo "ジョブID: $SLURM_JOB_ID / ノード: $(hostname)"
echo "best_ckpt: $BEST (epoch_*.ckpt本数=$N_CKPT)"
if [[ "$N_CKPT" -ne 1 ]]; then
    echo "ERROR: epoch_*.ckpt が${N_CKPT}本あり、ファイル名順のbest選定は保証されない。中止"
    exit 1
fi
sha256sum "$BEST"

JOBS=(
    "outdoor_siren_v11_valinfer val"
    "outdoor_siren_v10_scenario scenario"
    "outdoor_siren_v10_probe probe"
    "outdoor_siren_v10_scn2 scn2"
    "outdoor_siren_v10_v10a v10a"
    "outdoor_siren_v10_halluc halluc"
)
for job in "${JOBS[@]}"; do
    read -r experiment short <<< "$job"
    exp="infer_${EXP_NAME}_${short}"
    echo "=== 推論: $experiment -> $exp ==="
    python src/infer.py "experiment=${experiment}" mode=test \
        "ckpt_path=${BEST}" model.kwargs.pretrained_path=null \
        "experiment_name=${exp}" "paths.log_dir=${LOGDIR}"
    task="${experiment%_*}"
    python - "$LOGDIR" "$task" "$exp" "$OUTDIR" <<'PY'
import glob, os, sys
logdir, task, exp, outdir = sys.argv[1:5]
sub = f'{logdir}/{task}/runs/{exp}/submissions'
out_lines = []
for p in sorted(glob.glob(f'{sub}/*.csv')):
    stem = os.path.basename(p)[:-4]
    for line in open(p):
        if line.strip():
            out_lines.append(f'{stem},{line.strip()}')
assert out_lines, f'submissionsが空: {sub}'
out = f'{outdir}/{exp}_all.csv'
open(out, 'w').write('\n'.join(out_lines))
print('wrote', out, len(out_lines), 'lines')
PY
done
echo "=== 推論6セット完了 ==="
ls -la "$OUTDIR"/infer_${EXP_NAME}_*_all.csv
