cd ~/research/PSELDNet/PSELDNets
DS=outdoor_siren_probe192
CKPT=$HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_run2/checkpoints/epoch_094.ckpt
echo "ckpt exists: $(ls -l $CKPT 2>/dev/null | awk '{print $5}') bytes"
echo "=== infer via srun (a100_1g MIG) ==="
srun --qos=inter -p a100_1g --gres=gpu:1g.10gb:1 --time=00:15:00 \
  .venv/bin/python src/infer.py experiment=${DS} mode=test \
    ckpt_path="$CKPT" model.kwargs.pretrained_path=null \
    experiment_name=infer_probe192 paths.log_dir="$HOME/PSELDNets_logs" 2>&1 | tail -18
echo "=== collect submissions ==="
SUB="$HOME/PSELDNets_logs/${DS}/runs/infer_probe192/submissions"
echo "submissions csv: $(ls $SUB/*.csv 2>/dev/null | wc -l)"
.venv/bin/python - "$SUB" <<'PY'
import glob, os, sys
sub = sys.argv[1]
out = []
for p in sorted(glob.glob(f"{sub}/*.csv")):
    stem = os.path.basename(p)[:-4]
    for line in open(p):
        if line.strip():
            out.append(f"{stem},{line.strip()}")
dst = os.path.expanduser("~/probe192_all.csv")
open(dst, "w").write("\n".join(out))
print("wrote", dst, len(out), "lines")
PY
echo "=== INFER DONE ==="
