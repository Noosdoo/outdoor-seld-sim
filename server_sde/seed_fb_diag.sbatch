#!/bin/bash
#SBATCH -J seedfbdiag
#SBATCH -p a100_1g
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=low
#SBATCH -t 04:00:00
#SBATCH -o seedfbdiag-%j.out
# 種 2026（dev2s2・dev3s2）の交差 val の前後取り違え（約 2%・後方の多重音源）の原因さがし（2026-10-05・本人「原因を解明してほしい」）。
#   1) 因果 ft の 8 保存点（e019〜e149）の交差 val 予測（job 5301／5346 で済み）に知覚層の評価を掛け、取り違えがいつ現れるかを見る（対照 = dev3s1 の 8 点）。
#   2) 因果 ft の起点になった基準学習の ckpt（dev3s2・dev2s2 = epoch_084、対照 dev3 = epoch_089・dev3s1 = epoch_084）で交差 val を推論し、
#      ft の前から崩れているかを見る。推論の設定は v2final_dev_causal.sh の v12causal と同じ（高さ表なし・1.5 m 固定・KMIN40・REPEAT2）。
# 既存の推論・学習には触れない。出力 = out/seed_fb_diag_2026-10-05/。投入: sbatch server_sde/seed_fb_diag.sbatch
set -x
E2E=$HOME/research/outdoor_seld_e2e
PS=$HOME/research/PSELDNet/PSELDNets
export PS E2E PYTHONIOENCODING=utf-8
PY=$PS/.venv/bin/python
R=$HOME/PSELDNets_logs/outdoor_siren_v12/runs
OUT=$E2E/out/seed_fb_diag_2026-10-05; mkdir -p $OUT
JOBDIR=$OUT/job${SLURM_JOB_ID}; mkdir -p $JOBDIR
cp $E2E/server_sde/_causal_infer_v17.py $E2E/server_sde/_prediction_integrity.py $E2E/server_sde/hysteresis_patch.py $JOBDIR/
INFER=$JOBDIR/_causal_infer_v17.py
M12=$E2E/out/_eval_aux_v12/manifest_val_syn.csv
G12=$E2E/out/_eval_aux_v12/metadata_dist_h
cd $E2E
fail=0
# 1) ft の 8 点
for t in dev3s2 dev2s2 dev3s1; do
  ROWS=""
  for E in 019 039 059 079 099 119 139 149; do
    f=$R/infer_v2final${t}ft_e${E}_v12causal/val_all_causal.csv
    [ -f "$f" ] || { echo "無い: $f"; exit 2; }
    ROWS="$ROWS ${t}_e${E}=$f"
  done
  $PY scripts/_perception_eval.py $OUT/perception_cross_${t}_epochs.md --plan $M12 --meta $G12 --json $OUT/perception_cross_${t}_epochs.json --strict $ROWS > $OUT/_log_perception_${t}.txt 2>&1 || { echo FAIL_perception_$t; fail=1; }
done
# 2) 基準学習の ckpt（ft の起点）で交差 val を推論
cd $PS
export SDE_W_DIST=3.0 SDE_DIST_MODE=log CAUSAL_KMIN=40 CAUSAL_REPEAT=2 CAUSAL_SEED=0
export HEIGHT_COND=1 HEIGHT_JITTER=0 HEIGHT_DEFAULT=1.5 HEIGHT_OFFSET=0 HEIGHT_TABLE=""
export CAUSAL_SED_TH=0.7 CAUSAL_SED_STAY=0.3
export CAUSAL_PREFIX=val_syn_ CAUSAL_DSDIR=$PS/datasets_v12
unset CAUSAL_FOA CAUSAL_GLOB
BROWS=""
for t in dev3s2:epoch_084 dev2s2:epoch_084 dev3:epoch_089 dev3s1:epoch_084; do
  tag=${t%%:*}; ck=${t##*:}
  CK=$R/outdoor_siren_v2final_${tag}_sde_w3/checkpoints/$ck.ckpt
  [ -f "$CK" ] || { echo "基準の ckpt が無い: $CK"; exit 2; }
  export CAUSAL_CKPT=$CK CAUSAL_OUT=infer_v2final${tag}base_${ck/epoch_/e}_v12causal
  # RF02（Astra 再監査 10/6）: 完了印の存在だけで再利用しない。csv の sha・ckpt の sha・本数（1,800）が done.json と合うときだけ飛ばす。
  #   done.json があるのに合わなければ消さずに止める（別の走りの成果物の上書きを防ぐ）。done.json が無ければ消して作り直す。
  CKSHA=$(sha256sum $CK | cut -c1-64)
  if [ -f "$R/$CAUSAL_OUT/val_all_causal.done.json" ]; then
    if .venv/bin/python - "$R/$CAUSAL_OUT" "$CKSHA" 1800 <<'PY'
import hashlib, json, sys
from pathlib import Path
d, ck, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
done = json.loads((d / "val_all_causal.done.json").read_text(encoding="utf-8"))
csv = d / "val_all_causal.csv"
ok = (csv.is_file() and csv.stat().st_size > 0 and done.get("ckpt_sha256") == ck and done.get("n_clips") == n
      and done.get("sha256") == hashlib.sha256(csv.read_bytes()).hexdigest())
sys.exit(0 if ok else 1)
PY
    then echo "推論済み（ckpt sha・csv sha・本数が一致）: $CAUSAL_OUT"
    else echo "停止: $CAUSAL_OUT に done.json があるが ckpt／csv／本数が合わない（手で確かめてから消す）"; exit 6; fi
  else
    rm -rf "$R/$CAUSAL_OUT"
    .venv/bin/python $INFER || { echo "基準 ckpt の推論に失敗: $tag"; exit 4; }
  fi
  sha256sum $CK >> $OUT/sha256_job${SLURM_JOB_ID}.txt
  BROWS="$BROWS ${tag}_base_${ck/epoch_/e}=$R/$CAUSAL_OUT/val_all_causal.csv"
done
cd $E2E
$PY scripts/_perception_eval.py $OUT/perception_cross_base.md --plan $M12 --meta $G12 --json $OUT/perception_cross_base.json --strict $BROWS > $OUT/_log_perception_base.txt 2>&1 || { echo FAIL_perception_base; fail=1; }
$PY scripts/_score_unified.py $OUT/unified_cross_base.md --strict --raw --plan $M12 --meta $G12 $BROWS > $OUT/_log_unified_base.txt 2>&1 || { echo FAIL_unified_base; fail=1; }
sha256sum $INFER scripts/_perception_eval.py scripts/_score_unified.py $M12 >> $OUT/sha256_job${SLURM_JOB_ID}.txt
echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" >> $OUT/sha256_job${SLURM_JOB_ID}.txt
[ $fail -eq 0 ] || { echo "SEED_FB_DIAG_FAILED fail=$fail"; exit 1; }
echo "SEED_FB_DIAG_DONE $(date)"
