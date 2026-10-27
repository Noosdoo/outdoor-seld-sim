#!/bin/bash
# v13チェーン（v13gen→v13train→v13causal）を pro_6000 が空き次第そちらへ移す（冪等・監視から定期実行）。
#
#   bash ~/research/outdoor_seld_e2e/server_sde/_v13_move_to_pro.sh   → 1行出力（NOOP_… / MOVED_…）
#
# 方針（2026-09-02 本人「pro_6000が空き次第即移して」）:
#  - pro_6000 の空きGPUが無ければ何もしない
#  - 生成(v13gen)が a100 で実行中で残りが20%超なら: 取消→pro_6000で再投入（生成は既存クリップを
#    スキップして再開。中断で壊れた途中書きのflacは scene.json の有無で判定して消す）
#  - 学習/因果ft が a100 で**待機中**なら: 取消→pro_6000 で再投入（依存関係は張り直す）
#  - 実行中の学習/因果ftは動かさない（last.ckpt再開でも数十分は失うため）
# クラスタ制約: -p は1つだけ・GRESは型名必須・予約機能なし（2026-09-02 確認）。
set -u
cd ~/research/outdoor_seld_e2e
E2E=~/research/outdoor_seld_e2e
used=$(sinfo -h -N -p pro_6000 -O gresused | grep -o "pro_6000:[0-9]*" | head -1 | cut -d: -f2)
free=$(( 4 - ${used:-4} ))
[ "$free" -ge 1 ] || { echo "NOOP_pro_busy(used=$used)"; exit 0; }

q() { squeue -h -u "$USER" -n "$1" -o "%i %t %P" | head -1; }
read -r gid gst gpa <<< "$(q v13gen)"   || true
read -r tid tst tpa <<< "$(q v13train)" || true
read -r cid cst cpa <<< "$(q v13causal)" || true
PRO="-p pro_6000 --gres=gpu:pro_6000:1"

cleanup_partial() {   # scene.json が無い flac は途中書き → 消して再生成させる
  local n=0
  for f in out/dataset_outdoor_siren_v13/foa/*.flac; do
    [ -f "$f" ] || continue
    c=$(basename "$f" .flac)
    [ -f "out/dataset_outdoor_siren_v13/work/$c/scene.json" ] || { rm -f "$f"; n=$((n+1)); }
  done
  echo "cleanup_partial=$n" >&2
}

# ① 生成が a100 で実行中 → 残り20%超なら移す
if [ -n "${gid:-}" ] && [ "${gpa:-}" != "pro_6000" ]; then
  done_n=$(ls out/dataset_outdoor_siren_v13/foa 2>/dev/null | wc -l)
  if [ "$done_n" -lt 7200 ]; then
    scancel "$gid" ${tid:-} ${cid:-}
    sleep 8
    cleanup_partial
    G=$(sbatch --parsable $PRO server_sde/v13_gen.sbatch)
    T=$(sbatch --parsable $PRO --dependency=afterok:$G server_sde/v13_train.sbatch)
    C=$(sbatch --parsable $PRO --dependency=afterok:$T server_sde/v13_causal.sbatch)
    echo "MOVED_gen_to_pro(done=$done_n) gen=$G train=$T causal=$C"
  else
    echo "NOOP_gen_almost_done($done_n/9000)"
  fi
  exit 0
fi

# ② 学習が a100 で待機中（生成は完了 or 実行中）→ 移す
if [ -n "${tid:-}" ] && [ "${tst:-}" = "PD" ] && [ "${tpa:-}" != "pro_6000" ]; then
  scancel "$tid" ${cid:-}
  sleep 5
  dep=""; [ -n "${gid:-}" ] && dep="--dependency=afterok:$gid"
  T=$(sbatch --parsable $PRO $dep server_sde/v13_train.sbatch)
  C=$(sbatch --parsable $PRO --dependency=afterok:$T server_sde/v13_causal.sbatch)
  echo "MOVED_train_to_pro train=$T causal=$C"
  exit 0
fi

# ③ 因果ftが a100 で待機中（学習は実行中 or 完了）→ 移す
if [ -n "${cid:-}" ] && [ "${cst:-}" = "PD" ] && [ "${cpa:-}" != "pro_6000" ]; then
  scancel "$cid"
  sleep 5
  dep=""; [ -n "${tid:-}" ] && dep="--dependency=afterok:$tid"
  C=$(sbatch --parsable $PRO $dep server_sde/v13_causal.sbatch)
  echo "MOVED_causal_to_pro causal=$C"
  exit 0
fi
echo "NOOP_nothing_movable(gen=${gid:-}/${gpa:-} train=${tid:-}/${tst:-}/${tpa:-} causal=${cid:-}/${cst:-}/${cpa:-})"
