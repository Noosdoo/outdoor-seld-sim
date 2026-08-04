# -*- coding: utf-8 -*-
"""E1-full採点: 4条件比較（run3/E1fullモデル × plain/heavy val）。

事前登録ゲート(md/design/E1full_設計_2026-08-05.md 3節):
  G1: E1fullモデルのheavy-val 車至近≤5m中央誤差が run3比 -0.03m以上改善
  G2: E1fullモデルのplain-val SELD ≤ 0.075（学習ログで確認済みの値を転記）
  G3: E1fullモデルのplain-val 至近中央誤差 悪化+0.02m以内（0.24→0.26m以内）
集計は _e0_probe_score.py の型を流用。出力= out/e1full_probe/（新規）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "e0s", ROOT / "scripts" / "_e0_probe_score.py")
e0s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0s)

e0s.OUT = ROOT / "out" / "e1full_probe"
e0s.OUT.mkdir(parents=True, exist_ok=True)
e0s.CONDS = {
    "run3_plain": ROOT / "out" / "predictions_v11sde_run3" / "val_all.csv",
    "run3_heavy": ROOT / "out" / "predictions_v11sde_run3_e1aheavy" / "val_all.csv",
    "e1f_plain": ROOT / "out" / "predictions_v12heavy_plain" / "val_all.csv",
    "e1f_heavy": ROOT / "out" / "predictions_v12heavy_heavy" / "val_all.csv",
}


def main() -> None:
    import numpy as np
    import subprocess
    recalls = {}
    for cond, p in e0s.CONDS.items():
        assert p.exists(), p
        recalls[cond] = e0s.class_table(e0s.load_pred(p))
        print(f"[{cond}] recall done", flush=True)

    lines = ["# E1-full: 4条件比較（run3 vs v12heavy学習モデル × plain/heavy val）",
             "", "## クラス別 可聴検出率（車と代表クラスのみ抜粋は下表、全量は生出力）",
             "", "| クラス | run3/plain | run3/heavy | e1f/plain | e1f/heavy |",
             "| --- | --- | --- | --- | --- |"]
    for c in range(6):
        row = [e0s.CLASSES_JA[c]]
        for cond in e0s.CONDS:
            s = recalls[cond][c]
            row.append(f"{s['tp']/s['n']:.2%}" if s["n"] else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 車の方向誤差中央値[deg]", "",
              "| run3/plain | run3/heavy | e1f/plain | e1f/heavy |", "| --- | --- | --- | --- |",
              "| " + " | ".join(
                  f"{np.median(recalls[cond][e0s.CAR]['errs']):.2f}"
                  for cond in e0s.CONDS) + " |"]

    py = sys.executable
    for cond, p in e0s.CONDS.items():
        sub = e0s.OUT / cond
        sub.mkdir(exist_ok=True)
        for script in ["_score_sde_dist.py", "step12_notify_v3.py"]:
            r = subprocess.run([py, str(ROOT / "scripts" / script), str(p), str(sub)],
                               capture_output=True)
            print(f"[{cond}] {script} rc={r.returncode}", flush=True)

    lines += ["", "G1〜G3判定は各条件の dist_score.md（至近側の行）を突合して追記する。", ""]
    (e0s.OUT / "e1full_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", e0s.OUT / "e1full_report.md", flush=True)


if __name__ == "__main__":
    main()
