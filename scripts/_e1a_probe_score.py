# -*- coding: utf-8 -*-
"""E1a採点: 素(run3 val) vs 大型車低周波デルタ版のペア比較。

事前登録ゲート(md/design/E1a_低周波強化_設計_2026-08-05.md 4節):
  H1: 車の至近≤5m距離中央誤差が -0.03m以上改善
  H2: 安全車抑制率の悪化 -1.0pt以内
  H3: 警告音クラスの検出率±0.3pt・方向誤差±0.3°以内（負の統制）
集計ロジックは _e0_probe_score.py と同一（条件だけ差し替え）。出力= out/e1a_heavy_probe/
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

e0s.OUT = ROOT / "out" / "e1a_heavy_probe"
e0s.OUT.mkdir(parents=True, exist_ok=True)
e0s.CONDS = {
    "base": ROOT / "out" / "predictions_v11sde_run3" / "val_all.csv",
    "e1aheavy": ROOT / "out" / "predictions_v11sde_run3_e1aheavy" / "val_all.csv",
}


def main() -> None:
    import numpy as np
    import subprocess
    recalls = {}
    for cond, pred_path in e0s.CONDS.items():
        assert pred_path.exists(), pred_path
        recalls[cond] = e0s.class_table(e0s.load_pred(pred_path))
        print(f"[{cond}] recall done", flush=True)

    lines = ["# E1a: 大型車低周波デルタのペア比較（SDE run3 / val 1,200本）", "",
             "## クラス別 可聴(SNR≥0dB)検出率 / 方向誤差中央値", "",
             "| クラス | 素 検出率 | e1a 検出率 | 素 方向° | e1a 方向° |",
             "| --- | --- | --- | --- | --- |"]
    for c in range(6):
        row = [e0s.CLASSES_JA[c]]
        for cond in e0s.CONDS:
            s = recalls[cond][c]
            row.append(f"{s['tp']/s['n']:.2%}" if s["n"] else "n/a")
        for cond in e0s.CONDS:
            e = recalls[cond][c]["errs"]
            row.append(f"{np.median(e):.2f}" if e else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    py = sys.executable
    for cond, pred_path in e0s.CONDS.items():
        sub = e0s.OUT / cond
        sub.mkdir(exist_ok=True)
        for script in ["_score_sde_dist.py", "step12_notify_v3.py"]:
            r = subprocess.run([py, str(ROOT / "scripts" / script),
                                str(pred_path), str(sub)],
                               capture_output=True)
            print(f"[{cond}] {script} rc={r.returncode}", flush=True)

    # H3 判定材料（警告音クラス=車以外）
    lines += ["", "## H3 負の統制（警告音クラスのペア差）", ""]
    for c in [0, 1, 2, 3, 5]:
        rb, re_ = recalls["base"][c], recalls["e1aheavy"][c]
        d_rec = (re_["tp"]/re_["n"] - rb["tp"]/rb["n"]) * 100 if rb["n"] else 0
        d_le = (float(np.median(re_["errs"])) - float(np.median(rb["errs"]))
                if rb["errs"] and re_["errs"] else 0)
        ok = abs(d_rec) <= 0.3 and abs(d_le) <= 0.3
        lines.append(f"- {e0s.CLASSES_JA[c]}: 検出{d_rec:+.2f}pt / "
                     f"方向{d_le:+.2f}° {'✅' if ok else '⚠️'}")
    lines += ["", "H1(至近距離)・H2(抑制率)は base/ と e1aheavy/ の "
              "dist_score.md / notify_v3_val.md を突合して判定を追記する。", ""]
    (e0s.OUT / "e1a_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", e0s.OUT / "e1a_report.md", flush=True)


if __name__ == "__main__":
    main()
