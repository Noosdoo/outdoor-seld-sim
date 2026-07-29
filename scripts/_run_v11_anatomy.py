# -*- coding: utf-8 -*-
"""v11 run1 の本解剖ランナー(step13 を v11 パスで実行 + P19用クラス別まとめ表)。

step13_v9_anatomy.py は無改変で importlib 読み込み(v10.2解剖と同じrunner方式)。
追加で、ゼミP19用の「クラス別 可聴recall / 方向誤差中央値 / substitution」を集計する。
入力: out/predictions_v11/val_all.csv(run1) + dataset_outdoor_siren_v11(metadata/masks/work)
出力: out/v11_anatomy_2026-07-30.md + out/figures_v11_analysis/ + P19表を標準出力
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "step13", ROOT / "scripts" / "step13_v9_anatomy.py")
m13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m13)

m13.DS = ROOT / "out" / "dataset_outdoor_siren_v11"
m13.PRED = ROOT / "out" / "predictions_v11" / "val_all.csv"
m13.FIG = ROOT / "out" / "figures_v11_analysis"
m13.MD = ROOT / "out" / "v11_anatomy_2026-07-30.md"

CLASSES_JA = ["サイレン", "クラクション", "バック音", "自転車ベル", "車", "踏切"]


def p19_table() -> None:
    """P19用: クラス別 可聴(SNR>=0dB)recall・方向誤差中央値・substitution。"""
    data = m13.load_all()
    stats = {c: {"tp": 0, "n": 0, "errs": []} for c in range(6)}
    sub_frames = 0
    gt_frames = 0
    for clip, (pr, paz, gt, mask, scene) in data.items():
        for k, evs in gt.items():
            pset = pr.get(k, set())
            gset = set(evs)
            gt_frames += len(gset)
            fn = len(gset - pset)
            fp = len(pset - gset)
            sub_frames += min(fn, fp)
            for c, (gaz, gel) in evs.items():
                if mask.get((k, c), 99.0) < 0.0:     # 可聴ゲート(SNR>=0dB)のみ
                    continue
                stats[c]["n"] += 1
                if c in pset:
                    stats[c]["tp"] += 1
                    a, e = paz[(k, c)]
                    stats[c]["errs"].append(m13.ang_err(gaz, gel, a, e))
    lines = ["", "## P19差し替え表(v11 run1・val 1,200本・可聴フレーム SNR>=0dB)",
             "", "| クラス | 検出率(可聴) | 方向誤差(中央値) |", "| --- | --- | --- |"]
    for c in range(6):
        s = stats[c]
        rec = s["tp"] / s["n"] if s["n"] else float("nan")
        le = np.median(s["errs"]) if s["errs"] else float("nan")
        lines.append(f"| {CLASSES_JA[c]} | {rec:.1%} (n={s['n']:,}) | {le:.1f}° |")
    lines += ["",
              f"- 取り違え(同一フレームで見逃しと誤検出が同時=min(FN,FP)): "
              f"{sub_frames}/{gt_frames:,}正解フレーム ({sub_frames/gt_frames:.3%})",
              "- 注: 同一クラス2音源の同時フレームはクラス単位に集約(v10.2解剖と同じ扱い)", ""]
    out = "\n".join(lines)
    print(out)
    with open(m13.MD, "a", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    m13.main()
    p19_table()
