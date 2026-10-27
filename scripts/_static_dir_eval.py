# -*- coding: utf-8 -*-
"""static（停車中のサイレン・クラクション）の方位誤差と検出の事後分析（2026-09-09・採否には使わない探索）。

v19static の主指標（到達率）は天井と hold の上限で差が出なかった（`md/results/v19_結果_新4場面_2026-09-08.md`）。
ここでは正解の区間（クラスごと・frame≥minframe）に対して、そのフレームに同じクラスの予測があるか（検出率）と、
予測の方位と正解の方位の差（方位誤差）をクリップごとに取り、クリップの中央値の分布を v18 と v19static で比べる。
使い方:
  python scripts/_static_dir_eval.py <出力md> --plan <assignment_v19_static.csv> --meta <GT dir> [--minframe 40] <ラベル>=<csv> ...
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("score_unified", ROOT / "scripts/_score_unified.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
CLASSES = {0: "siren", 1: "horn"}


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def gt_frames(meta: Path, clip: str, cls: int, minframe: int):
    """frame -> az（そのクラスの正解・frame≥minframe）"""
    out = {}
    f = meta / f"{clip}.csv"
    if not f.exists():
        return out
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) >= 4 and int(g[1]) == cls and int(g[0]) >= minframe:
            out[int(g[0])] = float(g[3])
    return out


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    meta = ROOT / arg(argv, "--meta")
    minframe = int(arg(argv, "--minframe", "40"))
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]
    clips = [r["clip_id"] for r in csv.DictReader(open(plan, encoding="utf-8")) if r.get("split") == "fold2"]
    gts = {cls: {c: gt_frames(meta, c, cls, minframe) for c in clips} for cls in CLASSES}
    lines = [f"# static の方位誤差と検出（事後分析・採否には使わない）— frame≥{minframe}・fold2 {len(clips)} 本", "",
             "- 検出率 = 正解フレームのうち同じクラスの予測があるフレームの割合（クリップ内）。方位誤差 = 検出したフレームの |予測方位 − 正解方位| の中央値（クリップ内）。",
             "- 表はクリップの中央値と 90 パーセンタイル（クリップ分布の裾）。方位の誤差は通知の帰属（±30 度）より細かい差を見るため。", "",
             "| モデル | クラス | 本（正解あり） | 検出率 中央値 | 検出率 10%tile | 方位誤差 中央値 [度] | 方位誤差 90%tile [度] | 方位誤差 >30 度の本 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, path in items:
        pred = S.load_pred_rows(ROOT / path, horiz=False)
        for cls, name in CLASSES.items():
            det_rates, az_meds = [], []
            for c in clips:
                g = gts[cls][c]
                if not g:
                    continue
                frames = pred.get(c, {})
                hits, errs = 0, []
                for k, az in g.items():
                    pazs = [paz for (pc, paz, pel, pd) in frames.get(k, []) if pc == cls]
                    if pazs:
                        hits += 1
                        errs.append(min(dang(az, p) for p in pazs))
                det_rates.append(hits / len(g))
                az_meds.append(float(np.median(errs)) if errs else float("nan"))
            dr = np.array(det_rates)
            am = np.array(az_meds, dtype=float)
            ok = np.isfinite(am)
            lines.append(f"| {label} | {name} | {len(dr)} | {100 * np.median(dr):.1f}% | {100 * np.percentile(dr, 10):.1f}% | "
                         f"{np.median(am[ok]):.1f} | {np.percentile(am[ok], 90):.1f} | {int(np.sum(am[ok] > 30))} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
