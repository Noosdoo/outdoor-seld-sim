# -*- coding: utf-8 -*-
"""B02 の切り分け（2026-09-24）: 手元とサーバで発火数が 1 つ違う val_syn_clip00845 の発火の中身を出す（規則 v4.3＋hold の発火 = fires_of）。
使い方: python scripts/_b02_clip845.py --pred <pred.csv> [--clip val_syn_clip00845]
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_score_unified", ROOT / "scripts" / "_score_unified.py")
S = importlib.util.module_from_spec(_spec); sys.modules["_score_unified"] = S; _spec.loader.exec_module(S)


def main() -> int:
    argv = sys.argv[1:]
    def arg(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    pred_p = Path(arg("--pred")); clip = arg("--clip", "val_syn_clip00845")
    pred = S.load_pred_rows(pred_p, horiz=False, strict=False)
    frames = pred[clip]
    fires = S.fires_of(frames)
    print("env", platform.python_version(), np.__version__, platform.system())
    print("fires", len(fires))
    for f in fires:
        print("  frame=%d az=%.3f tier=%s d=%.6f cls=%d" % (f[0], f[1], f[2], f[3], f[4]))
    # 規則の内部（run_rule3）を直接呼んで、車クラスの結果を細かく出す
    frames_dist = {k: [(c, az, d) for c, az, el, d in v if np.isfinite(d)] for k, v in frames.items()}
    res = S.V43.run_rule3({"x": frames_dist}, S.C43, nframes=100).get("x", {})
    for cls, lst in res.items():
        print("  rule3 cls", cls, [(j, round(az, 3), tier, round(d, 6)) for j, az, tier, d in lst])
    # 距離クラスの車の予測（frame ごとの d・az）を 40〜99 で並べる（境界ぎりぎりの値を目で見る）
    for k in range(40, 100):
        row = [(c, round(az, 2), round(d, 6)) for c, az, el, d in frames.get(k, []) if c in S.DIST_CLS]
        if row:
            print("  k=%d %s" % (k, row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
