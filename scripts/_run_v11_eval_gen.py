# -*- coding: utf-8 -*-
"""v11評価拡張 全量生成ドライバ（3,246本、バックグラウンド実行用）。

使い方:
  python scripts/_run_v11_eval_gen.py --rows 0-1622 --skip-inspect   # 並列その1
  python scripts/_run_v11_eval_gen.py --rows 1623-3245 --skip-inspect # 並列その2
  python scripts/step11_v11_eval_render.py inspect                    # 全完了後に1回
セット順は step11_v11_eval_render.SETS の固定順（rowsは全セット連結のindex）。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v11_eval_render as mev  # noqa: E402
m9 = mev.m9


def main():
    assert m9.V10_1 and m9.V10_1B, "V10_1/V10_1Bフラグ継承が壊れている"
    assert m9.DS_NAME == "outdoor_siren_v11_eval"
    assert m9.sample_scene_v9 is mev.sample_scene_v11eval
    assert m9.generate_clip is mev.generate_clip_v11eval
    rows_all = []
    for which in mev.SETS:
        rows_all += m9.load_plan(which)
    assert len(rows_all) == 3246, len(rows_all)
    lo, hi = 0, len(rows_all) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    rows = rows_all[lo:hi + 1]
    t0 = time.time()
    print(f"=== v11eval gen rows {lo}-{hi} ({len(rows)} clips) ===", flush=True)
    for i, row in enumerate(rows):
        m9.generate_clip(row)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f"[{lo}-{hi}] {i + 1}/{len(rows)} {el:.0f}s elapsed, "
                  f"{el / (i + 1):.2f}s/clip", flush=True)
    print(f"\n=== generation done: {len(rows)} clips, {time.time() - t0:.0f}s ===",
          flush=True)
    if "--skip-inspect" in sys.argv:
        print("=== skip inspect (run separately after all ranges finish) ===",
              flush=True)
        return
    fail = m9.inspect_all()
    print(f"\n=== v11eval ALL DONE (inspect fail={fail}) ===", flush=True)


if __name__ == "__main__":
    main()
