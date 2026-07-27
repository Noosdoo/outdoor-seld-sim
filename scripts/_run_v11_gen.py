# -*- coding: utf-8 -*-
"""v11 core 全量生成ドライバ（バックグラウンド実行用）。

assignment_core.csv（7,200本）を生成する。生成は完全決定論なので中断・再実行しても
同一ビットに収束する（再開はログの進捗マーカーから --rows で範囲指定）。

使い方:
  python scripts/_run_v11_gen.py --rows 0-3599 --skip-inspect   # 並列その1
  python scripts/_run_v11_gen.py --rows 3600-7199 --skip-inspect # 並列その2
  python scripts/step11_v11_render.py inspect                    # 全完了後に1回
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v11_render as m11  # noqa: E402 (v11coreサンプラ差し替え込み)
m9 = m11.m9


def main():
    assert m9.V10_1 and m9.V10_1B, "V10_1/V10_1Bフラグ継承が壊れている"
    assert m9.DS_NAME == "outdoor_siren_v11"
    assert m9.sample_scene_v9 is m11.sample_scene_v11, "サンプラ差し替えが壊れている"
    rows_all = m9.load_plan("core")
    assert len(rows_all) == 7200
    lo, hi = 0, len(rows_all) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    rows = rows_all[lo:hi + 1]
    t0 = time.time()
    print(f"=== v11 gen rows {lo}-{hi} ({len(rows)} clips) ===", flush=True)
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
    print("=== running inspect_all() ===", flush=True)
    fail = m9.inspect_all()
    print(f"\n=== v11 ALL DONE (inspect fail={fail}) total {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    main()
