# -*- coding: utf-8 -*-
"""v12全量生成ドライバ（core 7,200 + v12ext 3,000 = 10,200本、シャード分割・再開可能）。

使い方:
  python scripts/_run_v12_gen.py --rows 0-849      # シャード（サーバーで12並列想定）
  python scripts/_run_v12_gen.py --list            # 総行数の確認のみ
決定論なので中断・再実行は同一ビットに収束（既存foaはスキップ）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v12_render as v12  # noqa: E402
m9 = v12.m9


def main() -> None:
    rows = m9.load_plan("core") + v12.load_plan_v12ext()
    assert len(rows) == 10200, len(rows)
    if "--list" in sys.argv:
        print(f"total rows: {len(rows)}")
        return
    lo, hi = 0, len(rows) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    part = rows[lo:hi + 1]
    t0 = time.time()
    done = skip = 0
    for i, row in enumerate(part):
        if (m9.DS / "foa" / f"{row['clip_id']}.flac").exists():
            skip += 1
            continue
        v12.generate_clip_v12(row)
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[{lo}-{hi}] {i+1}/{len(part)} done={done} skip={skip} "
                  f"{el/max(done,1):.1f}s/clip", flush=True)
    print(f"[{lo}-{hi}] FINISHED done={done} skip={skip} {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
