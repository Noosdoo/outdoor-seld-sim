# -*- coding: utf-8 -*-
"""v12確定評価セット1,800本の生成ドライバ（--rows分割・skip再開）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v12_conf_render as cf  # noqa: E402


def main() -> None:
    rows = cf.load_plan_v12conf()
    assert len(rows) == 1800, len(rows)
    lo, hi = 0, len(rows) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    t0 = time.time()
    done = skip = 0
    for row in rows[lo:hi + 1]:
        if (cf.DS / "foa" / f"{row['clip_id']}.flac").exists():
            skip += 1
            continue
        cf.m12.generate_clip_v12(row)
        done += 1
    print(f"[{lo}-{hi}] FINISHED done={done} skip={skip} {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
