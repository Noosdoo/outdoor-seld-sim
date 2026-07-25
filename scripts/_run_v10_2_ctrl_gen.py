# -*- coding: utf-8 -*-
"""v10.2均衡対照 生成ドライバ。--rows a-b で並列分担、--skip-inspect で検品分離。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_2_ctrl_render as mc  # noqa: E402
m9 = mc.m9


def main():
    assert m9.V10_1 and m9.V10_1B, "v10.2と同一物理（V10_1+V10_1B）が前提"
    assert m9.DS_NAME == "outdoor_siren_v10_2ctrl_add"
    rows_all = m9.load_plan("ctrlev") + m9.load_plan("ctrlclip")
    lo, hi = 0, len(rows_all) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    rows = rows_all[lo:hi + 1]
    t0 = time.time()
    print(f"=== v10_2ctrl gen rows {lo}-{hi} ({len(rows)} clips) ===", flush=True)
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
    print(f"\n=== v10_2ctrl ALL DONE (inspect fail={fail}) "
          f"total {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
