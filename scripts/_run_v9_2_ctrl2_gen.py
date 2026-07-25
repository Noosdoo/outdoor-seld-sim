# -*- coding: utf-8 -*-
"""v9.2均衡対照（ctrl2）生成ドライバ。270本+検品。決定論なので再実行安全。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v9_2_ctrl2_render as mc  # noqa: E402
m9 = mc.m9


def main():
    assert m9.V91 and not (m9.V92 or m9.V93 or m9.V10_1 or m9.V10_1B), \
        "ctrl2はv9.1条件（V91のみ）で生成する"
    assert m9.DS_NAME == "outdoor_siren_v9_2_ctrl2"
    rows = m9.load_plan("ctrl2")
    t0 = time.time()
    print(f"=== ctrl2: {len(rows)} rows ===", flush=True)
    for i, row in enumerate(rows):
        m9.generate_clip(row)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f"{i + 1}/{len(rows)} {el:.0f}s elapsed, {el/(i+1):.2f}s/clip",
                  flush=True)
    print(f"\n=== generation done: {len(rows)} clips, {time.time()-t0:.0f}s ===",
          flush=True)
    print("=== running inspect_all() ===", flush=True)
    fail = m9.inspect_all()
    print(f"\n=== ctrl2 ALL DONE (inspect fail={fail}) total {time.time()-t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    main()
