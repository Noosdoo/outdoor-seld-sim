# -*- coding: utf-8 -*-
"""v10 全量生成ドライバ（バックグラウンド実行用）。scripts/step11_v10_render.py の
モジュール設定をそのまま使い、5セット全行を1プロセス内でループ生成する
（gen呼び出し毎のプロセス起動コストを避けるため）。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_render as m10  # noqa: E402 (importするだけでモジュール設定が反映される)
m9 = m10.m9

SETS = ["core", "scenario", "probe", "scenario2", "v10a"]

t0 = time.time()
total_done = 0
for which in SETS:
    rows = m9.load_plan(which)
    print(f"=== {which}: {len(rows)} rows ===", flush=True)
    for i, row in enumerate(rows):
        m9.generate_clip(row)
        total_done += 1
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f"[{which}] {i + 1}/{len(rows)} (total {total_done}) "
                  f"{el:.0f}s elapsed, {el / total_done:.2f}s/clip", flush=True)

print(f"\n=== generation done: {total_done} clips, {time.time() - t0:.0f}s ===", flush=True)
print("=== running inspect_all() ===", flush=True)
fail = m9.inspect_all()
print(f"\n=== ALL DONE (inspect fail={fail}) total {time.time() - t0:.0f}s ===", flush=True)
