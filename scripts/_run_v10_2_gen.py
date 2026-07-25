# -*- coding: utf-8 -*-
"""v10.2 生成ドライバ（バックグラウンド実行用）。

assignment_v10_2add.csv（675本）と assignment_halluc.csv（30本）を全行生成し、
最後に inspect_all()（この独立フォルダ内の全クリップ検品）を回す。
生成は完全決定論なので中断・再実行しても同一ビットに収束する。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_2_render as m10_2  # noqa: E402
m9 = m10_2.m9

SETS = ["v10_2add", "halluc"]


def main():
    assert m9.V10_1 and m9.V10_1B, "V10_1/V10_1Bフラグ継承が壊れている"
    assert m9.DS_NAME == "outdoor_siren_v10_2_add"
    # 並列実行用: --rows a-b で全705行（v10_2add+halluc連結）の担当範囲を指定、
    # --skip-inspect で最終検品を省略（全プロセス完了後に inspect だけ別途1回回す）。
    # 引数なしは従来どおり全行+検品。
    rows_all = []
    for which in SETS:
        rows_all += m9.load_plan(which)
    lo, hi = 0, len(rows_all) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    rows = rows_all[lo:hi + 1]
    t0 = time.time()
    print(f"=== v10.2 gen rows {lo}-{hi} ({len(rows)} clips) ===", flush=True)
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
    print(f"\n=== v10.2 ALL DONE (inspect fail={fail}) total {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    main()
