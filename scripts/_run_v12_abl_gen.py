# -*- coding: utf-8 -*-
"""ablation arm の全量生成ドライバ（2026-08-15、確認run用に新規作成）。

`_run_v12_gen.py` は出力先が `out/dataset_outdoor_siren_v12` 固定で、既存foaを
スキップする再開設計になっている。そのため ABLATE を付けてそのまま流すと

  - 基準データセットが既に埋まっている場合 → **全部スキップして何も生成されない**
  - 空の場合 → **基準データセットをablation音声で上書きする**

という2通りの事故が起きる。本ドライバは基準を絶対に触らないための専用経路。

安全装置:
  1. ABLATE が空なら**起動を拒否**する（基準の再生成は _run_v12_gen.py の仕事）
  2. 出力先は out/dataset_outdoor_siren_v12_abl_<arm>/ に固定（基準と別ディレクトリ）
  3. 起動時に基準ディレクトリと出力先が別物であることを assert する

使い方（サーバ・12シャード並列を想定）:
  ABLATE=no_1r python scripts/_run_v12_abl_gen.py --rows 0-849
  ABLATE=no_1r python scripts/_run_v12_abl_gen.py --list

決定論なので中断・再実行は同一ビットに収束する（既存foaはスキップ）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ARM = os.environ.get("ABLATE", "").strip()
if not ARM:
    raise SystemExit(
        "ABLATE が空です。本ドライバは ablation arm 専用です。\n"
        "基準(full)の生成は scripts/_run_v12_gen.py を使ってください。")

import step11_v12_render as v12  # noqa: E402  (importでABLATEが読まれる)
m9 = v12.m9

BASE_DS = ROOT / "out" / "dataset_outdoor_siren_v12"
DS = ROOT / "out" / f"dataset_outdoor_siren_v12_abl_{ARM}"
assert DS.resolve() != BASE_DS.resolve(), "出力先が基準データセットと同一です"

# レンダとラベルが見る出力先をarm専用ディレクトリへ差し替える
m9.DS = DS
m9.WORK = DS / "work"
for sub in ("foa", "metadata", "masks", "work"):
    (DS / sub).mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = m9.load_plan("core") + v12.load_plan_v12ext()
    assert len(rows) == 10200, len(rows)
    if "--list" in sys.argv:
        print(f"arm={ARM} total rows: {len(rows)} -> {DS}")
        return
    lo, hi = 0, len(rows) - 1
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    part = rows[lo:hi + 1]
    print(f"arm={ARM} rows={lo}-{hi} out={DS}", flush=True)
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
    print(f"[{lo}-{hi}] FINISHED arm={ARM} done={done} skip={skip} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
