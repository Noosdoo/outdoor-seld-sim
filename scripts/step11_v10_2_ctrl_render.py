# -*- coding: utf-8 -*-
"""step11_v10_2_ctrl_render.py — v10.2均衡対照レンダ設定。

**物理はv10.2追加分と完全同一**（step11_v10_1b_renderのチェーンを継承:
v10のJP物理定数・車速11-36km/h・V10_1消防車・V10_1B backup混合）。
出力先だけを独立フォルダ out/dataset_outdoor_siren_v10_2ctrl_add/ に切替。
zipは pack_ctrl()（zip内パスをdatasets/outdoor_siren_v10/に付替=追記展開でマージ）。
設計= md/design/v10_2_ctrl_design_2026-07-22.md。
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1b_render as m10_1b  # noqa: E402 (V10_1+V10_1B+v10設定を継承)

m9 = m10_1b.m9
m9.DS_NAME = "outdoor_siren_v10_2ctrl_add"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = m9.DS / "plan"
m9.WORK = m9.DS / "work"


def pack_ctrl() -> None:
    zip_path = ROOT / "out" / "dataset_outdoor_siren_v10_2ctrl_add.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((m9.DS / sub).glob("*")):
                zf.write(p, f"datasets/outdoor_siren_v10/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pack_ctrl":
        pack_ctrl()
    else:
        print("usage: step11_v10_2_ctrl_render.py pack_ctrl  (生成は _run_v10_2_ctrl_gen.py)")
