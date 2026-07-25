# -*- coding: utf-8 -*-
"""step11_v9_2_ctrl2_render.py — v9.2均衡対照（ctrl2）レンダ設定。

**物理・音源は意図的にv9.1条件のまま**（V91のみON。旧horn 93-112dB@7m・
旧backup 87-112dB@1.2m・消防車なし・車速5-15m/s）。比較対象のv9.2/旧ctrlが
v9.1条件で学習されているため、対照だけ日本適合修正を入れると新たな交絡になる
（設計= md/design/v9_2_ctrl2_design_2026-07-22.md 1節）。

出力は独立フォルダ。zipは pack_ctrl2()（zip内パスをdatasets/outdoor_siren_v9_1/に
付け替え=Colabで追記展開するだけでマージ）。
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v9_render as m9  # noqa: E402

m9.V91 = True
m9.V92 = False
m9.V93 = False
m9.V10_1 = False
m9.V10_1B = False
m9.DS_NAME = "outdoor_siren_v9_2_ctrl2"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = m9.DS / "plan"
m9.WORK = m9.DS / "work"


def pack_ctrl2() -> None:
    zip_path = ROOT / "out" / "dataset_outdoor_siren_v9_2_ctrl2.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((m9.DS / sub).glob("*")):
                zf.write(p, f"datasets/outdoor_siren_v9_1/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pack_ctrl2":
        pack_ctrl2()
    else:
        print("usage: step11_v9_2_ctrl2_render.py pack_ctrl2  (生成は _run_v9_2_ctrl2_gen.py)")
