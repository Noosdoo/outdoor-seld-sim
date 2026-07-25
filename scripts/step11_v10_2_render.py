# -*- coding: utf-8 -*-
"""step11_v10_2_render.py — v10.2（複数車対応の学習追加675本＋幻覚評価30本）レンダ設定。

step11_v10_1b_render.py のチェーン（v10のJP物理定数・車速11-36km/h・V10_1消防車・
V10_1B backup混合）をそのまま継承し、出力先だけを独立フォルダ
out/dataset_outdoor_siren_v10_2_add/ に切り替える（v9.2 6節3項の版管理原則:
本体フォルダに追記せず独立生成・独立検品。マージはColab側の展開時のみ）。

複数車(traffic2/3)・車なし・同一クラス警告×2の挙動は全てplan列駆動
（scenario/car_side/w1==w2）なので、レンダ側の追加フラグは不要
（v10a・v9.2で生成実績のある経路をそのまま使う）。

使い方: 生成は scripts/_run_v10_2_gen.py。zipは本ファイルの pack_add()
（**zip内パスはdatasets/outdoor_siren_v10/のまま**にして、Colabで本体zipの上に
そのまま追記展開できる形にする）。
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1b_render as m10_1b  # noqa: E402 (V10_1+V10_1B+v10設定を継承)

m9 = m10_1b.m9
m9.DS_NAME = "outdoor_siren_v10_2_add"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = m9.DS / "plan"
m9.WORK = m9.DS / "work"


def pack_add() -> None:
    """追補zip。zip内パスは本体データセット名（outdoor_siren_v10）に付け替え、
    Colabで本体zipの後に追記展開するだけでマージが完了する形にする。"""
    zip_path = ROOT / "out" / "dataset_outdoor_siren_v10_2_add.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((m9.DS / sub).glob("*")):
                zf.write(p, f"datasets/outdoor_siren_v10/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pack_add":
        pack_add()
    else:
        print("usage: step11_v10_2_render.py pack_add  (生成は _run_v10_2_gen.py)")
