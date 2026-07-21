# -*- coding: utf-8 -*-
"""step11_v10_1_render.py — v10.1（本人指摘「消防車のサイレンの音なくない？」への対応）。

v10（step11_v10_render.py）をそのまま継承し、siren クラスに fire 型（消防車）を
追加する1点のみが差分。RNG最小撹乱設計（_warn_params内の分岐、
scripts/step11_v9_render.py の V10_1 節を参照）により、**peepo型のクリップと
siren以外のクリップは一切変更されない**。影響を受けるのは既存v10でwail型と
判定されたクリップのみ（その半分がfireに再割当される）。

出力先はv10と同一フォルダ（out/dataset_outdoor_siren_v10/）——新しいデータセットでは
なく、v10に対する**上書きパッチ**という位置づけ。対象クリップの特定・再生成は
scripts/_run_v10_1_patch.py が行う（このファイル自体はモジュール設定のみ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_render as m10  # noqa: E402 (importするだけでv10のモジュール設定を継承)

m9 = m10.m9
m9.V10_1 = True
