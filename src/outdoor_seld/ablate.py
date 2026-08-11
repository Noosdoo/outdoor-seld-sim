# -*- coding: utf-8 -*-
"""物理ablationの一元スイッチ（アブレーション実験計画 第5版・2026-08-11実装）。

環境変数 ABLATE ∈ {"", "no_doppler", "no_airabs", "no_1r", "no_ground"} を
プロセス起動時に1回だけ読む。レンダ（fastsim / step11系）とラベル（labels.py）が
同じ値を参照することで、音とラベルの規約が条件ごとに必ず一致する。

- no_doppler: 音源起因の時間伸縮を消す（一定遅延読み出し・v9設計8節の事前決定規約）。
  ラベルの発音区間判定も同じ一定遅延規約に切替わる（labels.py参照）。
  方向・距離の値は放射時刻ベースのまま＝FOAエンコードと同一経路で不変。
- no_airabs: ISO9613-1大気吸収FIRを適用しない
- no_1r   : 幾何減衰1/rを適用しない
- no_ground: 地面反射（鏡像レンダ）を加算しない
"""
from __future__ import annotations

import os

MODES = ("", "no_doppler", "no_airabs", "no_1r", "no_ground")
MODE = os.environ.get("ABLATE", "").strip()
assert MODE in MODES, f"ABLATE must be one of {MODES}, got {MODE!r}"

# no_dopplerの参照点計算に使うグリッド。レンダ側FS_SIMと一致していることが必須
# （step11_v9_render がimport時に自分のFS_SIMで上書きして整合を保証する）
NODOP_FS = 48_000


def render_flags() -> dict:
    """fastsim.render_mono へ渡す物理スイッチ。"""
    return dict(enable_doppler=MODE != "no_doppler",
                enable_spreading=MODE != "no_1r",
                enable_air_absorption=MODE != "no_airabs")


def ground_enabled() -> bool:
    return MODE != "no_ground"
