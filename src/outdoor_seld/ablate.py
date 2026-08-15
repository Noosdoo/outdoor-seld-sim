# -*- coding: utf-8 -*-
"""物理ablationの一元スイッチ（アブレーション実験計画 第5版・2026-08-11実装）。

環境変数 ABLATE ∈ {"", "no_doppler", "no_airabs", "no_1r", "no_ground"} を
プロセス起動時に1回だけ読む。レンダ（fastsim / step11系）とラベル（labels.py）が
同じ値を参照することで、音とラベルの規約が条件ごとに必ず一致する。

- no_doppler: 音源起因の時間伸縮を消す（一定遅延読み出し・v9設計8節の事前決定規約）。
  ラベルの発音区間判定も同じ一定遅延規約に切替わる（labels.py参照）。
  方向・距離の値は放射時刻ベースのまま＝FOAエンコードと同一経路で不変。
- no_airabs: ISO9613-1大気吸収FIRを適用しない
- no_1r   : 幾何減衰を**基準距離r_ref固定**にする（距離手がかりを消す。2026-08-16改訂）
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


# no_1r の基準距離 [m]（2026-08-16 改訂・確認runでのクリップ停止を受けた条件再定義）。
# 素朴に「1/rを掛けない」= g=1 は全音源を1m相当にすることであり、遠方源が桁違いに
# 大きくなって混合波形がフルスケールを超える（peak 1.11 で生成が停止した）。
# そこで **全音源を基準距離 r_ref に置いたとみなして一定の減衰を掛ける** 定義に変える。
# 距離手がかり（遠近の音量差・接近に伴う音量増大）は同じく完全に消える。
# 既定 10m は本研究の音源仕様の基準距離（車 60-67dB@10m 等）に合わせたもの。
# 環境変数 ABL_SPREAD_REF_M で上書きできる。
SPREAD_REF_M = float(os.environ.get("ABL_SPREAD_REF_M", "10.0"))


def render_flags() -> dict:
    """fastsim.render_mono へ渡す物理スイッチ。"""
    return dict(enable_doppler=MODE != "no_doppler",
                enable_spreading=True,
                spreading_ref_m=(SPREAD_REF_M if MODE == "no_1r" else None),
                enable_air_absorption=MODE != "no_airabs")


def ground_enabled() -> bool:
    return MODE != "no_ground"
