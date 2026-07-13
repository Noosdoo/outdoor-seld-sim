"""車のエンジン/走行音の合成（v5以降の妨害音用、クリーン合成）。

v3/v4は実録音(suv_dirt_road.wav)を妨害音に使っていたが、v5からはターゲット4クラス
（alert_sounds.py）と音源の質を揃えるため、妨害音もクリーン合成にする
（実録音は使わない、というユーザー指示。PROGRESS.md「v3 ドローン混入版の破棄」参照）。

2026-07-12 改良: 初版（倍音ハム+雑音の単純な足し算）が実物と質感がかけ離れて
いたため、外部調査を踏まえて作り直した。実際のエンジン音合成では「クランク
1回転を位相0→1のトリガ信号とみなし、そこから燃焼・バルブ挙動を作る」手法や、
「RPM依存の低周波倍音＋それに振幅変調された高周波狭帯域信号」という分解が
使われる（出典: ResearchGate "Physically informed car engine sound synthesis
for virtual and augmented environments"）。ここでは簡略化して:
  - 気筒の発火リズムを豊富な倍音を持つ準ノコギリ波で表現（純音の合成より
    機械的な質感が出る）
  - RPMのわずかな揺らぎ（低周波の乱数）でf0を微小変動させ、機械的な硬さを崩す
  - 発火周期に同期した振幅変調（チャギング感）
  - 路面/タイヤ由来の低域寄り広帯域ノイズを混合
"""
from __future__ import annotations

import numpy as np

from .noise import colored_noise


def make_car_driveby(duration_sec: float, fs: int, rng: np.random.Generator,
                     f0: float = 42.0, n_harmonics: int = 8,
                     jitter_depth: float = 0.04, fire_am_depth: float = 0.3,
                     rumble_slope: float = 1.3, rumble_mix: float = 0.35,
                     peak: float = 0.9) -> np.ndarray:
    """車の走行音（エンジン+タイヤ/路面ノイズ）を近似するクリーン合成音（改良版）。

    Args:
        rng: 揺らぎ・雑音成分の乱数生成器（呼び出し側でクリップ毎に別seedを渡す）
        f0: 気筒発火の基本周波数[Hz]（4気筒4stroke・アイドル回転数相当）
        jitter_depth: RPM揺らぎの深さ（f0に対する相対比率、1標準偏差）
        fire_am_depth: 発火周期に同期した振幅変調の深さ（チャギング感）
        rumble_mix: 路面/タイヤ広帯域ノイズの混合比（0=倍音のみ、1=ノイズのみ）
    """
    n = int(round(duration_sec * fs))

    # RPMのゆらぎ: ごくゆっくり変動する乱数でf0を微小変動させる（機械的な硬さを崩す）
    jitter = colored_noise(n, fs, rng, slope=2.5, f_lo=0.3)
    f_inst = f0 * (1.0 + jitter_depth * jitter)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs   # 周波数変動を位相に積分（クリック防止）

    # 気筒の発火リズム: 倍音を1/kで多数重ねる＝準ノコギリ波（純音より機械的でエッジが立つ）
    tonal = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        tonal += (1.0 / k) * np.sin(k * phase)
    tonal /= np.max(np.abs(tonal))

    # 発火周期に同期した振幅変調（チャギング/こもり感。1.0-depth〜1.0の範囲で変動）
    am = 1.0 - fire_am_depth * (0.5 + 0.5 * np.sin(phase))
    tonal = tonal * am

    # 路面/タイヤ由来の広帯域ノイズ（低域寄り、noise.pyの有色雑音を流用）
    rumble = colored_noise(n, fs, rng, slope=rumble_slope, f_lo=20.0)

    x = (1.0 - rumble_mix) * tonal + rumble_mix * rumble
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x
