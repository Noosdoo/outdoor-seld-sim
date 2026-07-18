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


def make_tire_noise(duration_sec: float, fs: int, rng: np.random.Generator,
                    f_lo: float = 600.0, f_hi: float = 2000.0) -> np.ndarray:
    """タイヤ/路面ノイズ（v9新規）: 1kHz中心600-2000Hzの帯域雑音（RMS=1）。

    根拠（out/v9_values_research_2026-07-16.md）: タイヤノイズは約1kHzが支配的で、
    スペクトル形状は速度にほぼ不変（レベルだけ変わる）。帯域端は半オクターブの
    コサインテーパで滑らかに落とす（矩形帯域のリンギング回避）。
    """
    n = int(round(duration_sec * fs))
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    inside = (f >= f_lo) & (f <= f_hi)
    shape[inside] = 1.0
    for edge, sign in ((f_lo, -1), (f_hi, +1)):   # 帯域外へ半オクターブで減衰
        lo, hi = (edge / np.sqrt(2.0), edge) if sign < 0 else (edge, edge * np.sqrt(2.0))
        m = (f >= lo) & (f < hi)
        frac = (np.log2(np.maximum(f[m], 1e-9)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
        shape[m] = 0.5 - 0.5 * np.cos(np.pi * (frac if sign < 0 else 1.0 - frac))
    x = np.fft.irfft(X * shape, n=n)
    return x / np.std(x)


def make_car_v9(duration_sec: float, fs: int, rng: np.random.Generator,
                f0: float = 42.0, tire_frac_a: float = 0.7,
                peak: float = 0.9) -> np.ndarray:
    """v9の車走行音 = エンジン(make_car_driveby) + タイヤ帯(make_tire_noise)。

    tire_frac_a: A特性パワーに占めるタイヤ成分の比率（既定0.7）。走行速度域では
    タイヤ音が支配的という知見に合わせ、かつA特性でほぼ消える低域エンジン音も
    質感として残す。配合はA特性RMSで合わせる（可聴性の議論と同じ土俵）。
    絶対レベルは呼び出し側が calibration.gain_for_spl_a で較正する。
    """
    from .calibration import a_weighted_rms
    eng = make_car_driveby(duration_sec, fs, rng, f0=f0, peak=1.0)
    tire = make_tire_noise(duration_sec, fs, rng)
    ra_e, ra_t = a_weighted_rms(eng, fs), a_weighted_rms(tire, fs)
    # 合成後のA特性パワー比が tire_frac_a : (1-tire_frac_a) になるゲイン
    g_t = np.sqrt(tire_frac_a) / ra_t
    g_e = np.sqrt(1.0 - tire_frac_a) / ra_e
    x = g_e * eng + g_t * tire
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x
