"""v12静音EV音源: タイヤ音＋PWMインバータトーン＋速度比例モータ次数トーン。

engine.py は無変更の別モジュール（v12設計書§2b、根拠= md/research/
不可聴帯域の有用場面_2026-08-05.md §2: EVはPWM由来のトーン性高周波5〜15kHzを発する）。

設計:
- 42Hzエンジン倍音列は持たない（内燃機関なし）
- 広帯域はタイヤ/路面ノイズ（engine.make_tire_noise 流用、速度域では支配的）
- ①PWMキャリアトーン: クリップ毎に8〜10kHzから抽選・固定周波数・微ゆらぎ
- ②モータ次数トーン: 速度比例 f_mot = MOTOR_ORDER_HZ_PER_MPS × v（+2倍音）。
  3〜10m/s → 約0.6〜2kHz帯に乗る
- 配合はA特性パワー比（make_car_v9と同じ流儀）: タイヤ0.75 / トーン計0.25。
  絶対レベルは呼び出し側が従来どおり較正
"""
from __future__ import annotations

import numpy as np

from .calibration import a_weighted_rms
from .engine import make_tire_noise
from .noise import colored_noise

MOTOR_ORDER_HZ_PER_MPS = 200.0     # 6m/s → 1.2kHz（文献の1〜10kHzトーン帯に整合）


def _tone(n: int, fs: int, f0: float, rng: np.random.Generator,
          jitter_depth: float = 0.002, am_depth: float = 0.15,
          am_rate_hz: float = 3.0) -> np.ndarray:
    """微ゆらぎ＋ゆっくりAMの正弦トーン（機械的な硬さを崩す、単位分散）。"""
    jitter = colored_noise(n, fs, rng, slope=2.5, f_lo=0.3)
    phase = 2.0 * np.pi * np.cumsum(f0 * (1.0 + jitter_depth * jitter)) / fs
    am = 1.0 - am_depth * (0.5 + 0.5 * np.sin(
        2.0 * np.pi * am_rate_hz * np.arange(n) / fs + rng.uniform(0, 2 * np.pi)))
    x = np.sin(phase) * am
    return x / np.std(x)


def make_car_ev(duration_sec: float, fs: int, rng: np.random.Generator,
                speed_mps: float = 6.0, pwm_hz: float | None = None,
                tire_frac_a: float = 0.75, pwm_frac_a: float = 0.10,
                motor_frac_a: float = 0.15, peak: float = 0.9) -> np.ndarray:
    """静音EVの走行音（単発クリップ用ドライ音源）。

    Args:
        speed_mps: 走行速度（モータ次数トーンの周波数を決める）
        pwm_hz: PWMキャリア周波数。None なら 8〜10kHz から抽選
    """
    n = int(round(duration_sec * fs))
    tire = make_tire_noise(duration_sec, fs, rng)
    if pwm_hz is None:
        pwm_hz = float(rng.uniform(8000.0, 10000.0))
    pwm = _tone(n, fs, pwm_hz, rng)
    f_mot = MOTOR_ORDER_HZ_PER_MPS * float(speed_mps)
    mot = (_tone(n, fs, f_mot, rng)
           + 0.5 * _tone(n, fs, 2.0 * f_mot, rng))   # 基本+2次（弱め）
    mot = mot / np.std(mot)

    parts = [(tire, tire_frac_a), (pwm, pwm_frac_a), (mot, motor_frac_a)]
    x = np.zeros(n)
    for sig, frac in parts:
        ra = a_weighted_rms(sig, fs)
        x = x + (np.sqrt(frac) / ra) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x


if __name__ == "__main__":   # 検品: トーンがスペクトルに卓越しているか
    fs = 48000
    rng = np.random.default_rng(0)
    x = make_car_ev(10.0, fs, rng, speed_mps=6.0, pwm_hz=9000.0)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)

    def band_db(f0, bw=100.0):
        m = (f >= f0 - bw) & (f < f0 + bw)
        return 10 * np.log10(spec[m].mean())

    base = band_db(3000.0)  # トーンの無い中間帯を基準
    for name, f0 in [("モータ1次(1.2k)", 1200.0), ("モータ2次(2.4k)", 2400.0),
                     ("PWM(9k)", 9000.0)]:
        print(f"{name}: 周辺帯域比 {band_db(f0) - base:+.1f} dB")
