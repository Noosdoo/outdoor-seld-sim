"""v12電動キックボード音源（新クラス7、日本の特定小型原動機付自転車準拠）。

根拠（2026-08-05リサーチ）:
- 法規: 特定小型原動機付自転車（2023年7月道交法改正）= 最高20km/h、歩道通行モード6km/h
  → 速度域 1.7〜5.6m/s
- 静音性の実測: 60dB(A)の街頭soundscapeでのe-スクータ検知率は**わずか23%**
  （Applied Acoustics "Development of electric scooter alerting sounds using
  psychoacoustical metrics"）＝新しい静音脅威としての定量的根拠
- 音の構成: 小径ハードタイヤの転がり音（車より高め・小レベル）＋ハブモータの
  PWM/次数トーン（加速時に顕著）。絶対レベルは実測文献レンジからの**設計値**
  55〜65dB(A)@1mとし、9月実録で実機実測が取れたら較正し直す（backup_beepの
  「実勢/新基準」2系統方式と同じ誠実表示）
"""
from __future__ import annotations

import numpy as np

from .calibration import a_weighted_rms
from .engine_ev import _tone

MOTOR_ORDER_HZ_PER_MPS = 400.0    # 5.6m/s(20km/h) → 約2.2kHz（小径ホイールの高次数）


def _rolling_noise(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """小径ハードタイヤの転がり音: 800〜4,000Hz帯（車タイヤ600-2kより高め）。"""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    inside = (f >= 800.0) & (f <= 4000.0)
    shape[inside] = 1.0
    for edge, sign in ((800.0, -1), (4000.0, +1)):
        lo, hi = (edge / np.sqrt(2), edge) if sign < 0 else (edge, edge * np.sqrt(2))
        m = (f >= lo) & (f < hi)
        frac = (np.log2(np.maximum(f[m], 1e-9)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
        shape[m] = 0.5 - 0.5 * np.cos(np.pi * (frac if sign < 0 else 1.0 - frac))
    x = np.fft.irfft(X * shape, n=n)
    # 路面の粗さによるゆっくりしたレベル変動（歩道の継ぎ目感）
    am = 1.0 + 0.25 * np.clip(
        np.interp(np.arange(n), np.arange(0, n, fs // 4),
                  rng.standard_normal(len(range(0, n, fs // 4)))), -2, 2) * 0.5
    return (x * am) / np.std(x * am)


def make_kickboard(duration_sec: float, fs: int, rng: np.random.Generator,
                   speed_mps: float = 4.0,
                   rolling_frac_a: float = 0.6, motor_frac_a: float = 0.4,
                   peak: float = 0.9) -> np.ndarray:
    """電動キックボードの走行音（ドライ音源）。speed_mps: 1.7〜5.6m/s（法規域）。"""
    n = int(round(duration_sec * fs))
    rolling = _rolling_noise(n, fs, rng)
    f_mot = MOTOR_ORDER_HZ_PER_MPS * float(speed_mps)
    motor = (_tone(n, fs, f_mot, rng, jitter_depth=0.004)
             + 0.4 * _tone(n, fs, 2.0 * f_mot, rng, jitter_depth=0.004))
    motor = motor / np.std(motor)
    x = np.zeros(n)
    for sig, frac in ((rolling, rolling_frac_a), (motor, motor_frac_a)):
        r = a_weighted_rms(sig, fs)
        x = x + (np.sqrt(frac) / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
