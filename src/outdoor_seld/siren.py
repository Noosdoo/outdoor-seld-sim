"""合成サイレン音源（ドライ信号）の生成。

救急車の wail 型サイレン: 基本周波数が 650-1450 Hz を周期 ~4.8s で往復する
正弦スイープ＋倍音2つ。位相積分で生成するためスイープが滑らか。
ドップラーがスペクトログラム上で見やすいトーン構造を持つ。
"""
from __future__ import annotations

import numpy as np


def make_siren(duration_sec: float, fs: int, f_lo: float = 650.0,
               f_hi: float = 1450.0, sweep_period_sec: float = 4.8,
               peak: float = 0.9, seed: int = 0) -> np.ndarray:
    """wail サイレンのモノラル信号を返す (float64, peak 正規化)。"""
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    f_center = 0.5 * (f_lo + f_hi)
    f_dev = 0.5 * (f_hi - f_lo)
    # 基本周波数の時間変化（正弦LFO）
    f_inst = f_center + f_dev * np.sin(2.0 * np.pi * t / sweep_period_sec - np.pi / 2)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = (1.00 * np.sin(phase)
         + 0.50 * np.sin(2.0 * phase)
         + 0.25 * np.sin(3.0 * phase))
    # ごく短いフェードイン/アウト（クリック防止）
    fade = int(0.01 * fs)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    x = x * env
    return peak * x / np.max(np.abs(x))
