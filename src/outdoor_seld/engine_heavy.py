"""E1a: 大型車の低周波デルタ成分（engine.py は無改変のまま別モジュール）。

大型ディーゼル（6気筒・低速走行 1000〜1500rpm 相当）の発火基本周波数は
50〜70Hz 帯に乗り、63Hz オクターブ帯が低速時の支配帯になる
（md/research/E低周波リサーチ_2026-08-04.md E-R1）。ここでは既存の
make_car_driveby と同じ「準ノコギリ倍音＋RPM揺らぎ＋発火同期AM」の文法で、
**250Hz以下に帯域制限した追加成分だけ**を作る（中高域の音色は変えない＝
E1a設計書2節の純化条件）。ゲインは呼び出し側が63Hz帯実測で決める。
"""
from __future__ import annotations

import numpy as np

from .noise import colored_noise


def heavy_f0_from_seed(audio_seed: int, lo: float = 50.0, hi: float = 70.0) -> float:
    """audio_seedから決定論的に大型車の発火基本周波数を選ぶ（再現可能）。"""
    u = ((int(audio_seed) * 2654435761) % (2 ** 32)) / 2.0 ** 32
    return lo + (hi - lo) * u


def make_heavy_delta(duration_sec: float, fs: int, rng: np.random.Generator,
                     f0: float = 60.0, fmax: float = 250.0,
                     jitter_depth: float = 0.04,
                     fire_am_depth: float = 0.3) -> np.ndarray:
    """帯域制限つき大型車デルタ（単位RMS）。倍音は k*f0 <= fmax のみ。"""
    n = int(round(duration_sec * fs))
    jitter = colored_noise(n, fs, rng, slope=2.5, f_lo=0.3)
    f_inst = f0 * (1.0 + jitter_depth * jitter)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    k_max = max(1, int(fmax // f0))
    tonal = np.zeros(n)
    for k in range(1, k_max + 1):
        tonal += (1.0 / k) * np.sin(k * phase)
    am = 1.0 - fire_am_depth * (0.5 + 0.5 * np.sin(phase))
    x = tonal * am
    rms = float(np.sqrt(np.mean(x ** 2)))
    return x / rms if rms > 0 else x
