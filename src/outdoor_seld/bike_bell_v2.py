"""自転車のベル・第 2 版 — 2026-10-07 の実録（本人の自転車 ESCAPE RX 3 の付属ベル・家のデモ 3 回）に合わせた試作。

実測（raw_demo/261007_003／004／006・ベルの区間を 1.5 kHz 以上で見た・3 回とも同じ）:
- 部分音: 基音 2,460〜2,470 Hz（0 dB）、6,264〜6,282 Hz（−2〜−5 dB）、11,233〜11,350 Hz の 2〜3 本の束（−1〜−14 dB・うなり）、
  4,913 Hz（−26 dB）、基音のすぐ下 2,412〜2,528 Hz（−27〜−30 dB）。→ 部分音が少なく間が広い「ピン」型のベル。
  既存の make_bike_bell（基音 3,000 Hz・比 1.0/1.003/2.4/2.7/3.8/5.3/6.8・うなりは基音）とは中身が違う。
- 減衰: 最初の打音から −10 dB まで 0.07 s（打撃の立ち上がりの成分）・−20 dB まで 0.17〜0.39 s・−30 dB まで 0.32〜0.72 s。
- 打ち方: 強い 1 打のあと 0.35〜0.9 s 後に 15〜20 dB 弱い 2 打目（レバーの戻り）。3 打のときもある。
限界: 本人の自転車 1 台の 3 回だけ。本番の自転車のベルが別物ならそのベルを録って合わせ直す。クラス名・場面の使い方は既存のまま。

使い方: make_bike_bell_v2(duration_sec, fs, f0=2465.0, second_gap_sec=0.7, second_db=None, repeat_period_sec=2.0, rng=None)
戻り値はピーク正規化（make_bike_bell と同じ）。f0 は本ごとに 2,300〜2,700 Hz でばらつかせる想定（場面側）。
既定は 1 打（本人 10/7 23:56「一回でいい」）。実録にあった弱い 2 打目（−16 dB・0.7 s 後）は second_db=-16.0 で足せる。
"""
from __future__ import annotations

import numpy as np

# (基音との周波数比, 振幅, 減衰時定数[s])。4.57/4.60 の対が 11.3 kHz の束のうなりを作る
PARTIALS_V2 = [
    (1.000, 1.00, 0.20),
    (0.980, 0.04, 0.15),
    (2.000, 0.05, 0.10),
    (2.550, 0.60, 0.12),
    (4.570, 0.35, 0.08),
    (4.600, 0.25, 0.08),
    (4.615, 0.12, 0.07),
]
STRIKE_TAU_SEC = 0.012     # 打撃の立ち上がり（広帯域の短い雑音・最初の −10 dB/0.07 s を作る）
STRIKE_AMP = 0.6


def _one_strike(n: int, fs: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(n) / fs
    tone = np.zeros(n)
    for ratio, amp, tau in PARTIALS_V2:
        tone += amp * np.exp(-tt / tau) * np.sin(2.0 * np.pi * f0 * ratio * tt + rng.uniform(0, 2 * np.pi))
    # 打撃の雑音（2〜12 kHz・12 ms）
    k = min(n, int(0.05 * fs))
    noise = rng.standard_normal(k)
    X = np.fft.rfft(noise); f = np.fft.rfftfreq(k, 1.0 / fs)
    noise = np.fft.irfft(X * ((f >= 2000) & (f <= 12000)), n=k)
    noise = noise / (np.max(np.abs(noise)) + 1e-12) * STRIKE_AMP * np.exp(-np.arange(k) / fs / STRIKE_TAU_SEC)
    tone[:k] += noise
    return tone


def make_bike_bell_v2(duration_sec: float, fs: int, f0: float = 2465.0,
                      second_gap_sec: float = 0.7, second_db: float | None = None,
                      repeat_period_sec: float = 2.0, peak: float = 0.9,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """ピン型の自転車のベル（実測に合わせた第 2 版）。1 打を repeat_period_sec ごとに繰り返す（既存と同じ周期の扱い）。
    second_db を与えると second_gap_sec 後に弱い 2 打目を足す（実録のレバーの戻り・既定は無し）。"""
    rng = rng or np.random.default_rng(0)
    n = int(round(duration_sec * fs))
    x = np.zeros(n)
    n_repeats = int(np.ceil(duration_sec / repeat_period_sec))
    strikes = [(0.0, 0.0)] + ([(second_gap_sec, float(second_db))] if second_db is not None else [])
    for r in range(n_repeats):
        base = r * repeat_period_sec
        for k, (gap, db) in enumerate(strikes):
            i0 = int((base + gap) * fs)
            if i0 >= n:
                break
            x[i0:] += _one_strike(n - i0, fs, f0, rng) * 10 ** (db / 20.0)
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
