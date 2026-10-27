"""自転車の走行音（ベルなし）のドライ音源・第 2 版 — 2026-10-07 の実測（本人の自転車・家・横 1 m・4〜12 km/h）に合わせた試作。

第 1 版（bicycle.py）との違い（根拠 = raw_bicycle_measure/analysis/bands.csv・summary.txt）:
- 音の中身は「1/3 オクターブごとのレベルが 500 Hz〜5 kHz でほぼ平ら（±1 dB）・5 kHz より上は約 −6 dB/oct で落ちる」
  ＝ ピンク雑音に近い形。暗騒音から 6 dB 以上出ていた帯域（1.26〜8 kHz）だけが確かで、500 Hz 以下は暗騒音に埋もれて読めない
  → 500 Hz 以下は設計値として −3 dB/oct で落とす（出典なし）。第 1 版の低い転がり音（80〜400 Hz）は根拠が無いので外した。
- 速さによるレベルの変化は 4〜12 km/h でほぼ平ら（6.5 log10 v）。レベルは描画側（l1m_db）が持つので、ここでは速さで音色を
  変えない（空転クリックは第 1 版と同じく既定で無し）。
- 漕いでいる区間のかすかな駆動音（チェーン・ケイデンスで振幅変調）と路面の粗さによるゆっくりした振幅のゆれは第 1 版のまま。

使い方: make_bicycle_v2(duration_sec, fs, rng, cadence_rpm=70) → ピーク正規化したドライ波形（他の make_* と同じ）。
絶対レベルは描画側（step11 の l1m_db）が当てる。規則 ②（合成側で同じ置き方を描画して l1m を合わせる）は別途。
"""
from __future__ import annotations

import numpy as np

from .calibration import a_weighted_rms

# 1/3 オクターブごとのレベルの形（dB・4 kHz 基準・実測の 500 Hz〜5 kHz は平ら・上は −6 dB/oct・下は設計値 −3 dB/oct）
SHAPE_FLAT_LO_HZ = 500.0
SHAPE_FLAT_HI_HZ = 5000.0
ROLL_HI_DB_PER_OCT = -6.0
ROLL_LO_DB_PER_OCT = -3.0     # 設計値（実測では暗騒音に埋もれて読めない帯域）
CHAIN_WEIGHT = 0.25           # 駆動音の A 特性実効値の比（タイヤ 1.0 に対して・設計値）


def _third_octave_shape(f: np.ndarray) -> np.ndarray:
    """周波数ごとの振幅の形（線形倍率）。1/3 オクターブで平ら = パワーが 1/f に比例（ピンク）。"""
    f = np.maximum(f, 1.0)
    db = np.zeros_like(f)
    lo = f < SHAPE_FLAT_LO_HZ
    hi = f > SHAPE_FLAT_HI_HZ
    db[lo] = ROLL_LO_DB_PER_OCT * np.log2(SHAPE_FLAT_LO_HZ / f[lo])
    db[hi] = ROLL_HI_DB_PER_OCT * np.log2(f[hi] / SHAPE_FLAT_HI_HZ)
    pink = 1.0 / np.sqrt(f)   # 1/3 オクターブ内のパワーを一定にする（帯域幅 ∝ f）
    return pink * 10 ** (db / 20.0)


def _pink_hiss(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = _third_octave_shape(f)
    shape[f < 40.0] = 0.0
    x = np.fft.irfft(X * shape, n=n)
    return x / (np.std(x) + 1e-12)


def _slow_am(n: int, fs: int, rng: np.random.Generator, depth: float, rate_hz: float) -> np.ndarray:
    step = max(1, int(fs / rate_hz))
    knots = rng.standard_normal(n // step + 2)
    env = np.interp(np.arange(n), np.arange(len(knots)) * step, knots)
    return 1.0 + depth * np.clip(env, -2.0, 2.0) * 0.5


def _band_noise(n: int, fs: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    m = (f >= lo) & (f <= hi)
    x = np.fft.irfft(X * m, n=n)
    return x / (np.std(x) + 1e-12)


def make_bicycle_v2(duration_sec: float, fs: int, rng: np.random.Generator,
                    cadence_rpm: float = 70.0, peak: float = 0.9) -> np.ndarray:
    """普通の自転車の走行音（ベルなし・ドライ音源・第 2 版）。速さで音色は変えない（レベルは描画側）。"""
    n = int(round(duration_sec * fs))
    tire = _pink_hiss(n, fs, rng) * _slow_am(n, fs, rng, depth=0.25, rate_hz=3.0)
    t = np.arange(n) / fs
    f_ped = 2.0 * float(cadence_rpm) / 60.0
    chain = _band_noise(n, fs, rng, 800.0, 3000.0) * (1.0 + 0.25 * np.sin(2 * np.pi * f_ped * t))
    x = np.zeros(n)
    for sig, w in ((tire, 1.0), (chain, CHAIN_WEIGHT)):
        r = a_weighted_rms(sig, fs)
        if r > 0:
            x = x + (w / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
