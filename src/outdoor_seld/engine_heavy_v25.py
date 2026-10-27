# -*- coding: utf-8 -*-
"""大型車（トラック・バス）の走行音・v25（2026-10-12・変更票 #52 D25・設計の案 r2 の ④）。

既存の `engine.py`（make_car_v9）と `engine_heavy.py`（make_heavy_delta）は変えない。ここでは #52 の決めごとどおりに組む:
  - エンジンの成分 = make_car_driveby（f0 = 発火の基本 U(55, 80) Hz・6 気筒 1,100〜1,600 rpm の換算・rumble_mix は既定のまま）
    ＋ make_heavy_delta（同じ f0・250 Hz 以下の追加・乱数は別の流れ）。追加の低域のゲインは、**エンジンの成分の 250 Hz 以下（重みなし）の RMS が
    追加前より +10 dB** になる値（CNOSSOS 30 km/h の 125〜500 Hz の差 +14〜15 dB を目安に控えめにした設計値）。
  - タイヤの成分 = make_tire_noise（独立に合成）。**A 特性でのタイヤの割合 U(0.10, 0.20)**（CNOSSOS-EU 2015 の大型の 30〜40 km/h の参考値・設計比）=
    独立に合成したタイヤの A 特性パワーがクリップ全体の A 特性パワーに占める比。make_car_v9 と同じ分け方で A 特性 RMS で比を合わせてから混ぜる。
  - 最後にピーク正規化（絶対の大きさは呼び出し側 = v24 の曲線 + U(4.8, 8.6) dB）。
  - 全帯域のピークとクリッピングは描画器の検査（peak）に任せる。

自己検査: python -m outdoor_seld.engine_heavy_v25
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .engine import make_car_driveby, make_tire_noise
    from .engine_heavy import make_heavy_delta
    from .calibration import a_weighted_rms
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from outdoor_seld.engine import make_car_driveby, make_tire_noise
    from outdoor_seld.engine_heavy import make_heavy_delta
    from outdoor_seld.calibration import a_weighted_rms

HEAVY_F0_HZ = (55.0, 80.0)          # 発火の基本（6 気筒・1,100〜1,600 rpm・仮定）
HEAVY_TIRE_FRAC_A = (0.10, 0.20)    # A 特性でのタイヤの割合（CNOSSOS 大型 30〜40 km/h の参考値・設計比）
HEAVY_LEVEL_OFFSET_DB = (4.8, 8.6)  # 同じ速さの乗用車の法則に足す dB（ASJ RTN-Model 2018 の中型 +4.8／大型 +6.5／+7.7 と 2023 年版 +8.6 をつないだ設計の幅）
HEAVY_LOW_BOOST_DB = 10.0           # エンジンの成分の 250 Hz 以下の RMS を追加前より +10 dB
HEAVY_LOW_BAND_HZ = 250.0
HEAVY_FRACTION = 0.15               # 車の音源の台数のうち大型車の割合（道路交通センサス R3 の一般国道の大型車混入率 約 15.2%・設計の仮定）


def band_rms(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    X = np.fft.rfft(np.asarray(x, dtype=np.float64)); f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= f_lo) & (f < f_hi)
    n = len(x)
    p = np.abs(X[m]) ** 2
    return float(np.sqrt(2.0 * np.sum(p) / (n * n)))


def heavy_params(rng: np.random.Generator) -> dict:
    return {"f0_hz": float(rng.uniform(*HEAVY_F0_HZ)), "tire_frac_a": float(rng.uniform(*HEAVY_TIRE_FRAC_A)),
            "offset_db": float(rng.uniform(*HEAVY_LEVEL_OFFSET_DB))}


def make_heavy_car(duration_sec: float, fs: int, rng: np.random.Generator, f0: float = 65.0, tire_frac_a: float = 0.15,
                   low_boost_db: float = HEAVY_LOW_BOOST_DB, peak: float = 0.9) -> tuple[np.ndarray, dict]:
    """大型車のドライ音と、検査用の内訳（低域の増分・タイヤの割合・追加のゲイン）。"""
    eng = make_car_driveby(duration_sec, fs, rng, f0=f0, peak=1.0)
    delta = make_heavy_delta(duration_sec, fs, np.random.default_rng(rng.integers(0, 2 ** 31)), f0=f0, fmax=HEAVY_LOW_BAND_HZ)
    e_low = band_rms(eng, fs, 0.0, HEAVY_LOW_BAND_HZ) ** 2
    d_low = band_rms(delta, fs, 0.0, HEAVY_LOW_BAND_HZ) ** 2
    target = e_low * 10.0 ** (low_boost_db / 10.0)
    g_d = float(np.sqrt(max(target - e_low, 0.0) / d_low)) if d_low > 0 else 0.0
    eng2 = eng + g_d * delta
    tire = make_tire_noise(duration_sec, fs, rng)
    ra_e, ra_t = a_weighted_rms(eng2, fs), a_weighted_rms(tire, fs)
    g_t = np.sqrt(tire_frac_a) / ra_t
    g_e = np.sqrt(1.0 - tire_frac_a) / ra_e
    x = g_e * eng2 + g_t * tire
    pk = float(np.max(np.abs(x)))
    x = peak * x / pk if pk > 0 else x
    low_gain_db = 20.0 * np.log10(band_rms(eng2, fs, 0.0, HEAVY_LOW_BAND_HZ) / max(np.sqrt(e_low), 1e-12))
    info = {"f0_hz": f0, "tire_frac_a": tire_frac_a, "delta_gain": g_d, "low_band_gain_db": round(float(low_gain_db), 3),
            "tire_frac_a_measured": round(float((g_t * ra_t) ** 2 / ((g_e * ra_e) ** 2 + (g_t * ra_t) ** 2)), 4)}
    return x, info


def selftest() -> int:
    fs = 48000
    ok = []
    for f0, tf in ((55.0, 0.10), (65.0, 0.15), (80.0, 0.20)):
        x, info = make_heavy_car(4.0, fs, np.random.default_rng(1), f0=f0, tire_frac_a=tf)
        ok.append((f"f0 {f0}: 250 Hz 以下のエンジンの成分が +10 dB（{info['low_band_gain_db']} dB・±0.3）", abs(info["low_band_gain_db"] - HEAVY_LOW_BOOST_DB) < 0.3))   # エンジンと追加の低域は同じ f0 の倍音を持つので交差項で ±0.2 dB ほど動く
        ok.append((f"f0 {f0}: A 特性のタイヤの割合 {tf}（測定 {info['tire_frac_a_measured']}）", abs(info["tire_frac_a_measured"] - tf) < 0.005))
        ok.append((f"f0 {f0}: ピーク正規化 0.9", abs(float(np.max(np.abs(x))) - 0.9) < 1e-6))
    car = make_car_driveby(4.0, fs, np.random.default_rng(1), f0=42.0, peak=1.0)
    hv, _ = make_heavy_car(4.0, fs, np.random.default_rng(1), f0=65.0, tire_frac_a=0.15)
    r_car = band_rms(car, fs, 0.0, 250.0) / band_rms(car, fs, 500.0, 4000.0)
    r_hv = band_rms(hv, fs, 0.0, 250.0) / band_rms(hv, fs, 500.0, 4000.0)
    ok.append((f"大型車は乗用車（エンジンだけ）より 250 Hz 以下／500〜4,000 Hz の比が大きい（{r_hv:.2f} vs {r_car:.2f}）", r_hv > r_car))
    rng = np.random.default_rng(7); draws = [heavy_params(rng) for _ in range(2000)]
    ok.append(("heavy_params: f0 55〜80・タイヤ 0.10〜0.20・大きさ +4.8〜8.6 の中", all(HEAVY_F0_HZ[0] <= d["f0_hz"] <= HEAVY_F0_HZ[1] and HEAVY_TIRE_FRAC_A[0] <= d["tire_frac_a"] <= HEAVY_TIRE_FRAC_A[1] and HEAVY_LEVEL_OFFSET_DB[0] <= d["offset_db"] <= HEAVY_LEVEL_OFFSET_DB[1] for d in draws)))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(selftest())
