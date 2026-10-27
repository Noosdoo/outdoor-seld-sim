"""v30 の背景の形（本体 §2-3・R1 (b)・2026-10-26）: 本ごとに折れ点と傾きを変える都市暗騒音（noise_v12 は無変更の別モジュール）。

v12 の形（noise_v12.URBAN_ANCHORS）: 63 Hz を 0 dB の峰・上へ −4 dB/oct・下へ −3 dB/oct（31.5 Hz で −3・それより下は端の値）。
v30 の形: 63 Hz から折れ点 f_k まで平ら（0 dB）・f_k より上は傾き s dB/oct・63 Hz より下は v12 と同じ（31.5 Hz で −3・それより下は端の値）。
  f_k = 63〜1,000 Hz の対数一様・s = U(−5, −3)（引くのは描画 v30 の別の乱数の流れ・この関数は形だけ）。
  家の背景の実測（md/research/根拠_可聴ゲートと背景の音_2026-10-25.md 1-6）= 125 Hz〜1 kHz がほぼ平らで 1 kHz より上が約 −3.7 dB/oct。
オクターブ帯の相対レベルの節点を log2(f) で直線につなぎ、PSD ∝ 10^(L/10)/f に直して白色雑音を整形する（noise_v12 と同じ流儀・端は外挿せず端の値）。
大きさ（dB(A)）は描画側で今までどおり gain_for_spl_a で目標値に合わせる。
"""
from __future__ import annotations

import numpy as np

KNEE_RANGE_HZ = (63.0, 1000.0)
SLOPE_RANGE_DB_OCT = (-5.0, -3.0)
F_TOP_HZ = 16000.0


def shape_anchors(knee_hz: float, slope_db_oct: float) -> tuple[np.ndarray, np.ndarray]:
    """節点（周波数 [Hz]・相対レベル [dB]）。31.5 Hz −3・63 Hz 0・折れ点 0・16 kHz は折れ点から傾きで。"""
    k = float(knee_hz)
    assert 20.0 <= k <= 4000.0, k                     # 決め直し 5（§9-7）: 幅は旗で替えられるので、ここは明らかな誤りだけ止める（既定の幅 63〜1,000 Hz の外も許す）
    f = [31.5, 63.0] + ([k] if k > 63.0 + 1e-9 else []) + [F_TOP_HZ]
    db = [-3.0, 0.0] + ([0.0] if k > 63.0 + 1e-9 else []) + [float(slope_db_oct) * np.log2(F_TOP_HZ / max(k, 63.0))]
    return np.array(f), np.array(db)


def shaped_colored_noise(n: int, fs: int, rng: np.random.Generator, knee_hz: float, slope_db_oct: float,
                         f_lo: float = 20.0) -> np.ndarray:
    """単位分散の暗騒音（オクターブ帯の相対レベルが shape_anchors に従う）。"""
    af, adb = shape_anchors(knee_hz, slope_db_oct)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    band = f >= f_lo
    lvl_db = np.interp(np.log2(np.maximum(f[band], 1e-9)), np.log2(af), adb)
    psd = (10.0 ** (lvl_db / 10.0)) / np.maximum(f[band], 1e-9)
    shape[band] = np.sqrt(psd)
    x = np.fft.irfft(spec * shape, n=n)
    return x / np.std(x)


def diffuse_foa_shaped_noise(n: int, fs: int, rng: np.random.Generator, knee_hz: float, slope_db_oct: float) -> np.ndarray:
    """等方拡散 FOA 版（noise_v12.diffuse_foa_urban_noise と同じパワーの約束 E|YZX|^2 = E|W|^2/3・同じ順で乱数を使う）。"""
    w = shaped_colored_noise(n, fs, rng, knee_hz, slope_db_oct)
    g = 1.0 / np.sqrt(3.0)
    y = g * shaped_colored_noise(n, fs, rng, knee_hz, slope_db_oct)
    z = g * shaped_colored_noise(n, fs, rng, knee_hz, slope_db_oct)
    x = g * shaped_colored_noise(n, fs, rng, knee_hz, slope_db_oct)
    return np.stack([w, y, z, x], axis=0)


def draw_shape(rng: np.random.Generator, knee_range: tuple | None = None, slope_range: tuple | None = None) -> dict:
    """折れ点（対数一様）と傾き（一様）を引く（2 回だけ乱数を使う）。幅（決め直し 5・§9-7 の旗）は引数・既定は KNEE_RANGE_HZ・SLOPE_RANGE_DB_OCT。
    rng.uniform(a, b) = a + (b − a)·U なので、同じ乱数の流れなら幅を替えても U は同じ（形の順位は保たれる）。"""
    kr = KNEE_RANGE_HZ if knee_range is None else (float(knee_range[0]), float(knee_range[1]))
    sr = SLOPE_RANGE_DB_OCT if slope_range is None else (float(slope_range[0]), float(slope_range[1]))
    lo, hi = np.log(kr[0]), np.log(kr[1])
    knee = float(np.exp(rng.uniform(lo, hi)))
    slope = float(rng.uniform(*sr))
    return {"knee_hz": knee, "slope_db_oct": slope}


def octave_levels(x: np.ndarray, fs: int, centers=(63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)) -> np.ndarray:
    """オクターブ帯のレベル [dB]（相対・検品用）。"""
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    out = []
    for fc in centers:
        m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        out.append(10.0 * np.log10(spec[m].sum() + 1e-30))
    return np.array(out)
