"""v12背景騒音: 完全ピンク→都市暗騒音スペクトル（noise.pyは無変更の別モジュール）。

根拠（md/design/v12設計書_2026-08-05.md §2a）:
- 都市の環境騒音は交通由来で約60Hzのオクターブ帯に峰を持つ
  （Applied Acoustics: 都市/郊外の1/3オクターブ実測、v12設計書の引用参照）
- 現行ピンク（オクターブ等パワー）は低域が現実比10〜20dB軽い（E0の解釈制限の正体）

設計: オクターブ帯相対レベルのアンカー（63Hz=0dB峰、上へ-4dB/oct、下へ-3dB/oct、
20Hz未満遮断は従来どおり）を log2(f) 空間で線形補間し、PSDに変換して整形する。
dB(A)較正・等方拡散FOA化は noise.py の既存関数と同じ流儀。
"""
from __future__ import annotations

import numpy as np

# オクターブ帯中心周波数 → 相対レベル[dB]（63Hz峰=0）
URBAN_ANCHORS_HZ = np.array([31.5, 63.0, 125.0, 250.0, 500.0,
                             1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
URBAN_ANCHORS_DB = np.array([-3.0, 0.0, -4.0, -8.0, -12.0,
                             -16.0, -20.0, -24.0, -28.0, -32.0])


def urban_colored_noise(n: int, fs: int, rng: np.random.Generator,
                        f_lo: float = 20.0) -> np.ndarray:
    """単位分散の都市暗騒音（オクターブ帯レベルがURBAN_ANCHORSに従う）。"""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    band = f >= f_lo
    # オクターブ帯レベル[dB] を log2(f) 空間で線形補間（端は外挿でなく端値を保持）
    lvl_db = np.interp(np.log2(np.maximum(f[band], 1e-9)),
                       np.log2(URBAN_ANCHORS_HZ), URBAN_ANCHORS_DB)
    # 帯域パワー→PSD変換: オクターブ帯パワー P_band ∝ PSD(f)·f なので PSD ∝ 10^(L/10)/f
    psd = (10.0 ** (lvl_db / 10.0)) / np.maximum(f[band], 1e-9)
    shape[band] = np.sqrt(psd)
    x = np.fft.irfft(spec * shape, n=n)
    return x / np.std(x)


def diffuse_foa_urban_noise(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """等方拡散FOA版（noise.diffuse_foa_noiseと同一のパワー規約 E|YZX|^2=E|W|^2/3）。"""
    w = urban_colored_noise(n, fs, rng)
    g = 1.0 / np.sqrt(3.0)
    y = g * urban_colored_noise(n, fs, rng)
    z = g * urban_colored_noise(n, fs, rng)
    x = g * urban_colored_noise(n, fs, rng)
    return np.stack([w, y, z, x], axis=0)


if __name__ == "__main__":   # 検品: 生成音のオクターブ帯実測 vs アンカー
    rng = np.random.default_rng(0)
    xw = urban_colored_noise(24000 * 30, 24000, rng)
    spec = np.abs(np.fft.rfft(xw)) ** 2
    f = np.fft.rfftfreq(len(xw), 1.0 / 24000)
    print("帯域   目標dB  実測dB")
    ref = None
    for fc, tgt in zip(URBAN_ANCHORS_HZ[:-1], URBAN_ANCHORS_DB[:-1]):
        m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        lv = 10 * np.log10(spec[m].sum())
        if ref is None:
            ref = lv - tgt
        print(f"{fc:7.1f} {tgt:6.1f} {lv - ref:7.2f}")
