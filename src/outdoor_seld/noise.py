"""拡散性背景雑音の生成と SNR 指定での混合。

設計:
- colored_noise: FFT整形の有色雑音。slope=1 でピンク（パワー 1/f）。
  20 Hz 未満は遮断（logmel特徴の f_min=20Hz に合わせる。可聴外の超低域に
  SNR予算を食われて「名目SNRより実効SNRが甘くなる」のを防ぐ）
- diffuse_foa_noise: 等方拡散音場の FOA (W,Y,Z,X / SN3D) 近似。
  4ch を互いに独立な同スペクトル雑音とし、パワー比
  E|Y|^2 = E|Z|^2 = E|X|^2 = E|W|^2 / 3 でスケールする
  （SN3D双極子の球面平均 <cos^2> = 1/3。ガウス雑音の拡散音場は
  この二次統計で完全に規定される）
- mix_at_snr: SNR[dB] を W チャンネル基準（クリップ全体の平均パワー比）で
  正確に合わせて加算する
"""
from __future__ import annotations

import numpy as np


def colored_noise(n: int, fs: int, rng: np.random.Generator,
                  slope: float = 1.0, f_lo: float = 20.0) -> np.ndarray:
    """単位分散の有色雑音（パワースペクトル ∝ f^-slope、f_lo未満は0）。"""
    white = rng.standard_normal(n)         # まず普通のホワイトノイズを作る
    spec = np.fft.rfft(white)              # 周波数領域に変換
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    band = f >= f_lo                       # 20Hz未満は0のまま（遮断）にする
    shape[band] = (f[band] / 1000.0) ** (-slope / 2.0)   # 振幅は周波数の-slope/2乗に比例させる
    x = np.fft.irfft(spec * shape, n=n)    # 整形したスペクトルを時間領域に戻す
    return x / np.std(x)                   # 標準偏差1に正規化（あとでSNR調整しやすくするため）


def diffuse_foa_noise(n: int, fs: int, rng: np.random.Generator,
                      slope: float = 1.0) -> np.ndarray:
    """等方拡散音場の FOA 雑音 (4, n)。W が単位分散、Y/Z/X はパワー1/3。"""
    w = colored_noise(n, fs, rng, slope)       # W chの雑音
    g = 1.0 / np.sqrt(3.0)                     # パワーを1/3にするための振幅ゲイン
    y = g * colored_noise(n, fs, rng, slope)   # 各chは独立な乱数列（rngを都度呼ぶので別系列）
    z = g * colored_noise(n, fs, rng, slope)
    x = g * colored_noise(n, fs, rng, slope)
    return np.stack([w, y, z, x], axis=0)      # チャンネル順はFOA規約と同じW,Y,Z,X


def mix_at_snr(foa_clean: np.ndarray, foa_noise_unit: np.ndarray,
               snr_db: float):
    """クリーンFOAに、Wチャンネル基準のSNR[dB]で雑音を混合する。

    Returns:
        foa_noisy: (4, n)
        noise_gain: 雑音に掛けたゲイン（記録用）
    """
    assert foa_clean.shape == foa_noise_unit.shape
    p_sig = float(np.mean(foa_clean[0] ** 2))          # 信号(W)の平均パワー
    p_noise_unit = float(np.mean(foa_noise_unit[0] ** 2))  # 雑音(単位振幅版)の平均パワー
    # 目標SNRになるように雑音のゲインを逆算する（dB→比率に変換して解く）
    noise_gain = float(np.sqrt(p_sig / (p_noise_unit * 10.0 ** (snr_db / 10.0))))
    return foa_clean + noise_gain * foa_noise_unit, noise_gain


def measure_snr_db(foa_clean: np.ndarray, foa_noisy: np.ndarray) -> float:
    """2つのファイルから実SNR（W基準）を独立に測る（検品用）。"""
    p_sig = float(np.mean(foa_clean[0] ** 2))
    p_noise = float(np.mean((foa_noisy[0] - foa_clean[0]) ** 2))  # 差分＝混ざった雑音そのもの
    return 10.0 * np.log10(p_sig / p_noise)
