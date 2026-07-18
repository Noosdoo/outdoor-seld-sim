"""絶対音量較正とA特性重み（v9）。dB SPL ⇔ デジタル値の対応の唯一の正規モジュール。

規約（out/v9_design_v2_2026-07-16.md 2節。137→143の変更は07-17、下記根拠）:
- **正弦波基準**: フルスケール（デジタル振幅1.0のピーク）の正弦波 ≡ 143 dB SPL(RMS)。
  よってデジタルRMS 1.0 ≡ 143 + 20log10(√2) ≈ 146.01 dB SPL(RMS)。
- 143 dB の根拠: 最悪ケース=サイレン120dB@20m（→146dB@1m）が横距離3mを通過し、
  地面反射（two-ray）が同相加算された場合の受音ピークは約 136.5+5 ≈ 141.5 dB SPL
  → ピーク≈−1.5dBFSで収まる。137dBでは反射同相加算でクリップするため変更した。
  静かな側: 暗騒音40dB(A) → デジタルRMS≈5.0e-6、24bit量子化雑音床(≈3.4e-8)から
  約43dB上 → 設計条件「床から≥40dB」を満たす。
- 音源・暗騒音とも **A特性RMS** で較正する（法規値はdB(A)前提と仮定、仮定は設計書に明記）。
- per-clipのpost_scale（正規化）は存在しない。この定数がデータセット全体で共通の唯一の換算。
"""
from __future__ import annotations

import numpy as np

FULLSCALE_SINE_SPL = 143.0
# デジタルRMS 1.0 が対応する dB SPL(RMS)（正弦波基準から導出）
K_RMS_SPL = FULLSCALE_SINE_SPL + 20.0 * np.log10(np.sqrt(2.0))  # ≈ 146.0103


def a_weight_amplitude(f: np.ndarray) -> np.ndarray:
    """A特性の振幅ゲイン（線形倍率）。IEC 61672 の解析式、1kHzで1.0に正規化。"""
    f = np.asarray(f, dtype=np.float64)
    f2 = f * f
    num = (12194.0 ** 2) * f2 * f2
    den = ((f2 + 20.6 ** 2)
           * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
           * (f2 + 12194.0 ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    # 1 kHz で 0 dB になる正規化定数（標準の +2.00 dB とほぼ同義、厳密に1kHz=1.0にする）
    f1k = 1000.0 ** 2
    r1k = ((12194.0 ** 2) * f1k * f1k
           / ((f1k + 20.6 ** 2)
              * np.sqrt((f1k + 107.7 ** 2) * (f1k + 737.9 ** 2))
              * (f1k + 12194.0 ** 2)))
    return r / r1k


def a_weighted_rms(x: np.ndarray, fs: int) -> float:
    """信号のA特性重み付きRMS（デジタル値）。周波数領域でA特性を掛けParsevalで戻す。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    X = np.fft.rfft(x)
    w = a_weight_amplitude(np.fft.rfftfreq(n, 1.0 / fs))
    p = np.abs(X * w) ** 2
    # 片側スペクトルのParseval（DCとNyquist以外は2倍）
    msq = (2.0 * np.sum(p[1:-1]) + p[0] + p[-1]) / (n * n)
    return float(np.sqrt(msq))


def spl_a(x: np.ndarray, fs: int) -> float:
    """信号のA特性音圧レベル dB SPL(A)（本プロジェクトの較正規約で）。"""
    r = a_weighted_rms(x, fs)
    return float(20.0 * np.log10(max(r, 1e-30)) + K_RMS_SPL)


def gain_for_spl_a(x: np.ndarray, fs: int, target_spl_a: float) -> float:
    """x に掛けると A特性レベルが target_spl_a dB(A) になるゲインを返す。"""
    r = a_weighted_rms(x, fs)
    assert r > 0, "cannot calibrate a silent signal"
    return float(10.0 ** ((target_spl_a - K_RMS_SPL) / 20.0) / r)


def frame_spl_a(x: np.ndarray, fs: int, frame_sec: float = 0.1) -> np.ndarray:
    """0.1秒フレームごとのA特性レベル dB SPL(A)。(n_frames,) を返す。

    可聴性マスク（設計4節）の土台: フレームSNR_A = frame_spl_a(音源) − frame_spl_a(雑音)。
    無音フレームは -inf ではなく非常に小さいdB値になる（1e-30の床）。
    """
    x = np.asarray(x, dtype=np.float64)
    spf = int(round(frame_sec * fs))
    n_frames = len(x) // spf
    frames = x[:n_frames * spf].reshape(n_frames, spf)
    X = np.fft.rfft(frames, axis=1)
    w = a_weight_amplitude(np.fft.rfftfreq(spf, 1.0 / fs))
    p = np.abs(X * w[None, :]) ** 2
    msq = (2.0 * np.sum(p[:, 1:-1], axis=1) + p[:, 0] + p[:, -1]) / (spf * spf)
    return 20.0 * np.log10(np.maximum(np.sqrt(msq), 1e-30)) + K_RMS_SPL
