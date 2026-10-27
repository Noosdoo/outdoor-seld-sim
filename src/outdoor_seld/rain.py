"""⑥ 雨音（背景雑音としての雨）の合成 — 試作（2026-09-02）。

【位置づけ】todo⑥「合成データに雨を入れる」の第1要素＝**背景雑音としての雨音**だけを扱う。
雨の日の残り2要素（②マイクに当たる雨滴の衝突音・③濡れた路面のタイヤ音の変化）は
このモジュールでは再現しない（todo⑥の注意書きどおり、①だけで「雨の日」と書かない）。

【モデル】雨音 = 無数の雨滴が路面・葉・傘などに当たる短い衝撃音の重ね合わせ。
  - 粒（droplet）: ポアソン過程で時刻を引き、1発ごとに振幅（対数正規）と
    中心周波数（0.8〜8kHzの対数一様。大粒が舗装に当たる低め の音も含む）を引いた 3〜8ms の減衰する帯域雑音バースト
  - 地（hiss）: 遠くの無数の小粒が溶け合った連続雑音。2.5kHz付近に峰を持つ幅広（±1.3oct）の帯域整形雑音
  強い雨ほど粒の発生率が上がり、粒どうしが重なって地に近づく（実際の雨の聴感と同じ）。
  雨の強さは「雨滴発生率」で連続的に指定し、絶対レベル(dB(A))は呼び出し側が
  calibration.gain_for_spl_a で別途決める（noise.py / noise_v12.py と同じ流儀）。

【空間】等方拡散FOA（noise.diffuse_foa_noise と同じパワー規約 E|YZX|^2 = E|W|^2/3）。
  実際の雨音は地面側（下）と周囲から来るが、試作では等方とする（⚠️ 仮定。実録との
  聴き比べで上下の偏りが要ると分かれば Z を弱める）。

【レベルの仮置き（⚠️ 要出典・10月の実録で検証）】
  弱い雨 ≈ 45〜50 dB(A) / 中程度 ≈ 50〜58 / 強い雨 ≈ 60〜68（舗装路・屋外・傘なし）。
  本モジュールは値を持たない。設計書側で決める。
"""
from __future__ import annotations

import numpy as np

from .noise import colored_noise  # 既存の有色雑音（変更なし）

# 雨の強さ → 雨滴発生率[1/s]（聴感でチューニングした試作値。層別の軸はこの値で持つ）
INTENSITY = {"light": 150.0, "moderate": 600.0, "heavy": 2500.0}


def _burst_kernel(fs: int, fc: float, dur_s: float, rng: np.random.Generator) -> np.ndarray:
    """1粒ぶんの衝撃音: 中心fcの帯域雑音 × 指数減衰。"""
    n = max(int(dur_s * fs), 8)
    t = np.arange(n) / fs
    env = np.exp(-t / (dur_s / 4.0))                      # 4τで減衰しきる
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # fc を中心に Q≈1.5 のガウス帯域（対数周波数で対称）
    shape = np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / 0.6) ** 2)
    x = np.fft.irfft(spec * shape, n=n) * env
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def rain_droplets(n: int, fs: int, rng: np.random.Generator,
                  rate_per_s: float, n_kernels: int = 24) -> np.ndarray:
    """粒の成分（単位分散）。ポアソン到着 × 対数正規振幅 × ランダム帯域バースト。"""
    kernels = [_burst_kernel(fs, float(np.exp(rng.uniform(np.log(800.0), np.log(8000.0)))),
                             float(rng.uniform(0.003, 0.008)), rng)
               for _ in range(n_kernels)]
    n_ev = rng.poisson(rate_per_s * n / fs)
    t_idx = np.sort(rng.integers(0, n, size=n_ev))
    amp = np.exp(rng.normal(0.0, 0.6, size=n_ev))        # 対数正規: 大粒がたまに混じる
    kid = rng.integers(0, n_kernels, size=n_ev)
    out = np.zeros(n + max(len(k) for k in kernels))
    for i, a, k in zip(t_idx, amp, kid):
        ker = kernels[k]
        out[i:i + len(ker)] += a * ker
    out = out[:n]
    return out / (np.std(out) + 1e-12)


def rain_hiss(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """地の成分（単位分散）: 2.5kHz付近に峰を持つ幅広（±1.3oct）の帯域整形雑音（ホワイトを整形）。"""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    fc = 2500.0
    shape = np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / 1.3) ** 2)
    shape[f < 100.0] = 0.0
    x = np.fft.irfft(spec * shape, n=n)
    return x / (np.std(x) + 1e-12)


def rain_mono(n: int, fs: int, rng: np.random.Generator,
              rate_per_s: float, hiss_db: float = -6.0) -> np.ndarray:
    """雨音モノ（単位分散）。hiss_db = 粒に対する地の相対レベル[dB]。

    強い雨ほど粒が重なって連続音に近づくので、地の比率は rate に応じて自動で上げる
    （150/s: hiss_db のまま → 2500/s: +6dB）。
    """
    drops = rain_droplets(n, fs, rng, rate_per_s)
    hiss = rain_hiss(n, fs, rng)
    h = hiss_db + 6.0 * np.clip((np.log10(rate_per_s) - np.log10(150.0))
                                / (np.log10(2500.0) - np.log10(150.0)), 0.0, 1.0)
    x = drops + (10.0 ** (h / 20.0)) * hiss
    return x / (np.std(x) + 1e-12)


def diffuse_foa_rain(n: int, fs: int, rng: np.random.Generator,
                     rate_per_s: float, hiss_db: float = -6.0) -> np.ndarray:
    """等方拡散FOA版 (4, n)。W が単位分散、Y/Z/X はパワー1/3（noise.py と同じ規約）。"""
    w = rain_mono(n, fs, rng, rate_per_s, hiss_db)
    g = 1.0 / np.sqrt(3.0)
    y = g * rain_mono(n, fs, rng, rate_per_s, hiss_db)
    z = g * rain_mono(n, fs, rng, rate_per_s, hiss_db)
    x = g * rain_mono(n, fs, rng, rate_per_s, hiss_db)
    return np.stack([w, y, z, x], axis=0)


__all__ = ["INTENSITY", "rain_mono", "diffuse_foa_rain", "rain_droplets", "rain_hiss",
           "colored_noise"]
