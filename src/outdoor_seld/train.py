"""v12列車走行音源（踏切クラスの音源拡張、日本の在来線準拠）。

根拠（2026-08-05リサーチ、md/design/v12設計書_2026-08-05.md 議題6の列車拡張）:
- 転動音の主成分は500〜2,000Hz（環境省 在来鉄道騒音測定マニュアル/音響学会誌73巻11号。
  主な速度域50〜120km/h）
- レール継目部の衝撃音は中間部より+5〜8dB（同マニュアル系資料）
- 通過騒音の実測: 普通列車90km/h・バラスト軌道・防音壁なしで最大82〜87dB@12.5m
  （東京都 在来線鉄道騒音・振動調査）→ レベル規定はplan側でこの実測に較正
- 幾何: 定尺レール25m・車両長約20m（ジョイント「ガタンゴトン」の周期の根拠）
- 想定シナリオ: 警報機のない第4種踏切（全国約2,200カ所・事故率は第1種の約2倍、
  総務省行政評価局/国交省）＝「警報が鳴らない場所で列車を音から検知する」

構成: ①転動音（500-2kHz帯雑音） ②低周波成分（40-100Hz、車体/構造系）
③継目衝撃（車両長周期の2連打「ガタンゴトン」、台車内軸距で2連の間隔が決まる）
"""
from __future__ import annotations

import numpy as np

from .calibration import a_weighted_rms
from .noise import colored_noise

RAIL_LEN_M = 25.0        # 定尺レール
CAR_LEN_M = 20.0         # 在来線一般車両長
BOGIE_AXLE_SPACING_M = 2.1   # 台車内軸距（2連打の間隔の根拠）


def _band_noise(n: int, fs: int, rng: np.random.Generator,
                f_lo: float, f_hi: float) -> np.ndarray:
    """半オクターブ・コサインテーパの帯域雑音（engine.make_tire_noiseと同じ流儀）。"""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    inside = (f >= f_lo) & (f <= f_hi)
    shape[inside] = 1.0
    for edge, sign in ((f_lo, -1), (f_hi, +1)):
        lo, hi = (edge / np.sqrt(2.0), edge) if sign < 0 else (edge, edge * np.sqrt(2.0))
        m = (f >= lo) & (f < hi)
        if m.any():
            frac = (np.log2(np.maximum(f[m], 1e-9)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
            shape[m] = 0.5 - 0.5 * np.cos(np.pi * (frac if sign < 0 else 1.0 - frac))
    x = np.fft.irfft(X * shape, n=n)
    return x / np.std(x)


def make_train_passby(duration_sec: float, fs: int, rng: np.random.Generator,
                      speed_mps: float = 20.0,
                      rolling_frac_a: float = 0.70, low_frac_a: float = 0.15,
                      joint_frac_a: float = 0.15, peak: float = 0.9) -> np.ndarray:
    """在来線列車の走行音（ドライ音源。移動・距離減衰はレンダラー側）。

    Args:
        speed_mps: 走行速度（継目リズムの周期を決める。在来線14〜33m/s）
    """
    n = int(round(duration_sec * fs))
    rolling = _band_noise(n, fs, rng, 500.0, 2000.0)
    low = colored_noise(n, fs, rng, slope=1.5, f_lo=40.0)   # 40Hz〜低域寄り

    # 継目衝撃: 車両長周期で「ガタン・ゴトン」（台車2軸=2連打×前後台車）
    t_car = CAR_LEN_M / speed_mps            # 1車両が通過する周期
    dt_axle = BOGIE_AXLE_SPACING_M / speed_mps   # 2連打の間隔
    env = np.zeros(n)
    t = rng.uniform(0.0, t_car)              # 位相はクリップ毎に乱数
    decay = int(0.03 * fs)                   # 打撃の減衰30ms
    kernel = np.exp(-np.arange(decay) / (0.008 * fs))
    while t < duration_sec:
        for off in (0.0, dt_axle,            # 前台車の2連打
                    t_car * 0.55, t_car * 0.55 + dt_axle):   # 後台車（車両後方）
            i = int((t + off) * fs)
            if 0 <= i < n:
                amp = 1.0 + 0.3 * rng.standard_normal()
                j = min(decay, n - i)
                env[i:i + j] += max(amp, 0.2) * kernel[:j]
        t += t_car
    joint_click = _band_noise(n, fs, rng, 200.0, 4000.0) * env
    ra = a_weighted_rms(joint_click, fs)
    joint_click = joint_click / ra if ra > 0 else joint_click

    x = np.zeros(n)
    for sig, frac in ((rolling, rolling_frac_a), (low, low_frac_a),
                      (joint_click / np.std(joint_click) if np.std(joint_click) > 0
                       else joint_click, joint_frac_a)):
        r = a_weighted_rms(sig, fs)
        x = x + (np.sqrt(frac) / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x


if __name__ == "__main__":
    fs = 48000
    x = make_train_passby(10.0, fs, np.random.default_rng(0), speed_mps=20.0)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)

    def band(f_lo, f_hi):
        return 10 * np.log10(spec[(f >= f_lo) & (f < f_hi)].sum())

    ref = band(500, 2000)
    print(f"転動帯500-2k: 0dB基準 / 低域40-100Hz: {band(40,100)-ref:+.1f}dB / "
          f"継目帯2-4k: {band(2000,4000)-ref:+.1f}dB")
    print(f"継目リズム周期(20m/s): 車両{CAR_LEN_M/20.0:.1f}s・2連打{BOGIE_AXLE_SPACING_M/20.0*1000:.0f}ms")
