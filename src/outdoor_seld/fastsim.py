"""DynamicSound と数学的に等価な高速モノラルレンダラ（ベクトル化）。

DynamicSound `_simulation.py run()` の処理を1サンプルずつのPythonループから
numpy ベクトル演算に置き換えたもの。物理・式・係数は同一:

  ① 放射時刻 te: geometry.solve_emission_times（DS内部と誤差ゼロ一致を単体テスト済み）
  ② 音源読み出し: ドライ信号の te*fs 位置を線形補間（AudioFile.get_sample と同一、
     loop=False 相当。te が無い/範囲外は 0）
  ③ 幾何減衰: 1/distance（distance=0 は 1.0、attenuations.geometric と同一）
  ④ 大気吸収: DynamicSound 自身の attenuation_coefficients（ISO 9613-1）から
     firwin2(513) で FIR 設計。DS は毎サンプル設計するが、距離は滑らかにしか
     変わらないため本実装は block_len サンプルごとに設計し区分一定で適用する
     （既定 240 サンプル=5ms、距離変化 ≤0.08m@15m/s → 係数差は無視できる）。

等価性は tests/test_fastsim.py で「同一シーンの DynamicSound 実出力」との
波形比較により検証する（int32量子化・FIR区分一定化の分だけの微小差を許容）。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin2

from dynamic_sound.acoustics.standards.ISO_9613_1_1993 import (
    attenuation_coefficients)

from .geometry import solve_emission_times, sound_speed

FIR_LEN = 513  # DynamicSound と同一


def render_mono(dry: np.ndarray, waypoints, mic_pos, fs: int,
                clip_len_sec: float, temperature_c: float = 20.0,
                pressure_atm: float = 1.0, rel_humidity: float = 50.0,
                gain_db: float = 0.0, block_len: int = 240) -> np.ndarray:
    """1音源×静止無指向1chマイクの物理適用済みモノラルを返す。

    Args:
        dry: ドライ音源（fs サンプリング）
        waypoints: (M,4) [[t,x,y,z],...] 音源軌道
        mic_pos: (3,) マイク位置
        fs: サンプリングレート [Hz]
        clip_len_sec: 出力長 [s]
        block_len: 大気吸収FIRの更新間隔（サンプル）
    """
    c = sound_speed(temperature_c)
    n = int(round(clip_len_sec * fs))
    tr = np.arange(n) / fs
    mic = np.asarray(mic_pos, dtype=np.float64)

    # ① 放射時刻と放射位置
    te, ps_te = solve_emission_times(tr, np.asarray(waypoints), mic, c)
    d = ps_te - mic[None, :]
    dist = np.sqrt(np.sum(d * d, axis=1))

    # ② ドライ読み出し（線形補間、loop=False 相当）
    dry = np.asarray(dry, dtype=np.float64) * 10.0 ** (gain_db / 20.0)
    pos = te * fs
    valid = np.isfinite(pos) & (pos >= 0.0) & (pos < len(dry) - 1)
    i0 = np.zeros(n, dtype=np.int64)
    i0[valid] = np.floor(pos[valid]).astype(np.int64)
    frac = np.zeros(n)
    frac[valid] = pos[valid] - i0[valid]
    s = np.zeros(n)
    s[valid] = (1.0 - frac[valid]) * dry[i0[valid]] + frac[valid] * dry[i0[valid] + 1]

    # ③ 幾何減衰 1/r
    g = np.ones(n)
    ok = valid & (dist > 0)
    g[ok] = 1.0 / dist[ok]
    s = s * g

    # ④ 大気吸収（DSと同一の周波数グリッド・係数・FIR設計、ブロック毎に更新）
    freqs = np.linspace(0.0, fs / 2.0, num=FIR_LEN - 1)
    alpha = attenuation_coefficients(
        frequency=freqs, temperature=temperature_c + 273.15,
        relative_humidity=rel_humidity, pressure=pressure_atm * 101.325)

    out = np.zeros(n)
    hist = np.zeros(FIR_LEN - 1)  # 直前ブロック末尾の入力（FIRの状態）
    ref_dist = np.where(np.isfinite(dist), dist, 0.0)
    for b0 in range(0, n, block_len):
        b1 = min(b0 + block_len, n)
        seg = np.concatenate([hist, s[b0:b1]])
        d_blk = ref_dist[b0:b1]
        d_use = float(np.median(d_blk[d_blk > 0])) if np.any(d_blk > 0) else 0.0
        if d_use > 0:
            coeff = 10.0 ** (-alpha * d_use / 20.0)
            h = firwin2(FIR_LEN, freqs, coeff, fs=fs)
            out[b0:b1] = fftconvolve(seg, h, mode="full")[
                FIR_LEN - 1: FIR_LEN - 1 + (b1 - b0)]
        # 音が全く届いていないブロックは 0 のまま（DSはbufferを進めない＝出力0）
        hist = seg[-(FIR_LEN - 1):]

    return out
