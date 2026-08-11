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

from .geometry import (receiver_positions_at, solve_emission_times,
                       sound_speed)

FIR_LEN = 513  # DynamicSound と同一（大気吸収フィルタのタップ数）


def _nodop_ref_from(te, ps_te):
    """no-doppler規約の参照点（中央放射時刻の音源位置）。解が無ければNone。"""
    finite = np.isfinite(te)
    if not np.any(finite):
        return None
    te_ref = float(np.median(te[finite]))
    i_ref = int(np.argmin(np.abs(np.where(finite, te, np.inf) - te_ref)))
    return ps_te[i_ref]


def nodop_delay(tr_query, waypoints, mic_pos, fs, clip_len_sec,
                c=None, temperature_c: float = 20.0):
    """no-doppler規約の読み出し遅延 delay_ref(tr) を返す（render_monoと同一規約）。

    ラベル生成（labels.py）がこの関数を使うことで、音声のドライ読み出し
    dry(tr − delay_ref) と発音区間判定の時刻規約が厳密に一致する。
    参照点は render_mono と同じく音声サンプル格子（fs）上の放射時刻から選ぶ。
    """
    if c is None:
        c = sound_speed(temperature_c)
    n = int(round(clip_len_sec * fs))
    tr = np.arange(n) / fs
    mic = np.asarray(mic_pos, dtype=np.float64)
    te, ps_te = solve_emission_times(tr, np.asarray(waypoints, np.float64),
                                     mic, c)
    ps_ref = _nodop_ref_from(te, ps_te)
    tr_query = np.asarray(tr_query, dtype=np.float64)
    if ps_ref is None:
        return np.full(tr_query.shape, np.nan)
    mic_at = (receiver_positions_at(tr_query, mic) if mic.ndim == 2
              else np.broadcast_to(mic, tr_query.shape + (3,)))
    return np.linalg.norm(mic_at - ps_ref[None, :], axis=1) / c


def render_mono(dry: np.ndarray, waypoints, mic_pos, fs: int,
                clip_len_sec: float, temperature_c: float = 20.0,
                pressure_atm: float = 1.0, rel_humidity: float = 50.0,
                gain_db: float = 0.0, block_len: int = 240,
                enable_doppler: bool = True,
                enable_spreading: bool = True,
                enable_air_absorption: bool = True) -> np.ndarray:
    """1音源×静止無指向1chマイクの物理適用済みモノラルを返す。

    Args:
        dry: ドライ音源（fs サンプリング）
        waypoints: (M,4) [[t,x,y,z],...] 音源軌道
        mic_pos: (3,) マイク位置
        fs: サンプリングレート [Hz]
        clip_len_sec: 出力長 [s]
        block_len: 大気吸収FIRの更新間隔（サンプル）
        enable_doppler / enable_spreading / enable_air_absorption:
            ablation用の物理スイッチ（既定は全ON=従来と完全同一の出力）。
            - enable_doppler=False: ドライ読み出しを「一定遅延」（全サンプル共通の
              中央値伝搬遅延）にする。時間伸縮（ピッチ変調）が消えるが、1/rと
              大気吸収は放射時刻ベースの距離のまま時変で残る。
              注意①: 静的RIR補間型の生成器（SpatialScaper等）も「ピッチ変調が
              消える」点は同じだが、遅延の扱いは異なる（あちらは区分的時変遅延）。
              「既存生成器の再現」とまでは言えない（レビューP10）。
              注意②: 一定遅延のため音のオンセット受信時刻がラベル窓（放射時刻
              基準）から最大0.1-0.3秒ずれる（レビューP1）。ablationでこの条件の
              データセットを作る際はラベル側も同じ一定遅延規約で生成すること。
            - enable_spreading=False: 幾何減衰 1/r を適用しない（距離によらず1倍）。
            - enable_air_absorption=False: ISO 9613-1 の大気吸収FIRを適用しない。
            地面反射のon/offは本関数の外（鏡像軌道レンダの加算有無）で制御する。
    """
    c = sound_speed(temperature_c)
    n = int(round(clip_len_sec * fs))
    tr = np.arange(n) / fs           # 出力の各サンプルに対応する受信時刻
    mic = np.asarray(mic_pos, dtype=np.float64)

    # ① 放射時刻と放射位置（全サンプル分を一括で解く。DynamicSoundは1サンプルずつ解く）
    te, ps_te = solve_emission_times(tr, np.asarray(waypoints), mic, c)
    if mic.ndim == 2:
        # 歩行マイク（mic_pos が (M,4) 軌道、2026-07-16拡張）: 距離は
        # 「受信時刻のマイク位置」から「放射時刻の音源位置」まで
        d = ps_te - receiver_positions_at(tr, mic)
    else:
        d = ps_te - mic[None, :]
    dist = np.sqrt(np.sum(d * d, axis=1))   # マイクから放射位置までの距離

    # ② ドライ読み出し（線形補間、loop=False 相当）
    dry = np.asarray(dry, dtype=np.float64) * 10.0 ** (gain_db / 20.0)  # dBをゲイン倍率に変換
    if enable_doppler:
        pos = te * fs                        # 放射時刻をドライ音源のサンプル位置に変換
    else:
        # ドップラーoff（2026-07-18 改訂・P1規約の実装）:
        # 「音源起因の時間伸縮だけを消し、観測者（歩行マイク）移動分は保持」
        # （設計= out/v9_design_v2_2026-07-16.md 8節の事前決定）。
        # 読み出し遅延 = ||mic(tr) − ps_ref|| / c。ps_ref は中央放射時刻での
        # 音源位置（固定点）なので、音源の移動はピッチに影響しなくなる。
        # マイクが動けば遅延は時変のまま＝観測者ドップラーは物理どおり残る。
        # 静止マイクでは一定遅延（旧実装の中央値遅延と同種）に退化する。
        # 振幅1/r・吸収は下の③④で従来どおり実距離の時変のまま
        ps_ref = _nodop_ref_from(te, ps_te)   # ラベル側(nodop_delay)と共通の参照点
        if ps_ref is not None:
            mic_at = (receiver_positions_at(tr, mic) if mic.ndim == 2
                      else np.broadcast_to(mic, (n, 3)))
            delay_ref = np.linalg.norm(mic_at - ps_ref[None, :], axis=1) / c
            pos = (tr - delay_ref) * fs
        else:
            pos = np.full(n, np.nan)
    valid = np.isfinite(pos) & (pos >= 0.0) & (pos < len(dry) - 1)  # 範囲内かつ解がある行だけ
    if not enable_doppler:
        # 未到達区間（te=NaN=音がまだ物理的に届いていない）はドップラーoffでも無音にする。
        # これが無いと到達前のサンプルが素通しになり、1/rも掛からない
        # （2026-07-14 敵対的レビューP2で発見・修正）
        valid &= np.isfinite(te)
    i0 = np.zeros(n, dtype=np.int64)
    i0[valid] = np.floor(pos[valid]).astype(np.int64)   # 整数側の読み出し位置
    frac = np.zeros(n)
    frac[valid] = pos[valid] - i0[valid]                 # 端数（線形補間の重み）
    s = np.zeros(n)                                      # 無効な行は0（＝まだ音が届いていない）
    s[valid] = (1.0 - frac[valid]) * dry[i0[valid]] + frac[valid] * dry[i0[valid] + 1]

    # ③ 幾何減衰 1/r（距離0=マイクと同じ位置は特例で1倍のまま）
    if enable_spreading:
        g = np.ones(n)
        ok = valid & np.isfinite(dist) & (dist > 0)
        g[ok] = 1.0 / dist[ok]
        s = s * g

    # ④ 大気吸収（DSと同一の周波数グリッド・係数・FIR設計、ブロック毎に更新）
    if not enable_air_absorption:
        # 吸収off: FIRを通さない。ただしFIR(線形位相513タップ)の群遅延
        # (FIR_LEN-1)/2=256サンプルだけonside出力は遅れるので、条件間で
        # 波形タイミングが揃うよう同量のゼロ遅延を入れて返す
        gd = (FIR_LEN - 1) // 2
        return np.concatenate([np.zeros(gd), s[:-gd]]) if n > gd else s
    freqs = np.linspace(0.0, fs / 2.0, num=FIR_LEN - 1)   # 0〜ナイキストの周波数グリッド
    # ISO9613-1の式から各周波数の減衰係数[dB/m]を計算（DynamicSound自身の関数を流用）
    alpha = attenuation_coefficients(
        frequency=freqs, temperature=temperature_c + 273.15,
        relative_humidity=rel_humidity, pressure=pressure_atm * 101.325)

    out = np.zeros(n)
    hist = np.zeros(FIR_LEN - 1)  # 直前ブロック末尾の入力（畳み込みをブロック間でつなぐための状態）
    ref_dist = np.where(np.isfinite(dist), dist, 0.0)
    for b0 in range(0, n, block_len):          # block_lenサンプルごとにFIRを設計し直す
        b1 = min(b0 + block_len, n)
        seg = np.concatenate([hist, s[b0:b1]])  # 前ブロックの末尾をつなげてフィルタの遅延を継続
        d_blk = ref_dist[b0:b1]
        # このブロック区間の代表距離（中央値）を1つ決める。距離は滑らかにしか変わらないため
        # ブロック内で一定とみなしても誤差は無視できる（docstring参照）
        d_use = float(np.median(d_blk[d_blk > 0])) if np.any(d_blk > 0) else 0.0
        if d_use > 0:
            coeff = 10.0 ** (-alpha * d_use / 20.0)     # dB減衰を振幅倍率に変換
            h = firwin2(FIR_LEN, freqs, coeff, fs=fs)   # 周波数特性からFIR係数を設計
            # 畳み込んでから、履歴分のオフセットを除いてこのブロック分だけ切り出す
            out[b0:b1] = fftconvolve(seg, h, mode="full")[
                FIR_LEN - 1: FIR_LEN - 1 + (b1 - b0)]
        # 音が全く届いていないブロックは 0 のまま（DSはbufferを進めない＝出力0）
        hist = seg[-(FIR_LEN - 1):]   # 次のブロックのために末尾を履歴として残す

    return out
