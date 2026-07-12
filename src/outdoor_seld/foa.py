"""解析エンコードによる FOA (一次アンビソニックス) 生成。

規約: チャンネル順 **W, Y, Z, X**（ACN）・**SN3D** 正規化。
PSELDNets の学習データ（DCASE FOA）および PSELDNets 論文 Eq.(1)(3) と同一:
    W = p
    Y = p * sin(az) cos(el) = p * uy
    Z = p * sin(el)         = p * uz
    X = p * cos(az) cos(el) = p * ux
（PSELDNets `src/data/data.py generate_spatial_samples` がまさにこの順で
 np.stack((w, y*audio, z*audio, x*audio)) を構成している）

物理適用済みモノラル圧力信号 p(t) に、受信時刻ごとの見かけDOAの
単位ベクトル u(t) をゲインとして掛けるだけでよい（トリガ関数は不要、
u の成分がそのまま球面調和ゲインになる）。
"""
from __future__ import annotations

import numpy as np

from .geometry import azel_deg_to_unit


def encode_foa_timevarying(mono: np.ndarray, u: np.ndarray) -> np.ndarray:
    """時変DOAで FOA 4ch を生成する。

    Args:
        mono: (N,) 物理適用済みモノラル信号（マイク位置の音圧）
        u: (N, 3) 受信時刻ごとの見かけDOA単位ベクトル (ux, uy, uz)。
           NaN 行（音が未到達）はゲイン0として扱う。
    Returns:
        foa: (4, N) [W, Y, Z, X]
    """
    mono = np.asarray(mono, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    assert u.shape == (mono.shape[0], 3), f"u shape {u.shape} != ({mono.shape[0]}, 3)"
    u = np.where(np.isfinite(u), u, 0.0)   # 音が届いていない(NaN)行は寄与ゼロにする
    w = mono                                # Wは無指向＝方向によらずモノラル音そのまま
    y = mono * u[:, 1]                      # Y chは左右成分(uy)に比例した感度
    z = mono * u[:, 2]                      # Z chは上下成分(uz)に比例した感度
    x = mono * u[:, 0]                      # X chは前後成分(ux)に比例した感度
    return np.stack([w, y, z, x], axis=0)   # チャンネル順は規約通りW,Y,Z,X


def encode_foa_static(mono: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """固定DOAで FOA 4ch を生成する（サニティチェック用）。"""
    u = azel_deg_to_unit(az_deg, el_deg)[0]         # 固定角度を単位ベクトルに変換
    N = len(mono)
    return encode_foa_timevarying(mono, np.tile(u, (N, 1)))  # 全フレーム同じuを使うだけ


def intensity_vector_doa(foa: np.ndarray, fs: int, frame_sec: float = 0.1,
                         nfft: int = 1024, hop: int = 240,
                         fmin: float = 200.0, fmax: float = 4000.0):
    """FOA から音響インテンシティベクトル法でフレームごとのDOAを推定する。

    ラベルとの独立照合用（Step 5 サニティチェック2）。
    I = Re{ conj(W) * [X, Y, Z] } を周波数帯 [fmin, fmax] で積算し、
    label_res 毎に平均して方向を求める。

    Args:
        foa: (4, N) [W, Y, Z, X]
    Returns:
        t_frames: (F,) 各フレーム中心時刻 [s]
        az_deg, el_deg: (F,) 推定DOA。無音フレームは NaN
        energy: (F,) フレームごとの |W|^2 合計（有効判定用）
    """
    from scipy.signal import stft

    w = foa[0]
    ych, zch, xch = foa[1], foa[2], foa[3]
    # 4chそれぞれを短時間フーリエ変換（時間×周波数の複素スペクトルにする）
    f, t, W = stft(w, fs=fs, nperseg=nfft, noverlap=nfft - hop, padded=False)
    _, _, X = stft(xch, fs=fs, nperseg=nfft, noverlap=nfft - hop, padded=False)
    _, _, Y = stft(ych, fs=fs, nperseg=nfft, noverlap=nfft - hop, padded=False)
    _, _, Z = stft(zch, fs=fs, nperseg=nfft, noverlap=nfft - hop, padded=False)

    band = (f >= fmin) & (f <= fmax)   # サイレンの主要な周波数帯だけを使う（帯域外雑音を無視）
    # 音響インテンシティ＝Wと各方向chの相関。方向のエネルギー流れを表す
    Ix = np.sum(np.real(np.conj(W[band]) * X[band]), axis=0)
    Iy = np.sum(np.real(np.conj(W[band]) * Y[band]), axis=0)
    Iz = np.sum(np.real(np.conj(W[band]) * Z[band]), axis=0)
    Ew = np.sum(np.abs(W[band]) ** 2, axis=0)   # フレームのエネルギー（無音判定に使う）

    frames_per_label = frame_sec * fs / hop     # ラベル1フレーム(0.1s)がSTFTの何コマ分か
    n_frames = int(np.floor(len(t) / frames_per_label))
    t_frames = np.zeros(n_frames)
    az = np.full(n_frames, np.nan)
    el = np.full(n_frames, np.nan)
    energy = np.zeros(n_frames)
    for k in range(n_frames):
        i0 = int(round(k * frames_per_label))
        i1 = int(round((k + 1) * frames_per_label))
        # ラベル1フレーム分のSTFTコマをまとめて積算し、方向を1つ決める
        ix, iy, iz = Ix[i0:i1].sum(), Iy[i0:i1].sum(), Iz[i0:i1].sum()
        t_frames[k] = (k + 0.5) * frame_sec
        energy[k] = Ew[i0:i1].sum()
        norm = np.sqrt(ix * ix + iy * iy + iz * iz)
        if norm > 0:   # エネルギーがゼロでなければ方向を計算（無音なら0のままNaN）
            az[k] = np.degrees(np.arctan2(iy, ix))
            el[k] = np.degrees(np.arctan2(iz, np.hypot(ix, iy)))
    return t_frames, az, el, energy
