"""座標変換と放射時刻(emission time)計算の唯一の正規モジュール。

このプロジェクトの音・ラベル両方が本モジュールを通ることで整合を保証する。

規約（PSELDNets / DCASE と同一。実コードで確認済み）:
- 座標系: x=前方, y=左, z=上 の右手系。DynamicSound のワールド座標をそのまま使う。
- 方位角 azimuth  = atan2(y, x)          [deg] 反時計回り正, 範囲 (-180, 180]
- 仰角   elevation = atan2(z, hypot(x,y)) [deg] 上が正,      範囲 [-90, 90]
- 単位DOAベクトル u = (cos el cos az, cos el sin az, sin el) = (ux, uy, uz)

根拠:
- PSELDNets `src/preproc/preprocess.py:600-601`（L3DAS22 の xyz→az/el 変換）
- PSELDNets `src/data/data.py generate_spatial_samples`（az/el→xyz、FOAゲイン）
- PSELDNets 論文 Eq.(1)(3)

放射時刻の解法（DynamicSound `_simulation.py _compute_emission` と同一の物理・
論文 arXiv:2601.15433 Eq.(12)(13)）:
  区分等速セグメント [t0,t1) 上の音源 ps(te) = p0 + v·(te−t0) について
  ||pr − ps(te)|| = c·(tr − te) を te の2次方程式として解く。
  A = |v|² − c², B = 2(c²(tr−t0) − d0·v), C = |d0|² − c²(tr−t0)², d0 = pr − p0
  亜音速では物理解は te ≤ tr を満たす側の根（もう一方は二乗により生じた偽解）。
"""
from __future__ import annotations

import numpy as np

# DynamicSound acoustics.standards.ISO_9613_1_1993 と同一
SOUND_SPEED_20C = 343.2  # [m/s] at 20 degC


def sound_speed(temperature_c: float) -> float:
    """気温[degC]から音速[m/s]。DynamicSound の式 c=343.2*sqrt(TK/293.15) と同一。"""
    return SOUND_SPEED_20C * np.sqrt((temperature_c + 273.15) / 293.15)


def solve_emission_times(tr, waypoints, receiver_pos, c=SOUND_SPEED_20C):
    """受信時刻列 tr に対する放射時刻 te と放射位置 ps(te) を求める（ベクトル化）。

    Args:
        tr: (N,) 受信時刻 [s]
        waypoints: (M, 4) の配列 [[t, x, y, z], ...]（区分等速の折れ線軌道）
        receiver_pos: (3,) 受信点（静止マイク）
        c: 音速 [m/s]

    Returns:
        te: (N,) 放射時刻。解なし（まだ音が届いていない等）は NaN
        ps_te: (N, 3) 放射位置。解なし行は NaN
    """
    tr = np.asarray(tr, dtype=np.float64)
    wp = np.asarray(waypoints, dtype=np.float64)
    pr = np.asarray(receiver_pos, dtype=np.float64)
    assert wp.ndim == 2 and wp.shape[1] == 4, "waypoints must be (M,4): [t,x,y,z]"

    te = np.full(tr.shape, np.nan)
    ps_te = np.full(tr.shape + (3,), np.nan)

    for i in range(len(wp) - 1):
        t0, t1 = wp[i, 0], wp[i + 1, 0]
        p0, p1 = wp[i, 1:4], wp[i + 1, 1:4]
        v = (p1 - p0) / (t1 - t0)
        d0 = pr - p0

        # 論文 Eq.(12): A te'^2 + B te' + C = 0, te' = te - t0
        A = float(v @ v) - c * c
        B = 2.0 * (c * c * (tr - t0) - float(d0 @ v))
        C = float(d0 @ d0) - (c * (tr - t0)) ** 2

        unresolved = np.isnan(te) & (tr >= t0)
        if abs(A) < 1e-12:  # |v| == c の退化ケース（通常起きない）
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = np.where(B != 0.0, -C / B, np.nan) + t0
            ok = unresolved & (cand >= t0) & (cand < t1) & (cand <= tr)
            te[ok] = cand[ok]
        else:
            disc = B * B - 4.0 * A * C
            with np.errstate(invalid="ignore"):
                sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
            for sign in (-1.0, +1.0):
                cand = (-B + sign * sq) / (2.0 * A) + t0
                ok = unresolved & np.isfinite(cand) \
                    & (cand >= t0) & (cand < t1) & (cand <= tr + 1e-12)
                te[ok] = cand[ok]
                unresolved &= np.isnan(te)

        solved_here = np.isfinite(te) & np.isnan(ps_te[:, 0]) & (te >= t0) & (te < t1)
        if np.any(solved_here):
            ps_te[solved_here] = p0[None, :] + (te[solved_here, None] - t0) * v[None, :]

    return te, ps_te


def doa_unit_vectors(ps, receiver_pos):
    """受信点から見た音源方向の単位ベクトル u = (ps − pr)/||ps − pr||。

    Args:
        ps: (N, 3) 音源位置（放射時刻補正済みを渡すこと）
        receiver_pos: (3,)
    Returns:
        u: (N, 3), dist: (N,)
    """
    ps = np.atleast_2d(np.asarray(ps, dtype=np.float64))
    pr = np.asarray(receiver_pos, dtype=np.float64)
    d = ps - pr[None, :]
    dist = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = d / dist[:, None]
    return u, dist


def unit_to_azel_deg(u):
    """単位方向ベクトル → (azimuth[deg], elevation[deg])。DCASE規約。"""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    az = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    el = np.degrees(np.arctan2(u[:, 2], np.hypot(u[:, 0], u[:, 1])))
    return az, el


def azel_deg_to_unit(az_deg, el_deg):
    """(azimuth[deg], elevation[deg]) → 単位方向ベクトル (N,3)。DCASE規約。"""
    az = np.deg2rad(np.atleast_1d(np.asarray(az_deg, dtype=np.float64)))
    el = np.deg2rad(np.atleast_1d(np.asarray(el_deg, dtype=np.float64)))
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def apparent_azel_deg(tr, waypoints, receiver_pos, c=SOUND_SPEED_20C):
    """受信時刻 tr における「見かけの方向」(放射時刻補正済みDOA) を返す。

    Returns:
        az, el: (N,) [deg]。音が未到達の時刻は NaN
        te: (N,) 放射時刻
        dist: (N,) 放射位置までの距離
    """
    te, ps_te = solve_emission_times(tr, waypoints, receiver_pos, c)
    u, dist = doa_unit_vectors(ps_te, receiver_pos)
    az, el = unit_to_azel_deg(u)
    return az, el, te, dist
