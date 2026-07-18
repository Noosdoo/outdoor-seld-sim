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

# DynamicSound acoustics.standards.ISO_9613_1_1993 と同一の基準音速
SOUND_SPEED_20C = 343.2  # [m/s] at 20 degC


def sound_speed(temperature_c: float) -> float:
    """気温[degC]から音速[m/s]。DynamicSound の式 c=343.2*sqrt(TK/293.15) と同一。"""
    # 絶対温度に変換してから基準(293.15K=20degC)との比の平方根をかけるだけ
    return SOUND_SPEED_20C * np.sqrt((temperature_c + 273.15) / 293.15)


def receiver_positions_at(tr, receiver):
    """受信点（マイク）の位置を各時刻について返す（歩行マイク対応の共通ヘルパ）。

    Args:
        tr: (N,) 時刻 [s]
        receiver: (3,) 静止マイク位置 か、(M,4) [[t,x,y,z],...] の折れ線軌道
                  （歩行マイク。区分等速、範囲外は端の値で一定）
    Returns:
        (N,3) 各時刻のマイク位置
    """
    r = np.asarray(receiver, dtype=np.float64)
    tr = np.asarray(tr, dtype=np.float64)
    if r.ndim == 1:
        return np.broadcast_to(r, tr.shape + (3,))
    assert r.ndim == 2 and r.shape[1] == 4, "receiver must be (3,) or (M,4)"
    out = np.empty(tr.shape + (3,))
    for k in range(3):
        out[..., k] = np.interp(tr, r[:, 0], r[:, k + 1])
    return out


def solve_emission_times(tr, waypoints, receiver_pos, c=SOUND_SPEED_20C):
    """受信時刻列 tr に対する放射時刻 te と放射位置 ps(te) を求める（ベクトル化）。

    「マイクが時刻trに音を受け取るには、音源は過去のいつ（te）その音を出したか」
    を解く。音源が動いているので te は単純な距離/音速の引き算では求まらず、
    音源軌道の区分（セグメント）ごとに2次方程式を解く必要がある。

    Args:
        tr: (N,) 受信時刻 [s]
        waypoints: (M, 4) の配列 [[t, x, y, z], ...]（区分等速の折れ線軌道）
        receiver_pos: (3,) 受信点（静止マイク）、または (M,4) のマイク軌道
                      （歩行マイク。2026-07-16拡張）。移動マイクでは
                      「時刻trにマイクがいる位置」で受信するとして
                      ||pr(tr) − ps(te)|| = c·(tr − te) を解く（媒質固定系で厳密。
                      観測者移動のドップラーも遅延の時間変化として自然に出る）
        c: 音速 [m/s]

    Returns:
        te: (N,) 放射時刻。解なし（まだ音が届いていない等）は NaN
        ps_te: (N, 3) 放射位置。解なし行は NaN
    """
    tr = np.asarray(tr, dtype=np.float64)
    wp = np.asarray(waypoints, dtype=np.float64)
    pr = np.asarray(receiver_pos, dtype=np.float64)
    assert wp.ndim == 2 and wp.shape[1] == 4, "waypoints must be (M,4): [t,x,y,z]"

    if pr.ndim == 2:
        # 歩行マイク: 受信位置が時刻依存。係数のうち d0 が (N,3) になるだけで
        # 2次方程式の構造は静止版と同一（静止版のコードパスには手を触れない）
        return _solve_emission_times_moving_receiver(tr, wp, pr, c)

    # 最終的な答え。まだ解けていないところはNaNのまま（＝音が未到達）
    te = np.full(tr.shape, np.nan)
    ps_te = np.full(tr.shape + (3,), np.nan)

    # 軌道は折れ線なので、区間(セグメント)ごとに「その区間内で等速直線運動」とみなして解く
    for i in range(len(wp) - 1):
        t0, t1 = wp[i, 0], wp[i + 1, 0]          # この区間の開始・終了時刻
        p0, p1 = wp[i, 1:4], wp[i + 1, 1:4]       # この区間の開始・終了位置
        v = (p1 - p0) / (t1 - t0)                 # この区間での音源速度ベクトル
        d0 = pr - p0                              # 区間開始時点でのマイク→音源始点の相対位置

        # 論文 Eq.(12): A te'^2 + B te' + C = 0 （te' = te - t0 を解く。te'なら数値が0付近で安定する）
        A = float(v @ v) - c * c                  # |v|^2 - c^2（音速との速度差）
        B = 2.0 * (c * c * (tr - t0) - float(d0 @ v))
        C = float(d0 @ d0) - (c * (tr - t0)) ** 2

        # まだ解けておらず、かつこの区間が受信時刻より前に始まっている行だけを対象にする
        unresolved = np.isnan(te) & (tr >= t0)
        if abs(A) < 1e-12:  # |v| == c の退化ケース（音速と同じ速さで動く。通常起きない）
            # 2次式でなく1次式 B*te'+C=0 として解く
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = np.where(B != 0.0, -C / B, np.nan) + t0
            # 解がこのセグメントの時間範囲内かつ因果的(未来の発射でない)ものだけ採用
            ok = unresolved & (cand >= t0) & (cand < t1) & (cand <= tr)
            te[ok] = cand[ok]
        else:
            disc = B * B - 4.0 * A * C             # 判別式
            with np.errstate(invalid="ignore"):
                sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))  # 実解がない行はNaNのまま
            for sign in (-1.0, +1.0):               # 2次方程式の解は2つ、両方試す
                cand = (-B + sign * sq) / (2.0 * A) + t0
                # 有効範囲: このセグメント内・かつ発射時刻が受信時刻を超えない(因果律)
                ok = unresolved & np.isfinite(cand) \
                    & (cand >= t0) & (cand < t1) & (cand <= tr + 1e-12)
                te[ok] = cand[ok]
                unresolved &= np.isnan(te)          # 採用した行は次のsignループで対象から外す

        # このセグメントで解けた行について、放射時刻での音源位置も計算して保存
        solved_here = np.isfinite(te) & np.isnan(ps_te[:, 0]) & (te >= t0) & (te < t1)
        if np.any(solved_here):
            ps_te[solved_here] = p0[None, :] + (te[solved_here, None] - t0) * v[None, :]

    return te, ps_te


def _solve_emission_times_moving_receiver(tr, wp, mic_wp, c):
    """solve_emission_times の歩行マイク版（receiver_pos が (M,4) 軌道のとき）。

    静止版との違いは d0 = pr(tr) − p0 が行ごとに変わる点のみ。B, C が
    行ごとのドット積になるが、根の選択・因果律・セグメント処理は静止版と同一。
    """
    pr_tr = receiver_positions_at(tr, mic_wp)     # (N,3) 各受信時刻のマイク位置
    te = np.full(tr.shape, np.nan)
    ps_te = np.full(tr.shape + (3,), np.nan)

    for i in range(len(wp) - 1):
        t0, t1 = wp[i, 0], wp[i + 1, 0]
        p0, p1 = wp[i, 1:4], wp[i + 1, 1:4]
        v = (p1 - p0) / (t1 - t0)
        d0 = pr_tr - p0[None, :]                  # (N,3) 行ごとの相対位置

        A = float(v @ v) - c * c                  # スカラのまま
        B = 2.0 * (c * c * (tr - t0) - d0 @ v)    # (N,)
        C = np.einsum("ij,ij->i", d0, d0) - (c * (tr - t0)) ** 2

        unresolved = np.isnan(te) & (tr >= t0)
        if abs(A) < 1e-12:
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
    if pr.ndim == 2:
        d = ps - pr                               # 歩行マイク: 行ごとのマイク位置 (N,3)
    else:
        d = ps - pr[None, :]                      # マイクから音源への相対ベクトル
    dist = np.linalg.norm(d, axis=1)               # その長さ＝距離
    with np.errstate(invalid="ignore", divide="ignore"):
        u = d / dist[:, None]                      # 長さ1に正規化＝方向だけを取り出す
    return u, dist


def unit_to_azel_deg(u):
    """単位方向ベクトル → (azimuth[deg], elevation[deg])。DCASE規約。"""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    az = np.degrees(np.arctan2(u[:, 1], u[:, 0]))              # y,xから水平角
    el = np.degrees(np.arctan2(u[:, 2], np.hypot(u[:, 0], u[:, 1])))  # zと水平成分から仰角
    return az, el


def azel_deg_to_unit(az_deg, el_deg):
    """(azimuth[deg], elevation[deg]) → 単位方向ベクトル (N,3)。DCASE規約。"""
    az = np.deg2rad(np.atleast_1d(np.asarray(az_deg, dtype=np.float64)))
    el = np.deg2rad(np.atleast_1d(np.asarray(el_deg, dtype=np.float64)))
    # 球面座標→直交座標の標準的な変換（unit_to_azel_degの逆変換）
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def apparent_azel_deg(tr, waypoints, receiver_pos, c=SOUND_SPEED_20C):
    """受信時刻 tr における「見かけの方向」(放射時刻補正済みDOA) を返す。

    音とラベルの両方がこの関数を呼ぶことで、方向のズレが起きない設計になっている。

    Returns:
        az, el: (N,) [deg]。音が未到達の時刻は NaN
        te: (N,) 放射時刻
        dist: (N,) 放射位置までの距離
    """
    te, ps_te = solve_emission_times(tr, waypoints, receiver_pos, c)  # まずいつ発射されたかを解く
    pr = np.asarray(receiver_pos, dtype=np.float64)
    if pr.ndim == 2:
        # 歩行マイク: 各受信時刻のマイク位置から見た方向（DCASE規約はマイク中心系）
        u, dist = doa_unit_vectors(ps_te, receiver_positions_at(tr, pr))
    else:
        u, dist = doa_unit_vectors(ps_te, receiver_pos)              # その発射位置から方向ベクトルを作る
    az, el = unit_to_azel_deg(u)                                     # 角度[deg]に変換
    return az, el, te, dist
