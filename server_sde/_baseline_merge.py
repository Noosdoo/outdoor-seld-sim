# -*- coding: utf-8 -*-
"""比べる相手 D（事前学習だけの PSELDNets・170 クラス × 3 トラック）の出力を、この研究の 9 クラス × 3 トラックへ写すときの重複の整理
（Astra R73-01・2026-10-20・Fable）。torch を使わない純 numpy なので手元で検査できる。

決まり（変更票 #73 に書く）:
  1. 対象クラスごとに、元クラス × トラックの候補（x, y, z・長さ = 確からしさ）を集める。
  2. 長さの大きい順に見て、既に残した候補と向きが MERGE_DEG（既定 20° = ヒステリシスの対応幅と同じ）以内なら「同じ音の複製」として落とす。
  3. 向きが違う候補は別の音として残し、残す数は N_TRACK（3）まで（長さの大きい順）。
  4. 長さが 0 の候補は見ない。ヒステリシス（入り／保持）はこのあと 9 クラスの doa に掛ける（v17 と同じ関数）。
使い方: python server_sde/_baseline_merge.py --selftest
"""
from __future__ import annotations

import math
import sys

import numpy as np

MERGE_DEG = 20.0
N_TRACK = 3


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def merge_frame_idx(cands: np.ndarray, merge_deg: float = MERGE_DEG, n_track: int = N_TRACK) -> tuple[np.ndarray, np.ndarray]:
    """cands: (n, 3) の x,y,z（長さ = 確からしさ）→ (out (n_track, 3), idx (n_track,))。重複をまとめ、向きの違う音を長さ順に n_track まで残す。
    idx は残した候補の行番号（残していない枠は −1）。Astra RI73-03（2026-10-21）: 距離を同じ音から付けるために、どの候補を残したかを返す。"""
    out = np.zeros((n_track, 3), dtype=np.float32)
    idx = np.full((n_track,), -1, dtype=np.int32)
    if cands.size == 0:
        return out, idx
    nrm = np.linalg.norm(cands, axis=1)
    order = np.argsort(-nrm, kind="stable")
    kept: list[np.ndarray] = []
    kept_i: list[int] = []
    for i in order:
        if nrm[i] <= 0:
            break
        v = cands[i]
        if any(_angle_deg(v, k) <= merge_deg for k in kept):
            continue                                            # 同じ向き = 同じ音の複製（元クラス違い・トラック違い）→ 落とす
        kept.append(v); kept_i.append(int(i))
        if len(kept) >= n_track:
            break
    for t, (v, i) in enumerate(zip(kept, kept_i)):
        out[t] = v; idx[t] = i
    return out, idx


def merge_frame(cands: np.ndarray, merge_deg: float = MERGE_DEG, n_track: int = N_TRACK) -> np.ndarray:
    """cands: (n, 3) → (n_track, 3)。merge_frame_idx の出力だけ（従来の形）。"""
    return merge_frame_idx(cands, merge_deg, n_track)[0]


def map_tracks(v: np.ndarray, src_of: dict[int, list[int]], n_ours: int = 9, merge_deg: float = MERGE_DEG, n_track: int = N_TRACK, return_src: bool = False):
    """v: (n_track_src, T, 3, C_src) → (n_track, T, 4*n_ours) の doa（x,y,z の 3 ブロック＋距離ブロック 0）。対象クラスごとに merge_frame。
    return_src=True なら (doa, src) を返す。src: (n_track, T, n_ours) int32 = 残した候補の**元トラック番号**（v の第 0 軸・残していない枠は −1）。
    相手 B はこれで同じ音の距離を付ける（RI73-03: 向きの内積で選ぶと別の音の距離が付く）。"""
    T = v.shape[1]
    out = np.zeros((n_track, T, 4 * n_ours), dtype=np.float32)
    src = np.full((n_track, T, n_ours), -1, dtype=np.int32)
    for ours, idxs in src_of.items():
        cand = v[:, :, :, idxs]                                   # (n_track_src, T, 3, n_src)
        n_src = len(idxs)
        for k in range(T):
            c = cand[:, k, :, :].transpose(0, 2, 1).reshape(-1, 3)   # (n_track_src*n_src, 3)・行 r = 元トラック r // n_src・元クラス idxs[r % n_src]
            m, idx = merge_frame_idx(c, merge_deg, n_track)       # (n_track, 3)・(n_track,)
            for t in range(n_track):
                for b in range(3):
                    out[t, k, b * n_ours + ours] = m[t, b]
                if idx[t] >= 0:
                    src[t, k, ours] = idx[t] // n_src
    return (out, src) if return_src else out


def attach_distance(doa: np.ndarray, src: np.ndarray, dist: np.ndarray, n_ours: int = 9) -> np.ndarray:
    """距離ブロックを埋める（相手 B・Astra RI73-03）: doa (n_track, T, 4*n_ours)・src (n_track, T, n_ours) = map_tracks の return_src・
    dist (n_track_src, T, n_ours) = 元トラックごとの距離 → 残した候補と同じ元トラックの距離を doa[..., 3*n_ours + c] に入れる（負は 0・src < 0 は触らない）。"""
    out = doa.copy()
    n_track, T, _ = src.shape
    for t in range(n_track):
        for k in range(T):
            for c in range(n_ours):
                s = int(src[t, k, c])
                if s >= 0:
                    out[t, k, 3 * n_ours + c] = max(0.0, float(dist[s, k, c]))
    return out


def _vec(norm: float, az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([norm * math.cos(e) * math.cos(a), norm * math.cos(e) * math.sin(a), norm * math.sin(e)])


def selftest() -> int:
    ok = []
    # 1 同じ向きの複製 3 本（元クラス違い・トラック違い）→ 1 本
    m = merge_frame(np.stack([_vec(0.9, 30), _vec(0.8, 31), _vec(0.7, 29)]))
    ok.append(("同じ音の複製 3 本 → 1 本（長さ最大を残す）", int((np.linalg.norm(m, axis=1) > 0).sum()) == 1 and abs(np.linalg.norm(m[0]) - 0.9) < 1e-6))
    # 2 別方向の 2 音 → 2 本
    m = merge_frame(np.stack([_vec(0.9, 30), _vec(0.5, 120)]))
    ok.append(("別方向（90°）の 2 音 → 2 本", int((np.linalg.norm(m, axis=1) > 0).sum()) == 2))
    # 3 同じ対象へ写す親子ラベル（Car・Car passing by）が同じ車を高く出し、別方向の車を少し低く出す → 別方向の車が消えない
    cands = np.stack([_vec(0.95, 10), _vec(0.9, 11), _vec(0.85, 9), _vec(0.8, 12), _vec(0.6, 100)])
    m = merge_frame(cands)
    dirs = [math.degrees(math.atan2(r[1], r[0])) for r in m if np.linalg.norm(r) > 0]
    ok.append(("親子ラベル 4 本が同じ車・別方向の車 1 本 → 2 本（90° 側の車が残る）", len(dirs) == 2 and any(abs(d - 100) < 1 for d in dirs)))
    # 4 向きの違う 4 音 → 長さ順に 3 本
    m = merge_frame(np.stack([_vec(0.9, 0), _vec(0.8, 90), _vec(0.7, 180), _vec(0.6, 270)]))
    kept = sorted(np.linalg.norm(m, axis=1).round(3), reverse=True)
    ok.append(("向きの違う 4 音 → 長さの大きい 3 本", kept == [0.9, 0.8, 0.7]))
    # 5 長さ 0 は見ない・空は空
    ok.append(("長さ 0 と空の候補 → 全部 0", np.all(merge_frame(np.zeros((3, 3))) == 0) and np.all(merge_frame(np.zeros((0, 3))) == 0)))
    # 6 20° ちょうどは複製・21° は別
    m20 = merge_frame(np.stack([_vec(0.9, 0), _vec(0.8, 20)])); m21 = merge_frame(np.stack([_vec(0.9, 0), _vec(0.8, 21)]))
    ok.append(("20° 以内は複製・21° は別の音", int((np.linalg.norm(m20, axis=1) > 0).sum()) == 1 and int((np.linalg.norm(m21, axis=1) > 0).sum()) == 2))
    # 7 map_tracks: (3, T, 3, C) → (3, T, 36)・対象クラスの列にだけ値・距離ブロックは 0
    C = 6; v = np.zeros((3, 2, 3, C), dtype=np.float32)
    v[0, 0, :, 1] = _vec(0.9, 30); v[1, 0, :, 4] = _vec(0.8, 31); v[2, 0, :, 1] = _vec(0.5, 150)      # クラス 1・4 を同じ対象 0 へ・複製 + 別方向
    out = map_tracks(v, {0: [1, 4], 3: [2]}, n_ours=9)
    n0 = np.linalg.norm(out[:, 0, [0, 9, 18]], axis=1)                                                    # 対象 0 の x,y,z
    ok.append(("map_tracks: 対象 0 に 2 本（0.9・0.5）・ほかの列と距離ブロックは 0", sorted(n0.round(3), reverse=True) == [0.9, 0.5, 0.0]
               and np.all(out[:, :, 27:] == 0) and np.all(out[:, 1, :] == 0) and np.all(out[:, 0, [3, 12, 21]] == 0)))
    # 8 Astra RI73-03: 同じクラスの 2 音（0°・長さ 0.8・距離 2 m と 25°・0.95・10 m）→ 2 本とも残り、0° の音に 2 m・25° の音に 10 m（別の音の距離を付けない）
    v = np.zeros((3, 1, 3, 9), dtype=np.float32); v[0, 0, :, 4] = _vec(0.8, 0); v[1, 0, :, 4] = _vec(0.95, 25)
    dist = np.zeros((3, 1, 9), dtype=np.float32); dist[0, 0, 4] = 2.0; dist[1, 0, 4] = 10.0
    out, src = map_tracks(v, {c: [c] for c in range(9)}, n_ours=9, return_src=True)
    out = attach_distance(out, src, dist, 9)
    got = {}
    for t in range(3):
        xyz = out[t, 0, [4, 13, 22]]
        if np.linalg.norm(xyz) > 0:
            got[round(float(np.degrees(np.arctan2(xyz[1], xyz[0]))))] = float(out[t, 0, 27 + 4])
    ok.append(("RI73-03: 0° の音に 2 m・25° の音に 10 m（src で同じ元トラックの距離を付ける）", got == {0: 2.0, 25: 10.0} and src[0, 0, 4] == 1 and src[1, 0, 4] == 0))
    # 8b 同じ向きの複製（0°・0.8・2 m と 3°・0.9・7 m）→ 1 本（長さの大きい 3° を残し、その距離 7 m）
    v2 = np.zeros((3, 1, 3, 9), dtype=np.float32); v2[0, 0, :, 4] = _vec(0.8, 0); v2[1, 0, :, 4] = _vec(0.9, 3)
    d2 = np.zeros((3, 1, 9), dtype=np.float32); d2[0, 0, 4] = 2.0; d2[1, 0, 4] = 7.0
    o2, s2 = map_tracks(v2, {c: [c] for c in range(9)}, n_ours=9, return_src=True); o2 = attach_distance(o2, s2, d2, 9)
    ok.append(("RI73-03: 同じ向きの複製は長さの大きい方を残し、その音の距離（7 m）", int((np.linalg.norm(o2[:, 0, [4, 13, 22]], axis=1) > 0).sum()) == 1 and float(o2[0, 0, 31]) == 7.0 and s2[0, 0, 4] == 1))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 2)
