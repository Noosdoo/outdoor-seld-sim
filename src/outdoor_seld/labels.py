"""DCASE 形式 SELD ラベル（メタデータ CSV）の生成。

形式（PSELDNets `src/utils/data_utilities.py load_output_format_file` の5列形式）:
    frame, class_idx, track_idx, azimuth_deg(int), elevation_deg(int)
- ヘッダなし・カンマ区切り・0.1秒/フレーム
- azimuth/elevation は int（DCASE準拠）。角度規約は geometry.py と同一
- DOA は「受信時刻における見かけの方向」（放射時刻補正済み）を用いる
  → 音（FOAエンコードに使う方向）とラベルが同一の計算経路で整合する
"""
from __future__ import annotations

import numpy as np

from . import ablate
from .geometry import SOUND_SPEED_20C, apparent_azel_deg


def frame_label_rows(waypoints, receiver_pos, clip_len_sec: float,
                     class_idx: int, track_idx: int = 0,
                     source_active_from: float = 0.0,
                     source_active_until: float | None = None,
                     label_res: float = 0.1, c: float = SOUND_SPEED_20C):
    """1音源分のフレームラベル行を生成する。

    フレーム k の代表時刻は中心 (k+0.5)*label_res。
    「音が届いている」（放射時刻の解が存在し、かつ放射時刻が音源の発音区間内）
    フレームのみ行を出す。到達前フレームはラベルなし＝物理的に正しい。

    Args:
        waypoints: (M,4) [[t,x,y,z],...] 音源軌道
        receiver_pos: (3,) マイク位置
        clip_len_sec: クリップ長 [s]
        class_idx: クラスID
        track_idx: トラックID（単一音源なら0）
        source_active_until: 音源が発音を止める時刻（Noneなら軌道終端まで）
    Returns:
        rows: list of (frame, class_idx, track_idx, az_int, el_int)
        debug: dict {t_frames, az, el, te, dist}（可視化用の連続値）
    """
    n_frames = int(round(clip_len_sec / label_res))
    t_frames = (np.arange(n_frames) + 0.5) * label_res   # 各フレームの代表(中心)時刻
    # 音とFOAエンコードで使うのと同じ関数を呼ぶ＝方向がズレない一番の理由
    az, el, te, dist = apparent_azel_deg(t_frames, waypoints, receiver_pos, c)

    active = np.isfinite(az)                 # 音がまだ届いていない(NaN)フレームは除外
    if ablate.MODE == "no_doppler":
        # 【ablation第5版・P1規約】no_dopplerの音声は dry(tr − delay_ref) の
        # 一定遅延読み出しなので、発音区間の判定も同じ規約で行う。
        # 方向・距離の値は放射時刻ベースのまま（FOAエンコードと同一経路）で、
        # 変わるのは「どのフレームに行があるか」だけ。
        from .fastsim import nodop_delay
        delay_ref = nodop_delay(t_frames, waypoints, receiver_pos,
                                ablate.NODOP_FS, clip_len_sec, c=c)
        t_nd = t_frames - delay_ref
        active &= np.isfinite(t_nd) & (t_nd >= source_active_from)
        if source_active_until is not None:
            active &= t_nd <= source_active_until
    else:
        active &= te >= source_active_from   # 発音開始前に発射された音は無視（v3のスパース発音用）
        if source_active_until is not None:
            active &= te <= source_active_until  # 発音終了後に発射された音も無視

    rows = []
    for k in range(n_frames):
        if not active[k]:
            continue                          # 無音フレームはCSVに行を書かない
        az_i = int(np.rint(az[k]))            # DCASE規約に合わせて整数角度に丸める
        el_i = int(np.rint(el[k]))
        if az_i == 180:
            az_i = -180                       # 範囲を(-180,180]に統一（180は-180と同じ意味）
        rows.append((k, class_idx, track_idx, az_i, el_i))
    debug = {"t_frames": t_frames, "az": az, "el": el, "te": te, "dist": dist}
    return rows, debug


def write_dcase_csv(path, rows):
    """行リストを DCASE メタデータ CSV として書き出す。"""
    with open(path, "w", newline="\n") as f:
        for frame, cls, track, az, el in rows:
            f.write(f"{frame},{cls},{track},{az},{el}\n")   # ヘッダなし・カンマ区切り5列


def read_dcase_csv(path):
    """DCASE メタデータ CSV を {frame: [(class, az, el), ...]} に読む（照合用）。"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            frame = int(vals[0])
            # 同じフレームに複数音源(複数行)がありうるのでリストに追加していく
            out.setdefault(frame, []).append(
                (int(vals[1]), float(vals[3]), float(vals[4])))
    return out
