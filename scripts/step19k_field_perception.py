# -*- coding: utf-8 -*-
"""step19k_field_perception.py — 実録の知覚層評価（変更票 #43・2026-09-28 本人「1 お願いします」・Fable）。

目印（2 m おき 11 点・x = −10〜+10）の通過時刻から相手の**軌道の正解**を作り、合成データと同じ知覚層の道具 `_perception_eval.py`
を実録の 10 秒クリップに掛ける。帯（1.5／3.2 m）に丸めない「装置の目の良さそのもの」の成績表（方位誤差・距離誤差・位置誤差・検出 F1・
動きの向き・距離帯／方向／歩行別）が、追加の録音なしで出る。

入力:
  --reads   step19h の読み（take,frame,cls,track,x,y[,z][,t]）。x,y は現場の座標 [m]（x = 南が正・y = 東が正）。車・自転車は step19l の reads_mid（車軸の中点に直したもの）を渡す。静止のマイクは ann の mic_x_m・mic_y_m（無ければ原点）。
            **歩行の本**は、歩く人（マイク）が目印を通った時刻も同じ表に書く: cls = mic・track = 0・x,y = その時刻の人の位置。
  --ann     step19b の ann_all.csv（clip_id, orig_file, cut_offset_s, scored, class[, mic_z, 状態, 三脚の向き]）
            #44: 静止は三脚の向き 0/180°（クラクション E2 は 90/270 も・H25-01）、歩行は時刻順の mic 読みから正面を決める。
            音声に三脚の向き（180°・E2 の 90°／270°）を足さず、方位だけをマイク基準にする。距離・時刻は不変。
  --out     出力 dir: gt/<clip_id>.csv（6 列 frame,cls,track,az,el,dist・学習の正解と同じ形）、manifest.csv（clip_id,split,motion,mic_z）、
            report.csv（clip ごとの行数・最接近・注意）。--pred があれば perception_field.md / .json も
  --mic-z   マイク高さの既定 [m]（ann の mic_z が優先）。--src-z 音源高さ（省略時はクラス別の既定〔#70 r16〕: car 0.325・kick 0.3・bike 0.6・bike_bell 1.0・bicycle 0.25）
  --pred    実録クリップの予測 csv（因果推論の val_all_causal.csv 形式）。あれば _perception_eval.py を呼ぶ
  --minframe 既定 40（暖機 4 s を除く・合成と同じ）
--warn-interval（2026-10-27・本人 12:35 (c)）: take,class,on_s,end_s の csv（全長の録音の秒・波形で読んだ鳴り始めと鳴り終わり）。あれば警告音（サイレン・クラクション・
          バック音・ベル）の正解を [鳴り始め, 鳴り終わり) にフレームの中心がある行だけに絞る（合成の「鳴っている間だけ」とそろえる）。区間の無い警告音の本があれば止める。無ければ今までどおり

正解の作り方: 同じ take・cls・track の点（t, x, y, z）を時刻順に並べて直線でつないだ軌道にし、マイクは静止なら (mic_x_m, mic_y_m, mic_z)（注釈に無ければ原点）、歩行なら mic 行の軌道
(t, x, y, mic_z)。クリップの時刻 = 原録音の時刻 − cut_offset_s。学習と同じ `frame_label_rows`（伝搬遅延込み・水平距離）で行を作る。
正解の不確かさ（結果に併記する）: 目印の読み ±0.1 s（3 m/s で ±0.3 m）、目印間は等速の近似、車・自転車は前輪の接地点で読み、step19l_when_where.py が車軸の中点に 1 回だけずらす（この道具には reads_mid_*.csv を渡す・変更票 #59 今の決まり・2026-10-12）、
歩行は人の位置の読み。合成の正解は誤差ゼロなので、実録の知覚層の数字はその分だけ悪く見える。

使い方:
  python scripts/step19k_field_perception.py --reads out/reads/reads_A.csv --ann out/realsmoke/ann_all.csv --out out/realsmoke/perception --mic-z 1.6 [--pred <csv>]
  python scripts/step19k_field_perception.py --selftest out/_test_step19k
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from _field_orientation import tripod_heading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.labels import frame_label_rows  # noqa: E402

CLASS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3, "car": 4, "car_drive": 4, "crossing": 5, "kick": 6, "kickboard": 6, "bike": 7, "motorcycle": 7, "bicycle": 8}   # bicycle = 2026-10-08 変更票 #48
SRC_Z = {4: 0.325, 6: 0.3, 7: 0.6, 3: 1.0, 8: 0.25}   # #70 r16（2026-10-19 本人）: 合成の範囲の中央値にそろえる（車 0.05〜0.6 → 0.325・自転車 0.1〜0.4 → 0.25・ベルは実物の高さ〔ann の高さ_m が優先・10/15 の記録は 1.0 m〕・未記入なら 1.0）。
                                                      # キック（旧 0.3）・バイク（旧 0.6）は実録でやめたので旧のまま。旧 = 車 0.5・ベル 0.6（2026-10-08）。車は --src-z 0.05／0.325／0.6 で仰角の感度を出す
CLIP = 10.0
LABEL_RES = 0.1
MIC_CLS = "mic"


def norm(name: str) -> str:
    s = name.strip().lower().replace("\\", "/").split("/")[-1]
    for ext in (".wav", ".flac", ".mov", ".mp4"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s


def cls_of(s: str) -> int:
    s = s.strip()
    if s.lstrip("-").isdigit():
        return int(s)
    if s not in CLASS_IDX:
        raise ValueError(f"unknown class name: {s!r}")
    return CLASS_IDX[s]


def _t_of(r: dict) -> float:
    if r.get("t") not in (None, "", "nan"):
        return float(r["t"])
    return int(r["frame"]) * LABEL_RES


from _field_hidden_fill import hidden_intervals, fill_hidden, strip_fill_flag, hold_ends, HIDE_MAX_S as FILL_MAX_S  # noqa: E402  変更票 #69 r31 ②・r34（2026-10-23）

HIDE_MAX_S = FILL_MAX_S                                # 変更票 #69 r31 ②（2026-10-23）: 隠れた区間は前後の読みに 2 次式を当てて埋める（上限 3 s・両側に読みがある区間だけ）。
                                                       #   埋められない区間（3 s 超・片側に読みが無い）には正解を作らない。旧 B4 R17-01（1 秒の線形）を置き換え
OLD_HIDE_MAX_S = 1.0                                   # 2026-10-25 Opus の案（Fable の確認待ち）: 旧の道（build の既定 = hidden_blocks・検査の固定値を守る）の上限は B4 R17-01 の 1 秒のまま。
                                                       #   10/23 に HIDE_MAX_S を 3 s にしたとき、旧の道まで 3 s になり test_step19_b2b4_static_hidden の K2・K3 が落ちていた（本番の CLI は新しい道）
RANGE_M = 10.0                                         # 変更票 #69 r31 ①: 知覚の正解の範囲（検出フレーム率の分母）= 道に沿って ±10（全部の日）
DIR_WIN = (2.5, 1.5)                                   # 向きの採点の窓 = 最接近の 2.5〜1.5 秒前（合成と同じ）。この窓の正解だけは ±12 の読み（と C の延長）も使う → gt_dirwin/


def hidden_blocks(reads: list[dict]) -> dict:
    """take -> {(cls, trk) か "mic": [(隠れる前の最後の読み, 見えてからの最初の読み)]}（1 秒より長い区間だけ・step19h の hidden = start／end の行）"""
    by = defaultdict(lambda: defaultdict(list)); times = defaultdict(lambda: defaultdict(list))
    for r in reads:
        take = norm(r["take"]); key = "mic" if r["cls"].strip().lower() == MIC_CLS else (cls_of(r["cls"]), int(r.get("track", 0) or 0))
        if (r.get("hidden") or "").strip():
            by[take][key].append((_t_of(r), (r.get("hidden") or "").strip()))
        else:
            times[take][key].append(_t_of(r))
    out = defaultdict(dict)
    for take, d in by.items():
        for key, lst in d.items():
            reads_t = sorted(times[take][key]); blocks, t_hide = [], None
            for t, kind in sorted(lst):
                if kind == "start":
                    if t_hide is not None:
                        raise ValueError(f"{take} {key}: 隠れの印が不完全（隠れ始め {t_hide:.2f} s のあと見え始めが無い・R72-05）")
                    t_hide = t
                elif kind == "end":
                    if t_hide is None:
                        raise ValueError(f"{take} {key}: 隠れの印が不完全（隠れ始めの無い見え始め {t:.2f} s・R72-05）")
                    if t - t_hide > OLD_HIDE_MAX_S:
                        blocks.append((max([x for x in reads_t if x <= t_hide], default=t_hide), min([x for x in reads_t if x >= t], default=t)))
                    t_hide = None
            if t_hide is not None:
                raise ValueError(f"{take} {key}: 隠れの印が不完全（隠れ始め {t_hide:.2f} s のあと見え始めが無い・押し忘れ・R72-05）")
            if blocks:
                out[take][key] = blocks
    return out


def group_reads(reads: list[dict]):
    """take -> ({(cls, trk): [(t, x, y, z|None)]}, [(t, x, y)] mic)。隠れた印の行（hidden・x 空）は位置に入れない（B4）"""
    src = defaultdict(lambda: defaultdict(list))
    mic = defaultdict(list)
    for r in reads:
        if (r.get("hidden") or "").strip():
            continue
        take = norm(r["take"])
        t = _t_of(r)
        x, y = float(r["x"]), float(r["y"])
        if r["cls"].strip().lower() == MIC_CLS:
            mic[take].append((t, x, y))
            continue
        c = cls_of(r["cls"]); trk = int(r.get("track", 0) or 0)
        z = None if r.get("z") in (None, "", "nan") else float(r["z"])
        src[take][(c, trk)].append((t, x, y, z))
    return src, mic


def group_fill(reads: list[dict]):
    """変更票 #69 r31 ②（2026-10-23）: group_reads と同じ形で、隠れた区間を前後の読みに 2 次式を当てて埋めた点列を返す。
    戻り = (src_by, mic_by, blk_by, log_by)。blk_by = 埋められなかった区間（3 s 超・片側に読みが無い）だけ（その中には正解を作らない）。
    log_by[take] = [{"key": …, "t_hide", "t_show", "filled", "reason", …}]（報告用）。印が不完全なら ValueError（R72-05）。"""
    src, mic = group_reads(reads)
    rows_by = defaultdict(lambda: defaultdict(list))
    for r in reads:
        take = norm(r["take"]); key = "mic" if r["cls"].strip().lower() == MIC_CLS else (cls_of(r["cls"]), int(r.get("track", 0) or 0))
        rows_by[take][key].append(r)
    blk_by, log_by = defaultdict(dict), defaultdict(list)
    for take, d in rows_by.items():
        for key, rows in d.items():
            iv = hidden_intervals(rows)
            if not iv:
                continue
            pts = mic[take] if key == "mic" else src[take][key]
            width = 3 if key == "mic" else 4
            filled, blocked, log = fill_hidden(sorted(pts), iv)
            filled = strip_fill_flag(filled, width)
            if key == "mic":
                mic[take] = filled
            else:
                src[take][key] = filled
            if blocked:
                blk_by[take][key] = blocked
            for g in log:
                log_by[take].append(dict(g, key=("mic" if key == "mic" else f"{key[0]}/{key[1]}")))
    return src, mic, blk_by, log_by


def walking_headings(mic_path, times):
    """Heading at reception from time-ordered walking marks (field x forward/y left).

    A stopped segment keeps the preceding walking heading (initial stops use
    the first subsequent heading). A wholly stationary path has no direction.
    """
    mp = np.asarray(sorted(set(tuple(p) for p in mic_path)), dtype=float)
    if len(mp) < 2 or np.any(np.diff(mp[:, 0]) <= 0):
        raise ValueError("歩行の向き: 異なる時刻の mic 読みが 2 点以上必要（同時刻の位置矛盾も不可）")
    delta = np.diff(mp[:, 1:3], axis=0)
    moving = np.linalg.norm(delta, axis=1) > 1e-9
    if not moving.any():
        raise ValueError("歩行の向き: mic の位置が動いていないため正面を決められません")
    headings = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    last = headings[np.flatnonzero(moving)[0]]
    for i in range(len(headings)):
        if moving[i]:
            last = headings[i]
        headings[i] = last
    indices = np.clip(np.searchsorted(mp[:, 0], times, side="right") - 1, 0, len(headings) - 1)
    return headings[indices]


def _segments(pts: list, blocks: list) -> list:
    """点列を、1 秒より長く隠れた区間（blocks = [(隠れる前の最後の読み, 見えてからの最初の読み)]・原録音の秒）で分ける（B4・R17-01）。
    区間の両端の読みはそれぞれの側の区切りに入り、区間の中には正解を作らない。区間の中の点（無いはず）は使わない。"""
    if not blocks:
        return [pts]
    segs, cur = [], []
    for p in pts:
        t = float(p[0])
        if any(a < t < b for a, b in blocks):
            continue
        cur.append(p)
        if any(abs(t - a) < 1e-9 for a, _ in blocks):
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return [sg for sg in segs if len(sg) >= 2]


def _extend_approach(pts: list, t_need: float, extend_m: float) -> tuple[list, dict]:
    """変更票 #69 r29／r30／r33 の C（H2-58・H2-59 だけ）: 向きの窓の始まり t_need（原録音の秒）が最初の読みより前なら、最初の 3 点から出した速さのまま
    目印の外 extend_m（2 m）まで手前に延ばす。要る分だけ（最大 extend_m）。窓が読みの中に収まる本（実際の速さで +10 の中）は延ばさない。
    pts = (t, x, y, z) 昇順（時刻順 = 近づく向きの順）。戻り = (延ばした点列, 記録)。"""
    info = dict(extended=False, ext_m=0.0, reason="")
    if len(pts) < 2 or extend_m <= 0:
        return pts, info
    t0, x0 = pts[0][0], pts[0][1]
    if t_need >= t0:
        info["reason"] = "窓の始まりが読みの中（延ばさない）"; return pts, info
    head = pts[:3]
    ts = np.array([p[0] for p in head]); xs = np.array([p[1] for p in head])
    if len(set(ts.tolist())) < 2:
        info["reason"] = "速さが出せない（同じ時刻の読み）"; return pts, info
    v = float(np.polyfit(ts, xs, 1)[0])                                             # 近づく向きの速さ（符号つき・m/s）
    if abs(v) < 1e-6:
        info["reason"] = "速さが 0"; return pts, info
    dt = min(t0 - t_need, extend_m / abs(v))                                        # 要る分だけ・最大 extend_m
    ext = (t0 - dt, x0 - v * dt, pts[0][2], pts[0][3])
    info.update(extended=True, ext_m=round(abs(v) * dt, 3), speed_mps=round(abs(v), 3), covers_window=bool(t0 - dt <= t_need + 1e-9))
    return [ext] + list(pts), info


def gt_lines_for_clip(sources: dict, mic_path: list, off: float, mic_z: float, src_z: float | None,
                      heading_deg: float = 0.0, walk_heading: bool = True, mic_xy=(0.0, 0.0), blocks: dict | None = None,
                      range_m: float | None = None, dirwin: tuple | None = None, extend_m: float = 0.0, extra: dict | None = None):
    """1 クリップの正解行（6 列の文字列）と、最接近の情報。時刻はクリップ基準（原録音 − off）。
    mic_xy = 静止の本人の現場座標（B2・R97-01・既定 (0, 0) = 原点）。blocks = 埋められなかった隠れた区間（(cls, trk) → [(a, b)]・"mic" → [(a, b)]・原録音の秒）。
    range_m（r31 ①・既定 None = 絞らない）: 相手の道に沿った位置 |x| ≤ range_m のフレームだけ正解に残す（検出フレーム率の分母）。
    dirwin（例 (2.5, 1.5)）: 最接近の 2.5〜1.5 s 前の窓の正解行を別に返す（±12 の読み・C の延長も使う）。extend_m > 0 で C の延長（この本だけ）。
    extra（dict）に range／dirwin／延長の記録を書き足す。戻り = (lines, best)（呼び方は今までどおり）。"""
    blocks = blocks or {}
    extra = extra if extra is not None else {}
    if mic_path:
        mp = sorted(mic_path)
        mp = hold_ends(mp, off - 1.0, off + CLIP + 1.0)                            # r34: 最初の読みより前・最後の読みより後ろはその位置に立っていた
        mic = np.array([[t - off, x, y, mic_z] for t, x, y in mp], dtype=float)
        if len(mic) < 2:
            mic = np.array([mic[0, 1], mic[0, 2], mic_z], dtype=float)
    else:
        mic = np.array([float(mic_xy[0]), float(mic_xy[1]), mic_z], dtype=float)
    headings = (walking_headings(mic_path, (np.arange(int(CLIP / LABEL_RES)) + 0.5) * LABEL_RES + off)
                if mic_path and walk_heading else np.full(int(CLIP / LABEL_RES), heading_deg))
    lines, best = [], None
    all_rows = []                                                                      # (k, 行, |x| along road)
    for (c, trk), pts in sorted(sources.items()):
        pts = sorted(pts)
        if len(pts) < 2:
            continue
        z0 = src_z if src_z is not None else SRC_Z.get(c, 0.5)
        mic_blk = blocks.get("mic", [])
        segs = _segments(pts, blocks.get((c, trk), []))
        for seg in segs:
          wp = np.array([[t - off, x, y, (z if z is not None else z0)] for t, x, y, z in seg], dtype=float)
          if wp[-1, 0] < -1.0 or wp[0, 0] > CLIP + 1.0:
            continue                                    # 読み取り点がクリップの外
          rows5, dbg = frame_label_rows(wp, mic, clip_len_sec=CLIP, class_idx=c, track_idx=trk,
                                        source_active_from=float(wp[0, 0]), source_active_until=float(wp[-1, 0]))
          t_frames = (np.arange(int(CLIP / LABEL_RES)) + 0.5) * LABEL_RES
          x_road = np.interp(t_frames, wp[:, 0], wp[:, 1])                           # 道に沿った位置（折れ線・r31 ①の範囲の判定に使う）
          for k, c_, t_, az_i, el_i in rows5:
            if any(a < (k + 0.5) * LABEL_RES + off < b for a, b in mic_blk):
                continue                                # 本人が隠れて埋められなかった区間の正解は作らない
            # Rotate the continuous angle before integer rounding. Keep heading
            # zero on the original path, byte-for-byte including boundary ties.
            if headings[k] != 0:
                az_i = int(np.rint((float(dbg["az"][k]) - headings[k] + 180.0) % 360.0 - 180.0))
                if az_i == 180:
                    az_i = -180
            d3 = float(dbg["dist"][k]); el = float(dbg["el"][k])
            dh = d3 * math.cos(math.radians(el))
            all_rows.append((k, f"{k},{c_},{t_},{az_i},{el_i},{dh:.2f}", float(abs(x_road[k])), (c, trk)))
            if best is None or dh < best[1]:
                best = (k, dh, az_i)
    all_rows.sort(key=lambda r: (r[0], r[3]))
    if range_m is None:
        lines = [s for _, s, _, _ in all_rows]
    else:
        lines = [s for _, s, ax, _ in all_rows if ax <= range_m + 1e-9]
        extra["n_rows_all"] = len(all_rows); extra["n_rows_in_range"] = len(lines); extra["range_m"] = range_m
    if dirwin is not None and best is not None:
        t_cpa = (best[0] + 0.5) * LABEL_RES
        w0, w1 = t_cpa - dirwin[0], t_cpa - dirwin[1]
        win_rows = [s for k, s, _, _ in all_rows if w0 - 1e-9 <= (k + 0.5) * LABEL_RES <= w1 + 1e-9]
        ext_info = dict(extended=False, ext_m=0.0, reason="")
        if extend_m > 0:
            # C: 窓の始まりが最初の読みより前なら、最初の 3 点の速さのまま目印の外 extend_m まで手前に延ばして窓の正解を作る（この本だけ）
            for (c, trk), pts in sorted(sources.items()):
                pts = sorted(pts)
                if len(pts) < 2:
                    continue
                z0 = src_z if src_z is not None else SRC_Z.get(c, 0.5)
                pts4 = [(t, x, y, (z if z is not None else z0)) for t, x, y, z in pts]
                ext_pts, ext_info = _extend_approach(pts4, w0 + off, extend_m)
                if not ext_info["extended"]:
                    continue
                wp = np.array([[t - off, x, y, z] for t, x, y, z in ext_pts], dtype=float)
                rows5, dbg = frame_label_rows(wp, mic, clip_len_sec=CLIP, class_idx=c, track_idx=trk,
                                              source_active_from=float(wp[0, 0]), source_active_until=float(wp[-1, 0]))
                have = {int(s.split(",")[0]) for s in win_rows}
                for k, c_, t_, az_i, el_i in rows5:
                    tk = (k + 0.5) * LABEL_RES
                    if not (w0 - 1e-9 <= tk <= w1 + 1e-9) or k in have:
                        continue
                    if headings[k] != 0:
                        az_i = int(np.rint((float(dbg["az"][k]) - headings[k] + 180.0) % 360.0 - 180.0))
                        if az_i == 180:
                            az_i = -180
                    d3 = float(dbg["dist"][k]); el = float(dbg["el"][k])
                    win_rows.append(f"{k},{c_},{t_},{az_i},{el_i},{d3 * math.cos(math.radians(el)):.2f}")
                win_rows.sort(key=lambda s: int(s.split(",")[0]))
        extra["dirwin_lines"] = win_rows; extra["dirwin"] = (round(w0, 2), round(w1, 2)); extra["ext"] = ext_info
    return lines, best


WARN_CLS = (0, 1, 2, 3)          # 警告音（サイレン・クラクション・バック音・ベル）


def load_warn_interval(path: Path) -> dict:
    """--warn-interval の csv（take,class,on_s,end_s・全長の録音の秒）→ {(take, cls): (on, end)}。鳴り終わり ≤ 鳴り始め・数でない値は止める。"""
    out = {}
    for r in _read_csv(path):
        take, c = norm(r["take"]), cls_of(r["class"])
        on, end = float(r["on_s"]), float(r["end_s"])
        if not (math.isfinite(on) and math.isfinite(end)) or end <= on:
            raise ValueError(f"警告音の区間が不正: {r['take']} {r['class']} {r['on_s']}〜{r['end_s']}")
        if (take, c) in out and out[(take, c)] != (on, end):
            raise ValueError(f"同じ本・同じクラスの警告音の区間が 2 つ: {r['take']} {r['class']}")
        out[(take, c)] = (on, end)
    return out


def warn_filter(lines: list, off: float, take: str, warn_iv: dict) -> tuple[list, int]:
    """警告音の正解の区間（2026-10-27・本人 12:35 (c)）: 警告音のクラスの行は、全長の録音の秒で [鳴り始め, 鳴り終わり) にフレームの中心がある行だけ残す。
    区間が無い警告音の行があれば止める（黙って通過の区間のまま使わない）。距離のクラスの行はそのまま。戻り = (残した行, 外した行の数)。"""
    kept, n_out = [], 0
    for s in lines:
        k, c = (int(v) for v in s.split(",")[:2])
        if c not in WARN_CLS:
            kept.append(s); continue
        iv = warn_iv.get((take, c))
        if iv is None:
            raise ValueError(f"警告音の区間（--warn-interval）が無い: {take} class {c}")
        t = (k + 0.5) * LABEL_RES + off
        if iv[0] <= t < iv[1]:
            kept.append(s)
        else:
            n_out += 1
    return kept, n_out


def build(reads: list[dict], ann: list[dict], out: Path, mic_z_default: float, src_z: float | None,
          range_m: float | None = None, dirwin: tuple | None = None, extend_takes: set | None = None, extend_m: float = 2.0,
          sync: dict | None = None, warn_iv: dict | None = None) -> list[dict]:
    """range_m／dirwin／extend_takes／sync は変更票 #69 r31・r29〜r33（2026-10-23）。既定（None）は今までどおりの動き（検査の固定値を守る）。
    sync = {take(小文字の stem): [録音側の拍手の秒, …]}（step19h の sync_*.csv）→ 拍手がクリップの中に入った本を数える（r31 ③・そのまま使う）。"""
    # Fixed positions are measured per take. Preserve the two actual sounding
    # endpoint times from reads; t_cpa for warnings is a scoring-window proxy,
    # not the observed end of the sound, and must not fabricate a duration.
    fixed = {}
    for a in ann:
        if a.get("class") not in ("backup_beep", "horn", "crossing"):
            continue
        xyz = [a.get(k, "") for k in ("x_m", "y_m", "高さ_m")]
        if not any(v not in (None, "") for v in xyz):
            continue
        if not all(v not in (None, "") for v in xyz):
            raise ValueError("固定音源の x_m・y_m・高さ_m は 3 欄とも必要")
        xyz = tuple(float(v) for v in xyz)
        if not all(math.isfinite(v) for v in xyz) or xyz[2] < 0:
            raise ValueError("固定音源の座標が不正")
        key = (norm(a.get("orig_file", "")), cls_of(a["class"]))
        if key in fixed and fixed[key] != xyz:
            raise ValueError("同じ原本・クラスの固定音源座標が不一致")
        fixed[key] = xyz
    positioned = []
    for r in reads:
        key = (norm(r["take"]), cls_of(r["cls"])) if r["cls"] != MIC_CLS else None
        if key in fixed:
            x, y, z = fixed[key]
            r = dict(r, x=str(x), y=str(y), z=str(z))
        positioned.append(r)
    if range_m is None and dirwin is None and not extend_takes and sync is None:
        src_by, mic_by = group_reads(positioned)
        blk_by = hidden_blocks(positioned)                   # 旧の動き（B4・1 秒の線形）: 検査の固定値を守る
        log_by = {}
    else:
        src_by, mic_by, blk_by, log_by = group_fill(positioned)   # r31 ②: 隠れた区間を 2 次式で埋める・埋められない区間だけ正解を作らない
    gt_dir = out / "gt"; gt_dir.mkdir(parents=True, exist_ok=True)
    win_dir = out / "gt_dirwin"
    if dirwin is not None:
        win_dir.mkdir(parents=True, exist_ok=True)
    extend_takes = {norm(t) for t in (extend_takes or set())}
    manifest, report = [], []
    summary = dict(range_m=range_m, dirwin=list(dirwin) if dirwin else None, extend_takes=sorted(extend_takes), extend_m=extend_m,
                   n_clips=0, n_hidden_filled=0, n_hidden_blocked=0, takes_with_fill=[], extended=[], clap_in_clip=[],
                   note_r35="ヘルメットのマイクの傾きは補正しない（#69 r35・歩く比較の仰角は割り引いて読む）")
    for a in ann:
        if str(a.get("scored", "1")).strip() != "1" or a.get("class", "").strip() == "none":
            continue
        take = norm(a.get("orig_file", "") or "")
        clip = a["clip_id"].strip()
        off = float(a["cut_offset_s"])
        mz = float(a["mic_z"]) if a.get("mic_z") not in (None, "", "nan") else mic_z_default
        state = (a.get("状態") or "").strip()
        mic_path = mic_by.get(take, [])
        motion = "walk" if (state == "歩行" or (not state and mic_path)) else "static"
        if motion == "static":
            mic_path = []  # 静止のヘルメットも原点。別の歩行読みを流用しない。
        if take not in src_by:
            report.append(dict(clip_id=clip, take=take, cut_offset_s=f"{off:.3f}", motion=motion, n_rows=0, cpa_frame="", min_dist="", az_at_cpa="", note="読みが無い（動画なし・未読み）"))
            continue
        if motion == "walk" and not mic_path:
            report.append(dict(clip_id=clip, take=take, cut_offset_s=f"{off:.3f}", motion=motion, n_rows=0, cpa_frame="", min_dist="", az_at_cpa="", note="歩行の本なのに mic 行（人の位置）が無い"))
            continue
        mic_xy = (float(a.get("mic_x_m") or 0.0), float(a.get("mic_y_m") or 0.0))      # B2: 静止の本人の現場座標（step19d の注釈・無ければ原点）
        extra = {}
        lines, best = gt_lines_for_clip(src_by[take], mic_path, off, mz, src_z,
                                       heading_deg=tripod_heading(a), walk_heading=(motion == "walk"), mic_xy=mic_xy, blocks=blk_by.get(take, {}),
                                       range_m=range_m, dirwin=dirwin, extend_m=(extend_m if take in extend_takes else 0.0), extra=extra)
        if warn_iv is not None:
            lines, n_wout = warn_filter(lines, off, take, warn_iv)
            if n_wout:
                summary.setdefault("warn_interval_trimmed", []).append(dict(clip_id=clip, take=take, n_rows_out=n_wout))
        (gt_dir / f"{clip}.csv").write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        note = "" if lines else "0 行（読み取り点がクリップの外）"
        if best and (best[0] < 5 or best[0] > 95):
            note = "最接近がクリップの端"
        row = dict(clip_id=clip, take=take, cut_offset_s=f"{off:.3f}", motion=motion, n_rows=len(lines),
                   cpa_frame=(best[0] if best else ""), min_dist=(f"{best[1]:.2f}" if best else ""), az_at_cpa=(best[2] if best else ""), note=note)
        if range_m is not None or dirwin is not None or extend_takes or sync is not None:
            win = extra.get("dirwin_lines", [])
            if dirwin is not None:
                (win_dir / f"{clip}.csv").write_bytes(("\n".join(win) + ("\n" if win else "")).encode("utf-8"))
            logs = log_by.get(take, [])
            n_f = sum(1 for g in logs if g["filled"]); n_b = sum(1 for g in logs if not g["filled"])
            ext = extra.get("ext", {}) or {}
            claps = [c for c in (sync or {}).get(take, []) if off <= c <= off + CLIP]
            row.update(n_rows_all=extra.get("n_rows_all", len(lines)), n_rows_dirwin=len(win), dirwin_s=(f"{extra['dirwin'][0]}〜{extra['dirwin'][1]}" if extra.get("dirwin") else ""),
                       n_hidden_filled=n_f, n_hidden_blocked=n_b, extended=int(bool(ext.get("extended"))), ext_m=(f"{ext['ext_m']:.2f}" if ext.get("extended") else ""),
                       ext_note=(ext.get("reason", "") if take in extend_takes else ""), clap_in_clip=len(claps), clap_s=";".join(f"{c:.3f}" for c in claps))
            summary["n_clips"] += 1; summary["n_hidden_filled"] += n_f; summary["n_hidden_blocked"] += n_b
            if n_f and take not in summary["takes_with_fill"]:
                summary["takes_with_fill"].append(take)
            if take in extend_takes:
                summary["extended"].append(dict(clip_id=clip, take=take, extended=bool(ext.get("extended")), ext_m=ext.get("ext_m", 0.0), reason=ext.get("reason", ""),
                                                speed_mps=ext.get("speed_mps"), covers_window=ext.get("covers_window")))
            if claps:
                summary["clap_in_clip"].append(dict(clip_id=clip, take=take, clap_s=[round(c, 3) for c in claps]))
        manifest.append(dict(clip_id=clip, split="eval", motion=motion, mic_z=f"{mz:.3f}"))
        report.append(row)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "split", "motion", "mic_z"]); w.writeheader(); w.writerows(manifest)
    cols = ["clip_id", "take", "cut_offset_s", "motion", "n_rows", "cpa_frame", "min_dist", "az_at_cpa", "note"]
    if range_m is not None or dirwin is not None or extend_takes or sync is not None:
        cols += ["n_rows_all", "n_rows_dirwin", "dirwin_s", "n_hidden_filled", "n_hidden_blocked", "extended", "ext_m", "ext_note", "clap_in_clip", "clap_s"]
        summary["n_clap_in_clip"] = len(summary["clap_in_clip"]); summary["n_extended_used"] = sum(1 for e in summary["extended"] if e["extended"])
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows({k: r.get(k, "") for k in cols} for r in report)
    return report


def load_sync(path: Path) -> dict:
    """step19h の sync_*.csv（take,clap,audio_s,video_s）→ {take（小文字・norm）: [録音側の拍手の秒]}（r31 ③の数え上げ用）。"""
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("audio_s") or "").strip():
                out[norm(r["take"])].append(float(r["audio_s"]))
    return dict(out)


def run_perception(out: Path, pred: Path, minframe: int, label: str = "field") -> int:
    cmd = [sys.executable, str(ROOT / "scripts/_perception_eval.py"), str(out / "perception_field.md"), "--plan", str(out / "manifest.csv"),
           "--split", "eval", "--meta", str(out / "gt"), "--minframe", str(minframe), "--json", str(out / "perception_field.json"), f"{label}={pred}"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    print((r.stdout or "")[-1500:]); print((r.stderr or "")[-800:], file=sys.stderr)
    return r.returncode


def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def selftest(tmp: Path) -> int:
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    marks = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
    # (a) 静止の本: 車が y = 2.0 を x = +10 → −10 へ 4 m/s、x = 0 を原録音 16.5 s に通る。切り出し 8.5 s → クリップの 8.0 s が最接近
    reads = [{"take": "ZOOM0011.wav", "frame": "", "cls": "car", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{16.5 - x / 4.0:.3f}"} for x in marks]
    # (b) 歩行の本: 人が y = 0 を x = −6 → +6 へ 1.2 m/s（原録音 10 s に x = −6）。車が y = 2.0 を +10 → −10 へ 4 m/s（原録音 11 s に x = +10）
    reads += [{"take": "ZOOM0012.wav", "frame": "", "cls": "mic", "track": "0", "x": str(x), "y": "0", "z": "", "t": f"{10.0 + (x + 6) / 1.2:.3f}"} for x in (-6, -4, -2, 0, 2, 4, 6)]
    reads += [{"take": "ZOOM0012.wav", "frame": "", "cls": "car", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{11.0 + (10 - x) / 4.0:.3f}"} for x in marks]
    ann = [{"clip_id": "eval_field_clip01003", "orig_file": "ZOOM0011.wav", "cut_offset_s": "8.500", "scored": "1", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6", "状態": "静止"},
           {"clip_id": "eval_field_clip01004", "orig_file": "ZOOM0012.wav", "cut_offset_s": "6.000", "scored": "1", "class": "car_drive", "t_cpa": "7.85", "mic_z": "1.6", "状態": "歩行"},
           {"clip_id": "eval_field_clip01005", "orig_file": "ZOOM0013.wav", "cut_offset_s": "5.000", "scored": "1", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6", "状態": "静止"},
           {"clip_id": "eval_field_clip01006", "orig_file": "ZOOM0011.wav", "cut_offset_s": "8.500", "scored": "0", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6", "状態": "静止"}]
    rep = {r["clip_id"]: r for r in build(reads, ann, tmp, 1.6, None)}
    ok = []
    a = rep["eval_field_clip01003"]
    ok.append(("静止: 最接近がクリップの 8.0 s（frame 80 ± 1）", a["n_rows"] > 0 and abs(int(a["cpa_frame"]) - 80) <= 1))
    # 方位はフレーム中心（8.05 s）と伝搬遅延のぶん最接近の後ろ（4 m/s で約 0.25 m → 約 +7°）になるので ±10° で見る
    ok.append(("静止: 最接近の水平距離 ≈ 2.0 m・方位 ≈ +90°（左・±10°）", a["min_dist"] != "" and abs(float(a["min_dist"]) - 2.0) < 0.1 and abs(int(a["az_at_cpa"]) - 90) <= 10))
    # 静止の本は step19i（既存の道具）と同じ正解になる（同じ frame_label_rows・静止マイク）
    spec = importlib.util.spec_from_file_location("s19i", ROOT / "scripts/step19i_rebase_video_gt.py")
    s19i = importlib.util.module_from_spec(spec); sys.modules["s19i"] = s19i; spec.loader.exec_module(s19i)
    s19i.rebase([r for r in reads if r["take"] == "ZOOM0011.wav"], ann[:1], tmp / "gt19i", 1.6, None)
    same = (tmp / "gt19i/eval_field_clip01003.csv").read_bytes() == (tmp / "gt/eval_field_clip01003.csv").read_bytes()
    ok.append(("静止: step19i の正解と 1 バイトも違わない", same))
    b = rep["eval_field_clip01004"]
    # 人 x_w = −6 + 1.2(t−10)、車 x_c = 10 − 4(t−11) → 一致は t = 13.846 s → クリップ 7.85 s（frame 78）
    ok.append(("歩行: 最接近がクリップの 7.85 s（frame 78 ± 2）", b["n_rows"] > 0 and abs(int(b["cpa_frame"]) - 78) <= 2))
    ok.append(("歩行: 最接近の水平距離 ≈ 2.0 m（人が動いても線からの距離）", b["min_dist"] != "" and abs(float(b["min_dist"]) - 2.0) < 0.15))
    gl = [s for s in (tmp / "gt/eval_field_clip01004.csv").read_text(encoding="utf-8").split("\n") if s]
    az_first = int(gl[0].split(",")[3])
    ok.append(("歩行: 車が前から来るので最初の方位は前寄り（|az| < 45°）", abs(az_first) < 45))
    ok.append(("読みの無いテイクは注意だけ・scored=0 は作らない", rep["eval_field_clip01005"]["n_rows"] == 0 and "eval_field_clip01006" not in rep))
    man = _read_csv(tmp / "manifest.csv")
    ok.append(("manifest は 2 本（静止 static・歩行 walk）", [(m["clip_id"], m["motion"]) for m in man] == [("eval_field_clip01003", "static"), ("eval_field_clip01004", "walk")]))
    # #44: frozen byte hashes of the selftest fixtures (no real takes). 2026-10-19 #70 r16: 車の既定の高さ 0.5 → 0.325 m で仰角の列が変わるので凍結し直した
    #   （旧 = 39b65dd1ecf7f01d… / c3cddf0fbef82a9b…・車 0.5 m）。向きの扱いが変われば、この固定値と食い違う。
    old_hashes = {"eval_field_clip01003": "dadbd2ecdebf6e6438d467b7ce1e93d6031916b6561a68034f530edce593b37b",
                  "eval_field_clip01004": "0bc5c6685d5e53601385541d945172b3bc5701a5a55b6a2b758facb80bfe9ed8"}
    ok.append(("#44 向き 0°: 静止・歩行とも旧出力 SHA256 と完全一致",
               all(hashlib.sha256((tmp / "gt" / (c + ".csv")).read_bytes()).hexdigest() == h for c, h in old_hashes.items())))
    reverse_reads = [dict(r, take="REVERSE.wav", x=str(-float(r["x"])))
                     for r in reads if r["take"] == "ZOOM0012.wav"]
    oriented_ann = [dict(ann[0], clip_id="static180", **{"三脚の向き": "180"}),
                    dict(ann[0], clip_id="explicit0", **{"三脚の向き": "0"}),
                    dict(ann[0], clip_id="legacy180", **{"特記": "三脚180°"}),
                    dict(ann[1], clip_id="reverse", orig_file="REVERSE.wav", cut_offset_s="6.050", **{"三脚の向き": "0"})]
    oriented_out = tmp / "orientation"
    oriented = {r["clip_id"]: r for r in build(reads + reverse_reads, oriented_ann, oriented_out, 1.6, None)}
    ok.append(("#44 三脚 180°: +y 線の相手は右（最接近 −85°、約 −90°）",
               abs(int(oriented["static180"]["az_at_cpa"]) + 90) <= 10))
    ok.append(("#44 逆歩行: +83° から −97°（三脚欄でなく歩く方向が正面）",
               int(oriented["reverse"]["az_at_cpa"]) == -97))
    p180 = (oriented_out / "gt/static180.csv").read_bytes()
    ok.append(("#44 旧特記チップ互換", p180 == (oriented_out / "gt/legacy180.csv").read_bytes()))
    ok.append(("#44 明示 0° も旧出力と完全一致",
               (oriented_out / "gt/explicit0.csv").read_bytes() == (tmp / "gt/eval_field_clip01003.csv").read_bytes()))
    def except_az(lines):
        return [tuple(x for i, x in enumerate(line.split(",")) if i != 3) for line in lines]
    src_rev, mic_rev = group_reads(reverse_reads)
    legacy_rev, _ = gt_lines_for_clip(src_rev["reverse"], mic_rev["reverse"], 6.05, 1.6, None, walk_heading=False)
    ok.append(("#44 回転前後で距離・時刻・行数・仰角が完全一致（静止・逆歩行）",
               except_az(p180.decode().splitlines()) == except_az((tmp / "gt/eval_field_clip01003.csv").read_text().splitlines())
               and except_az((oriented_out / "gt/reverse.csv").read_text().splitlines()) == except_az(legacy_rev)))
    ok.append(("#44 歩行方向は入力行順でなく時刻順", np.array_equal(
        walking_headings(mic_rev["reverse"], [12, 14, 16]), walking_headings(list(reversed(mic_rev["reverse"])), [12, 14, 16]))))
    for label, invalid_path in [("1 点だけ", [(0, 0, 0)]), ("全区間停止", [(0, 0, 0), (1, 0, 0)]),
                                 ("同時刻の位置矛盾", [(0, 0, 0), (0, 1, 0)])]:
        try:
            walking_headings(invalid_path, [0.5])
            rejected = False
        except ValueError:
            rejected = True
        ok.append(("#44 歩行方向を決められない入力を拒否: " + label, rejected))
    helmet_ann = dict(ann[0], clip_id="helmet180", mic_z="1.8", **{"装着":"ヘルメット", "三脚の向き":"180"})
    fixed_ann = dict(helmet_ann, clip_id="fixed", orig_file="FIXED.wav", cut_offset_s="0", **{"class":"horn", "x_m":"0", "y_m":"3", "高さ_m":"1.8"})
    fixed_reads = [dict(take="FIXED.wav", cls="horn", track="1", t=str(t), x="", y="") for t in (5, 6)]
    extra = {x["clip_id"]:x for x in build(reads + fixed_reads, [helmet_ann, fixed_ann], tmp / "fixed", 1.6, None)}
    ok.append(("#46 静止ヘルメット180°・実測高さ", abs(int(extra["helmet180"]["az_at_cpa"])+90)<=10))
    lines_fixed = (tmp/"fixed/gt/fixed.csv").read_text().splitlines()
    ok.append(("#47 固定音源の座標を欄から読取・実測発音区間のみ", bool(lines_fixed) and all(row.split(",")[3:]==["-90","0","3.00"] for row in lines_fixed) and all(50<=int(row.split(",")[0])<=60 for row in lines_fixed)))
    # (c) 正解に小さな揺れを足した予測（7 列）で _perception_eval.py が通り、方位・距離の誤差が小さい
    rng = np.random.default_rng(0)
    pred_lines = []
    for m in man:
        for s in (tmp / "gt" / f"{m['clip_id']}.csv").read_text(encoding="utf-8").split("\n"):
            if not s:
                continue
            k, c, t, az, el, d = s.split(",")
            pred_lines.append(f"{m['clip_id']},{k},{c},0,{float(az) + rng.normal(0, 3):.1f},{el},{float(d) * 1.05:.2f}")
    pred = tmp / "pred_field.csv"; pred.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    rc = run_perception(tmp, pred, minframe=0)
    js = json.loads((tmp / "perception_field.json").read_text(encoding="utf-8")) if (tmp / "perception_field.json").exists() else {}
    ok.append(("_perception_eval.py が通り json がある", rc == 0 and bool(js)))
    txt = (tmp / "perception_field.md").read_text(encoding="utf-8") if rc == 0 else ""
    ok.append(("結果 md に field の行と motion 別の行がある", "field" in txt and "motion:walk" in txt and "motion:static" in txt))
    # ---- 変更票 #69 r31 ①②③・r29〜r33 の C・r34（2026-10-23）
    marks12 = list(range(12, -13, -2))
    # (r) 速い本（6 m/s・±12 の読み・x = 0 を原録音 16.5 s）: 範囲 ±10 の正解と、向きの窓（2.5〜1.5 s 前 = 15〜9 m 手前）の正解
    fast = [{"take": "FAST.wav", "frame": "", "cls": "bicycle", "track": "1", "x": str(x), "y": "5.0", "z": "", "t": f"{16.5 - x / 6.0:.3f}"} for x in marks12]
    slow = [{"take": "SLOW.wav", "frame": "", "cls": "bicycle", "track": "1", "x": str(x), "y": "5.0", "z": "", "t": f"{16.5 - x / 4.0:.3f}"} for x in marks12]
    ann_r = [{"clip_id": "fast", "orig_file": "FAST.wav", "cut_offset_s": "8.500", "scored": "1", "class": "bicycle", "mic_z": "1.6", "状態": "静止"},
             {"clip_id": "slow", "orig_file": "SLOW.wav", "cut_offset_s": "8.500", "scored": "1", "class": "bicycle", "mic_z": "1.6", "状態": "静止"}]
    out_r = tmp / "range"
    rep_r = {r["clip_id"]: r for r in build(fast + slow, ann_r, out_r, 1.6, None, range_m=10.0, dirwin=(2.5, 1.5), extend_takes={"FAST.wav", "SLOW.wav"}, extend_m=2.0,
                                            sync={norm("FAST.wav"): [10.0, 30.0], norm("SLOW.wav"): [3.0]})}
    gf = [s for s in (out_r / "gt/fast.csv").read_text(encoding="utf-8").split("\n") if s]
    ok.append(("r31 ①: 範囲 ±10 → 正解の行は |x| ≤ 10 のフレームだけ（6 m/s: 約 3.3 s = 33 行・全部の行 40 より少ない）",
               rep_r["fast"]["n_rows"] < rep_r["fast"]["n_rows_all"] and 31 <= rep_r["fast"]["n_rows"] <= 35))
    gw = [s for s in (out_r / "gt_dirwin/fast.csv").read_text(encoding="utf-8").split("\n") if s]
    ks = [int(s.split(",")[0]) for s in gw]
    ok.append((f"向きの窓（最接近 8.0 s の 2.5〜1.5 s 前 = frame 55〜65）の正解を gt_dirwin/ に別に出す（{len(gw)} 行・延長込みで 14 m 手前から）",
               len(gw) >= 7 and min(ks) >= 54 and max(ks) <= 65 and rep_r["fast"]["n_rows_dirwin"] == len(gw)))
    S = json.loads((out_r / "summary.json").read_text(encoding="utf-8"))
    ef = [e for e in S["extended"] if e["clip_id"] == "fast"][0]; es = [e for e in S["extended"] if e["clip_id"] == "slow"][0]
    ok.append((f"C の延長: 速い本は窓の始まり（15 m 手前）が読み（12 m）の外 → 2 m だけ延ばす（ext_m {ef['ext_m']}・窓は覆えない）・ゆっくりの本は窓が読みの中 → 延ばさない",
               ef["extended"] and abs(ef["ext_m"] - 2.0) < 1e-6 and not ef["covers_window"] and not es["extended"] and "読みの中" in es["reason"]))
    ok.append(("r31 ③: 拍手がクリップ（8.5〜18.5 s）の中に入った本を数える（fast 1 回・slow 0 回・summary 1 本）",
               rep_r["fast"]["clap_in_clip"] == 1 and rep_r["slow"]["clap_in_clip"] == 0 and S["n_clap_in_clip"] == 1 and "r35" in S["note_r35"]))
    # (h) 隠れた区間: 2 s 隠れる（読みは両側にある）→ 2 次式で埋めて正解が続く・3.5 s 隠れる → 埋めず、その間の正解は無い
    # 速い本の読みは 14.5 s（x = 12）〜18.5 s（x = −12）。隠れは読みの中に置く（前後に読みがある区間だけ埋める）
    hid2 = [dict(fast[0], x="", y="", t="15.0", hidden="start"), dict(fast[0], x="", y="", t="16.9", hidden="end")]
    gap2 = [r for r in fast if not (15.0 <= float(r["t"]) <= 16.9)]
    rep_h = {r["clip_id"]: r for r in build(gap2 + hid2, ann_r[:1], tmp / "hid2", 1.6, None, range_m=10.0)}
    gh = [s for s in (tmp / "hid2/gt/fast.csv").read_text(encoding="utf-8").split("\n") if s]
    kh = {int(s.split(",")[0]) for s in gh}
    ok.append((f"r31 ②: 1.9 s 隠れた区間（15.0〜16.9 s = クリップ 6.5〜8.4 s）は 2 次式で埋めて正解が続く（埋め 1・frame 66〜83 がある・{rep_h['fast']['n_hidden_filled']}）",
               rep_h["fast"]["n_hidden_filled"] == 1 and rep_h["fast"]["n_hidden_blocked"] == 0 and all(k in kh for k in range(66, 84))))
    hid35 = [dict(fast[0], x="", y="", t="14.6", hidden="start"), dict(fast[0], x="", y="", t="18.2", hidden="end")]
    gap35 = [r for r in fast if not (14.6 <= float(r["t"]) <= 18.2)]
    rep_h3 = {r["clip_id"]: r for r in build(gap35 + hid35, ann_r[:1], tmp / "hid35", 1.6, None, range_m=10.0)}
    gh3 = [s for s in (tmp / "hid35/gt/fast.csv").read_text(encoding="utf-8").split("\n") if s]
    kh3 = {int(s.split(",")[0]) for s in gh3}
    ok.append(("3.6 s 隠れた区間（上限 3 s 超・前後に読みはある）は埋めず、その間（クリップ 6.1〜9.7 s = frame 61〜97）の正解は作らない（埋められず 1）",
               rep_h3["fast"]["n_hidden_blocked"] == 1 and rep_h3["fast"]["n_hidden_filled"] == 0 and not any(k in kh3 for k in range(62, 97))))
    # (w) r34: 歩く人の最初の読みより前はそこに立っていた（最初の読みが遅い歩行の本でも、クリップの頭から正解がある）
    late_mic = [{"take": "LATE.wav", "frame": "", "cls": "mic", "track": "0", "x": str(x), "y": "0", "z": "", "t": f"{13.0 + (x + 6) / 1.2:.3f}"} for x in (-6, -4, -2, 0, 2, 4, 6)]
    late_car = [{"take": "LATE.wav", "frame": "", "cls": "car", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{8.0 + (10 - x) / 4.0:.3f}"} for x in marks]
    ann_w = [{"clip_id": "late", "orig_file": "LATE.wav", "cut_offset_s": "6.000", "scored": "1", "class": "car_drive", "mic_z": "1.6", "状態": "歩行"}]
    rep_w = {r["clip_id"]: r for r in build(late_mic + late_car, ann_w, tmp / "late", 1.6, None, range_m=10.0)}
    gl_w = [s for s in (tmp / "late/gt/late.csv").read_text(encoding="utf-8").split("\n") if s]
    ok.append(("r34: 本人の最初の読み（13.0 s = クリップ 7.0 s）より前も、本人が −6 に立っていたとして正解がある（frame 25 前後の行あり）",
               rep_w["late"]["n_rows"] > 0 and any(int(s.split(",")[0]) < 60 for s in gl_w)))
    # ---- 2026-10-27（本人 12:35 (c)）: 警告音の正解を鳴り始め〜鳴り終わりに絞る（--warn-interval）
    _lines = [f"{k},3,0,90,-5,3.00" for k in range(30, 90)] + [f"{k},8,0,90,-5,3.00" for k in range(30, 90)]
    _kept, _nout = warn_filter(_lines, 10.0, "w_001", {("w_001", 3): (15.0, 16.5)})      # クリップの 5.0〜6.5 s = frame 50〜64
    _bell = [int(s.split(",")[0]) for s in _kept if s.split(",")[1] == "3"]
    _bike = [s for s in _kept if s.split(",")[1] == "8"]
    _stop = False
    try:
        warn_filter(_lines, 10.0, "w_002", {("w_001", 3): (15.0, 16.5)})
    except ValueError:
        _stop = True
    _p = tmp / "warn_iv.csv"
    _p.write_text("take,class,on_s,end_s\nW_001.WAV,bike_bell,15.0,16.5\n", encoding="utf-8")
    _bad = tmp / "warn_iv_bad.csv"
    _bad.write_text("take,class,on_s,end_s\nW_001.WAV,bike_bell,15.0,15.0\n", encoding="utf-8")
    _stop2 = False
    try:
        load_warn_interval(_bad)
    except ValueError:
        _stop2 = True
    ok.append(("警告音の区間: ベルは鳴り始め〜鳴り終わり（frame 50〜64・15 行）だけ残し、自転車（距離のクラス）は 60 行そのまま・区間の無い本は止める・鳴り終わり ≤ 鳴り始めも止める",
               _bell == list(range(50, 65)) and len(_bike) == 60 and _nout == 45 and _stop and _stop2
               and load_warn_interval(_p) == {(norm("W_001.WAV"), 3): (15.0, 16.5)}))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest(Path(argv[argv.index("--selftest") + 1]))

    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    reads = _read_csv(Path(opt("--reads")))
    ann = _read_csv(Path(opt("--ann")))
    out = Path(opt("--out"))
    mic_z = float(opt("--mic-z", "1.6"))
    src_z = opt("--src-z"); src_z = float(src_z) if src_z else None
    pred = opt("--pred")
    minframe = int(opt("--minframe", "40"))
    # 変更票 #69 r31・r29〜r33（2026-10-23）: --range-m（既定 10・"none" で絞らない）・--dir-window 2.5-1.5・--extend-takes <原本,…>（C の 2 本）・--extend-m 2.0・--sync <sync_*.csv>
    rm = opt("--range-m", str(RANGE_M)); range_m = None if rm.lower() == "none" else float(rm)
    dw = opt("--dir-window", f"{DIR_WIN[0]}-{DIR_WIN[1]}"); dirwin = None if dw.lower() == "none" else tuple(float(v) for v in dw.split("-"))
    extend_takes = {s.strip() for s in (opt("--extend-takes", "") or "").split(",") if s.strip()}
    extend_m = float(opt("--extend-m", "2.0"))
    sync_p = opt("--sync"); sync = load_sync(Path(sync_p)) if sync_p else None
    # 2026-10-27（本人 12:35 (c)・Fable 経由）: --warn-interval <csv>（take,class,on_s,end_s）= 警告音の正解を波形で読んだ鳴り始め〜鳴り終わりに絞る。無ければ今までどおり
    wi_p = opt("--warn-interval"); warn_iv = load_warn_interval(Path(wi_p)) if wi_p else None
    rep = build(reads, ann, out, mic_z, src_z, range_m=range_m, dirwin=dirwin, extend_takes=extend_takes, extend_m=extend_m, sync=sync, warn_iv=warn_iv)
    n_ok = sum(1 for r in rep if r["n_rows"])
    print(f"正解を作った clip {n_ok} / 対象 {len(rep)}（注意あり {sum(1 for r in rep if r['note'])}）-> {out / 'gt'}・{out / 'manifest.csv'}・{out / 'report.csv'}")
    if (out / "summary.json").is_file():
        S = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        print(f"範囲 ±{S['range_m']} m・向きの窓 {S['dirwin']}（gt_dirwin/）・隠れの埋め {S['n_hidden_filled']} 区間（{len(S['takes_with_fill'])} 本）・埋められず {S['n_hidden_blocked']}・"
              f"C の延長 {S.get('n_extended_used', 0)}/{len(S['extended'])} 本・拍手がクリップの中 {S.get('n_clap_in_clip', 0)} 本"
              + (f"・警告音を鳴り始め〜鳴り終わりに絞った {len(S.get('warn_interval_trimmed', []))} 本" if warn_iv is not None else "") + f" -> {out / 'summary.json'}")
    if pred:
        return run_perception(out, Path(pred), minframe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
