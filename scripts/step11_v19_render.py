# -*- coding: utf-8 -*-
"""v19 = 場面の幅を広げる 4 つの新場面（2026-09-08・本人の指示「順に全部」）。描画は v16 と同一（v13 の generate_clip_v13）で、
場面のサンプラだけ 4 つ足す。既存の scenario（v11core / v12kick / v12bike / v12train）は v16 の sample_scene_v15 にそのまま渡す。

  v19fast    幹線の高速域: 車 1 台・40〜60 km/h（11.1〜16.7 m/s）。タイヤ音は速さの 3 乗に比例（+30·log10(v/8)）で最大 +8 dB
  v19static  停車中の警告音: 救急車が現場で停車して鳴らし続ける（サイレン・5〜40 m・方位一様・連続）/ 停車した車のクラクション（3〜15 m・1〜3 回）
             30% で通りすがりの車 1 台を足す
  v19turn    交差点で曲がって寄ってくる車: 交差道路（x = ±8〜28 m）を 22〜40 km/h で来て、半径 4〜8 m で曲がり（11〜18 km/h）、
             マイクの車線に入って通り過ぎる（18〜36 km/h）。最接近は plan の危険層どおり
  v19platoon 車の連なり: 同方向・同車線の 3 台（ラベル上限）。車頭間隔 1.2〜2.2 s、速さ 18〜43 km/h ±5%
  v19launch  発進（2026-09-21・変更票 #36 の (b)・本人 23:27 採用・数値 23:52 確定）: 静止待機（t=−1〜t_on・v=0・無音）→ 等加速度
             1.0〜2.5 m/s² で 5〜15 km/h まで → 等速で通り過ぎる。最接近は plan の危険層どおり・時刻 6.5〜9.0 s。折れ線は 0.5 s 刻み
             （誤差 ≤ a·dt²/8 = 0.078 m）。EV は終端速度の一定音で近似（make_car_ev(speed_mps=終端速度)）。
             出典 = md/design/発進場面_数値の出典調査_2026-09-21.md（加速度の実測平均 約 2.0 m/s² に合わせて上端 2.5・他は据え置き）

乱数: rng = default_rng(seed)（ピーク超過時は [seed, 20260908, salt] で引き直す決定論ラダー）。マイクは v12 と同じ規約
（静止 1.5 m / 歩行 1.0〜1.4 m/s）、高さは plan の mic_z（v16 と同じく描画前に差し替え）。雨は plan の rain 列（空なら無し）。
使い方:
  python scripts/step11_v19_render.py --preview v19turn [--n 200]     # 描画せずに幾何の統計だけ
  python scripts/step11_v19_render.py --render <clip_id> [--plan <csv>]  # 1 本描画（ローカル試走）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v16_render as v16r  # noqa: E402

m9 = v16r.m9
v13 = v16r.v13
v12 = v13.v12

DS_V19 = ROOT / "out" / "dataset_outdoor_siren_v19"
PLAN_V19 = DS_V19 / "plan"
m9.DS = DS_V19
m9.WORK = DS_V19 / "work"

SALT19 = 20260908
_SALT = 0                                   # ピーク再抽選ラダー（generate_clip が設定）

# ---- v19fast ----
FAST_SPEED = (11.1, 16.7)                   # 40〜60 km/h
FAST_V_REF = 8.0                            # この速さで補正 0 dB（core の速さの上寄り）
FAST_ADJ_MAX = 8.0                          # タイヤ音の速度依存 +30·log10(v/v_ref) の上限 [dB]
# ---- v19static ----
STATIC_SIREN_P = 0.6
STATIC_SIREN_DIST = (5.0, 40.0)             # 対数一様 [m]
STATIC_HORN_DIST = (3.0, 15.0)
STATIC_HORN_N_P = (0.5, 0.3, 0.2)           # 1/2/3 回
STATIC_HORN_SEC = (0.4, 1.2)                # 1 回の長さ（対数一様）
STATIC_HORN_GAP = (0.3, 0.8)
STATIC_CAR_P = 0.3
# ---- v19turn ----
TURN_XC_EXTRA = (4.0, 14.0)                 # 交差道路の x = ±(R + これ) [m]
TURN_R = (4.0, 8.0)                         # 曲がりの半径
TURN_V_APPROACH = (6.0, 11.0)               # 22〜40 km/h
TURN_V_CORNER = (3.0, 5.0)                  # 11〜18 km/h
TURN_V_AFTER = (5.0, 10.0)                  # 18〜36 km/h
TURN_TCPA = (6.5, 9.0)
TURN_ARC_N = 4                              # 円弧の分割数
# ---- v19platoon ----
PLATOON_N = 3
PLATOON_V = (5.0, 12.0)
PLATOON_V_JITTER = 0.05
PLATOON_HEADWAY = (1.2, 2.2)
PLATOON_TCPA1 = (3.5, 5.5)
PLATOON_LANE_JITTER = 0.3
PLATOON_LANE_MIN = 0.5                      # 2・3 台目の横距離の下限 [m]（#70: v29 が 1.0 に。点検 K5）
# ---- v19launch（2026-09-21・本人 23:52 確定。出典 = md/design/発進場面_数値の出典調査_2026-09-21.md。設計上の仮定として限界に書く） ----
LAUNCH_T_ON = (1.0, 3.0)                    # 静止待機の長さ [s]（t=−1〜t_on は v=0・無音。HCM 発進損失 2.0 s・発進遅れ 約 1.1 s を包む幅）
LAUNCH_ACCEL = (1.0, 2.5)                   # 等加速度 [m/s²]（一時停止交差点の実測 平均 約 2.0 = Kodsi & Muttart 2010・信号交差点 0〜1 s 1.18・1〜3 s 2.55 = Danaher 2022 を包む幅。一様分布の平均 1.75。Opus 5.5 監査 9/23 F13）
LAUNCH_V_END = (5.0 / 3.6, 15.0 / 3.6)      # 終端速度 5〜15 km/h（徐行。D10 と同じ帯。東大構内 15 km/h・駐車場 8 km/h 級）
LAUNCH_TCPA = (6.5, 9.0)                    # 最接近の時刻 [s]（turn と同じ）
LAUNCH_DT = 0.5                             # 加速区間の折れ線の刻み [s]（区分等速の誤差 ≤ a·dt²/8 = 0.078 m）
LAUNCH_LEVEL_ADJ_DB = (0.0, 0.0)            # 音量補正 [dB]（0 dB = 通常走行と同じ・本人 9/21 23:52 決定で据え置き。同じ速さの徐行 D10（−10〜−6 dB）より 6〜10 dB 大きい仮定 = 監査 F12。値を変えるなら変更票）
LAUNCH_V_REL_MIN = 0.5                      # 最接近時の相対速度の下限 [m/s]（歩行マイクと同方向で追いつけない引きを捨てる）
LAUNCH_MAX_TRIES = 30                       # 引き直しの上限（超えたら対向に固定＝対向・静止マイクは常に成立する）


def _rng(row: dict) -> np.random.Generator:
    seed = int(row["seed"])
    return np.random.default_rng([seed, SALT19, _SALT]) if _SALT else np.random.default_rng(seed)


def _tier(row: dict, rng: np.random.Generator) -> str:
    t = row.get("danger_tier") or ""
    return t if t in m9.TIER_CPA else m9.TIERS[int(rng.integers(3))]


def _side(row: dict, rng: np.random.Generator) -> float:
    s = row.get("car_side") or ""
    if s in ("L", "R"):
        return 1.0 if s == "L" else -1.0
    return 1.0 if rng.random() < 0.5 else -1.0


def _lane(rng: np.random.Generator, tier: str, z: float, sign: float):
    """車線の横位置 = m9.lane_offset（#70 点検 K1: 式の写しをやめて 1 か所に）。旧 = 3 次元の最接近を引いて横に直す・v29 = 横距離を直接引く。"""
    return m9.lane_offset(rng, tier, z, sign)


def _car_src(track: int, wp, v: float, dirx: float, tier: str, cpa: float, t_cpa: float,
             rng: np.random.Generator, seed: int, grammar: str, **extra) -> dict:
    lv, l1m = m9._draw_level("car_drive", rng)
    f0 = 42.0 * float(rng.uniform(0.8, 1.2))
    src = {"class": "car_drive", "kind": "vehicle", "track": track,
           "wp": [[float(a) for a in p] for p in wp],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": float(v), "dir_x": float(dirx),
           "danger_tier": tier, "cpa_rel_target_m": float(cpa), "t_cpa_rel_s": float(t_cpa),
           "law_db": lv, "l1m_db": l1m, "grammar": grammar,
           "params": {"f0": f0, "audio_seed": seed * 7 + 5 + track}}
    src.update(extra)
    return src


def _finish(row: dict, rng: np.random.Generator, mic_arr, mic: dict, sources: list, scene: dict) -> dict:
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": sources,
            "noise": {"dba": dba, "seed": int(row["seed"]) * 7919 + 13}, "v19": scene}


# ---------------------------------------------------------------- 1) 幹線の高速域 ----
def sample_v19fast(row: dict) -> dict:
    rng = _rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    tier = _tier(row, rng)
    sign = _side(row, rng)
    v = float(rng.uniform(*FAST_SPEED))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    z = m9.draw_src_z(rng, "car_drive")
    cpa, y = _lane(rng, tier, z, sign)
    t_cpa = float(rng.uniform(*m9.CAR_TCPA))
    x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * v * t_cpa
    wp = [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * v * m9.CLIP, y, z]]
    src = _car_src(0, wp, v, dirx, tier, cpa, t_cpa, rng, int(row["seed"]), "v19_fast")
    adj = float(np.clip(30.0 * np.log10(v / FAST_V_REF), 0.0, FAST_ADJ_MAX))
    src["speed_level_adj_db"] = round(adj, 2)
    src["law_db"] = float(src["law_db"]) + adj
    src["l1m_db"] = float(src["l1m_db"]) + adj
    src["params"]["f0"] = src["params"]["f0"] * (1.0 + 0.4 * (v - FAST_V_REF) / FAST_V_REF)
    return _finish(row, rng, mic_arr, mic, [src], {"kind": "fast", "speed_kmh": round(v * 3.6, 1), "level_adj_db": round(adj, 2)})


# ---------------------------------------------------------------- 2) 停車中の警告音 ----
def sample_v19static(row: dict) -> dict:
    rng = _rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    sources = []
    has_car = rng.random() < STATIC_CAR_P
    if has_car:
        tier = _tier(row, rng)
        sign = _side(row, rng)
        v = float(rng.uniform(*m9.SPEED["car_drive"]))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        z = m9.draw_src_z(rng, "car_drive")
        cpa, y = _lane(rng, tier, z, sign)
        t_cpa = float(rng.uniform(*m9.CAR_TCPA))
        x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * v * t_cpa
        sources.append(_car_src(0, [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * v * m9.CLIP, y, z]],
                                v, dirx, tier, cpa, t_cpa, rng, int(row["seed"]), "v19_static_car"))
    kind = "siren" if rng.random() < STATIC_SIREN_P else "horn"
    az = float(rng.uniform(0.0, 360.0))
    z = m9.draw_src_z(rng, kind)                                              # #70: サイレンは v24 が上書き・クラクションはクラス別
    p5 = m9._mic_pos_at(mic_arr, 5.0)
    c = m9.sound_speed(m9.TEMP_C)
    if kind == "siren":
        d = float(np.exp(rng.uniform(np.log(STATIC_SIREN_DIST[0]), np.log(STATIC_SIREN_DIST[1]))))
        pos = [float(p5[0] + d * np.cos(np.radians(az))), float(p5[1] + d * np.sin(np.radians(az))), z]
        lv, l1m = m9._draw_level("siren", rng)
        params = m9._warn_params("siren", rng, int(row["seed"]))
        sources.append({"class": "siren", "kind": "static", "track": 0,
                        "wp": [[0.0, *pos], [m9.CLIP, *pos]], "t_on": 0.0, "t_off": m9.CLIP,
                        "dist_draw_m": d, "az_draw_deg": az, "speed_mps": 0.0,
                        "preroll_s": float(np.ceil(d / c) + 1.0), "start_dist_m": round(d, 1),
                        "law_db": lv, "l1m_db": l1m, "params": params,
                        "grammar": "v19_static_siren", "siren_kind": "stopped"})
        scene = {"kind": "static_siren", "dist_m": round(d, 1), "az_deg": round(az, 1), "has_car": has_car}
    else:
        d = float(rng.uniform(*STATIC_HORN_DIST))
        pos = [float(p5[0] + d * np.cos(np.radians(az))), float(p5[1] + d * np.sin(np.radians(az))), z]
        lv, l1m = m9._draw_level("horn", rng)
        params = m9._warn_params("horn", rng, int(row["seed"]))
        n = int(rng.choice([1, 2, 3], p=STATIC_HORN_N_P))
        t = float(rng.uniform(0.5, 4.0))
        for i in range(n):
            dur = float(np.exp(rng.uniform(np.log(STATIC_HORN_SEC[0]), np.log(STATIC_HORN_SEC[1]))))
            t_off = min(t + dur, m9.CLIP - 0.2)
            # 同じ車の 2 回目・3 回目の吹鳴は track 1・2（ラベル作りが (class, track) を一意に要求する。時間は重ならない）
            sources.append({"class": "horn", "kind": "static", "track": i,
                            "wp": [[0.0, *pos], [m9.CLIP, *pos]], "t_on": round(t, 3), "t_off": round(t_off, 3),
                            "dist_draw_m": d, "az_draw_deg": az, "speed_mps": 0.0,
                            "law_db": lv, "l1m_db": l1m, "params": dict(params),
                            "grammar": "v19_static_horn", "honk_index": i, "n_honks": n})
            t = t_off + float(rng.uniform(*STATIC_HORN_GAP))
            if t >= m9.CLIP - 0.6:
                break
        scene = {"kind": "static_horn", "dist_m": round(d, 1), "az_deg": round(az, 1), "n_honks": n, "has_car": has_car}
    return _finish(row, rng, mic_arr, mic, sources, scene)


# ---------------------------------------------------------------- 3) 交差点で曲がって寄ってくる車 ----
def _turn_path(rng: np.random.Generator, mic_arr, y_lane: float, z: float, t_cpa: float):
    """交差道路 → 円弧 → マイクの車線、の折れ線（区分等速）。t=−1 から CLIP+1 までを覆う。

    r2（2026-09-08 監査 F1 の修正）: 交差道路の位置 x_c は「最接近時のマイク位置 x_cpa」を基準に置く
    （x_c = x_cpa + s_x·(R + extra)）。r1 は原点基準だったので、歩行マイクが交差道路側へ進んでいると
    円弧の出口が最接近点より先になり、車が最接近点へ戻る（瞬時に 180 度反転する）軌道が 8/1,500 本できた。
    r2 では円弧の出口から最接近点までの区間が常に進行方向へ extra（4〜14 m）になる。
    乱数の消費順は r1 と同じ（s_x, s_y, R, extra, v1, vt, v2）。
    """
    s_x = 1.0 if rng.random() < 0.5 else -1.0                 # 交差道路が +x 側か −x 側か
    s_y = 1.0 if rng.random() < 0.5 else -1.0                 # 車が来る側（車線から見て +y か −y）
    R = float(rng.uniform(*TURN_R))
    extra = float(rng.uniform(*TURN_XC_EXTRA))
    v1 = float(rng.uniform(*TURN_V_APPROACH))
    vt = float(rng.uniform(*TURN_V_CORNER))
    v2 = float(rng.uniform(*TURN_V_AFTER))
    x_cpa = float(m9._mic_pos_at(mic_arr, t_cpa)[0])
    x_c = x_cpa + s_x * (R + extra)                           # r2: 最接近時のマイク位置を基準に
    cx, cy = x_c - s_x * R, y_lane + s_y * R                  # 円弧の中心
    pts, spd = [], []                                         # 点列と「その点へ向かう区間」の速さ
    A = (x_c, y_lane + s_y * R)
    pts.append(A)
    for k in range(1, TURN_ARC_N + 1):
        th = np.pi / 2 * k / TURN_ARC_N
        pts.append((cx + R * s_x * np.cos(th), cy - R * s_y * np.sin(th)))
        spd.append(vt)
    P = (x_cpa, y_lane)
    pts.append(P)
    spd.append(0.5 * (vt + v2))
    # 時刻: CPA を t_cpa に固定して後ろ向きに積む
    times = [0.0] * len(pts)
    times[-1] = t_cpa
    for i in range(len(pts) - 2, -1, -1):
        seg = float(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        times[i] = times[i + 1] - seg / spd[i]
    # 先頭を t=−1 以前まで交差道路に沿って延長（速さ v1）
    t_a = times[0]
    pre = max(1.0, t_a + 1.0)                                 # A の何秒前から始めるか
    A0 = (x_c, y_lane + s_y * (R + v1 * pre))
    pts.insert(0, A0)
    times.insert(0, t_a - pre)
    # 末尾を CLIP+1 まで車線に沿って延長（速さ v2）
    t_end = m9.CLIP + 1.0
    E = (x_cpa - s_x * v2 * (t_end - t_cpa), y_lane)
    pts.append(E)
    times.append(t_end)
    wp = [[float(t), float(x), float(y), z] for t, (x, y) in zip(times, pts)]
    check_polyline(wp)                                        # 反転・時刻の逆行があれば生成を止める
    info = {"x_cross_m": round(x_c, 2), "radius_m": round(R, 2), "s_x": s_x, "s_y": s_y, "extra_m": round(extra, 2),
            "v_approach_mps": round(v1, 2), "v_corner_mps": round(vt, 2), "v_after_mps": round(v2, 2),
            "t_turn_start_s": round(t_a, 2), "t_turn_end_s": round(times[TURN_ARC_N + 1], 2), "turn_rev": 2}
    return wp, -s_x, v2, info


def check_polyline(wp, max_turn_deg: float = 120.0, max_speed: float = 20.0) -> None:
    """折れ線軌道の健全性: 時刻が増加・区間の向きが max_turn_deg 以上変わらない（瞬時の反転なし）・区間速度が上限内。"""
    a = np.asarray(wp, dtype=float)
    assert a.ndim == 2 and a.shape[1] == 4 and len(a) >= 2, "wp は (M,4)"
    dt = np.diff(a[:, 0])
    assert np.all(dt > 1e-6), f"時刻が増加していない: {a[:, 0].tolist()}"
    seg = np.diff(a[:, 1:3], axis=0)
    spd = np.linalg.norm(seg, axis=1) / dt
    assert np.all(spd <= max_speed + 1e-6), f"区間速度が上限 {max_speed} m/s 超: {spd.round(2).tolist()}"
    for i in range(1, len(seg)):
        n0, n1 = np.linalg.norm(seg[i - 1]), np.linalg.norm(seg[i])
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(seg[i - 1], seg[i]) / (n0 * n1), -1.0, 1.0))))
        assert ang <= max_turn_deg, f"区間 {i} で向きが {ang:.1f} 度変わる（瞬時の反転）: t={a[i, 0]:.3f}"


def check_plan(plan: Path) -> int:
    """plan の v19 行を全部サンプルして（salt=0）、折れ線軌道の健全性を検査する。違反数を返す。"""
    rows = load_plan_v19(plan)
    n = bad = 0
    for r in rows:
        if not str(r.get("scenario", "")).startswith("v19"):
            continue
        try:
            s = sample_scene_v19(r)
            for src in s["sources"]:
                if len(src["wp"]) > 2:
                    check_polyline(src["wp"])
                if r["scenario"] == "v19launch" and src["class"] == "car_drive":
                    check_launch(src, s["_mic_arr"])
            n += 1
        except AssertionError as e:
            bad += 1
            print(f"NG {r['clip_id']}: {e}")
    print(f"check_plan {plan.name}: {n} 本 OK / {bad} 本 NG")
    return bad


def sample_v19turn(row: dict) -> dict:
    rng = _rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    tier = _tier(row, rng)
    sign = _side(row, rng)
    z = m9.draw_src_z(rng, "car_drive")
    cpa, y_lane = _lane(rng, tier, z, sign)
    t_cpa = float(rng.uniform(*TURN_TCPA))
    wp, dirx, v2, info = _turn_path(rng, mic_arr, y_lane, z, t_cpa)
    src = _car_src(0, wp, v2, dirx, tier, cpa, t_cpa, rng, int(row["seed"]), "v19_turn", turn=info)
    return _finish(row, rng, mic_arr, mic, [src], dict(kind="turn", **info))


# ---------------------------------------------------------------- 4) 車の連なり ----
def sample_v19platoon(row: dict) -> dict:
    rng = _rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    tier = _tier(row, rng)
    sign = _side(row, rng)
    z0 = m9.draw_src_z(rng, "car_drive")
    cpa, y_lane = _lane(rng, tier, z0, sign)
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    v0 = float(rng.uniform(*PLATOON_V))
    h = float(rng.uniform(*PLATOON_HEADWAY))
    t1 = float(rng.uniform(*PLATOON_TCPA1))
    sources = []
    for j in range(PLATOON_N):
        vj = v0 * float(rng.uniform(1.0 - PLATOON_V_JITTER, 1.0 + PLATOON_V_JITTER))
        zj = z0 if j == 0 else m9.draw_src_z(rng, "car_drive")
        yj = y_lane
        if j > 0:
            jit = float(rng.uniform(-PLATOON_LANE_JITTER, PLATOON_LANE_JITTER))
            yj = sign * max(abs(y_lane) + jit, PLATOON_LANE_MIN)
        t_cpa = t1 + j * h
        x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * vj * t_cpa
        wp = [[0.0, x0, yj, zj], [m9.CLIP, x0 + dirx * vj * m9.CLIP, yj, zj]]
        sources.append(_car_src(j, wp, vj, dirx, tier, cpa if j == 0 else float(np.hypot(yj, m9.MIC_STATIC[2] - zj)),
                                t_cpa, rng, int(row["seed"]), "v19_platoon", platoon_index=j, headway_s=round(h, 2)))
    return _finish(row, rng, mic_arr, mic, sources,
                   {"kind": "platoon", "n": PLATOON_N, "headway_s": round(h, 2), "speed_kmh": round(v0 * 3.6, 1)})


# ---------------------------------------------------------------- 5) 発進（静止待機 → 等加速度 → 等速） ----
def _launch_dist(a: float, v_end: float, tau: float) -> float:
    """発進から tau 秒後の走行距離 [m]（等加速度 a で v_end に達し、その後は等速）。tau ≤ 0 は 0。"""
    if tau <= 0.0:
        return 0.0
    t_acc = v_end / a
    if tau <= t_acc:
        return 0.5 * a * tau * tau
    return 0.5 * a * t_acc * t_acc + v_end * (tau - t_acc)


def _launch_path(mic_arr, y_lane: float, z: float, t_cpa: float, t_on: float, a: float, v_end: float, dirx: float):
    """静止（t=−1〜t_on）→ 等加速度 → 等速 の折れ線。最接近 t_cpa でマイクの x に一致するよう x0 を後ろ向きに解く。

    節点: −1, t_on, t_on + k·LAUNCH_DT（加速中）, 加速の終わり, t_cpa（加速中でも x(t_cpa)=x_cpa を厳密に）, CLIP+1。
    区分等速の位置誤差は加速中 ≤ a·dt²/8（a=2.5, dt=0.5 で 0.078 m）、等速区間は 0。"""
    x_cpa = float(m9._mic_pos_at(mic_arr, t_cpa)[0])
    x0 = x_cpa - dirx * _launch_dist(a, v_end, t_cpa - t_on)
    t_end = m9.CLIP + 1.0
    t_acc_end = min(t_on + v_end / a, t_end)
    ts = [-1.0, t_on]
    k = 1
    while t_on + k * LAUNCH_DT < t_acc_end - 1e-9:
        ts.append(t_on + k * LAUNCH_DT)
        k += 1
    ts += [t_acc_end, t_cpa, t_end]
    ts = sorted(set(float(t) for t in ts))                  # 丸めない（t_on・t_cpa の節点が src の値と厳密に一致する）
    wp = [[t, float(x0 + dirx * _launch_dist(a, v_end, t - t_on)), float(y_lane), float(z)] for t in ts]
    check_polyline(wp)
    return wp, x0


def _launch_ok(wp, mic_arr, t_on: float, t_cpa: float, y_lane: float, v_rel: float) -> bool:
    """発進の成立条件: 最接近（水平距離の最小）が t_cpa（±0.05 s）で、値が |y_lane|（±0.02 m）、
    静止中（t<t_on）にマイクが車の脇を通り過ぎない（無音の最接近を作らない）、最接近時の相対速度 ≥ LAUNCH_V_REL_MIN。"""
    if v_rel < LAUNCH_V_REL_MIN:
        return False
    tgrid = np.arange(0.0, m9.CLIP, 0.01)
    ps = m9.receiver_positions_at(tgrid, np.asarray(wp, dtype=np.float64))
    mic = np.asarray(mic_arr, dtype=np.float64)
    pm = m9.receiver_positions_at(tgrid, mic) if mic.ndim == 2 else np.broadcast_to(mic, ps.shape)
    dh = np.hypot(ps[:, 0] - pm[:, 0], ps[:, 1] - pm[:, 1])
    k = int(np.argmin(dh))
    if abs(float(tgrid[k]) - t_cpa) > 0.05 or abs(float(dh[k]) - abs(y_lane)) > 0.02:
        return False
    pre = dh[tgrid < t_on]
    return bool(len(pre) == 0 or float(np.min(pre)) > abs(y_lane) + 0.05)


def check_launch(src: dict, mic_arr) -> None:
    """発進 1 台の検査（check_plan から呼ぶ）。違反は AssertionError。"""
    L = src.get("launch") or {}
    assert L, "launch の情報が無い"
    wp = np.asarray(src["wp"], dtype=float)
    t_on, t_cpa, a, v_end = float(src["t_on"]), float(src["t_cpa_rel_s"]), float(L["accel_mps2"]), float(L["v_end_mps"])
    assert 0.0 < t_on < t_cpa < m9.CLIP, f"時刻の順序: t_on={t_on} t_cpa={t_cpa}"
    assert abs(wp[0, 0] + 1.0) < 1e-6 and abs(wp[1, 0] - t_on) < 1e-6, "先頭 2 節点が t=−1, t_on でない"
    assert np.allclose(wp[0, 1:], wp[1, 1:]), "静止区間で位置が動いている"
    dt = np.diff(wp[:, 0])
    spd = np.hypot(np.diff(wp[:, 1]), np.diff(wp[:, 2])) / dt
    assert np.all(np.diff(spd[1:]) >= -1e-6), f"発進後の区間速度が減っている: {spd.round(3).tolist()}"
    assert abs(float(spd[-1]) - v_end) < 1e-6, f"終端速度 {spd[-1]:.4f} ≠ v_end {v_end:.4f}"
    assert np.all(spd <= 20.0 + 1e-6), "区間速度が上限超"
    acc_nodes = wp[(wp[:, 0] >= t_on - 1e-9) & (wp[:, 0] <= t_on + v_end / a + 1e-9), 0]
    assert np.all(np.diff(acc_nodes) <= LAUNCH_DT + 1e-6), f"加速中の節点間隔が {LAUNCH_DT} s 超"
    x_cpa = float(np.interp(t_cpa, wp[:, 0], wp[:, 1]))
    assert abs(x_cpa - float(m9._mic_pos_at(mic_arr, t_cpa)[0])) < 1e-6, "x(t_cpa) がマイクの x と一致しない"
    v_rel = float(L["v_rel_cpa_mps"])
    assert _launch_ok(wp, mic_arr, t_on, t_cpa, float(wp[0, 2]), v_rel), "最接近の時刻/距離/無音の通過/相対速度の条件に反する"
    check_polyline(wp)


def sample_v19launch(row: dict) -> dict:
    rng = _rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    tier = _tier(row, rng)
    sign = _side(row, rng)
    z = m9.draw_src_z(rng, "car_drive")
    cpa, y_lane = _lane(rng, tier, z, sign)
    walk_v = float(mic.get("walk_speed_mps", 0.0)) * float(mic.get("walk_dir_x", 0.0))   # 符号つき（静止は 0）
    n_try, forced = 0, False
    while True:
        n_try += 1
        t_on = round(float(rng.uniform(*LAUNCH_T_ON)), 3)          # 発音開始（ms 単位に丸めて src["t_on"] と節点を同じ値にする）
        a = float(rng.uniform(*LAUNCH_ACCEL))
        v_end = float(rng.uniform(*LAUNCH_V_END))
        t_cpa = float(rng.uniform(*LAUNCH_TCPA))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        if n_try > LAUNCH_MAX_TRIES and walk_v:
            dirx, forced = -float(np.sign(walk_v)), True                    # 保険: 対向は常に成立する
        wp, x0 = _launch_path(mic_arr, y_lane, z, t_cpa, t_on, a, v_end, dirx)
        v_cpa = min(a * (t_cpa - t_on), v_end)
        v_rel = abs(dirx * v_cpa - walk_v)
        if _launch_ok(wp, mic_arr, t_on, t_cpa, y_lane, v_rel):
            break
        assert n_try <= LAUNCH_MAX_TRIES + 1, f"発進の引き直しが {LAUNCH_MAX_TRIES} 回を超えた: {row.get('clip_id')}"
    adj = float(rng.uniform(*LAUNCH_LEVEL_ADJ_DB))
    p_on = m9._mic_pos_at(mic_arr, t_on)
    info = {"t_on_s": t_on, "accel_mps2": a, "v_end_mps": v_end, "v_end_kmh": round(v_end * 3.6, 1),    # a・v_end は丸めない（check_launch が節点と突き合わせる）
            "t_acc_s": round(v_end / a, 3), "t_cpa_s": round(t_cpa, 3), "dir_x": dirx,
            "start_dist_m": round(float(np.hypot(x0 - p_on[0], y_lane - p_on[1])), 2),
            "v_rel_cpa_mps": round(v_rel, 4), "walk_v_signed_mps": round(walk_v, 3),
            "same_dir_as_walk": (bool(dirx * walk_v > 0) if walk_v else None),
            "n_tries": n_try, "forced_head_on": forced, "level_adj_db": round(adj, 2),
            "silent_before_t_on": True, "ev_speed_approx": "terminal", "polyline_dt_s": LAUNCH_DT}
    src = _car_src(0, wp, v_end, dirx, tier, cpa, t_cpa, rng, int(row["seed"]), "v19_launch", launch=info)
    src["t_on"] = t_on                                                      # 静止中は無音（本人決定・監査 §9）
    if adj:
        src["speed_level_adj_db"] = round(adj, 2)
        src["law_db"] = float(src["law_db"]) + adj
        src["l1m_db"] = float(src["l1m_db"]) + adj
    return _finish(row, rng, mic_arr, mic, [src], dict(kind="launch", **info))


_SAMPLERS19 = {"v19fast": sample_v19fast, "v19static": sample_v19static,
               "v19turn": sample_v19turn, "v19platoon": sample_v19platoon,
               "v19launch": sample_v19launch}


def sample_scene_v19(row: dict) -> dict:
    fn = _SAMPLERS19.get(row.get("scenario"))
    if fn is None:
        return v16r.sample_scene_v15(row)
    s = fn(row)
    s = v13._attach_rain(s, row)
    s["grammar"] = row["scenario"]
    return v16r.set_mic_height(s, float(row.get("mic_z", 1.5)))


# v13.generate_clip_v13 は module-global の sample_scene_v13 を呼ぶ → v19 版に差し替える
v13.sample_scene_v13 = sample_scene_v19


def generate_clip(row: dict) -> None:
    """v19 場面はピーク超過時だけ salt=1,2,... で引き直す（v11/v13 と同方式）。既存場面は v13 の梯子に委ねる。"""
    global _SALT
    if not str(row.get("scenario", "")).startswith("v19"):
        return v13.generate_clip(row)
    last = None
    for salt in range(9):
        _SALT = salt
        try:
            return v13.generate_clip_v13(row)
        except AssertionError as e:
            if "peak" not in str(e):
                raise
            last = e
            print(f"[peak-resample] {row['clip_id']} salt={salt} NG ({e}) -> retry", flush=True)
        finally:
            _SALT = 0
    raise last


def load_plan_v19(path: Path | None = None) -> list:
    path = path or (PLAN_V19 / "assignment_v19.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"] or 0)
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


# ---------------------------------------------------------------- 試走 ----
def _fake_row(scen: str, i: int) -> dict:
    tiers = ("critical", "caution", "safe")
    return {"clip_id": f"preview_{scen}_{i}", "split": "fold1", "motion": "walk" if i % 10 < 7 else "static",
            "n_warnings": 0, "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
            "danger_tier": tiers[i % 3], "car_side": "L" if i % 2 else "R", "scenario": scen,
            "seed": 9190000000 + i, "scene_type": "v19", "n_car": "1", "rain": "", "rain_rate": "", "rain_dba": "",
            "grammar": "v19", "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "",
            "mic_z": f"{1.4 + 0.7 * ((i * 37) % 100) / 100:.3f}", "h_copy": "1"}


def preview(scen: str, n: int) -> None:
    tgrid = np.arange(0.0, m9.CLIP, 0.01)
    mins, tmins, spd, ncls = [], [], [], {}
    for i in range(n):
        s = sample_scene_v19(_fake_row(scen, i))
        mic = s["_mic_arr"]
        for src in s["sources"]:
            wp = np.array(src["wp"], float)
            d = m9._dist_series(wp, mic, tgrid)
            mins.append(float(np.min(d)))
            tmins.append(float(tgrid[int(np.argmin(d))]))
            spd.append(float(src.get("speed_mps", 0.0)))
            ncls[src["class"]] = ncls.get(src["class"], 0) + 1
    mins, tmins, spd = map(np.array, (mins, tmins, spd))
    print(f"{scen}: {n} 場面 / 音源 {len(mins)} {ncls}")
    print(f"  最接近 [m] 5/50/95%: {np.percentile(mins, 5):.2f} / {np.percentile(mins, 50):.2f} / {np.percentile(mins, 95):.2f}")
    print(f"  最接近の時刻 [s] 5/50/95%: {np.percentile(tmins, 5):.1f} / {np.percentile(tmins, 50):.1f} / {np.percentile(tmins, 95):.1f}")
    print(f"  速さ [m/s] 5/50/95%: {np.percentile(spd, 5):.1f} / {np.percentile(spd, 50):.1f} / {np.percentile(spd, 95):.1f}")
    if scen == "v19turn":
        s = sample_scene_v19(_fake_row(scen, 0))
        print("  例（曲がる車の折れ線）:", json.dumps(s["sources"][0]["wp"]))
        print("  例（情報）:", json.dumps(s["v19"]))
    if scen == "v19launch":
        infos = [sample_scene_v19(_fake_row(scen, i))["v19"] for i in range(n)]
        tries = np.array([x["n_tries"] for x in infos])
        same = [x["same_dir_as_walk"] for x in infos if x["same_dir_as_walk"] is not None]
        print(f"  t_on 5/50/95%: {np.percentile([x['t_on_s'] for x in infos], [5, 50, 95]).round(2).tolist()} s・"
              f"発進位置の距離 5/50/95%: {np.percentile([x['start_dist_m'] for x in infos], [5, 50, 95]).round(1).tolist()} m")
        print(f"  引き直し回数 平均 {tries.mean():.2f}・最大 {tries.max()}・対向に固定 {sum(x['forced_head_on'] for x in infos)} 本・"
              f"歩行マイクと同方向 {sum(same)}/{len(same)}")
        s = sample_scene_v19(_fake_row(scen, 0))
        print("  例（発進の折れ線）:", json.dumps([[round(v, 3) for v in p] for p in s["sources"][0]["wp"]]))
        print("  例（情報）:", json.dumps(s["v19"]))


def main() -> int:
    a = sys.argv[1:]
    if "--preview" in a:
        scen = a[a.index("--preview") + 1]
        n = int(a[a.index("--n") + 1]) if "--n" in a else 200
        for sc in (_SAMPLERS19 if scen == "all" else [scen]):
            preview(sc, n)
        return 0
    if "--render" in a:
        cid = a[a.index("--render") + 1]
        plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else None
        rows = {r["clip_id"]: r for r in load_plan_v19(plan)}
        generate_clip(rows[cid])
        return 0
    if "--check-plan" in a:                     # 監査 F1 の再発防止: plan 全行の軌道検査（生成しない）
        return 1 if check_plan(Path(a[a.index("--check-plan") + 1])) else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
