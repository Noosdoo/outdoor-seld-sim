# -*- coding: utf-8 -*-
"""v14 描画 = v13 描画 ＋ D10「至近・低速」の車の書き換え（2026-09-03）。

v13（step11_v13_render.py）をそのまま使い、plan の close_slow=1 の行だけ、track 0 の車（car_drive）を
  速度 cs_speed_kmh・3D最接近 cs_cpa_m・音量 +cs_level_adj_db
に置き換える。通過側（L/R）・進行向き・最接近時刻・音色（f0・audio_seed）・高さ z は元のまま。
出力先は out/dataset_outdoor_siren_v14/（v13 以前には書かない）。

⚠️ 徐行の音は「同じエンジン音を小さくしただけ」（v9 の車ドライ音は速度で音色が変わらない）。
   タイヤ音の減少・エンジン回転の低下は入っていない＝近似。実録の徐行車で見直す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step11_v13_render as v13  # noqa: E402

m9 = v13.m9
DS_NAME_V14 = "outdoor_siren_v14"
m9.DS = ROOT / "out" / f"dataset_{DS_NAME_V14}"
m9.WORK = m9.DS / "work"
PLAN_V14 = m9.DS / "plan"

_sample_scene_v13 = v13.sample_scene_v13


def rewrite_close_slow(s: dict, row: dict) -> dict:
    """close_slow=1 の行: track0 の車を徐行・至近に置き換える（他の音源は不変）。"""
    if str(row.get("close_slow", "")) != "1":
        return s
    v = float(row["cs_speed_kmh"]) / 3.6
    cpa = float(row["cs_cpa_m"])
    adj = float(row["cs_level_adj_db"])
    mic = s["_mic_arr"]
    done = False
    for src in s["sources"]:
        if src.get("class") != "car_drive" or src.get("kind") != "vehicle" or src.get("track", 0) != 0 or done:
            continue
        wp = np.array(src["wp"], float)
        z = float(wp[0, 3])
        dz = m9.MIC_STATIC[2] - z
        cpa_eff = max(cpa, dz + 0.05)                       # 高さ差より小さい最接近は幾何的に不可能
        y_off = float(np.sqrt(max(cpa_eff * cpa_eff - dz * dz, 1e-6))) * (1.0 if wp[0, 2] >= 0 else -1.0)
        dirx = float(src.get("dir_x", 1.0))
        t_cpa = float(src["t_cpa_rel_s"])
        x_mic = float(m9._mic_pos_at(mic, t_cpa)[0])
        x0 = x_mic - dirx * v * t_cpa
        src["wp"] = [[0.0, x0, y_off, z], [m9.CLIP, x0 + dirx * v * m9.CLIP, y_off, z]]
        src["speed_mps"] = v
        src["cpa_rel_target_m"] = cpa_eff
        src["l1m_db"] = float(src["l1m_db"]) + adj
        src["law_db"] = float(src["law_db"]) + adj
        src["close_slow"] = True
        done = True
    assert done, f"close_slow 行に track0 の車が無い: {row.get('clip_id')}"
    return s


def sample_scene_v14(row: dict) -> dict:
    return rewrite_close_slow(_sample_scene_v13(row), row)


# v13 の generate_clip は module-global の sample_scene_v13 を呼ぶので、そこを差し替える
v13.sample_scene_v13 = sample_scene_v14
generate_clip = v13.generate_clip_v13


def load_plan_v14() -> list:
    import csv
    rows = []
    with open(PLAN_V14 / "assignment_v14.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def close_frames(row: dict, th=(1.0, 1.5)) -> dict:
    """描画せずに、track0 の車が 3D 距離 ≤ th にいるフレーム数（100 フレーム中）を数える。"""
    s = sample_scene_v14(row)
    mic = s["_mic_arr"]
    tk = np.arange(100) * 0.1
    out = {f"le{t}": 0 for t in th}
    out["has_car"] = 0
    for src in s["sources"]:
        if src.get("class") != "car_drive" or src.get("kind") != "vehicle":
            continue
        out["has_car"] = 1
        d = m9._dist_series(np.array(src["wp"], float), mic, tk)
        for t in th:
            out[f"le{t}"] += int(np.sum(d <= t))
    return out
