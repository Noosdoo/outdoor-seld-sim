# -*- coding: utf-8 -*-
"""v16 描画 = v15 描画（v14 の場面 ＋ D13 マイク高さ）そのまま。plan だけ v16（各場面を 2 つの高さで描く＝高さ増強、2026-09-05）。

v14（step11_v14_render.py）の場面サンプルをそのまま使い、マイクの高さ z だけを plan の mic_z に置き換える。
静止マイク (3,) は z を、歩行マイク (M,4) は z 列を差し替える。音源の高さ・横距離・時刻は不変。
出力先は out/dataset_outdoor_siren_v16/（v15 以前には書かない）。

⚠️ 横距離を保ったまま高さだけ変えるので、3D 最接近は高いマイクほど大きくなる（1.5 m で引いた危険層の 3D 境界は
   そのままでは成り立たない）。距離ラベルは 3D のまま（v1 との確定評価の比較を保つため）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step11_v14_render as v14  # noqa: E402

m9 = v14.m9
v13 = v14.v13
DS_NAME_V15 = "outdoor_siren_v16"
m9.DS = ROOT / "out" / f"dataset_{DS_NAME_V15}"
m9.WORK = m9.DS / "work"
PLAN_V15 = m9.DS / "plan"

_sample_scene_v14 = v14.sample_scene_v14


def set_mic_height(s: dict, mic_z: float) -> dict:
    mic = np.asarray(s["_mic_arr"], dtype=float)
    if mic.ndim == 1:
        mic = mic.copy(); mic[2] = mic_z
    else:
        mic = mic.copy(); mic[:, 3] = mic_z
    s["_mic_arr"] = mic
    s.setdefault("mic", {})["mic_z"] = float(mic_z)
    return s


def sample_scene_v15(row: dict) -> dict:
    s = _sample_scene_v14(row)
    return set_mic_height(s, float(row.get("mic_z", 1.5)))


# v13.generate_clip_v13 は module-global の sample_scene_v13 を呼ぶ（v14 が v14 版に差し替え済み）→ v15 版に差し替える
v13.sample_scene_v13 = sample_scene_v15
generate_clip = v13.generate_clip_v13


def load_plan_v15() -> list:
    import csv
    rows = []
    with open(PLAN_V15 / "assignment_v16.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def cpa_stats(row: dict) -> dict:
    """描画せずに、track0 の車の 3D 最接近と水平（横）距離を、1.5 m と mic_z の両方で返す。"""
    out = {}
    for tag, z in (("1.5", 1.5), ("v15", float(row.get("mic_z", 1.5)))):
        s = set_mic_height(_sample_scene_v14(row), z)
        mic = s["_mic_arr"]
        tk = np.arange(100) * 0.1
        for src in s["sources"]:
            if src.get("class") != "car_drive" or src.get("kind") != "vehicle" or src.get("track", 0) != 0:
                continue
            wp = np.array(src["wp"], float)
            d = m9._dist_series(wp, mic, tk)
            k = int(np.argmin(d))
            pm = m9.receiver_positions_at(np.array([tk[k]]), mic)[0] if mic.ndim == 2 else mic
            ps = m9.receiver_positions_at(np.array([tk[k]]), wp)[0]
            out[tag] = (float(d[k]), float(np.hypot(ps[0] - pm[0], ps[1] - pm[1])))
            break
    return out
