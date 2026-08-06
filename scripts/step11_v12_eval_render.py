# -*- coding: utf-8 -*-
"""v12評価専用セットのレンダラ（チェーン方式: step11_v12_renderを継承）。

plan= out/dataset_outdoor_siren_v12_eval/plan/assignment_v12eval.csv（1,500本）
  e_heavy: 単独大型車・safe幾何（CPA 3.2〜15m）——E系評価（至近/選別）の常設化
  e_far  : サイレン/車(50:50)を100〜300mからスポоン——「聞こえる前」の検知距離測定
  e_kick / e_bike: mix0001-0150=基本（騒音40〜65） / 0151-0300=苦手（騒音58〜65）
出力: out/dataset_outdoor_siren_v12_eval/（新規）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v12_render as m12  # noqa: E402
m9 = m12.m9

# 出力先を評価専用に切替（本体v12は不変更）
DS_EVAL = ROOT / "out" / "dataset_outdoor_siren_v12_eval"
m9.DS = DS_EVAL
m9.WORK = DS_EVAL / "work"
PLAN_EVAL = DS_EVAL / "plan"


def load_plan_v12eval() -> list:
    rows = []
    with open(PLAN_EVAL / "assignment_v12eval.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def _mixno(row) -> int:
    return int(row["clip_id"].split("mix")[1])


def sample_e_heavy(row: dict) -> dict:
    """単独大型車・safe幾何（v11 safe600と同型のCPA 3.2〜15m・強制heavy）。"""
    rng = np.random.default_rng(row["seed"])
    mic_arr, mic = m12._mic_sample(row, rng)
    side = float(rng.choice([-1.0, 1.0]))
    cpa = float(rng.uniform(3.2, 15.0))
    speed = float(rng.uniform(3.0, 10.0))
    dirx = float(rng.choice([-1.0, 1.0]))
    t_cpa = float(rng.uniform(4.0, 8.0))
    x0 = -dirx * speed * t_cpa
    f0 = float(rng.uniform(42.0 * 0.8, 42.0 * 1.2))
    law = float(rng.uniform(60.0, 67.0))
    l1m = law + 20.0 * np.log10(10.0)
    src = {"class": "car_drive", "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, cpa * side, 1.5],
                  [m9.CLIP, x0 + dirx * speed * m9.CLIP, cpa * side, 1.5]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "cpa_rel_target_m": cpa, "danger_tier": "safe",
           "law_db": law, "l1m_db": l1m,
           "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5,
                      "force_variant": "heavy"}}
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": [src],
            "noise": {"dba": dba, "seed": row["seed"] * 7919 + 13}}


def sample_e_far(row: dict) -> dict:
    """長距離接近: サイレン/車(50:50)を100〜300mからスポーン（横3m・直進接近）。"""
    rng = np.random.default_rng(row["seed"])
    mic_arr, mic = m12._mic_sample(row, rng)
    side = float(rng.choice([-1.0, 1.0]))
    L = float(rng.uniform(100.0, 300.0))
    dirx = float(rng.choice([-1.0, 1.0]))
    is_siren = rng.random() < 0.5
    if is_siren:
        speed = float(rng.uniform(5.0, 15.0))
        params = m9._warn_params("siren", rng, row["seed"])
        law = float(rng.uniform(90.0, 120.0))
        l1m = law + 20.0 * np.log10(20.0)
        cls, kind = "siren", "vehicle"
    else:
        speed = float(rng.uniform(3.0, 10.0))
        params = {"f0": float(rng.uniform(42.0 * 0.8, 42.0 * 1.2)),
                  "audio_seed": row["seed"] * 7 + 5}
        law = float(rng.uniform(60.0, 67.0))
        l1m = law + 20.0 * np.log10(10.0)
        cls, kind = "car_drive", "vehicle"
    x0 = -dirx * L                      # t=0で距離≈L、そのまま接近（クリップ内でCPA未到達可）
    src = {"class": cls, "kind": kind, "track": 0,
           "wp": [[0.0, x0, 3.0 * side, 1.5],
                  [m9.CLIP, x0 + dirx * speed * m9.CLIP, 3.0 * side, 1.5]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "spawn_dist_m": L, "law_db": law, "l1m_db": l1m, "params": params}
    if cls == "car_drive":
        src["danger_tier"] = "far"
        src["cpa_rel_target_m"] = 3.0
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": [src],
            "noise": {"dba": dba, "seed": row["seed"] * 7919 + 13}}


def sample_e_kick(row: dict) -> dict:
    s = m12.sample_v12kick(row)
    if _mixno(row) > 150:               # 苦手側: 騒音上限帯
        rng2 = np.random.default_rng(row["seed"] * 13 + 7)
        s["noise"]["dba"] = float(rng2.uniform(58.0, 65.0))
    return s


def sample_e_bike(row: dict) -> dict:
    s = m12.sample_v12bike(row)
    if _mixno(row) > 150:
        rng2 = np.random.default_rng(row["seed"] * 13 + 7)
        s["noise"]["dba"] = float(rng2.uniform(58.0, 65.0))
    return s


_EVAL_SAMPLERS = {"e_heavy": sample_e_heavy, "e_far": sample_e_far,
                  "e_kick": sample_e_kick, "e_bike": sample_e_bike}
_orig_sample_v12 = m12.sample_scene_v12


def sample_scene_v12e(row: dict) -> dict:
    fn = _EVAL_SAMPLERS.get(row["scenario"])
    return fn(row) if fn else _orig_sample_v12(row)


m12.sample_scene_v12 = sample_scene_v12e


def smoke() -> None:
    import json
    rows = load_plan_v12eval()
    picks = {}
    for r in rows:
        picks.setdefault(r["scenario"], r)
    for scen, r in picks.items():
        m12.generate_clip_v12(r)
        s = json.loads((m9.WORK / r["clip_id"] / "scene.json").read_text())
        info = [(x["class"], x.get("car_variant"), x.get("spawn_dist_m"),
                 x.get("n_label_frames")) for x in s["sources"]]
        print(f"  [{scen}] {r['clip_id']} {info} noise={s['noise']['dba']:.1f}")


if __name__ == "__main__":
    if "smoke" in sys.argv:
        smoke()
    else:
        print(__doc__)
