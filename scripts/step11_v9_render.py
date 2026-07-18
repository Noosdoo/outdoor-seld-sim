# -*- coding: utf-8 -*-
"""Step 11: v9 レンダラ = 絶対較正ミキサ＋事前チェック＋検品＋パッケージング。

設計の正: out/v9_design_v2_2026-07-16.md。割当表（step10_v9_plan.py の出力）だけを
入力に読む（idx算術なし）。v8までのstep6には一切触れない（凍結・新規ファイル方針）。

v8からの本質的な変更:
  - SNR/SIRのつまみは存在しない。全成分を物理の絶対音量（法規レンジ、A特性で較正、
    src/outdoor_seld/calibration.py が唯一の換算）で置き、ただ足す。post_scaleなし。
  - 車は妨害でなく第5クラス（ラベルあり、可聴フレームのみ）。第6クラス踏切=固定音源。
  - マイクは静止/歩行の混在（歩行=直線1.0-1.4m/s。geometry/fastsimの移動マイク経路）。
  - 危険層はマイク相対CPA min||car(t)-mic(t)|| で制御・記録（scene.json cpa_rel_*）。
  - 可聴性マスク（全クラス×全フレームのA特性SNR）を masks/<name>.csv に出力。

使い方（dynamic-sound venv の python で）:
  python scripts/step11_v9_render.py precheck            # 生成前チェック（レンダなし）
  python scripts/step11_v9_render.py gen 0-9 --set core  # core行のindex範囲を生成
  python scripts/step11_v9_render.py gen 0-19 --set scenario
  python scripts/step11_v9_render.py gen 0-47 --set probe
  python scripts/step11_v9_render.py inspect             # 全生成済みクリップの検品
  python scripts/step11_v9_render.py pack                # Colab用zip
"""
from __future__ import annotations

import csv
from collections import defaultdict
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.alert_sounds import (make_backup_beep, make_bike_bell,  # noqa: E402
                                       make_bike_bell_ring, make_crossing,
                                       make_crossing_v2, make_horn)
from outdoor_seld.calibration import (K_RMS_SPL, frame_spl_a,  # noqa: E402
                                      gain_for_spl_a, spl_a)
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.foa import encode_foa_timevarying, intensity_vector_doa  # noqa: E402
from outdoor_seld.geometry import (doa_unit_vectors,  # noqa: E402
                                   receiver_positions_at,
                                   solve_emission_times, sound_speed)
from outdoor_seld.labels import (frame_label_rows, read_dcase_csv,  # noqa: E402
                                 write_dcase_csv)
from outdoor_seld.noise import diffuse_foa_noise  # noqa: E402
from outdoor_seld.scene import decimate_to_out_rate  # noqa: E402
from outdoor_seld.siren import make_peepo_siren, make_siren  # noqa: E402

# --v91 = v9.1: 音源精査（out/source_audit_2026-07-17.md）の修正3点だけを反映した版。
#   ①踏切→make_crossing_v2（和音・打撃・余韻） ②wailサイレン→日本のパトカー仕様
#   （最高870Hz・周期4s/8s抽選・下限435Hz仮置き） ③自転車ベル→単打:引き打ち=50:50。
#   割当表・シード・他クラス・較正・検品はv9と完全同一。修正3クラスを含まない
#   クリップはv9とビット一致する。v9は凍結保存（比較材料）
# --v92 = v9.2追加分（学習追加180+対照180+幻覚評価30）を独立フォルダに生成する経路
#   （監査H5-3: v9.1フォルダへの追記は版管理違反 → out/dataset_outdoor_siren_v9_2_add/）。
#   音源はv9.1仕様（--v92は--v91を含意）。Colab側でのみ datasets/outdoor_siren_v9_1/ に
#   マージ展開する。設計= out/v9_2_design_2026-07-18.md（6節の改訂込み）
V92 = "--v92" in sys.argv
V91 = ("--v91" in sys.argv) or V92
DS_NAME = ("outdoor_siren_v9_2_add" if V92
           else ("outdoor_siren_v9_1" if V91 else "outdoor_siren_v9"))
DS = ROOT / "out" / f"dataset_{DS_NAME}"
# 割当表の正はv9のplan（v9.1も同一の表を使う。来歴用にv9.1側へもコピーされる）
PLAN = ROOT / "out" / "dataset_outdoor_siren_v9" / "plan"
WORK = DS / "work"
CLS_SRC = ROOT / "configs" / "cls_indices_v9.tsv"

FS_SIM, FS_OUT = 48000, 24000
CLIP = 10.0
MIC_STATIC = (0.0, 0.0, 1.5)
GROUND_Z = 0.0
TEMP_C, PRESS_ATM, RH = 20.0, 1.0, 50.0

CLASS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
             "car_drive": 4, "crossing": 5}

# 法規レンジ (dB(A) at 基準距離[m])。根拠: out/sound_class_research_2026-07-16.md /
# v9_values_research。較正は「1m相当レベル」= L + 20log10(ref_m)（幾何減衰1/r, r0=1m）
LAW = {"siren": (90.0, 120.0, 20.0), "horn": (93.0, 112.0, 7.0),
       "backup_beep": (87.0, 112.0, 1.2), "bike_bell": (80.0, 95.0, 1.0),
       "crossing": (75.0, 85.0, 1.0), "car_drive": (60.0, 67.0, 10.0)}
SPEED = {"siren": (5.0, 15.0), "horn": (5.0, 15.0), "backup_beep": (1.0, 3.0),
         "bike_bell": (3.0, 7.0), "car_drive": (5.0, 15.0)}
WALK_SPEED = (1.0, 1.4)
WARN_TCPA = (3.0, 7.0)
CAR_TCPA = (6.0, 9.0)          # 相対CPA時刻（設計1節: 接近の頭を収める）
EVENT_DUR = (2.0, 6.0)
WARN_OFFSET = (3.0, 15.0)      # 警告音車両の横距離（v8継承）
SRC_Z = (0.8, 1.3)
CROSS_DIST = (2.0, 20.0)       # 踏切: マイクからの水平距離
CROSS_Z = 2.5                  # 警報機スピーカ高さ（設計3節）
NOISE_DBA = (40.0, 65.0)       # 暗騒音 dB(A)（環境省基準アンカー）
TIER_CPA = {"critical": (0.6, 1.5), "caution": (1.55, 3.0), "safe": (3.2, 15.0)}
TIERS = ("critical", "caution", "safe")
GAP_FILL = 2                   # 車ラベルの可聴穴埋めフレーム数（パイロット規約）
AUDIBLE_SNR_DB = 0.0           # 可聴ゲート: フレームA特性SNR >= 0dB
PROBE_RECV_DBA = 70.0          # プローブ: 受聴A特性レベルをこの値に正規化
PROBE_NOISE_DBA = 45.0
PROBE_WIN = (3.0, 7.0)
PEAK_MAX = 0.99

# precheck用の波形クレスト率の見込み（ピーク上限の粗い事前計算にのみ使用。
# 最終的な正はgen時の実測ピーク assert）
CREST = {"siren": 1.8, "horn": 3.5, "backup_beep": 1.6, "bike_bell": 4.0,
         "crossing": 1.7, "car_drive": 4.0}


# ------------------------------------------------------------ 割当表の読込 ----

def load_plan(which: str) -> list:
    rows = []
    with open(PLAN / f"assignment_{which}.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


# ------------------------------------------------------- シーンのサンプル ----

def _mic_pos_at(mic, t: float) -> np.ndarray:
    mic = np.asarray(mic, dtype=np.float64)
    if mic.ndim == 1:
        return mic
    return receiver_positions_at(np.array([t]), mic)[0]


def _draw_level(cls: str, rng: np.random.Generator) -> tuple:
    lo, hi, ref = LAW[cls]
    lv = float(rng.uniform(lo, hi))
    return lv, lv + 20.0 * np.log10(ref)          # (法規値, 1m相当レベル)


def _warn_params(cls: str, rng: np.random.Generator, seed: int) -> dict:
    """クラス別のドライ音ジッタ（v8継承のレンジ）＋踏切の規格ジッタ。"""
    if cls == "siren":
        st = "peepo" if rng.random() < 0.5 else "wail"
        if st == "peepo":
            return {"siren_type": st,
                    "f_hi": 960.0 * float(rng.uniform(0.97, 1.03)),
                    "f_lo": 770.0 * float(rng.uniform(0.97, 1.03)),
                    "tone_sec": 0.65 * float(rng.uniform(0.9, 1.1))}
        if V91:
            # 日本のパトカー: 最高870Hz・折り返し4秒/8秒の2種（source_audit参照）。
            # 下限は出典未発見→1オクターブ下435Hzを仮置き（8月実測で検証）
            return {"siren_type": st,
                    "f_lo": 435.0 * float(rng.uniform(0.95, 1.05)),
                    "f_hi": 870.0 * float(rng.uniform(0.97, 1.03)),
                    "sweep_period_sec": (4.0 if rng.random() < 0.5 else 8.0)
                    * float(rng.uniform(0.9, 1.1))}
        return {"siren_type": st,
                "f_lo": 650.0 * float(rng.uniform(0.95, 1.05)),
                "f_hi": 1450.0 * float(rng.uniform(0.95, 1.05)),
                "sweep_period_sec": 4.8 * float(rng.uniform(0.85, 1.15))}
    if cls == "horn":
        return {"f_lo": 410.0 * float(rng.uniform(0.95, 1.05)),
                "f_hi": 500.0 * float(rng.uniform(0.95, 1.05)),
                "honk_sec": 0.35 * float(rng.uniform(0.9, 1.1)),
                "gap_sec": 0.15 * float(rng.uniform(0.9, 1.1)),
                "audio_seed": seed * 7 + 1}
    if cls == "backup_beep":
        return {"freq": 1000.0 * float(rng.uniform(0.95, 1.05)),
                "on_sec": 0.5 * float(rng.uniform(0.9, 1.1)),
                "off_sec": 0.5 * float(rng.uniform(0.9, 1.1))}
    if cls == "bike_bell":
        if V91 and rng.random() < 0.5:
            # 引き打ちベル「チリリン」（JIS D 9451のもう1方式、source_audit参照）
            return {"bell_type": "ring",
                    "f0": 3000.0 * float(rng.uniform(0.95, 1.05)),
                    "strike_rate_hz": 30.0 * float(rng.uniform(0.8, 1.2)),
                    "burst_sec": 0.65 * float(rng.uniform(0.8, 1.2)),
                    "gap_sec": 0.6 * float(rng.uniform(0.8, 1.2))}
        return {"f0": 3000.0 * float(rng.uniform(0.95, 1.05)),
                "ring_gap_sec": 0.45 * float(rng.uniform(0.9, 1.1)),
                "repeat_period_sec": 2.0 * float(rng.uniform(0.9, 1.1))}
    if cls == "crossing":
        out = {"f_a": 700.0 + float(rng.uniform(-15.0, 15.0)),
               "f_b": 750.0 + float(rng.uniform(-15.0, 15.0)),
               "rate_per_min": 130.0 + float(rng.uniform(-5.0, 5.0))}
        if V91:
            out["click_seed"] = seed * 7 + 9   # 打撃アタックのノイズ用
        return out
    raise ValueError(cls)


SCN2 = {"crossing_wait", "bell_overtake", "backup_reverse",
        "silent_negative", "siren_worstnoise"}


def sample_scenario2(row: dict, rng: np.random.Generator) -> dict:
    """追加5シナリオの幾何（out/scenario_design_2026-07-17.md）。--v91専用。

    乱数の消費順は各分岐内で固定（マイク→音源→暗騒音）。歩行は+x前進に固定
    （「背後」= −x側を定義するため。coreのランダム進行方向とは異なる=評価専用枠）。
    """
    scen = row["scenario"]
    side = 1.0 if row["w1_side"] == "L" else -1.0
    car_sign = 1.0 if row["car_side"] == "L" else -1.0
    s = {"row": dict(row), "sources": []}

    # ---- マイク ----
    if scen == "crossing_wait":
        v = float(rng.uniform(*WALK_SPEED))
        t_stop = float(rng.uniform(3.0, 5.0))
        mic = np.array([[0.0, -v * t_stop, 0.0, MIC_STATIC[2]],
                        [t_stop, 0.0, 0.0, MIC_STATIC[2]],
                        [CLIP, 0.0, 0.0, MIC_STATIC[2]]])
        mic_rec = {"motion": "walk_stop_x", "walk_speed_mps": v,
                   "t_stop_s": t_stop}
    elif row["motion"] == "walk":
        v = float(rng.uniform(*WALK_SPEED))
        mic = np.array([[0.0, -v * 5.0, 0.0, MIC_STATIC[2]],
                        [CLIP, v * 5.0, 0.0, MIC_STATIC[2]]])
        mic_rec = {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": 1.0}
    else:
        mic, mic_rec = np.array(MIC_STATIC), {"motion": "static"}

    # ---- 音源 ----
    if scen == "crossing_wait":
        lv, l1m = _draw_level("crossing", rng)
        params = _warn_params("crossing", rng, row["seed"])
        d = float(rng.uniform(2.5, 4.0))          # 停止位置から警報機まで（仮定）
        lat = float(rng.uniform(1.0, 2.0)) * side
        s["sources"].append({"class": "crossing", "kind": "static",
                             "wp": [[0.0, d, lat, CROSS_Z],
                                    [CLIP, d, lat, CROSS_Z]],
                             "t_on": 0.0, "t_off": CLIP, "law_db": lv,
                             "l1m_db": l1m, "params": params})
        vc = float(rng.uniform(3.0, 8.0))          # 踏切へ減速接近（道交法33条）の等速近似
        z = float(rng.uniform(*SRC_Z))
        dz = MIC_STATIC[2] - z
        cpa = float(rng.uniform(max(1.55, dz + 0.05), 3.0))
        y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * car_sign
        t_cpa = float(rng.uniform(*CAR_TCPA))
        lvc, l1mc = _draw_level("car_drive", rng)
        f0 = 42.0 * float(rng.uniform(0.8, 1.2))
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - vc * t_cpa   # 背後(−x)から+xへ
        s["sources"].append({"class": "car_drive", "kind": "vehicle",
                             "wp": [[0.0, x0, y, z], [CLIP, x0 + vc * CLIP, y, z]],
                             "t_on": 0.0, "t_off": CLIP, "speed_mps": vc,
                             "dir_x": 1.0, "cpa_rel_target_m": cpa,
                             "t_cpa_rel_s": t_cpa, "law_db": lvc, "l1m_db": l1mc,
                             "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
    elif scen == "bell_overtake":
        vb = float(rng.uniform(*SPEED["bike_bell"]))
        c = float(rng.uniform(0.8, 1.5))           # 側方間隔（改正道交法1m目安をアンカー）
        zb = float(rng.uniform(0.9, 1.1))          # ハンドル位置のベル高さ（仮定）
        dz = MIC_STATIC[2] - zb
        t_cpa = float(rng.uniform(*CAR_TCPA))
        lv, l1m = _draw_level("bike_bell", rng)
        params = _warn_params("bike_bell", rng, row["seed"])
        t_on = max(0.3, t_cpa - float(rng.uniform(2.0, 5.0)))
        dur = float(rng.uniform(*EVENT_DUR))
        t_off = min(t_on + dur, CLIP - 0.3)
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - vb * t_cpa
        s["sources"].append({"class": "bike_bell", "kind": "vehicle",
                             "wp": [[0.0, x0, c * side, zb],
                                    [CLIP, x0 + vb * CLIP, c * side, zb]],
                             "t_on": round(t_on, 3), "t_off": round(t_off, 3),
                             "speed_mps": vb, "dir_x": 1.0,
                             "cpa_rel_target_m": round(float(np.hypot(c, dz)), 3),
                             "t_cpa_rel_s": t_cpa, "law_db": lv, "l1m_db": l1m,
                             "params": params})
    elif scen == "backup_reverse":
        vc = float(rng.uniform(*SPEED["backup_beep"]))   # 後退速度1-3m/s
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        z = float(rng.uniform(*SRC_Z))
        dz = MIC_STATIC[2] - z
        lo, hi = TIER_CPA[row["danger_tier"]]
        cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
        y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * car_sign
        t_cpa = float(rng.uniform(*CAR_TCPA))
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - dirx * vc * t_cpa
        wp = [[0.0, x0, y, z], [CLIP, x0 + dirx * vc * CLIP, y, z]]
        lvc, l1mc = _draw_level("car_drive", rng)
        f0 = 42.0 * float(rng.uniform(0.8, 1.2))
        s["sources"].append({"class": "car_drive", "kind": "vehicle", "wp": wp,
                             "t_on": 0.0, "t_off": CLIP, "speed_mps": vc,
                             "dir_x": dirx, "cpa_rel_target_m": cpa,
                             "t_cpa_rel_s": t_cpa, "law_db": lvc, "l1m_db": l1mc,
                             "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
        lvb, l1mb = _draw_level("backup_beep", rng)
        pb = _warn_params("backup_beep", rng, row["seed"])
        s["sources"].append({"class": "backup_beep", "kind": "vehicle",
                             "wp": wp,               # 車体の音=同一軌道
                             "t_on": 0.0, "t_off": CLIP, "speed_mps": vc,
                             "dir_x": dirx, "law_db": lvb, "l1m_db": l1mb,
                             "params": pb})
    elif scen == "siren_worstnoise":
        lv, l1m = _draw_level("siren", rng)
        params = _warn_params("siren", rng, row["seed"])
        vs = float(rng.uniform(8.0, 15.0))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        y = float(rng.uniform(10.0, 15.0)) * side
        z = float(rng.uniform(*SRC_Z))
        t_cpa = float(rng.uniform(*CAR_TCPA))
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - dirx * vs * t_cpa
        s["sources"].append({"class": "siren", "kind": "vehicle",
                             "wp": [[0.0, x0, y, z], [CLIP, x0 + dirx * vs * CLIP, y, z]],
                             "t_on": 0.0, "t_off": CLIP, "speed_mps": vs,
                             "dir_x": dirx, "t_cpa_rel_s": t_cpa,
                             "law_db": lv, "l1m_db": l1m, "params": params})
    # silent_negative: 音源なし

    dba = (float(rng.uniform(60.0, 65.0)) if scen == "siren_worstnoise"
           else float(rng.uniform(*NOISE_DBA)))
    s["noise"] = {"dba": dba, "seed": row["seed"] * 7919 + 13}
    if mic.ndim == 2:
        mic_rec["waypoints"] = mic.tolist()
    s["mic"] = mic_rec
    s["_mic_arr"] = mic
    return s


def sample_v10a(row: dict, rng: np.random.Generator) -> dict:
    """v10a 交通量・複数車（scenario='trafficN'）。--v91専用・評価専用。

    車N台（1台目は割当表のside/tier、2台目以降はseedから）。各車は独立軌道で
    track=0..N-1。CPA時刻はU(4,9)に分散（通知の発火頻度を測る枠のため、
    coreの「接近の頭を収める」6-9s規約より広く取る=設計上の意図）。
    警告音は0/1個（クラスは割当表、幾何はcoreと同じ規約）。
    """
    ncars = int(row["scenario"][-1])
    s = {"row": dict(row), "sources": []}
    if row["motion"] == "walk":
        v = float(rng.uniform(*WALK_SPEED))
        mic = np.array([[0.0, -v * 5.0, 0.0, MIC_STATIC[2]],
                        [CLIP, v * 5.0, 0.0, MIC_STATIC[2]]])
        s["mic"] = {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": 1.0,
                    "waypoints": mic.tolist()}
    else:
        mic = np.array(MIC_STATIC)
        s["mic"] = {"motion": "static"}

    for j in range(ncars):
        if j == 0:
            sign = 1.0 if row["car_side"] == "L" else -1.0
            tier = row["danger_tier"]
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            tier = TIERS[int(rng.integers(3))]
        vc = float(rng.uniform(*SPEED["car_drive"]))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        z = float(rng.uniform(*SRC_Z))
        dz = MIC_STATIC[2] - z
        lo, hi = TIER_CPA[tier]
        cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
        y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * sign
        t_cpa = float(rng.uniform(4.0, 9.0))
        lv, l1m = _draw_level("car_drive", rng)
        f0 = 42.0 * float(rng.uniform(0.8, 1.2))
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - dirx * vc * t_cpa
        s["sources"].append({
            "class": "car_drive", "kind": "vehicle", "track": j,
            "wp": [[0.0, x0, y, z], [CLIP, x0 + dirx * vc * CLIP, y, z]],
            "t_on": 0.0, "t_off": CLIP, "speed_mps": vc, "dir_x": dirx,
            "danger_tier": tier, "cpa_rel_target_m": cpa, "t_cpa_rel_s": t_cpa,
            "law_db": lv, "l1m_db": l1m,
            "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5 + j}})

    cls = row["w1_class"]
    if cls:
        side = 1.0 if row["w1_side"] == "L" else -1.0
        lv, l1m = _draw_level(cls, rng)
        params = _warn_params(cls, rng, row["seed"])
        if cls == "crossing":
            d = float(rng.uniform(*CROSS_DIST))
            az = float(rng.uniform(5.0, 175.0)) * side
            p5 = _mic_pos_at(mic, 5.0)
            pos = [float(p5[0] + d * np.cos(np.radians(az))),
                   float(p5[1] + d * np.sin(np.radians(az))), CROSS_Z]
            s["sources"].append({"class": cls, "kind": "static",
                                 "wp": [[0.0, *pos], [CLIP, *pos]],
                                 "t_on": 0.0, "t_off": CLIP, "dist_draw_m": d,
                                 "law_db": lv, "l1m_db": l1m, "params": params})
        else:
            vw = float(rng.uniform(*SPEED[cls]))
            dirx = 1.0 if rng.random() < 0.5 else -1.0
            yw = float(rng.uniform(*WARN_OFFSET)) * side
            t_cpa = float(rng.uniform(*WARN_TCPA))
            z = float(rng.uniform(*SRC_Z))
            dur = float(rng.uniform(*EVENT_DUR))
            t_on = float(rng.uniform(0.3, CLIP - dur - 0.3))
            x0 = -dirx * vw * t_cpa
            s["sources"].append({"class": cls, "kind": "vehicle",
                                 "wp": [[0.0, x0, yw, z],
                                        [CLIP, x0 + dirx * vw * CLIP, yw, z]],
                                 "t_on": round(t_on, 3),
                                 "t_off": round(t_on + dur, 3),
                                 "speed_mps": vw, "dir_x": dirx,
                                 "t_cpa_s": t_cpa, "law_db": lv,
                                 "l1m_db": l1m, "params": params})

    s["noise"] = {"dba": float(rng.uniform(*NOISE_DBA)),
                  "seed": row["seed"] * 7919 + 13}
    s["_mic_arr"] = mic
    return s


def sample_scene_v9(row: dict) -> dict:
    """割当表の1行＋seedから、そのクリップの全条件を決定論的に確定する。

    乱数の消費順（変更禁止。変更すると全クリップが変わる）:
      ①マイク → ②車 → ③w1 → ④w2 → ⑤暗騒音
    返り値 s["_mic_arr"] = マイク表現（静止=(3,) / 歩行=(M,4)軌道）。
    """
    rng = np.random.default_rng(row["seed"])
    if row["scenario"] in SCN2:
        return sample_scenario2(row, rng)
    if row["scenario"].startswith("traffic"):
        return sample_v10a(row, rng)
    if row["scenario"] == "carfree_siren":
        # 幻覚評価: 車なし×連続サイレン（room9より広い幾何: 横3-15m・通常暗騒音）
        if row["motion"] == "walk":
            v = float(rng.uniform(*WALK_SPEED))
            mic = np.array([[0.0, -v * 5.0, 0.0, MIC_STATIC[2]],
                            [CLIP, v * 5.0, 0.0, MIC_STATIC[2]]])
            mic_rec = {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": 1.0,
                       "waypoints": mic.tolist()}
        else:
            mic, mic_rec = np.array(MIC_STATIC), {"motion": "static"}
        s = {"row": dict(row), "mic": mic_rec, "sources": [], "_mic_arr": mic}
        side = 1.0 if row["w1_side"] == "L" else -1.0
        lv, l1m = _draw_level("siren", rng)
        params = _warn_params("siren", rng, row["seed"])
        vs = float(rng.uniform(*SPEED["siren"]))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        y = float(rng.uniform(*WARN_OFFSET)) * side
        z = float(rng.uniform(*SRC_Z))
        t_cpa = float(rng.uniform(*CAR_TCPA))
        x0 = float(_mic_pos_at(mic, t_cpa)[0]) - dirx * vs * t_cpa
        s["sources"].append({"class": "siren", "kind": "vehicle",
                             "wp": [[0.0, x0, y, z],
                                    [CLIP, x0 + dirx * vs * CLIP, y, z]],
                             "t_on": 0.0, "t_off": CLIP, "speed_mps": vs,
                             "dir_x": dirx, "t_cpa_rel_s": t_cpa,
                             "law_db": lv, "l1m_db": l1m, "params": params})
        s["noise"] = {"dba": float(rng.uniform(*NOISE_DBA)),
                      "seed": row["seed"] * 7919 + 13}
        return s
    is_probe = row["scenario"].startswith("probe_")

    # ① マイク（歩行はx軸沿い直線・t=5sに原点。交差点サイレンだけ横断=y軸沿い）
    if row["motion"] == "static":
        mic, mic_rec = np.array(MIC_STATIC), {"motion": "static"}
    else:
        v = float(rng.uniform(*WALK_SPEED))
        if row["scenario"] == "intersection_siren":
            # 横断しようと歩いてきて、縁石（車道の0.5m手前）で立ち止まる歩行者。
            # 車線は常に y=+2.5〜3.5 → サイレンとの最小距離は3m以上が保証され、
            # 本編の最悪ケース（横距離3m）を超えない（precheckで発見した
            # 「車線に入り込んでクリップする」幾何の修正、07-17）
            t_stop = float(rng.uniform(4.0, 6.0))
            y_stop = -0.5
            y0 = y_stop - v * t_stop
            mic = np.array([[0.0, 0.0, y0, MIC_STATIC[2]],
                            [t_stop, 0.0, y_stop, MIC_STATIC[2]],
                            [CLIP, 0.0, y_stop, MIC_STATIC[2]]])
            mic_rec = {"motion": "walk_cross_y", "walk_speed_mps": v,
                       "t_stop_s": t_stop}
        else:
            dirx = 1.0 if rng.random() < 0.5 else -1.0
            x0 = -dirx * v * 5.0
            mic = np.array([[0.0, x0, 0.0, MIC_STATIC[2]],
                            [CLIP, x0 + dirx * v * CLIP, 0.0, MIC_STATIC[2]]])
            mic_rec = {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": dirx}
    s = {"row": dict(row), "mic": mic_rec, "sources": [], "_mic_arr": mic}

    # ② 車（car_sideが空なら車なし＝交差点シナリオ枠・警告プローブ）
    if row["car_side"] in ("L", "R"):
        v = float(rng.uniform(*SPEED["car_drive"]))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        z = float(rng.uniform(*SRC_Z))
        dz = MIC_STATIC[2] - z
        lo, hi = ((4.0, 8.0) if is_probe else TIER_CPA[row["danger_tier"]])
        cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
        # 両者ともx軸沿いなので 相対CPA = sqrt(y_off^2 + dz^2) が厳密に成り立つ
        y_off = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6)))
        y_off *= 1.0 if row["car_side"] == "L" else -1.0
        t_cpa = float(rng.uniform(*CAR_TCPA))
        lv, l1m = _draw_level("car_drive", rng)
        f0 = 42.0 * float(rng.uniform(0.8, 1.2))
        x_mic = float(_mic_pos_at(mic, t_cpa)[0])
        x0 = x_mic - dirx * v * t_cpa      # 相対x=0（=相対CPA）が t_cpa に来る配置
        s["sources"].append({
            "class": "car_drive", "kind": "vehicle",
            "wp": [[0.0, x0, y_off, z], [CLIP, x0 + dirx * v * CLIP, y_off, z]],
            "t_on": 0.0, "t_off": CLIP, "speed_mps": v, "dir_x": dirx,
            "cpa_rel_target_m": cpa, "t_cpa_rel_s": t_cpa,
            "law_db": lv, "l1m_db": l1m,
            "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})

    # ③④ 警告音（v9.2: w1==w2の同一クラス2本はw2をtrack=1にする）
    for key in ("w1", "w2"):
        cls = row[f"{key}_class"]
        if not cls:
            continue
        track = 1 if (key == "w2" and cls == row["w1_class"]) else 0
        side = 1.0 if row[f"{key}_side"] == "L" else -1.0
        lv, l1m = _draw_level(cls, rng)
        params = _warn_params(cls, rng, row["seed"])
        if cls == "crossing":
            d = float(rng.uniform(*(CROSS_DIST if not is_probe else (5.0, 10.0))))
            az = float(rng.uniform(5.0, 175.0)) * side
            p5 = _mic_pos_at(mic, 5.0)
            pos = [float(p5[0] + d * np.cos(np.radians(az))),
                   float(p5[1] + d * np.sin(np.radians(az))), CROSS_Z]
            src = {"class": cls, "kind": "static",
                   "wp": [[0.0, *pos], [CLIP, *pos]],
                   "t_on": 0.0, "t_off": CLIP, "dist_draw_m": d,
                   "law_db": lv, "l1m_db": l1m, "params": params}
        elif row["scenario"] == "intersection_siren":
            # 連続吹鳴サイレンが道路(x軸)を走り、横断中のマイクの前を横切る
            v = float(rng.uniform(8.0, 15.0))
            dirx = -1.0 if row[f"{key}_side"] == "L" else 1.0   # L=+x側から接近
            lane = float(rng.uniform(2.5, 3.5))   # マイク停止線(y=-0.5)の先の車線
            t_pass = float(rng.uniform(6.5, 9.0))
            z = float(rng.uniform(*SRC_Z))
            x0 = -dirx * v * t_pass
            src = {"class": cls, "kind": "vehicle",
                   "wp": [[0.0, x0, lane, z], [CLIP, x0 + dirx * v * CLIP, lane, z]],
                   "t_on": 0.0, "t_off": CLIP, "speed_mps": v, "dir_x": dirx,
                   "t_pass_s": t_pass, "law_db": lv, "l1m_db": l1m, "params": params}
        else:
            v = float(rng.uniform(*SPEED[cls]))
            dirx = 1.0 if rng.random() < 0.5 else -1.0
            y = float(rng.uniform(*(WARN_OFFSET if not is_probe else (5.0, 10.0)))) * side
            t_cpa = 5.0 if is_probe else float(rng.uniform(*WARN_TCPA))
            z = float(rng.uniform(*SRC_Z))
            dur = 4.0 if is_probe else float(rng.uniform(*EVENT_DUR))
            t_on = PROBE_WIN[0] if is_probe else float(rng.uniform(0.3, CLIP - dur - 0.3))
            x0 = -dirx * v * t_cpa
            src = {"class": cls, "kind": "vehicle",
                   "wp": [[0.0, x0, y, z], [CLIP, x0 + dirx * v * CLIP, y, z]],
                   "t_on": round(t_on, 3), "t_off": round(t_on + dur, 3),
                   "speed_mps": v, "dir_x": dirx, "t_cpa_s": t_cpa,
                   "law_db": lv, "l1m_db": l1m, "params": params}
        src["track"] = track
        s["sources"].append(src)

    # ⑤ 暗騒音（プローブは固定レベル）
    s["noise"] = {"dba": (PROBE_NOISE_DBA if is_probe
                          else float(rng.uniform(*NOISE_DBA))),
                  "seed": row["seed"] * 7919 + 13}
    return s


# ------------------------------------------------------------- レンダリング ----

def _make_dry(src: dict) -> np.ndarray:
    cls, p = src["class"], src["params"]
    if cls == "siren":
        gen = make_peepo_siren if p["siren_type"] == "peepo" else make_siren
        return gen(CLIP, FS_SIM, **{k: v for k, v in p.items()
                                    if k != "siren_type"})
    if cls == "horn":
        rng = np.random.default_rng(p["audio_seed"])
        return make_horn(CLIP, FS_SIM, rng,
                         **{k: v for k, v in p.items() if k != "audio_seed"})
    if cls == "backup_beep":
        return make_backup_beep(CLIP, FS_SIM, **p)
    if cls == "bike_bell":
        if p.get("bell_type") == "ring":
            return make_bike_bell_ring(
                CLIP, FS_SIM, **{k: v for k, v in p.items() if k != "bell_type"})
        return make_bike_bell(CLIP, FS_SIM, **p)
    if cls == "crossing":
        if V91:
            rng = np.random.default_rng(p["click_seed"])
            kw = {k: v for k, v in p.items() if k != "click_seed"}
            return make_crossing_v2(CLIP, FS_SIM, rng=rng, **kw)
        return make_crossing(CLIP, FS_SIM, **p)
    if cls == "car_drive":
        rng = np.random.default_rng(p["audio_seed"])
        return make_car_v9(CLIP, FS_SIM, rng, f0=p["f0"])
    raise ValueError(cls)


def _window(dry: np.ndarray, t_on: float, t_off: float) -> np.ndarray:
    if t_on <= 0.0 and t_off >= CLIP:
        return dry
    env = np.zeros(len(dry))
    i0, i1 = int(t_on * FS_SIM), int(t_off * FS_SIM)
    env[i0:i1] = 1.0
    fade = int(0.02 * FS_SIM)
    env[i0:i0 + fade] = np.linspace(0.0, 1.0, fade)
    env[i1 - fade:i1] = np.linspace(0.0, 1.0, fade)[::-1]
    return dry * env


def _render_stem(dry_cal: np.ndarray, wp: np.ndarray, mic, c: float):
    """1音源のFOAステム（直接のみ, 直接+地面反射）を返す。"""
    n24 = int(CLIP * FS_OUT)
    tr = np.arange(n24) / FS_OUT
    mic_arr = np.asarray(mic, dtype=np.float64)
    foas = {}
    for tag in ("direct", "mirror"):
        w = wp.copy()
        if tag == "mirror":
            w[:, 3] = 2.0 * GROUND_Z - w[:, 3]
        mono48 = render_mono(dry_cal, w, mic_arr, FS_SIM, CLIP,
                             temperature_c=TEMP_C, pressure_atm=PRESS_ATM,
                             rel_humidity=RH)
        mono24 = decimate_to_out_rate(mono48, FS_SIM, FS_OUT)
        te, ps = solve_emission_times(tr, w, mic_arr, c)
        mic_at = (receiver_positions_at(tr, mic_arr) if mic_arr.ndim == 2
                  else mic_arr)
        u, _ = doa_unit_vectors(ps, mic_at)
        foas[tag] = encode_foa_timevarying(mono24, u)
    return foas["direct"], foas["direct"] + foas["mirror"]


def _dist_series(wp, mic, tgrid: np.ndarray) -> np.ndarray:
    """音源-マイク間距離の時系列（同時刻の位置同士。CPA記録・precheck用）。"""
    ps = receiver_positions_at(tgrid, np.asarray(wp, dtype=np.float64))
    mic_arr = np.asarray(mic, dtype=np.float64)
    pm = (receiver_positions_at(tgrid, mic_arr) if mic_arr.ndim == 2
          else np.broadcast_to(mic_arr, ps.shape))
    return np.linalg.norm(ps - pm, axis=1)


def _fill_gaps(audible: np.ndarray, max_gap: int) -> np.ndarray:
    """可聴フレーム列の短い穴（<=max_gap）を埋める（パイロットstep9と同じ規約）。"""
    out = audible.copy()
    idx = np.where(audible)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a:b] = True
    return out


def generate_clip(row: dict) -> None:
    t_start = time.perf_counter()
    s = sample_scene_v9(row)
    mic = s.pop("_mic_arr")
    name = row["clip_id"]
    is_probe = row["scenario"].startswith("probe_")
    wdir = WORK / name
    wdir.mkdir(parents=True, exist_ok=True)
    for sub in ("foa", "metadata", "masks"):
        (DS / sub).mkdir(parents=True, exist_ok=True)
    c = sound_speed(TEMP_C)
    n24 = int(CLIP * FS_OUT)
    tgrid = np.arange(0.0, CLIP, 0.01)

    stems = []
    for src in s["sources"]:
        dry = _window(_make_dry(src), src["t_on"], src["t_off"])
        a0, a1 = int(src["t_on"] * FS_SIM), int(src["t_off"] * FS_SIM)
        g = gain_for_spl_a(dry[a0:a1], FS_SIM, src["l1m_db"])
        src["dry_gain"] = g
        stem_d, stem_wr = _render_stem(dry * g, np.array(src["wp"], float),
                                       mic, c)
        # レベル正規化プローブ: 受聴A特性レベルを共通値に再スケール（案A''の柱）
        if is_probe:
            b0, b1 = int(PROBE_WIN[0] * FS_OUT), int(PROBE_WIN[1] * FS_OUT)
            gp = gain_for_spl_a(stem_wr[0, b0:b1], FS_OUT, PROBE_RECV_DBA)
            stem_d, stem_wr = stem_d * gp, stem_wr * gp
            src["probe_gain"] = gp
        dist = _dist_series(src["wp"], mic, tgrid)
        src["min_dist_m"] = round(float(np.min(dist)), 3)
        src["t_min_dist_s"] = round(float(tgrid[int(np.argmin(dist))]), 3)
        if src["class"] == "car_drive" and src.get("track", 0) == 0:
            s["cpa_rel_dist_m"] = src["min_dist_m"]
            s["cpa_rel_time_s"] = src["t_min_dist_s"]
        # 較正の照合は【直接音ステム】で行う: 予測=発音窓内の 1/d^2 エネルギー平均
        # （動く音源はCPA近傍が支配するため中間時刻の距離では不可）。反射込みは
        # 純音のコム・ヌル（例: 995Hzビープが経路差半波長で−13dB。実物理）で
        # 予測から大きく外れうるので、ゲートには使わず記録列に残す
        win = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
        p_geo = float(np.mean(1.0 / dist[win] ** 2))
        src["recv_pred_db"] = round(src["l1m_db"] + 10.0 * np.log10(p_geo), 2)
        b0, b1 = int(src["t_on"] * FS_OUT), int(src["t_off"] * FS_OUT)
        src["recv_meas_db"] = round(spl_a(stem_d[0, b0:b1], FS_OUT), 2)
        src["recv_withrefl_db"] = round(spl_a(stem_wr[0, b0:b1], FS_OUT), 2)
        stems.append((src, stem_d, stem_wr))

    rng_n = np.random.default_rng(s["noise"]["seed"])
    noise = diffuse_foa_noise(n24, FS_OUT, rng_n)
    g_n = gain_for_spl_a(noise[0], FS_OUT, s["noise"]["dba"])
    noise = noise * g_n
    s["noise"]["gain"] = float(g_n)

    mix = noise.copy()
    for _, _, stem_wr in stems:
        mix = mix + stem_wr
    peak = float(np.max(np.abs(mix)))
    assert peak < PEAK_MAX, f"{name}: peak {peak:.3f} >= {PEAK_MAX}"

    # ラベルと可聴性マスク（全クラス。設計4節）
    fr_noise = frame_spl_a(noise[0], FS_OUT)
    label_rows, mask_rows = [], []
    for src, _, stem_wr in stems:
        cls_idx = CLASS_IDX[src["class"]]
        snr = frame_spl_a(stem_wr[0], FS_OUT) - fr_noise
        mask_rows += [(k, cls_idx, round(float(snr[k]), 2))
                      for k in range(len(snr))]
        rows, _ = frame_label_rows(np.array(src["wp"], float), mic,
                                   clip_len_sec=CLIP, class_idx=cls_idx,
                                   track_idx=src.get("track", 0),
                                   source_active_from=src["t_on"],
                                   source_active_until=src["t_off"], c=c)
        if src["class"] == "car_drive":
            audible = _fill_gaps(snr >= AUDIBLE_SNR_DB, GAP_FILL)
            rows = [r for r in rows if audible[r[0]]]
            src["audible_frac"] = round(float(np.mean(snr >= AUDIBLE_SNR_DB)), 3)
            aud_idx = np.where(snr >= AUDIBLE_SNR_DB)[0]
            src["t_first_audible_s"] = (round(float(aud_idx[0]) * 0.1, 2)
                                        if len(aud_idx) else None)
        else:
            k0, k1 = int(src["t_on"] * 10), max(int(src["t_off"] * 10), 1)
            src["audible_frac"] = round(float(np.mean(
                snr[k0:k1] >= AUDIBLE_SNR_DB)), 3)
        src["n_label_frames"] = len(rows)
        label_rows += rows
    label_rows.sort(key=lambda r: (r[0], r[1]))

    # 出力
    for i, (src, stem_d, stem_wr) in enumerate(stems):
        tag = f"src{i}_{src['class']}"
        sf.write(wdir / f"{tag}_direct_24k.flac",
                 stem_d.T.astype(np.float32), FS_OUT, subtype="PCM_24")
        sf.write(wdir / f"{tag}_withrefl_24k.flac",
                 stem_wr.T.astype(np.float32), FS_OUT, subtype="PCM_24")
    sf.write(DS / "foa" / f"{name}.flac", mix.T.astype(np.float32),
             FS_OUT, subtype="PCM_24")
    write_dcase_csv(DS / "metadata" / f"{name}.csv", label_rows)
    with open(DS / "masks" / f"{name}.csv", "w", newline="\n") as f:
        f.write("frame,class,snr_a_db\n")
        for k, ci, v in mask_rows:
            f.write(f"{k},{ci},{v}\n")

    s["stats"] = {"peak": round(peak, 4),
                  "mix_spl_a": round(spl_a(mix[0], FS_OUT), 2),
                  "n_label_rows": len(label_rows),
                  "gen_seconds": round(time.perf_counter() - t_start, 1)}
    (wdir / "scene.json").write_text(json.dumps(s, indent=2, default=str))
    srcs = "+".join(x["class"] for x in s["sources"]) or "none"
    print(f"[gen] {name} {row['motion']} tier={row['danger_tier']} {srcs} "
          f"noise={s['noise']['dba']:.1f}dBA peak={peak:.3f} "
          f"({s['stats']['gen_seconds']:.0f}s)")


# --------------------------------------------------------------- precheck ----

def precheck() -> int:
    """設計9節の生成前チェック（レンダなし・全行の解析計算）。"""
    sets = ["core", "scenario", "probe"]
    for extra in ("scenario2", "v10a"):
        if (PLAN / f"assignment_{extra}.csv").exists():
            sets.append(extra)
    all_rows = [(w, r) for w in sets for r in load_plan(w)]
    tgrid = np.arange(0.0, CLIP, 0.05)
    peak_list, aud = [], {}
    car_leads = []
    recv_by_class = {}
    cpa_err_max = 0.0
    for which, row in all_rows:
        s = sample_scene_v9(row)
        mic = s.pop("_mic_arr")
        noise_dba = s["noise"]["dba"]
        peak_bound = 4.0 * 10.0 ** ((noise_dba - K_RMS_SPL) / 20.0)
        for src in s["sources"]:
            dist = _dist_series(src["wp"], mic, tgrid)
            active = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
            recv = src["l1m_db"] - 20.0 * np.log10(np.maximum(dist, 0.3))
            dmin = float(np.min(dist[active]))
            wpm = np.array(src["wp"], float)
            wpm[:, 3] = 2.0 * GROUND_Z - wpm[:, 3]
            dmin_m = float(np.min(_dist_series(wpm, mic, tgrid)))
            rms = 10.0 ** ((src["l1m_db"] - K_RMS_SPL) / 20.0) / dmin
            # 反射同相加算の上限: 直接＋鏡像（距離比で小さい）が同位相で足される場合。
            # クレスト率(CREST)はピーク/RMS比そのもの（√2は含まれるので二重掛けしない）
            peak_bound += CREST[src["class"]] * rms * (1.0 + dmin / dmin_m)
            frac = float(np.mean(recv[active] >= noise_dba))
            aud.setdefault(src["class"], []).append(frac)
            if which == "core":
                # クリップの受音レベル（窓内エネルギー平均、AUC事前予測用）
                p_geo = float(np.mean(1.0 / np.maximum(dist[active], 0.3) ** 2))
                recv_by_class.setdefault(src["class"], []).append(
                    src["l1m_db"] + 10.0 * np.log10(p_geo))
            if src["class"] == "car_drive" and which == "core":
                # 相対CPAの検算（設計: 割当のtierレンジに一致するはず）
                cpa_err_max = max(cpa_err_max,
                                  abs(float(np.min(dist))
                                      - src["cpa_rel_target_m"]))
                audible_t = tgrid[recv >= noise_dba]
                car_leads.append(src["t_cpa_rel_s"] - float(audible_t[0])
                                 if len(audible_t) else -1.0)
        peak_list.append((float(peak_bound), row["clip_id"]))

    peak_list.sort(reverse=True)
    leads = np.array(car_leads)
    lines = ["# v9 生成前チェック（step11 precheck、レンダなしの解析計算）", ""]
    lines.append(f"- 行数: {len(all_rows)}（core+scenario+probe）")
    lines.append(f"- 相対CPAの検算: 設計値との最大差 {cpa_err_max:.3f} m"
                 "（0.05s格子の離散化誤差の範囲であること）")
    lines.append("- ピーク上限（クレスト率×反射同相の保守的バウンド）上位5:")
    for pb, cid in peak_list[:5]:
        lines.append(f"    {cid}: {pb:.3f}")
    lines.append(f"  -> 最大 {peak_list[0][0]:.3f}（1.0未満なら本番assertも確実。"
                 "超える行はスモークで実測ピークを確認する）")
    lines.append("- クラス別の可聴フレーム率の予測（発音窓内、吸収・反射無視の点近似）:")
    for k in sorted(aud):
        v = np.array(aud[k])
        lines.append(f"    {k}: mean={v.mean():.2f} p10={np.percentile(v, 10):.2f} "
                     f"完全不可聴率={(v < 0.05).mean():.2%}")
    if len(leads):
        lines.append(f"- 車のオラクル・リードタイム予測（可聴開始→相対CPA): "
                     f"median={np.median(leads):.1f}s  >=2.5s: "
                     f"{(leads >= 2.5).mean():.1%}  >=2.0s: "
                     f"{(leads >= 2.0).mean():.1%}  "
                     f"クリップ内不可聴: {(leads < 0).mean():.1%}")
    # 音量AUCの事前予測（設計9節3項の事前登録）: クリップ受音レベルだけで
    # クラス対を判別できてしまう度合い。法規レンジが重ならない対は≈1.0になるが、
    # それは「現実の正当な差」（案A''）。音色識別の証明はプローブ枠が担う
    lines.append("- 音量AUCの事前予測（受音レベルのみでのクラス対判別、案A''の事前登録値）:")
    cl = sorted(recv_by_class)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a = np.asarray(recv_by_class[cl[i]])[:, None]
            b = np.asarray(recv_by_class[cl[j]])[None, :]
            auc = float(((a > b).mean() + 0.5 * (a == b).mean()))
            auc = max(auc, 1.0 - auc)
            lines.append(f"    {cl[i]} vs {cl[j]}: AUC={auc:.3f}")
    report = "\n".join(lines) + "\n"
    outdir = DS / "plan"           # v9.1はv9のplanを読むが、レポートは自分側に書く
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "precheck_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


# ---------------------------------------------------------------- inspect ----

def inspect_all() -> int:
    clips = sorted((DS / "foa").glob("*.flac"))
    print(f"inspecting {len(clips)} clips ...")
    rows_out, n_fail = [], 0
    for flac in clips:
        name = flac.stem
        s = json.loads((WORK / name / "scene.json").read_text())
        mix = np.asarray(sf.read(flac)[0], np.float64).T
        labels = read_dcase_csv(DS / "metadata" / f"{name}.csv")
        stems = []
        for i, src in enumerate(s["sources"]):
            tag = f"src{i}_{src['class']}"
            d = np.asarray(sf.read(WORK / name / f"{tag}_direct_24k.flac")[0],
                           np.float64).T
            wr = np.asarray(
                sf.read(WORK / name / f"{tag}_withrefl_24k.flac")[0],
                np.float64).T
            stems.append((src, d, wr))
        # 1) 再構成: mix == Σステム + 雑音(シード再生成)。FLAC量子化ぶんだけ許容
        rng_n = np.random.default_rng(s["noise"]["seed"])
        noise = diffuse_foa_noise(mix.shape[1], FS_OUT, rng_n)
        noise = noise * gain_for_spl_a(noise[0], FS_OUT, s["noise"]["dba"])
        resid = mix - noise
        for _, _, wr in stems:
            resid = resid - wr
        recon_err = float(np.max(np.abs(resid)))
        # 2) 暗騒音の実測（mix−Σステム ≒ 雑音そのもの）
        res_noise = mix.copy()
        for _, _, wr in stems:
            res_noise = res_noise - wr
        noise_meas = spl_a(res_noise[0], FS_OUT)
        noise_ok = abs(noise_meas - s["noise"]["dba"]) < 0.3
        # 3) 音源ごと: IVラベル整合（直接音ステム）・受音レベルの粗ゲート。
        #    同一クラス複数トラック（v10a複数車）は自分のトラックのラベルとだけ照合
        #    （最近傍照合だと「自車が未可聴・他車がラベル中」の時刻に他車と照合して
        #    大誤差を出す偽FAILになる。2026-07-18修正）
        lab_track = defaultdict(dict)   # frame -> (class,track) -> (az,el)
        for line in open(DS / "metadata" / f"{name}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                lab_track[int(q[0])][(int(q[1]), int(q[2]))] = (
                    float(q[3]), float(q[4]))
        az_meds, el_meds = [], []
        recv_ok = True
        for src, stem_d, _ in stems:
            key = (CLASS_IDX[src["class"]], src.get("track", 0))
            frames = [k for k, d in lab_track.items() if key in d]
            if len(frames) >= 5:
                _, az_iv, el_iv, en = intensity_vector_doa(
                    stem_d, FS_OUT, frame_sec=0.1, fmin=200, fmax=4000)
                az_iv[en < np.max(en) * 1e-6] = np.nan
                errs_a, errs_e = [], []
                for k in frames:
                    if not (k < len(az_iv) and np.isfinite(az_iv[k])):
                        continue
                    gaz, gel = lab_track[k][key]
                    errs_a.append(abs((az_iv[k] - gaz + 180) % 360 - 180))
                    errs_e.append(abs(el_iv[k] - gel))
                if errs_a:
                    az_meds.append(float(np.median(errs_a)))
                    el_meds.append(float(np.median(errs_e)))
            # ±3.5dB: 打音系（ベル等）は「鳴る瞬間×距離」の偶然の相関で
            # 窓平均予測から±3dB程度まで正常にずれる（fold8_mix017で実測−3.14dB）
            if ("probe_gain" not in src
                    and abs(src["recv_meas_db"] - src["recv_pred_db"]) > 3.5):
                recv_ok = False
        az_med = max(az_meds) if az_meds else 0.0
        el_med = max(el_meds) if el_meds else 0.0
        peak = float(np.max(np.abs(mix)))
        n_mask = sum(1 for _ in open(DS / "masks" / f"{name}.csv")) - 1
        mask_ok = n_mask == 100 * len(stems)
        ok = (recon_err < 2e-6 and noise_ok and az_med < 2.0 and el_med < 2.0
              and peak < PEAK_MAX and mask_ok and recv_ok)
        n_fail += 0 if ok else 1
        rows_out.append({
            "name": name, "motion": s["row"]["motion"],
            "tier": s["row"]["danger_tier"], "scenario": s["row"]["scenario"],
            "n_src": len(stems), "peak": round(peak, 4),
            "recon_err": f"{recon_err:.2e}",
            "noise_dba_target": s["noise"]["dba"],
            "noise_dba_meas": round(noise_meas, 2),
            "az_med_max": round(az_med, 3), "el_med_max": round(el_med, 3),
            "cpa_rel_dist_m": s.get("cpa_rel_dist_m", ""),
            "cpa_rel_time_s": s.get("cpa_rel_time_s", ""),
            "car_audible_frac": next((x["audible_frac"] for x in s["sources"]
                                      if x["class"] == "car_drive"), ""),
            "recv_ok": recv_ok, "mask_ok": mask_ok,
            "result": "PASS" if ok else "FAIL"})
        print(f"  {name} {rows_out[-1]['result']} az={az_med:.2f} "
              f"el={el_med:.2f} peak={peak:.3f} recon={recon_err:.1e} "
              f"noise_d={noise_meas - s['noise']['dba']:+.2f}dB")
    with open(DS / "inspection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n{len(rows_out)} clips, {n_fail} FAIL -> {DS / 'inspection.csv'}")
    return n_fail


def pack() -> None:
    import shutil
    shutil.copyfile(CLS_SRC, DS / "cls_indices_train.tsv")
    zip_path = ROOT / "out" / f"dataset_{DS_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        zf.write(DS / "cls_indices_train.tsv", "datasets/cls_indices_train.tsv")
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((DS / sub).glob("*")):
                zf.write(p, f"datasets/{DS_NAME}/{sub}/{p.name}")
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def pack_v10a() -> None:
    """v10a（fold8_room1、交通量60本）の追補zip。既存datasets/へ追記展開。"""
    zip_path = ROOT / "out" / f"dataset_{DS_NAME}_v10a.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((DS / sub).glob("fold8_room1_*")):
                zf.write(p, f"datasets/{DS_NAME}/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


def pack_v92() -> None:
    """v9.2追加分のzip。展開先はColabの datasets/outdoor_siren_v9_1/（マージ）。"""
    zip_path = ROOT / "out" / "dataset_outdoor_siren_v9_2_add.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((DS / sub).glob("*")):
                zf.write(p, f"datasets/outdoor_siren_v9_1/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


def pack_scn2() -> None:
    """追加シナリオ（room4〜8）だけの追補zip。既存のdatasets/へ追記展開する。

    silent_negative の空メタデータCSVはPSELDNetsが読めないため、**zip内のコピーだけ**
    frame0にダミー行(0,0,0,0,0)を入れる（mix035の教訓）。採点は必ずローカルの
    scene.json/空CSVを正とする（設計書S4の実装注記）。
    """
    zip_path = ROOT / "out" / f"dataset_{DS_NAME}_scn2.zip"
    rooms = tuple(f"fold2_room{k}_" for k in (4, 5, 6, 7, 8))
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((DS / sub).glob("*")):
                if not p.name.startswith(rooms):
                    continue
                if sub == "metadata" and p.stat().st_size == 0:
                    zf.writestr(f"datasets/{DS_NAME}/{sub}/{p.name}",
                                "0,0,0,0,0\n")   # ダミー行（zip内のみ）
                else:
                    zf.write(p, f"datasets/{DS_NAME}/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    which = "core"
    if "--set" in sys.argv:
        which = sys.argv[sys.argv.index("--set") + 1]
    if mode == "precheck":
        sys.exit(precheck())
    elif mode == "gen":
        rows = load_plan(which)
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        for i in range(int(a), int(b) + 1):
            generate_clip(rows[i])
    elif mode == "inspect":
        sys.exit(1 if inspect_all() else 0)
    elif mode == "pack":
        pack()
    elif mode == "pack_scn2":
        pack_scn2()
    elif mode == "pack_v10a":
        pack_v10a()
    elif mode == "pack_v92":
        pack_v92()
    else:
        print("usage: precheck | gen A-B [--set core|scenario|probe|scenario2] "
              "| inspect | pack | pack_scn2")
