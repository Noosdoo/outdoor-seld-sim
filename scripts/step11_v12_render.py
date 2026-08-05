# -*- coding: utf-8 -*-
"""step11_v12_render.py — v12 リアリズム強化レンダラ（チェーン方式）。

設計= md/design/v12設計書_2026-08-05.md（追記2の統合チェックリスト準拠）。
step11_v11_render のチェーンを継承し、**物理・較正・RNG消費順は無変更**のまま:
  ①背景騒音: 完全ピンク → 都市暗騒音（noise_v12、63Hz峰・実測文献準拠）
  ②車: audio_seed決定論抽選で EV15%（engine_ev）/ 大型30%（63Hz帯+3dBデルタ）/ 通常55%
出力先: out/dataset_outdoor_siren_v12/（新規。v11は凍結のまま）

このファイルは既存7,200行（v11 plan）の同一seed再生成を担当する。
列車・キックボードの追加行は step10_v12_plan.py + 本ファイルのサンプラ拡張（次段）。

使い方:
  python scripts/step11_v12_render.py smoke        # 3クリップ生成＋検品出力
  （全量は _run_v12_gen.py を用意予定）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v11_render as m11v  # noqa: E402 (v11チェーン: サンプラ差替済み)
m9 = m11v.m9

from outdoor_seld.engine_ev import make_car_ev  # noqa: E402
from outdoor_seld.engine_heavy import heavy_f0_from_seed, make_heavy_delta  # noqa: E402
from outdoor_seld.noise_v12 import diffuse_foa_urban_noise  # noqa: E402

# ---- 出力先を v12 に切替（v11本体は不変更） --------------------------------
DS_NAME_V12 = "outdoor_siren_v12"
m9.DS = ROOT / "out" / f"dataset_{DS_NAME_V12}"
m9.WORK = m9.DS / "work"

# ---- ①背景騒音の差替（呼び出し規約・RNG消費構造は同一） --------------------
m9.diffuse_foa_noise = diffuse_foa_urban_noise

# ---- v12デジタルヘッドルーム規約: 全信号一律 -6dB -------------------------------
# 根拠: 都市背景はdB(A)等価でもピンクより低域の物理振幅が大きく、騒音上限付近の
# クリップでピークが0.99を超えた（job248 shard4: mix3642 peak1.248）。全音源・背景を
# 一律-6dBすることで相対関係（SNR・クラス間レベル・ラベル・マスク）は完全に不変のまま
# クリッピングだけを回避する。記録されるrecv_*_db等の絶対値も-6dBシフトする（規約）。
V12_HEADROOM_DB = 6.0
_orig_gain_for_spl_a = m9.gain_for_spl_a
m9.gain_for_spl_a = (lambda x, fs, db:
                     _orig_gain_for_spl_a(x, fs, db - V12_HEADROOM_DB))

# ---- ②車のバリアント振り分け（audio_seed決定論、車単位） --------------------
EV_FRAC = 0.15
HEAVY_FRAC = 0.30       # EVでない車のうちではなく全車比（0.15<=u<0.45が大型）
TARGET_63_OVER_1K_DB = 3.0

_orig_make_dry = m9._make_dry


def _car_variant(audio_seed: int) -> str:
    u = ((int(audio_seed) * 2654435761 + 97) % (2 ** 32)) / 2.0 ** 32
    if u < EV_FRAC:
        return "ev"
    if u < EV_FRAC + HEAVY_FRAC:
        return "heavy"
    return "normal"


def _band_energy(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(spec[(f >= f_lo) & (f < f_hi)].sum())


def _make_dry_v12(src: dict) -> np.ndarray:
    cls, p = src["class"], src.get("params", {})
    if cls != "car_drive":
        return _orig_make_dry(src)
    seed = p["audio_seed"]
    variant = _car_variant(seed)
    src["car_variant"] = variant            # scene.jsonに来歴を残す
    if variant == "ev":
        rng = np.random.default_rng(seed)
        speed = abs(float(src.get("speed_mps", 6.0)))
        return make_car_ev(m9.CLIP, m9.FS_SIM, rng, speed_mps=speed)
    dry = _orig_make_dry(src)               # normal / heavyの土台は現行車
    if variant == "heavy":
        b63 = _band_energy(dry, m9.FS_SIM, 44.0, 88.0)
        b1k = _band_energy(dry, m9.FS_SIM, 710.0, 1420.0)
        target = b1k * 10.0 ** (TARGET_63_OVER_1K_DB / 10.0)
        d = make_heavy_delta(m9.CLIP, m9.FS_SIM,
                             np.random.default_rng(seed * 31 + 17),
                             f0=heavy_f0_from_seed(seed))
        b63_d = _band_energy(d, m9.FS_SIM, 44.0, 88.0)
        need = max(0.0, target - b63)
        if need > 0 and b63_d > 0:
            dry = dry + float(np.sqrt(need / b63_d)) * d
    return dry


m9._make_dry = _make_dry_v12


# ==================== v12後半: 新クラス（キックボード/バイク/列車） ====================
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import soundfile as sf  # noqa: E402

from outdoor_seld.calibration import a_weighted_rms  # noqa: E402
from outdoor_seld.kickboard import make_kickboard  # noqa: E402
from outdoor_seld.motorcycle import make_motorcycle  # noqa: E402
from outdoor_seld.train import make_train_horn, make_train_passby  # noqa: E402

PLAN_V12 = ROOT / "out" / "dataset_outdoor_siren_v12" / "plan"
CLASS_IDX_V12 = dict(m9.CLASS_IDX, kickboard=6, motorcycle=7)
VEHICLE_GATED = {"car_drive", "kickboard", "motorcycle"}   # 可聴ゲート対象
m9.CREST.setdefault("kickboard", 5.0)
m9.CREST.setdefault("motorcycle", 8.0)


def load_plan_v12ext() -> list:
    rows = []
    with open(PLAN_V12 / "assignment_v12ext.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def _mic_sample(row: dict, rng: np.random.Generator):
    """coreと同一規約のマイク（歩行: 速度U(1.0,1.4)・±x、中心対称の2点軌道）。"""
    if row["motion"] == "walk":
        v = float(rng.uniform(1.0, 1.4))
        d = float(rng.choice([-1.0, 1.0]))
        x0 = -d * v * 5.0
        arr = np.array([[0.0, x0, 0.0, 1.5], [m9.CLIP, x0 + d * v * m9.CLIP, 0.0, 1.5]])
        return arr, {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": d}
    return np.array([0.0, 0.0, 1.5]), {"motion": "static"}


def sample_v12kick(row: dict) -> dict:
    """キックボード単独（特定小型原付: 歩道≤6km/h 40% / 車道≤20km/h 60%）。"""
    rng = np.random.default_rng(row["seed"])
    mic_arr, mic = _mic_sample(row, rng)
    side = float(rng.choice([-1.0, 1.0]))
    lat = float(rng.uniform(0.8, 2.5)) * side
    speed = float(rng.uniform(1.2, 1.7) if rng.random() < 0.4
                  else rng.uniform(1.7, 5.6))
    dirx = float(rng.choice([-1.0, 1.0]))
    t_cpa = float(rng.uniform(3.0, 8.0))
    x0 = -dirx * speed * t_cpa
    law = float(rng.uniform(55.0, 65.0))     # 設計値dB(A)@1m（9月実録で較正予定）
    src = {"class": "kickboard", "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, lat, 1.5], [m9.CLIP, x0 + dirx * speed * m9.CLIP, lat, 1.5]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "law_db": law, "l1m_db": law,
           "params": {"audio_seed": row["seed"] * 7 + 21, "speed_mps": speed}}
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": [src],
            "noise": {"dba": dba, "seed": row["seed"] * 7919 + 13}}


def sample_v12bike(row: dict) -> dict:
    """バイク単独（原付40%: 法定30km/h・法規79dB / 二輪60%: 30〜60km/h・82dBキャップ）。"""
    rng = np.random.default_rng(row["seed"])
    mic_arr, mic = _mic_sample(row, rng)
    side = float(rng.choice([-1.0, 1.0]))
    lat = float(rng.uniform(1.5, 4.0)) * side
    if rng.random() < 0.4:
        ec, speed, law = "moped", float(rng.uniform(5.6, 8.3)), float(rng.uniform(70.0, 79.0))
    else:
        ec, speed, law = "motorcycle", float(rng.uniform(8.3, 16.7)), float(rng.uniform(72.0, 82.0))
    dirx = float(rng.choice([-1.0, 1.0]))
    t_cpa = float(rng.uniform(3.0, 8.0))
    x0 = -dirx * speed * t_cpa
    l1m = law + 20.0 * np.log10(7.5)         # 加速走行騒音規制の7.5m測定→1m換算
    src = {"class": "motorcycle", "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, lat, 1.5], [m9.CLIP, x0 + dirx * speed * m9.CLIP, lat, 1.5]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "law_db": law, "l1m_db": l1m,
           "params": {"audio_seed": row["seed"] * 7 + 23, "engine_class": ec,
                      "speed_mps": speed}}
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": [src],
            "noise": {"dba": dba, "seed": row["seed"] * 7919 + 13}}


def sample_v12train(row: dict) -> dict:
    """第4種踏切の列車通過（複数点音源4〜8両・警笛50%・クラス=crossing意味拡張）。

    ラベルは先頭/中間/最後尾の3両=3トラック（ADPIT上限）。他車両はno_label（音のみ）。
    """
    rng = np.random.default_rng(row["seed"])
    mic_arr, mic = _mic_sample(row, rng)
    side = float(rng.choice([-1.0, 1.0]))
    y = float(rng.uniform(5.0, 10.0)) * side     # 第4種踏切の待機位置相当
    speed = float(rng.uniform(11.0, 25.0))       # 40〜90km/h（ローカル線）
    n_cars = int(rng.integers(4, 9))
    dirx = float(rng.choice([-1.0, 1.0]))
    t_cpa = float(rng.uniform(4.0, 7.0))
    x0 = -dirx * speed * t_cpa
    law = float(rng.uniform(82.0, 87.0))         # 実測82-87dB@12.5m（東京都調査）
    l1m_car = law + 20.0 * np.log10(12.5) - 10.0 * np.log10(n_cars)
    horn = None
    if rng.random() < 0.5:                       # 警笛吹鳴標識シナリオ（50%）
        horn = {"horn_type": "air" if rng.random() < 0.6 else "electric",
                "pattern": "long" if rng.random() < 0.5 else "short3",
                "t_start": float(rng.uniform(1.0, max(1.2, t_cpa - 1.0)))}
    labeled = {0: 0, n_cars // 2: 1, n_cars - 1: 2}
    sources = []
    for i in range(n_cars):
        xi = x0 - dirx * i * 20.0
        sources.append({
            "class": "crossing", "kind": "vehicle",
            "track": labeled.get(i, 0), "no_label": i not in labeled,
            "train_car_index": i,
            "wp": [[0.0, xi, y, 1.5], [m9.CLIP, xi + dirx * speed * m9.CLIP, y, 1.5]],
            "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
            "law_db": law, "l1m_db": l1m_car,
            "params": {"audio_seed": row["seed"] * 7 + 40 + i, "train_car": True,
                       "speed_mps": speed, "horn": horn if i == 0 else None}})
    dba = float(rng.uniform(*m9.NOISE_DBA))
    return {"row": dict(row), "mic": mic, "_mic_arr": mic_arr, "sources": sources,
            "noise": {"dba": dba, "seed": row["seed"] * 7919 + 13}}


_V12_SAMPLERS = {"v12kick": sample_v12kick, "v12bike": sample_v12bike,
                 "v12train": sample_v12train}


def sample_scene_v12(row: dict) -> dict:
    fn = _V12_SAMPLERS.get(row["scenario"])
    return fn(row) if fn else m9.sample_scene_v9(row)


# ---- _make_dry の新クラス対応（既存v12車バリアントの上に積む） ----
_car_variant_make_dry = m9._make_dry


def _make_dry_v12_full(src: dict) -> np.ndarray:
    cls, p = src["class"], src.get("params", {})
    if cls == "kickboard":
        return make_kickboard(m9.CLIP, m9.FS_SIM,
                              np.random.default_rng(p["audio_seed"]),
                              speed_mps=p["speed_mps"])
    if cls == "motorcycle":
        return make_motorcycle(m9.CLIP, m9.FS_SIM,
                               np.random.default_rng(p["audio_seed"]),
                               engine_class=p["engine_class"],
                               speed_mps=p["speed_mps"])
    if cls == "crossing" and p.get("train_car"):
        rng = np.random.default_rng(p["audio_seed"])
        body = make_train_passby(m9.CLIP, m9.FS_SIM, rng,
                                 speed_mps=p["speed_mps"], peak=1.0)
        h = p.get("horn")
        if h:
            if h["pattern"] == "long":
                seg = make_train_horn(float(rng.uniform(1.2, 1.8)), m9.FS_SIM, rng,
                                      horn_type=h["horn_type"])
            else:                                   # 短急数声（危険警告）
                parts = []
                for _ in range(3):
                    parts.append(make_train_horn(float(rng.uniform(0.3, 0.45)),
                                                 m9.FS_SIM, rng,
                                                 horn_type=h["horn_type"]))
                    parts.append(np.zeros(int(0.25 * m9.FS_SIM)))
                seg = np.concatenate(parts)
            g_h = (a_weighted_rms(body, m9.FS_SIM)
                   / max(a_weighted_rms(seg, m9.FS_SIM), 1e-12)) * 10.0 ** (6.0 / 20.0)
            i0 = int(h["t_start"] * m9.FS_SIM)
            j = min(len(seg), len(body) - i0)
            if j > 0:
                body[i0:i0 + j] += g_h * seg[:j]
        return body
    return _car_variant_make_dry(src)


m9._make_dry = _make_dry_v12_full


# ---- generate_clip の v12 複製（変更点: サンプラ/可聴ゲート集合/no_label/8クラス辞書）----
def generate_clip_v12(row: dict) -> None:
    t_start = time.perf_counter()
    s = sample_scene_v12(row)
    mic = s.pop("_mic_arr")
    name = row["clip_id"]
    is_probe = row["scenario"].startswith("probe_")
    wdir = m9.WORK / name
    wdir.mkdir(parents=True, exist_ok=True)
    for sub in ("foa", "metadata", "masks"):
        (m9.DS / sub).mkdir(parents=True, exist_ok=True)
    c = m9.sound_speed(m9.TEMP_C)
    n24 = int(m9.CLIP * m9.FS_OUT)
    tgrid = np.arange(0.0, m9.CLIP, 0.01)

    stems = []
    for src in s["sources"]:
        dry = m9._window(m9._make_dry(src), src["t_on"], src["t_off"])
        a0, a1 = int(src["t_on"] * m9.FS_SIM), int(src["t_off"] * m9.FS_SIM)
        g = m9.gain_for_spl_a(dry[a0:a1], m9.FS_SIM, src["l1m_db"])
        src["dry_gain"] = g
        stem_d, stem_wr = m9._render_stem(dry * g, np.array(src["wp"], float), mic, c)
        if is_probe:
            b0, b1 = int(m9.PROBE_WIN[0] * m9.FS_OUT), int(m9.PROBE_WIN[1] * m9.FS_OUT)
            gp = m9.gain_for_spl_a(stem_wr[0, b0:b1], m9.FS_OUT, m9.PROBE_RECV_DBA)
            stem_d, stem_wr = stem_d * gp, stem_wr * gp
            src["probe_gain"] = gp
        dist = m9._dist_series(src["wp"], mic, tgrid)
        src["min_dist_m"] = round(float(np.min(dist)), 3)
        src["t_min_dist_s"] = round(float(tgrid[int(np.argmin(dist))]), 3)
        if src["class"] == "car_drive" and src.get("track", 0) == 0:
            s["cpa_rel_dist_m"] = src["min_dist_m"]
            s["cpa_rel_time_s"] = src["t_min_dist_s"]
        win = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
        p_geo = float(np.mean(1.0 / np.maximum(dist[win], 1e-9) ** 2))
        src["recv_pred_db"] = round(src["l1m_db"] + 10.0 * np.log10(p_geo), 2)
        b0, b1 = int(src["t_on"] * m9.FS_OUT), max(int(src["t_off"] * m9.FS_OUT), 1)
        src["recv_meas_db"] = round(m9.spl_a(stem_d[0, b0:b1], m9.FS_OUT), 2)
        src["recv_withrefl_db"] = round(m9.spl_a(stem_wr[0, b0:b1], m9.FS_OUT), 2)
        stems.append((src, stem_d, stem_wr))

    rng_n = np.random.default_rng(s["noise"]["seed"])
    noise = m9.diffuse_foa_noise(n24, m9.FS_OUT, rng_n)
    g_n = m9.gain_for_spl_a(noise[0], m9.FS_OUT, s["noise"]["dba"])
    noise = noise * g_n
    s["noise"]["gain"] = float(g_n)

    mix = noise.copy()
    for _, _, stem_wr in stems:
        mix = mix + stem_wr
    peak = float(np.max(np.abs(mix)))
    assert peak < m9.PEAK_MAX, f"{name}: peak {peak:.3f} >= {m9.PEAK_MAX}"

    fr_noise = m9.frame_spl_a(noise[0], m9.FS_OUT)
    label_rows, mask_rows = [], []
    for src, _, stem_wr in stems:
        cls_idx = CLASS_IDX_V12[src["class"]]
        snr = m9.frame_spl_a(stem_wr[0], m9.FS_OUT) - fr_noise
        mask_rows += [(k, cls_idx, round(float(snr[k]), 2)) for k in range(len(snr))]
        if src.get("no_label"):
            src["audible_frac"] = round(float(np.mean(snr >= m9.AUDIBLE_SNR_DB)), 3)
            src["n_label_frames"] = 0
            continue
        rows, _ = m9.frame_label_rows(np.array(src["wp"], float), mic,
                                      clip_len_sec=m9.CLIP, class_idx=cls_idx,
                                      track_idx=src.get("track", 0),
                                      source_active_from=src["t_on"],
                                      source_active_until=src["t_off"], c=c)
        if src["class"] in VEHICLE_GATED:
            audible = m9._fill_gaps(snr >= m9.AUDIBLE_SNR_DB, m9.GAP_FILL)
            rows = [r for r in rows if audible[r[0]]]
            src["audible_frac"] = round(float(np.mean(snr >= m9.AUDIBLE_SNR_DB)), 3)
            aud_idx = np.where(snr >= m9.AUDIBLE_SNR_DB)[0]
            src["t_first_audible_s"] = (round(float(aud_idx[0]) * 0.1, 2)
                                        if len(aud_idx) else None)
        else:
            k0, k1 = int(src["t_on"] * 10), max(int(src["t_off"] * 10), 1)
            src["audible_frac"] = round(float(np.mean(
                snr[k0:k1] >= m9.AUDIBLE_SNR_DB)), 3)
        src["n_label_frames"] = len(rows)
        label_rows += rows
    label_rows.sort(key=lambda r: (r[0], r[1]))

    for i, (src, stem_d, stem_wr) in enumerate(stems):
        tag = f"src{i}_{src['class']}"
        sf.write(wdir / f"{tag}_direct_24k.flac",
                 stem_d.T.astype(np.float32), m9.FS_OUT, subtype="PCM_24")
        sf.write(wdir / f"{tag}_withrefl_24k.flac",
                 stem_wr.T.astype(np.float32), m9.FS_OUT, subtype="PCM_24")
    sf.write(m9.DS / "foa" / f"{name}.flac", mix.T.astype(np.float32),
             m9.FS_OUT, subtype="PCM_24")
    m9.write_dcase_csv(m9.DS / "metadata" / f"{name}.csv", label_rows)
    with open(m9.DS / "masks" / f"{name}.csv", "w", newline="\n") as f:
        f.write("frame,class,snr_a_db\n")
        for k, ci, v in mask_rows:
            f.write(f"{k},{ci},{v}\n")

    s["stats"] = {"peak": round(peak, 4),
                  "mix_spl_a": round(m9.spl_a(mix[0], m9.FS_OUT), 2),
                  "n_label_rows": len(label_rows),
                  "gen_seconds": round(time.perf_counter() - t_start, 1)}
    (wdir / "scene.json").write_text(json.dumps(s, indent=2, default=str))
    srcs = "+".join(x["class"] for x in s["sources"]) or "none"
    print(f"[gen12] {name} {row['motion']} {srcs} noise={s['noise']['dba']:.1f}dBA "
          f"peak={peak:.3f} ({s['stats']['gen_seconds']:.0f}s)", flush=True)


def verify_copy_equiv(clip_id: str = "fold2_room1_mix0002") -> None:
    """複製ドリフト検査: coreクリップをv12複製とm9原本の両方で生成→全ファイルsha一致。"""
    import hashlib
    rows = [r for r in m9.load_plan("core") if r["clip_id"] == clip_id]
    assert len(rows) == 1
    targets = [m9.DS / "foa" / f"{clip_id}.flac", m9.DS / "metadata" / f"{clip_id}.csv",
               m9.DS / "masks" / f"{clip_id}.csv"]
    generate_clip_v12(rows[0])
    h1 = [hashlib.sha256(p.read_bytes()).hexdigest() for p in targets]
    m9.generate_clip(rows[0])
    h2 = [hashlib.sha256(p.read_bytes()).hexdigest() for p in targets]
    assert h1 == h2, f"複製ドリフト検出: {clip_id}"
    print(f"copy-equiv PASS: {clip_id} 全ファイルsha一致（複製にドリフトなし）")


def smoke_ext() -> None:
    rows = load_plan_v12ext()
    picks = {}
    for r in rows:
        picks.setdefault(r["scenario"], r)
    for scen, r in picks.items():
        generate_clip_v12(r)
        s = json.loads((m9.WORK / r["clip_id"] / "scene.json").read_text())
        lab = [(x["class"], x.get("track"), x.get("n_label_frames"), x.get("no_label", False))
               for x in s["sources"]]
        print(f"  [{scen}] {r['clip_id']} sources={lab}")


def smoke(n_clips: int = 3) -> None:
    rows = m9.load_plan("core")
    # 車入りを優先して拾う（EV/大型/通常が最低1つずつ出るまで最大30行走査）
    seen = set()
    done = 0
    for row in rows:
        if done >= n_clips and {"ev", "heavy", "normal"} <= seen:
            break
        m9.generate_clip(row)
        import json
        s = json.loads((m9.WORK / row["clip_id"] / "scene.json").read_text())
        cars = [x for x in s["sources"] if x["class"] == "car_drive"]
        for c in cars:
            seen.add(c.get("car_variant", "?"))
        print(f"[smoke] {row['clip_id']} cars={[(c.get('car_variant'), round(c.get('recv_meas_db', 0), 1)) for c in cars]}")
        done += 1
    print(f"variants seen: {seen}")


if __name__ == "__main__":
    if "smoke" in sys.argv:
        smoke()
    elif "smoke_ext" in sys.argv:
        smoke_ext()
    elif "verify" in sys.argv:
        verify_copy_equiv()
    else:
        print(__doc__)
