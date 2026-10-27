# -*- coding: utf-8 -*-
"""音の大きさの合わせ（v24・規則 ②）: 本番の描画器と同じ部品で、車（とバイク）を側方 2 m・マイク 1.2 m に置いて描き、受音点の LAFmax と
1 m の設計値 l1m の差 Δ(v) を速さごとに求める（2026-10-12・`level_v24` の曲線に足して l1m にする）。20 log r の手計算は使わない。

置き方 = 交通安全環境研究所の測定と R138 の形: 走る線はマイクの横 2.0 m・マイク高さ 1.2 m・音源高さ 0.325 m（#70 r7／R108-07・2026-10-19: 車の高さ 0.05〜0.6 の真ん中。旧 1.05 m = 旧 SRC_Z 0.8〜1.3 の真ん中・旧の表は md/design/level_match_v24_srcz1p05_2026-10-12.json）・直進・
最接近をクリップの 5 s に。描画 = ドライ音 → gain_for_spl_a で l1m（1 m の A 特性 RMS）に → fastsim.render_mono（1/r・大気吸収・ドップラー）直接音＋地面の鏡像
→ 24 kHz に間引く（step11_v9_render._render_stem と同じ）。受音点の LAFmax = A 特性＋F 125 ms の最大。
Δ は音色（ドライ音）と速さ（F の時定数・ドップラー）で少し変わるので、車（normal）と バイク（moped・motorcycle）で別々に、速さの点ごとに出す。

使い方: python scripts/level_match_v24.py [--out md/design/level_match_v24.json] [--probe-l1m 70]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.motorcycle import make_motorcycle  # noqa: E402
from outdoor_seld.calibration import gain_for_spl_a  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.scene import decimate_to_out_rate  # noqa: E402
from outdoor_seld import ablate  # noqa: E402
from outdoor_seld import level_v24 as L24  # noqa: E402
from outdoor_seld.calibration import spl_a  # noqa: E402
from outdoor_seld import siren_v26 as S26  # noqa: E402
from outdoor_seld.siren import make_fire_siren, make_peepo_siren, make_siren  # noqa: E402

FS_SIM, FS_OUT, CLIP = 48000, 24000, 10.0
GROUND_Z, TEMP_C, PRESS_ATM, RH = 0.0, 20.0, 1.0, 50.0
G = L24.MATCH_GEOMETRY
SPEEDS_CAR = (5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 54.0)
SPEEDS_MOPED = (15.0, 20.0, 25.0, 30.0)
SPEEDS_MOTO = (25.0, 30.0, 40.0, 50.0, 60.0)


def received_lafmax(dry_cal: np.ndarray, v_mps: float) -> float:
    t_cpa = G["cpa_s"]
    wp = np.array([[t, (t - t_cpa) * v_mps, G["lateral_m"], G["src_z_m"]] for t in (0.0, CLIP)])
    mic = np.array([0.0, 0.0, G["mic_z_m"]])
    total = None
    for tag in (("direct", "mirror") if ablate.ground_enabled() else ("direct",)):
        w = wp.copy()
        if tag == "mirror":
            w[:, 3] = 2.0 * GROUND_Z - w[:, 3]
        mono48 = render_mono(dry_cal, w, mic, FS_SIM, CLIP, temperature_c=TEMP_C, pressure_atm=PRESS_ATM, rel_humidity=RH, **ablate.render_flags())
        mono24 = decimate_to_out_rate(mono48, FS_SIM, FS_OUT)
        total = mono24 if total is None else total + mono24
    return L24.lafmax_db(total, FS_OUT)


def siren_dry(p: dict, duration: float) -> np.ndarray:
    """サイレンのドライ音（v26 の params なら v26 の生成器・無ければ旧）。step11_v9 の _make_dry と v26 の振り分けと同じ呼び方。"""
    st = p["siren_type"]
    if p.get("v26"):
        kw = {k: v for k, v in p.items() if k not in ("siren_type", "v26", "v26_old", "audio_seed")}
        if st == "fire":
            return S26.make_fire_siren_v26(duration, FS_SIM, rng=np.random.default_rng(int(p.get("audio_seed", 0))), **kw)
        if st == "peepo":
            return S26.make_peepo_siren_v26(duration, FS_SIM, **kw)
        return S26.make_patrol_siren_v26(duration, FS_SIM, **kw)
    kw = {k: v for k, v in p.items() if k not in ("siren_type", "audio_seed")}
    if st == "fire":
        return make_fire_siren(duration, FS_SIM, rng=np.random.default_rng(int(p["audio_seed"])), **kw)
    if st == "peepo":
        return make_peepo_siren(duration, FS_SIM, **kw)
    return make_siren(duration, FS_SIM, **kw)


def received_siren_front(dry_cal: np.ndarray, duration: float) -> float:
    """前方 dist_m・静止・高さ src_z_m のサイレンを、マイク (0, 0, mic_z_m) で受けた dB(A)（直接音＋鏡像・時間平均）。"""
    SG = L24.SIREN_MATCH_GEOMETRY
    wp = np.array([[0.0, -SG["dist_m"], 0.0, SG["src_z_m"]], [duration, -SG["dist_m"], 0.0, SG["src_z_m"]]])
    mic = np.array([0.0, 0.0, SG["mic_z_m"]])
    total = None
    for tag in (("direct", "mirror") if ablate.ground_enabled() else ("direct",)):
        w = wp.copy()
        if tag == "mirror":
            w[:, 3] = 2.0 * GROUND_Z - w[:, 3]
        mono48 = render_mono(dry_cal, w, mic, FS_SIM, duration, temperature_c=TEMP_C, pressure_atm=PRESS_ATM, rel_humidity=RH, **ablate.render_flags())
        mono24 = decimate_to_out_rate(mono48, FS_SIM, FS_OUT)
        total = mono24 if total is None else total + mono24
    return spl_a(total, FS_OUT)


def siren_tables(n_per_type: int, probe_lv_20m: float = 100.0) -> dict:
    """Δ_siren(型) = 描画の前方 20 m の値 − 宣言値。本番の道で測る: 音源 dict を組んで m9._make_dry（v26 の生成器 → v24 の向きの差と l1m の直し）
    → v13 と同じ RMS 合わせ → 直接音＋鏡像で描画。v26 の params と旧の params（v26_old）の両方で、本番と同じ抽選（step11_v9 の _warn_params に v26 の hook）から型ごとに n_per_type 本。
    （17:4x: 向きの差なしの素のドライ音で測ると、マイク 1.2 m からスピーカー 2.4 m を見る 3.4° の仰角の分 −0.46 dB が本番に残った）"""
    import step11_v26_render  # noqa: F401  (m9._warn_params と生成器に v26 の hook が乗る)
    import step11_v9_render as m9
    SG = L24.SIREN_MATCH_GEOMETRY; dur = float(SG["clip_s"])
    assert abs(dur - m9.CLIP) < 1e-9, "サイレンの合わせの窓は本番の本の長さ（m9.CLIP）と同じにする"
    l1m0 = probe_lv_20m + 20.0 * np.log10(SG["dist_m"])
    mic = [0.0, 0.0, SG["mic_z_m"]]
    out = {"v26": {}, "old": {}}
    rng = np.random.default_rng(20261012); seed = 0
    need = ("peepo", "fire", "wail")
    while any(len(out["v26"].get(t, [])) < n_per_type for t in need):
        seed += 1
        p = m9._warn_params("siren", rng, 9000 + seed)
        st = p["siren_type"]
        if len(out["v26"].get(st, [])) >= n_per_type:
            continue
        old = dict(p["v26_old"])
        if "audio_seed" in p:
            old["audio_seed"] = p["audio_seed"]
        for chain, pp in (("v26", p), ("old", old)):
            src = {"class": "siren", "kind": "vehicle", "track": 0, "params": dict(pp), "t_on": 0.0, "t_off": dur, "speed_mps": 0.0, "dir_x": 1.0,
                   "wp": [[0.0, -SG["dist_m"], 0.0, SG["src_z_m"]], [dur, -SG["dist_m"], 0.0, SG["src_z_m"]]],
                   "law_db": probe_lv_20m, "l1m_db": l1m0, "v24": {"rule": "siren", "directivity_db": dict(L24.SIREN_DIR_DB)}, "_mic_arr_v24": mic}
            dry = m9._make_dry(src)                                                # 本番の連なり（向きの差で l1m_db が直る）
            got = received_siren_front(dry * gain_for_spl_a(dry, FS_SIM, float(src["l1m_db"])), dur)
            out[chain].setdefault(st, []).append(round(got - probe_lv_20m, 3))
            print(f"siren {chain:4s} {st:6s} seed {9000 + seed}: 前 20 m {got:6.2f} dB(A) → Δ = {got - probe_lv_20m:5.2f} dB（向きの差の平均 {src['v24'].get('dir_mean_db')} dB）")
    tables = {chain: {st: {"delta_db": round(float(np.mean(v)), 3), "min_db": min(v), "max_db": max(v), "n": len(v), "each_db": v} for st, v in d.items()} for chain, d in out.items()}
    return {"geometry": SG, "probe_lv_20m_db": probe_lv_20m, "n_per_type": n_per_type,
            "note": "delta_db = 本番の道（v26 の生成器 → v24 の向きの差 → v13 の RMS 合わせ → 直接音＋鏡像の描画）で測った前方 20 m の値（時間平均 dB(A)・10 s）− 宣言値。"
                    "本番は l1m = lv + 20 log10(20) − delta_db（step11_v24 apply_levels）。型は peepo／fire／wail（パトカー）。v26 はサイレンの形 v26（#62）、old は v25 までの形。"
                    "仰角 3.4° の向きの差（−0.46 dB）込み",
            "tables": tables}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "md/design/level_match_v24.json"))
    ap.add_argument("--probe-l1m", type=float, default=70.0)
    ap.add_argument("--siren-n", type=int, default=6, help="サイレンの型ごとの本数（0 で表を作らない）")
    a = ap.parse_args()
    rng = np.random.default_rng(20261012)
    out = {"version": "level_match_v24/4 (2026-10-13・サイレンの合わせのマイク高さを出典の 1.5 m に・Astra SV26-06)", "geometry": G, "probe_l1m_db": a.probe_l1m, "ground_reflection": bool(ablate.ground_enabled()),
           "render_flags": ablate.render_flags(), "fs_sim": FS_SIM, "fs_out": FS_OUT, "clip_s": CLIP, "tables": {}}
    t0 = time.perf_counter()
    for kind, speeds, maker in (("car", SPEEDS_CAR, lambda v: make_car_v9(CLIP, FS_SIM, np.random.default_rng(1), f0=42.0)),
                                ("moped", SPEEDS_MOPED, lambda v: make_motorcycle(CLIP, FS_SIM, np.random.default_rng(2), engine_class="moped", speed_mps=v / 3.6)),
                                ("motorcycle", SPEEDS_MOTO, lambda v: make_motorcycle(CLIP, FS_SIM, np.random.default_rng(3), engine_class="motorcycle", speed_mps=v / 3.6))):
        deltas = []
        for v in speeds:
            dry = maker(v)
            dry_cal = dry * gain_for_spl_a(dry, FS_SIM, a.probe_l1m)
            lmax = received_lafmax(dry_cal, v / 3.6)
            deltas.append(round(a.probe_l1m - lmax, 3))
            print(f"{kind:10s} {v:5.1f} km/h: 受音点 LAFmax {lmax:6.2f} dB(A) → Δ = l1m − LAFmax(2 m) = {a.probe_l1m - lmax:5.2f} dB")
        out["tables"][kind] = {"speeds_kmh": list(speeds), "delta_db": deltas}
    # 参考: 曲線を足した l1m（車）
    out["example_car_l1m_db"] = {str(v): round(L24.l2m_curve(v) + float(np.interp(v, SPEEDS_CAR, out["tables"]["car"]["delta_db"])), 2) for v in (6.5, 10.0, 15.0, 20.0, 30.0, 54.0)}
    if a.siren_n > 0:
        out["siren"] = siren_tables(a.siren_n, probe_lv_20m=100.0)
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out}  例（車の l1m・曲線 + Δ）: {out['example_car_l1m_db']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
