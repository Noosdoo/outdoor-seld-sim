# -*- coding: utf-8 -*-
"""規則 ②（PROGRESS 2026-10-07 10:40〜11:20「測った値の使い方」）: 合成側で今日の置き方を描画して、受音点の L_AFmax が実測と同じになる
l1m_db（1 m の設計値）を速さごとに求める。20 log r の手計算は使わない。

置き方 = 2026-10-07 の測定: 走行の線はマイクの横 1.1 m（本人申告 1.0〜1.2 m）・マイク高さ 1.5 m・音源（タイヤ）高さ 0.3 m 仮定・直進・CPA をクリップの 5 s に。
描画 = 本番の描画器と同じ部品: ドライ音源 → gain_for_spl_a で l1m_db（1 m の A 特性レベル）に合わせる → fastsim.render_mono（1/r・大気吸収・ドップラー）を
直接音と地面の鏡像（GROUND_Z = 0）で描画して足す → 24 kHz に間引く（step11_v9_render._render_stem と同じ手順・ablate の既定 = 地面反射あり）。
受音点の L_AFmax = A 特性 ＋ FAST 125 ms の最大（bicycle_measure_analyze と同じ関数）。
実測の目標 = 速さ 3 段の中央値（暗騒音を引いた値: 遅い 47.4・普通 48.7・速い 49.1 dB(A)。引く前は 48.0／49.2／49.5）。描画には暗騒音が無いので引いた値に合わせる。

使い方: python scripts/bicycle_l1m_match.py [--lateral 1.1] [--mic-z 1.5] [--src-z 0.3] [--probe-l1m 50]
出力: 速さごとの l1m_db と、横 1.0／1.2 m のときの幅。raw_bicycle_measure/analysis/l1m_match.txt にも書く。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from outdoor_seld.bicycle_v2 import make_bicycle_v2  # noqa: E402
from outdoor_seld.calibration import gain_for_spl_a  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.geometry import sound_speed  # noqa: E402
from outdoor_seld.scene import decimate_to_out_rate  # noqa: E402
from outdoor_seld import ablate  # noqa: E402
from bicycle_measure_analyze import a_weighted_signal, fast_level  # noqa: E402

FS_SIM, FS_OUT, CLIP = 48000, 24000, 10.0
GROUND_Z, TEMP_C, PRESS_ATM, RH = 0.0, 20.0, 1.0, 50.0
TARGETS = {"遅い 4.7 km/h": (4.7, 47.4, 48.0), "普通 8.3 km/h": (8.3, 48.7, 49.2), "速い 11.8 km/h": (11.8, 49.1, 49.5)}


def received_lafmax(dry_cal: np.ndarray, v_mps: float, lateral: float, mic_z: float, src_z: float) -> float:
    t_cpa = CLIP / 2
    ts = np.array([0.0, CLIP])
    wp = np.array([[t, (t - t_cpa) * v_mps, lateral, src_z] for t in ts])     # 直進・CPA を 5 s に
    mic = np.array([0.0, 0.0, mic_z])
    total = None
    for tag in (("direct", "mirror") if ablate.ground_enabled() else ("direct",)):
        w = wp.copy()
        if tag == "mirror":
            w[:, 3] = 2.0 * GROUND_Z - w[:, 3]
        mono48 = render_mono(dry_cal, w, mic, FS_SIM, CLIP, temperature_c=TEMP_C, pressure_atm=PRESS_ATM,
                             rel_humidity=RH, **ablate.render_flags())
        mono24 = decimate_to_out_rate(mono48, FS_SIM, FS_OUT)
        total = mono24 if total is None else total + mono24
    return float(fast_level(a_weighted_signal(total, FS_OUT), FS_OUT).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lateral", type=float, default=1.1); ap.add_argument("--mic-z", type=float, default=1.5); ap.add_argument("--src-z", type=float, default=0.3)
    ap.add_argument("--probe-l1m", type=float, default=50.0)
    a = ap.parse_args()
    rng = np.random.default_rng(20261007)
    dry = make_bicycle_v2(CLIP, FS_SIM, rng)
    g = gain_for_spl_a(dry, FS_SIM, a.probe_l1m)
    dry_cal = dry * g
    lines = [f"描画で合わせた l1m_db（横 {a.lateral} m・マイク {a.mic_z} m・音源 {a.src_z} m・地面反射 {'あり' if ablate.ground_enabled() else 'なし'}・探り値 l1m = {a.probe_l1m} dB(A)）"]
    results = {}
    for name, (kmh, tgt_corr, tgt_raw) in TARGETS.items():
        rec = received_lafmax(dry_cal, kmh / 3.6, a.lateral, a.mic_z, a.src_z)
        l1m_corr = a.probe_l1m + (tgt_corr - rec); l1m_raw = a.probe_l1m + (tgt_raw - rec)
        results[name] = (rec, l1m_corr, l1m_raw)
        lines.append(f"  {name}: 探り値での受音点 L_AFmax {rec:.1f} → 実測（暗騒音を引いた {tgt_corr}）に合う l1m_db = **{l1m_corr:.1f}** dB(A)@1 m（引く前の {tgt_raw} なら {l1m_raw:.1f}）")
    # 横の幅（1.0 と 1.2 m）
    for lat in (1.0, 1.2):
        rec = received_lafmax(dry_cal, 8.3 / 3.6, lat, a.mic_z, a.src_z)
        lines.append(f"  横 {lat} m のとき（普通 8.3 km/h・探り値）: 受音点 {rec:.1f} → l1m_db = {a.probe_l1m + (48.7 - rec):.1f}")
    vals = [r[1] for r in results.values()]
    lines.append(f"まとめ: l1m_db は {min(vals):.1f}〜{max(vals):.1f} dB(A)@1 m（速さ 4.7〜11.8 km/h・ほぼ平ら）。設計の範囲 45〜55 に{'入る' if 45 <= min(vals) and max(vals) <= 55 else '入らない'}。")
    txt = "\n".join(lines); print(txt)
    out = ROOT / "raw_bicycle_measure/analysis/l1m_match.txt"; out.write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
