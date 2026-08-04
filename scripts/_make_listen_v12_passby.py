# -*- coding: utf-8 -*-
"""v12新音源の「物理込み」通過プレビュー（本番と同じfastsim経路: 1/r・ドップラー・大気吸収）。

ドライ素材では「近づく感じ」は判定できない（本人指摘 2026-08-05）ため、
実走行の幾何で通過をレンダして試聴に出す。出力= out/listen_v12_sources/ に追加。
  06b: 列車 72km/h、第4種踏切の待機位置（線路から8m）を通過。150m手前から
  07b: キックボード 14km/h、歩道ですれ違い（横1.2m）。40m手前から
  04b: 静音EV 22km/h、路地で背後から接近・追い越し（横2m）。60m手前から
レベルは実測準拠: 列車=85dB(A)@12.5m（東京都実測の中央付近）、EV/キックは設計値。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.calibration import gain_for_spl_a  # noqa: E402
from outdoor_seld.engine_ev import make_car_ev  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.kickboard import make_kickboard  # noqa: E402
from outdoor_seld.train import make_train_passby  # noqa: E402

FS = 48000
OUT = ROOT / "out" / "listen_v12_sources"
MIC = np.array([0.0, 0.0, 1.5])


def l1m_from(law_db: float, ref_dist_m: float) -> float:
    """基準距離の規定dB(A)→1m相当（1/r幾何減衰、他クラスの規約と同じ）。"""
    return law_db + 20.0 * np.log10(ref_dist_m)


def render_pass(dry: np.ndarray, l1m_db: float, speed: float, lateral: float,
                x0: float, dur: float, name: str) -> None:
    g = gain_for_spl_a(dry, FS, l1m_db)
    wp = np.array([[0.0, x0, lateral, 1.5],
                   [dur, x0 + speed * dur, lateral, 1.5]])
    mono = render_mono(dry * g, wp, MIC, FS, dur,
                       temperature_c=20.0, pressure_atm=1.0, rel_humidity=50.0)
    x = mono / max(float(np.max(np.abs(mono))), 1e-9) * 0.8
    sf.write(OUT / name, x.astype(np.float32), FS, subtype="PCM_16")
    print("wrote", name)


def main() -> None:
    dur = 14.0
    # 列車: 72km/h(20m/s)。踏切待機=線路から8m。150m手前→+130m通過
    train = make_train_passby(dur, FS, np.random.default_rng(24), speed_mps=20.0)
    render_pass(train, l1m_from(85.0, 12.5), 20.0, 8.0, -150.0, dur,
                "06b_列車_通過_物理込み.wav")
    # キックボード: 14km/h(4m/s)。歩道すれ違い横1.2m。40m手前→+16m
    kick = make_kickboard(dur, FS, np.random.default_rng(25), speed_mps=4.0)
    render_pass(kick, 60.0, 4.0, 1.2, -40.0, dur,
                "07b_キックボード_接近_物理込み.wav")
    # 静音EV: 22km/h(6m/s)。背後から接近し追い越し 横2m。60m手前→+24m
    ev = make_car_ev(dur, FS, np.random.default_rng(26), speed_mps=6.0)
    render_pass(ev, l1m_from(62.0, 10.0), 6.0, 2.0, -60.0, dur,
                "04b_静音EV_接近_物理込み.wav")


if __name__ == "__main__":
    main()
