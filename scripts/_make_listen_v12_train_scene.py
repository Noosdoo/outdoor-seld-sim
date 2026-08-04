# -*- coding: utf-8 -*-
"""列車のシーン完成形試聴: 6両編成×複数点音源を本番と同じ物理でレンダし、
参照録音（電車通過）と同じ土俵で比較できる形にする。

- 幾何: 第4種踏切の待機位置（線路から8m）。72km/h、±130m通過（約13秒）
- 各車両=独立点音源（20m間隔、v12設計書の列車規約）＋先頭車が接近時に警笛（長緩一声）
- 背景: v12都市暗騒音 50dB(A)
- 出力: 10_列車通過_シーン完成形.wav / 99_参照_電車通過_実録.wav（比較用コピー、
  git管理外=再配布防止。データセットにはどちらも不使用）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.calibration import gain_for_spl_a  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.noise_v12 import urban_colored_noise  # noqa: E402
from outdoor_seld.train import (CAR_LEN_M, make_train_horn,  # noqa: E402
                                make_train_passby)

FS = 48000
OUT = ROOT / "out" / "listen_v12_sources"
MIC = np.array([0.0, 0.0, 1.5])
SP_REF = Path(r"C:\Users\satos\AppData\Local\Temp\claude\c--Users-satos-research"
              r"\f06c9494-5feb-49cd-8986-40f95882d5d7\scratchpad\se_ref\train-pass2.mp3")


def main() -> None:
    dur = 14.0
    speed = 20.0                     # 72km/h
    lateral = 8.0                    # 第4種踏切の待機位置相当
    n_cars = 6
    rng = np.random.default_rng(31)

    # 編成85dB(A)@12.5m（東京都実測の中央付近）→ 1両あたり等配分
    l1m_train = 85.0 + 20.0 * np.log10(12.5)
    l1m_car = l1m_train - 10.0 * np.log10(n_cars)

    n = int(dur * FS)
    mix = np.zeros(n)
    for i in range(n_cars):
        car_rng = np.random.default_rng(1000 + i)
        dry = make_train_passby(dur, FS, car_rng, speed_mps=speed, peak=1.0)
        if i == 0:                   # 先頭車: 接近中(t=2.5s)に長緩一声
            horn = make_train_horn(1.6, FS, np.random.default_rng(77),
                                   horn_type="air")
            g_h = 10.0 ** (6.0 / 20.0)   # 警笛は走行音より+6dB(設計値・省令追補待ち)
            s = int(2.5 * FS)
            dry[s:s + len(horn)] += g_h * horn * np.max(np.abs(dry))
        g = gain_for_spl_a(dry, FS, l1m_car)
        x0 = -130.0 - i * CAR_LEN_M      # 後続車は20mずつ後ろ
        wp = np.array([[0.0, x0, lateral, 1.5],
                       [dur, x0 + speed * dur, lateral, 1.5]])
        mono = render_mono(dry * g, wp, MIC, FS, dur,
                           temperature_c=20.0, pressure_atm=1.0, rel_humidity=50.0)
        mix += mono
        print(f"car {i} rendered", flush=True)

    bg = urban_colored_noise(n, FS, rng)
    mix += bg * gain_for_spl_a(bg, FS, 50.0)

    x = mix / max(float(np.max(np.abs(mix))), 1e-9) * 0.85
    sf.write(OUT / "10_列車通過_シーン完成形.wav", x.astype(np.float32), FS,
             subtype="PCM_16")
    print("wrote 10_列車通過_シーン完成形.wav")

    ref, fs_r = sf.read(SP_REF)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    ref = ref / max(float(np.max(np.abs(ref))), 1e-9) * 0.85
    sf.write(OUT / "99_参照_電車通過_実録.wav", ref.astype(np.float32), fs_r,
             subtype="PCM_16")
    print("wrote 99_参照_電車通過_実録.wav (git管理外)")


if __name__ == "__main__":
    main()
