# -*- coding: utf-8 -*-
"""v12新音源の試聴セット生成（本人の耳ゲート用）。出力= out/listen_v12_sources/

ペア構成（各8秒・A特性レベルを揃えて公平比較・ダブルクリック再生可）:
  01/02: 背景騒音 v11ピンク vs v12都市（63Hz峰）
  03/04: 車 v11現行（42Hzエンジン+タイヤ） vs v12静音EV（タイヤ+PWM9k+モータ次数）
  05:    v12大型車（現行車+63Hz帯強化デルタ、E1a較正と同じ+3dB規定）
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

from outdoor_seld.calibration import a_weighted_rms  # noqa: E402
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.engine_ev import make_car_ev  # noqa: E402
from outdoor_seld.engine_heavy import make_heavy_delta  # noqa: E402
from outdoor_seld.kickboard import make_kickboard  # noqa: E402
from outdoor_seld.noise import colored_noise  # noqa: E402
from outdoor_seld.noise_v12 import urban_colored_noise  # noqa: E402
from outdoor_seld.train import make_train_composite, make_train_horn  # noqa: E402

FS = 48000
DUR = 8.0
OUT = ROOT / "out" / "listen_v12_sources"
OUT.mkdir(parents=True, exist_ok=True)


def band_energy(x, fs, f_lo, f_hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(spec[(f >= f_lo) & (f < f_hi)].sum())


def main() -> None:
    n = int(DUR * FS)
    rng = np.random.default_rng(20260805)

    pink = colored_noise(n, FS, rng)
    urban = urban_colored_noise(n, FS, rng)
    car = make_car_v9(DUR, FS, np.random.default_rng(11), f0=42.0)
    ev = make_car_ev(DUR, FS, np.random.default_rng(12), speed_mps=6.0)

    # 大型車 = 現行車 + 63Hz帯デルタ（E1aと同じ「63Hz帯=1kHz帯+3dB」規定）
    b63 = band_energy(car, FS, 44.0, 88.0)
    b1k = band_energy(car, FS, 710.0, 1420.0)
    d = make_heavy_delta(DUR, FS, np.random.default_rng(13), f0=60.0)
    g = np.sqrt(max(0.0, b1k * 10 ** 0.3 - b63) / band_energy(d, FS, 44.0, 88.0))
    heavy = car + g * d

    train = make_train_composite(12.0, FS, np.random.default_rng(14),
                                 speed_mps=20.0, n_cars=6)[: int(DUR * FS)]
    kick = make_kickboard(DUR, FS, np.random.default_rng(15), speed_mps=4.0)

    # 警笛: 長緩一声(1.6s) と 短急数声(0.4s×3) を1本のデモにまとめる
    def horn_demo(horn_type, seed):
        r = np.random.default_rng(seed)
        one = make_train_horn(1.6, FS, r, horn_type=horn_type)
        gap = np.zeros(int(0.8 * FS))
        shorts = []
        for _ in range(3):
            shorts += [make_train_horn(0.4, FS, r, horn_type=horn_type),
                       np.zeros(int(0.25 * FS))]
        return np.concatenate([one, gap] + shorts)

    files = [("01_背景_v11ピンク.wav", pink), ("02_背景_v12都市_63Hz峰.wav", urban),
             ("03_車_v11現行.wav", car), ("04_車_v12静音EV.wav", ev),
             ("05_車_v12大型_低音強化.wav", heavy),
             ("06_列車_v12_72kmh.wav", train), ("07_キックボード_v12_14kmh.wav", kick),
             ("08_列車警笛_空気笛.wav", horn_demo("air", 16)),
             ("09_列車警笛_電子ホーン.wav", horn_demo("electric", 17))]
    # A特性レベルを全ファイルで統一 → 共通ゲインでピーク安全化（相対関係は保持）
    target = a_weighted_rms(car, FS)
    scaled = [(nm, x * (target / a_weighted_rms(x, FS))) for nm, x in files]
    gpk = 0.85 / max(float(np.max(np.abs(x))) for _, x in scaled)
    for nm, x in scaled:
        sf.write(OUT / nm, (x * gpk).astype(np.float32), FS, subtype="PCM_16")
        print("wrote", nm)

    (OUT / "README.txt").write_text(
        "v12新音源の試聴セット（A特性ラウドネスを揃えた公平比較）\n\n"
        "・01vs02: 背景騒音。02は実測文献準拠の都市スペクトル（63Hz峰）。\n"
        "  低音の「ゴー」という腹に来る成分が違い。ヘッドホン推奨\n"
        "  （ノートPCスピーカでは60Hzはほぼ再生されません）\n"
        "・03vs04: 車。04は静音EV（エンジン音なし・かすかな高音の笛=PWM 9kHz・\n"
        "  速度連動のモータ音1.2kHz）。「ヒーン」という電車っぽい音色が狙い\n"
        "・05: 大型車。03に63Hz帯の重低音を足したもの（dB(A)ほぼ不変=人間の\n"
        "  聞こえの大きさは同じで低音だけ増える統制）。ヘッドホンでどうぞ\n\n"
        "判定してほしいこと: ①02は01より「街の暗騒音」らしいか\n"
        "②04はEVらしいか（不自然な電子音になっていないか）\n"
        "③05は03より大型車らしいか\n"
        "・06: 在来線列車72km/h（転動音500-2kHz＋低域＋「ガタンゴトン」1秒周期。\n"
        "  第4種踏切シナリオ用）。列車らしいか\n"
        "・07: 電動キックボード14km/h（小径タイヤの転がり音＋モータ音1.6kHz）。\n"
        "  「静かだが確かに何か来る」感が出ているか\n", encoding="utf-8")
    print("\n->", OUT)


if __name__ == "__main__":
    main()
