# -*- coding: utf-8 -*-
"""自転車のベル 第 2 版（bike_bell_v2・実録に合わせた試作）の試聴ファイル（2026-10-07 23:45・本人「ベルも同じみたいに音色合成データ再作成」）。
出力 out/audition_2026-10-07c/（wav は git 追跡外・README.txt だけ残す）。全ファイル共通の聴くためのゲイン +40 dB（ベルは大きいので走行音より少なく）。
- real_*: 本人の自転車のベルの実録（家のデモ・W チャンネル・換算 −42.3 dB＝校正の 1 分）。demo1 はマイクの手前 2 m 付近・demo3 は 4 m 付近。
- synth_v2_*: 第 2 版を、実録と同じ L_AFmax になるように合わせたもの（音色の比較用・置き方は模さない）。
- synth_v1_*: 既存の make_bike_bell（ディンディン型・基音 3 kHz）と make_bike_bell_ring（リング型）を同じ大きさで。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from outdoor_seld.bike_bell_v2 import make_bike_bell_v2  # noqa: E402
from outdoor_seld.alert_sounds import make_bike_bell, make_bike_bell_ring  # noqa: E402
from bicycle_measure_analyze import a_weighted_signal, fast_level  # noqa: E402
FS = 48000; GAIN = 40.0; OUT = ROOT / "out" / "audition_2026-10-07c"; OUT.mkdir(parents=True, exist_ok=True)
REC_GAIN_DB = -42.3
lines = [__doc__.strip(), ""]
def lafmax(x): return float(fast_level(a_weighted_signal(x, FS), FS).max())
def save(name, x, note):
    y = x * 10 ** (GAIN / 20.0); pk = float(np.max(np.abs(y)))
    if pk > 0.99: y *= 0.99 / pk; note += f"（ピークを {20*np.log10(0.99/pk):+.1f} dB 下げた）"
    sf.write(str(OUT / name), (y * 32767).astype(np.int16), FS, subtype="PCM_16"); lines.append(f"{name}: {note}"); print("wrote", name)
def load(path, a, b):
    x, sr = sf.read(str(ROOT / path), dtype="float64", always_2d=True)
    return resample_poly(x[int(a*sr):int(b*sr), 0], 1, sr // FS) * 10 ** (REC_GAIN_DB / 20.0)
r1 = load("raw_demo/261007_003.WAV", 54.3, 57.3); save("real_bell_demo1_2m.wav", r1, f"実録 demo1（手前 2 m 付近・2 打）L_AFmax {lafmax(r1):.1f} dB(A)")
r3 = load("raw_demo/261007_006.WAV", 52.6, 55.6); save("real_bell_demo3_4m.wav", r3, f"実録 demo3（4 m 付近・3 打）L_AFmax {lafmax(r3):.1f} dB(A)")
target = lafmax(r1)
rng = np.random.default_rng(7)
for f0 in (2465.0, 2350.0, 2650.0):
    s = make_bike_bell_v2(3.0, FS, f0=f0, second_gap_sec=0.7, second_db=-16.0, repeat_period_sec=10.0, rng=rng)
    s = s * 10 ** ((target - lafmax(s)) / 20.0)
    save(f"synth_v2_bell_f{int(f0)}.wav", s, f"第 2 版・基音 {int(f0)} Hz・強い 1 打＋0.7 s 後に −16 dB の 2 打目・L_AFmax を demo1 に合わせた")
s = make_bike_bell(3.0, FS, f0=3000.0, n_rings=2, ring_gap_sec=0.45, repeat_period_sec=10.0); s = s * 10 ** ((target - lafmax(s)) / 20.0)
save("synth_v1_bell_dingding_3000.wav", s, "既存 make_bike_bell（ディンディン型・基音 3,000 Hz・2 打 0.45 s）同じ大きさ")
s = make_bike_bell_ring(3.0, FS, f0=3000.0); s = s * 10 ** ((target - lafmax(s)) / 20.0)
save("synth_v1_bell_ring_3000.wav", s, "既存 make_bike_bell_ring（リング型・基音 3,000 Hz）同じ大きさ")
(OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines[-7:]))
