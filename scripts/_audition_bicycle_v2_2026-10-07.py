# -*- coding: utf-8 -*-
"""自転車の走行音 第 2 版（bicycle_v2・実測に合わせた試作）の試聴ファイルを作る（2026-10-07 23:05・本人「合成データをまた作成して。試聴する」）。

出力 out/audition_2026-10-07b/（wav は git 追跡外・README.txt だけ残す）。全ファイルに共通の「聴くためのゲイン」+64 dB を掛けてある
（本プロジェクトの換算 RMS 1.0 = 146 dB なので、50 dB(A) は +64 dB しても −32 dBFS）。ファイル同士の大きさの差はそのまま本物の差。
- real_*: 本人の自転車の実録（261007_007.WAV の W チャンネル・騒音計で合わせた換算 −43.4 dB・横 1.0〜1.2 m・高さ 1.5 m）。
- synth_*: 第 2 版の合成を、簡易の通り過ぎ（1/r だけ・横 1.1 m・マイク 1.5 m・音源 0.3 m）で受音点の L_AFmax を実測の中央値に合わせたもの。
  実際の暗騒音（real_background）を下に重ねた版と、合成だけの版。規則 ② の l1m（本番の描画器で合わせる値）はここでは出さない。
- synth_v1_*: 比較用に第 1 版（bicycle.py）を同じ置き方・同じレベルで。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from outdoor_seld.bicycle_v2 import make_bicycle_v2  # noqa: E402
from outdoor_seld.bicycle import make_bicycle  # noqa: E402
from outdoor_seld.calibration import K_RMS_SPL  # noqa: E402
from bicycle_measure_analyze import a_weighted_signal, fast_level  # noqa: E402

FS = 48000
LISTEN_GAIN_DB = 64.0
OUT = ROOT / "out" / "audition_2026-10-07b"
OUT.mkdir(parents=True, exist_ok=True)
REC_GAIN_DB = -43.4            # 録音 → dB(A) の換算（騒音計との静かな秒の中央値・analysis/summary.txt）
LATERAL, MIC_Z, SRC_Z = 1.1, 1.5, 0.3
TARGETS = {5: 48.0, 9: 49.2, 12: 49.5}          # 受音点の L_AFmax の実測中央値（遅い・普通・速い）
rng = np.random.default_rng(20261007)
lines = [__doc__.strip(), ""]


def lafmax(x: np.ndarray) -> float:
    return float(fast_level(a_weighted_signal(x, FS), FS).max())


def save(name: str, x: np.ndarray, note: str):
    y = x * 10 ** (LISTEN_GAIN_DB / 20.0)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = y * (0.99 / pk); note += f"（ピークを {20*np.log10(0.99/pk):+.1f} dB 下げた）"
    sf.write(str(OUT / name), (y * 32767).astype(np.int16), FS, subtype="PCM_16")
    lines.append(f"{name}: {note}")
    print("wrote", name)


def passby(dry: np.ndarray, v_mps: float, t_cpa: float) -> np.ndarray:
    t = np.arange(len(dry)) / FS
    x = v_mps * (t - t_cpa)
    r = np.sqrt(x ** 2 + LATERAL ** 2 + (MIC_Z - SRC_Z) ** 2)
    return dry / np.maximum(r, 0.3)


def main():
    # --- 実録の抜き出し ---
    w, sr = sf.read(str(ROOT / "raw_bicycle_measure/261007_007.WAV"), dtype="float64", always_2d=True)
    w = resample_poly(w[:, 0], 1, sr // FS) * 10 ** (REC_GAIN_DB / 20.0)
    reads = {}
    for r in csv.DictReader(open(ROOT / "raw_bicycle_measure/reads/reads_A.csv", encoding="utf-8")):
        reads.setdefault(r["take"], {})[float(r["x"])] = float(r["t"])
    picks = [("pass02", "4.0 km/h・遅い"), ("pass06", "9.2 km/h・普通"), ("pass10", "11.8 km/h・速い")]
    DUR = 8.0
    for take, label in picks:
        t0 = reads[take][0.0]
        seg = w[int((t0 - DUR / 2) * FS): int((t0 + DUR / 2) * FS)]
        ta, tb = min(reads[take].values()) - 0.5, max(reads[take].values()) + 0.5     # 解析と同じ窓（目印 ±0.5 s）
        inwin = w[int(ta * FS): int(tb * FS)]
        save(f"real_{take}_{label.split(' ')[0].replace('.', 'p')}.wav", seg, f"実録 {take}（{label}・CPA を中央に 8 s）・通過の窓 {ta:.1f}〜{tb:.1f} s の L_AFmax {lafmax(inwin):.1f} dB(A)（8 s の中には通過の後の足音・ブレーキ等も入る）")
    # 暗騒音: 最も静かな 8 s
    sec = np.array([10 * np.log10(np.mean(w[i * FS:(i + 1) * FS] ** 2) + 1e-30) for i in range(len(w) // FS)])
    k = int(np.argmin(np.convolve(sec, np.ones(8) / 8, mode="valid")))
    bg = w[k * FS:(k + 8) * FS]
    save("real_background_8s.wav", bg, f"実録の暗騒音（録音 {k}〜{k+8} s）・L_Aeq {10*np.log10(np.mean(a_weighted_signal(bg, FS)**2))+K_RMS_SPL:.1f} dB(A)")
    # --- 合成（第 2 版） ---
    dry = make_bicycle_v2(DUR, FS, rng)
    for kmh, target in TARGETS.items():
        v = kmh / 3.6
        p = passby(dry, v, DUR / 2)
        g = 10 ** ((target - lafmax(p)) / 20.0)
        p = p * g
        save(f"synth_v2_pass_{kmh:02d}kmh_with_bg.wav", p + bg, f"合成 第 2 版・{kmh} km/h の通り過ぎ（横 {LATERAL} m）＋実録の暗騒音・受音点 L_AFmax を実測の中央値 {target} dB(A) に合わせた（合成だけの L_AFmax {lafmax(p):.1f}）")
        save(f"synth_v2_pass_{kmh:02d}kmh_dry.wav", p, f"同上・合成だけ（暗騒音なし）")
    # 20 km/h の 2 案（12 km/h より上は未測定）
    v = 20 / 3.6
    for name, target in (("flat", 49.5), ("30logv", 49.5 + 30 * np.log10(20 / 12))):
        p = passby(dry, v, DUR / 2); p = p * 10 ** ((target - lafmax(p)) / 20.0)
        save(f"synth_v2_pass_20kmh_{name}_with_bg.wav", p + bg, f"合成 第 2 版・20 km/h・案「{name}」= 受音点 {target:.1f} dB(A)（flat = 12 km/h と同じ／30logv = 12 km/h から 30 log v で +6.7 dB）＋実録の暗騒音")
    # 第 1 版の比較（9 km/h・同じレベル）
    dry1 = make_bicycle(DUR, FS, rng, speed_mps=9 / 3.6, coast_frac=0.0)
    p = passby(dry1, 9 / 3.6, DUR / 2); p = p * 10 ** ((49.2 - lafmax(p)) / 20.0)
    save("synth_v1_pass_09kmh_with_bg.wav", p + bg, "比較用・第 1 版（低い転がり音あり・400〜7 kHz）を 9 km/h・49.2 dB(A) で＋実録の暗騒音")
    # 定常（1 m・10 s）
    dry10 = make_bicycle_v2(10.0, FS, rng)
    r1 = np.sqrt(LATERAL ** 2 + (MIC_Z - SRC_Z) ** 2)
    p = dry10 / r1; p = p * 10 ** ((49.2 - lafmax(p)) / 20.0)
    save("synth_v2_steady_10s.wav", p, "合成 第 2 版・動かずに 1.66 m（横 1.1 m・高さ差 1.2 m）で 10 s・49.2 dB(A)（音色の確認用）")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
