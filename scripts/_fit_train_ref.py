# -*- coding: utf-8 -*-
"""実録参照（soundeffect-lab、本人指定）から列車合成の高解像度パラメータをフィットする。

出力: src/outdoor_seld/train_ref_params.json
  - body_env: 走行音の1/6オクターブ・スペクトル包絡（train-driving1の定常部）
  - impact: 打撃1発の実測（スペクトル包絡＋減衰時定数、train-pass2の孤立打撃平均）
  - horn_air / horn_electric: 部分音の実測周波数と相対振幅＋振幅エンベロープ概形

録音はパラメータ抽出のみに使用（データセットへのサンプル混入なし=クリーン合成方針）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SP = Path(r"C:\Users\satos\AppData\Local\Temp\claude\c--Users-satos-research"
          r"\f06c9494-5feb-49cd-8986-40f95882d5d7\scratchpad\se_ref")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_mono(name):
    x, fs = sf.read(SP / name)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), fs


def sixth_oct_env(x, fs, f_lo=25.0, f_hi=11000.0):
    """1/6オクターブ帯の相対レベル[dB]（最大=0）。"""
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    freqs, lvls = [], []
    fc = f_lo
    while fc < f_hi:
        m = (f >= fc * 2 ** (-1 / 12)) & (f < fc * 2 ** (1 / 12))
        if m.any():
            freqs.append(round(fc, 1))
            lvls.append(10 * np.log10(spec[m].mean() + 1e-18))
        fc *= 2 ** (1 / 6)
    lvls = np.array(lvls) - max(lvls)
    return freqs, [round(v, 2) for v in lvls]


def fit_body():
    x, fs = read_mono("train-driving1.mp3")
    x = x[int(5 * fs): int(35 * fs)]          # 定常部
    return dict(zip(("freqs", "db"), sixth_oct_env(x, fs)))


def fit_impact():
    x, fs = read_mono("train-pass2.mp3")
    # 60-2000Hz包絡のピークから孤立打撃を拾い、前後60msを平均
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    Xb = np.where((f >= 60) & (f <= 2000), X, 0)
    env = np.abs(np.fft.irfft(Xb, n=len(x)))
    k = int(0.005 * fs)
    env = np.convolve(env, np.ones(k) / k, "same")
    th = env.mean() + 2.5 * env.std()
    idx = np.where(env > th)[0]
    # 孤立ピーク（前後80ms内の最大）を最大10個
    picks, last = [], -10 ** 9
    for i in idx:
        if i - last > int(0.08 * fs):
            j = i + np.argmax(env[i:i + int(0.03 * fs)])
            picks.append(j)
            last = j
        if len(picks) >= 10:
            break
    wins = []
    for j in picks:
        a, b = j - int(0.005 * fs), j + int(0.055 * fs)
        if a >= 0 and b < len(x):
            wins.append(x[a:b])
    w = np.stack(wins)
    # スペクトル包絡（平均）と減衰時定数（包絡の対数直線フィット）
    freqs, db = sixth_oct_env(w.mean(axis=0), fs, 50.0, 6000.0)
    e = np.abs(w).mean(axis=0)
    e = np.convolve(e, np.ones(k) / k, "same")
    t = np.arange(len(e)) / fs
    m = (t > 0.008) & (e > e.max() * 0.05)
    tau = float(-1.0 / np.polyfit(t[m], np.log(e[m] + 1e-12), 1)[0])
    return {"freqs": freqs, "db": db, "tau_s": round(tau, 4), "n_hits": len(wins)}


def fit_horn(name):
    x, fs = read_mono(name)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    order = np.argsort(spec)[::-1]
    peaks = []
    for i in order:
        fi = float(f[i])
        if fi > 80 and all(abs(fi - p) / p > 0.03 for p, _ in peaks):
            peaks.append((fi, float(spec[i])))
        if len(peaks) >= 10:
            break
    peaks.sort()
    ref = max(p[1] for p in peaks)
    partials = [{"hz": round(p, 1), "amp": round(np.sqrt(v / ref), 3)}
                for p, v in peaks if v / ref > 0.001]
    # 振幅エンベロープ概形（20分割）
    seg = np.array_split(np.abs(x), 20)
    env = [round(float(np.sqrt((s ** 2).mean())), 4) for s in seg]
    env = [round(v / max(env), 3) for v in env]
    return {"partials": partials, "env20": env, "dur_s": round(len(x) / fs, 2)}


def main():
    params = {"_provenance": "soundeffect-lab.info machine SE (本人指定2026-08-05) から"
                             "パラメータ抽出。サンプル自体は不使用",
              "body_env": fit_body(),
              "impact": fit_impact(),
              "horn_air": fit_horn("train-horn2.mp3"),
              "horn_electric": fit_horn("train-horn1.mp3")}
    out = ROOT / "src" / "outdoor_seld" / "train_ref_params.json"
    out.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    print("impact: n_hits", params["impact"]["n_hits"], "tau", params["impact"]["tau_s"])
    print("horn_air partials:", params["horn_air"]["partials"][:6])
    print("horn_electric partials:", params["horn_electric"]["partials"][:6])


if __name__ == "__main__":
    main()
