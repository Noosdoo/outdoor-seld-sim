# -*- coding: utf-8 -*-
"""EV・HV の接近通報音（AVAS）・v26（2026-10-12・Fable・依頼文 r5 §3・§10「EV・HV の接近通報音」・順番の 5）。

既存の `engine_ev.py`（make_car_ev）は変えない。ここで作るのは通報音の成分だけ。混ぜ方と大きさは描画側（step11_v26_render）。
- 形（仮定）: 200〜1,700 Hz に 2〜4 本の音（少なくとも 2 つの 1/3 オクターブ帯に分かれ、そのうち 1 つは 1,600 Hz 帯以下 = R138 の要件の形。1,700 Hz までは 1,600 Hz 帯の中）。
  半分の車には 2.5〜5 kHz の成分も 1 本（リーフ 2010 の 600〜2,500 Hz・日本の市販品の 630 Hz と 4 kHz 付近の山の例が手がかり）。
  上限 1,000 → 1,700 Hz は 2026-10-12 18:24（Opus の調べ: 欧州 EV 1 台のトーン 549／738／1,650 Hz（DAGA 2024・事実）・リーフ 2010 のスイープ 600〜2,500 Hz（事実・メーカー発表））。
  正弦が何本かを測った日本車の資料は無い（プリウス・アクア・ノートは見つからない）→ 音色は「R138 の形の仮定」。
- 高さは速さで上がる: 1 km/h あたり U(0.8, 2.0)%（R138「5〜20 km/h で少なくとも 1 つの成分の平均の上がり方 0.8% 以上」= 事実・上の 2% と範囲の外への延長は仮定）。
  基準は 10 km/h（f = f_base × (1 + k × (v − 10))）。
- 大きさ（描画側）: 横 2 m・高さ 1.2 m で U(46, 70) dB(A)・速さでは変えない（仮定・上限 70 は 2026-10-12 18:24: 国内市販「M 社製」が前方 2 m で約 70 dB(A)（日大 2012・事実・1 例）・R138 の上限 75）。R138 の適合（10 km/h 50・20 km/h 56 dB(A)・帯域の最低値）は下地込みで
  音の検査で確かめる（SV10-M03・まだ）。
- 止める速さ（仮定）: v_stop = U(20, 30) km/h。弱め始め v_fade = max(20, v_stop − U(1, 5)) から v_stop の間で直線で消す。20 km/h までは鳴っている。
  止まっている間（発進の待機）は無音（以前の決定）。
- 今の EV の 8〜10 kHz の音（PWM）は、通報音を足す前の EV の音全体の A 特性のパワーの 2% 以下に（本人 10/10 16:43・仮定）。

自己検査: python -m outdoor_seld.avas_v26
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .noise import colored_noise
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from outdoor_seld.noise import colored_noise

AVAS_N_TONES = (2, 3, 4)
AVAS_F_LOW = (200.0, 1700.0)          # 低い帯の音（Hz・対数一様）。上限 1,700 = 実例に合わせた（DAGA 2024 の EV 1 台 549／738／1,650 Hz・リーフ 2010 は 600〜2,500 Hz のスイープ）・本人 10/12 18:24
AVAS_F_HIGH = (2500.0, 5000.0)        # 半分の車に足す高い成分
AVAS_P_HIGH = 0.5
AVAS_PITCH_PER_KMH = (0.008, 0.020)   # 1 km/h あたりの上がり方（0.8〜2%）
AVAS_V_REF_KMH = 10.0
AVAS_L2M_DBA = (46.0, 70.0)           # 横 2 m・高さ 1.2 m の値（仮定・上限 70 = 国内市販「M 社製」の前方 2 m 約 70 dB(A)（日大 2012）・R138 の上限 75 の下）・本人 10/12 18:24
AVAS_V_STOP_KMH = (20.0, 30.0)
AVAS_FADE_KMH = (1.0, 5.0)
AVAS_PWM_FRAC_A = (0.005, 0.02)       # 8〜10 kHz の音の A 特性パワーの割合（≤ 2%）


def third_octave_band(f: float) -> int:
    """1/3 オクターブ帯の番号（1,000 Hz = 30・IEC 61260 の名目中心）。"""
    return int(np.round(10.0 * np.log10(f / 1000.0) + 30.0))


def avas_params(rng: np.random.Generator) -> dict:
    """1 台分。少なくとも 2 つの 1/3 オクターブ帯に分かれるまで引き直す（同じ帯に 2 本では要件を満たさない・SV10）。"""
    for _ in range(100):
        n = int(rng.choice(AVAS_N_TONES))
        freqs = sorted(float(np.exp(rng.uniform(np.log(AVAS_F_LOW[0]), np.log(AVAS_F_LOW[1])))) for _ in range(n))
        bands = {third_octave_band(f) for f in freqs}
        if len(bands) >= 2:
            break
    amps = [float(rng.uniform(0.5, 1.0)) for _ in freqs]
    high = float(np.exp(rng.uniform(np.log(AVAS_F_HIGH[0]), np.log(AVAS_F_HIGH[1])))) if rng.random() < AVAS_P_HIGH else None
    v_stop = float(rng.uniform(*AVAS_V_STOP_KMH))
    return {"v26": True, "freqs_hz": freqs, "amps": amps, "high_hz": high, "high_amp": 0.3, "pitch_per_kmh": float(rng.uniform(*AVAS_PITCH_PER_KMH)),
            "l2m_dba": float(rng.uniform(*AVAS_L2M_DBA)), "v_stop_kmh": v_stop, "v_fade_kmh": max(AVAS_V_STOP_KMH[0], v_stop - float(rng.uniform(*AVAS_FADE_KMH))),
            "pwm_frac_a": float(rng.uniform(*AVAS_PWM_FRAC_A))}


def avas_gain(v_kmh: np.ndarray | float, p: dict) -> np.ndarray | float:
    """速さ → 利得（1 = 鳴る・0 = 止まる・間は直線）。止まっている（v < 0.5 km/h）ときも 0（発進の待機は無音・以前の決定）。"""
    v = np.asarray(v_kmh, dtype=float)
    g = np.clip((p["v_stop_kmh"] - v) / max(p["v_stop_kmh"] - p["v_fade_kmh"], 1e-6), 0.0, 1.0)
    return np.where(v < 0.5, 0.0, g)


def make_avas(duration_sec: float, fs: int, rng: np.random.Generator, p: dict, v_kmh: np.ndarray | float, apply_gain: bool = True) -> np.ndarray:
    """通報音のドライ音（単位 RMS・高さは v_kmh（スカラーか時系列・長さ n）に従う）。
    apply_gain=True なら速さの利得（avas_gain・止める速さで消える）も掛けて、そのあと単位 RMS にする（旧の形・試聴用）。
    apply_gain=False なら**減衰前の搬送音**を返す（描画側はこれで較正し、そのあと avas_gain(v) を掛ける＝減衰が正規化で消えない・Astra SV26-01・2026-10-13）。"""
    n = int(round(duration_sec * fs))
    v = np.full(n, float(v_kmh)) if np.isscalar(v_kmh) else np.asarray(v_kmh, dtype=float)
    assert len(v) == n
    mult = 1.0 + p["pitch_per_kmh"] * (v - AVAS_V_REF_KMH)
    jitter = colored_noise(n, fs, rng, slope=2.5, f_lo=0.3)
    x = np.zeros(n)
    for f, a in zip(p["freqs_hz"], p["amps"]):
        phase = 2.0 * np.pi * np.cumsum(f * mult * (1.0 + 0.002 * jitter)) / fs
        x += a * np.sin(phase + rng.uniform(0, 2 * np.pi))
    if p.get("high_hz"):
        phase = 2.0 * np.pi * np.cumsum(p["high_hz"] * mult) / fs
        x += p["high_amp"] * np.sin(phase + rng.uniform(0, 2 * np.pi))
    if apply_gain:
        x *= avas_gain(v, p)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return x / rms if rms > 0 else x


def selftest() -> int:
    fs = 48000
    ok = []
    rng = np.random.default_rng(1); ps = [avas_params(rng) for _ in range(3000)]
    ok.append(("params: 低い帯の音は 200〜1,700 Hz・少なくとも 2 つの 1/3 オクターブ帯・1,600 Hz 帯以下", all(all(AVAS_F_LOW[0] <= f <= AVAS_F_LOW[1] for f in p["freqs_hz"]) and len({third_octave_band(f) for f in p["freqs_hz"]}) >= 2 for p in ps)))
    ok.append((f"params: 高い成分は半分の車（{np.mean([p['high_hz'] is not None for p in ps]):.3f}）・2.5〜5 kHz", abs(np.mean([p["high_hz"] is not None for p in ps]) - 0.5) < 0.03 and all(p["high_hz"] is None or AVAS_F_HIGH[0] <= p["high_hz"] <= AVAS_F_HIGH[1] for p in ps)))
    ok.append(("params: 止める速さ 20〜30・弱め始めは 20 以上で止める速さより低い・上がり方 0.8〜2%・PWM ≤ 2%", all(AVAS_V_STOP_KMH[0] <= p["v_stop_kmh"] <= AVAS_V_STOP_KMH[1] and 20.0 <= p["v_fade_kmh"] <= p["v_stop_kmh"] and AVAS_PITCH_PER_KMH[0] <= p["pitch_per_kmh"] <= AVAS_PITCH_PER_KMH[1] and p["pwm_frac_a"] <= 0.02 for p in ps)))
    p = dict(ps[0], freqs_hz=[300.0, 630.0], amps=[1.0, 0.8], high_hz=None, pitch_per_kmh=0.01, v_stop_kmh=25.0, v_fade_kmh=22.0)
    x10 = make_avas(2.0, fs, np.random.default_rng(2), p, 10.0); x20 = make_avas(2.0, fs, np.random.default_rng(2), p, 20.0)
    def peaks(x):
        X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1.0 / fs); out = []
        for f0 in (300.0, 630.0):
            m = (f > f0 * 0.8) & (f < f0 * 1.35); out.append(float(f[m][int(np.argmax(X[m]))]))
        return out
    p10, p20 = peaks(x10), peaks(x20)
    ok.append((f"高さ: 10 km/h で 300／630 Hz（{p10[0]:.0f}／{p10[1]:.0f}）・20 km/h で 10% 上（{p20[0]:.0f}／{p20[1]:.0f}）", abs(p10[0] - 300) < 3 and abs(p10[1] - 630) < 3 and abs(p20[0] - 330) < 3 and abs(p20[1] - 693) < 3))
    g = avas_gain(np.array([0.0, 10.0, 20.0, 22.0, 23.5, 25.0, 30.0]), p)
    ok.append((f"利得: 止まると 0・20 km/h まで 1・22〜25 で直線で消える（{np.round(g, 2).tolist()}）", g[0] == 0 and g[1] == 1 and g[2] == 1 and g[3] == 1 and abs(g[4] - 0.5) < 1e-9 and g[5] == 0 and g[6] == 0))
    v = np.linspace(0.0, 30.0, int(4.0 * fs))                       # 発進: 0 → 30 km/h を 4 s で
    xl = make_avas(4.0, fs, np.random.default_rng(3), p, v)
    seg = lambda a, b: xl[int(a * fs): int(b * fs)]
    ok.append(("発進: 止まっている間は無音・途中は鳴る・止める速さを越えたあとは無音", float(np.max(np.abs(seg(0.0, 0.05)))) == 0.0 and float(np.std(seg(1.5, 2.0))) > 0 and float(np.max(np.abs(seg(3.6, 4.0)))) == 0.0))
    # SV26-01: 減衰前の搬送音（apply_gain=False）に avas_gain を掛ければ、23.5 km/h（利得 0.5）は 15 km/h（利得 1）より −6.02 dB
    pc = dict(ps[0], freqs_hz=[400.0, 700.0], amps=[1.0, 0.8], high_hz=None, pitch_per_kmh=0.0, v_stop_kmh=25.0, v_fade_kmh=22.0)
    car = make_avas(2.0, fs, np.random.default_rng(3), pc, 15.0, apply_gain=False)
    r1 = float(np.sqrt(np.mean((car * float(avas_gain(15.0, pc))) ** 2))); r2 = float(np.sqrt(np.mean((car * float(avas_gain(23.5, pc))) ** 2)))
    ok.append((f"搬送音 × 利得: 23.5 km/h は 15 km/h より {20 * np.log10(r2 / r1):.2f} dB（−6.02 ±0.05・SV26-01）", abs(20 * np.log10(r2 / r1) + 6.02) < 0.05))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(selftest())
