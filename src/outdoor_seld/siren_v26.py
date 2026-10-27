# -*- coding: utf-8 -*-
"""サイレンのドライ音・第 6 版（2026-10-12・Fable・依頼文 r5 §10「サイレン」・順番の 5）。既存の `siren.py` は変えない。

- 消防車のウー `make_fire_siren_v26`: 代表値 下 400 → 上 780 Hz・上がる → 一定（上で止まる）→ 下がる・1 周期 約 6 秒（事実 = 2009 年の予稿の本文・表 1・特許の実施例）。
  引く幅 上 U(780, 850)・下 U(300, 400)・周期 U(5.5, 6.5) s・上で止まる長さは周期の U(0.15, 0.30)（**仮定**）。鐘（カンカン・3 点鐘）は 1,150 ± 60 Hz・
  周期ごとに 3 打（今の make_fire_siren と同じ・打撃間隔 0.50 s・bell_mix 0.3）。
- パトカーのウー `make_patrol_siren_v26`: 代表値 450 → 850 Hz・上がる → 上で止まる → 下がる・周期 4 秒と 8 秒の例（事実 = 同予稿の本文・図 1）。
  引く幅 上 U(810, 910)・下 U(428, 473)・4 s か 8 s を半々 × U(0.9, 1.1)・上で止まる長さは周期の U(0.10, 0.30)・下では止まらない（**仮定**）。
- ピーポー `make_peepo_siren_v26`: 960 Hz と 770 Hz を各 0.65 s（事実・東京消防庁）。うなりは**近い 2 つの音を足して作る**（事実 = パトライトの特許【0016】の例）:
  ずらす幅 ピー U(7.15, 14.02) Hz・ポー U(9.72, 10.93) Hz（事実の範囲・一様は仮定）、深さ = 基本の音の 2 つの振幅の比 r で 20 log10((1+r)/(1−r)) dB を
  U(5, 18) dB（1993 の表 4 の 1 台 ピー 約 18・ポー 約 12 dB・幅は仮定）。30% の車はほとんど揺れない（深さ U(0, 2) dB・仮定）。倍音は車ごとに 2 倍音 U(−15, −2) dB・
  3 倍音 U(−20, −4) dB（1993 の実測の幅）。2 つの音それぞれに倍音を付けて足す（全体に大きさの変調は掛けない・SV10-M02）。
- 旧版: ピーポーは倍音固定（−6／−12 dB）で揺れなし、消防車は 361 → 710 Hz・2.97 s の正弦の往復、パトカーは 435 → 870 Hz の正弦の往復。

自己検査: python -m outdoor_seld.siren_v26
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIRE_F_HI = (780.0, 850.0); FIRE_F_LO = (300.0, 400.0); FIRE_PERIOD = (5.5, 6.5); FIRE_HOLD_FRAC = (0.15, 0.30)
FIRE_BELL_F0 = (1090.0, 1210.0)
PATROL_F_HI = (810.0, 910.0); PATROL_F_LO = (428.0, 473.0); PATROL_PERIODS = (4.0, 8.0); PATROL_PERIOD_JIT = (0.9, 1.1); PATROL_HOLD_FRAC = (0.10, 0.30)
PEEPO_F_HI, PEEPO_F_LO, PEEPO_TONE_SEC = 960.0, 770.0, 0.65
PEEPO_BEAT_HI = (7.15, 14.02); PEEPO_BEAT_LO = (9.72, 10.93)
PEEPO_DEPTH_DB = (5.0, 18.0); PEEPO_DEPTH_WEAK_DB = (0.0, 2.0); PEEPO_P_WEAK = 0.30
PEEPO_H2_DB = (-15.0, -2.0); PEEPO_H3_DB = (-20.0, -4.0)


def depth_to_ratio(depth_db: float) -> float:
    """深さ 20 log10((1+r)/(1−r)) dB → r（2 つの音の振幅の比）。"""
    k = 10.0 ** (depth_db / 20.0)
    return (k - 1.0) / (k + 1.0)


def _hold_sweep(n: int, fs: int, f_lo: float, f_hi: float, period: float, hold_frac: float, phase0: float = 0.0) -> np.ndarray:
    """上がる → 上で止まる → 下がる を繰り返す瞬時周波数（下では止まらない）。上がる・下がるは同じ長さ。"""
    t = (np.arange(n) / fs + phase0 * period) % period
    rise = 0.5 * (1.0 - hold_frac) * period
    hold = hold_frac * period
    f = np.empty(n)
    m1 = t < rise
    f[m1] = f_lo + (f_hi - f_lo) * (t[m1] / rise)
    m2 = (t >= rise) & (t < rise + hold)
    f[m2] = f_hi
    m3 = t >= rise + hold
    f[m3] = f_hi - (f_hi - f_lo) * ((t[m3] - rise - hold) / rise)
    return f


def _tone_with_harmonics(phase: np.ndarray, h2_db: float, h3_db: float) -> np.ndarray:
    return np.sin(phase) + 10 ** (h2_db / 20.0) * np.sin(2.0 * phase) + 10 ** (h3_db / 20.0) * np.sin(3.0 * phase)


def _fade(x: np.ndarray, fs: int) -> np.ndarray:
    fade = int(0.01 * fs)
    env = np.ones(len(x)); env[:fade] = np.linspace(0.0, 1.0, fade); env[-fade:] = np.linspace(1.0, 0.0, fade)
    return x * env


def fire_params_v26(rng: np.random.Generator) -> dict:
    return {"siren_type": "fire", "v26": True, "f_lo": float(rng.uniform(*FIRE_F_LO)), "f_hi": float(rng.uniform(*FIRE_F_HI)),
            "sweep_period_sec": float(rng.uniform(*FIRE_PERIOD)), "hold_frac": float(rng.uniform(*FIRE_HOLD_FRAC)),
            "bell_f0": float(rng.uniform(*FIRE_BELL_F0)), "h2_db": -6.0, "h3_db": -12.0}


def patrol_params_v26(rng: np.random.Generator) -> dict:
    base = PATROL_PERIODS[0] if rng.random() < 0.5 else PATROL_PERIODS[1]
    return {"siren_type": "wail", "v26": True, "f_lo": float(rng.uniform(*PATROL_F_LO)), "f_hi": float(rng.uniform(*PATROL_F_HI)),
            "sweep_period_sec": base * float(rng.uniform(*PATROL_PERIOD_JIT)), "hold_frac": float(rng.uniform(*PATROL_HOLD_FRAC)),
            "h2_db": -6.0, "h3_db": -12.0}


def peepo_params_v26(rng: np.random.Generator) -> dict:
    weak = bool(rng.random() < PEEPO_P_WEAK)
    depth = float(rng.uniform(*(PEEPO_DEPTH_WEAK_DB if weak else PEEPO_DEPTH_DB)))
    return {"siren_type": "peepo", "v26": True, "f_hi": PEEPO_F_HI, "f_lo": PEEPO_F_LO, "tone_sec": PEEPO_TONE_SEC,
            "beat_hi_hz": float(rng.uniform(*PEEPO_BEAT_HI)), "beat_lo_hz": float(rng.uniform(*PEEPO_BEAT_LO)), "depth_db": depth, "weak_beat": weak,
            "h2_db": float(rng.uniform(*PEEPO_H2_DB)), "h3_db": float(rng.uniform(*PEEPO_H3_DB))}


def make_fire_siren_v26(duration_sec: float, fs: int, f_lo: float, f_hi: float, sweep_period_sec: float, hold_frac: float,
                        bell_f0: float = 1150.0, h2_db: float = -6.0, h3_db: float = -12.0, bell_mix: float = 0.3,
                        peak: float = 0.9, rng: np.random.Generator | None = None, **_ignored) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    n = int(round(duration_sec * fs))
    f_inst = _hold_sweep(n, fs, f_lo, f_hi, sweep_period_sec, hold_frac)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = _fade(_tone_with_harmonics(phase, h2_db, h3_db), fs)
    # 鐘（3 点鐘・0.50 s 間隔・周期ごと・今の make_fire_siren と同じ作り）
    m_tone = int(1.2 * fs)
    tt = np.arange(m_tone) / fs
    partials = [(1.00, 1.00), (2.00, 0.35), (2.76, 0.22), (4.10, 0.10)]
    strike = np.zeros(m_tone)
    for ratio, amp in partials:
        strike += amp * np.sin(2.0 * np.pi * bell_f0 * ratio * tt)
    strike *= np.exp(-tt / 0.24)
    ns = int(0.006 * fs)
    click = rng.standard_normal(ns); spec = np.fft.rfft(click); fg = np.fft.rfftfreq(ns, 1.0 / fs); spec[(fg < 800) | (fg > 3000)] = 0.0
    click = np.fft.irfft(spec, n=ns); click = click / (np.max(np.abs(click)) + 1e-12)
    strike[:ns] += 10 ** (-18.0 / 20) * click * np.linspace(1, 0, ns)
    strike *= bell_mix
    ts = 0.58 * sweep_period_sec
    while ts < duration_sec:
        for gap in (0.0, 0.50, 1.00):
            i0 = int((ts + gap) * fs); m = min(m_tone, n - i0)
            if m > 0:
                x[i0:i0 + m] += strike[:m]
        ts += sweep_period_sec
    pk = float(np.max(np.abs(x)))
    return peak * x / pk if pk > 0 else x


def make_patrol_siren_v26(duration_sec: float, fs: int, f_lo: float, f_hi: float, sweep_period_sec: float, hold_frac: float,
                          h2_db: float = -6.0, h3_db: float = -12.0, peak: float = 0.9, **_ignored) -> np.ndarray:
    n = int(round(duration_sec * fs))
    f_inst = _hold_sweep(n, fs, f_lo, f_hi, sweep_period_sec, hold_frac)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = _fade(_tone_with_harmonics(phase, h2_db, h3_db), fs)
    pk = float(np.max(np.abs(x)))
    return peak * x / pk if pk > 0 else x


def make_peepo_siren_v26(duration_sec: float, fs: int, f_hi: float = PEEPO_F_HI, f_lo: float = PEEPO_F_LO, tone_sec: float = PEEPO_TONE_SEC,
                         beat_hi_hz: float = 10.0, beat_lo_hz: float = 10.0, depth_db: float = 12.0, h2_db: float = -6.0, h3_db: float = -12.0,
                         cross_sec: float = 0.015, peak: float = 0.9, **_ignored) -> np.ndarray:
    """ピーポー: 各音を「f と f + Δf の 2 つの音（それぞれ倍音つき）」の和で作る → Δf のうなり。深さは 2 音の振幅の比 r。"""
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    pos = (t % (2.0 * tone_sec)) / tone_sec
    frac = pos % 1.0
    ramp = np.clip(frac / (cross_sec / tone_sec), 0.0, 1.0)
    in_hi = pos < 1.0
    f_prev = np.where(in_hi, f_lo, f_hi); f_cur = np.where(in_hi, f_hi, f_lo)
    f1 = f_prev + (f_cur - f_prev) * ramp                               # 1 つ目の音の瞬時周波数
    df = np.where(in_hi, beat_hi_hz, beat_lo_hz)
    r = depth_to_ratio(depth_db)
    ph1 = 2.0 * np.pi * np.cumsum(f1) / fs
    ph2 = 2.0 * np.pi * np.cumsum(f1 + df) / fs
    x = _tone_with_harmonics(ph1, h2_db, h3_db) + r * _tone_with_harmonics(ph2, h2_db, h3_db)
    x = _fade(x, fs)
    pk = float(np.max(np.abs(x)))
    return peak * x / pk if pk > 0 else x


# ---------------------------------------------------------------- 検査の道具
def track_f0(x: np.ndarray, fs: int, frame_s: float = 0.05, f_lo: float = 200.0, f_hi: float = 1200.0) -> tuple[np.ndarray, np.ndarray]:
    """50 ms ごとの一番強い山（f_lo〜f_hi）の周波数。戻り = (時刻, 周波数)。"""
    n = int(frame_s * fs); m = len(x) // n
    ts, fsv = [], []
    for k in range(m):
        seg = x[k * n:(k + 1) * n] * np.hanning(n)
        X = np.abs(np.fft.rfft(seg, n=8 * n)); f = np.fft.rfftfreq(8 * n, 1.0 / fs)
        mm = (f >= f_lo) & (f <= f_hi)
        ts.append((k + 0.5) * frame_s); fsv.append(float(f[mm][int(np.argmax(X[mm]))]))
    return np.array(ts), np.array(fsv)


def envelope_depth_db(x: np.ndarray, fs: int, t0: float, t1: float, lp_hz: float = 40.0, band: tuple | None = None) -> float:
    """区間 [t0, t1] の包絡の最大と最小の差 [dB]（うなりの深さの測定）。
    band なし: |x| を lp_hz（40 Hz）で平滑した広帯域の包絡（耳で聞く揺れに近い・倍音は Δf の 2 倍・3 倍で揺れるので基本の音より浅く出る）。
    band=(f_lo, f_hi): その帯（例: 基本の音 850〜1,070 Hz）だけ取り出し、解析信号（Hilbert）の包絡を平滑なしで測る。
    2 音の和なら包絡は |a1 + a2 e^{iΔωt}| そのもので、深さ 20 log((1+r)/(1−r)) がそのまま出る（40 Hz の平滑だと深い谷（15 dB 超）が埋まって
    2〜3 dB 浅く出た＝音の検査 SIREN_PEEPO 2026-10-12）。SV10-M02: 基本の音の深さと広帯域の深さを別々に見る。"""
    from scipy.signal import butter, sosfiltfilt, hilbert
    seg = x[int(t0 * fs): int(t1 * fs)]
    if band is not None:
        bp = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band", output="sos")
        env = np.abs(hilbert(sosfiltfilt(bp, seg)))
    else:
        sos = butter(2, lp_hz / (fs / 2), btype="low", output="sos")
        env = sosfiltfilt(sos, np.abs(seg))
    env = env[int(0.05 * fs): -int(0.05 * fs)]
    return float(20.0 * np.log10(np.max(env) / max(np.min(env), 1e-9)))


def beat_rate_hz(x: np.ndarray, fs: int, t0: float, t1: float) -> float:
    """区間の包絡の揺れの速さ [Hz]（包絡の自己相関の最初の山）。"""
    from scipy.signal import butter, sosfiltfilt
    seg = np.abs(x[int(t0 * fs): int(t1 * fs)])
    sos = butter(2, 40.0 / (fs / 2), btype="low", output="sos")
    env = sosfiltfilt(sos, seg); env = env - np.mean(env)
    step = int(fs / 1000); e = env[::step]; fs2 = fs / step        # 1 kHz に間引く
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lo, hi = int(fs2 / 20.0), int(fs2 / 4.0)                       # 4〜20 Hz
    k = lo + int(np.argmax(ac[lo:hi]))
    return float(fs2 / k)


def selftest() -> int:
    fs = 48000
    ok = []
    # 消防車: 上で止まる・範囲・周期
    p = {"f_lo": 400.0, "f_hi": 780.0, "sweep_period_sec": 6.0, "hold_frac": 0.2}
    x = make_fire_siren_v26(13.0, fs, **p, bell_f0=1150.0, bell_mix=0.0)
    ts, f0 = track_f0(x, fs)
    ok.append((f"消防車: 基音が 400〜780 Hz の中（{f0.min():.0f}〜{f0.max():.0f}）", f0.min() >= 390 and f0.max() <= 790))
    top = np.sum(f0 >= 770) * 0.05
    ok.append((f"消防車: 1 周期 6 s のうち上で止まる時間が約 1.2 s（周期 × 0.2・測定 {top / 2:.2f} s／周期・2 周期で）", abs(top / 2 - 1.2) < 0.2))
    ok.append((f"消防車: 周期 6 s（6 s 後の基音が同じ・差 {abs(f0[int(1.0 / 0.05)] - f0[int(7.0 / 0.05)]):.0f} Hz）", abs(f0[int(1.0 / 0.05)] - f0[int(7.0 / 0.05)]) < 20))
    xb = make_fire_siren_v26(13.0, fs, **p, bell_f0=1150.0, bell_mix=0.3)
    X = np.abs(np.fft.rfft(xb)); fgrid = np.fft.rfftfreq(len(xb), 1.0 / fs)
    ok.append(("消防車: 鐘の 1,150 Hz の山がある", X[(fgrid > 1140) & (fgrid < 1160)].max() > 3 * np.median(X[(fgrid > 1050) & (fgrid < 1250)])))
    # パトカー: 上で止まる・下では止まらない
    q = {"f_lo": 450.0, "f_hi": 850.0, "sweep_period_sec": 4.0, "hold_frac": 0.2}
    y = make_patrol_siren_v26(9.0, fs, **q)
    ts, g0 = track_f0(y, fs)
    top = np.sum(g0 >= 840) * 0.05 / 2; bottom = np.sum(g0 <= 460) * 0.05 / 2
    ok.append((f"パトカー: 上で止まる時間 約 0.8 s／周期（測定 {top:.2f}）・下では止まらない（{bottom:.2f} s ≤ 0.15）", abs(top - 0.8) < 0.15 and bottom <= 0.15))
    ok.append((f"パトカー: 基音が 450〜850 Hz の中（{g0.min():.0f}〜{g0.max():.0f}）", g0.min() >= 440 and g0.max() <= 860))
    # ピーポー: うなりの速さと深さ・倍音の比
    z = make_peepo_siren_v26(6.0, fs, beat_hi_hz=10.0, beat_lo_hz=10.0, depth_db=12.0, h2_db=-6.0, h3_db=-12.0)
    d_hi = envelope_depth_db(z, fs, 2.6 + 0.05, 2.6 + 0.60, band=(850.0, 1070.0))   # ピーの区間（3 周期目）・基本の音の帯
    d_all = envelope_depth_db(z, fs, 2.6 + 0.05, 2.6 + 0.60)
    rate = beat_rate_hz(z, fs, 2.6 + 0.05, 2.6 + 0.60)
    ok.append((f"ピーポー: 基本の音のうなりの深さ 12 dB（測定 {d_hi:.1f}・±1.5）・広帯域の深さ {d_all:.1f} dB は別に記録", abs(d_hi - 12.0) < 1.5))
    ok.append((f"ピーポー: うなりの速さ 10 Hz（測定 {rate:.1f}・±1）", abs(rate - 10.0) < 1.0))
    zw = make_peepo_siren_v26(6.0, fs, beat_hi_hz=10.0, beat_lo_hz=10.0, depth_db=0.5)
    ok.append((f"ピーポー: 揺れの弱い車（深さ 0.5 dB → 測定 {envelope_depth_db(zw, fs, 2.65, 3.2, band=(850.0, 1070.0)):.1f} ≤ 1.5）", envelope_depth_db(zw, fs, 2.65, 3.2, band=(850.0, 1070.0)) <= 1.5))
    ts, h0 = track_f0(z, fs, f_lo=600.0, f_hi=1100.0)
    ok.append((f"ピーポー: 基音 960／770 Hz を 0.65 s で交互（測定 {np.median(h0[(ts % 1.3) < 0.6]):.0f}／{np.median(h0[(ts % 1.3) > 0.7]):.0f}）",
               abs(np.median(h0[(ts % 1.3) < 0.6]) - 960) < 15 and abs(np.median(h0[(ts % 1.3) > 0.7]) - 770) < 15))
    Xz = np.abs(np.fft.rfft(z[int(2.6 * fs): int(3.25 * fs)])); fz = np.fft.rfftfreq(int(0.65 * fs), 1.0 / fs)
    def pk(f):
        m = (fz > f - 30) & (fz < f + 30); return 20 * np.log10(Xz[m].max())
    ok.append((f"ピーポー: 2 倍音 −6 dB・3 倍音 −12 dB（測定 {pk(1920) - pk(960):+.1f}／{pk(2880) - pk(960):+.1f}・±1.5）", abs(pk(1920) - pk(960) + 6) < 1.5 and abs(pk(2880) - pk(960) + 12) < 1.5))
    rng = np.random.default_rng(3); ps = [peepo_params_v26(rng) for _ in range(4000)]
    ok.append((f"ピーポーの params: 揺れの弱い車 30%（{np.mean([p['weak_beat'] for p in ps]):.3f}）・ずらす幅と深さが幅の中",
               abs(np.mean([p["weak_beat"] for p in ps]) - 0.30) < 0.03 and all(PEEPO_BEAT_HI[0] <= p["beat_hi_hz"] <= PEEPO_BEAT_HI[1] and PEEPO_BEAT_LO[0] <= p["beat_lo_hz"] <= PEEPO_BEAT_LO[1] for p in ps)))
    fp = [fire_params_v26(rng) for _ in range(1000)]; pp = [patrol_params_v26(rng) for _ in range(1000)]
    ok.append(("消防車・パトカーの params: 幅の中・パトカーの周期は 4 と 8 の半々", all(FIRE_F_LO[0] <= p["f_lo"] <= FIRE_F_LO[1] and FIRE_F_HI[0] <= p["f_hi"] <= FIRE_F_HI[1] for p in fp)
               and all(PATROL_F_LO[0] <= p["f_lo"] <= PATROL_F_LO[1] for p in pp) and abs(np.mean([p["sweep_period_sec"] > 6 for p in pp]) - 0.5) < 0.05))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(selftest())
