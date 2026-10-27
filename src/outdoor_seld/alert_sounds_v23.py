# -*- coding: utf-8 -*-
"""警告音のドライ音・第 3 版（2026-10-12・Fable・依頼文 `md/handoff/Fable依頼_合成の音を現実に近づける_2026-10-10.md` r5 §1・§10）。

既存の `alert_sounds.py` は変えない（旧版の再現のため）。この版で変えるのは次の 3 つ。音の大きさの合わせ方はここでは変えない（別の決定）。

1. クラクション `make_horn_v23`: 「押し」の列をクリップの絶対時刻で描く。
   - 止まった車（1 回の押し = 1 つの窓）: 窓の中は連続音（細目告示 141 条「1 回押している間は鳴りっぱなし」・事実）。
   - 走る車（1 つの窓に複数の押し）: 回数と押しの長さは Takada 2013/2014 の 6 型の割合（単発 27.6%・短 2 回 27.3%・短 3 回以上 5.0%・
     長 1 回 28.6%・長 2 回 9.0%・長 3 回以上 2.4%・歩行者アンケート＝事実。3 回以上は 3 回＝仮定）、短い押し 0.15〜0.7 s・長い押し 0.8〜2.0 s・
     押しの間 0.2〜0.5 s（仮定・本人 10/10 16:43）。鳴り始め 5〜15 ms・鳴り終わりは指数で減る（時定数 10〜40 ms・仮定）。
   - 倍音は整数次（事実・Lemaitre 2009）。各次数の強さ k^(−α)（α は本ごとに引く・仮定）。単音と 2 音を半々（仮定）。
   - 旧 `make_horn` は 0 秒から 0.35 s 鳴って 0.15 s 休むの繰り返しを窓で切るだけだった → 1 回の押しが 2〜3 区間に切れていた（Astra AS10-02）。
2. バック音 `make_backup_beep_v23`: 先頭の音を鳴動窓の頭 t_on に合わせる。型は 3 つを同じ確率（仮定＝設計の分布）:
   - 単音スピーカー式 700〜1,900 Hz・単音圧電式 2,100〜3,200 Hz: 毎分 60〜100 回・鳴る割合 0.4〜0.6（YEC の試聴音 7 つ＋ミツバ RH-11・1 本の中では一定）。
   - 2 音チャイム: ピン（約 2,000 Hz）→ 0.25 s 後にポーン（約 1,590 Hz）を 1.25 s ごと（YEC BA1271506 の試聴音の測定＝事実・1 例）。
     高さは 2 音に同じ倍率 U(0.9, 1.1)、時間は「時間の倍率」U(0.9, 1.1) を 2 音の間と組の間の両方に（仮定）。減り方は仮定（時定数 0.15〜0.25 s）。
   - 旧 `make_backup_beep` は 1 kHz の単音・0.5 s 鳴って 0.5 s 休むを 0 秒から繰り返していた。
3. ベルの音の高さ `bell_f0_v23`: 自転車産業振興協会 2023 表 3（国内流通品 28 個・FFT で一番強い山の周波数＝事実）から、
   **製品の種類を同じ確率で 1 つ選び、その種類の試料から同じ確率で 1 つ選び、±3% ゆらす**（本人 10/10 22:26・選び方と ±3% は仮定）。
   チリリン型 ↔ 引きベル 8 種 22 個、ピン型 ↔ 単打ベル 4 種 5 個、ディンディン型 ↔ 28 個の 13 種全部（対応は推測）。
   個数は市場の割合ではない。2 回測った値は 1 回目を使う（測り直しを別の試料に数えない・SV10-L03）。ディンディン型も 1 回だけ鳴らす（本人 10/10 16:43）。

自己検査: python -m outdoor_seld.alert_sounds_v23（または python src/outdoor_seld/alert_sounds_v23.py）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .alert_sounds import colored_noise
except ImportError:                                   # スクリプトとして直接実行したとき（python src/outdoor_seld/alert_sounds_v23.py）
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from outdoor_seld.alert_sounds import colored_noise

# ---------------------------------------------------------------- クラクション
# Takada 2013（表 2・130 人）＋ 2014（表 5・関東 151・関西 106）を回答者数で合わせた 6 型の割合（Astra SV10 の再計算 27.586/27.321/5.040/28.647/9.019/2.387%）
HORN_TYPES = (
    ("single_short", 1, "short", 0.27586),
    ("double_short", 2, "short", 0.27321),
    ("triple_short", 3, "short", 0.05040),
    ("single_long", 1, "long", 0.28647),
    ("double_long", 2, "long", 0.09019),
    ("triple_long", 3, "long", 0.02387),
)
HORN_PRESS_SHORT = (0.15, 0.70)      # 仮定（本人 10/10 16:43）
HORN_PRESS_LONG = (0.80, 2.00)       # 仮定
HORN_GAP = (0.20, 0.50)              # 押しの間（仮定）
HORN_ONSET = (0.005, 0.015)          # 鳴り始め（仮定・実装の都合）
HORN_RELEASE_TAU = (0.010, 0.040)    # 鳴り終わりの時定数（仮定）
HORN_F_LO = (380.0, 440.0)           # 基音（旧 410 × U(0.95, 1.05) を包む幅・仮定）
HORN_DUAL_RATIO = (1.18, 1.26)       # 2 音の比（旧 500/410 = 1.22 を包む・仮定）
HORN_ALPHA = (0.5, 0.9)              # 倍音の減り方 k^(−α)（スペクトル重心が Lemaitre 2009 の 1,380〜3,790 Hz に入る幅・自己検査で確かめる・仮定）
HORN_N_HARMONICS = 12
HORN_P_DUAL = 0.5                    # 単音と 2 音を半々（仮定）


def horn_params_v23(rng: np.random.Generator) -> dict:
    """本ごとの音色（押しの列は別・`horn_press_plan`）。"""
    f_lo = float(rng.uniform(*HORN_F_LO))
    dual = bool(rng.random() < HORN_P_DUAL)
    ratio = float(rng.uniform(*HORN_DUAL_RATIO))
    return {"v23": True, "f_lo": f_lo, "f_hi": (f_lo * ratio if dual else None), "dual": dual,
            "alpha": float(rng.uniform(*HORN_ALPHA)), "onset_sec": float(rng.uniform(*HORN_ONSET)),
            "release_tau_sec": float(rng.uniform(*HORN_RELEASE_TAU))}


def horn_press_plan(rng: np.random.Generator) -> dict:
    """走る車の押しの列（窓の頭 t_on からの相対時刻）: {"type", "n", "kind", "presses": [(start, length), …], "total"}。"""
    p = np.array([t[3] for t in HORN_TYPES]); p = p / p.sum()
    k = int(rng.choice(len(HORN_TYPES), p=p))
    name, n, kind, _ = HORN_TYPES[k]
    lo, hi = HORN_PRESS_SHORT if kind == "short" else HORN_PRESS_LONG
    presses, t = [], 0.0
    for i in range(n):
        length = float(rng.uniform(lo, hi))
        presses.append((round(t, 4), round(length, 4)))
        t += length
        if i < n - 1:
            t += float(rng.uniform(*HORN_GAP))
    return {"type": name, "n": n, "kind": kind, "presses": presses, "total": round(t, 4)}


def _press_envelope(n: int, fs: int, presses: list, onset_sec: float, tau: float) -> np.ndarray:
    env = np.zeros(n)
    for start, length in presses:
        i0 = int(round(start * fs)); i1 = int(round((start + length) * fs))
        if i0 >= n:
            continue
        i1 = min(i1, n)
        env[i0:i1] = np.maximum(env[i0:i1], 1.0)
        k = min(int(onset_sec * fs), i1 - i0)
        if k > 1:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)        # 上がり: 余弦の半分
            env[i0:i0 + k] = np.maximum(env[i0:i0 + k] * 0 + ramp, env[i0:i0 + k] * 0)
        m = min(int(5.0 * tau * fs), n - i1)                             # 下がり: 指数（5τ で切る）
        if m > 0:
            tail = np.exp(-np.arange(m) / (tau * fs))
            env[i1:i1 + m] = np.maximum(env[i1:i1 + m], tail)
    return env


def make_horn_v23(duration_sec: float, fs: int, rng: np.random.Generator, presses: list,
                  f_lo: float = 410.0, f_hi: float | None = 500.0, alpha: float = 0.7,
                  onset_sec: float = 0.01, release_tau_sec: float = 0.02,
                  n_harmonics: int = HORN_N_HARMONICS, noise_mix: float = 0.06, peak: float = 0.9) -> np.ndarray:
    """クラクションのドライ音。presses = [(開始秒, 長さ秒), …]（クリップの絶対時刻）。押しの中は連続音、押しの外は無音（鳴り終わりの短い尾だけ）。"""
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    env = _press_envelope(n, fs, presses, onset_sec, release_tau_sec)

    def tone(f: float) -> np.ndarray:
        x = np.zeros(n)
        for k in range(1, n_harmonics + 1):
            jit = 1.0 + (rng.uniform(-0.002, 0.002) if k > 1 else 0.0)     # 整数次からのわずかなずれ（非調和の雑味・仮定）
            x += (k ** (-alpha)) * np.sin(2.0 * np.pi * f * k * jit * t + rng.uniform(0, 2 * np.pi))
        return x

    x = tone(f_lo) if f_hi is None else 0.5 * tone(f_lo) + 0.5 * tone(f_hi)
    if noise_mix > 0:
        breath = colored_noise(n, fs, rng, slope=0.5, f_lo=200.0)
        breath = breath / (np.max(np.abs(breath)) + 1e-12) * np.max(np.abs(x))
        x = (1.0 - noise_mix) * x + noise_mix * breath
    x = x * env
    pk = float(np.max(np.abs(x)))
    return peak * x / pk if pk > 0 else x


# ---------------------------------------------------------------- バック音
BACKUP_KINDS = ("speaker", "piezo", "chime")       # 3 つを同じ確率（仮定＝設計の分布・市場の割合ではない）
BACKUP_FREQ = {"speaker": (700.0, 1900.0), "piezo": (2100.0, 3200.0)}   # YEC の公称値の幅（圧電式の下は BA3301500 の 2,400 ± 300）
BACKUP_RATE_PER_MIN = (60.0, 100.0)                # 単音 2 型だけ（本人 10/10 17:02・YEC 7 つ 60〜81＋RH-11 90 ± 10 を包む・一様は仮定）
BACKUP_DUTY = (0.40, 0.60)                         # 鳴る時間の割合（試聴音 0.42〜0.58）
BACKUP_RAMP_SEC = 0.005
CHIME_PING_HZ, CHIME_PONG_HZ = 2000.0, 1590.0      # YEC BA1271506 の試聴音（事実・1 例）
CHIME_GAP_SEC, CHIME_PERIOD_SEC = 0.25, 1.25       # ピン → 0.25 s 後にポーン・1.25 s ごと（事実・Astra 再測定 0.2555／1.2489 s）
CHIME_PITCH_MULT = (0.9, 1.1)                      # 仮定
CHIME_TIME_MULT = (0.9, 1.1)                       # 仮定（時間の倍率に固定・速さの倍率は引かない）
CHIME_TAU = (0.15, 0.25)                           # 減り方（仮定・試聴音の減衰は部屋を含むので使わない）
CHIME_H2_DB = -12.0                                # 2 倍音（仮定）


def backup_params_v23(rng: np.random.Generator) -> dict:
    kind = BACKUP_KINDS[int(rng.integers(0, len(BACKUP_KINDS)))]
    if kind == "chime":
        return {"v23": True, "kind": kind, "pitch_mult": float(rng.uniform(*CHIME_PITCH_MULT)),
                "time_mult": float(rng.uniform(*CHIME_TIME_MULT)), "tau_sec": float(rng.uniform(*CHIME_TAU))}
    return {"v23": True, "kind": kind, "freq": float(rng.uniform(*BACKUP_FREQ[kind])),
            "rate_per_min": float(rng.uniform(*BACKUP_RATE_PER_MIN)), "duty": float(rng.uniform(*BACKUP_DUTY))}


def make_backup_beep_v23(duration_sec: float, fs: int, t_on: float, params: dict, peak: float = 0.9) -> np.ndarray:
    """バック音のドライ音。先頭の音が t_on（秒・クリップの絶対時刻）に始まる。t_on より前は無音。"""
    n = int(round(duration_sec * fs))
    x = np.zeros(n)
    kind = params["kind"]
    if kind in ("speaker", "piezo"):
        period = 60.0 / float(params["rate_per_min"]); on = period * float(params["duty"])
        k_on = int(round(on * fs)); k_ramp = max(2, int(BACKUP_RAMP_SEC * fs))
        env1 = np.ones(k_on)
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k_ramp) / k_ramp)
        env1[:k_ramp] = ramp; env1[-k_ramp:] = ramp[::-1]
        tt = np.arange(k_on) / fs
        beep = np.sin(2.0 * np.pi * float(params["freq"]) * tt) * env1
        k = 0
        while True:
            i0 = int(round((t_on + k * period) * fs))
            if i0 >= n:
                break
            m = min(k_on, n - i0)
            x[i0:i0 + m] += beep[:m]
            k += 1
    else:
        pm, tm, tau = float(params["pitch_mult"]), float(params["time_mult"]), float(params["tau_sec"])
        f1, f2 = CHIME_PING_HZ * pm, CHIME_PONG_HZ * pm
        gap, period = CHIME_GAP_SEC * tm, CHIME_PERIOD_SEC * tm
        m_tone = int(6.0 * tau * fs)
        tt = np.arange(m_tone) / fs
        att = np.minimum(1.0, tt / 0.003)
        h2 = 10 ** (CHIME_H2_DB / 20.0)

        def strike(f: float) -> np.ndarray:
            return (np.sin(2 * np.pi * f * tt) + h2 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / tau) * att
        s1, s2 = strike(f1), strike(f2)
        k = 0
        while True:
            t1 = t_on + k * period
            i1 = int(round(t1 * fs))
            if i1 >= n:
                break
            m = min(m_tone, n - i1); x[i1:i1 + m] += s1[:m]
            i2 = int(round((t1 + gap) * fs))
            if i2 < n:
                m = min(m_tone, n - i2); x[i2:i2 + m] += s2[:m]
            k += 1
    pk = float(np.max(np.abs(x)))
    return peak * x / pk if pk > 0 else x


# ---------------------------------------------------------------- ベルの音の高さ（JBPI 2023 表 3・1 回目・Hz）
BELL_SAMPLES = {
    # 引きベル 8 種 22 個
    "a": (2657, 2617, 2737, 2762, 2721, 2657, 2689, 2745, 2785, 2649, 2705, 2790),
    "b": (2185, 2377), "c": (3337, 3536), "e": (2569,), "f": (2553,), "h": (2545,), "j": (2449,), "m": (2601, 2641),
    # 単打ベル 4 種 5 個
    "d": (2457, 2505), "i": (2929,), "k": (2769,), "l": (2633,),
    # スプリングベル 1 種 1 個
    "g": (2073,),
}
BELL_PULL_SPECIES = ("a", "b", "c", "e", "f", "h", "j", "m")
BELL_SINGLE_SPECIES = ("d", "i", "k", "l")
BELL_SPRING_SPECIES = ("g",)
BELL_TYPE_SPECIES = {"ring": BELL_PULL_SPECIES,                                           # チリリン型 ↔ 引きベル
                     "pin": BELL_SINGLE_SPECIES,                                          # ピン型 ↔ 単打ベル
                     "ding": BELL_PULL_SPECIES + BELL_SINGLE_SPECIES + BELL_SPRING_SPECIES}   # ディンディン型 ↔ 13 種全部
BELL_JITTER = (0.97, 1.03)


def bell_f0_v23(kind: str, rng: np.random.Generator) -> tuple[float, str]:
    """(f0 [Hz], 試料の印 '<種類>-<番号>')。種類 → 試料 → ±3%。"""
    species = BELL_TYPE_SPECIES[kind]
    sp = species[int(rng.integers(0, len(species)))]
    vals = BELL_SAMPLES[sp]
    j = int(rng.integers(0, len(vals)))
    return float(vals[j]) * float(rng.uniform(*BELL_JITTER)), f"{sp}-{j + 1}"


def bell_f0_range(kind: str) -> tuple[float, float]:
    vals = [v for sp in BELL_TYPE_SPECIES[kind] for v in BELL_SAMPLES[sp]]
    return min(vals) * BELL_JITTER[0], max(vals) * BELL_JITTER[1]


# ---------------------------------------------------------------- 検査の道具（音の検査でも使う）
def sounding_segments(x: np.ndarray, fs: int, rel_db: float = -30.0, frame_sec: float = 0.002, min_gap_sec: float = 0.05) -> list:
    """包絡（2 ms RMS）がピークから rel_db 以内の区間を [(開始秒, 終了秒), …] で返す（min_gap_sec より短い切れ目はつなぐ）。"""
    n = int(frame_sec * fs); m = len(x) // n
    rms = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1)) + 1e-12
    db = 20 * np.log10(rms); on = db >= db.max() + rel_db
    segs, start = [], None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segs.append([start * frame_sec, i * frame_sec]); start = None
    if start is not None:
        segs.append([start * frame_sec, m * frame_sec])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= min_gap_sec:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged]


def spectral_centroid_hz(x: np.ndarray, fs: int, f_lo: float = 100.0, f_hi: float = 10000.0) -> float:
    X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.sum(f[m] * X[m]) / (np.sum(X[m]) + 1e-12))


def peak_frequency_hz(x: np.ndarray, fs: int, f_lo: float = 100.0, f_hi: float = 12000.0) -> float:
    X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    return float(f[m][int(np.argmax(X[m]))])


def selftest() -> int:
    fs = 48000
    ok = []
    rng = np.random.default_rng(3)
    # 1 クラクション: 押しの列（絶対時刻）どおりに鳴り、押しの間は無音
    presses = [(1.0, 0.3), (1.7, 0.5)]
    x = make_horn_v23(4.0, fs, rng, presses, f_lo=410.0, f_hi=500.0, alpha=0.7, onset_sec=0.01, release_tau_sec=0.02)
    segs = sounding_segments(x, fs)
    ok.append(("horn: 押し 2 回 → 鳴る区間 2 つ", len(segs) == 2))
    ok.append(("horn: 各区間の始まりが押しの開始 ±10 ms", len(segs) == 2 and all(abs(segs[i][0] - presses[i][0]) <= 0.010 for i in range(2))))
    ok.append(("horn: 各区間の終わりは押しの終わり + 尾（≤ 0.15 s）", len(segs) == 2 and all(0 <= segs[i][1] - (presses[i][0] + presses[i][1]) <= 0.15 for i in range(2))))
    gap_rms = float(np.sqrt(np.mean(x[int(1.45 * fs): int(1.65 * fs)] ** 2)))
    ok.append(("horn: 押しの間は無音（ピークの −50 dB 以下）", gap_rms < 10 ** (-50 / 20) * np.max(np.abs(x))))
    x1 = make_horn_v23(4.0, fs, rng, [(2.0, 1.1)], f_lo=410.0, f_hi=None, alpha=0.7)
    ok.append(("horn: 止まった車の 1 押し（窓 1.1 s）は 1 つの連続音", len(sounding_segments(x1, fs)) == 1))
    cents = []
    for a in (HORN_ALPHA[0], 0.7, HORN_ALPHA[1]):
        for f_hi in (None, 500.0):
            y = make_horn_v23(2.0, fs, np.random.default_rng(1), [(0.2, 1.5)], f_lo=410.0, f_hi=f_hi, alpha=a, noise_mix=0.0)
            cents.append(spectral_centroid_hz(y[int(0.3 * fs): int(1.6 * fs)], fs))
    ok.append((f"horn: スペクトル重心が Lemaitre 2009 の 1,380〜3,790 Hz に入る（α の両端・単音と 2 音）: {[round(c) for c in cents]}", all(1380 <= c <= 3790 for c in cents)))
    # 押しの列の割合と長さ
    rng2 = np.random.default_rng(5); cnt = {}; bad = 0
    for _ in range(20000):
        pl = horn_press_plan(rng2); cnt[pl["type"]] = cnt.get(pl["type"], 0) + 1
        lo, hi = HORN_PRESS_SHORT if pl["kind"] == "short" else HORN_PRESS_LONG
        if len(pl["presses"]) != pl["n"] or not all(lo - 1e-9 <= L <= hi + 1e-9 for _, L in pl["presses"]):
            bad += 1
        for (s0, L0), (s1, _) in zip(pl["presses"], pl["presses"][1:]):
            if not (HORN_GAP[0] - 1e-6 <= s1 - (s0 + L0) <= HORN_GAP[1] + 1e-6):
                bad += 1
    frac = {t[0]: cnt.get(t[0], 0) / 20000 for t in HORN_TYPES}
    ok.append(("horn: 6 型の割合が Takada の値 ±1.5 pt（20,000 回）", all(abs(frac[t[0]] - t[3]) <= 0.015 for t in HORN_TYPES)))
    ok.append(("horn: 押しの長さと間が決めた幅の中", bad == 0))
    n1 = sum(frac[t[0]] for t in HORN_TYPES if t[1] == 1); n2 = sum(frac[t[0]] for t in HORN_TYPES if t[1] == 2)
    ok.append((f"horn: 回数に直すと 1 回 56.2%・2 回 36.3%（±1.5 pt）: {n1:.3f}/{n2:.3f}", abs(n1 - 0.562) <= 0.015 and abs(n2 - 0.363) <= 0.015))
    # 2 バック音: 先頭の音が t_on に合う・速さ・チャイムの 2 音
    p_sp = {"kind": "speaker", "freq": 1200.0, "rate_per_min": 75.0, "duty": 0.5}
    xb = make_backup_beep_v23(10.0, fs, 0.6, p_sp)
    segs = sounding_segments(xb, fs, rel_db=-20.0)
    ok.append(("backup speaker: 先頭の音が t_on = 0.6 s（±5 ms）", segs and abs(segs[0][0] - 0.6) <= 0.005))
    starts = [s for s, _ in segs]
    rate = 60.0 / float(np.median(np.diff(starts)))
    ok.append((f"backup speaker: 毎分 75 回（測定 {rate:.1f}）・鳴る割合 0.5（測定 {np.mean([e - s for s, e in segs]) / (60 / 75):.2f}）",
               abs(rate - 75.0) <= 1.0 and abs(np.mean([e - s for s, e in segs]) / (60 / 75) - 0.5) <= 0.03))
    ok.append(("backup speaker: t_on より前は無音", float(np.max(np.abs(xb[: int(0.59 * fs)]))) == 0.0))
    ok.append(("backup speaker: 山の周波数 1,200 Hz（±1%）", abs(peak_frequency_hz(xb, fs) - 1200.0) <= 12.0))
    p_pz = {"kind": "piezo", "freq": 2777.0, "rate_per_min": 81.0, "duty": 0.58}
    xp = make_backup_beep_v23(5.0, fs, 1.4, p_pz)
    segs = sounding_segments(xp, fs, rel_db=-20.0)
    ok.append(("backup piezo: t_on = 1.4 s の途中からではなく頭から始まる（±5 ms）", segs and abs(segs[0][0] - 1.4) <= 0.005))
    p_ch = {"kind": "chime", "pitch_mult": 1.0, "time_mult": 1.0, "tau_sec": 0.2}
    xc = make_backup_beep_v23(6.0, fs, 0.6, p_ch)
    seg_c = sounding_segments(xc, fs, rel_db=-25.0, min_gap_sec=0.02)
    ok.append(("chime: 先頭のピンが t_on = 0.6 s（±5 ms）", seg_c and abs(seg_c[0][0] - 0.6) <= 0.005))
    ping = xc[int(0.60 * fs): int(0.84 * fs)]; pong = xc[int(0.86 * fs): int(1.10 * fs)]
    ok.append((f"chime: ピン 2,000 Hz → ポーン 1,590 Hz（測定 {peak_frequency_hz(ping, fs):.0f}／{peak_frequency_hz(pong, fs):.0f}）",
               abs(peak_frequency_hz(ping, fs) - 2000) <= 20 and abs(peak_frequency_hz(pong, fs) - 1590) <= 20))
    groups = [s for s, _ in seg_c]          # ピンの尾（時定数 0.2 s）がポーンに重なるので、1 組 = 1 区間
    ok.append((f"chime: 組のくり返し 1.25 s（測定 {np.median(np.diff(groups)) if len(groups) >= 3 else -1:.3f}）",
               len(groups) >= 3 and abs(float(np.median(np.diff(groups))) - 1.25) <= 0.01))
    # 3 ベルの高さ: 種類 → 試料 → ±3%・範囲・種類の出やすさ
    rng3 = np.random.default_rng(9)
    for kind in ("ring", "pin", "ding"):
        lo, hi = bell_f0_range(kind); draws = [bell_f0_v23(kind, rng3) for _ in range(6000)]
        ok.append((f"bell {kind}: 6,000 回とも幅 {lo:.0f}〜{hi:.0f} Hz の中・試料の印つき", all(lo - 1e-6 <= f <= hi + 1e-6 and "-" in sid for f, sid in draws)))
        sp_cnt = {}
        for _, sid in draws:
            sp_cnt[sid.split("-")[0]] = sp_cnt.get(sid.split("-")[0], 0) + 1
        n_sp = len(BELL_TYPE_SPECIES[kind])
        ok.append((f"bell {kind}: {n_sp} 種類が同じ確率（各 1/{n_sp} ±3 pt）", all(abs(sp_cnt.get(sp, 0) / 6000 - 1 / n_sp) <= 0.03 for sp in BELL_TYPE_SPECIES[kind])))
    a_frac = sum(1 for f, sid in [bell_f0_v23("ring", rng3) for _ in range(6000)] if sid.startswith("a-")) / 6000
    ok.append((f"bell ring: a 種（12 個）が 1/8（{a_frac:.3f}）で、28 個から選んだときの 12/22 にならない", abs(a_frac - 0.125) <= 0.03))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(selftest())
