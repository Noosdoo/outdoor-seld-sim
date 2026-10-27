"""自転車の走行音（ベルなし）のドライ音源 — クラス bicycle の試作（2026-10-07・本人「1 回作成してみて。試聴する」）。

舞台は日本の普通の自転車（軽快車・ママチャリ・通学用）。電動アシストは入れない（本人 10/7 03:27）。

根拠と限界（2026-10-07 の文献調べ・scratchpad 先行研究_自転車の*_2026-10-07.md）:
- 日本の自転車の走行音を「何 m・何 km/h・何 dB」で測った公開資料は無い（環境省の騒音規制は自動車・原付だけで自転車は対象外）。
  スペクトルを測った文献も無い。→ ここに書く数値は **設計値** で、採用時は変更票に「設計値・出典なし」と明記する。
- 速さ: 歩道の徐行 4〜5 km/h（警察庁答弁）・交差点進入の平均 11.1 km/h（松井ら 2016）。場面側で 4〜25 km/h を引く。
- 空転（フリーホイール）のクリック: 漕いでいないときだけ出る。繰り返し周波数 f = 歯数 × 車輪の回転数（26 インチ周長 2.07 m・歯数 18〜36）。
  15 km/h・24 歯で約 48 Hz。
- ケイデンス（漕ぐ回転数）: 移動用で 60〜80 rpm。漕いでいる区間はこれで振幅がゆれる。
- 音の構成（設計）: ①タイヤと路面の広帯域ヒス音（主成分・400〜7,000 Hz・高域はなだらかに落ちる）＋低い転がり音（80〜400 Hz・弱）
  ②漕いでいない区間: 空転クリック列（3〜9 kHz の短い打音）③漕いでいる区間: かすかな駆動音（チェーン・800〜3,000 Hz・ケイデンスで振幅変調）
  ④カゴ・泥除けのガタつき（まれ・既定はオフ）。
- 絶対レベル（dB(A)@1 m）はここでは決めない。描画側（step11）が l1m_db を当てる（キック・車と同じ流れ）。速さによる増減は場面側の
  レベルの引き方で扱う（タイヤ音の 30 log10(v) を仮に使う）。

使い方: make_bicycle(duration_sec, fs, rng, speed_mps=4.2, coast_frac=0.0, teeth=24, cadence_rpm=70, rattle_rate=0.0)
既定は空転クリック無し（coast_frac=0.0・本人 10/7 03:36「チチチは要らない・現実もあまり聞こえない」）。必要なら coast_frac で戻せる。
戻り値はピーク正規化したドライ波形（他の make_* と同じ）。
"""
from __future__ import annotations

import numpy as np

from .calibration import a_weighted_rms

WHEEL_CIRC_M = 2.07          # 26 インチ軽快車の周長（仮定・文献なし）
V_REF_MPS = 15.0 / 3.6       # レベルの基準の速さ（場面側で使う）
TEETH_RANGE = (18, 36)       # ラチェットの歯数（一般的なハブ・要確認）
CADENCE_RANGE = (60.0, 80.0)  # rpm


def _shaped_noise(n: int, fs: int, rng: np.random.Generator, lo: float, hi: float,
                  tilt_db_per_oct: float = 0.0) -> np.ndarray:
    """lo〜hi Hz の帯域雑音（端は半オクターブのコサインで落とす）。tilt は lo を基準にした高域の傾き [dB/oct]。標準偏差 1。"""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    shape[(f >= lo) & (f <= hi)] = 1.0
    for edge, sign in ((lo, -1), (hi, +1)):
        a, b = (edge / np.sqrt(2), edge) if sign < 0 else (edge, edge * np.sqrt(2))
        m = (f >= a) & (f < b)
        frac = (np.log2(np.maximum(f[m], 1e-9)) - np.log2(a)) / (np.log2(b) - np.log2(a))
        shape[m] = 0.5 - 0.5 * np.cos(np.pi * (frac if sign < 0 else 1.0 - frac))
    if tilt_db_per_oct:
        shape = shape * 10 ** (tilt_db_per_oct * np.log2(np.maximum(f, 1.0) / max(lo, 1.0)) / 20.0)
    x = np.fft.irfft(X * shape, n=n)
    return x / (np.std(x) + 1e-12)


def _slow_am(n: int, fs: int, rng: np.random.Generator, depth: float, rate_hz: float) -> np.ndarray:
    """路面の粗さ・継ぎ目によるゆっくりした振幅のゆれ（1 ± depth）。"""
    step = max(1, int(fs / rate_hz))
    knots = rng.standard_normal(n // step + 2)
    env = np.interp(np.arange(n), np.arange(len(knots)) * step, knots)
    return 1.0 + depth * np.clip(env, -2.0, 2.0) * 0.5


def _segments(n: int, fs: int, rng: np.random.Generator, coast_frac: float) -> np.ndarray:
    """漕ぐ／空転の区間（True = 空転）。漕ぐ区間 3〜8 秒・空転区間 2〜6 秒を coast_frac に合わせて伸縮。"""
    coast = np.zeros(n, dtype=bool)
    if coast_frac <= 0.0:
        return coast
    if coast_frac >= 1.0:
        coast[:] = True
        return coast
    scale = coast_frac / 0.3
    i = 0
    state = bool(rng.random() < coast_frac)
    while i < n:
        sec = rng.uniform(2.0, 6.0) * scale if state else rng.uniform(3.0, 8.0)
        j = min(n, i + int(sec * fs))
        coast[i:j] = state
        i = j
        state = not state
    return coast


def _click_train(n: int, fs: int, rng: np.random.Generator, f_click: float, mask: np.ndarray) -> np.ndarray:
    """空転のクリック列: 3〜9 kHz の短い打音（約 1.2 ms・指数減衰）を f_click [Hz] で並べ、mask が True の区間だけ鳴らす。"""
    x = np.zeros(n)
    if f_click <= 0.0 or not mask.any():
        return x
    k_len = int(0.0012 * fs)
    t = np.arange(k_len) / fs
    kernel = _shaped_noise(k_len, fs, rng, 3000.0, 9000.0) * np.exp(-t / 0.0004)
    kernel = kernel / (np.max(np.abs(kernel)) + 1e-12)
    period = fs / f_click
    pos = 0.0
    while pos < n:
        p = int(pos)
        if p < n and mask[p]:
            amp = 1.0 + 0.2 * rng.standard_normal()
            e = min(n, p + k_len)
            x[p:e] += amp * kernel[: e - p]
        pos += period * (1.0 + 0.01 * rng.standard_normal())
    return x


def _rattle(n: int, fs: int, rng: np.random.Generator, rate_per_s: float) -> np.ndarray:
    """カゴ・泥除けのガタつき（150〜400 Hz の減衰音・まれ）。"""
    x = np.zeros(n)
    if rate_per_s <= 0.0:
        return x
    k_len = int(0.06 * fs)
    t = np.arange(k_len) / fs
    n_ev = rng.poisson(rate_per_s * n / fs)
    for _ in range(int(n_ev)):
        f0 = rng.uniform(150.0, 400.0)
        kernel = np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.015)
        p = int(rng.uniform(0, max(1, n - k_len)))
        x[p:p + k_len] += kernel * rng.uniform(0.5, 1.0)
    return x


def make_bicycle(duration_sec: float, fs: int, rng: np.random.Generator,
                 speed_mps: float = V_REF_MPS, coast_frac: float = 0.0,
                 teeth: int = 24, cadence_rpm: float = 70.0,
                 rattle_rate: float = 0.0, peak: float = 0.9) -> np.ndarray:
    """普通の自転車の走行音（ベルなし・ドライ音源）。

    speed_mps: 走る速さ（場面側で 4〜25 km/h = 1.1〜6.9 m/s を引く）。coast_frac: 漕いでいない時間の割合の目安（0 で常に漕ぐ・1 で常に空転）。
    teeth: ラチェットの歯数。cadence_rpm: 漕ぐ回転数。rattle_rate: ガタつきの頻度 [回/秒]（既定 0 = 無し）。
    """
    n = int(round(duration_sec * fs))
    v = max(float(speed_mps), 0.3)
    # ① タイヤと路面のヒス音（主成分）＋ 低い転がり音
    hiss = _shaped_noise(n, fs, rng, 400.0, 7000.0, tilt_db_per_oct=-3.0)
    rumble = _shaped_noise(n, fs, rng, 80.0, 400.0)
    tire = hiss + 0.35 * rumble
    tire = tire * _slow_am(n, fs, rng, depth=0.25, rate_hz=3.0)
    tire = tire / (np.std(tire) + 1e-12)
    coast = _segments(n, fs, rng, coast_frac)
    # ② 空転のクリック列（漕いでいない区間だけ）: f = 歯数 × 車輪の回転数
    f_click = float(teeth) * v / WHEEL_CIRC_M
    clicks = _click_train(n, fs, rng, f_click, coast)
    # ③ 漕いでいる区間のかすかな駆動音（ケイデンスで 2 回/回転の振幅変調）
    f_ped = 2.0 * float(cadence_rpm) / 60.0
    t = np.arange(n) / fs
    chain = _shaped_noise(n, fs, rng, 800.0, 3000.0) * (1.0 + 0.25 * np.sin(2 * np.pi * f_ped * t))
    chain = chain * (~coast)
    # ④ ガタつき（既定オフ）
    rattle = _rattle(n, fs, rng, rattle_rate)
    # A 特性の実効値で混ぜる: タイヤ 1.0・クリック 0.5・駆動音 0.3・ガタつき 0.4（設計値）
    x = np.zeros(n)
    for sig, w in ((tire, 1.0), (clicks, 0.5), (chain, 0.3), (rattle, 0.4)):
        r = a_weighted_rms(sig, fs)
        if r > 0:
            x = x + (w / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
