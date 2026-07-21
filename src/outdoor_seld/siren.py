"""合成サイレン音源（ドライ信号）の生成。

救急車の wail 型サイレン: 基本周波数が 650-1450 Hz を周期 ~4.8s で往復する
正弦スイープ＋倍音2つ。位相積分で生成するためスイープが滑らか。
ドップラーがスペクトログラム上で見やすいトーン構造を持つ。
"""
from __future__ import annotations

import numpy as np


def make_peepo_siren(duration_sec: float, fs: int, f_hi: float = 960.0,
                     f_lo: float = 770.0, tone_sec: float = 0.65,
                     cross_sec: float = 0.015, peak: float = 0.9) -> np.ndarray:
    """「ピーポー」型サイレン（日本の救急車: 960/770 Hz を各0.65秒で交互）。

    周波数を短いランプ(15ms)で切り替えて位相積分するのでクリックが出ない。
    音程が階段状のため、ドップラーによる周波数シフトが線の平行移動として
    スペクトログラム上で最も分かりやすい。
    """
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    pos = (t % (2.0 * tone_sec)) / tone_sec   # 0..2 (0-1: ピー, 1-2: ポー)。2音1周期の中の位置
    frac = pos % 1.0                          # 各トーン内での位置 0..1（今のトーンの何%地点か）
    # トーン切り替え直後(cross_sec間)だけ0→1で滑らかに遷移させる比率
    ramp = np.clip(frac / (cross_sec / tone_sec), 0.0, 1.0)
    in_hi = pos < 1.0                          # 前半(0-1)が「ピー」、後半(1-2)が「ポー」
    f_prev = np.where(in_hi, f_lo, f_hi)      # 直前のトーン（切り替わる前の周波数）
    f_cur = np.where(in_hi, f_hi, f_lo)       # いまのトーン（切り替わった後の周波数）
    f_inst = f_prev + (f_cur - f_prev) * ramp  # rampで前後を線形補間＝瞬時周波数
    # 周波数を積分して位相にする（瞬時周波数が変化しても位相が連続でクリックが出ない）
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = (1.00 * np.sin(phase)              # 基本波
         + 0.50 * np.sin(2.0 * phase)      # 第2倍音
         + 0.25 * np.sin(3.0 * phase))     # 第3倍音
    fade = int(0.01 * fs)                   # 10msのフェード長
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)   # クリップ先頭のフェードイン
    env[-fade:] = np.linspace(1.0, 0.0, fade)  # クリップ末尾のフェードアウト（クリック防止）
    x = x * env
    return peak * x / np.max(np.abs(x))         # ピーク値をpeakに正規化して返す


def make_siren(duration_sec: float, fs: int, f_lo: float = 650.0,
               f_hi: float = 1450.0, sweep_period_sec: float = 4.8,
               peak: float = 0.9, seed: int = 0) -> np.ndarray:
    """wail サイレンのモノラル信号を返す (float64, peak 正規化)。"""
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    f_center = 0.5 * (f_lo + f_hi)   # 周波数スイープの中心
    f_dev = 0.5 * (f_hi - f_lo)      # 中心からの振れ幅
    # 基本周波数の時間変化（正弦LFOでf_loとf_hiの間をなめらかに往復）
    f_inst = f_center + f_dev * np.sin(2.0 * np.pi * t / sweep_period_sec - np.pi / 2)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs   # 瞬時周波数を積分して連続位相にする
    x = (1.00 * np.sin(phase)
         + 0.50 * np.sin(2.0 * phase)
         + 0.25 * np.sin(3.0 * phase))
    # ごく短いフェードイン/アウト（クリック防止）
    fade = int(0.01 * fs)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    x = x * env
    return peak * x / np.max(np.abs(x))


def make_fire_siren(duration_sec: float, fs: int, f_lo: float = 361.0,
                    f_hi: float = 710.0, sweep_period_sec: float = 2.97,
                    bell_f0: float = 1150.0, bell_gap1_sec: float = 0.50,
                    bell_gap2_sec: float = 0.50, bell_decay_sec: float = 0.24,
                    bell_phase_frac: float = 0.58, bell_mix: float = 0.3,
                    peak: float = 0.9,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """消防車サイレン: 「ウー」(wail)は止めず鳴らし続け、その上に「カン・カン・カン」
    （警鐘=半鐘、3点鐘）を周期的に重ねる（2026-07-21追加、本人指摘「ちりんの音も
    あるはず」→さらに「現実と違う」との指摘を受け、実録音を周波数解析して再設計）。

    **全パラメータの既定値は実測に基づく**（[OtoLogic](https://otologic.jp/free/se/transportation02.html)
    「消防車　サイレン01」、CC BY 4.0、5本のDry版クリップをlibrosa.pyinのf0追跡＋
    帯域エネルギー解析でクロス検証、2026-07-21）。当初の想定（ウーと鐘を交互・鐘750Hz・
    周期6-8s）も、その後の1回目の再測定（打撃間隔0.50s/0.81sの不均等）も実測と異なり、
    複数クリップでの再検証で確定: ①ウーは止まらず連続②鐘は約1150Hz（自転車ベルより
    低いがウーよりずっと高い）③鐘は3点鐘で**打撃間隔はほぼ均等に約0.50s**
    （0.505/0.511s・0.505/0.459s等、複数クリップで一貫）④ウーの掃引周期は約2.97sで鐘バーストと
    ほぼ同期（トラフから約58%位相の位置でバーストが始まる）、と判明したため全面改訂した。

    **bell_mix=0.3（2026-07-21、v10.1本番生成で発覚した安全域調整）**: この規約は
    絶対較正（gain_for_spl_aによるA特性RMS一致）を使うため、ゲインは信号のRMSを
    目標dB SPLに合わせて決まり、**クレスト比（ピーク/RMS）が高いほど較正後の
    実ピークも高くなる**。鐘を実測どおりのフル振幅で重ねるとクレスト比が
    ウー単体(約1.7)の2倍(約3.5)に達し、他音源・反射との組み合わせで
    peak<0.99の安全アサートを稀に破る実例が発生した（fold1_room1_mix1381で
    peak=1.013）。bell_mix=0.3でクレスト比を約2.3に抑え、実用上の安全マージンを
    確保する（鐘の可聴性は保ったまま）。
    """
    n = int(round(duration_sec * fs))
    if rng is None:
        rng = np.random.default_rng(0)

    def bell_strike(m: int) -> np.ndarray:
        tt = np.arange(m) / fs
        env = np.exp(-tt / bell_decay_sec)
        # 鐘らしい非整数次の部分音（make_bike_bell系と同じ加算合成の考え方、
        # ただし半鐘は自転車ベルより低く重い音を想定）
        partials = [(1.00, 1.00), (2.00, 0.35), (2.76, 0.22), (4.10, 0.10)]
        tone = np.zeros(m)
        for ratio, amp in partials:
            tone += amp * np.sin(2.0 * np.pi * bell_f0 * ratio * tt)
        # 打撃アタック: ごく短い金属質ノイズ（make_crossing_v2と同じ手法）
        ns = min(int(0.006 * fs), m)
        if ns > 0:
            click = rng.standard_normal(ns)
            spec = np.fft.rfft(click)
            fgrid = np.fft.rfftfreq(ns, 1.0 / fs)
            spec[(fgrid < 800) | (fgrid > 3000)] = 0.0
            click = np.fft.irfft(spec, n=ns)
            click = click / (np.max(np.abs(click)) + 1e-12)
            tone[:ns] += 10 ** (-18.0 / 20) * click * np.linspace(1, 0, ns)
        return tone * env * bell_mix

    # --- ウー(wail)は全長で連続して鳴らす ---
    t = np.arange(n) / fs
    f_center = 0.5 * (f_lo + f_hi)
    f_dev = 0.5 * (f_hi - f_lo)
    # sin(2*pi*t/T - pi/2) はt=0でトラフ（実測のトラフ基準に合わせる）
    f_inst = f_center + f_dev * np.sin(2.0 * np.pi * t / sweep_period_sec - np.pi / 2)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = (1.00 * np.sin(phase) + 0.50 * np.sin(2.0 * phase)
         + 0.25 * np.sin(3.0 * phase))
    fade = int(0.01 * fs)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    x = x * env

    # --- カン・カン・カン(警鐘3点鐘)をウー周期に同期して重ねる ---
    ts = bell_phase_frac * sweep_period_sec
    while ts < duration_sec:
        for gap in (0.0, bell_gap1_sec, bell_gap1_sec + bell_gap2_sec):
            i0 = int((ts + gap) * fs)
            m = n - i0
            if m <= 0:
                continue
            x[i0:] += bell_strike(m)
        ts += sweep_period_sec

    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x
