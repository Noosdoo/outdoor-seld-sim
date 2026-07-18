"""クラクション・バック警告音・自転車ベルの合成（v5マルチクラス拡張用）。

siren.py と同じ方針: すべてクリーン合成音源（実録音は使わない）。
理由: 実録音を混ぜるとクリップごとに音源の質が変わり「土俵」の解釈可能性が
崩れるため（car+drone混在版が破棄された理由と同じ、PROGRESS.md参照）。

2026-07-12 改良: 初版（純音の重ね合わせのみ）が実物とかけ離れていたため、
外部調査を踏まえて音色を作り込み直した。
- クラクション: 実物は振動板駆動のリード楽器的な音色（奇数次倍音が強い、
  純音より硬い）で、デュアルトーンは約400/500Hz(3度の音程)が一般的
  （出典: soundcy.com "Mastering Car Horn Sounds"、bosshorn.com）
- 自転車ベル: 非整数次倍音（モード）の集合＋近接デチューンによる「warble」
  （うなり）が特徴。各倍音が独自の減衰時間を持つ
  （出典: DAFX02 Karjalainen "Efficient Modeling And Synthesis Of Bell-Like
  Sounds"、CCRMA "Risset's bell"）
"""
from __future__ import annotations

import numpy as np

from .noise import colored_noise


def make_horn(duration_sec: float, fs: int, rng: np.random.Generator | None = None,
             f_lo: float = 410.0, f_hi: float = 500.0, n_harmonics: int = 5,
             honk_sec: float = 0.35, gap_sec: float = 0.15, ramp_sec: float = 0.02,
             noise_mix: float = 0.06, peak: float = 0.9) -> np.ndarray:
    """車のクラクション: 2音同時（うなり）のリード楽器的な断続音。

    実物のダイアフラム式ホーンは奇数次倍音が強く、純音より硬い/割れた質感を
    持つため、各トーンを奇数次倍音の重ね合わせ（矩形波に近い波形）で作る。
    さらに honk 区間だけわずかに息成分（帯域雑音）を混ぜて機械的な硬さを崩す。
    honk_sec鳴って gap_sec休む、を周期的に繰り返す。
    """
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    period = honk_sec + gap_sec
    phase_pos = t % period                    # 1周期(鳴る+休む)の中の位置
    env = np.zeros(n)
    in_honk = phase_pos < honk_sec
    env[in_honk] = 1.0
    fade_in = phase_pos < ramp_sec             # 鳴り始めのフェード
    env[fade_in] = phase_pos[fade_in] / ramp_sec
    fade_out = (phase_pos >= honk_sec - ramp_sec) & (phase_pos < honk_sec)  # 鳴り終わりのフェード
    env[fade_out] = (honk_sec - phase_pos[fade_out]) / ramp_sec

    def reedy_tone(f: float) -> np.ndarray:
        # 奇数次倍音を1/kで重ねる＝矩形波に近い、リード楽器的な硬い音色
        x = np.zeros(n)
        for k in range(1, n_harmonics * 2, 2):
            x += (1.0 / k) * np.sin(2.0 * np.pi * f * k * t)
        return x

    x = 0.55 * reedy_tone(f_lo) + 0.55 * reedy_tone(f_hi)
    if rng is not None and noise_mix > 0:
        breath = colored_noise(n, fs, rng, slope=0.5, f_lo=200.0)  # 息成分(高域寄り)
        x = (1.0 - noise_mix) * x + noise_mix * breath
    x = x * env
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x


def make_backup_beep(duration_sec: float, fs: int, freq: float = 1000.0,
                     on_sec: float = 0.5, off_sec: float = 0.5,
                     ramp_sec: float = 0.02, peak: float = 0.9) -> np.ndarray:
    """車のバック(後退)警告音: 単一トーンの断続ビープ（on/off等間隔）。

    実物はピエゾブザー由来の素朴な電子音なので、純音のままで質感は近い
    （クラクション・ベルほど作り込みは要らない）。
    """
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs
    period = on_sec + off_sec
    phase_pos = t % period
    env = np.zeros(n)
    in_on = phase_pos < on_sec
    env[in_on] = 1.0
    fade_in = phase_pos < ramp_sec
    env[fade_in] = phase_pos[fade_in] / ramp_sec
    fade_out = (phase_pos >= on_sec - ramp_sec) & (phase_pos < on_sec)
    env[fade_out] = (on_sec - phase_pos[fade_out]) / ramp_sec
    x = np.sin(2.0 * np.pi * freq * t) * env
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x


def make_bike_bell(duration_sec: float, fs: int, f0: float = 3000.0,
                   n_rings: int = 2, ring_gap_sec: float = 0.45,
                   repeat_period_sec: float = 2.0, peak: float = 0.9) -> np.ndarray:
    """自転車のベル: 非整数次倍音＋近接デチューンによる warble を n_rings 回。

    ベル音は調和しない部分音（モード）の集合で、しかも近接した部分音同士が
    わずかにビートする「warble」が特徴（Risset bell 等の加算合成の定番手法）。
    高次の部分音ほど振幅が小さく、減衰も速い。

    n_rings回のチリンを repeat_period_sec ごとにクリップ全体で繰り返す
    （horn/backup_beepと同様に周期的にする。1回だけ鳴らす設計だと、発音区間
    ウィンドウがクリップ後半に来た場合に無音になってしまうバグがあったため
    2026-07-12 修正）。
    """
    n = int(round(duration_sec * fs))
    x = np.zeros(n)
    # (基音との周波数比, 振幅, 減衰時定数[s])。1.00と1.003の対がwarble(うなり)を作る
    partials = [
        (1.000, 1.00, 0.35), (1.003, 0.85, 0.30),
        (2.40, 0.45, 0.18),
        (2.70, 0.30, 0.15),
        (3.80, 0.22, 0.10),
        (5.30, 0.12, 0.07),
        (6.80, 0.06, 0.05),
    ]
    n_repeats = int(np.ceil(duration_sec / repeat_period_sec))
    for r in range(n_repeats):
        base_sec = r * repeat_period_sec
        for k in range(n_rings):
            i0 = int((base_sec + k * ring_gap_sec) * fs)
            if i0 >= n:
                break
            tt = np.arange(n - i0) / fs
            tone = np.zeros(n - i0)
            for ratio, amp, tau in partials:
                tone += amp * np.exp(-tt / tau) * np.sin(2.0 * np.pi * f0 * ratio * tt)
            x[i0:] += tone
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x


def make_crossing(duration_sec: float, fs: int,
                  f_a: float = 700.0, f_b: float = 750.0,
                  rate_per_min: float = 130.0, h2_db: float = -15.0,
                  fade_sec: float = 0.01, peak: float = 0.9) -> np.ndarray:
    """踏切警報音（v9新規、設計= out/v9_design_v2_2026-07-16.md 3節）。

    規格準拠（out/v9_values_research_2026-07-16.md）: 700/750Hz±15Hzの2音交互、
    毎分130±5回（=1音の長さ 60/rate 秒）、電子音らしさとして第2倍音を h2_db で足す。
    列車接近中は鳴りやまない現実に合わせ、クリップ全長で鳴る（発音窓なし）。
    """
    n = int(round(duration_sec * fs))
    tone_len = 60.0 / rate_per_min          # 1音の長さ[s]（この後で交互に切り替わる）
    spt = int(round(tone_len * fs))
    x = np.zeros(n)
    fade = int(fade_sec * fs)
    env = np.ones(spt)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    k = 0
    for i0 in range(0, n, spt):
        f = f_a if k % 2 == 0 else f_b
        m = min(spt, n - i0)
        tt = np.arange(m) / fs
        tone = np.sin(2.0 * np.pi * f * tt) \
            + 10.0 ** (h2_db / 20.0) * np.sin(2.0 * np.pi * 2.0 * f * tt)
        x[i0:i0 + m] = tone * env[:m]
        k += 1
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x


def make_crossing_v2(duration_sec: float, fs: int,
                     f_a: float = 700.0, f_b: float = 750.0,
                     rate_per_min: float = 130.0,
                     h2_db: float = -14.0, h3_db: float = -22.0,
                     detune: float = 1.004, detune_db: float = -6.0,
                     decay_tau_sec: float = 0.30,
                     strike_noise_db: float = -20.0,
                     strike_noise_sec: float = 0.008,
                     peak: float = 0.9,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """踏切警報音 改訂版（2026-07-17。make_crossing=v9凍結版は不変更）。

    実物の構造（鉄道総研 人間科学ニュース200号・Wikipedia踏切警報機、
    生録音との試聴比較で改訂）:
      - 700Hz±15と750Hz±15は【同時に鳴る和音】（半音違いの濁った不協和音）
      - 【毎分130±5回】は打撃（カン）の繰り返し。電鐘（金属打音）由来なので
        各打は鐘のように励起→指数減衰し、**余韻は次の打まで残る**（ゲートで
        切らない。初版v2のオン/オフゲートは切り際が不自然だった）
      - 打撃瞬間の金属アタック（短い広帯域ノイズ）と、ベル系のデチューン
        （わずかにずれた対でうなる）を付けて電鐘らしさを出す
    """
    n = int(round(duration_sec * fs))
    x = np.zeros(n)
    period = 60.0 / rate_per_min
    if rng is None:
        rng = np.random.default_rng(0)

    def chord(tt):
        y = np.zeros(len(tt))
        for f in (f_a, f_b):
            y += np.sin(2 * np.pi * f * tt)
            y += 10 ** (detune_db / 20) * np.sin(2 * np.pi * f * detune * tt)
            y += 10 ** (h2_db / 20) * np.sin(2 * np.pi * 2 * f * tt)
            y += 10 ** (h3_db / 20) * np.sin(2 * np.pi * 3 * f * tt)
        return y

    ts = 0.0
    while ts < duration_sec:
        i0 = int(ts * fs)
        m = n - i0
        if m <= 0:
            break
        tt = np.arange(m) / fs
        env = np.exp(-tt / decay_tau_sec)
        na = min(int(0.003 * fs), len(env))   # 終端間際の打撃はランプも切り詰める
        env[:na] *= np.linspace(0.0, 1.0, int(0.003 * fs))[:na]
        strike = chord(tt) * env
        # 打撃アタック: ごく短い金属質ノイズ（1〜4kHz帯）
        ns = int(strike_noise_sec * fs)
        click = rng.standard_normal(ns)
        spec = np.fft.rfft(click)
        fgrid = np.fft.rfftfreq(ns, 1.0 / fs)
        spec[(fgrid < 1000) | (fgrid > 4000)] = 0.0
        click = np.fft.irfft(spec, n=ns)
        click = click / (np.max(np.abs(click)) + 1e-12)
        # クリップ終端間際の打撃はアタックノイズが残り長を超えることがある（切り詰める）
        ne = min(ns, len(strike))
        strike[:ne] += (10 ** (strike_noise_db / 20)
                        * click[:ne] * np.linspace(1, 0, ns)[:ne])
        x[i0:] += strike
        ts += period
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x


def make_bike_bell_ring(duration_sec: float, fs: int, f0: float = 3000.0,
                        strike_rate_hz: float = 30.0, burst_sec: float = 0.65,
                        gap_sec: float = 0.6, peak: float = 0.9) -> np.ndarray:
    """自転車ベルの引き打ち（回転）式「チリリンチリリン」（2026-07-17追加）。

    JIS D 9451 は引き打ちベル・単打ベル・スプリングベルを規定。既存の
    make_bike_bell=単打「チーン」に対し、本関数はレバーを引く間だけ小刻みに
    連打される音（連打中は互いに減衰を打ち切るため各打の減衰は短い）。
    部分音は make_bike_bell と同系（同じベルを速く叩いた音）。
    """
    n = int(round(duration_sec * fs))
    x = np.zeros(n)
    partials = [
        (1.00, 1.00, 0.30), (2.40, 0.45, 0.18), (2.70, 0.30, 0.15),
        (3.80, 0.22, 0.10), (5.30, 0.12, 0.07), (6.80, 0.06, 0.05),
    ]
    tau_scale = 0.35   # 連打で減衰が打ち切られるぶん短く
    t_cursor = 0.0
    while t_cursor < duration_sec:
        burst_end = min(t_cursor + burst_sec, duration_sec)
        ts = t_cursor
        while ts < burst_end:
            i0 = int(ts * fs)
            m = n - i0
            if m <= 0:
                break
            tt = np.arange(m) / fs
            tone = np.zeros(m)
            for ratio, amp, tau in partials:
                tone += amp * np.exp(-tt / (tau * tau_scale)) \
                    * np.sin(2.0 * np.pi * f0 * ratio * tt)
            x[i0:] += tone
            ts += 1.0 / strike_rate_hz
        t_cursor = burst_end + gap_sec
    peak_val = np.max(np.abs(x))
    return peak * x / peak_val if peak_val > 0 else x
