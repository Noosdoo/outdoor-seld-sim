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
