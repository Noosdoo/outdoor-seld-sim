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
