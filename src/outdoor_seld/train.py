"""v12列車走行音源＋警笛（踏切クラスの音源拡張、日本の在来線準拠）。

根拠（2026-08-05リサーチ＋実録参照分析）:
- 文献: 転動音の主成分500〜2,000Hz・継目衝撃+5〜8dB（環境省 在来鉄道騒音測定
  マニュアル系）/ 通過騒音 実測82〜87dB@12.5m（東京都調査）
- **実録参照の実測**（soundeffect-lab.info の電車通過・走行3本を分析。本人指定
  2026-08-05。録音は較正の参照のみ・データセットには不使用=クリーン合成方針）:
  - オクターブ形状: **125〜250Hz峰**、63-125:-5 / 250-500:-1.5 / 500-1k:-4.5 /
    1-2k:-8 / 2-4k:-10.5 / 31-63:-12 dB（3本平均の丸め）
  - 打撃リズム: 連打0.05〜0.15s＋クラスタ間隔0.6〜0.7s（軸幾何モデルと整合）
  - 警笛: 空気笛=基本約317Hz+倍音列 / 電子ホーン=900〜2400Hz帯に倍音エネルギー
- 幾何: 定尺レール25m・車両長約20m・台車中心=車端から約3m・軸距2.1m
- 警笛の運用法規: 気笛合図（危険警告=短急数声・接近通知=長緩一声）、警笛吹鳴標識
  （踏切手前での吹鳴指示）。空気式・電子式の2方式（民鉄協会/実施基準）。
  音圧の省令数値は要追補（planレベルはサイレン級の設計値で仮置きし追補時に較正）
- シナリオ: 警報機のない第4種踏切（約2,200カ所・事故率第1種の約2倍、総務省/国交省）

実装規約（v12設計書追記）: 本番レンダは**車両毎の複数点音源**（editable: 4〜8両、
20m間隔・位相結合）＋ラベルは最近接車両の単一トラック。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import a_weighted_rms
from .noise import colored_noise

# 実録参照からの高解像度フィット（scripts/_fit_train_ref.py が生成、2026-08-05）
_REF = json.loads((Path(__file__).parent / "train_ref_params.json")
                  .read_text(encoding="utf-8"))

RAIL_LEN_M = 25.0
CAR_LEN_M = 20.0
BOGIE_AXLE_SPACING_M = 2.1
BOGIE_FROM_END_M = 3.0

# 実録参照3本の平均オクターブ形状（125-250Hz峰=0dB）
TRAIN_ANCHORS_HZ = np.array([31.5, 63.0, 125.0, 250.0, 500.0,
                             1000.0, 2000.0, 4000.0, 8000.0])
TRAIN_ANCHORS_DB = np.array([-12.0, -5.0, 0.0, -1.5, -4.5,
                             -8.0, -10.5, -14.0, -18.0])


def _shaped_noise(n: int, fs: int, rng: np.random.Generator,
                  anchors_hz: np.ndarray, anchors_db: np.ndarray,
                  f_lo: float = 25.0) -> np.ndarray:
    """オクターブ帯アンカー形状の雑音（noise_v12と同じPSD変換規約、単位分散）。"""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(f)
    band = f >= f_lo
    lvl = np.interp(np.log2(np.maximum(f[band], 1e-9)),
                    np.log2(anchors_hz), anchors_db)
    shape[band] = np.sqrt((10.0 ** (lvl / 10.0)) / np.maximum(f[band], 1e-9))
    x = np.fft.irfft(X * shape, n=n)
    return x / np.std(x)


def make_train_passby(duration_sec: float, fs: int, rng: np.random.Generator,
                      speed_mps: float = 20.0,
                      body_frac_a: float = 0.70, joint_frac_a: float = 0.30,
                      peak: float = 0.9) -> np.ndarray:
    """1車両ぶんの走行音（本番は車両毎に本関数を1音源として並べる）。"""
    n = int(round(duration_sec * fs))
    body = _shaped_noise(n, fs, rng,
                         np.array(_REF["body_env"]["freqs"], float),
                         np.array(_REF["body_env"]["db"], float))

    # 継目打撃: 軸位置の実幾何から発生時刻を導出（実録の連打0.05-0.15s/クラスタ0.6-0.7sと整合）
    dt_axle = BOGIE_AXLE_SPACING_M / speed_mps
    axle_offsets = np.array([BOGIE_FROM_END_M - dt_axle * speed_mps / 2.0,
                             BOGIE_FROM_END_M + dt_axle * speed_mps / 2.0,
                             CAR_LEN_M - BOGIE_FROM_END_M - dt_axle * speed_mps / 2.0,
                             CAR_LEN_M - BOGIE_FROM_END_M + dt_axle * speed_mps / 2.0])
    t_rail = RAIL_LEN_M / speed_mps
    env = np.zeros(n)
    tau = float(_REF["impact"]["tau_s"])     # 実測減衰 0.213s（フィット値）
    decay = int(3.0 * tau * fs)
    kernel = np.exp(-np.arange(decay) / (tau * fs))
    t = rng.uniform(0.0, t_rail)
    while t < duration_sec:
        for k, off_m in enumerate(axle_offsets):
            i = int((t + off_m / speed_mps) * fs)
            acc = 1.0 if k < 2 else 0.8
            if 0 <= i < n:
                amp = acc * (1.0 + 0.25 * rng.standard_normal())
                j = min(decay, n - i)
                env[i:i + j] += max(amp, 0.2) * kernel[:j]
        t += t_rail
    click = (_shaped_noise(n, fs, rng,
                           np.array(_REF["impact"]["freqs"], float),
                           np.array(_REF["impact"]["db"], float)) * env)
    rc = a_weighted_rms(click, fs)
    click = click / rc if rc > 0 else click

    x = np.zeros(n)
    for sig, frac in ((body, body_frac_a), (click, joint_frac_a)):
        r = a_weighted_rms(sig, fs)
        x = x + (np.sqrt(frac) / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x


def make_train_composite(duration_sec: float, fs: int, rng: np.random.Generator,
                         speed_mps: float = 20.0, n_cars: int = 6,
                         peak: float = 0.9) -> np.ndarray:
    """編成合成（試聴・検品用）。位相結合=各車両が同一ジョイントを車両長/速度ずつ遅れて踏む。"""
    n = int(round(duration_sec * fs))
    dt_car = CAR_LEN_M / speed_mps
    base_phase = float(rng.uniform(0.0, RAIL_LEN_M / speed_mps))
    x = np.zeros(n)
    for i in range(n_cars):
        car_rng = np.random.default_rng(rng.integers(2 ** 31) + i)
        car = make_train_passby(duration_sec + dt_car, fs, car_rng,
                                speed_mps=speed_mps, peak=1.0)
        s = int((i * dt_car % (RAIL_LEN_M / speed_mps)) * fs)
        x += car[s:s + n]
    _ = base_phase
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x


# ------------------------------------------------------------------ 警笛 ----

def make_train_horn(duration_sec: float, fs: int, rng: np.random.Generator,
                    horn_type: str = "air", peak: float = 0.9) -> np.ndarray:
    """列車警笛1声（実録参照の実測に較正した2方式）。

    horn_type:
      "air"      空気笛（タイフォン）: 基本約317Hz＋倍音列（実測 317/632/946/1263/1577Hz）
      "electric" 電子ホーン: 300Hz系列の3〜8倍音帯にエネルギー（実測 900/1200/1500/1800/2400Hz）
    吹鳴パターン（長緩一声・短急数声）は呼び出し側がこの1声を並べて作る。
    """
    n = int(round(duration_sec * fs))
    ref = _REF["horn_air" if horn_type == "air" else "horn_electric"]
    detune = float(rng.uniform(0.98, 1.02))   # 個体差（部分音の相対構造は実測のまま）
    jitter = colored_noise(n, fs, rng, slope=2.5, f_lo=0.3)
    x = np.zeros(n)
    for p in ref["partials"]:
        phase = (2.0 * np.pi * np.cumsum(
            p["hz"] * detune * (1.0 + 0.002 * jitter)) / fs
            + rng.uniform(0.0, 2.0 * np.pi))
        x += float(p["amp"]) * np.sin(phase)
    # 実測の振幅エンベロープ概形（20分割）を時間伸縮して適用＋端の立上り/立下り
    env = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 20),
                    np.array(ref["env20"], float))
    a, r = int(0.02 * fs), int(0.06 * fs)
    env[:a] *= np.linspace(0.0, 1.0, a)
    env[-r:] *= np.linspace(1.0, 0.0, r)
    x = x * env
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
