"""v12バイク（自動二輪・原付）音源 — 新クラス8、日本の騒音法規準拠。

根拠（2026-08-05リサーチ）:
- 加速走行騒音規制（7.5m測定）: 原付一種/二種 79dB(A)、軽二輪/小型二輪 82dB(A)
  （平成22年規制、JMCA全国二輪車用品連合会の規制値表/環境省中環審資料）
  → planのレベル帯はこの規制上限をキャップに 70〜82dB(A)@7.5m の実勢レンジ
- 近接排気騒音（0.5m・45°）: 84/90/94/94dB(A) — 参考（測定条件が違うため直接は使わない）
- 速度域: 原付一種は法定30km/h(8.3m/s)、市街地の二輪 30〜60km/h(8.3〜16.7m/s)

音の設計（車 make_car_v9 との識別点）:
- 車=なめらか42Hzノコギリ＋タイヤ帯主体。バイク=**排気パルス列**（発火周波数で
  鋭いパルス→倍音が2kHz超まで立つ「バラバラ/パンパン」感）＋発火同期の強いAM＋
  回転数のゆっくりした変動（スロットル操作感）。タイヤ音は無視できる小ささ
- 発火周波数: 4スト単気筒 rpm/120（例: 5000rpm→41.7Hz）。原付=高回転小排気量で
  やや高め、二輪=低め太め
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import a_weighted_rms
from .noise import colored_noise

# 実録参照（soundeffect-lab バイク通過1/3）からフィットした1/6oct包絡51点
# （2026-08-05。二輪=125-250Hz峰・500Hz以上急峻ロールオフ / 原付=500-1k峰の甲高系）
_BREF = json.loads((Path(__file__).parent / "bike_ref_params.json")
                   .read_text(encoding="utf-8"))


def _pulse_train(n: int, fs: int, f_inst: np.ndarray, duty: float,
                 rng: np.random.Generator) -> np.ndarray:
    """発火周波数 f_inst[Hz] の鋭い排気パルス列（位相積分でクリックレス）。"""
    phase = np.cumsum(f_inst) / fs          # 発火位相（1.0ごとに1発）
    frac = phase - np.floor(phase)
    # dutyの間だけ立つ半波状パルス（角を丸めて耳障りな折返しを回避）
    p = np.clip(1.0 - frac / duty, 0.0, 1.0) ** 2
    # パルス毎の強さゆらぎ（燃焼ばらつき）
    idx = np.floor(phase).astype(np.int64)
    gains = 1.0 + 0.2 * rng.standard_normal(int(idx.max()) + 2)
    return p * gains[idx]


def make_motorcycle(duration_sec: float, fs: int, rng: np.random.Generator,
                    engine_class: str = "motorcycle", speed_mps: float = 12.0,
                    exhaust_frac_a: float = 0.75, rasp_frac_a: float = 0.15,
                    mech_frac_a: float = 0.10, peak: float = 0.9) -> np.ndarray:
    """バイク走行音（ドライ）。engine_class: "moped"(原付) | "motorcycle"(軽/小型二輪)。"""
    n = int(round(duration_sec * fs))
    # 発火周波数（実録参照の推定に合わせ調整）: 原付=85〜120Hz（高回転・甲高い）/
    # 二輪=45〜62Hz（bike-pass1/2の実測 47〜62Hz）
    lo, hi = (85.0, 120.0) if engine_class == "moped" else (45.0, 62.0)
    f0 = float(rng.uniform(lo, hi))
    # 回転数のゆっくりした変動（スロットル感、±12%）
    drift = colored_noise(n, fs, rng, slope=2.5, f_lo=0.2)
    f_inst = f0 * (1.0 + 0.12 * drift)

    pulses = _pulse_train(n, fs, f_inst, duty=0.25, rng=rng)
    # 色付け: 実録参照のフィット包絡（1/6oct 51点）をそのまま適用
    ref = _BREF["moped" if engine_class == "moped" else "motorcycle"]
    X = np.fft.rfft(pulses - pulses.mean())
    f = np.fft.rfftfreq(n, 1.0 / fs)
    lvl = np.interp(np.log2(np.maximum(f, 1e-9)),
                    np.log2(np.array(ref["freqs"], float)),
                    np.array(ref["db"], float))
    exhaust = np.fft.irfft(X * (10.0 ** (lvl / 20.0)), n=n)
    exhaust /= np.std(exhaust)

    # ラスプ（発火同期の乱流ノイズ、同じ実測包絡で色付け=浮かない）
    rasp_env = 0.4 + 0.6 * pulses / max(pulses.max(), 1e-9)
    white = rng.standard_normal(n)
    rasp = np.fft.irfft(np.fft.rfft(white) * (10.0 ** (lvl / 20.0)), n=n) * rasp_env
    rasp /= np.std(rasp)

    x = np.zeros(n)
    for sig, frac in ((exhaust, exhaust_frac_a + mech_frac_a),
                      (rasp, rasp_frac_a)):
        r = a_weighted_rms(sig, fs)
        x = x + (np.sqrt(frac) / r) * sig
    peak_val = float(np.max(np.abs(x)))
    return peak * x / peak_val if peak_val > 0 else x
