# -*- coding: utf-8 -*-
"""E2ミニ: 40kHzパーキングセンサ・ピング検知の成立性実証（96kHz合成・自己完結）。

設計= md/design/E2ミニ_設計_2026-08-06.md（事前登録: 虚報1%で検出90%を満たす
最大距離を雑音床3水準で報告）。物理は自前のISO 9613実装（本体と同一コード）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dynamic_sound.acoustics.standards.ISO_9613_1_1993 import (  # noqa: E402
    attenuation_coefficients)

FS = 96000
PING_HZ = 40000.0
BURST_S = 0.001            # 1msバースト
RATE_HZ = 10.0             # 繰返し約10Hz
SPL_REF = 100.0            # 設計値: 100dB SPL @0.3m（感度解析でレベル依存も出す）
REF_D = 0.3
ALPHA = float(attenuation_coefficients(
    frequency=np.array([PING_HZ]), temperature=293.15,
    relative_humidity=50.0, pressure=101.325)[0])   # [dB/m] @40kHz
OUT = ROOT / "out" / "e2_mini"


def make_clip(dur, d0, d1, noise_db_rel, rng, with_ping=True):
    """後退接近シナリオ1本（dB値は@10mのピング振幅を0dB基準に正規化した相対系）。"""
    n = int(dur * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    if with_ping:
        d = d0 + (d1 - d0) * t / dur                    # 距離の線形接近
        # 振幅: 1/r + 大気吸収（dB/m）。@10mを0dB基準に正規化
        amp_db = (-20.0 * np.log10(d / 10.0)) - ALPHA * (d - 10.0)
        amp = 10.0 ** (amp_db / 20.0)
        t0 = rng.uniform(0.0, 1.0 / RATE_HZ)
        k = t0
        while k < dur:
            i0 = int(k * FS)
            nb = int(BURST_S * FS)
            if i0 + nb < n:
                tt = np.arange(nb) / FS
                v = 1.0 + (d0 - d1) / dur / 343.0        # 接近ドップラー（受信周波数up）
                burst = np.sin(2 * np.pi * PING_HZ * v * tt) * np.hanning(nb)
                x[i0:i0 + nb] += amp[i0] * burst
            k += 1.0 / RATE_HZ * (1.0 + 0.02 * rng.standard_normal())
    # 雑音床（広帯域白色、38-42kHz帯内の実効レベルで規定）
    w = rng.standard_normal(n)
    bw_frac = 4000.0 / (FS / 2)                          # 帯域幅比
    g = 10.0 ** (noise_db_rel / 20.0) / np.sqrt(bw_frac)
    return x + g * w / np.std(w)


def band_burst_score(x):
    """検知器: 38-42kHz帯域エネルギー包絡 → バースト周期性（8-15Hz）のスコア。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    Xb = np.where((f >= 38000) & (f <= 42000), X, 0)
    env = np.abs(np.fft.irfft(Xb, n=len(x)))
    dec = 480                                            # →200Hz包絡
    env = env[: len(env) // dec * dec].reshape(-1, dec).mean(axis=1)
    env = env - env.mean()
    E = np.abs(np.fft.rfft(env))
    fe = np.fft.rfftfreq(len(env), dec / FS)
    sig = E[(fe >= 8) & (fe <= 15)].max()
    noise = np.median(E[(fe >= 20) & (fe <= 60)]) + 1e-12
    return float(sig / noise)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260806)
    dur = 2.0
    floors = [-20.0, -10.0, 0.0]     # 雑音床（@10mピング基準）
    dists = [3, 5, 8, 10, 13, 16, 20]
    lines = ["# E2ミニ結果: 40kHzピング検知（96kHz合成・古典DSP検知器）", "",
             f"物理: 大気吸収 {ALPHA:.2f} dB/m @40kHz（自前ISO 9613実装）・1/r・"
             "ドップラー。事前登録=虚報1%で検出90%の最大距離。", "",
             "| 雑音床 | " + " | ".join(f"{d}m" for d in dists) + " | 検知可能距離 |",
             "|---|" + "---|" * (len(dists) + 1)]
    for nf in floors:
        # 虚報しきい値: ピング無し200本の99パーセンタイル
        null_scores = sorted(band_burst_score(
            make_clip(dur, 10, 10, nf, rng, with_ping=False)) for _ in range(200))
        th = null_scores[int(0.99 * len(null_scores))]
        row = [f"{nf:+.0f}dB"]
        det_range = "-"
        for d in dists:
            hits = sum(band_burst_score(
                make_clip(dur, d + 1.0, d - 1.0 if d > 1 else d, nf, rng)) > th
                for _ in range(50))
            rate = hits / 50
            row.append(f"{rate:.0%}")
            if rate >= 0.9:
                det_range = f"≥{d}m"
        lines.append("| " + " | ".join(row) + f" | {det_range} |")
        print(lines[-1], flush=True)

    lines += ["", "検知器=帯域エネルギー×バースト周期性（学習なし）。",
              "雑音床は@10mピング振幅基準の相対dB（-20=静かな駐車場想定〜0=劣悪）。"]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # ヘテロダイン試聴（40kHz→2kHz: 機械の聞いている世界を可聴化）
    (OUT / "listen").mkdir(exist_ok=True)
    x = make_clip(4.0, 10.0, 1.0, -20.0, np.random.default_rng(7))
    t = np.arange(len(x)) / FS
    het = x * np.cos(2 * np.pi * 38000.0 * t)            # 40k→2kに周波数シフト
    X = np.fft.rfft(het)
    f = np.fft.rfftfreq(len(het), 1.0 / FS)
    X[f > 6000] = 0
    het = np.fft.irfft(X, n=len(het))
    het = het / max(abs(het.max()), abs(het.min())) * 0.8
    sf.write(OUT / "listen" / "ピング接近10m→1m_可聴化.wav",
             het[::2].astype(np.float32), FS // 2, subtype="PCM_16")
    print("wrote report + 可聴化wav ->", OUT)


if __name__ == "__main__":
    main()
