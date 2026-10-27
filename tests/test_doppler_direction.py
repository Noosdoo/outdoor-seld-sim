# -*- coding: utf-8 -*-
"""ドップラーの向きの回帰テスト（2026-09-06・本人「通り過ぎた後に音が上がっていないか」への確認）。

純音 1 kHz の音源が 30 km/h で横 1.2 m を通過するとき、受音のピッチは
  接近中 > 1 kHz > 遠ざかり
になる（fastsim.render_mono は伝搬遅延の時間変化として音源・観測者の両方のドップラーを出す）。
歩くマイク（1.4 m/s・車と同じ向き）では相対速度ぶんだけ変化が小さくなる（観測者ドップラー）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from outdoor_seld.fastsim import render_mono, sound_speed  # noqa: E402

FS, T, F0 = 24000, 10.0, 1000.0
V = 30 / 3.6


def _inst_freq(x, t0, t1):
    seg = x[int(t0 * FS):int(t1 * FS)]
    zc = np.where(np.diff(np.signbit(seg)))[0]
    return FS * (len(zc) - 1) / 2.0 / (zc[-1] - zc[0])


def _dry():
    t = np.arange(int(T * FS)) / FS
    return (0.3 * np.sin(2 * np.pi * F0 * t)).astype(np.float32)


def test_static_mic_pitch_drops_after_pass():
    c = sound_speed(20.0)
    wp = np.array([[0.0, -V * 5, 1.2, 0.5], [10.0, V * 5, 1.2, 0.5]])   # CPA 5 s
    y = render_mono(_dry(), wp, np.array([0.0, 0.0, 1.5]), FS, T)
    fb, fa = _inst_freq(y, 1.5, 2.5), _inst_freq(y, 7.5, 8.5)
    assert fb > F0 > fa, (fb, fa)
    assert abs(fb - F0 * c / (c - V)) < 2.0 and abs(fa - F0 * c / (c + V)) < 2.0, (fb, fa)


def test_walking_mic_observer_doppler_included():
    c = sound_speed(20.0)
    vo = 1.4
    mic = np.array([[0.0, -vo * 5, 0.0, 1.5], [10.0, vo * 5, 0.0, 1.5]])   # 車と同じ向きに歩く
    wp = np.array([[0.0, -V * 5, 1.2, 0.5], [10.0, V * 5, 1.2, 0.5]])
    y = render_mono(_dry(), wp, mic, FS, T)
    fb, fa = _inst_freq(y, 1.5, 2.5), _inst_freq(y, 7.5, 8.5)
    # 観測者が音源から遠ざかりながら接近を受ける: f = f0 (c − vo)/(c − V)、遠ざかり: f0 (c + vo)/(c + V)
    assert abs(fb - F0 * (c - vo) / (c - V)) < 2.0, fb
    assert abs(fa - F0 * (c + vo) / (c + V)) < 2.0, fa
    assert fb > F0 > fa
