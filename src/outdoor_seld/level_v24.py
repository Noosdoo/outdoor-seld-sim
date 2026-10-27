# -*- coding: utf-8 -*-
"""音の大きさの決め方・第 4 版の道具（2026-10-12・本人決定 15:28〜15:33・依頼文 r5 §2）。

1. 車の大きさの曲線 `l2m_curve(v_kmh)`: 側方 2 m・LAmax・dB(A)。交通安全環境研究所（平成 18 年度・国交省の検討会 資料 4(1)・
   https://www.mlit.go.jp/common/000044181.pdf ）のガソリン車 2 台（GE1・GE2）を図から読んだ値の平均（事実・読み取り値）:
   停止 47.75・6.5 km/h 51.5・10 km/h 56.5・15 km/h 58.25・20 km/h 62.25・30 km/h 68.5〜69.5 → 69.0。点の間は直線でつなぐ。
   30 km/h より上は 69.0 + 30·log10(v/30)（ASJ RTN-Model の一定の速さの傾き b = 30・仮定）。
   車ごとのばらつきは ±4 dB の一様（仮定・本人 10/12 15:28「1a」）。
   合成の 1 m の値 l1m は「曲線の値 + Δ(v)」。Δ(v) は本番の描画器で車を側方 2 m・マイク 1.2 m に置いて描いた LAFmax と l1m の差
   （`scripts/level_match_v24.py` が計算して `md/design/level_match_v24.json` に書く。20 log r の手計算は使わない＝自転車の規則 ② と同じ）。
2. 発進: 曲線（その本の最後の速さ）+ U(4, 8) dB（仮定。Sandberg 2010: 1 速 0.5 m/s² の加速は同じ速さの一定走行より約 +6 dB・7.5 m）。
3. バイク: 車の曲線 + U(2.4, 3.8) dB（ASJ RTN-Model 2018／2023 と CNOSSOS-EU の二輪と小型車類の差・仮定の幅）+ 車と同じ ±4 dB。
4. サイレン: 前方 20 m で U(90, 105) dB（東京消防庁 1993・1996・1997 の救急車 9 台 90〜103・105 超なし＝事実・幅は仮定）、
   スピーカーの高さ U(2.2, 2.6) m（屋根の前の上・全高 約 2.5 m・仮定の幅）、向きの差 `directivity_db(theta)`: 前 0・横 −12・後ろ −14 dB
   （2000 年の 5 台の中央値 横 −9〜−14・後ろ −10〜−18 の中の値・成分別の測定を広帯域に一律に掛ける近似）。
5. 最大値で合わせる音（ベル・クラクション・バック音・踏切）: 鳴動窓の中の LAFmax（A 特性・F 時間重み 125 ms）を 1 m の値にそろえる
   （JIS D 9451・JBPI・細目告示 141 条・R165 はどれも最大値で書かれている）。走行音（車・バイク・自転車・キック）とサイレンは今までどおり
   窓の中の A 特性 RMS。
6. バック音の 1 m の値: 「通常」60〜75 dB(A) は車の後ろ 7 m の最大値 → 1 m では約 77〜92（20 log 7 = 16.9 dB）。実勢のブザー 85〜95 dB/1 m
   （ミツバ RH-11・YEC）との 50:50 の混合は v10.1b のまま。

自己検査: python -m outdoor_seld.level_v24
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .calibration import K_RMS_SPL, a_weight_amplitude
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from outdoor_seld.calibration import K_RMS_SPL, a_weight_amplitude

# 側方 2 m の LAmax（dB(A)）・GE1 と GE2 の平均（図から読んだ値）
L2M_POINTS_KMH = (0.0, 6.5, 10.0, 15.0, 20.0, 30.0)
L2M_POINTS_DB = (47.75, 51.5, 56.5, 58.25, 62.25, 69.0)
L2M_SLOPE_ABOVE_30 = 30.0           # dB per decade（ASJ の一定の速さ b = 30・仮定）
CAR_SPREAD_DB = (-4.0, 4.0)         # 車ごとのばらつき（仮定）
LAUNCH_BONUS_DB = (4.0, 8.0)        # 発進の加速の分（仮定）
BIKE_OFFSET_DB = (2.4, 3.8)         # 二輪 − 小型車類（予測モデル・仮定の幅）
SIREN_LV_20M_DB = (90.0, 105.0)     # 前方 20 m（事実の範囲・幅は仮定）
SIREN_REF_M = 20.0
SIREN_Z_M = (2.2, 2.6)              # スピーカーの高さ（仮定）
SIREN_DIR_DB = {"front": 0.0, "side": -12.0, "rear": -14.0}   # 向きの差（近似）
BACKUP_LEGACY_1M = (85.0, 95.0)     # 実勢のブザー（ミツバ RH-11・YEC・1 m）
BACKUP_NORMAL_1M = (77.0, 92.0)     # R165「通常」60〜75 @7 m → 1 m（本人 10/12 決定 6）
MATCH_MAX_CLASSES = ("bike_bell", "horn", "backup_beep", "crossing")
MATCH_GEOMETRY = {"lateral_m": 2.0, "mic_z_m": 1.2, "src_z_m": 0.325, "cpa_s": 5.0, "clip_s": 10.0}  # R138／交通安全環境研究所の形。音源の高さは #70 r7／R108-07（2026-10-19）で 1.05（旧 SRC_Z 0.8〜1.3 の真ん中）→ 0.325 m（新 0.05〜0.6 の真ん中・数で固定）。旧の Δ は md/design/level_match_v24_srcz1p05_2026-10-12.json
SIREN_MATCH_GEOMETRY = {"dist_m": 20.0, "src_z_m": 2.4, "mic_z_m": 1.5, "clip_s": 10.0}
# サイレンの 20 m の値を描画で合わせる置き方: 前方 20 m・静止・スピーカー高さは SIREN_Z_M の中央・**マイク 1.5 m（東京消防庁 1993 の報告 本文 p.109 3(3) の測定高さ＝事実・Astra SV26-06・2026-10-13。
# 1.2 m は「資料に無い」としていた誤り。鏡像との干渉があるので 1.2 → 1.5 で 1.5〜2.5 dB 変わる）**・
# 直接音＋地面の鏡像・本番の本と同じ 10 s（ピーポーのピーとポーの割合を本番と同じに）。20 m の報告値は現場の値（地面込み・マイクから見た 3.4° の仰角の向きの差も込み）なので、
# 逆 2 乗だけの換算では反射の分（約 +2 dB）だけ大きくなる（音の検査 SIREN_LEVEL_20M 2026-10-12）→ 車と同じく level_match_v24.json の Δ を引く。
# Δ は本番の道（v26 の生成器 → v24 の向きの差 → v13 の RMS 合わせ → 描画）で測る（向きの差なしで測ると仰角の分 −0.46 dB が残った・17:4x）。


def l2m_curve(v_kmh: float) -> float:
    """側方 2 m の LAmax dB(A)。点の間は直線・30 km/h より上は 30 log。"""
    v = float(v_kmh)
    if v <= L2M_POINTS_KMH[-1]:
        return float(np.interp(max(v, 0.0), L2M_POINTS_KMH, L2M_POINTS_DB))
    return L2M_POINTS_DB[-1] + L2M_SLOPE_ABOVE_30 * float(np.log10(v / L2M_POINTS_KMH[-1]))


def directivity_db(theta_deg: np.ndarray | float) -> np.ndarray | float:
    """進む向きとマイクの方向の角度 θ（0 = 前・180 = 後ろ）→ 大きさの差 [dB]。前 0・横 −12・後ろ −14 の間を直線でつなぐ。"""
    th = np.abs(np.asarray(theta_deg, dtype=float))
    return np.interp(th, [0.0, 90.0, 180.0], [SIREN_DIR_DB["front"], SIREN_DIR_DB["side"], SIREN_DIR_DB["rear"]])


def lafmax_db(x: np.ndarray, fs: int, tau: float = 0.125) -> float:
    """A 特性・F 時間重み（指数平均 125 ms）の最大の音圧レベル（本プロジェクトの換算 K_RMS_SPL）。"""
    from scipy.signal import lfilter
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    X = np.fft.rfft(x); X *= a_weight_amplitude(np.fft.rfftfreq(n, 1.0 / fs))
    wa = np.fft.irfft(X, n=n)
    alpha = 1.0 - np.exp(-1.0 / (fs * tau))
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], wa * wa)
    return float(10.0 * np.log10(max(float(np.max(y)), 1e-30)) + K_RMS_SPL)


def selftest() -> int:
    ok = []
    ok.append(("曲線: 測った点をそのまま通る（6.5→51.5・10→56.5・15→58.25・20→62.25・30→69.0）",
               all(abs(l2m_curve(v) - d) < 1e-9 for v, d in zip(L2M_POINTS_KMH, L2M_POINTS_DB))))
    ok.append(("曲線: 間は直線（12.5 km/h → 57.375）", abs(l2m_curve(12.5) - 57.375) < 1e-9))
    ok.append(("曲線: 30 より上は 30 log（54 km/h → 69 + 30 log10(1.8) = 76.66）", abs(l2m_curve(54.0) - (69.0 + 30 * np.log10(1.8))) < 1e-9))
    ok.append(("曲線: 単調に増える（0〜100 km/h）", all(l2m_curve(a) <= l2m_curve(b) for a, b in zip(np.arange(0, 100, 0.5), np.arange(0.5, 100.5, 0.5)))))
    ok.append(("向きの差: 前 0・横 −12・後ろ −14・45° で −6", directivity_db(0) == 0.0 and directivity_db(90) == -12.0 and directivity_db(180) == -14.0 and abs(directivity_db(45) + 6.0) < 1e-9 and abs(directivity_db(-90) + 12.0) < 1e-9))
    fs = 48000; t = np.arange(int(2.0 * fs)) / fs
    tone = np.sin(2 * np.pi * 1000 * t)                     # 1 kHz の連続音は A 特性 0 dB → LAFmax ≈ RMS の値
    from outdoor_seld.calibration import spl_a
    ok.append((f"LAFmax: 1 kHz の連続音で RMS の値と 0.1 dB 以内（{lafmax_db(tone, fs):.2f} vs {spl_a(tone, fs):.2f}）", abs(lafmax_db(tone, fs) - spl_a(tone, fs)) < 0.1))
    burst = tone.copy(); burst[int(0.5 * fs):] = 0.0       # 0.5 s だけ鳴る → 窓 2 s の RMS は −6 dB・LAFmax はほぼ同じ
    ok.append((f"LAFmax: 窓の 4 分の 1 だけ鳴る音では RMS より約 6 dB 大きい（差 {lafmax_db(burst, fs) - spl_a(burst, fs):.2f}）", 5.5 <= lafmax_db(burst, fs) - spl_a(burst, fs) <= 6.5))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(selftest())
