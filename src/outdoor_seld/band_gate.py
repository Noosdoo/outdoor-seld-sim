# -*- coding: utf-8 -*-
"""帯域ゲートの判定量（決め直し 4・本人 10/26 19:32・仕様 §9-5 = v30 の既定の正解の線・§9-6 の停止検査つき）。

音源 i の各枠（0.1 s・calibration.frame_spl_a と同じ切り方・窓なし rfft）で、1/3 オクターブ帯（中心 125 Hz〜10 kHz の 20 帯・
帯の端 = 中心 × 2^(±1/6)・`.tmp/band_snr_20261026/fable_band_snr.py` と同じ）の SN 比 = 10·log10(P_stem,b ÷ P_bg,b)。
背景 = 「暗騒音＋雨＋同じ場面の自分以外の全 stem」（描画では mix − 自分）。判定量 = 一番有利な帯の SN 比（best_band_db）と、その帯の中心（band_hz）。
A 特性は掛けない（帯の中の比なので重みは打ち消す）。

停止検査と床（§9-6 の 1・Astra R4-30-01）:
  - 入力（stem・bg）と帯のパワーに非有限（NaN・Inf）があれば BandGateError で止める（黙って 0% にしない）
  - stem と背景が**両方とも**パワー床（FLOOR = 1e-30）以下の帯は無効（勝ち帯の候補から除く・音の無い帯どうしを「対象 ≥ 背景」にしない）
  - 片側だけ床以下は有効（stem だけ無音 → 大きな負の値・背景だけ無音 → 大きな正の値。床で割る）
  - 全帯が無効な枠は valid = False（ゲート不通過）。best_band_db は INVALID_DB（−999.0）・band_hz は 0.0 を入れる（非有限は返さない）
  - 0 dB ちょうど（stem = 背景）は線 0 を通る（≥）
来歴: CONTRACT（帯・枠・床・無効の値）と code_sha256()（このファイルの内容 hash）を scene・小標本の記録に入れる。
試験: scripts/test_v30_band_gate.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FC = 1000.0 * 2.0 ** (np.arange(-9, 11) / 3.0)            # 125 Hz … 10 079 Hz（20 帯）
FLOOR = 1e-30
INVALID_DB = -999.0
FRAME_SEC = 0.1
CONTRACT = {"bands": "1/3-octave fc=1000*2^(k/3), k=-9..10 (20), edges fc*2^(+-1/6), [lo, hi)", "frame_sec": FRAME_SEC,
            "fft": "rfft without window, |X|^2 summed in band", "floor": FLOOR, "invalid_db": INVALID_DB,
            "invalid_rule": "band invalid if stem and bg both <= floor; frame invalid (gate fail) if all bands invalid"}


class BandGateError(ValueError):
    pass


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def band_edges(fc: np.ndarray = FC) -> tuple[np.ndarray, np.ndarray]:
    return fc / 2.0 ** (1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)


def band_powers(x: np.ndarray, fs: int, frame_sec: float = FRAME_SEC) -> np.ndarray:
    """(n_frames, 20) の帯のパワー（窓なし rfft の |X|² を帯の中の bin で足す）。非有限の入力・パワーは止める。"""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise BandGateError(f"帯域ゲートの入力に非有限の値がある（{int(np.sum(~np.isfinite(x)))} 個）")
    spf = int(round(frame_sec * fs))
    n = len(x) // spf
    P = np.abs(np.fft.rfft(x[:n * spf].reshape(n, spf), axis=1)) ** 2
    f = np.fft.rfftfreq(spf, 1.0 / fs)
    lo, hi = band_edges()
    B = np.stack([P[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(lo, hi)], axis=1)
    if not np.all(np.isfinite(B)):
        raise BandGateError("帯のパワーに非有限の値がある")
    return B


def band_snr_frames(stem: np.ndarray, bg: np.ndarray, fs: int, frame_sec: float = FRAME_SEC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """枠ごとの (一番有利な帯の SN 比 [dB], その帯の中心 [Hz], valid)。stem・bg は同じ長さの 1 チャンネル（描画では W）。
    無効な枠は (INVALID_DB, 0.0, False)。返す値はすべて有限。"""
    Ps, Pb = band_powers(stem, fs, frame_sec), band_powers(bg, fs, frame_sec)
    if Ps.shape != Pb.shape:
        raise BandGateError(f"stem と背景の枠の数が違う: {Ps.shape} {Pb.shape}")
    ok = ~((Ps <= FLOOR) & (Pb <= FLOOR))                    # 両方とも床以下の帯は無効
    snr = 10.0 * np.log10(np.maximum(Ps, FLOOR)) - 10.0 * np.log10(np.maximum(Pb, FLOOR))
    snr = np.where(ok, snr, -np.inf)
    valid = ok.any(axis=1)
    k = np.argmax(snr, axis=1)
    best = np.where(valid, snr[np.arange(len(k)), k], INVALID_DB)
    hz = np.where(valid, FC[k], 0.0)
    if not np.all(np.isfinite(best)):
        raise BandGateError("帯域ゲートの判定量に非有限の値が残った")
    return best, hz, valid


def passes(best: np.ndarray, valid: np.ndarray, line_db: float) -> np.ndarray:
    """ゲートの通過（valid かつ best ≥ 線・未丸めの値で判定）。"""
    return np.asarray(valid, dtype=bool) & (np.asarray(best) >= float(line_db))
