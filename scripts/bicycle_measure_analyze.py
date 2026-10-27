# -*- coding: utf-8 -*-
"""自転車の走行音の測定（2026-10-07・家・新クラス bicycle の作成用）を読む道具。評価データではない。

入力:
  - H3-VR の録音（AmbiX 4ch・W を使う）                         raw_bicycle_measure/261007_007.WAV
  - 騒音計 CENTER 392 の 1 秒ごとの dB(A・FAST)（slm_ghf_to_csv.py） raw_bicycle_measure/slm_20261007_bicycle.csv
  - 動画で読んだ目印の通過時刻（step19h の reads・t = 録音の秒）    raw_bicycle_measure/reads/reads_A.csv
やること（PROGRESS 2026-10-07 10:40〜11:20「測った値の使い方」の ①・④・⑤ の下ごしらえ。② の l1m_db は合成側で描画して合わせるので別）:
  1. 騒音計と録音の時刻合わせ（1 秒ごとの包絡の相互相関）と、録音の dB 換算（静かな秒の 騒音計 − 録音 の中央値）。
     校正の 1 分（001・騒音計の目視 42.6 dB）とも突き合わせる。
  2. 通過ごとに 速さ（x = +2 → −2 の 4 m の時間）・受音点の L_AFmax（A 特性・FAST 125 ms・録音から）・
     騒音計の最大（1 秒抽出なので下限）・窓の L_Aeq・暗騒音を引いた L_AFmax。
  3. 速さ 3 段ごとの中央値と、log10(v) に対する傾き（n log v の n）。
  4. 1/3 オクターブ帯域（100 Hz〜10 kHz）の、通過の最大付近 ±0.3 s の平均と暗騒音の差（音色の比較用・⑤）。
出力: raw_bicycle_measure/analysis/passes.csv・bands.csv・summary.txt
使い方: python scripts/bicycle_measure_analyze.py [--wav ...] [--meter ...] [--reads ...] [--out raw_bicycle_measure/analysis]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from outdoor_seld.calibration import a_weight_amplitude, spl_a, K_RMS_SPL  # noqa: E402

FS = 48000
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_w(path: Path, fs: int = FS) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    w = x[:, 0]
    if sr != fs:
        w = resample_poly(w, fs, sr) if sr < fs else resample_poly(w, 1, sr // fs)
    return w


def a_weighted_signal(w: np.ndarray, fs: int) -> np.ndarray:
    """A 特性を周波数領域で掛けた時間信号（FAST の時間重みはこの後に掛ける）。"""
    n = len(w)
    X = np.fft.rfft(w)
    X *= a_weight_amplitude(np.fft.rfftfreq(n, 1.0 / fs))
    return np.fft.irfft(X, n=n)


def fast_level(wa: np.ndarray, fs: int, tau: float = 0.125) -> np.ndarray:
    """A 特性信号 → FAST（指数平均 125 ms）の音圧レベル dB（本プロジェクトの換算 K_RMS_SPL・較正ゲインは外で足す）。"""
    alpha = 1.0 - np.exp(-1.0 / (fs * tau))
    p2 = wa * wa
    # 一次 IIR の指数平均
    from scipy.signal import lfilter
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], p2)
    return 10.0 * np.log10(np.maximum(y, 1e-30)) + K_RMS_SPL


def sec_levels(w: np.ndarray, fs: int) -> np.ndarray:
    n = len(w) // fs
    return np.array([spl_a(w[i * fs:(i + 1) * fs], fs) for i in range(n)])


def align_meter(rec_sec: np.ndarray, meter: np.ndarray) -> tuple[int, float]:
    """騒音計の lag 秒目 = 録音の 0 秒目 となる lag と相関を返す。"""
    best = (0, -1.0)
    rz = (rec_sec - rec_sec.mean()) / rec_sec.std()
    for lag in range(0, len(meter) - len(rec_sec) + 1):
        seg = meter[lag:lag + len(rec_sec)]
        c = float(np.corrcoef(rz, (seg - seg.mean()) / (seg.std() + 1e-9))[0, 1])
        if c > best[1]:
            best = (lag, c)
    return best


def third_octave_bands(lo: float = 100.0, hi: float = 10000.0) -> list[tuple[float, float, float]]:
    out = []
    fc = 1000.0 * 2.0 ** (np.arange(-30, 31) / 3.0)
    for f in fc:
        if lo <= f <= hi:
            out.append((float(f), float(f / 2 ** (1 / 6)), float(f * 2 ** (1 / 6))))
    return out


def band_levels(seg: np.ndarray, fs: int, bands) -> np.ndarray:
    n = len(seg)
    X = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    out = []
    for _, lo, hi in bands:
        m = (fr >= lo) & (fr < hi)
        p = (2.0 * X[m].sum()) / (n * n)
        out.append(10.0 * np.log10(max(p, 1e-30)) + K_RMS_SPL)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(ROOT / "raw_bicycle_measure/261007_007.WAV"))
    ap.add_argument("--meter", default=str(ROOT / "raw_bicycle_measure/slm_20261007_bicycle.csv"))
    ap.add_argument("--reads", default=str(ROOT / "raw_bicycle_measure/reads/reads_A.csv"))
    ap.add_argument("--calib-wav", default=str(ROOT / "raw_demo/261007_001.WAV"), help="校正の 1 分（任意・突き合わせ用）")
    ap.add_argument("--calib-laeq", type=float, default=42.6, help="校正の 1 分の騒音計の読み（記録シート）")
    ap.add_argument("--out", default=str(ROOT / "raw_bicycle_measure/analysis"))
    ap.add_argument("--lateral", type=float, default=1.0); ap.add_argument("--mic-z", type=float, default=1.5); ap.add_argument("--src-z", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []

    w = load_w(Path(a.wav)); fs = FS
    meter = np.array([float(r["dB"]) for r in csv.DictReader(l for l in open(a.meter, encoding="utf-8") if not l.startswith("#"))])
    rec_sec = sec_levels(w, fs)
    lag, corr = align_meter(rec_sec, meter)
    quiet = meter[lag:lag + len(rec_sec)] < 45.0
    diff = meter[lag:lag + len(rec_sec)] - rec_sec
    gain = float(np.median(diff[quiet])) if quiet.sum() >= 10 else float(np.median(diff))
    lines.append(f"時刻合わせ: 騒音計の {lag} 秒目 = 録音の 0 秒目（相関 {corr:.2f}）")
    lines.append(f"録音の dB 換算（静かな {int(quiet.sum())} 秒の 騒音計 − 録音 の中央値）: {gain:+.1f} dB（四分位 {np.percentile(diff[quiet],25):+.1f}〜{np.percentile(diff[quiet],75):+.1f}）")
    if Path(a.calib_wav).exists():
        wc = load_w(Path(a.calib_wav))[: 60 * fs]
        g2 = a.calib_laeq - spl_a(wc, fs)
        lines.append(f"校正の 1 分（{Path(a.calib_wav).name}・騒音計 {a.calib_laeq} dB）からの換算: {g2:+.1f} dB（測定の中の静かな秒からの値との差 {g2 - gain:+.1f} dB）")

    wa = a_weighted_signal(w, fs)
    lf = fast_level(wa, fs) + gain
    # 暗騒音: 録音全体の L_A90（1 秒値の下位 10%）と、最も静かな 10 秒
    l90 = float(np.percentile(rec_sec + gain, 10))
    lines.append(f"暗騒音: 1 秒値の L_A90 = {l90:.1f} dB(A)・最も静かな 10 秒の平均 = {np.sort(rec_sec + gain)[:10].mean():.1f} dB(A)")

    reads = {}
    for r in csv.DictReader(open(a.reads, encoding="utf-8")):
        reads.setdefault(r["take"], {})[float(r["x"])] = float(r["t"])
    bands = third_octave_bands()
    rows = []; band_rows = []
    bg_seg = None
    # 暗騒音の帯域: 最も静かな 10 秒のうち連続する 3 秒
    order = np.argsort(rec_sec)
    for i in order:
        if i + 3 <= len(rec_sec) and rec_sec[i:i + 3].max() <= np.sort(rec_sec)[10]:
            bg_seg = w[i * fs:(i + 3) * fs]; bg_t = i; break
    if bg_seg is None:
        bg_t = int(order[0]); bg_seg = w[bg_t * fs:(bg_t + 1) * fs]
    bg_bands = band_levels(bg_seg, fs, bands) + gain
    for take, mk in reads.items():
        if 2.0 not in mk or -2.0 not in mk or 0.0 not in mk:
            lines.append(f"{take}: 目印が足りない {sorted(mk)} → 飛ばす"); continue
        v = 4.0 / abs(mk[-2.0] - mk[2.0])                     # m/s（x = +2 → −2 の 4 m）
        t0 = mk[0.0]
        xs = sorted(mk); dt_all = abs(mk[xs[-1]] - mk[xs[0]]); v_all = (xs[-1] - xs[0]) / dt_all if dt_all > 0 else v
        ta, tb = min(mk.values()) - 0.5, max(mk.values()) + 0.5
        ia, ib = int(ta * fs), int(tb * fs)
        seg_lf = lf[ia:ib]
        k = int(np.argmax(seg_lf)); t_max = ta + k / fs
        lmax = float(seg_lf.max())
        leq = spl_a(w[ia:ib], fs) + gain
        # 騒音計の最大（窓の中・1 秒抽出）
        ma, mb = int(ta) + lag, int(np.ceil(tb)) + lag
        mmax = float(meter[max(0, ma):min(len(meter), mb)].max()) if mb > ma else float("nan")
        lmax_corr = 10 * np.log10(max(10 ** (lmax / 10) - 10 ** (l90 / 10), 1e-9))
        rows.append({"take": take, "v_kmh": round(v * 3.6, 2), "v_all_kmh": round(v_all * 3.6, 2), "t_cpa_s": round(t0, 3),
                     "t_max_s": round(t_max, 3), "LAFmax_rec_dB": round(lmax, 1), "LAFmax_meter1s_dB": round(mmax, 1),
                     "LAeq_window_dB": round(leq, 1), "LAFmax_minus_bg_dB": round(lmax_corr, 1), "window_s": f"{ta:.2f}-{tb:.2f}"})
        # 帯域: 最大付近 ±0.3 s
        j0, j1 = int((t_max - 0.3) * fs), int((t_max + 0.3) * fs)
        bl = band_levels(w[j0:j1], fs, bands) + gain
        band_rows.append({"take": take, "v_kmh": round(v * 3.6, 2), **{f"{fc:.0f}": round(float(b - g), 1) for (fc, _, _), b, g in zip(bands, bl, bg_bands)}})
    with open(out / "passes.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(out / "bands.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(band_rows[0].keys())); wr.writeheader(); wr.writerows(band_rows)
        wr.writerow({"take": "暗騒音(dB)", "v_kmh": "", **{f"{fc:.0f}": round(float(g), 1) for (fc, _, _), g in zip(bands, bg_bands)}})

    lines.append("")
    lines.append("通過ごと（受音点 = 線から横 %.1f m・マイク高さ %.1f m・音源高さ %.1f m 仮定・暗騒音は L_A90）:" % (a.lateral, a.mic_z, a.src_z))
    for r in rows:
        lines.append(f"  {r['take']}: {r['v_kmh']:5.1f} km/h  L_AFmax 録音 {r['LAFmax_rec_dB']:5.1f}  騒音計(1 s 抽出) {r['LAFmax_meter1s_dB']:5.1f}  窓 L_Aeq {r['LAeq_window_dB']:5.1f}  暗騒音を引いた L_AFmax {r['LAFmax_minus_bg_dB']:5.1f} dB(A)")
    # 3 段（速さで 3 等分）
    vs = np.array([r["v_kmh"] for r in rows]); ls = np.array([r["LAFmax_minus_bg_dB"] for r in rows]); lraw = np.array([r["LAFmax_rec_dB"] for r in rows])
    order = np.argsort(vs); groups = np.array_split(order, 3)
    lines.append("")
    lines.append("速さ 3 段（速さ順に 3 等分）の中央値:")
    for name, g in zip(("遅い", "普通", "速い"), groups):
        lines.append(f"  {name}: 速さ {np.median(vs[g]):.1f} km/h（{vs[g].min():.1f}〜{vs[g].max():.1f}）  L_AFmax 中央値 {np.median(lraw[g]):.1f} dB(A)・暗騒音を引いて {np.median(ls[g]):.1f} dB(A)・n={len(g)}")
    # 傾き n log v
    X = np.log10(vs / 3.6); A = np.vstack([X, np.ones_like(X)]).T
    n_raw = np.linalg.lstsq(A, lraw, rcond=None)[0][0]; n_corr = np.linalg.lstsq(A, ls, rcond=None)[0][0]
    rr = np.corrcoef(X, ls)[0, 1]
    lines.append(f"速さ依存（最小二乗・10 本）: 暗騒音を引いた L_AFmax = {n_corr:.1f} log10(v) + …（相関 {rr:.2f}）・引く前 {n_raw:.1f} log10(v)。設計は 30 log v・許す範囲 20〜40")
    lines.append("")
    lines.append("1/3 オクターブ（通過の最大付近 ±0.3 s − 暗騒音・dB・速さ順）: bands.csv。6 dB を超える帯域が音色合わせの対象（本番が別の個体のとき・⑤）")
    lines.append("")
    lines.append("注意: 20 log r の手計算で 1 m の値にしない（規則 ②）。l1m_db は合成側で同じ置き方を描画して合わせる。今日の測定は評価には使わない（⑥）。")
    txt = "\n".join(lines); (out / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
