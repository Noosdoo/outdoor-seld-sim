# -*- coding: utf-8 -*-
"""Step 19j: 録音ファイルの波形から拍手の時刻を自動で拾う（2026-09-27・敵対レビュー D の D02。2026-10-12 変更票 #59 r6 で「人は波形を開かない」道具に改訂）。

背景: 2026-10-10 の変更票 #59 で現場の拍手ボタンと通過ボタンをやめた。「いつ」は動画の目印と録音の波形から読む。
      動画の秒を録音の秒に直すのに拍手（両方に入る 1 瞬）を使うが、録った日の夜は本人が録音の波形を開かない（#59 r6）。
      この道具が録音の先頭近くの拍手の立ち上がりの**候補**を自動で拾い、録音ファイルの秒を出す。
      **候補は同期に自動で採用しない**（Astra NT12-01・2026-10-12）: step19d --video-only --clap-csv はこの csv を読み、読む準備の表
      （*_video_takes.csv）の clap_candidate_s・clap_candidate_quality の列に書くだけで、clap_audio_s（同期に使う値）は本人が
      合成を固めたあとに波形で読んで events.csv の 拍手_録音側秒 に入れる（変更票 #59 r7）。step19d は本人の読みと候補が 10 ms より違う本を知らせる。
      旧: 記録シートの「拍手_音側秒」（ストップウォッチ）との差（delta_s）も出せる（--events・#59 より前の記録の互換）。

選び方（2026-10-12・既定 --pick first）:
  高域（>1.5 kHz）の 2 ms 枠の RMS で、探索窓の中の「強い立ち上がり」（枠の値が直前 0.3 s の中央値より ONSET_DB = 15 dB 以上高く、
  探索窓全体の中央値（背景）より HEIGHT_DB = 30 dB 以上高い）を候補にし、--min-start（既定 0.5 s・REC ボタンの操作音は 0.1〜0.3 s に出る）
  より後の**一番早い候補**を拍手の候補とする（強さで先の候補を捨てない＝弱い拍手のあとの大きいベルを採らない・NT12-01）。
  --pick loudest は旧来の「探索窓で一番大きい山」。
  品質（どれも「候補」で、拍手と断定しない）:
    候補（最初の強い音）= 一番早い候補が一番強い候補から WITHIN_DB = 12 dB 以内で、前後 NEAR_S = 2 s に同じ強さの別の候補が無い／
    候補（単独）= 候補が 1 つだけ（拍手が無くベルだけの録音でもこうなる）／
    確認（最初の音があとの音より 12 dB 以上弱い）／確認（近くに同じ強さの音）／
    弱い = 強い立ち上がりが無く、一番大きい山が背景 +10〜20 dB／不明 = +10 dB 未満／確認（立ち上がりが鈍い）= 山は +20 dB 以上あるが立ち上がりが条件に届かない。
  立ち上がりの秒 = 選んだ山から手前へ戻って背景 + 12 dB を初めて下回る枠の次から、山の 30% を初めて超えるサンプル（ms 単位）。
  10/7 の家の録音 4 本（本人が波形で読んだ値 10.934／4.544／6.416／3.944 s）との差は +1〜+3 ms。

使い方:
  python scripts/step19j_clap_from_audio.py --dir raw/ --out out/realsmoke/clap_audio.csv [--search 0-20] [--pick first|loudest] [--min-start 0.5]
  python scripts/step19j_clap_from_audio.py raw/261015_003.WAV raw/261015_004.WAV --out out/reads/claps.csv
  python scripts/step19j_clap_from_audio.py --dir raw/ --out out/reads/claps.csv --events events.csv   # 旧記録: 拍手_音側秒（ラップ）との差も出す
  python scripts/step19j_clap_from_audio.py --selftest out/_test_step19j
出力 csv: file, clap_audio_s（候補の秒・拍手の立ち上がり）, peak_dbfs, snr_db（選んだ山の高さ・背景比）, quality（候補（…）／確認（…）／弱い（…）／不明（…））,
          pick, n_candidates, candidates（"秒:高さdB;…"・時刻順・最大 6 つ）, lap_s, delta_s（clap_audio_s − lap_s・--events があるときだけ）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRAME_S = 0.002
HP_HZ = 1500.0
PREV_S = 0.3          # 立ち上がりを測る「直前」の長さ
ONSET_DB = 15.0       # 候補の条件 1: 枠 − 直前 0.3 s の中央値
HEIGHT_DB = 30.0      # 候補の条件 2: 枠 − 探索窓の中央値（背景）
WITHIN_DB = 12.0      # 一番強い候補からこの範囲を「同じ強さ」と見る
NEAR_S = 2.0          # 品質「確認」: この範囲に同じ強さの別の候補がある
MERGE_S = 0.2         # この間隔より近い候補は 1 つにまとめる（減衰の尾・反射）
MIN_START_DEFAULT = 0.5
OUT_COLS = ["file", "clap_audio_s", "peak_dbfs", "snr_db", "quality", "pick", "n_candidates", "candidates", "lap_s", "delta_s"]


def _frames_db(x: np.ndarray, fs: int):
    sos = butter(4, HP_HZ / (fs / 2), btype="high", output="sos")
    y = sosfiltfilt(sos, x)
    n = int(FRAME_S * fs)
    m = len(y) // n
    rms = np.sqrt(np.mean(y[: m * n].reshape(m, n) ** 2, axis=1)) + 1e-12
    return y, n, 20 * np.log10(rms)


def _candidates(db: np.ndarray, bg: float) -> list[tuple[int, float, float]]:
    """強い立ち上がりの枠を時刻順に返す: (枠番号, 高さ dB = 枠 − 背景, 立ち上がり dB = 枠 − 直前 0.3 s の中央値)。近い枠はまとめる。"""
    k = int(PREV_S / FRAME_S)
    raw = []
    for i in np.flatnonzero(db >= bg + HEIGHT_DB):
        prev = db[max(0, i - k): i]
        onset = float(db[i] - (np.median(prev) if len(prev) else db[i]))
        if onset >= ONSET_DB:
            raw.append((int(i), float(db[i] - bg), onset))
    merged: list[tuple[int, float, float]] = []
    for c in raw:                                                # raw は枠番号の順
        if merged and (c[0] - merged[-1][0]) * FRAME_S <= MERGE_S:
            if c[1] > merged[-1][1]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _rise_sample(y: np.ndarray, db: np.ndarray, n: int, k_peak: int, bg: float) -> int:
    """選んだ山の立ち上がりのサンプル番号（y の中）: 背景 + 12 dB を初めて下回る枠の次から、山の 30 % を初めて超える所。"""
    th = bg + 12.0
    k = k_peak
    while k > 0 and db[k - 1] > th:
        k -= 1
    s0 = k * n
    s1 = min(len(y), (k_peak + 1) * n)
    seg_abs = np.abs(y[s0:s1])
    thr = 0.3 * float(np.max(seg_abs)) if len(seg_abs) else 0.0
    idx = int(np.argmax(seg_abs >= thr)) if len(seg_abs) else 0
    return s0 + idx


def detect_clap(x: np.ndarray, fs: int, search: tuple[float, float] = (0.0, 20.0), pick: str = "first",
                min_start: float = MIN_START_DEFAULT) -> dict:
    """x: (n,) 1 ch。探索窓 [search[0], search[1]] s の中で拍手の立ち上がりを返す（docstring の「選び方」）。"""
    a, b = int(search[0] * fs), min(len(x), int(search[1] * fs))
    seg = x[a:b].astype("float64")
    base = {"clap_audio_s": "", "peak_dbfs": "", "snr_db": "", "quality": "", "pick": pick, "n_candidates": "0", "candidates": ""}
    if len(seg) < fs // 10:
        return {**base, "quality": "不明（短い）"}
    y, n, db = _frames_db(seg, fs)
    bg = float(np.median(db))
    cands = [c for c in _candidates(db, bg) if (a + c[0] * n) / fs >= min_start]
    cand_txt = ";".join(f"{(a + c[0] * n) / fs:.3f}:{c[1]:.1f}" for c in cands[:6])
    if cands:
        hmax = max(c[1] for c in cands)
        if pick == "loudest":
            chosen = max(cands, key=lambda c: c[1])
        else:
            chosen = cands[0]                                   # 一番早い候補（強さで捨てない・NT12-01）
        near = [c for c in cands if c is not chosen and abs(c[0] - chosen[0]) * FRAME_S <= NEAR_S and c[1] >= chosen[1] - WITHIN_DB]
        later_stronger = [c for c in cands if c[0] > chosen[0] and c[1] > chosen[1] + WITHIN_DB]
        k_peak, snr = chosen[0], chosen[1]
        if len(cands) == 1:
            quality = "候補（単独・拍手と断定しない）"
        elif pick == "loudest":
            quality = "確認（先に別の強い音がある）" if chosen is not cands[0] else "候補（一番大きい音）"
        elif later_stronger:
            quality = "確認（最初の音があとの音より 12 dB 以上弱い: " + ", ".join(f"{(a + c[0] * n) / fs:.2f}s" for c in later_stronger[:3]) + "）"
        elif near:
            quality = "確認（近くに同じ強さの音: " + ", ".join(f"{(a + c[0] * n) / fs:.2f}s" for c in near[:3]) + "）"
        else:
            quality = "候補（最初の強い音）"
    else:
        # 強い立ち上がりが無い: 一番大きい山を返すが OK にはしない（min_start より前の山は使わない）
        k0 = int(min_start * fs - a) // n if a / fs < min_start else 0
        k0 = max(0, min(k0, len(db) - 1))
        k_peak = k0 + int(np.argmax(db[k0:]))
        snr = float(db[k_peak] - bg)
        quality = "弱い（拍手か疑わしい）" if snr >= 10.0 else "不明（山が無い）"
        if snr >= 20.0:
            quality = "確認（立ち上がりが鈍い）"
    s_rise = _rise_sample(y, db, n, k_peak, bg)
    t = (a + s_rise) / fs
    lo, hi = max(0, s_rise - int(0.05 * fs)), min(len(seg), s_rise + int(0.2 * fs))
    peak_dbfs = 20 * np.log10(float(np.max(np.abs(seg[lo:hi]))) + 1e-12)
    return {**base, "clap_audio_s": f"{t:.3f}", "peak_dbfs": f"{peak_dbfs:.1f}", "snr_db": f"{snr:.1f}", "quality": quality,
            "n_candidates": str(len(cands)), "candidates": cand_txt}


def read_mono(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as f:
        fs = f.samplerate
        x = f.read(min(len(f), int(seconds * fs)), dtype="float64", always_2d=True)
    return x[:, 0], fs        # W（AmbiX の 1 ch 目 = 無指向）


def run(files: list[Path], out: Path, search: tuple[float, float], events: Path | None, pick: str = "first",
        min_start: float = MIN_START_DEFAULT) -> list[dict]:
    laps = {}
    if events and events.exists():
        with open(events, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = (r.get("拍手_音側秒") or "").strip()
                if v and r.get("原本"):
                    laps[r["原本"].strip().lower()] = v
    rows = []
    for p in files:
        x, fs = read_mono(p, search[1] + 1.0)
        d = detect_clap(x, fs, search, pick, min_start)
        lap = laps.get(p.name.lower(), "")
        delta = f"{float(d['clap_audio_s']) - float(lap):+.3f}" if (lap and d["clap_audio_s"]) else ""
        rows.append({"file": p.name, **d, "lap_s": lap, "delta_s": delta})
        print(f"{p.name}: 拍手 {d['clap_audio_s'] or '—'} s（{d['quality']}・高さ {d['snr_db']} dB・候補 {d['n_candidates']}）"
              + (f"  ラップ {lap} s → ずれ {delta} s" if delta else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(rows)
    n_c = sum(r["quality"].startswith("候補") for r in rows)
    print(f"{len(rows)} 本 → {out}（候補 {n_c}・確認／弱い／不明 {len(rows) - n_c}。どれも同期には自動で入れない＝本人の読みとの突き合わせ用・NT12-01）")
    return rows


def _burst(rng, fs: int, dur_s: float, tau_s: float, amp: float) -> np.ndarray:
    n = int(dur_s * fs)
    return rng.standard_normal(n) * np.exp(-np.arange(n) / (tau_s * fs)) * amp


def selftest(tmp: Path) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    fs = 96000
    rng = np.random.default_rng(1)
    dur = 12.0
    noise = rng.standard_normal((int(dur * fs), 4)) * 10 ** (-50 / 20)        # 暗騒音 −50 dBFS
    x = noise.copy()
    t_clap = 3.217
    i0 = int(t_clap * fs)
    burst = _burst(rng, fs, 0.03, 0.004, 0.5)                                 # 30 ms の減衰する破裂音（拍手）
    x[i0:i0 + len(burst), :] += burst[:, None]
    i1 = int(8.0 * fs); n1 = int(1.0 * fs)                                     # 車の音のような低域の山を 8 s に（高域で拍手を選ぶことの検査）
    x[i1:i1 + n1, :] += (np.sin(2 * np.pi * 120 * np.arange(n1) / fs) * 0.3)[:, None]
    p = tmp / "ZOOM0007.wav"; sf.write(p, x, fs, subtype="PCM_24")
    ev = tmp / "events.csv"; ev.write_text("原本,拍手_音側秒\nZOOM0007.wav,3.5\n", encoding="utf-8")
    r = run([p], tmp / "claps.csv", (0.0, 10.0), ev)[0]
    ok = [("拍手の時刻が ±5 ms", r["clap_audio_s"] != "" and abs(float(r["clap_audio_s"]) - t_clap) <= 0.005),
          ("品質 = 候補（単独）: 候補 1 つだけなので拍手と断定しない", r["quality"].startswith("候補（単独")),
          ("低域の大きい音（8 s）に引かれない", r["clap_audio_s"] != "" and float(r["clap_audio_s"]) < 5.0),
          ("ラップとの差 = −0.283 s（±0.005）", r["delta_s"] != "" and abs(float(r["delta_s"]) - (t_clap - 3.5)) <= 0.005),
          ("csv が書けた・列がそろう", (tmp / "claps.csv").exists() and next(csv.reader(open(tmp / "claps.csv", encoding="utf-8"))) == OUT_COLS)]
    # 拍手が無い録音 → 不明か弱い
    q = tmp / "ZOOM0008.wav"
    sf.write(q, rng.standard_normal((int(5 * fs), 4)) * 10 ** (-50 / 20), fs, subtype="PCM_24")
    r2 = run([q], tmp / "claps2.csv", (0.0, 5.0), None)[0]
    ok.append(("拍手の無い録音は『候補（最初の強い音）』にしない", not r2["quality"].startswith("候補（最初")))
    # 2026-10-12（#59 r6）: REC の操作音（0.15 s・弱い）→ 拍手（3.217 s）→ もっと大きいベル（7.0 s・2.6 kHz）の順の録音
    x3 = noise.copy()
    rec = _burst(rng, fs, 0.01, 0.002, 0.05); x3[int(0.15 * fs): int(0.15 * fs) + len(rec), :] += rec[:, None]
    x3[i0:i0 + len(burst), :] += burst[:, None]
    i2 = int(7.0 * fs); n2 = int(0.3 * fs)
    bell = np.sin(2 * np.pi * 2600 * np.arange(n2) / fs) * np.exp(-np.arange(n2) / (0.08 * fs)) * 0.9
    x3[i2:i2 + n2, :] += bell[:, None]
    p3 = tmp / "ZOOM0009.wav"; sf.write(p3, x3, fs, subtype="PCM_24")
    r3 = run([p3], tmp / "claps3.csv", (0.0, 10.0), None, pick="first")[0]
    r3l = run([p3], tmp / "claps3l.csv", (0.0, 10.0), None, pick="loudest")[0]
    ok.append(("first: REC の操作音（0.5 s より前）を飛ばし、あとの大きいベルではなく最初の強い音（拍手）を選ぶ（±5 ms・候補（最初の強い音））",
               r3["clap_audio_s"] != "" and abs(float(r3["clap_audio_s"]) - t_clap) <= 0.005 and r3["quality"].startswith("候補（最初") and int(r3["n_candidates"]) == 2))
    ok.append(("loudest: 一番大きい山（ベル 7.0 s）を選び、品質は『確認（先に別の強い音）』",
               r3l["clap_audio_s"] != "" and abs(float(r3l["clap_audio_s"]) - 7.0) <= 0.01 and r3l["quality"].startswith("確認（先に")))
    # Astra NT12-01 の反例: 拍手がベルより 12 dB 以上弱い → それでも最初の候補（拍手）を返し、品質は『確認』（同期に入れない）
    x5 = noise.copy() * 10 ** (-10 / 20)                                   # 暗騒音 −60 dBFS（弱い拍手も候補の条件 +30 dB に届く）
    weak = _burst(rng, fs, 0.03, 0.004, 0.12); x5[i0:i0 + len(weak), :] += weak[:, None]
    x5[i2:i2 + n2, :] += bell[:, None]
    p5 = tmp / "ZOOM0011.wav"; sf.write(p5, x5, fs, subtype="PCM_24")
    r5w = run([p5], tmp / "claps5w.csv", (0.0, 10.0), None)[0]
    ok.append((f"NT12-01: 弱い拍手（{r5w['candidates']}）のあとの大きいベルを採らず、最初の候補 3.217 s を返して品質は『確認（最初の音が…弱い）』",
               r5w["clap_audio_s"] != "" and abs(float(r5w["clap_audio_s"]) - t_clap) <= 0.005 and r5w["quality"].startswith("確認（最初の音")))
    # ベルだけ（拍手なし）→ 候補は出るが『候補（単独）』で、拍手と断定しない
    x6 = noise.copy(); x6[i2:i2 + n2, :] += bell[:, None]
    p6 = tmp / "ZOOM0012.wav"; sf.write(p6, x6, fs, subtype="PCM_24")
    r6 = run([p6], tmp / "claps6.csv", (0.0, 10.0), None)[0]
    ok.append(("NT12-01: ベルだけの録音は『候補（単独）』で、『候補（最初の強い音）』にならない", r6["quality"].startswith("候補（単独")))
    # 二重の拍手（3.217 と 3.9 s・同じ強さ）→ 最初の方を返すが品質は『確認』
    x4 = noise.copy(); x4[i0:i0 + len(burst), :] += burst[:, None]
    i4 = int(3.9 * fs); x4[i4:i4 + len(burst), :] += burst[:, None]
    p4 = tmp / "ZOOM0010.wav"; sf.write(p4, x4, fs, subtype="PCM_24")
    r4 = run([p4], tmp / "claps4.csv", (0.0, 10.0), None)[0]
    ok.append(("二重の拍手: 最初の方の秒を返し、品質は『確認』（近くに同じ強さの音）",
               r4["clap_audio_s"] != "" and abs(float(r4["clap_audio_s"]) - t_clap) <= 0.005 and r4["quality"].startswith("確認") and "3.90" in r4["quality"]))
    # 探索窓: 拍手より前で切ると見つからない（OK にしない）
    r5 = run([p], tmp / "claps5.csv", (0.0, 2.0), None)[0]
    ok.append(("探索窓の外の拍手は『候補（最初の強い音）』にしない", not r5["quality"].startswith("候補（最初")))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    st = opt("--selftest")
    if st:
        return selftest(Path(st))
    out = opt("--out"); d = opt("--dir"); ev = opt("--events"); se = opt("--search", "0-20")
    pick = opt("--pick", "first"); min_start = float(opt("--min-start", str(MIN_START_DEFAULT)))
    if pick not in ("first", "loudest"):
        print(f"--pick は first か loudest: {pick}"); return 2
    a, b = se.split("-"); search = (float(a), float(b))
    files = [Path(f) for f in argv]
    if d:
        files += sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".wav", ".flac"))
    if not out or not files:
        print(__doc__); return 2
    run(files, Path(out), search, Path(ev) if ev else None, pick, min_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
