# -*- coding: utf-8 -*-
"""人と比べる測定（聞き取り）の準備（2026-10-27 Opus・本人 11:14「人と比べる測定はやりたいです・デモのみ対象」）。

デモの組（学内の使わない本・開発用・tune_field_）の 18 本を、モデルが聞いたのと同じ 10 秒の切り出しから両耳の音にして、
聞き取りの道具（scripts/listen_test_run.py）で鳴らす wav を作る。本番（評価の本・eval_field_）は読まない（名前と置き場で止める）。

  - 入力: out/_demo_field_unused_149/infer/clips/tune_field_clip*.flac（4ch FOA・ACN/SN3D = W,Y,Z,X・24 kHz・10 s・最接近は 8.0 s 目）と
          infer/ann_all.csv（クラス・t_start・t_cpa・cut_offset_s）
  - 両耳の音: 水平 8 方向の仮想スピーカー（2 次元・1 次・2 帯域〔500 Hz 未満は基本の重み・900 Hz 超は max-rE〕）→ 球の頭のモデル（Brown & Duda 1998: 頭の影の 1 次フィルタ＋時間差）。
    1 次なので高い帯の時間差は小さく出る（左右は高い帯では大きさの差で伝わる）。
    耳たぶの手がかりが無いので前後の区別は弱い（ヘッドホン・頭を動かせない条件）。仰角は使わない（水平に投影）。
  - クラスごとに本数をそろえる（本人 11:21）: 多いクラスから、採点に使える正解の時間（暖機のあと・除外区間を引く）が短い本を外す。
  - 始まりを本ごとに 0〜2.0 s 切り詰める（種で固定）: 最接近が毎回 8 秒目に来ると読めてしまうため。切り詰めた分は人に不利（モデルより短い履歴）。
    拍手が切り出しの中にある本（10/22 の多く・頭の 0.3〜1.1 s）は、拍手の 0.5 s 後まで切り詰める（拍手は聞かせない・一番大きい標本にもしない）。拍手は reads/audio_reads_*.json。
  - 全部の本に同じ倍率を掛ける（本ごとに正規化しない＝本どうしの大きさの差はそのまま）。一番大きい標本が −1 dBFS。
  - 反応の速さの組: 小さな背景に 1 kHz の短い音が無作為の時刻に鳴る 6 本（同じ道具・同じヘッドホンの遅れごと測り、あとで差し引く）。
  - 音量の確かめ: ある本の最初の 4 秒（自転車が近づく前の背景）。
出力: out/_demo_field_unused_149/listen/（git の外）: listen_NN.wav（鳴らす順）・rt_NN.wav・volume.wav・trials.csv（道具が読む: block,order,file）・
      _key.csv（答え: clip_id・クラス・切り詰め・最接近とベルの鳴り始めの秒・本人は開かない）・prepare.json（種・倍率・入力の sha）。
使い方: python scripts/listen_test_prepare.py [--seed 20261027] [--out out/_demo_field_unused_149/listen]
試験: scripts/test_listen_test.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

E2E = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEMO = E2E / "out" / "_demo_field_unused_149"
SR_IN, SR_OUT = 24000, 48000
CPA_IN_CLIP_S = 8.0                 # demo_field_unused_chain.py: 最接近を 8 秒目に 10 秒切り出し
HEAD_A, SOUND_C = 0.0875, 343.0     # 頭の半径 [m]・音速 [m/s]
N_SPK = 8
TRIM_MAX_S = 2.0
CLAP_SKIP_S = 0.5                   # 拍手が切り出しの中にあれば、拍手の 0.5 s 後から聞かせる（10/22 の本は拍手が頭の 0.3〜1.1 s にある）
N_RT = 6


class GuardError(RuntimeError):
    pass


def _ear_filter(freqs: np.ndarray, theta: float) -> np.ndarray:
    """Brown & Duda (1998) の球の頭: theta = 音の来る向きと耳の軸のなす角 [rad]（0 = 耳の真横から）。頭の影の 1 次フィルタ × 時間差。"""
    w = 2 * np.pi * freqs
    w0 = SOUND_C / HEAD_A
    a_min, th_min = 0.1, np.deg2rad(150.0)
    alpha = (1 + a_min / 2) + (1 - a_min / 2) * np.cos(theta / th_min * np.pi)
    h = (1 + 1j * alpha * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
    t = -(HEAD_A / SOUND_C) * np.cos(theta) if theta < np.pi / 2 else (HEAD_A / SOUND_C) * (theta - np.pi / 2)
    return h * np.exp(-1j * w * (t + HEAD_A / SOUND_C))      # a/c だけ遅らせて因果に


def foa_to_binaural(foa: np.ndarray, sr: int) -> np.ndarray:
    """foa: (n, 4) ACN/SN3D（W,Y,Z,X）→ (n, 2) 左・右。方位は左が正（atan2(y, x)）。"""
    n = foa.shape[0]
    pad = int(0.05 * sr)
    nfft = n + pad
    W, Yc, Xc = (np.fft.rfft(foa[:, k], nfft) for k in (0, 1, 3))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    # 2 帯域の復号: 低い帯（< 500 Hz）は基本の重み 1（音場の再現が正しく、耳の時間差が出る）・高い帯（> 900 Hz）は max-rE の cos(π/4)・間は線形
    g1 = np.interp(f, [500.0, 900.0], [1.0, np.cos(np.pi / 4)])
    out = []
    for ear_az in (np.pi / 2, -np.pi / 2):                   # 左耳 +90°・右耳 −90°
        A = np.zeros_like(f, dtype=complex); Bx = np.zeros_like(A); By = np.zeros_like(A)
        for i in range(N_SPK):
            th = 2 * np.pi * i / N_SPK
            inc = float(np.arccos(np.clip(np.cos(th - ear_az), -1.0, 1.0)))
            h = _ear_filter(f, inc)
            A += h / N_SPK
            Bx += h * 2 * g1 * np.cos(th) / N_SPK
            By += h * 2 * g1 * np.sin(th) / N_SPK
        out.append(np.fft.irfft(W * A + Xc * Bx + Yc * By, nfft)[:n])
    return np.stack(out, axis=1)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_claps(base: Path) -> dict:
    """録音の拍手の秒（本人が波形で読んだ値）: <base>/<日付>/reads/audio_reads_*.json の {原本: {clap_audio_s}}。"""
    out = {}
    for p in sorted(base.glob("*/reads/audio_reads_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in (d.items() if isinstance(d, dict) else []):
            if isinstance(v, dict) and v.get("clap_audio_s") is not None:
                out[Path(k).name] = float(v["clap_audio_s"])
    return out


def load_targets(infer: Path) -> list:
    """infer/ann_all.csv と clips/ から、聞く本の一覧（clip_id・クラス・最接近・ベルの鳴り始め・拍手〔切り出しの秒〕）。本番の名前と置き場は止める。"""
    if "_demo_field_unused" not in str(infer).replace("\\", "/"):
        raise GuardError(f"デモの組の置き場ではない: {infer}")
    claps = load_claps(infer.parent)
    rows = list(csv.DictReader(open(infer / "ann_all.csv", encoding="utf-8-sig")))
    out, seen = [], set()
    for r in rows:
        cid = r["clip_id"]
        if not cid.startswith("tune_field_clip"):
            raise GuardError(f"開発用の名前（tune_field_clip）ではない: {cid}")
        if cid in seen:
            continue
        seen.add(cid)
        fl = infer / "clips" / f"{cid}.flac"
        if not fl.is_file():
            raise GuardError(f"切り出しが無い: {fl}")
        off = float(r["cut_offset_s"])
        cls = r["class"]
        onset = (float(r["t_start"]) - off) if (cls == "bike_bell" and (r.get("t_start") or "").strip()) else None
        cpa = (float(r["t_cpa"]) - off) if (r.get("t_cpa") or "").strip() else CPA_IN_CLIP_S
        c = claps.get(Path(r.get("orig_file", "")).name)
        clap = (c - off) if (c is not None and 0.0 <= c - off < 10.0) else None
        out.append({"clip_id": cid, "class": cls, "flac": fl, "cpa_clip_s": cpa, "bell_onset_clip_s": onset, "clap_clip_s": clap})
    return out


def valid_label_frames(infer: Path, clip_id: str, cls: str) -> int:
    """採点に使える正解のフレーム数 = 目的のクラスの正解フレーム ∩ 暖機のあと（40〜99）− 除外区間（決め直し 6 と同じ考え）。"""
    c = {"bicycle": 8, "bike_bell": 3}.get(cls)
    fr = set()
    gp = infer / "gt" / f"{clip_id}.csv"
    if gp.is_file():
        for row in csv.reader(open(gp, encoding="utf-8")):
            if len(row) >= 6 and row[0].strip().isdigit() and int(row[1]) == c:
                fr.add(int(row[0]))
    ex = set()
    ep = infer / "exclusions.csv"
    if ep.is_file():
        for r in csv.DictReader(open(ep, encoding="utf-8-sig")):
            if r["clip_id"] == clip_id:
                a, b = float(r["入り_クリップの秒"]), float(r["出_クリップの秒"])
                ex |= set(range(int(np.floor(a / 0.1)), int(np.ceil(b / 0.1))))
    return len({f for f in fr if 40 <= f <= 99} - ex)


def balance(tg: list, infer: Path) -> tuple:
    """クラスごとの本数をそろえる（本人 11:21「クラスごとに種類は均等に」）。多いクラスから、採点に使える正解の時間が短い本を外す
    （結果を見る前に決まる基準・同じ長さなら clip_id の大きい方を外す）。返り値 = (残す本, 外した本)。"""
    by = {}
    for t in tg:
        by.setdefault(t["class"], []).append(t)
    n = min(len(v) for v in by.values())
    keep, drop = [], []
    for cls, v in by.items():
        v = sorted(v, key=lambda t: (-valid_label_frames(infer, t["clip_id"], cls), t["clip_id"]))
        keep += v[:n]
        drop += v[n:]
    keep.sort(key=lambda t: t["clip_id"])
    return keep, drop


def _beep_track(rng: np.random.Generator, sr: int, bg_rms: float) -> tuple:
    dur = 6.0
    onset = float(rng.uniform(1.5, 4.5))
    x = rng.standard_normal(int(dur * sr)) * bg_rms * 0.3          # 小さな背景（白色・背景の 0.3 倍）
    t = np.arange(int(0.1 * sr)) / sr
    beep = np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t)) * bg_rms * 3 * np.sqrt(2)   # はっきり聞こえる短い音
    i0 = int(onset * sr)
    x[i0:i0 + len(beep)] += beep
    return np.stack([x, x], axis=1), onset


def prepare(out: Path, seed: int = 20261027, infer: Path | None = None, do_balance: bool = True) -> dict:
    infer = infer or (DEMO / "infer")
    tg = load_targets(infer)
    dropped = []
    if do_balance:
        tg, dropped = balance(tg, infer)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tg))
    trims = rng.uniform(0.0, TRIM_MAX_S, size=len(tg))
    rendered, keys = [], []
    for k, idx in enumerate(order):
        t = tg[idx]
        foa, sr = sf.read(str(t["flac"]), dtype="float64", always_2d=True)
        if sr != SR_IN or foa.shape[1] != 4:
            raise GuardError(f"{t['flac'].name}: sr={sr} ch={foa.shape[1]}")
        trim = float(round(trims[k], 2))
        why = "無作為"
        if t["clap_clip_s"] is not None and trim < t["clap_clip_s"] + CLAP_SKIP_S:
            trim, why = float(round(t["clap_clip_s"] + CLAP_SKIP_S, 2)), "拍手のあと"     # 拍手は聞かせない（頭の合図・一番大きい標本になるため）
        b = foa_to_binaural(foa[int(round(trim * sr)):], sr)
        rendered.append(b)
        keys.append({"block": "main", "order": k + 1, "file": f"listen_{k + 1:02d}.wav", "clip_id": t["clip_id"], "class": t["class"],
                     "trim_s": trim, "trim_why": why, "clap_clip_s": ("" if t["clap_clip_s"] is None else round(t["clap_clip_s"], 3)),
                     "cpa_file_s": round(t["cpa_clip_s"] - trim, 3),
                     "bell_onset_file_s": (round(t["bell_onset_clip_s"] - trim, 3) if t["bell_onset_clip_s"] is not None else ""),
                     "flac_sha256": _sha(t["flac"])})
    peak = max(float(np.max(np.abs(b))) for b in rendered)
    gain = 10 ** (-1 / 20) / peak
    out.mkdir(parents=True, exist_ok=True)
    for b, kk in zip(rendered, keys):
        sf.write(str(out / kk["file"]), resample_poly(b * gain, SR_OUT // SR_IN, 1, axis=0), SR_OUT, subtype="PCM_16")
    # 音量の確かめ: 鳴らす順で最初の本の最初の 4 秒（近づく前の背景）
    vol = rendered[0][: int(4.0 * SR_IN)] * gain
    sf.write(str(out / "volume.wav"), resample_poly(vol, SR_OUT // SR_IN, 1, axis=0), SR_OUT, subtype="PCM_16")
    bg_rms = float(np.sqrt(np.mean(vol ** 2)))
    rt_keys = []
    for j in range(N_RT):
        x, onset = _beep_track(rng, SR_OUT, bg_rms)
        fn = f"rt_{j + 1:02d}.wav"
        sf.write(str(out / fn), np.clip(x, -1, 1), SR_OUT, subtype="PCM_16")
        rt_keys.append({"block": "rt", "order": j + 1, "file": fn, "clip_id": "", "class": "beep", "trim_s": "", "trim_why": "", "clap_clip_s": "",
                        "cpa_file_s": "", "bell_onset_file_s": round(onset, 3), "flac_sha256": ""})
    cols = ["block", "order", "file", "clip_id", "class", "trim_s", "trim_why", "clap_clip_s", "cpa_file_s", "bell_onset_file_s", "flac_sha256"]
    with open(out / "_key.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rt_keys + keys)
    with open(out / "trials.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "order", "file"]); w.writeheader()
        w.writerows({c: r[c] for c in ("block", "order", "file")} for r in rt_keys + keys)
    meta = {"seed": seed, "gain": gain, "peak_before_gain": peak, "n_main": len(keys), "n_rt": N_RT, "trim_max_s": TRIM_MAX_S,
            "per_class": {c: sum(k["class"] == c for k in keys) for c in sorted({k["class"] for k in keys})},
            "dropped_for_balance": [{"clip_id": t["clip_id"], "class": t["class"], "valid_label_frames": valid_label_frames(infer, t["clip_id"], t["class"])} for t in dropped],
            "balance_rule": "クラスごとに本数をそろえる・多いクラスから採点に使える正解のフレーム（40〜99・除外区間を引く）が少ない本を外す",
            "renderer": "8 方向の仮想スピーカー（2D・1 次・500 Hz 未満は基本・900 Hz 超は max-rE）＋ Brown & Duda 1998 の球の頭（a = 0.0875 m）・仰角なし",
            "sr_out": SR_OUT, "infer": str(infer), "ann_all_sha256": _sha(infer / "ann_all.csv")}
    (out / "prepare.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", type=Path, default=DEMO / "listen")
    a = ap.parse_args(argv)
    if a.out.exists() and any(a.out.glob("results_*.csv")):
        print(f"止める: {a.out} に聞いた結果がある（作り直すと順番と切り詰めが変わる）"); return 2
    m = prepare(a.out, a.seed)
    print(f"聞く本 {m['n_main']}（{m['per_class']}）・反応の速さ {m['n_rt']}・倍率 {m['gain']:.3g} → {a.out}")
    for d in m["dropped_for_balance"]:
        print(f"  数をそろえるために外した: {d['clip_id']}（{d['class']}・採点に使える正解 {d['valid_label_frames']} フレーム）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
