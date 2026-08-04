# -*- coding: utf-8 -*-
"""データ精査A1/A2/A4: クリーン音源を再合成して周波数・時間パターンを実測（読み取り専用）。

- A1: 支配周波数（FFTピーク上位）を公称値と突合
- A2: 時間パターン（包絡の周期を自己相関で推定）
- A4: 較正検証 recv_meas_db - recv_pred_db の全ソース分布
出力: 標準出力 + out/audit_sources_wave_2026-08-05.md（新規）
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# scipy DLLブロック回避＋監査の独立性のため、step11を経由せず音源モジュールを
# 直接importし、_make_dry相当のディスパッチを独立に再実装する（v9.1以降=v11規約）
from outdoor_seld.alert_sounds import (make_backup_beep, make_bike_bell,  # noqa: E402
                                       make_bike_bell_ring, make_crossing_v2,
                                       make_horn)
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.siren import (make_fire_siren, make_peepo_siren,  # noqa: E402
                                make_siren)

DS = ROOT / "out" / "dataset_outdoor_siren_v11"
FS = 48000
CLIP = 10.0
L = []


def make_dry_independent(src: dict) -> np.ndarray:
    """step11._make_dryの独立再実装（v11規約: v9.1以降の分岐）。"""
    cls, p_ = src["class"], src["params"]
    if cls == "siren":
        if p_["siren_type"] == "fire":
            rng = np.random.default_rng(p_["audio_seed"])
            return make_fire_siren(CLIP, FS, rng=rng,
                                   **{k: v for k, v in p_.items()
                                      if k not in ("siren_type", "audio_seed")})
        gen = make_peepo_siren if p_["siren_type"] == "peepo" else make_siren
        return gen(CLIP, FS, **{k: v for k, v in p_.items() if k != "siren_type"})
    if cls == "horn":
        rng = np.random.default_rng(p_["audio_seed"])
        return make_horn(CLIP, FS, rng,
                         **{k: v for k, v in p_.items() if k != "audio_seed"})
    if cls == "backup_beep":
        return make_backup_beep(CLIP, FS, **p_)
    if cls == "bike_bell":
        if p_.get("bell_type") == "ring":
            return make_bike_bell_ring(
                CLIP, FS, **{k: v for k, v in p_.items() if k != "bell_type"})
        return make_bike_bell(CLIP, FS, **p_)
    if cls == "crossing":
        rng = np.random.default_rng(p_["click_seed"])
        return make_crossing_v2(CLIP, FS, rng=rng,
                                **{k: v for k, v in p_.items() if k != "click_seed"})
    if cls == "car_drive":
        rng = np.random.default_rng(p_["audio_seed"])
        return make_car_v9(CLIP, FS, rng, f0=p_["f0"])
    raise ValueError(cls)


def p(s=""):
    print(s)
    L.append(s)


def top_peaks(x, fs, n=4, fmin=25.0):
    """スペクトルのピーク上位n個の周波数[Hz]（粗い山検出、近接±3%は統合）。"""
    spec_ = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = f >= fmin
    spec_, f = spec_[m], f[m]
    order = np.argsort(spec_)[::-1]
    peaks = []
    for i in order:
        fi = f[i]
        if all(abs(fi - q) / q > 0.03 for q in peaks):
            peaks.append(float(fi))
        if len(peaks) >= n:
            break
    return sorted(peaks)


def envelope_period(x, fs, lo=0.1, hi=5.0):
    """包絡の主周期[s]（自己相関の最大、lo〜hi秒の範囲）。"""
    env = np.abs(x)
    dec = 100                      # 480Hzまで間引き
    env = env[: len(env) // dec * dec].reshape(-1, dec).mean(axis=1)
    fs_e = fs / dec
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    i0, i1 = int(lo * fs_e), int(hi * fs_e)
    if i1 >= len(ac):
        i1 = len(ac) - 1
    return float((i0 + np.argmax(ac[i0:i1])) / fs_e)


def main():
    # 各クラス・サブタイプの実例を scene.json から拾う（最大3例）
    examples = defaultdict(list)
    for sj in sorted((DS / "work").glob("*mix*/scene.json"))[:1500]:
        s = json.loads(sj.read_text())
        for src in s["sources"]:
            cls = src["class"]
            key = cls
            if cls == "siren":
                key = f"siren_{src['params'].get('siren_type')}"
            if cls == "bike_bell":
                key = ("bell_ring" if src["params"].get("bell_type") == "ring"
                       else "bell_single")
            if len(examples[key]) < 3:
                examples[key].append(src)

    NOMINAL = {
        "siren_peepo": "960/770Hz交互 各0.65s（±ゆらぎ）",
        "siren_wail": "435→870Hzスイープ 周期4s/8s",
        "siren_fire": "ウー音＋警鐘（実測361-710Hz掃引・鐘1150Hz）",
        "horn": "410+500Hz+倍音 0.35s鳴/0.15s休（周期0.5s）",
        "backup_beep": "1000Hz純音 0.5s on/off（周期1.0s）",
        "bell_single": "3000Hz 単打0.45s間隔・2s周期",
        "bell_ring": "3000Hz 毎秒30打・0.65s持続",
        "crossing": "700+750Hz和音 毎分130±5回（周期≈0.46s）",
        "car_drive": "f0(33.6-50.4Hz)+倍音 / タイヤ600-2000Hz",
    }
    p("# A1/A2 クリーン音源の波形実測（各3例、48kHz再合成）")
    for key in sorted(examples):
        p(f"\n## {key} — 公称: {NOMINAL.get(key.split('_')[0] if key not in NOMINAL else key, '?')}")
        for src in examples[key]:
            dry = make_dry_independent(src)
            pk = top_peaks(dry, FS)
            per = envelope_period(dry, FS)
            p(f"- f0={src['params'].get('f0', '')} ピーク周波数 "
              f"{[round(q, 1) for q in pk]} Hz / 包絡主周期 {per:.2f}s")

    # A4 較正検証（全ソース）
    p("\n# A4 較正検証: recv_meas_db - recv_pred_db（直接音ステム、全学習ソース）")
    diffs = defaultdict(list)
    for sj in sorted((DS / "work").glob("*mix*/scene.json")):
        s = json.loads(sj.read_text())
        for src in s["sources"]:
            d = src.get("recv_meas_db"), src.get("recv_pred_db")
            if None not in d:
                diffs[src["class"]].append(d[0] - d[1])
    for cls in sorted(diffs):
        v = np.array(diffs[cls])
        p(f"- {cls}: n={len(v)} 平均{v.mean():+.2f}dB / 中央{np.median(v):+.2f}dB / "
          f"|差|>1dB: {(np.abs(v) > 1.0).mean():.1%}")

    (ROOT / "out" / "audit_sources_wave_2026-08-05.md").write_text(
        "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
