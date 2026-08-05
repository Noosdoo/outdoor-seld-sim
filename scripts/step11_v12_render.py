# -*- coding: utf-8 -*-
"""step11_v12_render.py — v12 リアリズム強化レンダラ（チェーン方式）。

設計= md/design/v12設計書_2026-08-05.md（追記2の統合チェックリスト準拠）。
step11_v11_render のチェーンを継承し、**物理・較正・RNG消費順は無変更**のまま:
  ①背景騒音: 完全ピンク → 都市暗騒音（noise_v12、63Hz峰・実測文献準拠）
  ②車: audio_seed決定論抽選で EV15%（engine_ev）/ 大型30%（63Hz帯+3dBデルタ）/ 通常55%
出力先: out/dataset_outdoor_siren_v12/（新規。v11は凍結のまま）

このファイルは既存7,200行（v11 plan）の同一seed再生成を担当する。
列車・キックボードの追加行は step10_v12_plan.py + 本ファイルのサンプラ拡張（次段）。

使い方:
  python scripts/step11_v12_render.py smoke        # 3クリップ生成＋検品出力
  （全量は _run_v12_gen.py を用意予定）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v11_render as m11v  # noqa: E402 (v11チェーン: サンプラ差替済み)
m9 = m11v.m9

from outdoor_seld.engine_ev import make_car_ev  # noqa: E402
from outdoor_seld.engine_heavy import heavy_f0_from_seed, make_heavy_delta  # noqa: E402
from outdoor_seld.noise_v12 import diffuse_foa_urban_noise  # noqa: E402

# ---- 出力先を v12 に切替（v11本体は不変更） --------------------------------
DS_NAME_V12 = "outdoor_siren_v12"
m9.DS = ROOT / "out" / f"dataset_{DS_NAME_V12}"
m9.WORK = m9.DS / "work"

# ---- ①背景騒音の差替（呼び出し規約・RNG消費構造は同一） --------------------
m9.diffuse_foa_noise = diffuse_foa_urban_noise

# ---- ②車のバリアント振り分け（audio_seed決定論、車単位） --------------------
EV_FRAC = 0.15
HEAVY_FRAC = 0.30       # EVでない車のうちではなく全車比（0.15<=u<0.45が大型）
TARGET_63_OVER_1K_DB = 3.0

_orig_make_dry = m9._make_dry


def _car_variant(audio_seed: int) -> str:
    u = ((int(audio_seed) * 2654435761 + 97) % (2 ** 32)) / 2.0 ** 32
    if u < EV_FRAC:
        return "ev"
    if u < EV_FRAC + HEAVY_FRAC:
        return "heavy"
    return "normal"


def _band_energy(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(spec[(f >= f_lo) & (f < f_hi)].sum())


def _make_dry_v12(src: dict) -> np.ndarray:
    cls, p = src["class"], src.get("params", {})
    if cls != "car_drive":
        return _orig_make_dry(src)
    seed = p["audio_seed"]
    variant = _car_variant(seed)
    src["car_variant"] = variant            # scene.jsonに来歴を残す
    if variant == "ev":
        rng = np.random.default_rng(seed)
        speed = abs(float(src.get("speed_mps", 6.0)))
        return make_car_ev(m9.CLIP, m9.FS_SIM, rng, speed_mps=speed)
    dry = _orig_make_dry(src)               # normal / heavyの土台は現行車
    if variant == "heavy":
        b63 = _band_energy(dry, m9.FS_SIM, 44.0, 88.0)
        b1k = _band_energy(dry, m9.FS_SIM, 710.0, 1420.0)
        target = b1k * 10.0 ** (TARGET_63_OVER_1K_DB / 10.0)
        d = make_heavy_delta(m9.CLIP, m9.FS_SIM,
                             np.random.default_rng(seed * 31 + 17),
                             f0=heavy_f0_from_seed(seed))
        b63_d = _band_energy(d, m9.FS_SIM, 44.0, 88.0)
        need = max(0.0, target - b63)
        if need > 0 and b63_d > 0:
            dry = dry + float(np.sqrt(need / b63_d)) * d
    return dry


m9._make_dry = _make_dry_v12


def smoke(n_clips: int = 3) -> None:
    rows = m9.load_plan("core")
    # 車入りを優先して拾う（EV/大型/通常が最低1つずつ出るまで最大30行走査）
    seen = set()
    done = 0
    for row in rows:
        if done >= n_clips and {"ev", "heavy", "normal"} <= seen:
            break
        m9.generate_clip(row)
        import json
        s = json.loads((m9.WORK / row["clip_id"] / "scene.json").read_text())
        cars = [x for x in s["sources"] if x["class"] == "car_drive"]
        for c in cars:
            seen.add(c.get("car_variant", "?"))
        print(f"[smoke] {row['clip_id']} cars={[(c.get('car_variant'), round(c.get('recv_meas_db', 0), 1)) for c in cars]}")
        done += 1
    print(f"variants seen: {seen}")


if __name__ == "__main__":
    if "smoke" in sys.argv:
        smoke()
    else:
        print(__doc__)
