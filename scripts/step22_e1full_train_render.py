# -*- coding: utf-8 -*-
"""E1-full: fold1(学習4,800本)の車30%を大型車化した学習用派生を作る。

設計= md/design/E1full_設計_2026-08-05.md。E1aのstep21と同じデルタ差分加算経路。
- 大型車の抽選: audio_seedの決定論ハッシュ < HEAVY_FRAC（クリップでなく車単位）
- 大型車を1台も含まないクリップは**書き出さない**（サーバー側でv11本体へsymlink）
- 出力: out/dataset_v12_heavy/foa/（変更クリップのみ）+ manifest.jsonl

使い方:
  python scripts/step22_e1full_train_render.py --smoke 2
  python scripts/step22_e1full_train_render.py --rows 0-1599      # 並列分割
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.engine_heavy import heavy_f0_from_seed, make_heavy_delta  # noqa: E402

spec21 = importlib.util.spec_from_file_location(
    "step21", ROOT / "scripts" / "step21_e1a_delta_render.py")
m21 = importlib.util.module_from_spec(spec21)
spec21.loader.exec_module(m21)
m11 = m21.m11

DS = ROOT / "out" / "dataset_outdoor_siren_v11"
OUT_DS = ROOT / "out" / "dataset_v12_heavy"
FS = m11.FS_SIM
HEAVY_FRAC = 0.30
TARGET_63_OVER_1K_DB = 3.0


def is_heavy(audio_seed: int) -> bool:
    u = ((int(audio_seed) * 1103515245 + 12345) % (2 ** 31)) / 2.0 ** 31
    return u < HEAVY_FRAC


def process_clip(clip: str, c: float):
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    cars = [s for s in scene["sources"] if s["class"] == "car_drive"]
    heavies = [s for s in cars if is_heavy(s["params"]["audio_seed"])]
    if not heavies:
        return {"clip": clip, "cars": len(cars), "heavy": 0, "written": False}
    dst_flac = OUT_DS / "foa" / f"{clip}.flac"
    if dst_flac.exists():
        return {"clip": clip, "skipped": True}

    src_flac = DS / "foa" / f"{clip}.flac"
    info = sf.info(src_flac)
    mic = m21.mic_array(scene)
    mix, fs_out = sf.read(src_flac)
    mix = np.asarray(mix, np.float64).T
    delta_sum = np.zeros_like(mix)
    log = {"clip": clip, "cars": len(cars), "heavy": len(heavies),
           "written": True, "per_car": []}
    for src in heavies:
        dry = m11._window(m11._make_dry(src), src["t_on"], src["t_off"])
        car_cal = dry * src["dry_gain"]
        a0, a1 = int(src["t_on"] * FS), max(int(src["t_off"] * FS), 1)
        act = car_cal[a0:a1]
        b63_car = m21.band_energy(act, FS, 44.0, 88.0)
        b1k_car = m21.band_energy(act, FS, 710.0, 1420.0)
        target = b1k_car * 10.0 ** (TARGET_63_OVER_1K_DB / 10.0)
        seed = src["params"]["audio_seed"]
        f0h = heavy_f0_from_seed(seed)
        rng = np.random.default_rng(seed * 31 + 17)
        d_unit = m11._window(make_heavy_delta(m11.CLIP, FS, rng, f0=f0h),
                             src["t_on"], src["t_off"])
        b63_d = m21.band_energy(d_unit[a0:a1], FS, 44.0, 88.0)
        need = max(0.0, target - b63_car)
        g = float(np.sqrt(need / b63_d)) if (need > 0 and b63_d > 0) else 0.0
        rec = {"f0h": round(f0h, 2), "g_delta": g}
        if g > 0.0:
            delta_cal = d_unit * g
            rec["dba_diff"] = round(
                m11.spl_a(act + delta_cal[a0:a1], FS) - m11.spl_a(act, FS), 3)
            _, stem_wr = m11._render_stem(delta_cal, np.array(src["wp"], float),
                                          mic, c)
            delta_sum = delta_sum + stem_wr
        log["per_car"].append(rec)
    new_mix = mix + delta_sum
    log["peak"] = round(float(np.max(np.abs(new_mix))), 4)
    sf.write(dst_flac, new_mix.T.astype(np.float64), fs_out, subtype=info.subtype)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--rows", type=str, default=None)
    args = ap.parse_args()
    (OUT_DS / "foa").mkdir(parents=True, exist_ok=True)

    clips = sorted(p.stem for p in (DS / "foa").glob("fold1_*.flac"))
    assert len(clips) == 4800, len(clips)
    tag = "all"
    if args.rows:
        a, b = args.rows.split("-")
        clips = clips[int(a):int(b) + 1]
        tag = args.rows
    if args.smoke:
        clips = clips[: args.smoke * 20]  # 大型車入りに当たるまで余裕を持って走査

    c = m11.sound_speed(m11.TEMP_C)
    logf = open(OUT_DS / f"manifest_{tag}.jsonl", "a", encoding="utf-8")
    n_written = n_heavy = 0
    warn = []
    for i, clip in enumerate(clips):
        log = process_clip(clip, c)
        logf.write(json.dumps(log, ensure_ascii=False) + "\n")
        if log.get("written"):
            n_written += 1
            n_heavy += log["heavy"]
            if log.get("peak", 0) >= 0.99:
                warn.append(clip)
            if args.smoke:
                print(json.dumps(log, ensure_ascii=False))
                if n_written >= args.smoke:
                    break
        if not args.smoke and (i + 1) % 200 == 0:
            print(f"[{tag}] {i+1}/{len(clips)} written={n_written}", flush=True)
    logf.close()
    print(f"[{tag}] done: 変更{n_written}本 / 大型車{n_heavy}台 / peak警告{len(warn)} {warn[:3]}")


if __name__ == "__main__":
    main()
