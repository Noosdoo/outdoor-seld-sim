# -*- coding: utf-8 -*-
"""E1a: val 1,200本の車に大型車の低周波デルタを差分加算した派生セットを作る。

設計= md/design/E1a_低周波強化_設計_2026-08-05.md。v11本体は不変更、出力は
out/dataset_v11_e1a_heavy/foa/（新規）。レンダ経路は step11 のものを importlib で
そのまま流用する（_make_dry/_window/_render_stem/定数、独自実装しない）。

手順（クリップ毎）:
  1. scene.json から車を復元: dry = _make_dry(src) → _window → × 保存済み dry_gain
  2. デルタ = make_heavy_delta（f0はaudio_seedから50〜70Hzを決定論選択、≤250Hz）
  3. ゲイン規定: 車の63Hzオクターブ帯(44-88Hz)エネルギーが1kHz帯(710-1420Hz)+3dB
     になるよう g_delta を解く（既に超えていれば g=0）
  4. QC: dB(A)変化 <0.5dB を車毎に実測記録（超過は警告列挙）
  5. デルタを direct+mirror でFOAレンダ（_render_stem）し既存flacに加算、ピーク検査

使い方:
  python scripts/step21_e1a_delta_render.py --smoke 3   # 検品モード（試聴wavも出力）
  python scripts/step21_e1a_delta_render.py             # 全量
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.engine_heavy import heavy_f0_from_seed, make_heavy_delta  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "step11", ROOT / "scripts" / "step11_v9_render.py")
m11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11)

DS = ROOT / "out" / "dataset_outdoor_siren_v11"
OUT_DS = ROOT / "out" / "dataset_v11_e1a_heavy"
PROBE = ROOT / "out" / "e1a_heavy_probe"
FS = m11.FS_SIM
TARGET_63_OVER_1K_DB = 3.0


def band_energy(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    spec_ = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(spec_[(f >= f_lo) & (f < f_hi)].sum())


def mic_array(scene: dict) -> np.ndarray:
    """scene.jsonのmic記述からマイク軌道を復元（step12_notify_v9._mic_wpと同規約）。"""
    m = scene["mic"]
    if "waypoints" in m:
        return np.array(m["waypoints"], dtype=np.float64)
    if m["motion"] == "static":
        return np.array([0.0, 0.0, 1.5])
    if m["motion"] == "walk":
        v, d = m["walk_speed_mps"], m["walk_dir_x"]
        x0 = -d * v * 5.0
        return np.array([[0.0, x0, 0.0, 1.5], [10.0, x0 + d * v * 10.0, 0.0, 1.5]])
    raise ValueError(f"unknown mic motion: {m}")


def process_clip(clip: str, c: float, listen_dir: Path | None):
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    cars = [s for s in scene["sources"] if s["class"] == "car_drive"]
    src_flac = DS / "foa" / f"{clip}.flac"
    dst_flac = OUT_DS / "foa" / f"{clip}.flac"
    if dst_flac.exists():                     # 再実行時は続きから（中断耐性）
        return {"clip": clip, "skipped": True}
    info = sf.info(src_flac)
    if not cars:
        shutil.copy2(src_flac, dst_flac)
        return {"clip": clip, "cars": 0}

    mic = mic_array(scene)
    mix, fs_out = sf.read(src_flac)
    mix = np.asarray(mix, np.float64).T          # (4, n)
    assert fs_out == m11.FS_OUT
    delta_sum = np.zeros_like(mix)
    log = {"clip": clip, "cars": len(cars), "per_car": []}

    for src in cars:
        dry = m11._window(m11._make_dry(src), src["t_on"], src["t_off"])
        car_cal = dry * src["dry_gain"]
        a0, a1 = int(src["t_on"] * FS), max(int(src["t_off"] * FS), 1)
        act = car_cal[a0:a1]
        b63_car = band_energy(act, FS, 44.0, 88.0)
        b1k_car = band_energy(act, FS, 710.0, 1420.0)
        target = b1k_car * 10.0 ** (TARGET_63_OVER_1K_DB / 10.0)
        seed = src["params"]["audio_seed"]
        f0h = heavy_f0_from_seed(seed)
        rng = np.random.default_rng(seed * 31 + 17)
        d_unit = m11._window(make_heavy_delta(m11.CLIP, FS, rng, f0=f0h),
                             src["t_on"], src["t_off"])
        b63_d = band_energy(d_unit[a0:a1], FS, 44.0, 88.0)
        need = max(0.0, target - b63_car)
        g = float(np.sqrt(need / b63_d)) if (need > 0 and b63_d > 0) else 0.0
        rec = {"f0h": round(f0h, 2), "g_delta": g,
               "b63_over_b1k_before_db": round(10 * np.log10(b63_car / b1k_car), 2)
               if b1k_car > 0 else None}
        if g > 0.0:
            delta_cal = d_unit * g
            dba0 = m11.spl_a(act, FS)
            dba1 = m11.spl_a(act + delta_cal[a0:a1], FS)
            rec["dba_diff"] = round(dba1 - dba0, 3)
            _, stem_wr = m11._render_stem(delta_cal, np.array(src["wp"], float),
                                          mic, c)
            delta_sum = delta_sum + stem_wr
        log["per_car"].append(rec)

    new_mix = mix + delta_sum
    peak = float(np.max(np.abs(new_mix)))
    log["peak"] = round(peak, 4)
    sf.write(dst_flac, new_mix.T.astype(np.float64), fs_out, subtype=info.subtype)
    if listen_dir is not None:
        sf.write(listen_dir / f"{clip}_delta_only_W.wav",
                 delta_sum[0], fs_out, subtype="PCM_16")
        sf.write(listen_dir / f"{clip}_new_mix_W.wav",
                 new_mix[0] / max(peak, 1e-9) * 0.9, fs_out, subtype="PCM_16")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    (OUT_DS / "foa").mkdir(parents=True, exist_ok=True)
    PROBE.mkdir(parents=True, exist_ok=True)
    listen_dir = None
    if args.smoke:
        listen_dir = PROBE / "listen_smoke"
        listen_dir.mkdir(exist_ok=True)

    clips = sorted(p.stem for p in (DS / "foa").glob("fold2_*.flac"))
    if args.smoke:
        with_car = []
        for cl in clips:
            sc = json.loads((DS / "work" / cl / "scene.json").read_text())
            if any(s["class"] == "car_drive" for s in sc["sources"]):
                with_car.append(cl)
            if len(with_car) >= args.smoke:
                break
        clips = with_car

    c = m11.sound_speed(m11.TEMP_C)
    logf = open(PROBE / ("render_log_smoke.jsonl" if args.smoke
                         else "render_log.jsonl"), "a", encoding="utf-8")
    peak_warn, dba_warn = [], []
    for i, clip in enumerate(clips):
        log = process_clip(clip, c, listen_dir)
        logf.write(json.dumps(log, ensure_ascii=False) + "\n")
        if log.get("peak", 0.0) >= 0.99:
            peak_warn.append(clip)
        for rec in log.get("per_car", []):
            if abs(rec.get("dba_diff", 0.0)) >= 0.5:
                dba_warn.append((clip, rec["dba_diff"]))
        if args.smoke:
            print(json.dumps(log, ensure_ascii=False, indent=1))
        elif (i + 1) % 100 == 0:
            print(f"{i+1}/{len(clips)}", flush=True)
    logf.close()
    print(f"done {len(clips)}本 / peak>=0.99: {len(peak_warn)}本 {peak_warn[:5]} / "
          f"dBA変化>=0.5: {len(dba_warn)}件 {dba_warn[:5]}")


if __name__ == "__main__":
    main()
