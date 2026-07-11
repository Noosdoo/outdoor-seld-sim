"""fastsim.py の等価性検証: DynamicSound の実出力との波形比較。

試作クリップ（DynamicSound で生成済みの mono_direct/mirror wav）と、
同一シーン条件を fastsim.render_mono で再現した波形を比較する。
許容差はブロック一定FIR化とint32量子化に由来する微小差のみ。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.scene import SceneConfig  # noqa: E402
from outdoor_seld.siren import make_peepo_siren, make_siren  # noqa: E402

DS_DIR = Path(__file__).resolve().parents[1] / "out" / "dataset_outdoor_siren_v1"
PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(bool(cond))
    print(f"[{status}] {name} {detail}")


def compare_clip(name: str):
    s = json.loads((DS_DIR / "work" / name / "scene.json").read_text())
    scene = SceneConfig(**s["scene_config"])
    gen = make_peepo_siren if s["siren_type"] == "peepo" else make_siren
    dry = gen(scene.clip_len_sec, scene.fs_sim, **s["siren_params"])

    t0 = time.perf_counter()
    for tag, wp in [("direct", scene.waypoints_direct()),
                    ("mirror", scene.waypoints_mirror())]:
        ref_path = DS_DIR / "work" / name / f"mono_{tag}_48k_DSref.wav"
        if not ref_path.exists():  # 保全リネーム前の旧名にフォールバック
            ref_path = DS_DIR / "work" / name / f"mono_{tag}_48k.wav"
        ref, sr = sf.read(ref_path)
        ref = np.asarray(ref, np.float64)
        mine = render_mono(dry, wp, scene.mic_pos, scene.fs_sim,
                           scene.clip_len_sec,
                           temperature_c=scene.temperature_c,
                           pressure_atm=scene.pressure_atm,
                           rel_humidity=scene.rel_humidity,
                           gain_db=scene.source_gain_db)
        nmin = min(len(ref), len(mine))
        diff = mine[:nmin] - ref[:nmin]
        rel = float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(ref ** 2)))
        maxd = float(np.max(np.abs(diff)))
        check(f"{name} {tag}: waveform match", rel < 5e-3,
              f"rel_rms={rel:.2e} max_abs_diff={maxd:.2e}")
    dt = time.perf_counter() - t0
    print(f"  render time (direct+mirror): {dt:.1f}s "
          f"(DynamicSound was ~{2 * s['stats']['sim_seconds']:.0f}s... "
          f"recorded {s['stats']['sim_seconds']:.0f}s total)")


if __name__ == "__main__":
    for name in ["fold1_room1_mix001", "fold1_room1_mix002", "fold1_room1_mix003"]:
        compare_clip(name)
    n_fail = PASS.count(False)
    print(f"\n{len(PASS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
