"""Step 1: シーン定義と1クリップ生成。

- 合成サイレン（ドライ）を 48 kHz で生成
- DynamicSound で 2 回シミュレーション（直接音源／地面反射=鏡像音源、各モノラル48kHz）
- シーン設定（r0=1m 含む）と RMS を JSON に記録

出力: out/clip/{siren_dry_48k.wav, mono_direct_48k.wav, mono_mirror_48k.wav,
               scene_config.json, clip_stats.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out" / "clip"
OUT.mkdir(parents=True, exist_ok=True)

import tqdm as _tqdm_mod  # noqa: E402

_tqdm_mod.tqdm = lambda iterable=None, **kw: iterable  # type: ignore

import soundfile as sf  # noqa: E402

from outdoor_seld.scene import SceneConfig, run_mono_sim  # noqa: E402
from outdoor_seld.siren import make_siren  # noqa: E402


def main():
    scene = SceneConfig()
    scene.save_json(OUT / "scene_config.json")

    # ドライ音源（クリップ長と同じ10秒。放射時刻 te は常に < clip_len なので足りる）
    dry = make_siren(scene.clip_len_sec, scene.fs_sim)
    dry_path = OUT / "siren_dry_48k.wav"
    sf.write(dry_path, dry.astype(np.float32), scene.fs_sim)
    print(f"dry siren: {dry_path.name} rms={np.sqrt(np.mean(dry**2)):.4f} "
          f"peak={np.max(np.abs(dry)):.3f}")

    stats = {"r0_m": scene.r0_m,
             "note_r0": "DynamicSound attenuations.geometric=1/distance (gain 1 at 1 m, fixed)",
             "dry_rms": float(np.sqrt(np.mean(dry ** 2))),
             "dry_peak": float(np.max(np.abs(dry)))}

    for tag, mirror in [("direct", False), ("mirror", True)]:
        out_wav = OUT / f"mono_{tag}_48k.wav"
        t0 = time.perf_counter()
        run_mono_sim(scene, str(dry_path), str(out_wav), mirror=mirror)
        dt = time.perf_counter() - t0
        x, sr = sf.read(out_wav)
        x = np.asarray(x, dtype=np.float64)
        stats[f"mono_{tag}"] = {
            "rms": float(np.sqrt(np.mean(x ** 2))),
            "peak": float(np.max(np.abs(x))),
            "sim_seconds": round(dt, 1),
            "n_samples": int(len(x)), "fs": int(sr),
        }
        print(f"{tag}: rms={stats[f'mono_{tag}']['rms']:.5f} "
              f"peak={stats[f'mono_{tag}']['peak']:.4f} ({dt:.0f}s)")

    (OUT / "clip_stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote scene_config.json / clip_stats.json")


if __name__ == "__main__":
    main()
