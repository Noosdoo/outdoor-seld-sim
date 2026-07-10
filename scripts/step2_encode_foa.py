"""Step 2: 解析エンコードによる FOA 変換（★本研究の中核）。

- 物理適用済みモノラル（直接／鏡像）を 24 kHz にデシメート
- 各音源の「受信時刻における見かけのDOA」（放射時刻補正、geometry.py）から
  時変ゲインで FOA (W,Y,Z,X / SN3D) を生成
- 直接音のみ版と、直接＋地面反射（鏡像を加算）版の 2 種を FLAC 出力

出力: out/clip/{foa_direct_24k.flac, foa_withrefl_24k.flac} + clip_stats.json 追記
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out" / "clip"

from outdoor_seld.foa import encode_foa_timevarying  # noqa: E402
from outdoor_seld.geometry import (SOUND_SPEED_20C, apparent_azel_deg,  # noqa: E402
                                   azel_deg_to_unit, solve_emission_times,
                                   doa_unit_vectors, sound_speed)
from outdoor_seld.scene import SceneConfig, decimate_to_out_rate  # noqa: E402


def encode_one(mono_24k: np.ndarray, waypoints, mic_pos, fs_out: int, c: float):
    tr = np.arange(len(mono_24k)) / fs_out
    te, ps_te = solve_emission_times(tr, waypoints, np.array(mic_pos), c)
    u, dist = doa_unit_vectors(ps_te, np.array(mic_pos))
    return encode_foa_timevarying(mono_24k, u)


def main():
    scene = SceneConfig(**json.loads((OUT / "scene_config.json").read_text()))
    c = sound_speed(scene.temperature_c)

    monos = {}
    for tag in ["direct", "mirror"]:
        x, sr = sf.read(OUT / f"mono_{tag}_48k.wav")
        assert sr == scene.fs_sim
        monos[tag] = decimate_to_out_rate(np.asarray(x, np.float64),
                                          scene.fs_sim, scene.fs_out)

    foa_direct = encode_one(monos["direct"], scene.waypoints_direct(),
                            scene.mic_pos, scene.fs_out, c)
    foa_mirror = encode_one(monos["mirror"], scene.waypoints_mirror(),
                            scene.mic_pos, scene.fs_out, c)
    foa_withrefl = foa_direct + foa_mirror

    stats = json.loads((OUT / "clip_stats.json").read_text())
    for name, foa in [("foa_direct_24k", foa_direct),
                      ("foa_withrefl_24k", foa_withrefl)]:
        peak = float(np.max(np.abs(foa)))
        assert peak < 0.99, f"clipping risk: peak={peak}"
        path = OUT / f"{name}.flac"
        sf.write(path, foa.T.astype(np.float32), scene.fs_out, subtype="PCM_24")
        stats[name] = {
            "peak": peak,
            "rms_per_ch_WYZX": [float(np.sqrt(np.mean(ch ** 2))) for ch in foa],
            "fs": scene.fs_out, "n_samples": int(foa.shape[1]),
            "channel_order": "W,Y,Z,X (ACN)", "normalization": "SN3D",
        }
        print(f"{path.name}: peak={peak:.4f} "
              f"rms(W,Y,Z,X)={[f'{v:.4f}' for v in stats[name]['rms_per_ch_WYZX']]}")

    # 参考: 最接近時刻付近の見かけDOA（ログ用）
    az, el, te, dist = apparent_azel_deg(
        np.array([0.5, 2.5, 5.0, 7.5, 9.5]), scene.waypoints_direct(),
        np.array(scene.mic_pos), c)
    stats["apparent_doa_samples"] = [
        {"t": t, "az_deg": round(float(a), 2), "el_deg": round(float(e), 2),
         "dist_m": round(float(d), 2)}
        for t, a, e, d in zip([0.5, 2.5, 5.0, 7.5, 9.5], az, el, dist)]
    (OUT / "clip_stats.json").write_text(json.dumps(stats, indent=2))
    print("updated clip_stats.json")


if __name__ == "__main__":
    main()
