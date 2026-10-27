# -*- coding: utf-8 -*-
"""自転車の走行音（bicycle 試作・src/outdoor_seld/bicycle.py）の試聴用 wav を out/audition_2026-10-07/ に書く（2026-10-07）。

- dry_*: ドライ音源そのもの（10 秒・速さ別・漕ぐ／空転）。ピーク 0.5 に揃える（絶対レベルは未定なので音色だけを聞く）。
- passby_*: 簡易の通り過ぎ（横 1 m／3 m・時速 15 km・前から後ろへ・マイク高さ 1.6 m・音源高さ 0.3 m）。
  距離の減衰（1/r）だけで、ドップラー・空気吸収・反射は無し。本番の描画（step11・FOA・反射・暗騒音）とは別物。
- compare_*: 比べる用に、今のキック（make_kickboard・4 m/s）と車（make_car_v9）を同じ扱いで。
- passby_bell_*: 通り過ぎにベル（make_bike_bell）を重ねた例。レベル差は設計値（自転車 50 dB(A)@1 m・ベル 85 dB(A)@1 m・35 dB 差）。
実行: PSELDNets の venv の python で（numpy・scipy が要る）。リポジトリの既存ファイルは変えない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from outdoor_seld.bicycle import make_bicycle  # noqa: E402
from outdoor_seld.kickboard import make_kickboard  # noqa: E402
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.alert_sounds import make_bike_bell  # noqa: E402
from outdoor_seld.calibration import gain_for_spl_a  # noqa: E402

FS = 48000
DUR = 10.0
OUT = ROOT / "out" / "audition_2026-10-07"
OUT.mkdir(parents=True, exist_ok=True)


def save(name: str, x: np.ndarray, peak: float = 0.5):
    m = float(np.max(np.abs(x))) or 1.0
    y = (peak * x / m * 32767).astype(np.int16)
    wavfile.write(OUT / name, FS, y)
    print("wrote", name)


def passby(dry: np.ndarray, v_mps: float, lateral_m: float, mic_z: float = 1.6, src_z: float = 0.3, t_cpa: float = 5.0):
    """簡易の通り過ぎ: 1/r の減衰だけ（r は 3 次元距離・1 m を基準）。"""
    n = len(dry)
    t = np.arange(n) / FS
    x = v_mps * (t - t_cpa)
    r = np.sqrt(x ** 2 + lateral_m ** 2 + (mic_z - src_z) ** 2)
    return dry / np.maximum(r, 0.3)


def main():
    rng = np.random.default_rng(20261007)
    lines = ["自転車の走行音（試作）の試聴ファイル 2026-10-07。全部ピーク 0.5。絶対レベルは未定（設計値）。", ""]
    for kmh in (8, 12, 15, 20):
        v = kmh / 3.6
        x = make_bicycle(DUR, FS, rng, speed_mps=v, coast_frac=0.3, teeth=24, cadence_rpm=70.0)
        save(f"dry_bicycle_{kmh:02d}kmh_mix.wav", x)
        lines.append(f"dry_bicycle_{kmh:02d}kmh_mix.wav: 時速 {kmh} km・漕ぐと空転が混ざる（空転 3 割・24 歯・ケイデンス 70）")
    x = make_bicycle(DUR, FS, rng, speed_mps=15 / 3.6, coast_frac=0.0)
    save("dry_bicycle_15kmh_pedal_only.wav", x); lines.append("dry_bicycle_15kmh_pedal_only.wav: 時速 15 km・ずっと漕ぐ（クリック無し）")
    x = make_bicycle(DUR, FS, rng, speed_mps=15 / 3.6, coast_frac=1.0)
    save("dry_bicycle_15kmh_coast_only.wav", x); lines.append("dry_bicycle_15kmh_coast_only.wav: 時速 15 km・ずっと空転（クリック約 48 回/秒）")
    x = make_bicycle(DUR, FS, rng, speed_mps=15 / 3.6, coast_frac=1.0, teeth=36)
    save("dry_bicycle_15kmh_coast_36teeth.wav", x); lines.append("dry_bicycle_15kmh_coast_36teeth.wav: 同上・36 歯（クリック約 72 回/秒）")
    x = make_bicycle(DUR, FS, rng, speed_mps=12 / 3.6, coast_frac=0.3, rattle_rate=0.3)
    save("dry_bicycle_12kmh_rattle.wav", x); lines.append("dry_bicycle_12kmh_rattle.wav: 時速 12 km・カゴ／泥除けのガタつきあり（0.3 回/秒・既定はオフ）")
    for lat in (1.0, 3.0):
        x = make_bicycle(DUR, FS, rng, speed_mps=15 / 3.6, coast_frac=0.3)
        save(f"passby_bicycle_15kmh_{lat:.0f}m.wav", passby(x, 15 / 3.6, lat))
        lines.append(f"passby_bicycle_15kmh_{lat:.0f}m.wav: 時速 15 km・横 {lat:.0f} m を通り過ぎる（簡易・距離の減衰だけ）")
    # 比べる用
    xk = make_kickboard(DUR, FS, rng, speed_mps=4.0)
    save("compare_kickboard_4mps_dry.wav", xk); lines.append("compare_kickboard_4mps_dry.wav: 今のキックボード（4 m/s）のドライ音源")
    xc = make_car_v9(DUR, FS, rng, f0=42.0)
    save("compare_car_dry.wav", xc); lines.append("compare_car_dry.wav: 今の車（make_car_v9・f0 42 Hz）のドライ音源")
    # ベルあり（設計値のレベル差で重ねる）
    xb = make_bicycle(DUR, FS, rng, speed_mps=15 / 3.6, coast_frac=0.3)
    bell = make_bike_bell(DUR, FS, f0=3000.0, n_rings=2, ring_gap_sec=0.45, repeat_period_sec=4.0)
    gb = gain_for_spl_a(xb, FS, 50.0)
    gbell = gain_for_spl_a(bell, FS, 85.0)
    mix = passby(gb * xb, 15 / 3.6, 1.0) + passby(gbell * bell, 15 / 3.6, 1.0)
    save("passby_bell_bicycle_15kmh_1m.wav", mix); lines.append("passby_bell_bicycle_15kmh_1m.wav: 横 1 m の通り過ぎにベル（自転車 50 dB(A)@1m・ベル 85 dB(A)@1m の設計値の差で重ねた）")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
