"""noise.py の単体テスト: 拡散音場の統計・SNRの正確さ・スペクトル形状。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outdoor_seld.noise import (colored_noise, diffuse_foa_noise,  # noqa: E402
                                measure_snr_db, mix_at_snr)

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(bool(cond))
    print(f"[{status}] {name} {detail}")


def test_diffuse_statistics():
    rng = np.random.default_rng(0)
    n, fs = 240000, 24000
    foa = diffuse_foa_noise(n, fs, rng)
    pw = np.mean(foa[0] ** 2)
    ratios = [np.mean(foa[i] ** 2) / pw for i in (1, 2, 3)]
    check("diffuse: E|Y,Z,X|^2 / E|W|^2 ~ 1/3",
          all(abs(r - 1 / 3) < 0.03 for r in ratios),
          f"ratios={[f'{r:.3f}' for r in ratios]}")
    cors = [abs(np.corrcoef(foa[i], foa[j])[0, 1])
            for i in range(4) for j in range(i + 1, 4)]
    check("diffuse: channels uncorrelated", max(cors) < 0.02,
          f"max|corr|={max(cors):.4f}")


def test_snr_accuracy():
    rng = np.random.default_rng(1)
    n, fs = 240000, 24000
    sig = np.zeros((4, n))
    sig[0] = 0.05 * np.sin(2 * np.pi * 960 * np.arange(n) / fs)
    sig[1] = sig[0] * 0.7
    for target in [0.0, 6.0, 12.0, 20.0]:
        noise = diffuse_foa_noise(n, fs, rng)
        noisy, g = mix_at_snr(sig, noise, target)
        meas = measure_snr_db(sig, noisy)
        check(f"SNR target {target:.0f} dB", abs(meas - target) < 0.01,
              f"measured={meas:.3f} dB gain={g:.5f}")


def test_spectrum_shape():
    rng = np.random.default_rng(2)
    n, fs = 480000, 24000
    x = colored_noise(n, fs, rng, slope=1.0)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)

    def band_power(f0, f1):
        m = (f >= f0) & (f < f1)
        return np.mean(spec[m])

    # ピンク: パワー1/f → 1ディケードで -10 dB
    ratio_db = 10 * np.log10(band_power(100, 200) / band_power(1000, 2000))
    check("pink: -10 dB/decade", abs(ratio_db - 10.0) < 1.5,
          f"measured={ratio_db:.2f} dB")
    sub = np.sum(spec[f < 20.0]) / np.sum(spec)
    check("pink: no energy below 20 Hz", sub < 1e-6, f"frac={sub:.2e}")


if __name__ == "__main__":
    test_diffuse_statistics()
    test_snr_accuracy()
    test_spectrum_shape()
    n_fail = PASS.count(False)
    print(f"\n{len(PASS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
