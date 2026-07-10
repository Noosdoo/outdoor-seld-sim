"""foa.py の単体テスト（サニティチェック1: FOA規約の確認を含む）。

- 既知方向の静的エンコードでチャンネルゲインが規約どおりか
  （az=+90°,el=0° → Y=W, X≈0, Z≈0 なら ACN[W,Y,Z,X]/SN3D が正しい。
   取り違え（FuMa順やN3D）ならここで即検出される）
- インテンシティベクトル法DOAがエンコードDOAを復元するか（自己整合）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outdoor_seld.foa import (encode_foa_static, encode_foa_timevarying,
                              intensity_vector_doa)

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(bool(cond))
    print(f"[{status}] {name} {detail}")


def test_static_encoding_convention():
    fs = 24000
    rng = np.random.default_rng(0)
    mono = rng.standard_normal(fs)  # 1s noise

    foa = encode_foa_static(mono, az_deg=90.0, el_deg=0.0)  # 真横=左
    w, y, z, x = foa
    check("az=90: W == mono", np.allclose(w, mono))
    check("az=90: Y == W (gain 1)", np.allclose(y, w, atol=1e-12))
    check("az=90: X ~ 0", np.max(np.abs(x)) < 1e-12)
    check("az=90: Z ~ 0", np.max(np.abs(z)) < 1e-12)

    foa = encode_foa_static(mono, az_deg=0.0, el_deg=0.0)  # 正面
    check("az=0: X == W", np.allclose(foa[3], foa[0]))
    check("az=0: Y ~ 0, Z ~ 0",
          np.max(np.abs(foa[1])) < 1e-12 and np.max(np.abs(foa[2])) < 1e-12)

    foa = encode_foa_static(mono, az_deg=0.0, el_deg=90.0)  # 真上
    check("el=90: Z == W", np.allclose(foa[2], foa[0]))

    foa = encode_foa_static(mono, az_deg=45.0, el_deg=30.0)
    g = np.array([np.cos(np.pi/6)*np.cos(np.pi/4),
                  np.cos(np.pi/6)*np.sin(np.pi/4),
                  np.sin(np.pi/6)])  # (x,y,z)
    check("az=45,el=30: SN3D gains",
          np.allclose(foa[1], g[1]*mono) and np.allclose(foa[2], g[2]*mono)
          and np.allclose(foa[3], g[0]*mono))


def test_iv_doa_recovers_encoding():
    fs = 24000
    rng = np.random.default_rng(1)
    mono = rng.standard_normal(fs * 2)
    for az_t, el_t in [(0, 0), (90, 0), (-135, 20), (30, -30)]:
        foa = encode_foa_static(mono, az_deg=az_t, el_deg=el_t)
        _, az, el, _ = intensity_vector_doa(foa, fs, fmin=200, fmax=8000)
        az_m, el_m = np.nanmedian(az), np.nanmedian(el)
        d_az = abs((az_m - az_t + 180) % 360 - 180)
        check(f"IV-DOA recovers ({az_t},{el_t})",
              d_az < 1.0 and abs(el_m - el_t) < 1.0,
              f"got ({az_m:.2f},{el_m:.2f})")


def test_timevarying_nan_handling():
    mono = np.ones(100)
    u = np.full((100, 3), np.nan)
    u[50:, :] = [1.0, 0.0, 0.0]
    foa = encode_foa_timevarying(mono, u)
    check("NaN DOA -> zero gains (W untouched)",
          np.all(foa[1:, :50] == 0.0) and np.allclose(foa[3, 50:], 1.0)
          and np.allclose(foa[0], 1.0))


if __name__ == "__main__":
    test_static_encoding_convention()
    test_iv_doa_recovers_encoding()
    test_timevarying_nan_handling()
    n_fail = PASS.count(False)
    print(f"\n{len(PASS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
