"""geometry.py の単体テスト。

検証内容:
1. 静止音源: te = tr − d/c が厳密に成り立つ
2. 移動音源: 解が伝搬方程式 ||pr − ps(te)|| = c(tr − te) を満たす（残差ゼロ）
3. DynamicSound `Simulation._compute_emission` との一致（同一実装の照合）
4. 数値微分 dte/dtr がドップラー理論式 c/(c − v·u_sr) と一致（論文Eq.4, vr=0）
5. az/el 変換: 既知方向（前/左/上/右後方）と往復変換
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outdoor_seld.geometry import (SOUND_SPEED_20C, apparent_azel_deg,
                                   azel_deg_to_unit, doa_unit_vectors,
                                   solve_emission_times, unit_to_azel_deg)

C = SOUND_SPEED_20C
PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(bool(cond))
    print(f"[{status}] {name} {detail}")


def test_static_source():
    pr = np.array([0.0, 0.0, 1.5])
    ps = np.array([100.0, -30.0, 1.0])
    d = np.linalg.norm(ps - pr)
    wp = np.array([[0.0, *ps], [10.0, *ps]])
    tr = np.linspace(d / C + 0.01, 9.9, 500)
    te, ps_te = solve_emission_times(tr, wp, pr, C)
    err_te = np.max(np.abs(te - (tr - d / C)))
    err_ps = np.max(np.linalg.norm(ps_te - ps, axis=1))
    check("static: te == tr - d/c", err_te < 1e-9, f"max_err={err_te:.2e}")
    check("static: ps(te) == ps", err_ps < 1e-9, f"max_err={err_ps:.2e}")
    # 到達前は NaN
    te0, _ = solve_emission_times(np.array([0.5 * d / C]), wp, pr, C)
    check("static: pre-arrival is NaN", np.isnan(te0[0]))


def test_moving_residual():
    pr = np.array([0.0, 0.0, 1.5])
    wp = np.array([[0.0, -50.0, 5.0, 1.0], [10.0, 50.0, 5.0, 1.0]])  # 10 m/s
    tr = np.linspace(0.0, 9.999, 2000)
    te, ps_te = solve_emission_times(tr, wp, pr, C)
    ok = np.isfinite(te)
    lhs = np.linalg.norm(ps_te[ok] - pr, axis=1)
    rhs = C * (tr[ok] - te[ok])
    res = np.max(np.abs(lhs - rhs))
    check("moving: |pr-ps(te)| = c(tr-te)", res < 1e-6, f"max_residual={res:.2e} m")
    check("moving: causality te<=tr", np.all(te[ok] <= tr[ok] + 1e-12))
    t_arr = np.linalg.norm(np.array([-50, 5, 1.0]) - pr) / C
    first = tr[ok][0]
    check("moving: first arrival ~ d0/c", abs(first - t_arr) < 6e-3,
          f"first={first:.4f}s theory={t_arr:.4f}s")


def test_against_dynamicsound():
    import dynamic_sound as ds
    pr = np.array([0.0, 0.0, 1.5])
    quat = [1, 0, 0, 0]
    src_path = ds.Path([[0.0, -50, 5, 1, *quat], [10.0, 50, 5, 1, *quat]])
    wp = np.array([[0.0, -50, 5, 1.0], [10.0, 50, 5, 1.0]])
    rng = np.random.default_rng(0)
    tr = np.sort(rng.uniform(0.2, 9.99, 200))
    te_mine, ps_mine = solve_emission_times(tr, wp, pr, C)
    max_dte, max_dps = 0.0, 0.0
    for i, t in enumerate(tr):
        te_ds, ps_ds = ds.Simulation._compute_emission(
            position_receiver=pr, time_receiver=float(t),
            source_path=src_path, c=C)
        if te_ds is None:
            assert np.isnan(te_mine[i])
            continue
        max_dte = max(max_dte, abs(te_ds - te_mine[i]))
        max_dps = max(max_dps, float(np.max(np.abs(ps_ds - ps_mine[i]))))
    check("vs DynamicSound: te match", max_dte < 1e-9, f"max_dte={max_dte:.2e}")
    check("vs DynamicSound: ps(te) match", max_dps < 1e-6, f"max_dps={max_dps:.2e}")


def test_doppler_derivative():
    """dte/dtr（数値微分）とドップラー係数 c/(c − v・u_sr) の一致（受信静止）。"""
    pr = np.array([0.0, 0.0, 1.5])
    wp = np.array([[0.0, -50.0, 5.0, 1.0], [10.0, 50.0, 5.0, 1.0]])
    v = (wp[1, 1:] - wp[0, 1:]) / 10.0
    tr = np.linspace(0.5, 9.5, 400)
    h = 1e-5
    te1, ps_te = solve_emission_times(tr, wp, pr, C)
    te2, _ = solve_emission_times(tr + h, wp, pr, C)
    dte_dtr = (te2 - te1) / h
    u_sr, _ = doa_unit_vectors(np.full_like(ps_te, pr) , pr)  # placeholder
    # u_sr: 放射位置→受信点 の単位ベクトル
    d = pr[None, :] - ps_te
    u_sr = d / np.linalg.norm(d, axis=1)[:, None]
    theory = C / (C - (u_sr @ v))
    err = np.max(np.abs(dte_dtr - theory) / theory)
    check("doppler: dte/dtr == c/(c - v.u)", err < 1e-4, f"max_rel_err={err:.2e}")


def test_azel_conversions():
    cases = [((1, 0, 0), (0, 0)), ((0, 1, 0), (90, 0)), ((0, 0, 1), (0, 90)),
             ((-1, 0, 0), (180, 0)), ((0, -1, 0), (-90, 0))]
    ok = True
    for u, (az_t, el_t) in cases:
        az, el = unit_to_azel_deg(np.array([u], dtype=float))
        if not (abs(az[0] - az_t) < 1e-9 or abs(az_t) == 180) or abs(el[0] - el_t) > 1e-9:
            ok = False
    check("azel: known directions", ok)
    rng = np.random.default_rng(1)
    az = rng.uniform(-179.9, 179.9, 100)
    el = rng.uniform(-89.9, 89.9, 100)
    u = azel_deg_to_unit(az, el)
    az2, el2 = unit_to_azel_deg(u)
    check("azel: round-trip", np.max(np.abs(az2 - az)) < 1e-9
          and np.max(np.abs(el2 - el)) < 1e-9)
    # 進行方向の見かけの方位: t=5s(最接近 az=90) では放射時刻補正で
    # 音源は少し「過去」= まだ左後方寄りに見えるはず (az > 90)
    pr = np.array([0.0, 0.0, 1.5])
    wp = np.array([[0.0, -50.0, 5.0, 1.0], [10.0, 50.0, 5.0, 1.0]])
    az5, el5, te5, dist5 = apparent_azel_deg(np.array([5.0]), wp, pr, C)
    check("apparent DOA lags geometric at CPA", az5[0] > 90.0,
          f"az(t=5)={az5[0]:.2f} deg (geometric would be 90.00)")


if __name__ == "__main__":
    test_static_source()
    test_moving_residual()
    test_against_dynamicsound()
    test_doppler_derivative()
    test_azel_conversions()
    n_fail = PASS.count(False)
    print(f"\n{len(PASS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
