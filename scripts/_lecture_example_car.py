# -*- coding: utf-8 -*-
"""講義ノート「矢印の長さと閾値」（md/thesis・2026-09-17）の例 1〜6 の数値を計算する（左 2 m を通る車・マイク高さ 1.6 m・音源高さ 0.5 m・5 m/s）。
使い方: python scripts/_lecture_example_car.py"""
import math
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
LOG_K = math.log1p(100.0)
MIC_Z, SRC_Z, V, Y = 1.6, 0.5, 5.0, 2.0


def state(t):
    x = 10.0 - V * t
    dz = SRC_Z - MIC_Z
    dh = math.hypot(x, Y)
    az = math.degrees(math.atan2(Y, x))
    el = math.degrees(math.atan2(dz, dh))
    u = (math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el)))
    d0 = math.log1p(dh) / LOG_K
    return x, dh, az, el, u, d0


print("== 正解（t, x, 水平距離, az, el, u, d0）")
for t in (0.0, 1.0, 1.5, 1.6, 2.0, 2.4, 3.0):
    x, dh, az, el, u, d0 = state(t)
    print(f"t={t:.1f}s x={x:+.1f} m  dist_h={dh:.2f} m  az={az:.1f}°  el={el:.1f}°  u=({u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f})  d0={d0:.3f}  遅延={dh/343:.3f}s")


def decode(a, d):
    n = math.sqrt(sum(v * v for v in a))
    az = math.degrees(math.atan2(a[1], a[0]))
    el = math.degrees(math.atan2(a[2], math.hypot(a[0], a[1])))
    dist = math.expm1(d * LOG_K)
    return n, az, el, dist


def loss(a, u, d, d0):
    lx = sum((ai - ui) ** 2 for ai, ui in zip(a, u)) / 3
    ld = (d - d0) ** 2
    return lx, ld, lx + 3.0 * ld


x, dh, az, el, u, d0 = state(2.0)
print("\n== t=2.0 s の正解", u, d0)
for name, a, d in [("A 自信あり", (0.05, 0.85, -0.45), 0.25), ("B 弱い", (0.45, 0.15, -0.05), 0.20), ("A2 枠2", (0.02, 0.80, -0.40), 0.27), ("A3 枠3", (0.30, 0.10, 0.00), 0.40)]:
    n, paz, pel, dist = decode(a, d)
    lx, ld, L = loss(a, u, d, d0)
    print(f"{name}: a={a} d={d} → ‖a‖={n:.3f} az={paz:.1f}° el={pel:.1f}° dist={dist:.2f} m | L_xyz={lx:.4f} L_d={ld:.5f} L={L:.4f}")

# 統合: 枠1 と 枠2 の平均
a1, a2 = (0.05, 0.85, -0.45), (0.02, 0.80, -0.40)
am = tuple((p + q) / 2 for p, q in zip(a1, a2))
n, paz, pel, _ = decode(am, 0.26)
print(f"統合 (枠1+枠2)/2 = ({am[0]:.3f}, {am[1]:.3f}, {am[2]:.3f}) → ‖a‖={n:.3f} az={paz:.1f}° 距離=(2.17+2.47)/2")
d1 = math.expm1(0.25 * LOG_K); d2 = math.expm1(0.27 * LOG_K)
print(f"  距離 枠1 {d1:.2f} m 枠2 {d2:.2f} m 平均 {(d1+d2)/2:.2f} m")
# 角度差 枠1-枠2
def angdiff(a, b):
    na = np.array(a) / np.linalg.norm(a); nb = np.array(b) / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(na, nb), -1, 1))))
print(f"  枠1 と 枠2 の角度差 {angdiff(a1, a2):.1f}°（15° 以内 → 統合）、枠1 と 枠3 {angdiff(a1, (0.3,0.1,0.0)):.1f}°")

# 前後の迷い: t=1.0 s (az 14°) は前だが、モデルが 前(14°) と 後(166°) で半々
x, dh, az, el, u1, d0 = state(1.0)
az2 = 180 - az
u2 = (math.cos(math.radians(el)) * math.cos(math.radians(az2)), math.cos(math.radians(el)) * math.sin(math.radians(az2)), math.sin(math.radians(el)))
avg = tuple((p + q) / 2 for p, q in zip(u1, u2))
n, paz, pel, _ = decode(avg, 0.0)
print(f"\n== 前後の迷い t=1.0 s: 正解 az={az:.1f}° el={el:.1f}° u1=({u1[0]:.3f},{u1[1]:.3f},{u1[2]:.3f})  裏 az={az2:.1f}° u2=({u2[0]:.3f},{u2[1]:.3f},{u2[2]:.3f})")
print(f"  半々の平均 = ({avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}) → ‖a‖={n:.3f} az={paz:.1f}°（cos(φ/2) with φ={angdiff(u1,u2):.1f}° → {math.cos(math.radians(angdiff(u1,u2)/2)):.3f}）")
# 7:3
avg73 = tuple(0.7 * p + 0.3 * q for p, q in zip(u1, u2))
n, paz, pel, _ = decode(avg73, 0.0)
print(f"  7:3 の平均 = ({avg73[0]:.3f}, {avg73[1]:.3f}, {avg73[2]:.3f}) → ‖a‖={n:.3f} az={paz:.1f}°")

# 距離規則の例
for dh in (1.2, 2.0, 2.83, 3.2, 5.0):
    print(f"距離 {dh} m → d0={math.log1p(dh)/LOG_K:.3f}")
print("3.2 m に入る時刻:", (10 - math.sqrt(3.2**2 - 4)) / 5, "s ; 3.2 m を出る時刻:", (10 + math.sqrt(3.2**2 - 4)) / 5, "s")
