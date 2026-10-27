# -*- coding: utf-8 -*-
"""通知v4（TTC規則）の単体テスト（2026-08-18）。

V1 : 速度推定＝最小二乗の傾き（等速接近で正確・遠ざかりで負）
V2 : 窓が埋まるまでは None（判定しない＝検知遅れの源）
V3 : TTC = (d − r_danger)/v。遠ざかり・低速・遠すぎは None
V4 : **速いトラックを遠くで捕まえる**（距離規則では黙る場面で鳴る）
V5 : **遠ざかる相手には鳴らさない**（距離規則は鳴らしうる）
V6 : 既に危険域内なら速度が測れなくても鳴る（距離規則の保険が効く）
V7 : 距離規則(--rule dist)がv3.4と同じ挙動（回帰）
V8 : エピソード統合は v3.4 と同一規則（フレーム差≤1かつ方位差≤25°）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "nv4", ROOT / "scripts" / "step12_notify_v4_ttc.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

fails, n = [], 0


def check(name, cond, note=""):
    global n
    n += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        fails.append(name)


def approach(d0, v, nframes=40):
    """等速で近づく距離系列（フレーム0でd0、毎フレーム v/FPS ずつ減る）。"""
    return {j: max(0.1, d0 - v * j / v4.FPS) for j in range(nframes)}


# ---- V1: 速度推定 ----
d_in = approach(20.0, 10.0)
d_out = {j: 2.0 + 5.0 * j / v4.FPS for j in range(40)}
v_in = v4.closing_speed(d_in, 10)
v_out = v4.closing_speed(d_out, 10)
check("V1 速度推定（接近+10m/s・離反−5m/s）",
      abs(v_in - 10.0) < 0.05 and abs(v_out + 5.0) < 0.05,
      f"(接近={v_in:.2f}, 離反={v_out:.2f})")

# ---- V2: 窓が埋まるまでNone ----
check("V2 窓未充足ではNone（＝検知遅れの源）",
      v4.closing_speed(d_in, v4.VEL_WIN - 2) is None
      and v4.closing_speed(d_in, v4.VEL_WIN - 1) is not None,
      f"(VEL_WIN={v4.VEL_WIN})")

# ---- V3: TTCの計算と除外条件 ----
t_ok = v4.ttc_of(20.0, 10.0)            # (20-1.5)/10 = 1.85s
t_recede = v4.ttc_of(2.0, -3.0)         # 遠ざかり → None
t_slow = v4.ttc_of(20.0, 0.1)           # 低速 → None
t_far = v4.ttc_of(50.0, 10.0)           # 遠すぎ → None
check("V3 TTC計算と除外（遠ざかり/低速/遠すぎはNone）",
      abs(t_ok - 1.85) < 1e-9 and t_recede is None
      and t_slow is None and t_far is None, f"(TTC={t_ok:.2f}s)")

# ---- V4: 速いトラックを遠くで捕まえる ----
# 40m先から14m/s（時速50km）で接近。距離規則は3.2m以内まで黙る
d_truck = approach(40.0, 14.0, nframes=60)
az = {j: 0.0 for j in range(60)}
f_ttc = v4.fires_ttc(d_truck, az, 60)
f_dist = v4.fires_dist(d_truck, az, 60)
d_ttc_fire = f_ttc[0][3] if f_ttc else None
d_dist_fire = f_dist[0][3] if f_dist else None
check("V4 【本命】速いトラックをTTC規則は遠くで捕まえる",
      d_ttc_fire is not None and d_dist_fire is not None
      and d_ttc_fire > d_dist_fire + 10.0,
      f"(TTC規則={d_ttc_fire:.1f}m / 距離規則={d_dist_fire:.1f}m)")
lead_gain = (d_ttc_fire - d_dist_fire) / 14.0
check("V4b リードタイムの伸び（14m/s換算・1秒以上）",
      lead_gain > 1.0, f"(+{lead_gain:.2f}秒)")
# D_MAX_TTC が効いていること（30m超では鳴らない＝遠方の距離誤差への保険）
check("V4c D_MAX_TTC=30mより遠くでは鳴らない",
      d_ttc_fire <= v4.D_MAX_TTC + 1e-6, f"(発火={d_ttc_fire:.1f}m)")

# ---- V5: 遠ざかる相手 ----
# 1.4mから3m/sで遠ざかる。距離規則は「一度は危険域内」なので強で鳴る。
# TTC規則は接近していないのでTTC由来では鳴らず、距離保険のみが働く。
d_away = {j: 1.4 + 3.0 * j / v4.FPS for j in range(40)}
f_ttc_away = v4.fires_ttc(d_away, az, 40)
f_dist_away = v4.fires_dist(d_away, az, 40)
# 遠ざかる相手では両規則とも「距離が近い間」しか鳴らない＝発火時距離が同じはず
same_dist = (bool(f_ttc_away) and bool(f_dist_away)
             and abs(f_ttc_away[0][3] - f_dist_away[0][3]) < 1e-9)
check("V5 遠ざかる相手ではTTCが効かず距離保険のみ（両規則が一致）",
      same_dist, f"(ttc={f_ttc_away[0] if f_ttc_away else None} / "
                 f"dist={f_dist_away[0] if f_dist_away else None})")
# 接近側と比べると、TTC規則の利得は接近時にだけ出ることを確認
check("V5b TTCの利得は接近時のみ（遠ざかりでは差ゼロ）",
      abs((f_ttc_away[0][3] if f_ttc_away else 0)
          - (f_dist_away[0][3] if f_dist_away else 0)) < 1e-9
      and d_ttc_fire > d_dist_fire,
      "(接近では差あり・離反では差なし)")

# ---- V6: 既に危険域内なら速度不明でも鳴る（保険） ----
d_inside = {0: 1.0, 1: 1.0, 2: 1.0}     # 窓が埋まらない＝速度None
f_inside = v4.fires_ttc(d_inside, {j: 0.0 for j in d_inside}, 3)
check("V6 危険域内は速度が測れなくても強で鳴る（距離保険）",
      len(f_inside) == 1 and f_inside[0][2] == "強", f"({f_inside})")

# ---- V7: 距離規則の回帰（v3.4と同じしきい値・確認フレーム） ----
d_reg = {0: 3.5, 1: 3.0, 2: 2.9, 3: 1.4, 4: 1.3}
f_reg = v4.fires_dist(d_reg, {j: 0.0 for j in d_reg}, 5)
# 3.0,2.9で中が2連続→中成立(j=2)。1.4,1.3で強が2連続→強成立(j=4)。
# 同一エピソード内なので**強へ昇格**し、発火時刻は強成立の j=4（v3.4と同一規約）
check("V7 強/中は独立列・同一エピソード内なら強へ昇格（v3.4規約）",
      len(f_reg) == 1 and f_reg[0][2] == "強" and f_reg[0][0] == 4,
      f"({[(x[0], x[2]) for x in f_reg]})")

# ---- V8: エピソード統合 ----
eps_gap = v4.group_episodes([(10, 0.0, "中", None, None, 1.0),
                             (11, 0.0, "中", None, None, 1.0),
                             (20, 0.0, "中", None, None, 1.0)])
eps_az = v4.group_episodes([(10, 0.0, "中", None, None, 1.0),
                            (11, 50.0, "中", None, None, 1.0)])
check("V8 統合規則がv3.4と同一（フレーム差>1で分割・方位差>25°で分割）",
      len(eps_gap) == 2 and len(eps_az) == 2,
      f"(gap={len(eps_gap)}, az={len(eps_az)})")

# ---- V9〜V12: v4.1（最接近予測）----
# TTC規則が壊れた原因は「近づいてさえいれば鳴る」こと。横をすり抜ける相手にも鳴った。
# v4.1 は**このまま進むとどこまで近づくか**を解くので、その差が出るはずである。


def passby(b, v, d0=30.0, nframes=80):
    """横オフセット b [m] で等速 v [m/s] ですれ違う軌跡の (距離, 方位)。

    直線 x(t)=b, y(t)=y0−v·t（受聴者は原点）。b=0 なら正面衝突コース。
    """
    y0 = float(np.sqrt(max(d0 * d0 - b * b, 0.0)))
    d_at, az_at = {}, {}
    for j in range(nframes):
        y = y0 - v * j / v4.FPS
        d_at[j] = float(np.hypot(b, y))
        az_at[j] = float(np.degrees(np.arctan2(b, y)))
    return d_at, az_at


# 幾何そのものの検算（規則を通す前に、式が正しいことを確かめる）
dc0, tc0 = v4.cpa_of(20.0, -14.0, 0.0)              # 正面・14m/s
b_, d_, v_ = 5.0, 20.0, 14.0
vt = v_ * b_ / d_
vr = -v_ * float(np.sqrt(1 - (b_ / d_) ** 2))
dc5, tc5 = v4.cpa_of(d_, vr, vt / d_)               # 5m横をすり抜け
# t_cpa は「最接近する時刻」まで（危険域に入るまでの TTC とは別物）
check("V9 最接近の式（正面=0m / 5m横すれ違い=5m・到達20/14秒）",
      abs(dc0) < 1e-6 and abs(tc0 - 20.0 / 14.0) < 1e-6 and abs(dc5 - 5.0) < 1e-6,
      f"(正面 {dc0:.3f}m {tc0:.3f}s / 横 {dc5:.3f}m {tc5:.3f}s)")

# 本命: 正面衝突コースには鳴り、5m横を通るだけの相手には鳴らない
d_head, az_head = passby(0.3, 14.0)
d_side, az_side = passby(5.0, 14.0)
f_head = v4.fires_cpa(d_head, az_head, 80)
f_side = v4.fires_cpa(d_side, az_side, 80)
head_d = f_head[0][3] if f_head else None
check("V10 【本命】正面に来る相手には遠くで鳴る",
      bool(f_head) and head_d is not None and head_d > v4.SUPP,
      f"(発火時の距離 {head_d if head_d is None else round(head_d, 1)}m)")
check("V11 【本命】5m横を通り過ぎるだけの相手には鳴らない",
      not f_side, f"({f_side})")

# 同じ2本を旧TTC規則に通すと、横すれ違いにも鳴ってしまう（v4が使えなかった理由）
t_side = v4.fires_ttc(d_side, az_side, 80)
check("V12 旧TTC規則は横すれ違いにも鳴る（v4.1で直したこと）",
      bool(t_side) and not f_side,
      f"(TTC規則={len(t_side)}件 / 最接近予測={len(f_side)}件)")

print()
if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print(f"ALL PASS ({n} checks)")
sys.exit(0)
