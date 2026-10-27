# -*- coding: utf-8 -*-
"""v21 = キックボードの遠い横距離（2026-09-28・変更票 #42・D21・本人 14:14「学習データの追加はやってください。試聴もします」）。

v19 の描画（step11_v19_render）に場面を 1 つ登録するだけ。既存の 5 場面（fast/static/turn/platoon/launch）と土台
（v11core / v12kick / v12bike / v12train）のコードと乱数の消費順には触らない（_SAMPLERS19 に鍵を 1 つ足す）。

  v19kickfar  キックボード 1 台。横距離 2.5〜8.0 m の**対数一様**（近い側を厚く。実録の 3.0／5.0 m の線を覆う）。
              速さ・音量・向き・最接近の時刻は v12kick と同じ規則: 歩道 1.2〜1.7 m/s 40% / 車道 1.7〜5.6 m/s 60%、
              55〜65 dB(A)@1m、最接近 3〜8 s、高さ 1.5 m。左右は plan の car_side（L/R 交互・無ければ乱数）。
              最接近の位置は「最接近時のマイク位置」を基準に置く（v19 の車と同じ式。v12kick は原点基準だったので、
              歩行マイクでは最接近が t_cpa から最大 1.4 m/s × 8 s ずれていた）。
              雨（D3/D2）・歩行比（D4）・高さ（D13）は plan の列から掛かる（発進 D20 と同じ扱い）。

なぜ足すか: 学習のキックボードは横 0.8〜2.5 m だけで、実録の新台本 F-21〜F-24（3.0／5.0 m の線）は学習外だった。
クラクションの近い距離は足さない（実録は 3 m・学習の 3〜15 m の下端に入る。理由は変更票 #42）。

乱数: v19 と同じ（rng = default_rng(seed)・ピーク超過時は [seed, 20260908, salt] の決定論ラダー）。
使い方:
  python scripts/step11_v21_render.py --preview [--n 300]                 # 描画せずに幾何の統計だけ
  python scripts/step11_v21_render.py --render <clip_id> [--plan <csv>]   # 1 本描画（ローカル試走）
  python scripts/step11_v21_render.py --check-plan <csv>                  # plan の v19 行（kickfar 込み）の軌道検査
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v19_render as v19r  # noqa: E402

m9 = v19r.m9
v13 = v19r.v13
v12 = v19r.v12

SCENARIO = "v19kickfar"                      # plan の scenario（step10_v19_plan の f"v19{scene}"・scene = kickfar・k=5・clip 30001〜31750）
# ---- v19kickfar（2026-09-28・変更票 #42） ----
KICKFAR_LAT = (2.5, 8.0)                     # 横距離 [m]・対数一様（v12kick の 0.8〜2.5 の続き。実録の線 3.0／5.0 m を覆う）
KICKFAR_SPEED_WALKWAY = (1.2, 1.7)           # 歩道モード [m/s]（特定小型原付 ≤6 km/h。v12kick と同じ）
KICKFAR_SPEED_ROAD = (1.7, 5.6)              # 車道モード [m/s]（≤20 km/h。v12kick と同じ）
KICKFAR_P_WALKWAY = 0.4                      # 歩道モードの割合（v12kick と同じ）
KICKFAR_LEVEL_DBA_1M = (55.0, 65.0)          # 音量 dB(A)@1m（v12kick と同じ設計値。実機での較正は 2026-09-28 本人決定で行わない・差は限界に書く）
KICKFAR_TCPA = (3.0, 8.0)                    # 最接近の時刻 [s]（v12kick と同じ）
KICKFAR_Z = 1.5                              # 旧の音源の高さ [m]（記録用）。実際の値は m9.src_z_fixed("kickboard") = 既定 1.5・v29 は 0.05〜0.2 m（#70・点検 K4）


def sample_v19kickfar(row: dict) -> dict:
    rng = v19r._rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    sign = v19r._side(row, rng)
    lat = float(np.exp(rng.uniform(np.log(KICKFAR_LAT[0]), np.log(KICKFAR_LAT[1]))))
    walkway = bool(rng.random() < KICKFAR_P_WALKWAY)
    speed = float(rng.uniform(*(KICKFAR_SPEED_WALKWAY if walkway else KICKFAR_SPEED_ROAD)))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    t_cpa = float(rng.uniform(*KICKFAR_TCPA))
    law = float(rng.uniform(*KICKFAR_LEVEL_DBA_1M))
    y = lat * sign
    x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * speed * t_cpa      # 最接近時のマイク位置を基準（v19 の車と同じ）
    zk = m9.src_z_fixed("kickboard", row)                                        # #70: 旧 KICKFAR_Z 1.5・v29 は 0.05〜0.2 m
    src = {"class": "kickboard", "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, y, zk], [m9.CLIP, x0 + dirx * speed * m9.CLIP, y, zk]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "law_db": law, "l1m_db": law, "lateral_m": lat,           # 丸めない（wp の |y| と厳密に同じ値）
           "grammar": "v19_kickfar",
           "params": {"audio_seed": int(row["seed"]) * 7 + 21, "speed_mps": speed}}      # audio_seed は v12kick と同じ式
    info = {"kind": "kickfar", "lateral_m": round(lat, 2), "side": "L" if sign > 0 else "R", "speed_kmh": round(speed * 3.6, 1),
            "walkway": walkway, "dir_x": dirx, "t_cpa_s": round(t_cpa, 2), "level_dba_1m": round(law, 1)}
    return v19r._finish(row, rng, mic_arr, mic, [src], info)


# 登録: sample_scene_v19 は呼ばれるたびに _SAMPLERS19 を見るので、鍵を足すだけで v19 の generate_clip（"v19" 始まり）・check_plan・
# v13.sample_scene_v13 の差し替えがそのまま効く。既存 5 場面の関数オブジェクトは変えない。
assert SCENARIO not in v19r._SAMPLERS19 or v19r._SAMPLERS19[SCENARIO] is sample_v19kickfar, "v19kickfar が別の関数で登録済み"
v19r._SAMPLERS19[SCENARIO] = sample_v19kickfar
generate_clip = v19r.generate_clip
sample_scene_v19 = v19r.sample_scene_v19


def check_kickfar(src: dict, mic_arr) -> None:
    """kickfar 1 台の検査（AssertionError で違反を返す）: 横距離が範囲内・最接近（水平）が t_cpa（±0.02 s）で値が横距離（±0.02 m）・
    速さ・音量・時刻が範囲内。"""
    lat = float(src["lateral_m"])
    assert KICKFAR_LAT[0] - 1e-6 <= lat <= KICKFAR_LAT[1] + 1e-6, f"横距離 {lat} が範囲外"
    v = float(src["speed_mps"])
    assert min(KICKFAR_SPEED_WALKWAY[0], KICKFAR_SPEED_ROAD[0]) - 1e-6 <= v <= max(KICKFAR_SPEED_WALKWAY[1], KICKFAR_SPEED_ROAD[1]) + 1e-6, f"速さ {v} が範囲外"
    assert KICKFAR_LEVEL_DBA_1M[0] - 1e-6 <= float(src["l1m_db"]) <= KICKFAR_LEVEL_DBA_1M[1] + 1e-6, "音量が範囲外"
    t_cpa = float(src["t_cpa_s"])
    assert KICKFAR_TCPA[0] - 1e-6 <= t_cpa <= KICKFAR_TCPA[1] + 1e-6, "最接近の時刻が範囲外"
    tgrid = np.arange(0.0, m9.CLIP, 0.01)
    wp = np.asarray(src["wp"], dtype=np.float64)
    ps = m9.receiver_positions_at(tgrid, wp)
    mic = np.asarray(mic_arr, dtype=np.float64)
    pm = m9.receiver_positions_at(tgrid, mic) if mic.ndim == 2 else np.broadcast_to(mic, ps.shape)
    dh = np.hypot(ps[:, 0] - pm[:, 0], ps[:, 1] - pm[:, 1])
    k = int(np.argmin(dh))
    assert abs(float(tgrid[k]) - t_cpa) <= 0.02, f"最接近の時刻 {tgrid[k]:.2f} ≠ t_cpa {t_cpa:.2f}"
    assert abs(float(dh[k]) - lat) <= 0.02, f"最接近の水平距離 {dh[k]:.3f} ≠ 横距離 {lat:.3f}"
    v19r.check_polyline(src["wp"])


def preview(n: int) -> dict:
    lat, spd, tc, lv, walk, side, ok = [], [], [], [], 0, {"L": 0, "R": 0}, 0
    for i in range(n):
        row = v19r._fake_row(SCENARIO, i)
        s = sample_scene_v19(row)
        assert len(s["sources"]) == 1 and s["sources"][0]["class"] == "kickboard"
        src = s["sources"][0]
        try:
            check_kickfar(src, s["_mic_arr"]); ok += 1
        except AssertionError as e:
            print(f"NG {row['clip_id']}: {e}")
        lat.append(src["lateral_m"]); spd.append(src["speed_mps"]); tc.append(src["t_cpa_s"]); lv.append(src["l1m_db"])
        walk += int(s["v19"]["walkway"]); side[s["v19"]["side"]] += 1
    lat, spd, tc, lv = map(np.array, (lat, spd, tc, lv))
    out = {"n": n, "ok": ok, "lat_p5_50_95": np.percentile(lat, [5, 50, 95]).round(2).tolist(),
           "lat_ge_3m": float(np.mean(lat >= 3.0)), "lat_ge_5m": float(np.mean(lat >= 5.0)),
           "speed_kmh_p5_50_95": (np.percentile(spd, [5, 50, 95]) * 3.6).round(1).tolist(), "walkway_frac": walk / n,
           "t_cpa_p5_50_95": np.percentile(tc, [5, 50, 95]).round(2).tolist(), "level_p5_50_95": np.percentile(lv, [5, 50, 95]).round(1).tolist(),
           "side": side}
    print(f"{SCENARIO}: {n} 場面・検査 OK {ok}/{n}")
    print(f"  横距離 [m] 5/50/95%: {out['lat_p5_50_95']}（≥3 m {out['lat_ge_3m']:.0%}・≥5 m {out['lat_ge_5m']:.0%}）")
    print(f"  速さ [km/h] 5/50/95%: {out['speed_kmh_p5_50_95']}・歩道モード {out['walkway_frac']:.0%}")
    print(f"  最接近の時刻 [s] 5/50/95%: {out['t_cpa_p5_50_95']}・音量 dB(A)@1m 5/50/95%: {out['level_p5_50_95']}・左右 {side}")
    s = sample_scene_v19(v19r._fake_row(SCENARIO, 0))
    print("  例（軌道）:", json.dumps([[round(v, 3) for v in p] for p in s["sources"][0]["wp"]]), "情報:", json.dumps(s["v19"]))
    return out


def main() -> int:
    a = sys.argv[1:]
    if "--preview" in a:
        preview(int(a[a.index("--n") + 1]) if "--n" in a else 300)
        return 0
    if "--render" in a:
        cid = a[a.index("--render") + 1]
        plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else None
        rows = {r["clip_id"]: r for r in v19r.load_plan_v19(plan)}
        generate_clip(rows[cid])
        return 0
    if "--check-plan" in a:
        return 1 if v19r.check_plan(Path(a[a.index("--check-plan") + 1])) else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
