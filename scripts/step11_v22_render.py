# -*- coding: utf-8 -*-
"""v22 = 新クラス bicycle（自転車の走行音・番号 8・距離クラス）とピン型ベル（2026-10-08・変更票 #48 D22・#49 D23）。
v21（kickfar）と同じく、v19 の描画に場面を登録するだけ。既存の場面のコードと乱数の消費順には触らない。

場面（plan の scenario）:
  v19bicycle  自転車 1 台（走行音だけ・ベルなし）。横距離は 0.8〜2.5 m（v12kick と同じ対数一様）と 2.5〜8.0 m（v19kickfar と同じ）を半々。
              速さ 4〜25 km/h の一様（徐行 4〜5・交差点進入の平均 11.1・アシストなし）。音源の高さ 0.1〜0.4 m。最接近 3〜8 s。
              音量 l1m = U(45, 55) dB(A)@1 m ＋ 速さの法則（4〜12 km/h: +6.5 log10(v/8.3)・12 km/h 超: 12 km/h の値 +30 log10(v/12)）。
              左右は plan の car_side（L/R 交互・無ければ乱数）。最接近の位置は「最接近時のマイク位置」基準（v19 の車と同じ）。
  v19bellbike 上の自転車と同じ軌道の上で bike_bell を鳴らす（走行音＋ベル）。ベルの型はディンディン・チリリン・ピンを等分。
              鳴動は v13 の規則（1〜3 回・短い鳴動窓）で、自転車が最接近の前後 ±4 s の間に鳴り始める。
  v19bellpin  既存のベルの本（v11core のベル行）の双子。同じ割当条件（列の写し）から新しい種で場面を引き直し、ベルの型だけピン型に固定（乱数の実現値は元と一致しない）。
              双子の元の行は plan の twin_of 列（step10_v22_plan が書く）。

ピン型ベル: src/outdoor_seld/bike_bell_v2.py（基音 2,450〜3,000 Hz の一様・JBPI 2023 の単打ベルの実測・1 打・repeat_period 2.0 s ±10%）。
bicycle の音: src/outdoor_seld/bicycle_v2.py（実測の音色・レベルは描画側）。
乱数: v19 と同じ（rng = default_rng(seed)・ピーク超過時は決定論ラダー）。
使い方:
  python scripts/step11_v22_render.py --preview [--n 300] [--scen v19bicycle|v19bellbike|v19bellpin|all]
  python scripts/step11_v22_render.py --render <clip_id> [--plan <csv>]
  python scripts/step11_v22_render.py --check-plan <csv>
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
import step11_v21_render as v21r  # noqa: E402  v21 までの登録を読み込む（kickfar の登録が要る）
v19r = v21r.v19r
m9 = v19r.m9
v13 = v19r.v13
v12 = v19r.v12
from outdoor_seld.bicycle_v2 import make_bicycle_v2  # noqa: E402
from outdoor_seld.bike_bell_v2 import make_bike_bell_v2  # noqa: E402

CLASS_BICYCLE = "bicycle"
IDX_BICYCLE = 8
# ---- bicycle（変更票 #48） ----
BIC_LAT_NEAR = (0.8, 2.5)        # v12kick と同じ
BIC_LAT_FAR = (2.5, 8.0)         # v19kickfar と同じ
BIC_P_FAR = 0.5
BIC_SPEED_KMH = (4.0, 25.0)
BIC_Z = (0.1, 0.4)
BIC_TCPA = (3.0, 8.0)
BIC_LEVEL_DBA_1M = (45.0, 55.0)  # 規則 ③ 据え置き（実測 48.6〜50.6 が中央）
BIC_V_REF_KMH = 8.3              # 実測の「普通」の中央値
BIC_SLOPE_LOW = 6.5              # 4〜12 km/h の実測の傾き（log10 v）
BIC_V_KNEE_KMH = 12.0
BIC_SLOPE_HIGH = 30.0            # 12 km/h 超の仮定（未測定）
BELL_TYPES = ("ding", "ring", "pin")
PIN_F0 = (2450.0, 3000.0)        # JBPI 2023 の単打ベル（案 B）
PIN_PERIOD = 2.0
BELL_START_WINDOW_S = 4.0        # ベルは最接近の ±4 s の間に鳴り始める
SCEN_BICYCLE, SCEN_BELLBIKE, SCEN_BELLPIN = "v19bicycle", "v19bellbike", "v19bellpin"   # 生成の振り分けが「v19」始まりなので慣例どおり（kickfar と同じ）


def bicycle_level_adj_db(v_kmh: float) -> float:
    """速さの法則（変更票 #48）: 4〜12 km/h は +6.5 log10(v/8.3)、12 km/h 超は 12 km/h の値 + 30 log10(v/12)。"""
    v = max(float(v_kmh), 0.1)
    if v <= BIC_V_KNEE_KMH:
        return BIC_SLOPE_LOW * np.log10(v / BIC_V_REF_KMH)
    return BIC_SLOPE_LOW * np.log10(BIC_V_KNEE_KMH / BIC_V_REF_KMH) + BIC_SLOPE_HIGH * np.log10(v / BIC_V_KNEE_KMH)


# ---- クラス番号・ゲート・ドライ音源の登録（v12 がキックと原付を足したときと同じ手口） ----
for table in (m9.CLASS_IDX, v12.CLASS_IDX_V12):
    assert table.get(CLASS_BICYCLE, IDX_BICYCLE) == IDX_BICYCLE, "bicycle の番号が衝突"
    table[CLASS_BICYCLE] = IDX_BICYCLE
v12.VEHICLE_GATED.add(CLASS_BICYCLE)
v13.VEHICLE_GATED_V13.add(CLASS_BICYCLE)
m9.CREST.setdefault(CLASS_BICYCLE, 5.0)
m9.LAW.setdefault(CLASS_BICYCLE, (BIC_LEVEL_DBA_1M[0], BIC_LEVEL_DBA_1M[1], 1.0))   # (lo, hi, ref 1 m) = 法規値 = 1 m 相当


EXTRA_BELL_KEYS = ("align_strike", "audio_seed")      # v22 が足した鍵。旧 make_bike_bell／make_bike_bell_ring は受け取らない（D08-07）
ALIGN_BASE_BELLS = True                               # 土台（v11core）のベルにも打音合わせを掛ける（変更票 #51 D24・本人 2026-10-08 13:32「土台も作り直しましょう」）


def _bell_dry(src: dict, p: dict, _prev):
    if p.get("bell_type") == "pin":
        return make_bike_bell_v2(m9.CLIP, m9.FS_SIM, f0=p["f0"], repeat_period_sec=p["repeat_period_sec"],
                                 rng=np.random.default_rng(p.get("audio_seed", 0)))
    src2 = dict(src); src2["params"] = {k: v for k, v in p.items() if k not in EXTRA_BELL_KEYS}
    return _prev(src2)


def _make_dry_v22(_prev):
    def fn(src: dict) -> np.ndarray:
        cls, p = src["class"], src.get("params", {})
        if cls == CLASS_BICYCLE:
            return make_bicycle_v2(m9.CLIP, m9.FS_SIM, np.random.default_rng(p["audio_seed"]), cadence_rpm=p.get("cadence_rpm", 70.0))
        if cls == "bike_bell" and (p.get("bell_type") == "pin" or p.get("align_strike") or ALIGN_BASE_BELLS):
            dry = _bell_dry(src, p, _prev)
            # 打音合わせ（Astra 監査 D08-01・2026-10-08）: ベルの打音は 0, period, 2×period… の固定時刻に生成されるので、鳴動窓 [t_on, t_off] に
            # 打音が入らず減衰の尻尾だけが目標の音量まで持ち上がる本があった（ピン型で 37%・土台のベルでも 14〜24%）。先頭の打音を t_on に動かす。
            shift = int(round(float(src.get("t_on", 0.0)) * m9.FS_SIM))
            if shift:
                dry = np.roll(dry, shift)       # 窓の外は _window が 0 にするので、巻き戻った尻尾は使われない
            return dry
        return _prev(src)
    fn._v22 = True
    return fn


if not getattr(m9._make_dry, "_v22", False):
    m9._make_dry = _make_dry_v22(m9._make_dry)


def pin_params(rng: np.random.Generator, seed: int) -> dict:
    """ピン型ベルの params（v13 の _rewrite_bell の else 枝 = repeat_period ごとに 1 打・ring_gap 0 が使える形）。"""
    return {"bell_type": "pin", "f0": float(rng.uniform(*PIN_F0)),
            "repeat_period_sec": PIN_PERIOD * float(rng.uniform(0.9, 1.1)), "ring_gap_sec": 0.0,
            "audio_seed": int(seed) * 7 + 23, "align_strike": True}


def bell_params_of_type(kind: str, rng: np.random.Generator, seed: int) -> dict:
    """型を指定してベルの params を引く。ding/ring は m9._warn_params の式と同じ（V91 の抽選を型で置き換える）。"""
    if kind == "pin":
        return pin_params(rng, seed)
    if kind == "ring":
        return {"bell_type": "ring", "f0": 3000.0 * float(rng.uniform(0.95, 1.05)),
                "strike_rate_hz": 30.0 * float(rng.uniform(0.8, 1.2)),
                "burst_sec": 0.65 * float(rng.uniform(0.8, 1.2)), "gap_sec": 0.6 * float(rng.uniform(0.8, 1.2))}
    return {"f0": 3000.0 * float(rng.uniform(0.95, 1.05)), "ring_gap_sec": 0.45 * float(rng.uniform(0.9, 1.1)),
            "repeat_period_sec": 2.0 * float(rng.uniform(0.9, 1.1))}


# ---- 場面 1: v22bicycle ----
def _bicycle_src(row: dict, rng: np.random.Generator, mic_arr, grammar: str) -> tuple[dict, dict]:
    sign = v19r._side(row, rng)
    far = bool(rng.random() < BIC_P_FAR)
    lo, hi = BIC_LAT_FAR if far else BIC_LAT_NEAR
    lat = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    v_kmh = float(rng.uniform(*BIC_SPEED_KMH)); speed = v_kmh / 3.6
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    z = float(rng.uniform(*BIC_Z))
    t_cpa = float(rng.uniform(*BIC_TCPA))
    base = float(rng.uniform(*BIC_LEVEL_DBA_1M))
    adj = float(bicycle_level_adj_db(v_kmh))
    law = base + adj
    y = lat * sign
    x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * speed * t_cpa
    cadence = float(rng.uniform(60.0, 80.0))
    src = {"class": CLASS_BICYCLE, "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * speed * m9.CLIP, y, z]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "law_db": law, "l1m_db": law, "lateral_m": lat,
           "level_base_db": base, "speed_level_adj_db": round(adj, 3),
           "grammar": grammar,
           "params": {"audio_seed": int(row["seed"]) * 7 + 22, "speed_mps": speed, "cadence_rpm": cadence}}
    info = {"lateral_m": round(lat, 2), "far": far, "side": "L" if sign > 0 else "R", "speed_kmh": round(v_kmh, 1),
            "dir_x": dirx, "t_cpa_s": round(t_cpa, 2), "level_base_db": round(base, 1), "level_adj_db": round(adj, 2), "src_z": round(z, 2)}
    return src, info


def sample_v22bicycle(row: dict) -> dict:
    rng = v19r._rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    src, info = _bicycle_src(row, rng, mic_arr, "v22_bicycle")
    return v19r._finish(row, rng, mic_arr, mic, [src], dict(kind="bicycle", **info))


# ---- 場面 2: v22bellbike ----
def sample_v22bellbike(row: dict) -> dict:
    rng = v19r._rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    src, info = _bicycle_src(row, rng, mic_arr, "v22_bellbike")
    kind = BELL_TYPES[int(rng.integers(0, len(BELL_TYPES)))]
    lv, l1m = m9._draw_level("bike_bell", rng)
    params = bell_params_of_type(kind, rng, int(row["seed"]))      # audio_seed はピン型だけ（ding／ring の make_* は受け取らない・10/8 ジョブ 5735 の失敗の原因）
    params["align_strike"] = True                                   # 3 型とも先頭の打音を鳴動窓の頭に合わせる（D08-01）
    bell = {"class": "bike_bell", "kind": "warn_moving", "track": 1,
            "wp": [list(p) for p in src["wp"]], "t_on": 0.0, "t_off": m9.CLIP,
            "speed_mps": src["speed_mps"], "dir_x": src["dir_x"], "t_cpa_s": src["t_cpa_s"], "lateral_m": src["lateral_m"],
            "law_db": lv, "l1m_db": l1m, "params": params, "grammar": "v22_bellbike_bell"}
    # 鳴動の窓（v13 の規則: 1〜3 回の短い鳴動）。鳴り始めは最接近の ±4 s の中（クリップ内に収める）
    bell = v13._rewrite_bell(bell, rng)
    t0 = float(np.clip(src["t_cpa_s"] + rng.uniform(-BELL_START_WINDOW_S, BELL_START_WINDOW_S), 0.2, m9.CLIP - 0.7))
    dur = float(bell["t_off"] - bell["t_on"])
    bell["t_on"], bell["t_off"] = t0, float(min(t0 + dur, m9.CLIP - 0.1))
    info.update({"bell_type": kind, "bell_f0": round(params["f0"], 1), "bell_n": bell.get("n_ring"), "bell_t_on": round(bell["t_on"], 2), "bell_t_off": round(bell["t_off"], 2)})
    return v19r._finish(row, rng, mic_arr, mic, [src, bell], dict(kind="bellbike", **info))


# ---- 場面 3: v23bellpin（既存のベルの本の双子・型だけピン） ----
_FORCE_PIN = {"on": False}


def _warn_params_v22(_prev):
    def fn(cls: str, rng: np.random.Generator, seed: int) -> dict:
        if cls == "bike_bell" and _FORCE_PIN["on"]:
            _prev(cls, rng, seed)              # 元の抽選を 1 回消費してから（この後 pin_params が uniform を 2 回引くので、以降の乱数は元の行とは一致しない = 「同じ割当条件から新しい種で引き直す」設計・Astra D08-09）
            return pin_params(rng, seed)
        return _prev(cls, rng, seed)
    fn._v22 = True
    return fn


if not getattr(m9._warn_params, "_v22", False):
    m9._warn_params = _warn_params_v22(m9._warn_params)


def sample_v23bellpin(row: dict) -> dict:
    """双子: plan の twin_of（元の v11core の行の clip_id）は記録だけ。場面は元と同じ規則（v13 の v11core）で、この行の種から引く。"""
    twin = dict(row); twin["scenario"] = "v11core"
    _FORCE_PIN["on"] = True
    try:
        s = v19r.v16r.sample_scene_v15(twin)      # 土台（v11core）の場面を直接引く。雨と grammar は外側の sample_scene_v19 が付ける
    finally:
        _FORCE_PIN["on"] = False
    bells = [src for src in s["sources"] if src["class"] == "bike_bell"]
    assert bells and all(src["params"].get("bell_type") == "pin" for src in bells), f"{row['clip_id']}: ピン型のベルが無い"
    s["row"] = dict(row)
    s["v22"] = {"kind": "bellpin", "twin_of": row.get("twin_of", ""), "n_bell": len(bells), "bell_f0": [round(b["params"]["f0"], 1) for b in bells]}
    return s


# 登録（v21 と同じ: _SAMPLERS19 に鍵を足す。v23bellpin は v13 を直接呼ぶので sample_scene_v19 経由でも動く）
for scen, fn in ((SCEN_BICYCLE, sample_v22bicycle), (SCEN_BELLBIKE, sample_v22bellbike), (SCEN_BELLPIN, sample_v23bellpin)):
    assert scen not in v19r._SAMPLERS19 or v19r._SAMPLERS19[scen] is fn, f"{scen} が別の関数で登録済み"
    v19r._SAMPLERS19[scen] = fn
generate_clip = v19r.generate_clip
sample_scene_v19 = v19r.sample_scene_v19


# ---- 検査 ----
def check_bicycle(src: dict, mic_arr) -> None:
    lat = float(src["lateral_m"])
    assert BIC_LAT_NEAR[0] - 1e-6 <= lat <= BIC_LAT_FAR[1] + 1e-6, f"横距離 {lat} が範囲外"
    v_kmh = float(src["speed_mps"]) * 3.6
    assert BIC_SPEED_KMH[0] - 1e-6 <= v_kmh <= BIC_SPEED_KMH[1] + 1e-6, f"速さ {v_kmh} が範囲外"
    assert BIC_Z[0] - 1e-6 <= float(src["wp"][0][3]) <= BIC_Z[1] + 1e-6, "高さが範囲外"
    base = float(src["level_base_db"])
    assert BIC_LEVEL_DBA_1M[0] - 1e-6 <= base <= BIC_LEVEL_DBA_1M[1] + 1e-6, "音量の基準が範囲外"
    assert abs(float(src["l1m_db"]) - (base + bicycle_level_adj_db(v_kmh))) < 1e-6, "速さの法則が合わない"
    t_cpa = float(src["t_cpa_s"])
    assert BIC_TCPA[0] - 1e-6 <= t_cpa <= BIC_TCPA[1] + 1e-6, "最接近の時刻が範囲外"
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


def check_bellbike(s: dict) -> None:
    srcs = s["sources"]
    assert len(srcs) == 2 and srcs[0]["class"] == CLASS_BICYCLE and srcs[1]["class"] == "bike_bell"
    check_bicycle(srcs[0], s["_mic_arr"])
    b = srcs[1]
    assert b["wp"] == srcs[0]["wp"], "ベルと走行音の軌道が違う"
    assert b["params"].get("bell_type", "ding") in ("ding", "ring", "pin")
    if b["params"].get("bell_type") == "pin":
        assert PIN_F0[0] - 1e-6 <= b["params"]["f0"] <= PIN_F0[1] + 1e-6, "ピン型の基音が範囲外"
    assert 0.0 <= b["t_on"] < b["t_off"] <= m9.CLIP, "鳴動窓が不正"
    assert b["params"].get("align_strike") and strike_at_window_start(b), "鳴動窓の頭に打音が無い（D08-01）"
    assert abs(b["t_on"] - srcs[0]["t_cpa_s"]) <= BELL_START_WINDOW_S + 1e-6 or b["t_on"] in (0.2,), "鳴り始めが最接近の ±4 s の外"


def strike_at_window_start(src: dict) -> bool:
    """鳴動窓の先頭 60 ms の実効値が、ドライ全体（窓を掛ける前）の最大 60 ms 実効値の 0.3 倍以上なら「打音が窓の頭にある」とみなす（D08-01 の検査）。
    減衰の尻尾だけの窓（0.2 s の時定数で 0.77 s 後なら 0.02 倍）は落ち、チリリン型の立ち上がり（0.37〜0.48 倍）は通る。"""
    full = m9._make_dry(src)
    a0, a1 = int(src["t_on"] * m9.FS_SIM), int(src["t_off"] * m9.FS_SIM)
    w = int(0.06 * m9.FS_SIM)
    if a1 - a0 < 2 * w:
        return False
    rms_full = np.sqrt(np.convolve(full ** 2, np.ones(w) / w, mode="valid"))
    start = float(np.sqrt(np.mean(full[a0:a0 + w] ** 2)))
    return start >= 0.3 * float(rms_full.max())


def check_bellpin(s: dict) -> None:
    bells = [src for src in s["sources"] if src["class"] == "bike_bell"]
    assert bells, "ベルが無い"
    for b in bells:
        assert b["params"].get("bell_type") == "pin", "ピン型でない"
        assert PIN_F0[0] - 1e-6 <= b["params"]["f0"] <= PIN_F0[1] + 1e-6, "ピン型の基音が範囲外"
        assert strike_at_window_start(b), "鳴動窓の頭に打音が無い（D08-01）"


def check_scene(scen: str, s: dict) -> None:
    if scen == SCEN_BICYCLE:
        assert len(s["sources"]) == 1 and s["sources"][0]["class"] == CLASS_BICYCLE
        check_bicycle(s["sources"][0], s["_mic_arr"])
    elif scen == SCEN_BELLBIKE:
        check_bellbike(s)
    elif scen == SCEN_BELLPIN:
        check_bellpin(s)
    else:
        raise ValueError(scen)


def preview(n: int, scen: str = "all") -> dict:
    out = {}
    for sc in ([SCEN_BICYCLE, SCEN_BELLBIKE, SCEN_BELLPIN] if scen == "all" else [scen]):
        ok, lat, spd, lv, adj, far, types = 0, [], [], [], [], 0, {}
        for i in range(n):
            row = v19r._fake_row(sc, i)
            if sc == SCEN_BELLPIN:
                row["n_warnings"] = 1; row["w1_class"] = "bike_bell"; row["w2_class"] = ""
            s = sample_scene_v19(row)
            try:
                check_scene(sc, s); ok += 1
            except AssertionError as e:
                print(f"NG {row['clip_id']}: {e}")
            if sc != SCEN_BELLPIN:
                src = s["sources"][0]
                lat.append(src["lateral_m"]); spd.append(src["speed_mps"] * 3.6); lv.append(src["l1m_db"]); adj.append(src["speed_level_adj_db"]); far += int(s["v19"]["far"])
                if sc == SCEN_BELLBIKE:
                    t = s["v19"]["bell_type"]; types[t] = types.get(t, 0) + 1
        print(f"{sc}: {n} 場面・検査 OK {ok}/{n}")
        if sc != SCEN_BELLPIN:
            lat, spd, lv, adj = map(np.array, (lat, spd, lv, adj))
            print(f"  横距離 [m] 5/50/95%: {np.percentile(lat, [5, 50, 95]).round(2).tolist()}・遠い側 {far / n:.0%}")
            print(f"  速さ [km/h] 5/50/95%: {np.percentile(spd, [5, 50, 95]).round(1).tolist()}・音量 l1m 5/50/95%: {np.percentile(lv, [5, 50, 95]).round(1).tolist()}・速さ補正 5/50/95%: {np.percentile(adj, [5, 50, 95]).round(1).tolist()}")
            if types:
                print(f"  ベルの型: {types}")
        out[sc] = {"n": n, "ok": ok}
    return out


def check_plan(path: Path) -> int:
    import csv
    bad = 0; n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("scenario") not in (SCEN_BICYCLE, SCEN_BELLBIKE, SCEN_BELLPIN):
                continue
            n += 1
            r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0)
            try:
                check_scene(r["scenario"], sample_scene_v19(r))
            except AssertionError as e:
                bad += 1; print(f"NG {r['clip_id']}: {e}")
    print(f"check_plan v22/v23: {n} 本 OK / {bad} 本 NG")
    return bad


def main() -> int:
    a = sys.argv[1:]
    if "--preview" in a:
        preview(int(a[a.index("--n") + 1]) if "--n" in a else 300, a[a.index("--scen") + 1] if "--scen" in a else "all")
        return 0
    if "--check-plan" in a:
        return 1 if check_plan(Path(a[a.index("--check-plan") + 1])) else 0
    if "--render" in a:
        import csv
        clip = a[a.index("--render") + 1]
        plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
        with open(plan, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["clip_id"] == clip]
        assert rows, f"{clip} が plan に無い"
        r = rows[0]; r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0)
        generate_clip(r); print("描画:", clip)
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
