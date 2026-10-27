# -*- coding: utf-8 -*-
"""v29 描画 = v27 の鎖 ＋ 変更票 #70（2026-10-19・本人 00:14〜03:48・r16 11:35）の値（引き継ぎの値を根拠で決め直したもの）。

変えるのは **場面の作り方の値** だけ（音の生成器・ラベル・描画の物理は変えない）。土台（v9・v11・v12・v19・v21・v27）に足した差し込み口
（m9.draw_src_z・m9.src_z_fixed・m9.src_z_range・m9.lane_offset・m9.WALK_SPEED・m9.WARN_OFFSET・v12.HEAVY_FRAC・v19r.PLATOON_LANE_MIN）と
v14.rewrite_close_slow を、この module を import した時点で差し替える。import しなければ土台の出力は 1 バイトも変わらない（自己検査 1）。

  1. 音源の高さはクラスごと（点検 K3・K4）: 車 0.05〜0.6・バイク 0.05〜0.6・キック 0.05〜0.2・自転車 0.1〜0.4（v22 のまま）・
     クラクションを鳴らす車 0.4〜0.9・ベルは全部 0.95〜1.10（v11core のベル・ピン型・組み合わせの中のベル。ベルつき走行は v22 で元から）・
     サイレン 2.2〜2.6（v24 が上書き・ここでは引くだけ）・電車の車両 1.5・踏切の警報機 2.5（m9.CROSS_Z・据え置き）・
     バック音 0.8〜1.3（#70 r17・据え置き・3.5 t 超のトラック・バスの後部を想定）。
     乱数の消費は旧と同じ（1 回）。キック・バイクの高さは旧が固定値（乱数なし）だったので、別の種 rng([seed, SALT29, k]) で引き、本体の乱数列を変えない。
  2. 圏は横距離で抽選（K1）: 車の至近 1.0〜1.5・注意 1.5〜3.2・安全 3.2〜15 m（空きなし）。頭（マイクの真下）から車の中心線までの横距離を 1 回の一様乱数で引く。
     記録する cpa_rel_target_m は旧と同じ意味（マイク 1.5 m 基準の 3 次元の最接近）= hypot(横, 1.5 − z)。probe（4〜8 m）と休眠の場面の直書きは旧の式のまま。
  3. 警告音を鳴らす車の横距離 WARN_OFFSET 3〜15 → 1.5〜15 m。
  4. 重い車 HEAVY_FRAC 0.30 → 0.10（EV 15% はそのまま）。
  5. 歩く速さ WALK_SPEED 1.0〜1.4 → 1.0〜1.6 m/s（v11core と v12._mic_sample の両方・K2）。
  6. 隊列の 2・3 台目の横距離の下限 0.5 → 1.0 m（K5）。
  7. D10（至近の徐行）は plan の新しい列 cs_lat_m（横距離 1.0〜1.5 m）を読む。旧 cs_cpa_m（3 次元）は読まない。音量補正 cs_level_adj_db も足さない
     （v24 が速さの曲線で上書きするので旧でも効いていなかった）。
  8. D19（注意・安全の徐行・#69 r2／r3）: plan の列 d19=1・d19_tier・d19_band・d19_speed_kmh・d19_lat_m を読み、track0 の車を D10 と同じ作りで
     その速さ・横距離に置き直す（ほかの音源は変えない・音量は v24 の曲線だけ）。

雨の dB(A)（#70 r13）と雨の割合（r16 = 旧 50/35/15 のまま）は plan の行の値（step10_v29_plan.py が引き直す）で、描画器は読むだけ。

使い方: import するだけ（_run_v2final_gen.py が鎖の最後に import）。自己検査: python scripts/step11_v29_render.py selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v9_render as m9  # noqa: E402
import step11_v12_render as v12  # noqa: E402
import step11_v14_render as v14  # noqa: E402
import step11_v19_render as v19r  # noqa: E402
import step11_v27_render as v27  # noqa: E402,F401  鎖の前（v9 → … → v27）を全部読む

SALT29 = 20261019
# 用途ごとの乱数の鍵（rng([seed, SALT29, KEYS[用途]])）。**用途ごとに一意**（Astra R70-01: 旧は雨の dba とキックの高さが同じ 1 で結び付いていた）。
# plan 作成器（step10_v29_plan）もこの表を読む。eval v3 の作成器は d10_lat・d19 を同じ鍵で引く。新しい用途を足すときは必ずここに足し、selftest 11 が一意を確かめる。
KEYS = {"rain_dba": 1, "d10_lat": 2, "d19": 3, "z_kickboard": 11, "z_motorcycle": 12, "z_train_car": 13}

# ---------------------------------------------------------------- #70 の値（正 = 変更票 #70 の頭「最終の値」・md/design/v2final_params.json の D70） ----
SRC_Z_CLASS = {
    "car_drive": (0.05, 0.60),      # 走行音（ASJ RTN-Model 0 m・押野 1993 路面上・CNOSSOS 0.05 m・米フロリダ 1989 の等価高さ 約 0.6 m）
    "horn": (0.40, 0.90),           # クラクションを鳴らす車（車体の形からの推測・根拠は弱い）
    "bike_bell": (0.95, 1.10),      # ベル全部（#60 パナソニックの車種のハンドル高さ 965〜1,085 mm）
    "siren": (2.20, 2.60),          # サイレン（v24 が同じ範囲で上書きする・ここで引く値は使われない）
    "backup_beep": (0.80, 1.30),    # バック音（#70 r17・本人 10/19 13:15: 旧 0.8〜1.3 のまま。3.5 t 超のトラック・バスの後部〔荷台の床 0.81〜1.07 m・海外の取り付け案内 1.2 m〕を想定。日本の法令に高さの決まりは無い）
}
SRC_Z_FIXED_CLASS = {               # 旧は 1.5 m 固定（乱数なし）だったもの。(lo, hi) で引く。lo == hi なら固定
    "kickboard": (0.05, 0.20),
    "motorcycle": (0.05, 0.60),
    "train_car": (1.50, 1.50),      # #70 r9 据え置き（根拠は未確認の設計値）
}
TIER_LAT = {"critical": (1.0, 1.5), "caution": (1.5, 3.2), "safe": (3.2, 15.0)}   # 横距離 [m]（空きなし・採点の境 1.5／3.2 と同じ）
WARN_OFFSET = (1.5, 15.0)
WALK_SPEED = (1.0, 1.6)
HEAVY_FRAC = 0.10
PLATOON_LANE_MIN = 1.0
D10_LAT = (1.0, 1.5)                 # 至近の徐行（plan の cs_lat_m の範囲・検査用）
D19_TIER_LAT = {"caution": (1.5, 3.2), "safe": (3.2, 15.0)}                     # d19_lat_m の範囲（検査用・境は採点器と同じ 1.5 < d ≤ 3.2 / d > 3.2）
D19_BANDS = ((5.0, 7.5), (7.5, 10.0), (10.0, 12.5), (12.5, 15.0))               # 速さの帯 [km/h]・半開 [lo, hi)・最後の帯だけ 15.0 を含む（#69 r3）

_FIXED_IDX = {"kickboard": KEYS["z_kickboard"], "motorcycle": KEYS["z_motorcycle"], "train_car": KEYS["z_train_car"]}   # R70-01: 雨（1）・D10（2）・D19（3）と重ならない


def band_of(speed_kmh: float) -> int:
    """速さ [km/h] → 帯の番号 1〜4（半開区間・15.0 は 4）。範囲の外は ValueError。"""
    v = float(speed_kmh)
    if not (D19_BANDS[0][0] <= v <= D19_BANDS[-1][1]) or not np.isfinite(v):
        raise ValueError(f"D19 の速さ {v} km/h は 5〜15 の外")
    for i, (lo, hi) in enumerate(D19_BANDS, start=1):
        if lo <= v < hi:
            return i
    return len(D19_BANDS)


def tier_of_lateral(lat_m: float) -> str:
    """横距離 [m] → 圏（採点器と同じ境: ≤1.5 至近・1.5< d ≤3.2 注意・>3.2 安全）。"""
    d = float(lat_m)
    if d <= 1.5:
        return "critical"
    return "caution" if d <= 3.2 else "safe"


# ---------------------------------------------------------------- 差し替える関数 ----
def draw_src_z(rng: np.random.Generator, cls: str) -> float:
    """音源の高さ [m]（乱数 1 回・旧と同じ消費）。クラスの表に無い／None のクラスは旧 SRC_Z。"""
    rngs = SRC_Z_CLASS.get(cls)
    lo, hi = rngs if rngs is not None else m9.SRC_Z
    return float(rng.uniform(lo, hi))


def src_z_fixed(cls: str, row: dict) -> float:
    """旧が固定値だった音源の高さ。別の種で引く（本体の乱数列を変えない）。lo == hi なら固定値。"""
    lo, hi = SRC_Z_FIXED_CLASS[cls]
    if lo == hi:
        return float(lo)
    r = np.random.default_rng([int(row["seed"]), SALT29, _FIXED_IDX[cls]])
    return float(r.uniform(lo, hi))


def src_z_range(cls: str) -> tuple:
    if cls in SRC_Z_FIXED_CLASS:
        return tuple(float(x) for x in SRC_Z_FIXED_CLASS[cls])
    rngs = SRC_Z_CLASS.get(cls)
    return tuple(float(x) for x in (rngs if rngs is not None else m9.SRC_Z))


def lane_offset(rng: np.random.Generator, tier: str, z: float, sign: float, cpa_range=None):
    """車の横位置: 圏の横距離を 1 回の一様乱数で引く（旧と同じ消費）。戻り = (cpa_3d_at_1p5m, y_signed)。
    cpa_range が来たとき（probe 4〜8 m・休眠の場面の直書き）は旧の式のまま。"""
    if cpa_range is not None:
        return _ORIG_LANE_OFFSET(rng, tier, z, sign, cpa_range)
    lo, hi = TIER_LAT[tier]
    lat = float(rng.uniform(lo, hi))
    dz = m9.MIC_STATIC[2] - float(z)
    return float(np.hypot(lat, dz)), lat * float(sign)


def _rewrite_track0_car(s: dict, v_mps: float, lat_m: float, tag: str, extra: dict, side: str = "") -> dict:
    """track0 の車の速さと横距離を置き直す（D10・D19 共通・ほかの音源は変えない・音量は v24 の曲線だけ）。side（plan の car_side）があれば y の符号と照合する。"""
    mic = s["_mic_arr"]
    for src in s["sources"]:
        if src.get("class") != "car_drive" or src.get("kind") != "vehicle" or src.get("track", 0) != 0:
            continue
        wp = np.array(src["wp"], float)
        assert wp.shape[0] == 2 and wp[0, 2] == wp[1, 2], f"{tag}: track0 の車が直進でない（{tag} の元は v11core の直進だけ）: {wp.tolist()}"
        z = float(wp[0, 3])
        sign = 1.0 if wp[0, 2] >= 0 else -1.0
        if side in ("L", "R"):
            assert sign == (1.0 if side == "L" else -1.0), f"{tag}: 車の側（y の符号 {sign:+.0f}）が plan の car_side {side} と違う"
        y = float(lat_m) * sign
        dirx = float(src.get("dir_x", 1.0))
        t_cpa = float(src["t_cpa_rel_s"])
        x_mic = float(m9._mic_pos_at(mic, t_cpa)[0])
        x0 = x_mic - dirx * v_mps * t_cpa
        src["wp"] = [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * v_mps * m9.CLIP, y, z]]
        src["speed_mps"] = float(v_mps)
        src["cpa_rel_target_m"] = float(np.hypot(lat_m, m9.MIC_STATIC[2] - z))
        src["lateral_m"] = float(lat_m)
        src.update(extra)
        return s
    raise AssertionError(f"{tag}: track0 の車が無い")


def rewrite_close_slow(s: dict, row: dict) -> dict:
    """D10（至近の徐行・close_slow=1）: 横距離は plan の cs_lat_m（1.0〜1.5 m・新しい plan で引く）。cs_cpa_m（3 次元・旧）は読まない。"""
    if str(row.get("close_slow", "")) != "1":
        return s
    lat = row.get("cs_lat_m", "")
    assert lat not in ("", None), f"{row.get('clip_id')}: close_slow=1 なのに cs_lat_m が空（新しい plan〔step10_v29_plan.py〕が要る・旧 cs_cpa_m は読まない）"
    lat = float(lat); v_kmh = float(row["cs_speed_kmh"])
    assert np.isfinite(lat) and np.isfinite(v_kmh), (row.get("clip_id"), lat, v_kmh)
    assert D10_LAT[0] - 1e-9 <= lat <= D10_LAT[1] + 1e-9, (row.get("clip_id"), lat)
    assert 5.0 - 1e-9 <= v_kmh <= 15.0 + 1e-9, f"{row.get('clip_id')}: D10 の速さ {v_kmh} km/h が 5〜15 の外"
    return _rewrite_track0_car(s, v_kmh / 3.6, lat, "D10", {"close_slow": True, "cs_lat_m": lat, "cs_speed_kmh": v_kmh}, side=str(row.get("car_side", "")))


def rewrite_d19(s: dict, row: dict) -> dict:
    """D19（注意・安全の徐行・d19=1）: track0 の車を d19_speed_kmh・d19_lat_m に置き直す。帯と圏の整合は読み直した値で確かめる（#69 r3）。
    Astra R70-04: 有限・横距離の上限（注意 ≤3.2・安全 ≤15）・速さ 5〜15・側も確かめる。"""
    if str(row.get("d19", "")) != "1":
        return s
    tier, band = str(row["d19_tier"]), int(row["d19_band"])
    v_kmh, lat = float(row["d19_speed_kmh"]), float(row["d19_lat_m"])
    assert tier in D19_TIER_LAT, (row.get("clip_id"), tier)
    assert np.isfinite(v_kmh) and np.isfinite(lat), (row.get("clip_id"), v_kmh, lat)
    assert band_of(v_kmh) == band, f"{row.get('clip_id')}: 速さ {v_kmh} の帯 {band_of(v_kmh)} ≠ 記録 {band}"
    lo, hi = D19_TIER_LAT[tier]
    assert lo < lat <= hi + 1e-9, f"{row.get('clip_id')}: 横距離 {lat} が {tier} の範囲 {lo}<d≤{hi} の外"
    assert tier_of_lateral(lat) == tier, f"{row.get('clip_id')}: 横距離 {lat} の圏 {tier_of_lateral(lat)} ≠ 記録 {tier}"
    assert str(row.get("danger_tier")) == tier, f"{row.get('clip_id')}: plan の danger_tier {row.get('danger_tier')} ≠ d19_tier {tier}"
    return _rewrite_track0_car(s, v_kmh / 3.6, lat, "D19",
                               {"d19": True, "d19_tier": tier, "d19_band": band, "d19_speed_kmh": v_kmh, "d19_lat_m": lat}, side=str(row.get("car_side", "")))


def rewrite_slow_v29(s: dict, row: dict) -> dict:
    """v14 の rewrite_close_slow の差し替え: D10 → D19（1 本に両方は立たない）。"""
    assert not (str(row.get("close_slow", "")) == "1" and str(row.get("d19", "")) == "1"), f"{row.get('clip_id')}: close_slow と d19 が両方 1"
    return rewrite_d19(rewrite_close_slow(s, row), row)


# ---------------------------------------------------------------- 差し替え（import 時） ----
_ORIG = {"draw_src_z": m9.draw_src_z, "src_z_fixed": m9.src_z_fixed, "src_z_range": m9.src_z_range, "lane_offset": m9.lane_offset,
         "WALK_SPEED": m9.WALK_SPEED, "WARN_OFFSET": m9.WARN_OFFSET, "HEAVY_FRAC": v12.HEAVY_FRAC, "PLATOON_LANE_MIN": v19r.PLATOON_LANE_MIN,
         "rewrite_close_slow": v14.rewrite_close_slow}
_ORIG_LANE_OFFSET = _ORIG["lane_offset"]
assert _ORIG["WALK_SPEED"] == (1.0, 1.4) and _ORIG["WARN_OFFSET"] == (3.0, 15.0) and _ORIG["HEAVY_FRAC"] == 0.30 and _ORIG["PLATOON_LANE_MIN"] == 0.5, \
    "土台の既定値が想定（旧の値）と違う＝鎖の途中で誰かが変えている"
m9.draw_src_z = draw_src_z
m9.src_z_fixed = src_z_fixed
m9.src_z_range = src_z_range
m9.lane_offset = lane_offset
m9.WALK_SPEED = WALK_SPEED
m9.WARN_OFFSET = WARN_OFFSET
v12.HEAVY_FRAC = HEAVY_FRAC
v19r.PLATOON_LANE_MIN = PLATOON_LANE_MIN
v14.rewrite_close_slow = rewrite_slow_v29
m9.SRC_Z_FIXED = {k: v[0] for k, v in SRC_Z_FIXED_CLASS.items()}      # 記録用（検査は src_z_range を使う）


def contract_mismatches(P: dict) -> list:
    """保存された params（現物）と、この描画器の定数の食い違いを並べる（Astra R70-05: 書き直してから比べない）。空なら一致。"""
    bad = []
    d = (P or {}).get("D70_HEIGHTS_LATERAL") or {}
    if not d:
        return ["params に D70_HEIGHTS_LATERAL が無い"]
    want_z = {k: list(v) for k, v in SRC_Z_CLASS.items()} | {k: list(v) for k, v in SRC_Z_FIXED_CLASS.items()} | {"bicycle": [0.1, 0.4], "crossing_static": [float(m9.CROSS_Z), float(m9.CROSS_Z)]}
    got_z = d.get("src_z_m") or {}
    for k, v in want_z.items():
        if [float(x) for x in got_z.get(k, [])] != [float(x) for x in v]:
            bad.append(f"高さ {k}: params {got_z.get(k)} ≠ 描画器 {v}")
    for k in got_z:
        if k not in want_z:
            bad.append(f"高さ {k}: 描画器に無いクラスが params にある")
    if d.get("src_z_undecided"):
        bad.append(f"高さの未決が残っている: {d.get('src_z_undecided')}（#70 r17 で全部決まった）")
    pairs = (("tier_lateral_m", {k: list(v) for k, v in TIER_LAT.items()}), ("warn_offset_m", list(WARN_OFFSET)), ("walk_speed_mps", list(WALK_SPEED)),
             ("heavy_frac", HEAVY_FRAC), ("ev_frac", 0.15), ("platoon_lane_min_m", PLATOON_LANE_MIN), ("d10_lateral_m", list(D10_LAT)), ("level_match_src_z_m", 0.325),
             ("rng_keys", dict(KEYS)))
    for k, want in pairs:
        if d.get(k) != want:
            bad.append(f"{k}: params {d.get(k)} ≠ 描画器 {want}")
    D2 = (P or {}).get("D2_RAIN_DBA") or {}
    if {k: [float(x) for x in v] for k, v in D2.items()} != {"light": [34.0, 58.0], "moderate": [45.0, 61.0], "heavy": [49.0, 64.0]}:
        bad.append(f"D2_RAIN_DBA {D2} ≠ 34〜58／45〜61／49〜64（#70 r13）")
    if (P or {}).get("RAIN_KIND_P") != {"light": 0.5, "moderate": 0.35, "heavy": 0.15}:
        bad.append(f"RAIN_KIND_P {(P or {}).get('RAIN_KIND_P')} ≠ 0.5／0.35／0.15（#70 r16）")
    d19 = (P or {}).get("D19_SLOW_ADD") or {}
    if not d19.get("adopt") or d19.get("bands_kmh") != [list(b) for b in D19_BANDS] or d19.get("lateral_m") != {k: list(v) for k, v in D19_TIER_LAT.items()} or d19.get("n_clips") != 1600:
        bad.append(f"D19_SLOW_ADD（adopt／帯／横距離／本数）が描画器と違う: {d19.get('adopt')} {d19.get('bands_kmh')} {d19.get('lateral_m')} {d19.get('n_clips')}")
    try:
        from outdoor_seld import level_v24 as _L24
        if float(_L24.MATCH_GEOMETRY["src_z_m"]) != 0.325:
            bad.append(f"level_v24.MATCH_GEOMETRY.src_z_m {_L24.MATCH_GEOMETRY['src_z_m']} ≠ 0.325")
    except Exception as e:  # noqa: BLE001
        bad.append(f"level_v24 を読めない: {e}")
    return bad


def height_table() -> list:
    """全部の音源の高さの一覧（本人に出す・params の D70_SRC_Z と同じ内容）。"""
    rows = [
        ("車の走行音（car_drive）", SRC_Z_CLASS["car_drive"], "変更（旧 0.8〜1.3）", "ASJ RTN-Model 0 m・押野 1993 路面上・CNOSSOS 0.05・米 1989 等価 0.6 m"),
        ("バイク（motorcycle・v12bike／v19motofar／組み合わせ）", SRC_Z_FIXED_CLASS["motorcycle"], "変更（旧 1.5 固定）", "車と同じ範囲（車体の形からの推測）"),
        ("キックボード（kickboard・v12kick／v19kickfar／組み合わせ）", SRC_Z_FIXED_CLASS["kickboard"], "変更（旧 1.5 固定）", "デッキの高さ（推測）"),
        ("自転車の走行音（bicycle・v22）", (0.1, 0.4), "据え置き", "タイヤ・チェーンの高さ（#48）"),
        ("クラクションを鳴らす車（horn）", SRC_Z_CLASS["horn"], "変更（旧 0.8〜1.3）", "グリル裏の取り付け位置（推測）"),
        ("ベル（bike_bell・v11core の単独／ピン型／組み合わせ）", SRC_Z_CLASS["bike_bell"], "変更（旧 0.8〜1.3）", "#60 ハンドル高さ 965〜1,085 mm"),
        ("ベルつき走行（v19bellbike・v22）", (0.95, 1.10), "据え置き", "#60（元から）"),
        ("サイレン（siren）", (2.2, 2.6), "据え置き（v24 が上書き）", "#61"),
        ("バック音（backup_beep）", SRC_Z_CLASS["backup_beep"], "据え置き（#70 r17・3.5 t 超のトラック・バスの後部を想定）", "md/research/根拠_バック音の取り付け高さ_2026-10-19.md（荷台の床 0.81〜1.07・海外の案内 1.2 m・法令に高さの決まりなし）"),
        ("電車の車両（crossing・v12train の車両）", SRC_Z_FIXED_CLASS["train_car"], "据え置き", "#70 r9（根拠は未確認の設計値）"),
        ("踏切の警報機（crossing・静止）", (m9.CROSS_Z, m9.CROSS_Z), "据え置き", "#70 r9（根拠は未確認の設計値）"),
    ]
    return [{"source": a, "z_m": [float(b[0]), float(b[1])], "status": c, "basis": d} for a, b, c, d in rows]


# ---------------------------------------------------------------- 検査の道具 ----
def _interp_path(wp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """wp [[t,x,y,z],...] を時刻 t に線形補間（端は保持）。"""
    out = np.empty((t.size, 3))
    for k in range(3):
        out[:, k] = np.interp(t, wp[:, 0], wp[:, k + 1])
    return out


def mic_path(mic_arr, t: np.ndarray) -> np.ndarray:
    m = np.asarray(mic_arr, float)
    if m.ndim == 1:
        return np.tile(m[:3], (t.size, 1))
    return _interp_path(m, t)


def mic_arr_from_scene(s: dict) -> np.ndarray:
    """scene.json（_mic_arr は保存されない）からマイクの軌道を作り直す（v12._mic_sample と同じ式: 歩行は x0 = −d·v·5・y = 0・高さは mic_z）。"""
    if "_mic_arr" in s:
        return np.asarray(s["_mic_arr"], float)
    mic = s.get("mic") or {}
    z = float(mic.get("mic_z", (s.get("row") or {}).get("mic_z", 1.5)) or 1.5)
    if mic.get("motion") == "walk":
        v = float(mic["walk_speed_mps"]); d = float(mic["walk_dir_x"]); x0 = -d * v * 5.0
        return np.array([[0.0, x0, 0.0, z], [m9.CLIP, x0 + d * v * m9.CLIP, 0.0, z]])
    return np.array([0.0, 0.0, z])


def horizontal_cpa(s: dict, src: dict, dt: float = 0.005) -> float:
    """音源とマイクの水平距離（マイクの真下から）の最小 [m]（0〜CLIP を dt 刻み）。"""
    t = np.arange(0.0, m9.CLIP + 1e-9, dt)
    ps = _interp_path(np.array(src["wp"], float), t)
    pm = mic_path(s["_mic_arr"], t)
    return float(np.hypot(ps[:, 0] - pm[:, 0], ps[:, 1] - pm[:, 1]).min())


def _speed_from_wp(wp: np.ndarray) -> float | None:
    """2 点の直進の wp から速さ [m/s]（曲がり・発進の折れ線は None）。"""
    if wp.shape[0] != 2 or wp[1, 0] <= wp[0, 0]:
        return None
    return float(np.hypot(wp[1, 1] - wp[0, 1], wp[1, 2] - wp[0, 2]) / (wp[1, 0] - wp[0, 0]))


def check_scene_v29(s: dict, row: dict) -> list:
    """1 本の場面の #70 の受け入れ（Astra R70-04 で強化）: 高さがクラスの範囲・track0 の車の横距離が圏の範囲（上限 15 を含む）・隊列の下限・歩く速さ・
    直進の音源は wp から求めた速さ = speed_mps・D10／D19 は plan の速さ・横距離・側・帯・圏と一致・値は有限。戻り = 違反の一覧。"""
    bad = []
    for src in s["sources"]:
        cls = src.get("class"); wp = np.array(src["wp"], float)
        if wp.ndim != 2 or wp.shape[1] != 4 or not np.all(np.isfinite(wp)):
            bad.append(f"{cls} の wp が数でない・形が違う: {np.asarray(src['wp']).shape}"); continue
        if wp.shape[0] < 2 or not np.all(np.diff(wp[:, 0]) > 0):
            bad.append(f"{cls} の wp の時刻が増えていない（節点 {wp.shape[0]}・RR70-03）"); continue
        z = float(wp[0, 3])
        if cls == "crossing" and src.get("params", {}).get("train_car"):
            lo, hi = SRC_Z_FIXED_CLASS["train_car"]
        elif cls == "crossing":
            lo, hi = m9.CROSS_Z, m9.CROSS_Z
        elif cls in ("kickboard", "motorcycle"):
            lo, hi = SRC_Z_FIXED_CLASS[cls]
        elif cls == "bicycle":
            lo, hi = 0.1, 0.4
        elif cls in ("car_drive", "bike_bell", "horn", "backup_beep"):
            lo, hi = SRC_Z_CLASS[cls]
        elif cls == "siren":
            lo, hi = 2.2, 2.6
        else:
            bad.append(f"知らないクラス {cls}"); continue
        if not (lo - 1e-9 <= z <= hi + 1e-9) or not np.all(np.abs(wp[:, 3] - z) < 1e-9):
            bad.append(f"{cls} の高さ {z:.3f} が範囲 {lo}〜{hi} の外か途中で変わる")
        # 直進の音源: wp から求めた速さ = speed_mps（速さの欄と軌道の食い違いを見る・R70-04）
        v_wp = _speed_from_wp(wp) if src.get("kind") == "vehicle" and "launch" not in src and "turn" not in src else None
        if v_wp is not None and src.get("speed_mps") is not None and abs(v_wp - float(src["speed_mps"])) > 1e-6 * max(1.0, float(src["speed_mps"])):
            bad.append(f"{cls} の wp から求めた速さ {v_wp:.4f} m/s ≠ speed_mps {float(src['speed_mps']):.4f}")
        is_track0_car = cls == "car_drive" and src.get("kind") == "vehicle" and src.get("track", 0) == 0
        if is_track0_car and src.get("danger_tier") in TIER_LAT and not src.get("d19") and not src.get("close_slow"):
            lo_l, hi_l = TIER_LAT[src["danger_tier"]]
            if wp.shape[0] == 2 and wp[0, 2] == wp[1, 2]:                              # 直進: 横距離 = |y|
                lat = abs(float(wp[0, 2]))
                if not (lo_l - 1e-9 <= lat <= hi_l + 1e-9):
                    bad.append(f"track0 の車の横距離 {lat:.3f} が圏 {src['danger_tier']} の範囲 {lo_l}〜{hi_l} の外")
                exp = float(np.hypot(lat, m9.MIC_STATIC[2] - z))
                if abs(float(src.get("cpa_rel_target_m", exp)) - exp) > 1e-6:
                    bad.append(f"cpa_rel_target_m {src.get('cpa_rel_target_m')} ≠ hypot(横, 1.5−z) {exp:.4f}")
        if cls == "car_drive" and src.get("platoon_index", 0) > 0:
            if abs(float(wp[0, 2])) < PLATOON_LANE_MIN - 1e-9:
                bad.append(f"隊列の {src['platoon_index']} 台目の横距離 {abs(float(wp[0, 2])):.3f} < {PLATOON_LANE_MIN}")
        for tag, key_lat, key_v, key_lat_plan, key_v_plan in (("D10", "cs_lat_m", "cs_speed_kmh", "cs_lat_m", "cs_speed_kmh"), ("D19", "d19_lat_m", "d19_speed_kmh", "d19_lat_m", "d19_speed_kmh")):
            if not src.get(tag.lower() if tag == "D19" else "close_slow"):
                continue
            lat = abs(float(wp[0, 2])); v_kmh = float(src.get("speed_mps", 0.0)) * 3.6
            if wp.shape[0] != 2 or wp[0, 2] != wp[1, 2]:
                bad.append(f"{tag} の車が直進でない"); continue
            if src.get(key_lat) is None:
                bad.append(f"{tag} の車に {key_lat} が無い（v29 より前の描き方の本＝旧 cs_cpa_m の 3 次元で置いた本）"); continue
            if abs(lat - float(src[key_lat])) > 1e-9:
                bad.append(f"{tag} の横距離が wp と違う")
            if str(row.get(key_lat_plan, "")) != "" and abs(lat - float(row[key_lat_plan])) > 1e-9:
                bad.append(f"{tag} の横距離 {lat:.3f} が plan の {key_lat_plan} {row[key_lat_plan]} と違う")
            if str(row.get(key_v_plan, "")) != "" and abs(v_kmh - float(row[key_v_plan])) > 1e-6:
                bad.append(f"{tag} の速さ {v_kmh:.3f} km/h が plan の {key_v_plan} {row[key_v_plan]} と違う")
            if not (5.0 - 1e-9 <= v_kmh <= 15.0 + 1e-9):
                bad.append(f"{tag} の速さ {v_kmh:.3f} km/h が 5〜15 の外")
            if tag == "D10" and not (D10_LAT[0] - 1e-9 <= lat <= D10_LAT[1] + 1e-9):
                bad.append(f"D10 の横距離 {lat:.3f} が {D10_LAT} の外")
            if tag == "D19":
                tier = str(src.get("d19_tier")); lo_t, hi_t = D19_TIER_LAT.get(tier, (np.nan, np.nan))
                if not (lo_t < lat <= hi_t + 1e-9) or tier_of_lateral(lat) != tier or str(row.get("danger_tier")) != tier or str(row.get("d19_tier")) != tier:
                    bad.append(f"D19 の横距離 {lat:.3f}・圏 {tier} が範囲 {lo_t}<d≤{hi_t}／plan と合わない")
                try:
                    if band_of(v_kmh) != int(src.get("d19_band")) or str(row.get("d19_band")) != str(src.get("d19_band")):
                        bad.append(f"D19 の速さ {v_kmh:.3f} の帯 ≠ 記録 {src.get('d19_band')}／plan {row.get('d19_band')}")
                except ValueError as e:
                    bad.append(f"D19 の速さが帯の外: {e}")
            side = str(row.get("car_side", ""))
            if side in ("L", "R") and (1.0 if wp[0, 2] >= 0 else -1.0) != (1.0 if side == "L" else -1.0):
                bad.append(f"{tag} の車の側（y {wp[0, 2]:+.2f}）が plan の car_side {side} と違う")
    mic = s.get("mic", {})
    if mic.get("motion") == "walk":
        v = float(mic.get("walk_speed_mps", 0.0))
        if not (WALK_SPEED[0] - 1e-9 <= v <= WALK_SPEED[1] + 1e-9):
            bad.append(f"歩く速さ {v:.3f} が {WALK_SPEED} の外")
    if str(row.get("close_slow", "")) == "1" and not any(x.get("close_slow") for x in s["sources"]):
        bad.append("close_slow=1 なのに書き換えた車が無い")
    if str(row.get("d19", "")) == "1" and not any(x.get("d19") for x in s["sources"]):
        bad.append("d19=1 なのに書き換えた車が無い")
    return bad


# ---------------------------------------------------------------- 自己検査 ----
def _fake_core_row(i: int, tier: str, side: str, motion: str, **kw) -> dict:
    r = {"clip_id": f"preview_v29_{i}", "split": "train", "motion": motion, "n_warnings": 0, "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
         "danger_tier": tier, "car_side": side, "scenario": "v11core", "seed": 9290000000 + i, "scene_type": "residential", "n_car": "1",
         "rain": "", "rain_rate": "", "rain_dba": "", "grammar": "v16", "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "",
         "mic_z": "1.5", "h_copy": "1", "twin_of": "", "w3_class": "", "w3_side": "", "mix": "", "cs_lat_m": "", "d19": "", "d19_tier": "", "d19_band": "",
         "d19_speed_kmh": "", "d19_lat_m": ""}
    r.update(kw)
    return r


def _ref_without_v29(rows: list) -> list:
    """別プロセス（v29 を import しない鎖）で同じ行の場面を作る。"""
    code = ("import sys, json; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')\n"
            "import step11_v27_render\nimport step11_v19_render as v19r\nimport numpy as np\n"
            "def conv(o):\n"
            "    if isinstance(o, np.ndarray): return o.tolist()\n"
            "    if isinstance(o, (np.floating, np.integer)): return o.item()\n"
            "    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}\n"
            "    if isinstance(o, (list, tuple)): return [conv(v) for v in o]\n"
            "    return o\n"
            "rows = json.loads(sys.stdin.read())\n"
            "print(json.dumps([conv(v19r.sample_scene_v19(r)) for r in rows]))\n") % (str(ROOT / "scripts"), str(ROOT / "src"))
    p = subprocess.run([sys.executable, "-c", code], input=json.dumps(rows), capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_conv(v) for v in o]
    return o


_KEEP_SRC = ("class", "kind", "track", "no_label", "t_on", "t_off", "speed_mps", "dir_x", "t_cpa_rel_s", "t_cpa_s", "params", "grammar", "danger_tier",
             "platoon_index", "headway_s", "train_car_index", "dist_draw_m", "az_draw_deg", "car_variant")


def _strip_geometry(s: dict) -> dict:
    """高さ・横距離・音量に依存しない欄だけ残す（速さ・時刻・向き・params・暗騒音）。これが旧と一致すれば、乱数の消費が旧と同じ。"""
    s = json.loads(json.dumps(s))
    static = (s.get("mic") or {}).get("motion") != "walk"
    out = {"row": s.get("row"), "noise": s.get("noise"), "mic": {k: v for k, v in s.get("mic", {}).items() if k != "walk_speed_mps"},
           "v19_kind": (s.get("v19") or {}).get("kind"),
           # Astra R70-03: wp を丸ごと落とさず、時刻の列と（静止のマイクなら）x の列も比べる＝横距離 y と高さ z だけが変わってよい
           "sources": [{**{k: v for k, v in src.items() if k in _KEEP_SRC},
                        **({"wp_t": [round(float(q[0]), 9) for q in src["wp"]],
                            **({"wp_x": [round(float(q[1]), 6) for q in src["wp"]]} if static and src.get("class") not in ("siren", "horn", "backup_beep", "bike_bell") else {})}
                           if len(src.get("wp", [])) == 2 else {"wp_n": len(src.get("wp", []))})} for src in s["sources"]]}
    for src in out["sources"]:
        p = src.get("params")
        if isinstance(p, dict):
            src["params"] = {k: v for k, v in p.items() if k not in ("speed_mps",)}
        if src.get("class") in ("siren", "horn", "backup_beep", "bike_bell"):
            src.pop("t_cpa_s", None)            # 警告音の t_cpa_s と x は受音の時刻（音の伝わる時間 ≈ 横距離/343 を含む）で、WARN_OFFSET が変わると末尾の桁が動く＝正しい変化
    return out


def selftest() -> int:
    ok = []
    tiers = ("critical", "caution", "safe")
    # 1. 乱数の消費が旧と同じ: v29 で作った場面から高さ・横・速さ・音量の欄を落とすと、v29 なしの場面と一致（v11core・fast・turn・platoon・launch・kick・bike・mix）
    rows = [_fake_core_row(i, tiers[i % 3], "L" if i % 2 else "R", "walk" if i % 3 else "static") for i in range(12)]
    # 発進は静止のマイクで比べる（歩くマイクだと歩く速さの変更で引き直しの採否が変わる本がある＝R70-03・意図した変化・一覧は out/v29_regen_check_2026-10-19/launch_resample_diff.csv）
    rows += [dict(v19r._fake_row(sc, i), seed=9291000000 + i, **({"motion": "static"} if sc == "v19launch" else {})) for i, sc in enumerate(("v19fast", "v19turn", "v19platoon", "v19launch") * 3)]
    rows += [dict(_fake_core_row(100 + i, tiers[i % 3], "L", "walk"), scenario=sc, scene_type="v12ext") for i, sc in enumerate(("v12kick", "v12bike", "v12train") * 2)]
    # 組み合わせの本は静止のマイクで比べる（歩くマイクは WALK_SPEED が変わるので、マイク基準で置く自転車の t_cpa_s が末尾の桁で動く＝乱数の消費とは別の、正しい変化）
    rows += [v27.fake_row("car+bicycle+siren", 500 + i, "static") for i in range(3)] + [v27.fake_row("car+horn", 600 + i, "static") for i in range(3)]
    for r in rows:
        r.setdefault("cs_lat_m", ""); r.setdefault("d19", "")
    ref = _ref_without_v29(rows)
    mine = [_conv(v19r.sample_scene_v19(dict(r))) for r in rows]
    same = [_strip_geometry(a) == _strip_geometry(b) for a, b in zip(ref, mine)]
    ok.append((f"1. 乱数の消費は旧と同じ（高さ・横・速さ・音量の欄を除けば {sum(same)}/{len(rows)} 本が一致）", all(same)))
    # 2. 高さ・横距離・隊列・歩く速さの受け入れ（check_scene_v29 で違反 0）
    bad = [(r["clip_id"], b) for r, s in zip(rows, mine) for b in check_scene_v29(s, r)]
    ok.append((f"2. 受け入れの検査: 違反 {len(bad)}（高さの範囲・track0 の横距離が圏の範囲・隊列の下限 1.0・歩く速さ 1.0〜1.6）", not bad))
    if bad:
        print("   例:", bad[:5])
    # 3. 横距離の抽選: 多数の v11core で至近 1.0〜1.5・注意 1.5〜3.2・安全 3.2〜15 を埋める（端も出る）
    lats = {t: [] for t in tiers}; zs = []
    for i in range(600):
        r = _fake_core_row(1000 + i, tiers[i % 3], "L" if i % 2 else "R", "static")
        s = v19r.sample_scene_v19(dict(r))
        car = next(x for x in s["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
        lats[r["danger_tier"]].append(abs(float(car["wp"][0][2]))); zs.append(float(car["wp"][0][3]))
    span_ok = all(min(lats[t]) < TIER_LAT[t][0] + 0.1 * (TIER_LAT[t][1] - TIER_LAT[t][0]) and max(lats[t]) > TIER_LAT[t][1] - 0.1 * (TIER_LAT[t][1] - TIER_LAT[t][0]) for t in tiers)
    ok.append((f"3. 横距離は圏の範囲を端まで埋める（至近 {min(lats['critical']):.2f}〜{max(lats['critical']):.2f}・注意 {min(lats['caution']):.2f}〜{max(lats['caution']):.2f}・"
               f"安全 {min(lats['safe']):.2f}〜{max(lats['safe']):.2f}）・車の高さ {min(zs):.3f}〜{max(zs):.3f}", span_ok and min(zs) >= 0.05 and max(zs) <= 0.6 and max(zs) - min(zs) > 0.4))
    # 4. 警告音の車の横距離 1.5〜15・クラクション 0.4〜0.9・ベル 0.95〜1.10・バック音は旧のまま（未決）
    hz, bz, bk, lat_w = [], [], [], []
    for i in range(300):
        cls = ("horn", "bike_bell", "backup_beep")[i % 3]
        r = _fake_core_row(2000 + i, "safe", "L", "static", n_warnings=1, w1_class=cls, w1_side="R")
        s = v19r.sample_scene_v19(dict(r))
        w = next(x for x in s["sources"] if x["class"] == cls)
        z = float(w["wp"][0][3]); lat_w.append(abs(float(w["wp"][0][2])))
        (hz if cls == "horn" else bz if cls == "bike_bell" else bk).append(z)
    ok.append((f"4. 警告音の車の横距離 {min(lat_w):.2f}〜{max(lat_w):.2f}（1.5〜15）・クラクション {min(hz):.2f}〜{max(hz):.2f}（0.4〜0.9）・ベル {min(bz):.3f}〜{max(bz):.3f}（0.95〜1.10）・"
               f"バック音 {min(bk):.2f}〜{max(bk):.2f}（未決＝旧 0.8〜1.3）",
               1.5 <= min(lat_w) and max(lat_w) <= 15.0 and min(lat_w) < 2.5 and 0.4 <= min(hz) and max(hz) <= 0.9 and 0.95 <= min(bz) and max(bz) <= 1.10
               and 0.8 <= min(bk) and max(bk) <= 1.3))
    # 5. 重い車 10%・EV 15%（audio_seed のハッシュで決まる・5,000 台）
    cnt = {"ev": 0, "heavy": 0, "normal": 0}
    for i in range(5000):
        cnt[v12._car_variant(9290000000 + i * 7 + 5)] += 1
    fe, fh = cnt["ev"] / 5000, cnt["heavy"] / 5000
    ok.append((f"5. 車の種別（5,000 台）: EV {fe:.1%}・重い {fh:.1%}・ふつう {cnt['normal'] / 5000:.1%}", abs(fe - 0.15) < 0.02 and abs(fh - 0.10) < 0.02))
    # 6. D10: cs_lat_m を読む・cs_cpa_m は読まない・音量補正を足さない
    r = _fake_core_row(3000, "critical", "L", "walk", close_slow="1", cs_cpa_m="0.60", cs_speed_kmh="8.0", cs_level_adj_db="-8.0", cs_lat_m="1.234")
    s = v19r.sample_scene_v19(dict(r))
    car = next(x for x in s["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    hc = horizontal_cpa(s, car)
    r_plain = dict(r, close_slow="", cs_cpa_m="", cs_speed_kmh="", cs_level_adj_db="", cs_lat_m="")
    s_plain = v19r.sample_scene_v19(r_plain)
    car_plain = next(x for x in s_plain["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    ok.append((f"6. D10: 横距離 {abs(car['wp'][0][2]):.3f} = cs_lat_m 1.234・実際の水平の最接近 {hc:.3f}・速さ {car['speed_mps'] * 3.6:.1f} km/h・"
               f"音量は v24 の曲線（補正なし: 1m 値 {car['l1m_db']:.1f} vs 置き換え前 {car_plain['l1m_db']:.1f}）",
               abs(abs(car["wp"][0][2]) - 1.234) < 1e-9 and abs(hc - 1.234) < 0.01 and abs(car["speed_mps"] * 3.6 - 8.0) < 1e-6 and car.get("close_slow") is True))
    try:
        v19r.sample_scene_v19(dict(r, cs_lat_m=""))
        ok.append(("6b. D10 で cs_lat_m が空 → 止まる", False))
    except AssertionError as e:
        ok.append(("6b. D10 で cs_lat_m が空 → 止まる（旧 cs_cpa_m を黙って読まない）", "cs_lat_m" in str(e)))
    # 7. D19: 速さの帯・圏の整合・置き直し・不整合は止まる
    r = _fake_core_row(3100, "safe", "R", "static", d19="1", d19_tier="safe", d19_band="3", d19_speed_kmh="10.000", d19_lat_m="7.250")
    s = v19r.sample_scene_v19(dict(r))
    car = next(x for x in s["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    ok.append((f"7. D19: 横距離 {abs(car['wp'][0][2]):.3f}（safe）・速さ {car['speed_mps'] * 3.6:.1f} km/h（帯 3）・実際の水平の最接近 {horizontal_cpa(s, car):.3f}",
               abs(abs(car["wp"][0][2]) - 7.25) < 1e-9 and abs(car["speed_mps"] * 3.6 - 10.0) < 1e-6 and car.get("d19") is True and car["wp"][0][2] < 0))
    bads = 0
    for kw in (dict(d19_speed_kmh="7.5", d19_band="1"), dict(d19_lat_m="3.2", d19_tier="safe"), dict(d19_tier="caution"), dict(d19_speed_kmh="15.1", d19_band="4"),
               dict(d19_lat_m="100"), dict(d19_speed_kmh="nan", d19_band="1"), dict(d19_lat_m="15.500")):
        try:
            v19r.sample_scene_v19(dict(r, **kw));
        except (AssertionError, ValueError):
            bads += 1
    ok.append((f"7b. D19 の不整合（7.5 は帯 2・3.2 は注意・danger_tier と違う・15.1 は範囲外・横 100 m・速さ nan・横 15.5 m〔安全の上限の外〕）は止まる {bads}/7", bads == 7))
    ok.append(("7c. 帯の境: 5.0→1・7.5→2・10.0→3・12.5→4・15.0→4・7.4999→1", [band_of(x) for x in (5.0, 7.5, 10.0, 12.5, 15.0, 7.4999)] == [1, 2, 3, 4, 4, 1]))
    ok.append(("7d. 圏の境: 1.5→至近・1.5001→注意・3.2→注意・3.2001→安全", [tier_of_lateral(x) for x in (1.5, 1.5001, 3.2, 3.2001)] == ["critical", "caution", "caution", "safe"]))
    # 8. 高さ違いの双子（mic_z だけ違う行）は横距離と xy の軌道が同じ
    r1 = _fake_core_row(3200, "caution", "L", "walk", mic_z="1.45"); r2 = dict(r1, mic_z="2.05", clip_id="preview_v29_3200b")
    s1, s2 = v19r.sample_scene_v19(dict(r1)), v19r.sample_scene_v19(dict(r2))
    xy_same = all(np.allclose(np.array(a["wp"], float)[:, :3], np.array(b["wp"], float)[:, :3]) for a, b in zip(s1["sources"], s2["sources"]))
    ok.append(("8. 高さ違いの双子は xy の軌道と横距離が同じ（マイクの高さだけ違う）", xy_same and float(s1["_mic_arr"][0][3] if np.ndim(s1["_mic_arr"]) == 2 else s1["_mic_arr"][2]) != float(s2["_mic_arr"][0][3] if np.ndim(s2["_mic_arr"]) == 2 else s2["_mic_arr"][2])))
    # 9. 二輪の高さは別の種（本体の乱数列を変えない）: kick 0.05〜0.2・bike 0.05〜0.6・電車 1.5
    kz, mz, tz = [], [], []
    for i in range(200):
        for sc, lst in (("v12kick", kz), ("v12bike", mz), ("v12train", tz)):
            s = v19r.sample_scene_v19(dict(_fake_core_row(4000 + i, "safe", "L", "static"), scenario=sc, scene_type="v12ext"))
            lst.append(float(s["sources"][0]["wp"][0][3]))
    ok.append((f"9. キック {min(kz):.3f}〜{max(kz):.3f}（0.05〜0.2）・バイク {min(mz):.3f}〜{max(mz):.3f}（0.05〜0.6）・電車 {min(tz)}〜{max(tz)}（1.5）",
               0.05 <= min(kz) and max(kz) <= 0.2 and 0.05 <= min(mz) and max(mz) <= 0.6 and min(tz) == max(tz) == 1.5 and max(kz) - min(kz) > 0.1))
    # 10. 高さの一覧が全クラスを持つ
    ht = height_table()
    ok.append((f"10. 高さの一覧 {len(ht)} 行（未決なし・バック音は r17 で据え置き）", len(ht) == 11 and sum("未決" in h["status"] for h in ht) == 0))
    # 11. 乱数の鍵は用途ごとに一意で、別の流れになる（R70-01: 旧は雨の dba とキックの高さが同じ鍵 1 だった）
    ok.append(("11. KEYS の値は一意（雨 1・D10 2・D19 3・キック 11・バイク 12・電車 13）", len(set(KEYS.values())) == len(KEYS) and KEYS["rain_dba"] != KEYS["z_kickboard"]))
    u_rain = np.array([np.random.default_rng([9290000000 + i, SALT29, KEYS["rain_dba"]]).uniform() for i in range(400)])
    u_kick = np.array([np.random.default_rng([9290000000 + i, SALT29, KEYS["z_kickboard"]]).uniform() for i in range(400)])
    u_d10 = np.array([np.random.default_rng([9290000000 + i, SALT29, KEYS["d10_lat"]]).uniform() for i in range(400)])
    u_moto = np.array([np.random.default_rng([9290000000 + i, SALT29, KEYS["z_motorcycle"]]).uniform() for i in range(400)])
    c1 = float(np.corrcoef(u_rain, u_kick)[0, 1]); c2 = float(np.corrcoef(u_d10, u_moto)[0, 1])
    ok.append((f"11b. 同じ seed で雨の dba とキックの高さの乱数が別の流れ（相関 {c1:+.3f}）・D10 の横距離とバイクの高さも（{c2:+.3f}）", not np.allclose(u_rain, u_kick) and abs(c1) < 0.2 and abs(c2) < 0.2))
    # 12. 受け入れの検査が誤値を見つける（R70-04）: 速さの欄の倍・横距離 100・側の反転・wp の nan
    r = _fake_core_row(3300, "safe", "R", "static", d19="1", d19_tier="safe", d19_band="2", d19_speed_kmh="8.000", d19_lat_m="6.000")
    s_ok = v19r.sample_scene_v19(dict(r)); car = next(x for x in s_ok["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    found = []
    s1 = json.loads(json.dumps(_conv(s_ok))); c1s = next(x for x in s1["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0); c1s["speed_mps"] *= 2; found.append(bool(check_scene_v29(s1, r)))
    s2 = json.loads(json.dumps(_conv(s_ok))); c2s = next(x for x in s2["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    for q in c2s["wp"]: q[2] = -100.0
    c2s["d19_lat_m"] = 100.0; found.append(bool(check_scene_v29(s2, dict(r, d19_lat_m="100.000"))))
    s3 = json.loads(json.dumps(_conv(s_ok))); c3s = next(x for x in s3["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    for q in c3s["wp"]: q[2] = -q[2]
    found.append(bool(check_scene_v29(s3, r)))
    s4 = json.loads(json.dumps(_conv(s_ok))); c4s = next(x for x in s4["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0); c4s["wp"][0][3] = float("nan"); found.append(bool(check_scene_v29(s4, r)))
    s5 = json.loads(json.dumps(_conv(s_ok))); c5s = next(x for x in s5["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
    c5s["wp"][1][1] = c5s["wp"][0][1] + 2 * (c5s["wp"][1][1] - c5s["wp"][0][1]); found.append(bool(check_scene_v29(s5, r)))
    s6 = json.loads(json.dumps(_conv(s_ok))); c6s = next(x for x in s6["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0); c6s["wp"][1][0] = c6s["wp"][0][0]; found.append(bool(check_scene_v29(s6, r)))
    ok.append((f"12. 受け入れの検査が誤値を見つける: 速さの欄 2 倍・横 100 m・側の反転・高さ nan・軌道の x だけ 2 倍・時刻が増えない（RR70-03） → {sum(found)}/6 で違反・正しい本は違反 0", all(found) and not check_scene_v29(_conv(s_ok), r)))
    # 13. 契約の検査（R70-05）: 現物の params と一致・誤値を入れると見つける
    P_now = json.loads((ROOT / "md/design/v2final_params.json").read_text(encoding="utf-8"))
    mm = contract_mismatches(P_now)
    P_bad = json.loads(json.dumps(P_now)); P_bad["D70_HEIGHTS_LATERAL"]["walk_speed_mps"] = [1.0, 9.0]
    ok.append((f"13. 契約: 現物の params と描画器が一致（食い違い {len(mm)}: {mm[:2]}）・歩く速さを [1, 9] にした写しは見つかる", not mm and any("walk_speed" in x for x in contract_mismatches(P_bad))))
    n_ok = sum(1 for _, b in ok if b)
    for msg, b in ok:
        print(("OK " if b else "NG ") + msg)
    print(f"selftest {n_ok}/{len(ok)}")
    return 0 if n_ok == len(ok) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "height-table":
        for h in height_table():
            print(f"- {h['source']}: {h['z_m'][0]}〜{h['z_m'][1]} m・{h['status']}・{h['basis']}")
        sys.exit(0)
    print(__doc__)
