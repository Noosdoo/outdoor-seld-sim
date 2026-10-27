# -*- coding: utf-8 -*-
"""step11_v24_render.py — 音の大きさの決め方（2026-10-12・本人決定 15:28〜15:33 の 6 点・依頼文 r5 §2・順番の 6）。v23 の上に重ねる版。

変えるのは**音の大きさ（1 m の値 l1m_db と法規の値 law_db）・サイレンの高さと向きの差・最大値で合わせる音の較正**だけ。
場面の乱数の消費は変えない（独立の乱数 np.random.default_rng([seed, 20261013, 印, track, 元の l1m から作った整数])）。

1. 車（car_drive・v11core／v19 fast・turn・platoon・launch／D10 徐行）: l1m = 曲線 l2m(v) + Δ(v) + U(−4, 4)。曲線は側方 2 m の実測（level_v24.l2m_curve）、
   Δ(v) は描画で合わせた差（md/design/level_match_v24.json・scripts/level_match_v24.py）。law_db には 2 m の値（曲線 + ばらつき）を入れる。
   旧: 80〜87 dB(A)@1 m（速さに関係なく）・徐行 −10〜−6・発進 ±0。v19fast の +30 log(v/8) の補正・D10 の補正・発進の補正は使わない（曲線が速さを含む）。
2. 発進（v19launch）: 上に + U(4, 8) dB（その本の最後の速さの曲線に・1 本の中で大きさは一定＝道具の限界）。
3. バイク（motorcycle・moped）: l1m = 曲線 l2m(v) + Δ_バイク(v) + U(2.4, 3.8) + U(−4, 4)。旧: 原付 87.5〜96.5・二輪 89.5〜99.5。
4. サイレン: 前方 20 m で U(90, 105)（旧 90〜120）→ l1m = lv + 20 log 20。スピーカーの高さ z = U(2.2, 2.6) m（旧 0.8〜1.3・軌道の xy と時刻は同じ）。
   向きの差: ドライ音に、放射の時刻ごとの「進む向きとマイクの方向の角度」から 前 0・横 −12・後ろ −14 dB（直線でつなぐ）の利得を掛ける。
   較正は**前（0°）の値が l1m になる**ように、v13 の RMS 合わせに渡す l1m_db を 20 log(RMS(利得つき)/RMS(素)) だけ下げる（l1m_decl_db に宣言値を残す）。
5. 最大値で合わせる音（bike_bell・horn・backup_beep・crossing）: 鳴動窓の中の LAFmax（A・F 125 ms）が l1m になるように、v13 の RMS 合わせに渡す l1m_db を
   （LAFmax − RMS）だけ下げる（l1m_decl_db に宣言値・v24.crest_db に差を残す）。走行音とサイレンは RMS のまま。
6. バック音の 1 m の値: 50% 実勢 U(85, 95)・50% 「通常」U(77, 92)（旧の R165 側 60〜92 @1 m は 7 m の値を 1 m に写していた）。

使い方:
  python scripts/level_match_v24.py                      # 先に Δ の表を作る
  python scripts/step11_v24_render.py selftest           # 代表の場面で検査（描画は少し）
  python scripts/step11_v24_render.py audition <出力dir>  # 大きさだけ旧（v23 まで）と新（v24）を同じ場面で描いて W チャンネルを並べる
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v23_render as v23  # noqa: E402
from outdoor_seld import level_v24 as L24  # noqa: E402
from outdoor_seld.calibration import gain_for_spl_a, spl_a  # noqa: E402
from outdoor_seld.geometry import receiver_positions_at  # noqa: E402

v22, m9, v13, v19r = v23.v22, v23.m9, v23.v13, v23.v19r
SALT = 20261013
MATCH_JSON = ROOT / "md/design/level_match_v24.json"
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
ENABLED = {"on": True}          # 試聴で「大きさだけ旧」を描くときだけ False
_MATCH = {}


def match_table() -> dict:
    if not _MATCH:
        if not MATCH_JSON.is_file():
            raise FileNotFoundError(f"{MATCH_JSON} が無い → 先に python scripts/level_match_v24.py")
        _MATCH.update(json.loads(MATCH_JSON.read_text(encoding="utf-8")))
    return _MATCH


def delta_db(kind: str, v_kmh: float) -> float:
    t = match_table()["tables"][kind]
    return float(np.interp(v_kmh, t["speeds_kmh"], t["delta_db"]))


def siren_delta_db(siren_type: str, v26: bool) -> float:
    """サイレンの 20 m の値 → 1 m の値の換算で引く描画の差（level_match_v24.json の siren・型ごと・v26 と旧で別の表）。表が無ければ落とす（0 で黙って通さない）。"""
    t = match_table().get("siren")
    if not t:
        raise KeyError(f"{MATCH_JSON} に siren の表が無い → python scripts/level_match_v24.py を回す")
    chain = "v26" if v26 else "old"
    return float(t["tables"][chain][siren_type]["delta_db"])


def base_l1m(kind: str, v_kmh: float) -> float:
    """曲線（側方 2 m）+ Δ(v)（描画で合わせた 1 m との差）。kind = car / moped / motorcycle。"""
    return L24.l2m_curve(v_kmh) + delta_db(kind, v_kmh)


def _irng(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


# ---------------------------------------------------------------- 場面のあと: 大きさ・サイレンの高さ
def apply_levels(s: dict, row: dict) -> dict:
    seed = int(row["seed"]); mic = np.asarray(s["_mic_arr"], dtype=float)
    for src in s["sources"]:
        cls = src["class"]; old = float(src.get("l1m_db", 0.0)); tr = int(src.get("track", 0) or 0)
        if cls == "car_drive":
            r = _irng(seed, SALT, 1, tr, round(old * 1000))
            v = float(src["speed_mps"]) * 3.6
            l2m = L24.l2m_curve(v); d = delta_db("car", v)
            spread = float(r.uniform(*L24.CAR_SPREAD_DB))
            bonus = float(r.uniform(*L24.LAUNCH_BONUS_DB)) if src.get("launch") else 0.0
            hoff = float((src.get("heavy") or {}).get("offset_db", 0.0))        # 大型車（v25・#52）: 同じ速さの乗用車の法則に足す dB
            src["v24"] = {"rule": "car_curve", "v_kmh": round(v, 2), "l2m_db": round(l2m, 2), "delta_db": round(d, 2), "spread_db": round(spread, 2),
                          "launch_bonus_db": round(bonus, 2), "heavy_offset_db": round(hoff, 2), "l1m_old_db": round(old, 2), "close_slow": bool(src.get("close_slow"))}
            src["law_db"] = l2m + spread + bonus + hoff
            src["l1m_db"] = l2m + d + spread + bonus + hoff
        elif cls == "motorcycle":
            kind = src.get("params", {}).get("engine_class", "motorcycle"); v = float(src["speed_mps"]) * 3.6
            r = _irng(seed, SALT, 2, tr, round(old * 1000))
            off = float(r.uniform(*L24.BIKE_OFFSET_DB)); spread = float(r.uniform(*L24.CAR_SPREAD_DB))
            l2m = L24.l2m_curve(v); d = delta_db(kind, v)
            src["v24"] = {"rule": "bike_curve", "engine_class": kind, "v_kmh": round(v, 2), "l2m_db": round(l2m, 2), "delta_db": round(d, 2),
                          "bike_offset_db": round(off, 2), "spread_db": round(spread, 2), "l1m_old_db": round(old, 2)}
            src["law_db"] = l2m + off + spread
            src["l1m_db"] = l2m + d + off + spread
        elif cls == "siren":
            r = _irng(seed, SALT, 3, tr, round(old * 1000))
            lv = float(r.uniform(*L24.SIREN_LV_20M_DB)); z = float(r.uniform(*L24.SIREN_Z_M))
            sp = src.get("params", {}); dsir = siren_delta_db(str(sp.get("siren_type")), bool(sp.get("v26")))   # 描画で合わせた差（地面の反射の分）
            src["law_db"] = lv; src["l1m_db"] = lv + 20.0 * float(np.log10(L24.SIREN_REF_M)) - dsir
            src["wp"] = [[float(p[0]), float(p[1]), float(p[2]), z] for p in src["wp"]]
            src["v24"] = {"rule": "siren", "lv_20m_db": round(lv, 2), "z_m": round(z, 3), "directivity_db": dict(L24.SIREN_DIR_DB), "delta_db": round(dsir, 2),
                          "siren_chain": "v26" if sp.get("v26") else "old", "l1m_old_db": round(old, 2)}
            src["_mic_arr_v24"] = mic.tolist()
        elif cls == "backup_beep":
            r = _irng(seed, SALT, 4, tr, round(old * 1000))
            legacy = bool(r.random() < 0.5)
            lv = float(r.uniform(*(L24.BACKUP_LEGACY_1M if legacy else L24.BACKUP_NORMAL_1M)))
            src["law_db"] = lv; src["l1m_db"] = lv
            src["v24"] = {"rule": "backup_1m", "legacy": legacy, "l1m_old_db": round(old, 2)}
        is_train_car = bool((src.get("params") or {}).get("train_car")) or "train_car_index" in src   # 列車の車両の印は params.train_car（step11_v12 の v12train）
        if cls in L24.MATCH_MAX_CLASSES and not is_train_car:
            # 最大値合わせは出典が最大値の警告音（ベル・クラクション・バック音・踏切の警報機）だけ。列車の車両（class は crossing だが train_car=True・
            # v12train の背景の音・宣言値は通過音の平均）は RMS 合わせのまま（AR12-08・2026-10-12）
            src["match_max"] = True
            src.setdefault("v24", {})["match"] = "LAFmax"
        elif cls in L24.MATCH_MAX_CLASSES and is_train_car:
            src.setdefault("v24", {})["match"] = "RMS（列車の車両・AR12-08）"
    return s


def _scene_v24(prev):
    def fn(row: dict) -> dict:
        s = prev(row)
        return apply_levels(s, row) if ENABLED["on"] else s
    fn._v24 = True
    return fn


# ---------------------------------------------------------------- ドライ音のあと: 向きの差・最大値合わせ（l1m_db を v13 の RMS 合わせ用に直す）
def _directivity_gain(src: dict, n: int, fs: int, pre: float) -> np.ndarray:
    wp = np.array(src["wp"], dtype=float)
    if pre > 0:
        wp = v13._extend_back(wp, pre)
    t = np.arange(n) / fs - pre                               # 放射の時刻（ドライ音の標本 = 放射時刻）
    step = max(1, int(0.01 * fs))
    tc = t[::step]
    p = receiver_positions_at(tc, wp)
    mic = np.asarray(src["_mic_arr_v24"], dtype=float)
    m = receiver_positions_at(tc, mic) if mic.ndim == 2 else np.broadcast_to(mic[:3], p.shape)
    d = m - p
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    head = np.array([float(src.get("dir_x", 1.0)), 0.0, 0.0])
    cos = np.clip(dn @ head, -1.0, 1.0)
    gdb = L24.directivity_db(np.degrees(np.arccos(cos)))
    g = 10.0 ** (np.asarray(gdb, dtype=float) / 20.0)
    return np.interp(t, tc, g)


def _seg(src: dict, n: int, fs: int, pre: float) -> slice:
    if pre > 0:
        return slice(0, n)
    a0, a1 = int(float(src["t_on"]) * fs), int(float(src["t_off"]) * fs)
    return slice(a0, max(a1, a0 + 1))


def _make_dry_v24(prev):
    def fn(src: dict) -> np.ndarray:
        dry = prev(src)
        if not ENABLED["on"]:
            return dry
        fs = m9.FS_SIM; n = len(dry); pre = float(src.get("preroll_s", 0.0))
        seg = _seg(src, n, fs, pre)
        if src["class"] == "siren" and src.get("v24", {}).get("directivity_db") and src.get("_mic_arr_v24") is not None:
            g = _directivity_gain(src, n, fs, pre)
            X = dry * g
            decl = src.setdefault("l1m_decl_db", float(src["l1m_db"]))
            src["l1m_db"] = decl + (spl_a(X[seg], fs) - spl_a(dry[seg], fs))     # 前（0°）の値が decl になる
            src["v24"]["dir_mean_db"] = round(float(20.0 * np.log10(float(np.mean(g[seg])) + 1e-12)), 2)
            return X
        if src.get("match_max"):
            decl = src.setdefault("l1m_decl_db", float(src["l1m_db"]))
            w = dry if pre > 0 else m9._window(dry, float(src["t_on"]), float(src["t_off"]))
            R = spl_a(w[seg], fs); M = L24.lafmax_db(w[seg], fs)
            src["l1m_db"] = decl - (M - R)
            src.setdefault("v24", {})["crest_db"] = round(M - R, 2)
            return dry
        return dry
    fn._v24 = True
    return fn


# ---------------------------------------------------------------- 登録
if not getattr(v19r.sample_scene_v19, "_v24", False):
    _wrapped = _scene_v24(v19r.sample_scene_v19)
    v19r.sample_scene_v19 = _wrapped
    v13.sample_scene_v13 = _wrapped                 # generate_clip_v13 はこの名前を呼ぶ
    v22.sample_scene_v19 = _wrapped; v23.sample_scene_v19 = _wrapped
if not getattr(m9._make_dry, "_v24", False):
    m9._make_dry = _make_dry_v24(m9._make_dry)
sample_scene_v19 = v19r.sample_scene_v19
generate_clip = v19r.generate_clip


# ---------------------------------------------------------------- 検査
def check_levels(s: dict) -> None:
    for src in s["sources"]:
        cls = src["class"]; info = src.get("v24", {})
        if cls == "car_drive":
            assert info.get("rule") == "car_curve", "車に v24 が無い"
            exp = L24.l2m_curve(info["v_kmh"]) + delta_db("car", info["v_kmh"]) + info["spread_db"] + info["launch_bonus_db"] + info.get("heavy_offset_db", 0.0)
            assert abs(float(src["l1m_db"]) - exp) < 0.02, f"車の l1m {src['l1m_db']} ≠ 曲線 {exp}"
            assert L24.CAR_SPREAD_DB[0] - 1e-6 <= info["spread_db"] <= L24.CAR_SPREAD_DB[1] + 1e-6
            if src.get("launch"):
                assert L24.LAUNCH_BONUS_DB[0] - 1e-6 <= info["launch_bonus_db"] <= L24.LAUNCH_BONUS_DB[1] + 1e-6, "発進の分が範囲外"
            else:
                assert info["launch_bonus_db"] == 0.0
        elif cls == "motorcycle":
            assert info.get("rule") == "bike_curve" and L24.BIKE_OFFSET_DB[0] - 1e-6 <= info["bike_offset_db"] <= L24.BIKE_OFFSET_DB[1] + 1e-6
        elif cls == "siren":
            assert L24.SIREN_LV_20M_DB[0] - 1e-6 <= info["lv_20m_db"] <= L24.SIREN_LV_20M_DB[1] + 1e-6, "サイレンの 20 m の値が範囲外"
            assert all(L24.SIREN_Z_M[0] - 1e-6 <= p[3] <= L24.SIREN_Z_M[1] + 1e-6 for p in src["wp"]), "サイレンの高さが範囲外"
        elif cls == "backup_beep":
            lo, hi = (L24.BACKUP_LEGACY_1M if info["legacy"] else L24.BACKUP_NORMAL_1M)
            assert lo - 1e-6 <= float(src["l1m_db"]) <= hi + 1e-6, "バック音の 1 m の値が範囲外"
        if cls in L24.MATCH_MAX_CLASSES:
            if (src.get("params") or {}).get("train_car") or "train_car_index" in src:
                assert not src.get("match_max"), "列車の車両は最大値合わせの対象外（AR12-08）"
            else:
                assert src.get("match_max") is True


def check_match_max_dry(src: dict) -> float:
    """最大値合わせ: v13 と同じ RMS 合わせの利得を掛けたあと、窓の中の LAFmax が宣言値になる。戻り値 = 差 [dB]。"""
    fs = m9.FS_SIM
    X = m9._make_dry(src)                                       # l1m_db が RMS 用に直る
    w = m9._window(X, float(src["t_on"]), float(src["t_off"]))
    seg = _seg(src, len(X), fs, 0.0)
    g = gain_for_spl_a(w[seg], fs, float(src["l1m_db"]))
    return L24.lafmax_db(w[seg] * g, fs) - float(src["l1m_decl_db"])


def check_siren_front_level(src: dict) -> float:
    """向きの差: 前の値が宣言どおり。素のドライ音（向きの差なし）に v13 の利得を掛けた RMS と宣言値の差 [dB] を返す。"""
    fs = m9.FS_SIM
    X = m9._make_dry(src)
    pre = float(src.get("preroll_s", 0.0)); seg = _seg(src, len(X), fs, pre)
    g = gain_for_spl_a(X[seg], fs, float(src["l1m_db"]))
    ENABLED["on"] = False
    try:
        plain = m9._make_dry(src)                               # 向きの差なし・大きさの直しなし（v23 まで）
    finally:
        ENABLED["on"] = True
    return spl_a(plain[seg] * g, fs) - float(src["l1m_decl_db"])


def _plan_rows(pred, n: int) -> list:
    out = []
    with open(PLAN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if pred(r):
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); out.append(r)
                if len(out) >= n:
                    break
    return out


_V23_DUMP = r"""
import json, sys
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import step11_v23_render as v23
rows = json.loads(sys.stdin.read()); out = []
for r in rows:
    s = v23.sample_scene_v19(r)
    out.append({"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "kind": x["kind"], "track": x["track"], "wp_xy": [[p[0], p[1], p[2]] for p in x["wp"]],
                 "t_on": x["t_on"], "t_off": x["t_off"], "speed": x.get("speed_mps"), "t_cpa": x.get("t_cpa_s")} for x in s["sources"]]})
print(json.dumps(out))
"""


def selftest() -> int:
    ok = []
    rows = {
        "car": _plan_rows(lambda r: r["scenario"] == "v11core" and r["close_slow"] != "1", 3),
        "slow": _plan_rows(lambda r: r["scenario"] == "v11core" and r["close_slow"] == "1", 2),
        "launch": _plan_rows(lambda r: r["scenario"] == "v19launch", 2),
        "fast": _plan_rows(lambda r: r["scenario"] == "v19fast", 2),
        "bike": _plan_rows(lambda r: r["scenario"] == "v12bike", 3),
        "siren": _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "siren", 3),
        "beep": _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "backup_beep", 2),
        "horn": _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "horn", 2),
        "bell": _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "bike_bell", 2),
        "train": _plan_rows(lambda r: r["scenario"] == "v12train", 1),
    }
    scenes = {k: [sample_scene_v19(r) for r in v] for k, v in rows.items()}
    for k, v in scenes.items():
        for s in v:
            check_levels(s)
    ok.append(("全部の代表の場面で check_levels（車の曲線・発進の分・バイクの差・サイレンの範囲と高さ・バック音の範囲・最大値合わせの印）が通る", True))
    cars = [x for s in scenes["car"] + scenes["slow"] + scenes["fast"] for x in s["sources"] if x["class"] == "car_drive"]
    ok.append((f"車 {len(cars)} 台: 新しい l1m は旧（80〜87・徐行 70〜81・fast +8 まで）より小さい（例 {[(c['v24']['v_kmh'], round(c['l1m_db'], 1), c['v24']['l1m_old_db']) for c in cars[:4]]}）",
               all(float(c["l1m_db"]) < c["v24"]["l1m_old_db"] + 0.01 for c in cars)))
    la = [x for s in scenes["launch"] for x in s["sources"] if x["class"] == "car_drive"]
    ok.append((f"発進 {len(la)} 台: 発進の分 +4〜8 dB が付く（{[x['v24']['launch_bonus_db'] for x in la]}）", all(4.0 <= x["v24"]["launch_bonus_db"] <= 8.0 for x in la)))
    bk = [x for s in scenes["bike"] for x in s["sources"] if x["class"] == "motorcycle"]
    ok.append((f"バイク {len(bk)} 台: 車の曲線 + 2.4〜3.8（例 {[(x['v24']['engine_class'], x['v24']['v_kmh'], round(x['l1m_db'], 1), x['v24']['l1m_old_db']) for x in bk[:3]]}）", len(bk) >= 2))
    sr = [x for s in scenes["siren"] for x in s["sources"] if x["class"] == "siren"]
    fronts = [check_siren_front_level(x) for x in sr]
    ok.append((f"サイレン {len(sr)} 台: 向きの差を掛けても前の値が宣言どおり（差 {[round(f, 3) for f in fronts]} dB）・高さ 2.2〜2.6（{[x['v24']['z_m'] for x in sr]}）",
               all(abs(f) < 0.05 for f in fronts)))
    _tc = lambda x: bool((x.get("params") or {}).get("train_car")) or "train_car_index" in x
    mm = [x for k in ("beep", "horn", "bell", "train") for s in scenes[k] for x in s["sources"] if x["class"] in L24.MATCH_MAX_CLASSES and not x.get("preroll_s") and not _tc(x)]
    tc = [x for s in scenes["train"] for x in s["sources"] if _tc(x)]
    ok.append((f"列車の車両 {len(tc)} 台は最大値合わせの対象外（RMS のまま・AR12-08）、踏切の警報機 {sum(1 for s in scenes['train'] for x in s['sources'] if x['class'] == 'crossing' and not _tc(x))} 本は対象",
               len(tc) >= 1 and all(not x.get("match_max") for x in tc)))
    diffs = [check_match_max_dry(x) for x in mm]
    ok.append((f"最大値合わせ {len(mm)} 本（{sorted(set(x['class'] for x in mm))}）: RMS 合わせの利得を掛けたあとの LAFmax が宣言値 ±0.1 dB（差 {[round(d, 3) for d in diffs]}・crest {[x['v24']['crest_db'] for x in mm]}）",
               len(mm) >= 4 and all(abs(d) < 0.1 for d in diffs)))
    # 場面の乱数: v23 だけの別プロセスと、大きさ以外（軌道の xy・時刻・暗騒音）が一致
    sample = rows["car"][:1] + rows["slow"][:1] + rows["launch"][:1] + rows["bike"][:1] + rows["siren"][:1] + rows["beep"][:1] + rows["horn"][:1]
    res = subprocess.run([sys.executable, "-c", _V23_DUMP % (str(ROOT / "scripts"), str(ROOT / "src"))], input=json.dumps(sample), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        ref = json.loads(res.stdout.strip().splitlines()[-1]); diffs2 = []
        for a, r in zip(ref, sample):
            s = sample_scene_v19(r)
            b = {"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "kind": x["kind"], "track": x["track"], "wp_xy": [[p[0], p[1], p[2]] for p in x["wp"]],
                                                           "t_on": x["t_on"], "t_off": x["t_off"], "speed": x.get("speed_mps"), "t_cpa": x.get("t_cpa_s")} for x in s["sources"]]}
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                diffs2.append(r["clip_id"])
        ok.append((f"場面の乱数: v23 だけの別プロセスと {len(sample)} 本の軌道 xy・時刻・暗騒音が一致（違いは大きさとサイレンの z だけ）{'' if not diffs2 else diffs2}", not diffs2))
    else:
        ok.append(("場面の乱数: 別プロセスの v23 が動く", False)); print(res.stderr[-600:])
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# ---------------------------------------------------------------- 試聴（大きさだけ旧 vs 新・同じ場面・W チャンネル・同じ再生ゲイン）
def audition(out_dir: Path) -> int:
    import shutil
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_render"; shutil.rmtree(tmp, ignore_errors=True); (tmp / "work").mkdir(parents=True)
    keep = (m9.DS, m9.WORK, v19r.DS_V19)
    m9.DS, m9.WORK, v19r.DS_V19 = tmp, tmp / "work", tmp
    REF_DB = 90.0          # 再生ゲイン: 90 dB(A) = 0 dBFS（全ファイル共通。音量を上げすぎない）
    from outdoor_seld.calibration import K_RMS_SPL
    gain = 10.0 ** ((0.0 - (REF_DB - K_RMS_SPL)) / 20.0)
    picks = [("car_slow_D10", lambda r: r["scenario"] == "v11core" and r["close_slow"] == "1" and r["n_warnings"] == "0"),
             ("car_core", lambda r: r["scenario"] == "v11core" and r["close_slow"] != "1" and r["n_warnings"] == "0" and r["danger_tier"] == "critical"),
             ("car_fast", lambda r: r["scenario"] == "v19fast"),
             ("car_launch", lambda r: r["scenario"] == "v19launch"),
             ("bike", lambda r: r["scenario"] == "v12bike"),
             ("siren", lambda r: r["scenario"] == "v11core" and r["w1_class"] == "siren" and r["n_warnings"] == "1"),
             ("horn", lambda r: r["scenario"] == "v11core" and r["w1_class"] == "horn" and r["n_warnings"] == "1"),
             ("bell", lambda r: r["scenario"] == "v11core" and r["w1_class"] == "bike_bell" and r["n_warnings"] == "1"),
             ("backup", lambda r: r["scenario"] == "v11core" and r["w1_class"] == "backup_beep" and r["n_warnings"] == "1")]
    lines = ["音の大きさの直し（v24・2026-10-12）の試聴: 同じ場面を「大きさだけ旧（v23 まで）」と「新（v24）」で描いた W チャンネル（24 kHz・暗騒音と雨込み）。"
             f"全ファイル同じ再生ゲイン（{REF_DB:.0f} dB(A) = 0 dBFS）なので、ファイル間の大きさの違いがそのまま設計の違い。wav は git に入れない（README だけ）。", ""]
    try:
        for name, pred in picks:
            rows = _plan_rows(pred, 1)
            if not rows:
                lines.append(f"{name}: plan に該当の本が無い"); continue
            row = rows[0]
            for tag, on in (("old", False), ("new", True)):
                ENABLED["on"] = on
                shutil.rmtree(tmp / "foa", ignore_errors=True); shutil.rmtree(tmp / "work" / row["clip_id"], ignore_errors=True)
                generate_clip(row)
                f = next((tmp / "foa").glob(row["clip_id"] + ".*"))
                x, sr = sf.read(str(f), dtype="float64", always_2d=True)
                w = x[:, 0] * gain
                pk = float(np.max(np.abs(w)))
                if pk > 0.99:
                    w = w * 0.99 / pk
                sf.write(str(out_dir / f"{name}_{tag}.wav"), (w * 32767).astype(np.int16), sr, subtype="PCM_16")
                s = json.loads((tmp / "work" / row["clip_id"] / "scene.json").read_text(encoding="utf-8"))
                mains = [z for z in s["sources"] if z["class"] in ("car_drive", "motorcycle", "siren", "horn", "bike_bell", "backup_beep")]
                desc = "; ".join(f"{z['class']} v={float(z.get('speed_mps', 0)) * 3.6:.0f} km/h l1m={float(z.get('l1m_decl_db', z['l1m_db'])):.1f}" + (f" 2m={float(z['law_db']):.1f}" if z["class"] in ("car_drive", "motorcycle") and on else "") for z in mains[:3])
                lines.append(f"{name}_{tag}.wav: {row['clip_id']}（{row['scenario']}・暗騒音 {s['noise']['dba']:.0f} dB(A)・雨 {row.get('rain') or 'なし'}）{'新' if on else '旧'}: {desc}" + (f"（ピーク {20 * np.log10(0.99 / pk):+.1f} dB 下げ）" if pk > 0.99 else ""))
                print("wrote", f"{name}_{tag}.wav")
    finally:
        ENABLED["on"] = True
        m9.DS, m9.WORK, v19r.DS_V19 = keep
        shutil.rmtree(tmp, ignore_errors=True)
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "audition":
        sys.exit(audition(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "audition_2026-10-12_level"))
    print(__doc__); sys.exit(2)
