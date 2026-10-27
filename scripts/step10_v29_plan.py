# -*- coding: utf-8 -*-
"""step10_v29_plan.py — 変更票 #70（2026-10-19）の新しい plan（Astra R108-01: 雨と D10 は plan の行の値を描画器が読むので、定数を変えて描き直すだけでは新しい値にならない）。

入力 = 最終の学習 plan（out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv・60,502 行・v28 まで）。先に assignment_v2final_pre_v29.csv に写す（バイト単位）。
変えるもの（3 つ）:
  (1) 雨: 行ごとの雨の有無・種類・rate は保つ（割合は旧 50／35／15 のまま・#70 r16）。**rain_dba だけ** params の D2_RAIN_DBA（佐藤ら 2020 図 9 の範囲・#70 r13）から
      rng([seed, SALT29, 1]) で引き直す（同じ種の 2 本〔高さ違い〕は同じ値）。
  (2) D10（至近の徐行・close_slow=1）: 新しい列 cs_lat_m = 横距離 1.0〜1.5 m（rng([seed, SALT29, 2])・#70 r7）。cs_cpa_m（旧・3 次元）は記録として残すが描画器は読まない。
  (3) D19（注意・安全の徐行・#69 r2／r3・#70 で全部の描き直しに一緒に入れる）: 圏 2 × 速さの帯 4 × 左右 2 = 16 の組 × 50 場面 = 800 場面 × 高さ 2 = 1,600 行。
      元 = v11core・住宅街／日常・車ありの既存の場面から条件（scene_type・n_car・警告音）を写す。danger_tier = 圏・car_side = 左右。
      d19=1・d19_tier・d19_band（1〜4）・d19_speed_kmh（半開の帯 [5,7.5) [7.5,10) [10,12.5) [12.5,15]・小数 3 桁で保存し、読み直した値の帯が一致するまで引く）・
      d19_lat_m（注意 1.5 < d ≤ 3.2・安全 3.2 < d ≤ 15・小数 3 桁・読み直した値の圏が一致するまで引く）。
      train／val = 組ごとに 40／10（場面単位・同じ場面の 2 本は同じ側）。clip 番号 h1 91001+i / h2 92001+i（i < 800）・seed 9203000000+i。
      歩行・雨・高さは v28 と同じ規則（rng([seed, 7])・p13 の雨・rng([seed, SALT_H1／H2])）。grammar = "v29"。
総数 60,502 + 1,600 = 62,102 = params の N_ROWS_TOTAL_EXPECTED_V29。
params にも書く: D70_HEIGHTS_LATERAL（#70 の値・描画器 step11_v29_render の定数と一致を検査）・D19_SLOW_ADD（実績）・D2_RAIN_DBA・status は draft に戻す（fixed は本人）。

使い方: python scripts/step10_v29_plan.py            （plan と params を書く）
        python scripts/step10_v29_plan.py --check-params  （現物の params と描画器の契約だけ見る・R70-05）
        python scripts/step10_v29_plan.py --dry-run  （書かずに検査だけ）
        python scripts/step10_v29_plan.py selftest
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import step10_v27_plan as p27  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final" / "plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
PLAN_PRE = PLAN_DIR / "assignment_v2final_pre_v29.csv"
MANIFEST = PLAN_DIR / "manifest_v29_d19.csv"
README = PLAN_DIR / "README_plan_v29.md"
PARAMS = ROOT / "md" / "design" / "v2final_params.json"
NEW_COLS = ["cs_lat_m", "d19", "d19_tier", "d19_band", "d19_speed_kmh", "d19_lat_m"]
COLS = p27.COLS + NEW_COLS

from step11_v29_render import SALT29, KEYS as _KEYS  # noqa: E402  R70-01: 乱数の鍵は描画器の表が正（用途ごとに一意）
SALT_RAIN, SALT_D10, SALT_D19 = _KEYS["rain_dba"], _KEYS["d10_lat"], _KEYS["d19"]
ID1, ID2 = 91001, 92001
SEED0 = 9203000000
D19_TIERS = ("caution", "safe")
D19_SIDES = ("L", "R")
D19_BANDS = ((5.0, 7.5), (7.5, 10.0), (10.0, 12.5), (12.5, 15.0))
D19_SCENES_PER_CELL = 50
D19_VAL_PER_CELL = 10
D19_ELIGIBLE_SCENE = {"residential", "daily"}
D10_LAT = (1.0, 1.5)
TIER_LAT = {"critical": (1.0, 1.5), "caution": (1.5, 3.2), "safe": (3.2, 15.0)}
MANIFEST_COLS = ["tier", "band", "side", "speed_lo_kmh", "speed_hi_kmh", "lat_lo_m", "lat_hi_m", "scenes", "clips", "train", "val",
                 "clip_h1_first", "clip_h1_last", "clip_h2_first", "clip_h2_last", "seed_first", "seed_last"]


def band_of(v_kmh: float) -> int:
    v = float(v_kmh)
    if not np.isfinite(v) or not (D19_BANDS[0][0] <= v <= D19_BANDS[-1][1]):
        raise ValueError(f"D19 の速さ {v} km/h は 5〜15 の外")
    for i, (lo, hi) in enumerate(D19_BANDS, start=1):
        if lo <= v < hi:
            return i
    return len(D19_BANDS)


def tier_of_lateral(lat_m: float) -> str:
    d = float(lat_m)
    return "critical" if d <= 1.5 else ("caution" if d <= 3.2 else "safe")


def _draw_in_band(rng: np.random.Generator, band: int) -> str:
    lo, hi = D19_BANDS[band - 1]
    for _ in range(1000):
        s = f"{rng.uniform(lo, hi):.3f}"
        if band_of(float(s)) == band:
            return s
    raise RuntimeError("帯の中の速さが引けない")


def _draw_lat(rng: np.random.Generator, tier: str) -> str:
    lo, hi = TIER_LAT[tier]
    for _ in range(1000):
        s = f"{rng.uniform(lo, hi):.3f}"
        if tier_of_lateral(float(s)) == tier:
            return s
    raise RuntimeError("圏の中の横距離が引けない")


def rain_dba_v29(seed: int, kind: str, D2: dict) -> str:
    lo, hi = D2[kind]
    return f"{np.random.default_rng([int(seed), SALT29, SALT_RAIN]).uniform(float(lo), float(hi)):.1f}"


def cs_lat_v29(seed: int) -> str:
    return f"{np.random.default_rng([int(seed), SALT29, SALT_D10]).uniform(*D10_LAT):.3f}"


def update_existing(base: list, P: dict) -> list:
    """(1) 雨の dba を引き直す・(2) D10 に cs_lat_m を足す。ほかの列は 1 文字も変えない。"""
    D2 = {k: (float(v[0]), float(v[1])) for k, v in P["D2_RAIN_DBA"].items()}
    out = []
    for r in base:
        q = dict(r)
        for c in NEW_COLS:
            q.setdefault(c, "")
        if q.get("rain"):
            q["rain_dba"] = rain_dba_v29(int(q["seed"]), q["rain"], D2)
        if q.get("close_slow") == "1":
            q["cs_lat_m"] = cs_lat_v29(int(q["seed"]))
        out.append(q)
    return out


def build_d19(base: list, P: dict) -> tuple[list, list]:
    """(3) D19 の 1,600 行。"""
    p13 = p27._p13(P)
    walk = float(P["D4_WALK_FRAC"]); zr = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    pool = [r for r in base if r["scenario"] == "v11core" and r.get("scene_type") in D19_ELIGIBLE_SCENE and r.get("car_side") in ("L", "R")
            and r.get("h_copy") == "1" and r.get("close_slow") != "1"]
    assert len(pool) >= 16 * D19_SCENES_PER_CELL, len(pool)
    rng_pick = np.random.default_rng([SALT29, SALT_D19])
    picks = rng_pick.choice(len(pool), size=16 * D19_SCENES_PER_CELL, replace=False)
    rows, manifest = [], []
    i = 0
    for tier in D19_TIERS:
        for band in range(1, len(D19_BANDS) + 1):
            for side in D19_SIDES:
                first = i
                for k in range(D19_SCENES_PER_CELL):
                    tmpl = pool[int(picks[i])]
                    seed = SEED0 + i
                    split = "val" if k % 5 == 4 else "train"
                    motion = "walk" if np.random.default_rng([seed, 7]).random() < walk else "static"
                    rd = np.random.default_rng([seed, SALT29, SALT_D19])
                    r0 = {c: "" for c in COLS}
                    r0.update({"split": split, "motion": motion, "n_warnings": tmpl["n_warnings"], "w1_class": tmpl["w1_class"], "w1_side": tmpl["w1_side"],
                               "w2_class": tmpl["w2_class"], "w2_side": tmpl["w2_side"], "w3_class": "", "w3_side": "", "danger_tier": tier, "car_side": side,
                               "scenario": "v11core", "seed": str(seed), "scene_type": tmpl["scene_type"], "n_car": tmpl["n_car"], "grammar": "v29",
                               "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "", "cs_lat_m": "", "twin_of": "", "mix": "",
                               "d19": "1", "d19_tier": tier, "d19_band": str(band), "d19_speed_kmh": _draw_in_band(rd, band), "d19_lat_m": _draw_lat(rd, tier)})
                    for h, salt, idb in ((1, p27.SALT_H1, ID1), (2, p27.SALT_H2, ID2)):
                        r = dict(r0)
                        r["clip_id"] = f"{split}_syn_clip{idb + i:05d}"
                        r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*zr)):.3f}"
                        r["h_copy"] = str(h)
                        p27._rain(r, P, p13)
                        rows.append(r)
                    i += 1
                last = i - 1
                lo, hi = D19_BANDS[band - 1]; llo, lhi = TIER_LAT[tier]
                manifest.append({"tier": tier, "band": band, "side": side, "speed_lo_kmh": lo, "speed_hi_kmh": hi, "lat_lo_m": llo, "lat_hi_m": lhi,
                                 "scenes": D19_SCENES_PER_CELL, "clips": 2 * D19_SCENES_PER_CELL, "train": 2 * (D19_SCENES_PER_CELL - D19_VAL_PER_CELL),
                                 "val": 2 * D19_VAL_PER_CELL, "clip_h1_first": f"{ID1 + first:05d}", "clip_h1_last": f"{ID1 + last:05d}",
                                 "clip_h2_first": f"{ID2 + first:05d}", "clip_h2_last": f"{ID2 + last:05d}", "seed_first": SEED0 + first, "seed_last": SEED0 + last})
    assert i == 16 * D19_SCENES_PER_CELL and ID1 + i <= ID2, i
    return rows, manifest


def check(pre: list, upd: list, new: list, P: dict) -> list:
    """受け入れ（Astra R108-01: 変えてよい列と保つ列・場面と split と高さの組・雨の種類の実数・D10／D19 の横距離・同じ抽選の決まり）。"""
    msgs = []
    D2 = {k: (float(v[0]), float(v[1])) for k, v in P["D2_RAIN_DBA"].items()}
    msgs.append((f"元の行数 {len(pre):,} が保たれ、新しい列 {NEW_COLS} 以外の列の集まりは同じ", len(pre) == len(upd) and set(upd[0].keys()) == set(COLS)))
    keep = [c for c in p27.COLS if c != "rain_dba"]
    same_keep = all(all(a.get(c, "") == b.get(c, "") for c in keep) for a, b in zip(pre, upd))
    msgs.append(("既存の行: rain_dba と新しい列以外は 1 文字も変わらない（clip_id・seed・場面・split・高さ・雨の有無と種類・D10 の速さ）", same_keep))
    rain_rows = [(a, b) for a, b in zip(pre, upd) if a.get("rain")]
    in_rng = all(D2[b["rain"]][0] <= float(b["rain_dba"]) <= D2[b["rain"]][1] for _, b in rain_rows)
    changed = sum(a["rain_dba"] != b["rain_dba"] for a, b in rain_rows)
    norain_same = all(b["rain_dba"] == "" for a, b in zip(pre, upd) if not a.get("rain"))
    kinds = Counter(b["rain"] for _, b in rain_rows)
    msgs.append((f"雨: 種類は保ち（{dict(kinds)}・{len(rain_rows):,} 本 = {len(rain_rows) / len(upd):.1%}）・dba は全部新しい範囲 {D2}・引き直した本 {changed:,}・雨なしは空のまま",
                 in_rng and norain_same and len(rain_rows) == sum(1 for a in pre if a.get("rain"))))
    by_seed_dba = defaultdict(set)
    for _, b in rain_rows:
        by_seed_dba[b["seed"]].add(b["rain_dba"])
    msgs.append(("雨: 同じ種（高さ違いの 2 本）は同じ dba", all(len(v) == 1 for v in by_seed_dba.values())))
    cs = [b for b in upd if b.get("close_slow") == "1"]
    cs_ok = all(b["cs_lat_m"] != "" and D10_LAT[0] <= float(b["cs_lat_m"]) <= D10_LAT[1] for b in cs) and all(b["cs_lat_m"] == "" for b in upd if b.get("close_slow") != "1")
    by_seed_cs = defaultdict(set)
    for b in cs:
        by_seed_cs[b["seed"]].add(b["cs_lat_m"])
    lat_cs = [float(b["cs_lat_m"]) for b in cs]
    msgs.append((f"D10: {len(cs):,} 本に cs_lat_m（{min(lat_cs):.3f}〜{max(lat_cs):.3f}・1.0〜1.5）・同じ種は同じ値・徐行でない行は空", cs_ok and all(len(v) == 1 for v in by_seed_cs.values())))
    # D19
    msgs.append((f"D19: {len(new):,} 本 = 16 組 × {D19_SCENES_PER_CELL} 場面 × 2 高さ", len(new) == 16 * D19_SCENES_PER_CELL * 2))
    ids = [r["clip_id"] for r in upd + new]
    msgs.append(("clip_id は全部ユニーク", len(ids) == len(set(ids))))
    nums = [int(r["clip_id"][-5:]) for r in new]
    msgs.append((f"D19 の番号 {min(nums)}〜{max(nums)} は 91001〜92800 の中・既存の最大 {max(int(r['clip_id'][-5:]) for r in upd)} より上", min(nums) >= ID1 and max(nums) < ID2 + 16 * D19_SCENES_PER_CELL
                 and min(nums) > max(int(r["clip_id"][-5:]) for r in upd)))
    seeds = [r["seed"] for r in new]
    msgs.append(("D19 の seed は既存と重ならない・同じ seed は 2 本（高さ 1 と 2）", not (set(seeds) & set(r["seed"] for r in upd)) and all(c == 2 for c in Counter(seeds).values())))
    good = True; cells = Counter(); split_cells = Counter()
    for r in new:
        v = float(r["d19_speed_kmh"]); lat = float(r["d19_lat_m"])
        good &= r["d19"] == "1" and r["scenario"] == "v11core" and r["scene_type"] in D19_ELIGIBLE_SCENE and r["grammar"] == "v29"
        good &= band_of(v) == int(r["d19_band"]) and tier_of_lateral(lat) == r["d19_tier"] == r["danger_tier"] and r["car_side"] in D19_SIDES
        good &= r["close_slow"] == "" and r["cs_lat_m"] == "" and 1.4 <= float(r["mic_z"]) <= 2.1 and r["mix"] == "" and r["twin_of"] == ""
        good &= (r["d19_speed_kmh"] == f"{v:.3f}") and (r["d19_lat_m"] == f"{lat:.3f}")
        cells[(r["d19_tier"], int(r["d19_band"]), r["car_side"])] += 1
        split_cells[(r["d19_tier"], int(r["d19_band"]), r["car_side"], r["split"])] += 1
    msgs.append(("D19 の行: d19=1・v11core・住宅街／日常・grammar v29・帯 = 読み直した速さの帯・圏 = 読み直した横距離の圏 = danger_tier・高さ 1.4〜2.1・D10 の列は空", good))
    msgs.append((f"D19 の組: 16 組 × {2 * D19_SCENES_PER_CELL} 本", len(cells) == 16 and all(c == 2 * D19_SCENES_PER_CELL for c in cells.values())))
    msgs.append((f"D19 の train／val: 組ごとに {2 * (D19_SCENES_PER_CELL - D19_VAL_PER_CELL)}／{2 * D19_VAL_PER_CELL} 本",
                 all(split_cells[(t, b, s, "train")] == 2 * (D19_SCENES_PER_CELL - D19_VAL_PER_CELL) and split_cells[(t, b, s, "val")] == 2 * D19_VAL_PER_CELL
                     for (t, b, s) in cells)))
    by_seed = defaultdict(list)
    for r in new:
        by_seed[r["seed"]].append(r)
    twin_ok = all(len(v) == 2 and {v[0]["h_copy"], v[1]["h_copy"]} == {"1", "2"} and all(v[0][c] == v[1][c] for c in COLS if c not in ("clip_id", "mic_z", "h_copy")) for v in by_seed.values())
    msgs.append(("D19 の同じ場面の 2 本: 高さと clip_id 以外は同じ（側・圏・速さ・横距離・雨・歩行）", twin_ok))
    walk = np.mean([r["motion"] == "walk" for r in new]); rain = np.mean([bool(r["rain"]) for r in new])
    msgs.append((f"D19 の歩行 {walk:.1%}（0.70）・雨 {rain:.1%}（0.20）は 3σ 内", abs(walk - 0.7) < 3 * np.sqrt(0.21 / 800) and abs(rain - 0.2) < 3 * np.sqrt(0.16 / 800)))
    sp = [float(r["d19_speed_kmh"]) for r in new]; lt = {t: [float(r["d19_lat_m"]) for r in new if r["d19_tier"] == t] for t in D19_TIERS}
    msgs.append((f"D19 の速さ {min(sp):.3f}〜{max(sp):.3f} km/h・横距離 注意 {min(lt['caution']):.3f}〜{max(lt['caution']):.3f}・安全 {min(lt['safe']):.3f}〜{max(lt['safe']):.3f}",
                 5.0 <= min(sp) and max(sp) <= 15.0 and min(lt["caution"]) > 1.5 and max(lt["caution"]) <= 3.2 and min(lt["safe"]) > 3.2 and max(lt["safe"]) <= 15.0))
    msgs.append((f"総数 {len(upd) + len(new):,} = 62,102", len(upd) + len(new) == 62102))
    # 描画器の定数との一致（値の正は変更票 #70 → params D70 → 描画器）
    try:
        import step11_v29_render as v29
        agree = (v29.TIER_LAT == TIER_LAT and v29.D10_LAT == D10_LAT and v29.D19_BANDS == D19_BANDS and v29.D19_TIER_LAT == {t: TIER_LAT[t] for t in D19_TIERS})
        msgs.append(("描画器 step11_v29_render の圏・帯・D10 の範囲と一致", agree))
    except Exception as e:  # noqa: BLE001
        msgs.append((f"描画器 step11_v29_render を読めない: {e}", False))
    return msgs


def params_d70(P: dict) -> dict:
    """params に #70 の値を書く（宣言。描画器の定数と一致することを check_params_vs_renderer が見る）。"""
    import step11_v29_render as v29
    P["D70_HEIGHTS_LATERAL"] = {
        "decided": "2026-10-19 00:14〜03:48 本人（変更票 #70 r0〜r15）・r16 11:35（雨の割合は旧のまま・高さの下端 0.05 のまま・仰角を表に）・r17 13:15（バック音 0.8〜1.3 のまま・合成のベルは実測で変えない・試聴 OK・eval v3 の D19 は比例）",
        "src_z_m": {k: list(v) for k, v in v29.SRC_Z_CLASS.items() if v is not None} | {k: list(v) for k, v in v29.SRC_Z_FIXED_CLASS.items()} | {"bicycle": [0.1, 0.4], "crossing_static": [2.5, 2.5]},
        "rng_keys": dict(v29.KEYS),
        "tier_lateral_m": {k: list(v) for k, v in v29.TIER_LAT.items()},
        "tier_note": "圏は横距離（頭＝マイクの真下から車の中心線まで）で抽選。空きなし。旧 TIER_CPA（3 次元 0.6〜1.5／1.55〜3.0／3.2〜15）は使わない。記録の cpa_rel_target_m はマイク 1.5 m 基準の 3 次元 = hypot(横, 1.5−z)",
        "warn_offset_m": list(v29.WARN_OFFSET), "walk_speed_mps": list(v29.WALK_SPEED), "heavy_frac": v29.HEAVY_FRAC, "ev_frac": 0.15,
        "platoon_lane_min_m": v29.PLATOON_LANE_MIN, "d10_lateral_m": list(v29.D10_LAT),
        "level_match_src_z_m": 0.325, "wind": "none（#70 r11）", "avas": "今のまま（#70 r15・低い音 200〜1,700 Hz ＋ 半分の車に 2,500〜5,000 Hz）",
        "renderer": "scripts/step11_v29_render.py（土台の差し込み口 m9.draw_src_z／src_z_fixed／lane_offset／WALK_SPEED／WARN_OFFSET・v12.HEAVY_FRAC・v19r.PLATOON_LANE_MIN・v14.rewrite_close_slow を差し替える）",
        "plan": "scripts/step10_v29_plan.py（雨の dba を引き直す・D10 に cs_lat_m・D19 1,600 本）",
    }
    P["D2_RAIN_DBA"] = {"light": [34.0, 58.0], "moderate": [45.0, 61.0], "heavy": [49.0, 64.0]}
    P["D2_source"] = {"light": "sato2020_fig9", "moderate": "sato2020_fig9", "heavy": "sato2020_fig9"}
    P["D2_note"] = ("#70 r13（本人 2026-10-19 03:25）: 佐藤・栗栖・森本・前田 2020（Applied Acoustics 172:107616・採択済みの著者原稿・掲載 2021）の図 9 の、雨の音が主な 6 本の回帰式を "
                    "小雨 1.3〜9・中 10〜20・強 20〜40 mm/h の両端で計算した最小〜最大を丸めた設計の範囲（中を一様に引く・実測の分布ではない）。合成の小雨・中・強を 10 mm/h 未満・10〜20・20 以上とみなす仮定（r7）。"
                    "値は雨の成分だけの dB(A)（gain_for_spl_a・step11_v13_render）。rain_rate は合成の 1 秒あたりの粒の数で mm/h ではない（R109-05）。"
                    "旧 45〜50／50〜58／60〜68 は 9/2 の仮置き。新しい plan（step10_v29_plan）で行の rain_dba だけ引き直す（雨の有無・種類・rate は保つ）")
    P["RAIN_KIND_P_note"] = "#70 r16（本人 2026-10-19 11:31「網羅的にしましょう」）: 学習も評価も旧 50／35／15 のまま。千葉の実際の頻度（97／2.6／0.65%・10 mm/h 未満／10〜20／20 以上）は考察に書く（r4／r7／r10 の 0.9675／0.0260／0.0065 は取り消し）"
    P["D10_note_v29"] = "#70 r7: 至近の徐行の横距離は plan の cs_lat_m（1.0〜1.5 m・step10_v29_plan）。旧 cs_cpa_m（3 次元 0.6〜1.2）は記録だけで描画器は読まない。音量補正 cs_level_adj_db は v24 の曲線で上書きされるので効かない"
    d19 = {"adopt": True, "decided": "2026-10-17 05:28〜16:02 本人（#69 r2: 帯 4 × 圏 2 × 各 100 場面・左右 50/50・train/val 8:2）・10/17 19:xx（#69 r3: 横距離で数える・半開の帯・組ごとに 40/10）・"
                                       "2026-10-19（#70: 積み増しではなく全部の描き直しに一緒に入れる・圏の範囲は #70 の横距離）",
           "sampler": "v11core（住宅街／日常・車あり）の行を写し、step11_v29_render.rewrite_d19 が track0 の車を d19_speed_kmh・d19_lat_m に置き直す（ほかの音源は変えない・音量は v24 の速さの曲線だけ）",
           "tiers": list(D19_TIERS), "bands_kmh": [list(b) for b in D19_BANDS], "band_rule": "半開 [lo, hi)・15.0 は最後の帯・小数 3 桁で保存し読み直した値の帯が一致するまで引く",
           "lateral_m": {t: list(TIER_LAT[t]) for t in D19_TIERS}, "lateral_rule": "注意 1.5 < d ≤ 3.2・安全 3.2 < d ≤ 15（採点器と同じ境）・小数 3 桁",
           "scenes_per_cell": D19_SCENES_PER_CELL, "val_per_cell": D19_VAL_PER_CELL, "n_scenes": 16 * D19_SCENES_PER_CELL, "n_clips": 32 * D19_SCENES_PER_CELL,
           "train": 2 * 16 * (D19_SCENES_PER_CELL - D19_VAL_PER_CELL), "val": 2 * 16 * D19_VAL_PER_CELL,
           "clip_ids": f"h1 {ID1}〜{ID1 + 16 * D19_SCENES_PER_CELL - 1} / h2 {ID2}〜{ID2 + 16 * D19_SCENES_PER_CELL - 1}", "seed_base": SEED0,
           "note": "旧 D19_SLOW_TIERS（置換・adopt=false・2026-09-21）とは別物。割合は実際の道の割合ではなく、速さと圏の足りない所を埋める割り当て（#69）。eval v3 の帯ごとの本数は本人の判断（A3）"}
    P["D19_SLOW_ADD"] = d19
    P["D19_SLOW_TIERS"]["note_v29"] = "2026-10-19: 積み増し版は D19_SLOW_ADD（別物）。この鍵（置換版）は adopt=false のまま"
    return P


def check_params_vs_renderer(P: dict) -> list:
    """保存された params（現物）と描画器の契約（Astra R70-05: 書き直したあとの写しではなく、渡された dict をそのまま比べる）。"""
    import step11_v29_render as v29
    mm = v29.contract_mismatches(P)
    return [("params と描画器 step11_v29_render の契約が一致（高さ全クラス・圏・警告音の車・歩く速さ・車種割合・隊列・D10・雨の dB と割合・D19・較正の高さ・乱数の鍵）" + (f"・食い違い {mm[:3]}" if mm else ""), not mm)]


def write_outputs(upd: list, new: list, manifest: list, P: dict, msgs: list) -> None:
    if not PLAN_PRE.exists():
        shutil.copyfile(PLAN, PLAN_PRE)
    p27.write_rows(PLAN, upd + new, COLS)
    p27.write_rows(MANIFEST, manifest, MANIFEST_COLS)
    P["N_ROWS_TOTAL_EXPECTED_V29"] = len(upd) + len(new)
    P["_note_v29"] = (f"2026-10-19: #70 の新しい plan（step10_v29_plan）: 雨の dba を引き直し・D10 に cs_lat_m・D19 {len(new):,} 本を足した総数 = N_ROWS_TOTAL_EXPECTED_V29 = {len(upd) + len(new):,}。"
                      "V28 の 60,502 は据え置き。全部をこの plan と step11_v29_render で描き直す（出力先は新しい置き場）")
    P["status"] = "draft"
    P["draft_note"] = ("2026-10-19 Fable: #70（r0〜r16）の値を入れた（D70_HEIGHTS_LATERAL・D2_RAIN_DBA・D19_SLOW_ADD・N_ROWS_TOTAL_EXPECTED_V29）。fixed は本人（バック音の高さの判断・試聴・検査のあと）。"
                      "前の fixed = 2026-10-13T14:14（v28 まで）")
    PARAMS.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    L = [f"# plan v29（変更票 #70・2026-10-19・step10_v29_plan.py）", "",
         f"- 元の plan（{len(upd):,} 行・v28 まで）= assignment_v2final_pre_v29.csv sha256 {p27.sha256(PLAN_PRE)}",
         f"- 新しい plan（{len(upd) + len(new):,} 行）sha256 {p27.sha256(PLAN)}",
         f"- 雨: 有無・種類・rate は保ち、rain_dba だけ D2_RAIN_DBA {P['D2_RAIN_DBA']} から rng([seed, {SALT29}, {SALT_RAIN}]) で引き直した（#70 r13・割合は旧 50/35/15・r16）",
         f"- D10: close_slow=1 の {sum(r.get('close_slow') == '1' for r in upd):,} 本に cs_lat_m（横距離 {D10_LAT} m・rng([seed, {SALT29}, {SALT_D10}])）。cs_cpa_m は記録だけ",
         f"- D19: {len(new):,} 本（16 組 × {D19_SCENES_PER_CELL} 場面 × 2 高さ・clip h1 {ID1}〜 / h2 {ID2}〜・seed {SEED0}〜・manifest_v29_d19.csv）",
         "- 受け入れの検査:"] + [f"  - {'OK' if g else 'NG'} {m}" for m, g in msgs] + [""]
    README.write_text("\n".join(L), encoding="utf-8")


def selftest() -> int:
    ok = []
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    raw = check_params_vs_renderer(P)                      # R70-05: 現物（書き直す前）を先に見る
    ok.append(("現物の params（書き直す前）: " + raw[0][0], raw[0][1]))
    P = params_d70(json.loads(json.dumps(P)))
    base = p27.read_rows(PLAN_PRE if PLAN_PRE.exists() else PLAN)
    base = [{k: v for k, v in r.items() if k in p27.COLS} for r in base]          # 既存の列だけに戻す（2 回目以降の実行でも同じ入口）
    upd = update_existing(base, P)
    new, manifest = build_d19(base, P)
    msgs = check(base, upd, new, P) + check_params_vs_renderer(P)
    for m, g in msgs:
        ok.append((m, g))
    # 決定論: 2 回作って同じ
    upd2 = update_existing(base, P); new2, _ = build_d19(base, P)
    ok.append(("決定論: 2 回作って同じ", upd == upd2 and new == new2))
    # 帯と圏の境
    ok.append(("帯: 5.0→1・7.5→2・10.0→3・12.5→4・15.0→4・7.499→1", [band_of(x) for x in (5.0, 7.5, 10.0, 12.5, 15.0, 7.499)] == [1, 2, 3, 4, 4, 1]))
    ok.append(("圏: 1.5→至近・1.501→注意・3.2→注意・3.201→安全", [tier_of_lateral(x) for x in (1.5, 1.501, 3.2, 3.201)] == ["critical", "caution", "caution", "safe"]))
    # 生成器の入口が V29 を V28 より先に読む
    drv = (ROOT / "scripts" / "_run_v2final_gen.py").read_text(encoding="utf-8")
    ok.append(("生成器の入口が N_ROWS_TOTAL_EXPECTED_V29 を V28 より先に読み、step11_v29_render を鎖に入れている",
               drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V29")') != -1 and drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V29")') < drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V28")')
               and "import step11_v29_render" in drv and '"step11_v29_render.py"' in drv))
    n_ok = sum(1 for _, g in ok if g)
    for m, g in ok:
        print(("OK " if g else "NG ") + m)
    print(f"selftest {n_ok}/{len(ok)}")
    return 0 if n_ok == len(ok) else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    if "--check-params" in sys.argv[1:]:                  # 配備の前・検査器から: 現物の params と描画器の契約だけ見る（書かない）
        msgs = check_params_vs_renderer(json.loads(PARAMS.read_text(encoding="utf-8")))
        for m, g in msgs:
            print(("OK " if g else "NG ") + m)
        return 0 if all(g for _, g in msgs) else 1
    dry = "--dry-run" in sys.argv[1:]
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    src = PLAN_PRE if PLAN_PRE.exists() else PLAN
    base = p27.read_rows(src)
    assert len(base) == int(P["N_ROWS_TOTAL_EXPECTED_V28"]), (len(base), P["N_ROWS_TOTAL_EXPECTED_V28"])
    base = [{k: v for k, v in r.items() if k in p27.COLS} for r in base]
    P = params_d70(P)
    upd = update_existing(base, P)
    new, manifest = build_d19(base, P)
    msgs = check(base, upd, new, P) + check_params_vs_renderer(P)
    for m, g in msgs:
        print(("OK " if g else "NG ") + m)
    if not all(g for _, g in msgs):
        print("NG があるので書かない"); return 1
    if dry:
        print(f"--dry-run: 書かない（{len(upd) + len(new):,} 行）"); return 0
    write_outputs(upd, new, manifest, P, msgs)
    print("書いた:", PLAN, "/ 前の plan:", PLAN_PRE, "/ manifest:", MANIFEST, "/ README:", README, "/ params: D70・D2・D19_SLOW_ADD・N_ROWS_TOTAL_EXPECTED_V29・status draft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
