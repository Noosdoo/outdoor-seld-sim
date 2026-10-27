# -*- coding: utf-8 -*-
"""step11_v25_render.py — 大型車（変更票 #52 D25）と、ベルつき走行のベルの時刻（#53 D26）（2026-10-12・順番の 4）。v24 の上に重ねる版。

1. 大型車 v25heavy（**2026-10-12 18:17 本人が不採用**＝試聴で車と区別がつかず、耳で分かる要素の大きさに測定の根拠が無い。plan は 31,274 行に戻し、この枝は params の D25_HEAVY.adopt が true のときだけ使う）（plan は step10_v25_plan.py・manifest_v25_heavy.csv）: 元の場面の種類（twin_of の scenario）のサンプラを新しい seed で引き、車の中から
   固定 seed で一様に 1 台を大型車にする（params["heavy"]）。ドライ音 = engine_heavy_v25.make_heavy_car（発火 U(55, 80) Hz・250 Hz 以下 +10 dB・A 特性のタイヤの割合
   U(0.10, 0.20)）。大きさ = v24 の車の曲線 + U(4.8, 8.6) dB（v24.apply_levels が src["heavy"]["offset_db"] を足す）。ラベルは car_drive（番号 4）のまま。
2. ベルつき走行のベルの時刻 v19bellbike（D26）: 「決めた回数が全部窓に収まる長さ d」を型と回数から先に出し、上限 U = 9.9 − d。前の区間 [max(0.2, CPA−4), min(U, CPA−0.5)]・
   後の区間 (max(0.2, CPA+0.1), min(U, CPA+4)]。後の区間が空でない場面の中から固定 seed で 11 場面（22/500 = 4.4%・土台のベルの本と同じ割合）を「後」に、
   残りは「前」で一様に引く（独立の乱数・自転車の軌道・速さ・大きさは変えない）。区間が空なら黙って反対側へ移さず止める。
   回数の単位: ピン型・ディンディン型 = 打の回数（repeat_period ごと）・チリリン型 = バーストの回数。全部の開始が窓の中・次の周期が窓に入らないことを検査。

使い方:
  python scripts/step11_v25_render.py selftest
  python scripts/step11_v25_render.py manifest-bellbike        # plan/manifest_v26_bellbike.csv（500 本の側・t_on・t_off・型・回数・d）
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

import step11_v24_render as v24  # noqa: E402
from outdoor_seld import engine_heavy_v25 as H25  # noqa: E402

v23, v22, m9, v13, v19r = v24.v23, v24.v22, v24.m9, v24.v13, v24.v19r
PLAN_DIR = ROOT / "out/dataset_outdoor_siren_v2final/plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
MANIFEST_HEAVY = PLAN_DIR / "manifest_v25_heavy.csv"
MANIFEST_BELLBIKE = PLAN_DIR / "manifest_v26_bellbike.csv"
SCEN_HEAVY = "v25heavy"
SALT25, SALT26, SALT26_SELECT = 20261014, 20261015, 20261016
BELL_WINDOW_END = 9.9
BELL_TAIL_S = {"pin": 0.5, "ding": 0.5, "ring": 0.1}      # 最後の打のあとの余韻の余裕（設計値）
BELL_N_AFTER = 11                                        # 250 場面のうち「後」にする場面（22/500 = 4.4%）
_TWIN_SCEN: dict = {}
_BELL_ASSIGN: dict = {}
HEAVY_ENABLED = {"on": True}       # 試聴で「同じ場面を乗用車のまま」描くときだけ False


def _irng(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


# ---------------------------------------------------------------- 1. 大型車
def set_twin_scenarios(mapping: dict) -> None:
    """plan を書く前の検査用: clip_id → 元の場面の種類。"""
    _TWIN_SCEN.clear(); _TWIN_SCEN.update(mapping)


def twin_scenario(row: dict) -> str:
    if not _TWIN_SCEN:
        if MANIFEST_HEAVY.is_file():
            with open(MANIFEST_HEAVY, newline="", encoding="utf-8") as f:
                for m in csv.DictReader(f):
                    _TWIN_SCEN[m["clip_id"]] = m["src_scenario"]
        if row["clip_id"] not in _TWIN_SCEN and PLAN.is_file():            # manifest が無いときは plan の twin_of から
            with open(PLAN, newline="", encoding="utf-8") as f:
                by_id = {r["clip_id"]: r["scenario"] for r in csv.DictReader(f)}
            _TWIN_SCEN[row["clip_id"]] = by_id[row["twin_of"]]
    return _TWIN_SCEN[row["clip_id"]]


def sample_v25heavy(row: dict) -> dict:
    scen = twin_scenario(row)
    twin = dict(row); twin["scenario"] = scen
    fn = v19r._SAMPLERS19.get(scen)
    s = fn(twin) if fn is not None else v19r.v16r.sample_scene_v15(twin)
    cars = [x for x in s["sources"] if x["class"] == "car_drive" and x.get("kind") == "vehicle"]
    assert cars, f"{row['clip_id']}: 元の場面 {scen} に車が無い"
    r = _irng(row["seed"], SALT25)
    hv = cars[int(r.integers(0, len(cars)))]
    hp = H25.heavy_params(r)
    s["row"] = dict(row)
    if not HEAVY_ENABLED["on"]:                     # 試聴用: 同じ場面・同じ車を乗用車のまま
        s["v25"] = {"kind": "heavy_off", "twin_of": row.get("twin_of", ""), "src_scenario": scen, "heavy_track": int(hv.get("track", 0) or 0), "n_car": len(cars)}
        return s
    hv["heavy"] = dict(hp, track=int(hv.get("track", 0) or 0))
    hv["params"] = dict(hv["params"], heavy=True, heavy_f0=hp["f0_hz"], heavy_tire_frac_a=hp["tire_frac_a"], heavy_seed=int(row["seed"]) * 7 + 31)
    s["v25"] = {"kind": "heavy", "twin_of": row.get("twin_of", ""), "src_scenario": scen, "heavy_track": hv["heavy"]["track"], "n_car": len(cars),
                "heavy_is_track0": hv["heavy"]["track"] == 0, **{k: round(v, 4) for k, v in hp.items()}}
    return s


def _make_dry_v25(prev):
    def fn(src: dict) -> np.ndarray:
        p = src.get("params", {})
        if src["class"] == "car_drive" and p.get("heavy"):
            x, info = H25.make_heavy_car(m9.CLIP, m9.FS_SIM, np.random.default_rng(int(p["heavy_seed"])), f0=float(p["heavy_f0"]), tire_frac_a=float(p["heavy_tire_frac_a"]))
            src["car_variant"] = "heavy_v25"
            src.setdefault("heavy", {})["dry_info"] = info
            return x
        return prev(src)
    fn._v25 = True
    return fn


# ---------------------------------------------------------------- 2. ベルつき走行のベルの時刻（D26）
def _bell_kind(p: dict) -> str:
    bt = p.get("bell_type", "ding")
    return "pin" if bt == "pin" else ("ring" if bt == "ring" else "ding")


def bell_needed_duration(bell: dict) -> float:
    """決めた回数が全部窓に収まる長さ d（最後の打の開始 + 余韻の余裕）。"""
    p = bell["params"]; n = int(bell["n_ring"]); kind = _bell_kind(p)
    if kind == "ring":
        return n * float(p["burst_sec"]) + (n - 1) * float(p["gap_sec"]) + BELL_TAIL_S["ring"]
    return (n - 1) * float(p["repeat_period_sec"]) + BELL_TAIL_S[kind]


def bell_intervals(cpa: float, d: float) -> tuple[tuple[float, float], tuple[float, float]]:
    u = BELL_WINDOW_END - d
    before = (max(0.2, cpa - v22.BELL_START_WINDOW_S), min(u, cpa - 0.5))
    after = (max(0.2, cpa + 0.1), min(u, cpa + v22.BELL_START_WINDOW_S))
    return before, after


def bell_strike_starts(bell: dict) -> list:
    """予定の打撃（バースト）の開始時刻の列（窓の頭 t_on から）。"""
    p = bell["params"]; n = int(bell["n_ring"]); kind = _bell_kind(p); t0 = float(bell["t_on"])
    if kind == "ring":
        step = float(p["burst_sec"]) + float(p["gap_sec"])
    else:
        step = float(p["repeat_period_sec"])
    return [t0 + k * step for k in range(n)], t0 + n * step


def _bellbike_rows(plan: Path = PLAN) -> list:
    rows = []
    with open(plan, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == v22.SCEN_BELLBIKE and r["h_copy"] == "1":
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); rows.append(r)
    return rows


def _assignment_from_rows(prev, rows: list, n_after: int) -> dict:
    """行の一覧（h_copy=1 の場面）から seed → "after"／"before"。後の区間が空でない場面から固定 seed（SALT26_SELECT）で n_after 場面を後に。"""
    eligible, all_seeds = [], []
    for r in rows:
        s = prev(r)
        bell = next(x for x in s["sources"] if x["class"] == "bike_bell")
        cpa = float(s["sources"][0]["t_cpa_s"])
        _, after = bell_intervals(cpa, bell_needed_duration(bell))
        all_seeds.append(int(r["seed"]))
        if after[0] < after[1]:
            eligible.append(int(r["seed"]))
    assert len(eligible) >= n_after, f"後に置ける場面が {len(eligible)} < {n_after}"
    chosen = set(int(x) for x in np.random.default_rng(SALT26_SELECT).choice(sorted(eligible), n_after, replace=False)) if n_after > 0 else set()
    out = {sd: ("after" if sd in chosen else "before") for sd in all_seeds}
    out["_n_eligible"] = len(eligible); out["_n_after"] = n_after
    return out


def _bellbike_assignment(prev) -> dict:
    """学習 plan の割り当て（変えない）: seed → "after"／"before"。後の区間が空でない場面から固定 seed で 11 場面を後に。"""
    if _BELL_ASSIGN:
        return _BELL_ASSIGN
    _BELL_ASSIGN.update(_assignment_from_rows(prev, _bellbike_rows(), BELL_N_AFTER))
    return _BELL_ASSIGN


# 学習 plan 以外の plan（確定評価 eval v3・2026-10-14 凍結 r11）のベルつき走行: 同じ規則（後 = 11/250 = 4.4%・同じ固定 seed）を、その plan の場面の数に比例した場面数で掛ける
_BELL_ASSIGN_EXTRA: dict = {}
_BELL_ROWS_OVERRIDE: dict = {"tag": None, "rows": None}
BELL_AFTER_FRAC = BELL_N_AFTER / 250.0


def set_bellbike_rows_override(rows: list | None, tag: str | None = None) -> None:
    """plan を書く前の検査用: その plan のベルつき走行の行（h_copy=1 の場面・seed は int）を直接渡す。None で解除。"""
    _BELL_ROWS_OVERRIDE["rows"] = None if rows is None else [dict(r, seed=int(r["seed"]), n_warnings=int(r.get("n_warnings") or 0)) for r in rows if r.get("h_copy", "1") == "1"]
    _BELL_ROWS_OVERRIDE["tag"] = tag
    _BELL_ASSIGN_EXTRA.clear()


def _bellbike_assignment_extra(prev) -> dict:
    """学習 plan に無い seed の割り当て: 渡された行の一覧 → それも無ければ いまの出力先の plan（v19r.PLAN_V19/assignment_v2final.csv）。"""
    if _BELL_ROWS_OVERRIDE["rows"] is not None:
        key = f"override:{_BELL_ROWS_OVERRIDE['tag']}"; rows = _BELL_ROWS_OVERRIDE["rows"]
    else:
        plan = Path(v19r.PLAN_V19) / "assignment_v2final.csv"
        if not plan.is_file() or plan.resolve() == PLAN.resolve():
            return {}
        key = str(plan.resolve())
        rows = [r for r in _bellbike_rows(plan)]
    if key not in _BELL_ASSIGN_EXTRA:
        n_after = int(round(BELL_AFTER_FRAC * len(rows)))
        _BELL_ASSIGN_EXTRA[key] = _assignment_from_rows(prev, rows, n_after) if rows else {}
    return _BELL_ASSIGN_EXTRA[key]


def _bellbike_v25(prev):
    def fn(row: dict) -> dict:
        s = prev(row)
        assign = _bellbike_assignment(prev)
        seed = int(row["seed"]); side = assign.get(seed)
        if side is None:                                            # 学習 plan に無い seed（確定評価 eval v3 など）→ その plan の割り当て（同じ規則・同じ割合）
            side = _bellbike_assignment_extra(prev).get(seed)
        assert side in ("after", "before"), f"{row['clip_id']}: seed {seed} が plan のベルつき走行に無い"
        bell = next(x for x in s["sources"] if x["class"] == "bike_bell")
        cpa = float(s["sources"][0]["t_cpa_s"]); d = bell_needed_duration(bell)
        before, after = bell_intervals(cpa, d)
        lo, hi = after if side == "after" else before
        assert lo < hi, f"{row['clip_id']}: {side} の区間が空（CPA {cpa:.2f}・d {d:.2f}）"
        t0 = float(_irng(seed, SALT26).uniform(lo, hi))
        bell["t_on"], bell["t_off"] = round(t0, 3), round(min(t0 + d, BELL_WINDOW_END), 3)
        info = {"side": side, "needed_s": round(d, 3), "t_on": bell["t_on"], "t_off": bell["t_off"], "n_ring": int(bell["n_ring"]), "type": _bell_kind(bell["params"])}
        s.setdefault("v25", {})["bell_timing"] = info
        if "v19" in s:
            s["v19"]["bell_t_on"], s["v19"]["bell_t_off"] = bell["t_on"], bell["t_off"]
        return s
    fn._v25 = True
    return fn


# ---------------------------------------------------------------- 登録
if SCEN_HEAVY not in v19r._SAMPLERS19:
    v19r._SAMPLERS19[SCEN_HEAVY] = sample_v25heavy
if not getattr(m9._make_dry, "_v25", False):
    m9._make_dry = _make_dry_v25(m9._make_dry)
if not getattr(v22.sample_v22bellbike, "_v25", False):
    v22.sample_v22bellbike = _bellbike_v25(v22.sample_v22bellbike)
    v19r._SAMPLERS19[v22.SCEN_BELLBIKE] = v22.sample_v22bellbike
sample_scene_v19 = v19r.sample_scene_v19
generate_clip = v19r.generate_clip


# ---------------------------------------------------------------- 検査
def check_heavy_scene(s: dict, row: dict) -> None:
    cars = [x for x in s["sources"] if x["class"] == "car_drive" and x.get("kind") == "vehicle"]
    hv = [x for x in cars if x.get("params", {}).get("heavy")]
    assert len(hv) == 1, f"大型車が {len(hv)} 台（1 台でない）"
    assert len(cars) == int(row["n_car"]), f"車の台数 {len(cars)} ≠ plan の n_car {row['n_car']}"
    h = hv[0]["heavy"]
    assert H25.HEAVY_F0_HZ[0] <= h["f0_hz"] <= H25.HEAVY_F0_HZ[1] and H25.HEAVY_TIRE_FRAC_A[0] <= h["tire_frac_a"] <= H25.HEAVY_TIRE_FRAC_A[1]
    assert H25.HEAVY_LEVEL_OFFSET_DB[0] <= h["offset_db"] <= H25.HEAVY_LEVEL_OFFSET_DB[1]
    info = hv[0].get("v24", {})
    assert info.get("rule") == "car_curve" and abs(info.get("heavy_offset_db", 0.0) - h["offset_db"]) < 0.006, "v24 の大きさに大型車の分が足されていない"   # v24 は 2 桁に丸めて記録
    for x in cars:
        if x is not hv[0]:
            assert not x.get("params", {}).get("heavy") and x.get("v24", {}).get("heavy_offset_db", 0.0) == 0.0


def check_heavy_dry(src: dict) -> dict:
    x = m9._make_dry(src)
    info = src["heavy"]["dry_info"]
    assert abs(info["low_band_gain_db"] - H25.HEAVY_LOW_BOOST_DB) < 0.3, f"低域の増分 {info['low_band_gain_db']}"
    assert abs(info["tire_frac_a_measured"] - src["heavy"]["tire_frac_a"]) < 0.005, "タイヤの割合"
    assert abs(float(np.max(np.abs(x))) - 0.9) < 1e-6
    return info


def check_bellbike_timing(s: dict) -> dict:
    bell = next(x for x in s["sources"] if x["class"] == "bike_bell")
    cpa = float(s["sources"][0]["t_cpa_s"]); info = s["v25"]["bell_timing"]
    starts, next_start = bell_strike_starts(bell)
    assert all(bell["t_on"] - 1e-9 <= t < bell["t_off"] for t in starts), f"打撃の開始が窓の外 {starts} 窓 {bell['t_on']}〜{bell['t_off']}"
    assert next_start >= bell["t_off"] - 1e-9, f"次の周期 {next_start:.3f} が窓 {bell['t_off']} に入る"
    assert bell["t_off"] <= BELL_WINDOW_END + 1e-9
    if info["side"] == "after":
        assert bell["t_on"] > cpa + 0.1 - 1e-9 and bell["t_on"] <= cpa + v22.BELL_START_WINDOW_S + 1e-9
    else:
        assert bell["t_on"] <= cpa - 0.5 + 1e-9 and bell["t_on"] >= max(0.2, cpa - v22.BELL_START_WINDOW_S) - 1e-9
    v22.check_scene(v22.SCEN_BELLBIKE, s)        # v23 の検査（xy と時刻の軌道・z・打音・±4 s）
    return info


def write_bellbike_manifest(path: Path = MANIFEST_BELLBIKE, plan: Path = PLAN) -> dict:
    """plan のベルつき走行の全行の時刻の manifest。学習 plan 以外（eval v3）は、その plan の行を割り当てに渡してから引く。"""
    if plan.resolve() != PLAN.resolve():
        with open(plan, newline="", encoding="utf-8") as f:
            set_bellbike_rows_override([r for r in csv.DictReader(f) if r["scenario"] == v22.SCEN_BELLBIKE], plan.name)
    rows = []
    with open(plan, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == v22.SCEN_BELLBIKE:
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); rows.append(r)
    out, cnt = [], {"after": 0, "before": 0}
    for r in rows:
        s = sample_scene_v19(r); info = check_bellbike_timing(s); cnt[info["side"]] += 1
        out.append({"clip_id": r["clip_id"], "seed": r["seed"], "h_copy": r["h_copy"], "split": r["split"], **info, "t_cpa": round(float(s["sources"][0]["t_cpa_s"]), 3)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return {"n": len(out), **cnt, "path": str(path)}


def _plan_rows(pred, n: int) -> list:
    out = []
    with open(PLAN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if pred(r):
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); out.append(r)
                if len(out) >= n:
                    break
    return out


_V24_DUMP = r"""
import json, sys
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import step11_v24_render as v24
rows = json.loads(sys.stdin.read()); out = []
for r in rows:
    s = v24.sample_scene_v19(r)
    out.append({"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"], "t_off": x["t_off"], "l1m": x.get("l1m_db")} for x in s["sources"]]})
print(json.dumps(out))
"""


def selftest() -> int:
    ok = []
    import step10_v25_plan as p25
    d25 = (json.loads(p25.PARAMS.read_text(encoding="utf-8")).get("D25_HEAVY") or {})
    if not d25.get("adopt"):
        # D25 不採用（本人 2026-10-12 18:17）: plan に v25heavy は無い。大型車の枝はコードに残すが検査は飛ばす
        n_heavy = len(_plan_rows(lambda r: r["scenario"] == SCEN_HEAVY, 10)); n_all = sum(1 for _ in csv.DictReader(open(PLAN, newline="", encoding="utf-8")))
        P_all = json.loads(p25.PARAMS.read_text(encoding="utf-8")); n_exp = int(P_all.get("N_ROWS_TOTAL_EXPECTED_V29") or P_all.get("N_ROWS_TOTAL_EXPECTED_V28") or P_all.get("N_ROWS_TOTAL_EXPECTED_V27") or p25.N_EXISTING_V22)   # 2026-10-13 v27（#64）・10/14 v28（積み増し）で総数が増えた
        ok.append((f"D25 不採用（本人 10/12 18:17）: plan に v25heavy が無い（{n_heavy} 本）・総行数 {n_all:,}（= params の総数 {n_exp:,}）", n_heavy == 0 and n_all == n_exp))
    else:
        base = p25.read_rows(p25.PLAN_PRE25 if p25.PLAN_PRE25.exists() else p25.PLAN)
        base = base[:p25.N_EXISTING_V22]
        scenes = p25.car_scenes(base); chosen, stat = p25.select_scenes(scenes, p25.total_car_sources(base))
        new, manifest = p25.build(chosen)
        set_twin_scenarios({m["clip_id"]: m["src_scenario"] for m in manifest})
        ok.append((f"plan: 車のある場面 {len(scenes):,} から {stat['n_scenes']:,} 場面（{len(new):,} 本）を選ぶと大型車の割合 {stat['fraction_final']:.4f}（1 つ前 {stat['fraction_before_last']:.4f}）", stat["fraction_final"] >= 0.15 > stat["fraction_before_last"]))
        kinds = {}
        for m in manifest:
            kinds[m["src_scenario"]] = kinds.get(m["src_scenario"], 0) + 1
        picks = []
        for scen in ("v11core", "v19fast", "v19turn", "v19platoon", "v19launch"):
            picks += [r for r in new if dict(zip([x["clip_id"] for x in manifest], [x["src_scenario"] for x in manifest]))[r["clip_id"]] == scen][:2]
        n_ok, tr0 = 0, 0
        for r in picks:
            rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
            s = sample_scene_v19(rr); check_heavy_scene(s, rr); n_ok += 1; tr0 += int(s["v25"]["heavy_is_track0"])
            hv = next(x for x in s["sources"] if x.get("params", {}).get("heavy")); check_heavy_dry(hv)
        ok.append((f"heavy: 元の場面の種類ごと 2 本 × 5 = {n_ok} 本で、大型車 1 台・台数が plan と同じ・v24 の大きさに +4.8〜8.6・ドライ音（低域 +10 dB・タイヤの割合）（track0 が大型 {tr0}/{n_ok}・元の種類の内訳 {kinds}）", n_ok == 10))
    # D26
    bb = _plan_rows(lambda r: r["scenario"] == v22.SCEN_BELLBIKE, 500)
    sides = {"after": 0, "before": 0}; bad = 0
    for r in bb:
        try:
            info = check_bellbike_timing(sample_scene_v19(r)); sides[info["side"]] += 1
        except AssertionError as e:
            bad += 1
            if bad <= 3:
                print("NG", r["clip_id"], e)
    ok.append((f"bellbike D26: 500 本で打撃が全部窓の中・次の周期は外・後 {sides['after']}（=22）・前 {sides['before']}・後に置ける場面 {_BELL_ASSIGN.get('_n_eligible')}", bad == 0 and sides["after"] == 22 and sides["before"] == 478))
    # 乱数: v24 だけの別プロセスと、ベルつき走行以外の本が一致・ベルつき走行は窓だけ違う
    sample = _plan_rows(lambda r: r["scenario"] == "v11core" and r["n_car"] == "2", 2) + _plan_rows(lambda r: r["scenario"] == "v19platoon", 1) + bb[:2]
    res = subprocess.run([sys.executable, "-c", _V24_DUMP % (str(ROOT / "scripts"), str(ROOT / "src"))], input=json.dumps(sample), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        ref = json.loads(res.stdout.strip().splitlines()[-1]); diffs = []
        for a, r in zip(ref, sample):
            s = sample_scene_v19(r)
            mine = {"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"], "t_off": x["t_off"], "l1m": x.get("l1m_db")} for x in s["sources"]]}
            if r["scenario"] == v22.SCEN_BELLBIKE:
                for d in (a, mine):
                    for x in d["sources"]:
                        if x["class"] == "bike_bell":
                            x["t_on"] = x["t_off"] = None          # 意図して変える列
            if json.dumps(a, sort_keys=True) != json.dumps(mine, sort_keys=True):
                diffs.append(r["clip_id"])
        ok.append((f"場面の乱数: v24 だけの別プロセスと {len(sample)} 本が一致（ベルつき走行はベルの窓だけ違う）{'' if not diffs else diffs}", not diffs))
    else:
        ok.append(("場面の乱数: 別プロセスの v24 が動く", False)); print(res.stderr[-600:])
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# ---------------------------------------------------------------- 試聴（大型車: ドライ音の比較と、同じ場面を乗用車／大型車で描いた W チャンネル）
def audition(out_dir: Path) -> int:
    import shutil
    import soundfile as sf
    from outdoor_seld.engine import make_car_v9
    from outdoor_seld.calibration import K_RMS_SPL
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = m9.FS_SIM
    lines = ["大型車（v25・変更票 #52 D25）の試聴（2026-10-12）。wav は git に入れない（README だけ）。", ""]

    def save_dry(name, x, note):
        y = x / (np.max(np.abs(x)) + 1e-12) * 10 ** (-3 / 20)
        sf.write(str(out_dir / name), (y * 32767).astype(np.int16), fs, subtype="PCM_16"); lines.append(f"{name}: {note}"); print("wrote", name)
    # ドライ音（伝搬なし・同じ大きさ＝ピーク −3 dBFS・音色の比較）
    save_dry("dry_car_f0_42.wav", make_car_v9(5.0, fs, np.random.default_rng(1), f0=42.0), "乗用車（今の作り・4 気筒 42 Hz・タイヤ 7 割）")
    for f0, tf in ((55.0, 0.20), (65.0, 0.15), (80.0, 0.10)):
        x, info = H25.make_heavy_car(5.0, fs, np.random.default_rng(1), f0=f0, tire_frac_a=tf)
        save_dry(f"dry_truck_f0_{int(f0)}_tire{int(tf * 100)}.wav", x, f"大型車（発火 {f0:.0f} Hz・250 Hz 以下 +{info['low_band_gain_db']:.1f} dB・A 特性のタイヤの割合 {tf}）")
    # 同じ場面を乗用車／大型車で（v24 の大きさ込み・W チャンネル・同じ再生ゲイン 90 dB(A) = 0 dBFS）
    tmp = out_dir / "_render"; shutil.rmtree(tmp, ignore_errors=True); (tmp / "work").mkdir(parents=True)
    keep = (m9.DS, m9.WORK, v19r.DS_V19); m9.DS, m9.WORK, v19r.DS_V19 = tmp, tmp / "work", tmp
    REF_DB = 90.0; gain = 10.0 ** ((0.0 - (REF_DB - K_RMS_SPL)) / 20.0)
    picks = []
    with open(MANIFEST_HEAVY, newline="", encoding="utf-8") as f:
        ms = list(csv.DictReader(f))
    for scen in ("v11core", "v19fast", "v19launch"):
        m = next((x for x in ms if x["src_scenario"] == scen and x["h_copy"] == "1"), None)
        if m:
            picks.append((scen, m["clip_id"]))
    rows = {r["clip_id"]: r for r in _plan_rows(lambda r: r["clip_id"] in {c for _, c in picks}, len(picks))}
    try:
        for scen, cid in picks:
            row = rows[cid]
            for tag, on in (("car", False), ("truck", True)):
                HEAVY_ENABLED["on"] = on
                shutil.rmtree(tmp / "foa", ignore_errors=True); shutil.rmtree(tmp / "work" / cid, ignore_errors=True)
                generate_clip(row)
                fpath = next((tmp / "foa").glob(cid + ".*"))
                x, sr = sf.read(str(fpath), dtype="float64", always_2d=True)
                w = x[:, 0] * gain; pk = float(np.max(np.abs(w)))
                if pk > 0.99:
                    w = w * 0.99 / pk
                sf.write(str(out_dir / f"scene_{scen}_{tag}.wav"), (w * 32767).astype(np.int16), sr, subtype="PCM_16")
                sj = json.loads((tmp / "work" / cid / "scene.json").read_text(encoding="utf-8"))
                cars = [z for z in sj["sources"] if z["class"] == "car_drive"]
                desc = "; ".join(f"{'大型' if z.get('params', {}).get('heavy') else '乗用'} v={float(z.get('speed_mps', 0)) * 3.6:.0f} km/h 1m={float(z['l1m_db']):.1f} 2m={float(z['law_db']):.1f}" for z in cars[:3])
                lines.append(f"scene_{scen}_{tag}.wav: {cid}（元の場面の種類 {scen}・暗騒音 {sj['noise']['dba']:.0f} dB(A)）{'大型車あり' if on else '全部乗用車'}: {desc}" + (f"（ピーク {20 * np.log10(0.99 / pk):+.1f} dB 下げ）" if pk > 0.99 else ""))
                print("wrote", f"scene_{scen}_{tag}.wav")
    finally:
        HEAVY_ENABLED["on"] = True
        m9.DS, m9.WORK, v19r.DS_V19 = keep
        shutil.rmtree(tmp, ignore_errors=True)
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "manifest-bellbike":
        a = sys.argv[2:]
        plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else PLAN
        out = Path(a[a.index("--out") + 1]) if "--out" in a else (MANIFEST_BELLBIKE if plan.resolve() == PLAN.resolve() else plan.parent / "manifest_v26_bellbike.csv")
        print(write_bellbike_manifest(out, plan)); sys.exit(0)
    if cmd == "audition":
        sys.exit(audition(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "audition_2026-10-12_truck"))
    print(__doc__); sys.exit(2)
