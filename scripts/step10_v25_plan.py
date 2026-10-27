# -*- coding: utf-8 -*-
"""v25 plan = v22 の plan（31,274 行）の末尾に、変更票 #52（D25 大型車）の行を足す（2026-10-12・設計の案 r2 どおり）。

足す行（params md/design/v2final_params.json の D25_HEAVY.adopt が true のときだけ）:
  v25heavy = 今の car_drive を含む本（v11core・v19fast・v19turn・v19platoon・v19launch で n_car ≥ 1）の「高さの組（場面）」を**一様に・重複なし・固定 seed の順**に選び、
  割当の条件（split・motion・警告音の列・層・左右・scene_type・n_car・雨・mic_z・h_copy・D10 の列）を写した新しい行を作る。変えるのは clip_id・seed（新しい）・
  scenario（v25heavy）・grammar（v19）・twin_of（元の clip_id）。場面は新しい seed で引き直す（双子ではない・位置・速さ・時刻は元と同じとは主張しない）。
  大型車にするのは、新しい場面の車の中から固定 seed で一様に 1 台（描画側 step11_v25_render）。
  本数: 選んだ順に足していき、**H ÷ (全部の車の音源の台数 + 選んだ元の車の台数の合計) が初めて 15% 以上になったところで止める**（H = 大型車の台数 = 新しい本の数）。
clip 番号（既存の最大は 41250・重ならない）: h1 42001+i / h2 46001+i。seed: 9199000000+i。既存 31,274 行は 1 バイトも変えない（前の plan を assignment_v2final_pre_v25.csv に残す）。
manifest（plan の既存列は増やさない・#52）: plan/manifest_v25_heavy.csv = clip_id・twin_of・src_scenario・seed・h_copy・n_car・split（描画側が元の場面の種類を知るのに使う）。
params に書き戻す: D25_HEAVY.n_scenes／n_clips／heavy_fraction_final と N_ROWS_TOTAL_EXPECTED_V25（書くときだけ・dry-run では書かない）。

使い方: PYTHONPATH=scripts:src python scripts/step10_v25_plan.py [--dry-run]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final" / "plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
PLAN_PRE25 = PLAN_DIR / "assignment_v2final_pre_v25.csv"
MANIFEST = PLAN_DIR / "manifest_v25_heavy.csv"
PARAMS = ROOT / "md" / "design" / "v2final_params.json"
README = PLAN_DIR / "README_plan_v2final.md"
COLS_BASE = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side", "w2_class", "w2_side", "danger_tier",
             "car_side", "scenario", "seed", "scene_type", "n_car", "rain", "rain_rate", "rain_dba", "grammar",
             "close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db", "mic_z", "h_copy"]
COLS = COLS_BASE + ["twin_of"]
MANIFEST_COLS = ["clip_id", "twin_of", "src_scenario", "seed", "h_copy", "n_car", "split"]
SCEN = "v25heavy"
ID_H1, ID_H2, SEED0 = 42001, 46001, 9199000000
N_EXISTING_V22, MAX_EXISTING_ID = 31274, 41250
CAR_SCENARIOS = ("v11core", "v19fast", "v19turn", "v19platoon", "v19launch")
SALT_SELECT = 20261064          # 場面を選ぶ順の固定 seed（#52）
FRACTION = 0.15


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list, cols: list) -> None:
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def car_scenes(base: list) -> list:
    """car_drive を含む本の高さの組（同じ seed の 2 本）を clip_id の順に。"""
    by_seed = defaultdict(list)
    for r in base:
        if r["scenario"] in CAR_SCENARIOS and int(r["n_car"] or 0) >= 1:
            by_seed[r["seed"]].append(r)
    scenes = sorted(by_seed.values(), key=lambda v: min(r["clip_id"] for r in v))
    assert all(len(v) == 2 and {x["h_copy"] for x in v} == {"1", "2"} for v in scenes), "車の場面が高さ 2 本になっていない"
    return scenes


def total_car_sources(rows: list) -> int:
    return sum(int(r["n_car"] or 0) for r in rows)


def select_scenes(scenes: list, total_cars: int) -> tuple[list, dict]:
    order = np.random.default_rng(SALT_SELECT).permutation(len(scenes))
    chosen, H, S = [], 0, 0
    for idx in order:
        pair = scenes[int(idx)]
        chosen.append(pair); H += 2; S += 2 * int(pair[0]["n_car"])
        if H / (total_cars + S) >= FRACTION:
            break
    frac = H / (total_cars + S)
    prev = (H - 2) / (total_cars + S - 2 * int(chosen[-1][0]["n_car"])) if len(chosen) > 1 else 0.0
    return chosen, {"n_scenes": len(chosen), "H": H, "added_cars": S, "total_cars_before": total_cars, "fraction_final": frac, "fraction_before_last": prev}


def build(chosen: list) -> tuple[list, list]:
    rows, manifest = [], []
    for i, pair in enumerate(chosen):
        for r in sorted(pair, key=lambda x: int(x["h_copy"])):
            t = {c: r.get(c, "") for c in COLS_BASE}
            h = int(r["h_copy"])
            t["clip_id"] = f"{r['split']}_syn_clip{(ID_H1 if h == 1 else ID_H2) + i:05d}"
            t["seed"] = str(SEED0 + i)
            t["scenario"] = SCEN; t["grammar"] = "v19"; t["twin_of"] = r["clip_id"]
            rows.append(t)
            manifest.append({"clip_id": t["clip_id"], "twin_of": r["clip_id"], "src_scenario": r["scenario"], "seed": str(SEED0 + i),
                             "h_copy": str(h), "n_car": r["n_car"], "split": r["split"]})
    return rows, manifest


def check(base: list, new: list, manifest: list, stat: dict) -> list:
    allrows = base + new
    ids = [r["clip_id"] for r in allrows]
    base_by_id = {r["clip_id"]: r for r in base}
    msgs = [("clip_id が全部ユニーク", len(ids) == len(set(ids))),
            (f"既存 {N_EXISTING_V22:,} 行が先頭にそのまま", len(base) == N_EXISTING_V22),
            ("新しい番号が既存の最大より大きく 50000 未満", all(MAX_EXISTING_ID < int(r["clip_id"][-5:]) < 50000 for r in new)),
            ("新しい seed が 9199000000 から連番で既存と重ならない", sorted(int(r["seed"]) for r in new) == sorted([SEED0 + i for i in range(len(new) // 2)] * 2) and not ({r["seed"] for r in new} & {r["seed"] for r in base})),
            ("1 場面 = 高さ 2 本（同じ seed・h_copy 1 と 2）", all(len(v) == 2 and {x["h_copy"] for x in v} == {"1", "2"} for v in _group(new).values())),
            (f"大型車の割合 = 止めた時点で初めて 15% 以上（{stat['fraction_final']:.4f}・1 つ前 {stat['fraction_before_last']:.4f}）", stat["fraction_final"] >= FRACTION > stat["fraction_before_last"]),
            ("manifest の行数・clip_id が新しい行と一致", [m["clip_id"] for m in manifest] == [r["clip_id"] for r in new])]
    bad = 0
    for r in new:
        tw = base_by_id.get(r["twin_of"])
        if tw is None or tw["scenario"] not in CAR_SCENARIOS or int(tw["n_car"] or 0) < 1:
            bad += 1; continue
        for c in COLS_BASE:
            if c in ("clip_id", "seed", "scenario", "grammar"):
                continue
            if r.get(c, "") != tw.get(c, ""):
                bad += 1; break
    msgs.append(("割当の列（split・motion・警告音・層・左右・scene_type・n_car・雨・D10・mic_z・h_copy）が元の行の写し・元は車のある本", bad == 0))
    msgs.append(("元の場面は重複なし", len({r["twin_of"] for r in new}) == len(new)))
    return msgs


def _group(rows: list) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r["seed"]].append(r)
    return g


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    d25 = P.get("D25_HEAVY") or {}
    if not d25.get("adopt"):
        print("D25_HEAVY.adopt が true でない → 何もしない"); return 0
    base = read_rows(PLAN_PRE25 if PLAN_PRE25.exists() else PLAN)
    assert len(base) == N_EXISTING_V22, f"元の plan が {len(base)} 行（31,274 でない）"
    scenes = car_scenes(base)
    total = total_car_sources(base)
    assert total == 25550, f"車の音源の台数の合計 {total}（#52 の 25,550 と違う）"
    chosen, stat = select_scenes(scenes, total)
    new, manifest = build(chosen)
    msgs = check(base, new, manifest, stat)
    for name, ok in msgs:
        print(("OK " if ok else "NG ") + name)
    print(f"場面 {stat['n_scenes']:,}（車のある場面 {len(scenes):,} の中から）→ 新しい本 {len(new):,}（大型車 {stat['H']:,} 台）・足した乗用車 {stat['added_cars']:,} 台・割合 {stat['fraction_final']:.4f}")
    c = Counter(r["split"] for r in new); cs = Counter(m["src_scenario"] for m in manifest)
    print(f"  split: train {c['train']:,} / val {c['val']:,}・元の場面の種類: {dict(cs)}・歩行 {sum(r['motion'] == 'walk' for r in new) / len(new):.1%}・雨 {sum(bool(r['rain']) for r in new) / len(new):.1%}")
    # 場面の検査（描画なし）: dry-run は先頭 200 本・書くときは全部
    import step11_v25_render as v25r
    sel = new if not dry else new[:200]
    bad = 0
    tmp_manifest = {m["clip_id"]: m["src_scenario"] for m in manifest}
    v25r.set_twin_scenarios(tmp_manifest)
    for r in sel:
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            v25r.check_heavy_scene(v25r.sample_scene_v19(rr), rr)
        except AssertionError as e:
            bad += 1
            if bad <= 5:
                print("NG 場面", r["clip_id"], e)
    print(f"場面の検査（大型車 1 台・車の台数が元と同じ・v24 の大きさに +4.8〜8.6）: {len(sel)} 本中 NG {bad}")
    if not all(ok for _, ok in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    if not PLAN_PRE25.exists():
        PLAN_PRE25.write_bytes(PLAN.read_bytes())
    write_rows(PLAN, base + new, COLS)
    write_rows(MANIFEST, manifest, MANIFEST_COLS)
    d25["n_scenes"], d25["n_clips"] = stat["n_scenes"], len(new)
    d25["heavy_fraction_final"] = round(stat["fraction_final"], 5); d25["added_cars"] = stat["added_cars"]
    P["D25_HEAVY"] = d25
    P["N_ROWS_TOTAL_EXPECTED_V25"] = len(base) + len(new)
    P["_note_v25"] = f"2026-10-12: v25 の追加分（大型車 {len(new):,}）を足した総数 = N_ROWS_TOTAL_EXPECTED_V25（step10_v25_plan が書く）。V22 の 31,274 は据え置き"
    PARAMS.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = [f"\n## v25 追加（2026-10-12・変更票 #52 D25 大型車）: {len(new):,} 行を末尾に足した → 総行数 {len(base) + len(new):,}",
             f"- 元の plan（31,274 行）= assignment_v2final_pre_v25.csv sha256 {sha256(PLAN_PRE25)}", f"- 新しい plan sha256 {sha256(PLAN)}",
             f"- 場面 {stat['n_scenes']:,}（車のある場面 {len(scenes):,} から固定 seed {SALT_SELECT} の順に・大型車 {stat['H']:,} 台 ÷ (25,550 + {stat['added_cars']:,}) = {stat['fraction_final']:.4f}）・train {c['train']:,} / val {c['val']:,}・元の場面の種類 {dict(cs)}",
             f"- manifest = manifest_v25_heavy.csv（clip_id・twin_of・src_scenario・seed・h_copy・n_car・split）sha256 {sha256(MANIFEST)}",
             "- 場面の検査（step11_v25_render.check_heavy_scene）: 違反 0"]
    README.write_text(README.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    print("書いた:", PLAN, "/ 前の plan:", PLAN_PRE25, "/ manifest:", MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
