# -*- coding: utf-8 -*-
"""v22 plan = v2final の plan（25,500 行）の末尾に、変更票 #48（D22 bicycle）と #49（D23 ピン型ベル）の 5,774 行を足す（2026-10-08）。

足す行（params md/design/v2final_params.json の D22_BICYCLE / D22_BELLBIKE / D23_BELLPIN の adopt が true のものだけ）:
  v19bellpin  774 本 = 土台（v11core）のベルの本 774 場面（同じ種で高さ 2 本）を clip_id の順に 1 つおきに 387 場面選び、双子を作る。
              列は元の行の写し（motion・警告音の列・層・左右・scene_type・n_car・雨・mic_z・h_copy）。変えるのは clip_id・seed（新しい）・
              scenario（v19bellpin）・grammar（v19）と twin_of（元の clip_id）。ベルの型はピン型に固定（描画側 step11_v22_render）。
  v19bicycle  4,500 本 = 2,250 場面 × 高さ 2 本（本人 10/8 00:46「(b)」= 可聴ゲートでラベルが残る本が 22% だったので 2 倍に）。v19 の規則（歩行 70%・層の巡回・左右交互・mic_z は種と塩から・雨は v2final の規則）。
  v19bellbike 500 本 = 250 場面 × 高さ 2 本。同上。
clip 番号（既存の最大は 31750・重ならない）: bellpin h1 32001+i / h2 33001+i（i<387）、bicycle h1 34001+i / h2 37001+i（i<2,250）、
  bellbike h1 40001+i / h2 41001+i（i<250）。split は i%5==4 を val（v19 と同じ）。双子は元の split を写す。
seed: bellpin 9196000000+i、bicycle 9197000000+i、bellbike 9198000000+i（v19 の SEED_BASE 9190000000 の続き・既存と重ならない）。
twin_of 列を plan に足す（既存の行は空）。既存 25,500 行は 1 バイトも変えない（前の plan を assignment_v2final_pre_v22.csv に残す）。

使い方: PYTHONPATH=scripts:src python scripts/step10_v22_plan.py [--dry-run]   # dry-run は書かずに本数と検査だけ
検査: scripts/test_v2final_bicycle_bellpin.py
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final" / "plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
PLAN_PRE = PLAN_DIR / "assignment_v2final_pre_v22.csv"
PARAMS = ROOT / "md" / "design" / "v2final_params.json"
README = PLAN_DIR / "README_plan_v2final.md"
COLS_BASE = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side", "w2_class", "w2_side", "danger_tier",
             "car_side", "scenario", "seed", "scene_type", "n_car", "rain", "rain_rate", "rain_dba", "grammar",
             "close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db", "mic_z", "h_copy"]
COLS = COLS_BASE + ["twin_of"]
BLOCKS = {  # key: (scenario, params key, id base h1, id base h2, seed base)
    "bellpin": ("v19bellpin", "D23_BELLPIN", 32001, 33001, 9196000000),
    "bicycle": ("v19bicycle", "D22_BICYCLE", 34001, 37001, 9197000000),   # 2,250 場面 → h1 34001〜36250・h2 37001〜39250
    "bellbike": ("v19bellbike", "D22_BELLBIKE", 40001, 41001, 9198000000),
}
N_EXISTING = 25500
MAX_EXISTING_ID = 31750
TIERS = ("critical", "caution", "safe")
SALT_H1, SALT_H2 = 20260908, 20260918
RAIN_SALT_EXTRA = 19


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list, cols: list) -> None:
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _rain(r: dict, P: dict, p13) -> None:
    kinds, ps = zip(*p13.RAIN_KIND_P.items())
    rng = np.random.default_rng([int(r["seed"]), p13.SALT, RAIN_SALT_EXTRA])
    if rng.random() < p13.RAIN_FRAC:
        kind = str(rng.choice(kinds, p=ps))
        r["rain"], r["rain_rate"], r["rain_dba"] = kind, f"{p13.RAIN_RATE[kind]:.0f}", f"{rng.uniform(*p13.RAIN_DBA[kind]):.1f}"
    else:
        r["rain"], r["rain_rate"], r["rain_dba"] = "", "", ""


def build_twins(base: list, n_scenes: int, scenario: str, id1: int, id2: int, seed0: int) -> list:
    bell = [r for r in base if r["scenario"] == "v11core" and (r["w1_class"] == "bike_bell" or r["w2_class"] == "bike_bell")]
    by_seed = defaultdict(list)
    for r in bell:
        by_seed[r["seed"]].append(r)
    scenes = sorted(by_seed.values(), key=lambda v: min(r["clip_id"] for r in v))
    assert all(len(v) == 2 for v in scenes), "土台のベルの場面が高さ 2 本になっていない"
    chosen = scenes[::2][:n_scenes]
    assert len(chosen) == n_scenes, f"双子の元が足りない {len(chosen)} < {n_scenes}"
    out = []
    for i, pair in enumerate(chosen):
        pair = sorted(pair, key=lambda r: int(r["h_copy"]))
        for r in pair:
            t = {c: r.get(c, "") for c in COLS_BASE}
            n = (id1 if r["h_copy"] == "1" else id2) + i
            t["clip_id"] = f"{r['split']}_syn_clip{n:05d}"
            t["seed"] = str(seed0 + i)
            t["scenario"] = scenario; t["grammar"] = "v19"
            t["twin_of"] = r["clip_id"]
            out.append(t)
    return out


def build_new(scenario: str, n_scenes: int, id1: int, id2: int, seed0: int, P: dict, p13, n_car: str) -> list:
    walk = float(P["D4_WALK_FRAC"]); zr = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    out = []
    for i in range(n_scenes):
        seed = seed0 + i
        split = "val" if i % 5 == 4 else "train"
        motion = "walk" if np.random.default_rng([seed, 7]).random() < walk else "static"
        base = {"split": split, "motion": motion, "n_warnings": "0", "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
                "danger_tier": TIERS[i % 3], "car_side": "L" if i % 2 else "R", "scenario": scenario, "seed": str(seed),
                "scene_type": "v19", "n_car": n_car, "grammar": "v19", "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "",
                "cs_level_adj_db": "", "twin_of": ""}
        for h, salt, idb in ((1, SALT_H1, id1), (2, SALT_H2, id2)):
            r = dict(base)
            r["clip_id"] = f"{split}_syn_clip{idb + i:05d}"
            r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*zr)):.3f}"
            r["h_copy"] = str(h)
            _rain(r, P, p13)
            out.append(r)
    return out


def build_all(base: list, P: dict) -> tuple[list, dict]:
    import step10_v13_plan as p13
    new, summary = [], {}
    for key, (scen, pkey, id1, id2, seed0) in BLOCKS.items():
        d = P.get(pkey) or {}
        if not d.get("adopt"):
            continue
        n = int(d["n_scenes"])
        rows = build_twins(base, n, scen, id1, id2, seed0) if key == "bellpin" else build_new(scen, n, id1, id2, seed0, P, p13, "0")
        assert len(rows) == int(d["n_clips"]) == 2 * n, (key, len(rows), d)
        c = Counter(r["split"] for r in rows)
        summary[scen] = dict(n=len(rows), train=c["train"], val=c["val"], walk=sum(r["motion"] == "walk" for r in rows) / len(rows),
                             rain=sum(bool(r["rain"]) for r in rows) / len(rows), ids=(rows[0]["clip_id"][-5:], rows[-1]["clip_id"][-5:]))
        new += rows
    return new, summary


def check(base: list, new: list, P: dict) -> list:
    msgs = []
    allrows = base + new
    ids = [r["clip_id"] for r in allrows]
    msgs.append(("clip_id が全部ユニーク", len(ids) == len(set(ids))))
    msgs.append(("既存 25,500 行が先頭にそのまま", len(base) == N_EXISTING))
    msgs.append(("新しい番号が既存の最大より大きい", all(int(r["clip_id"][-5:]) > MAX_EXISTING_ID for r in new)))
    exp = P.get("N_ROWS_TOTAL_EXPECTED_V22")
    msgs.append((f"総行数 {len(allrows):,} = 期待 {exp}", exp is not None and len(allrows) == int(exp)))
    seeds = [r["seed"] for r in allrows]
    msgs.append(("seed が全部ユニーク（高さの対を除く）", len(set(seeds)) * 2 == len(seeds) or len(set(seeds)) == len(seeds) // 2))
    twins = [r for r in new if r["scenario"] == "v19bellpin"]
    base_by_id = {r["clip_id"]: r for r in base}
    ok_twin = all(r["twin_of"] in base_by_id and base_by_id[r["twin_of"]]["scenario"] == "v11core"
                  and ("bike_bell" in (base_by_id[r["twin_of"]]["w1_class"], base_by_id[r["twin_of"]]["w2_class"]))
                  and all(r[c] == base_by_id[r["twin_of"]][c] for c in ("motion", "n_warnings", "w1_class", "w1_side", "w2_class", "w2_side", "danger_tier", "car_side", "scene_type", "n_car", "rain", "rain_rate", "rain_dba", "mic_z", "h_copy", "split"))
                  for r in twins)
    msgs.append(("双子の元が全部 v11core のベルの本で、列が写しになっている", ok_twin))
    msgs.append(("双子の元に重複が無い", len({r["twin_of"] for r in twins}) == len(twins)))
    return msgs


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    base = read_rows(PLAN_PRE if PLAN_PRE.exists() else PLAN)
    assert len(base) == N_EXISTING, f"元の plan が {len(base)} 行（25,500 でない）"
    new, summary = build_all(base, P)
    msgs = check(base, new, P)
    for name, ok in msgs:
        print(("OK " if ok else "NG ") + name)
    import step11_v22_render as v22r
    bad = 0
    for r in new:
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            v22r.check_scene(r["scenario"], v22r.sample_scene_v19(rr))
        except AssertionError as e:
            bad += 1
            if bad <= 5:
                print("NG 場面", r["clip_id"], e)
    print(f"場面の検査: {len(new)} 本中 NG {bad}")
    for scen, d in summary.items():
        print(f"- {scen}: {d['n']:,} 本（train {d['train']:,} / val {d['val']:,}）・歩行 {d['walk']:.1%}・雨 {d['rain']:.1%}・clip {d['ids'][0]}〜{d['ids'][1]}")
    if not all(ok for _, ok in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    if not PLAN_PRE.exists():
        PLAN_PRE.write_bytes(PLAN.read_bytes())
    write_rows(PLAN, base + new, COLS)
    lines = [f"\n## v22 追加（2026-10-08・変更票 #48 D22・#49 D23）: {len(new):,} 行を末尾に足した → 総行数 {len(base) + len(new):,}",
             f"- 元の plan（25,500 行）= assignment_v2final_pre_v22.csv sha256 {sha256(PLAN_PRE)}", f"- 新しい plan sha256 {sha256(PLAN)}（twin_of 列を追加・既存行は空）"]
    for scen, d in summary.items():
        lines.append(f"- {scen}: {d['n']:,} 本（train {d['train']:,} / val {d['val']:,}）・歩行 {d['walk']:.1%}・雨 {d['rain']:.1%}・clip {d['ids'][0]}〜{d['ids'][1]}")
    lines.append("- 場面の検査（step11_v22_render.check_scene）: 違反 0")
    README.write_text(README.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    print("書いた:", PLAN, "/ 前の plan:", PLAN_PRE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
