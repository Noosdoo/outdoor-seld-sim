# -*- coding: utf-8 -*-
"""v27 plan = v2final の plan（31,274 行）の末尾に、変更票 #64（D31 組み合わせの網羅・v19mix）の行を足す（2026-10-13・Fable）。

本人の決定（2026-10-13 12:28〜12:46）: 9 クラス（車・自転車・キック・バイク・電車・サイレン・クラクション・バック音・踏切の警報機・ベル）の
2 つ組と 3 つ組（同じクラスの重なりも含む）を全部そろえる。除くのは「電車＋バック音」だけ（電車 3 編成同時も置かない）。
電車 2 編成・同じクラス 3 つ・警告音 3 つ・同じ警告音 2 つ（車なし）は残す。学習時間は気にしない。

足す組 = 規則に合う組のうち、今の plan に同じ組み合わせ（音源の構成がちょうど同じ本）が floor_existing（20）本未満のもの
  （既に v11core にある 車＋警告音 の組などは足さない）。1 組あたり 2 つ組 n_scenes_pair（150）場面・3 つ組 n_scenes_triple（20）場面 × 高さ 2 本。
clip 番号（既存の最大は 41250・v25 の 42001〜 は不採用で空き）: h1 50001+i / h2 60001+i（i = 場面の通し番号 < 10,000）。
seed: 9200000000+i。split: i%5==4 を val（v19／v22 と同じ）。motion: rng([seed,7]) < 歩行比。危険層は巡回・車の側は交互・警告音の側は rng([seed, 20261013])。
列を足す: w3_class・w3_side（警告音 3 つ目）・mix（組み合わせの文字列・step11_v27_render.TOKENS の順）。既存 31,274 行は 1 バイトも変えない
  （前の plan を assignment_v2final_pre_v27.csv に残す。新しい列は既存行では空）。
組ごとの一覧 = manifest_v27_mix.csv（mix・大きさ・場面数・本数・clip の範囲・元からあった本数）。

使い方: PYTHONPATH=scripts:src python scripts/step10_v27_plan.py [--dry-run] [--sample N]
  --dry-run = 書かずに本数と検査だけ。--sample N = 場面の検査（サンプラで引いて check_scene）を N 本に絞る（既定は全部）。
検査: scripts/test_v2final_mix.py
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final" / "plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
PLAN_PRE = PLAN_DIR / "assignment_v2final_pre_v27.csv"
MANIFEST = PLAN_DIR / "manifest_v27_mix.csv"
PARAMS = ROOT / "md" / "design" / "v2final_params.json"
README = PLAN_DIR / "README_plan_v2final.md"
COLS_V22 = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side", "w2_class", "w2_side", "danger_tier",
            "car_side", "scenario", "seed", "scene_type", "n_car", "rain", "rain_rate", "rain_dba", "grammar",
            "close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db", "mic_z", "h_copy", "twin_of"]
NEW_COLS = ["w3_class", "w3_side", "mix"]
COLS = COLS_V22 + NEW_COLS
MANIFEST_COLS = ["mix", "size", "n_scenes", "n_clips", "clip_h1_first", "clip_h1_last", "clip_h2_first", "clip_h2_last", "seed_first", "n_existing_before"]
SCENARIO = "v19mix"
SCEN_MOTOFAR = "v19motofar"                    # D32（本人 2026-10-13 13:09「バイクの遠い横距離は足してください」）: v12bike の規則で横距離だけ 4〜12 m
N_EXISTING, MAX_EXISTING_ID = 31274, 41250
ID1, ID2 = 50001, 60001                        # v19mix の高さ 1／2（場面の通し番号 i < 10,000）
IDM1, IDM2 = 70001, 71001                      # v19motofar の高さ 1／2（j < 1,000）
ID_MAX = 72000
SEED0 = 9200000000
SEEDM0 = 9201000000
TIERS = ("critical", "caution", "safe")
SALT_H1, SALT_H2 = 20260908, 20260918          # v19／v22 と同じ（高さの乱数）
SALT_SIDE = 20261013                           # 警告音の側
RAIN_SALT_EXTRA = 19

# 既存の本の「音源の構成」（exact composition）を数えるための対応（scenario → 札）
MOBILE_OF_SCEN = {"v12kick": ["kick"], "v19kickfar": ["kick"], "v12bike": ["moto"], "v19motofar": ["moto"], "v12train": ["train"],
                  "v19bicycle": ["bicycle"], "v19bellbike": ["bicycle", "bike_bell"]}
CAR_SCENS = ("v11core", "v19fast", "v19turn", "v19platoon", "v19launch", "v19static", "v19bellpin")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list, cols: list) -> None:
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def composition(r: dict) -> tuple:
    """既存の行の音源の構成（札の多重集合・TOKENS の順）。電車は 1 編成 = 1 札。ベルつき走行は bicycle+bike_bell。"""
    import step11_v27_render as v27
    s = r["scenario"]; toks = []
    if s in CAR_SCENS:
        toks += ["car"] * int(r.get("n_car") or 0)
    toks += MOBILE_OF_SCEN.get(s, [])
    for w in ("w1_class", "w2_class", "w3_class"):
        if r.get(w):
            toks.append(r[w])
    if s == SCENARIO:
        toks = list(v27.parse_mix(r["mix"]).elements())
    return tuple(sorted(toks, key=v27.TOKENS.index))


def patterns(base: list, P: dict) -> list:
    """足す組の一覧 [(mix, size, n_existing)]（規則に合い、既存に floor 本未満のもの・決定論的な並び）。"""
    import step11_v27_render as v27
    d = P["D31_MIX"]; floor = int(d["rules"]["floor_existing"])
    have = Counter(composition(r) for r in base)
    out = []
    for k in d["rules"]["sizes"]:
        for m in itertools.combinations_with_replacement(v27.TOKENS, int(k)):
            m = tuple(sorted(m, key=v27.TOKENS.index))
            if v27.mix_ok(m):
                continue
            if have.get(m, 0) >= floor:
                continue
            out.append((v27.mix_string(m), int(k), int(have.get(m, 0))))
    return out


def _rain(r: dict, P: dict, p13) -> None:
    kinds, ps = zip(*p13.RAIN_KIND_P.items())
    rng = np.random.default_rng([int(r["seed"]), p13.SALT, RAIN_SALT_EXTRA])
    if rng.random() < p13.RAIN_FRAC:
        kind = str(rng.choice(kinds, p=ps))
        r["rain"], r["rain_rate"], r["rain_dba"] = kind, f"{p13.RAIN_RATE[kind]:.0f}", f"{rng.uniform(*p13.RAIN_DBA[kind]):.1f}"
    else:
        r["rain"], r["rain_rate"], r["rain_dba"] = "", "", ""


def _p13(P):
    """雨の規則は params を正にする（step10_v2final_plan.build_base と同じつまみ）。"""
    import step10_v13_plan as p13
    p13.WALK_FRAC = float(P["D4_WALK_FRAC"])
    p13.RAIN_FRAC = float(P["D3_RAIN_FRAC"])
    p13.RAIN_KIND_P = {k: float(v) for k, v in P["RAIN_KIND_P"].items()}
    p13.RAIN_DBA = {k: (float(v[0]), float(v[1])) for k, v in P["D2_RAIN_DBA"].items()}
    p13.SALT = int(P["SALTS"]["v13"])
    return p13


def build(base: list, P: dict) -> tuple[list, list]:
    import step11_v27_render as v27
    d = P["D31_MIX"]
    p13 = _p13(P)
    walk = float(P["D4_WALK_FRAC"]); zr = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    n_of = {2: int(d["n_scenes_pair"]), 3: int(d["n_scenes_triple"])}
    rows, manifest = [], []
    i = 0
    for mix, size, n_ex in patterns(base, P):
        comp = v27.parse_mix(mix)
        warns = [w for w in v27.WARN for _ in range(comp[w])]
        n = n_of[size]
        first = i
        for _ in range(n):
            seed = SEED0 + i
            split = "val" if i % 5 == 4 else "train"
            motion = "walk" if np.random.default_rng([seed, 7]).random() < walk else "static"
            rs = np.random.default_rng([seed, SALT_SIDE])
            r0 = {"split": split, "motion": motion, "n_warnings": str(len(warns)), "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
                  "w3_class": "", "w3_side": "", "danger_tier": TIERS[i % 3], "car_side": "L" if i % 2 else "R", "scenario": SCENARIO,
                  "seed": str(seed), "scene_type": "v19", "n_car": str(comp["car"]), "grammar": "v19", "close_slow": "", "cs_cpa_m": "",
                  "cs_speed_kmh": "", "cs_level_adj_db": "", "twin_of": "", "mix": mix}
            for k, w in enumerate(warns):
                r0[f"w{k + 1}_class"] = w
                r0[f"w{k + 1}_side"] = "L" if rs.random() < 0.5 else "R"
            for h, salt, idb in ((1, SALT_H1, ID1), (2, SALT_H2, ID2)):
                r = dict(r0)
                r["clip_id"] = f"{split}_syn_clip{idb + i:05d}"
                r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*zr)):.3f}"
                r["h_copy"] = str(h)
                _rain(r, P, p13)
                rows.append(r)
            i += 1
        last = i - 1
        manifest.append({"mix": mix, "size": size, "n_scenes": n, "n_clips": 2 * n,
                         "clip_h1_first": f"{ID1 + first:05d}", "clip_h1_last": f"{ID1 + last:05d}",
                         "clip_h2_first": f"{ID2 + first:05d}", "clip_h2_last": f"{ID2 + last:05d}",
                         "seed_first": SEED0 + first, "n_existing_before": n_ex})
    assert i < ID2 - ID1, f"場面 {i} 本は番号の枡（{ID2 - ID1}）に入らない"
    # D32 v19motofar（遠いバイク）: v22 の build_new と同じ形（車 0 台・警告音なし・mix は空）
    dm = P.get("D32_MOTOFAR") or {}
    if dm.get("adopt"):
        nm = int(dm["n_scenes"])
        assert nm < IDM2 - IDM1, nm
        for j in range(nm):
            seed = SEEDM0 + j
            split = "val" if j % 5 == 4 else "train"
            motion = "walk" if np.random.default_rng([seed, 7]).random() < walk else "static"
            r0 = {"split": split, "motion": motion, "n_warnings": "0", "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
                  "w3_class": "", "w3_side": "", "danger_tier": TIERS[j % 3], "car_side": "L" if j % 2 else "R", "scenario": SCEN_MOTOFAR,
                  "seed": str(seed), "scene_type": "v19", "n_car": "0", "grammar": "v19", "close_slow": "", "cs_cpa_m": "",
                  "cs_speed_kmh": "", "cs_level_adj_db": "", "twin_of": "", "mix": ""}
            for h, salt, idb in ((1, SALT_H1, IDM1), (2, SALT_H2, IDM2)):
                r = dict(r0)
                r["clip_id"] = f"{split}_syn_clip{idb + j:05d}"
                r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*zr)):.3f}"
                r["h_copy"] = str(h)
                _rain(r, P, p13)
                rows.append(r)
        manifest.append({"mix": SCEN_MOTOFAR, "size": 1, "n_scenes": nm, "n_clips": 2 * nm,
                         "clip_h1_first": f"{IDM1:05d}", "clip_h1_last": f"{IDM1 + nm - 1:05d}",
                         "clip_h2_first": f"{IDM2:05d}", "clip_h2_last": f"{IDM2 + nm - 1:05d}",
                         "seed_first": SEEDM0, "n_existing_before": sum(1 for r in base if r["scenario"] == "v12bike")})
    return rows, manifest


def check(base: list, new: list, manifest: list, P: dict) -> list:
    import step11_v27_render as v27
    d = P["D31_MIX"]
    msgs = []
    ids = [r["clip_id"] for r in base + new]
    msgs.append(("clip_id が全部ユニーク", len(ids) == len(set(ids))))
    msgs.append((f"新しい番号は {MAX_EXISTING_ID} より大きく {ID_MAX} 未満", all(MAX_EXISTING_ID < int(r["clip_id"][-5:]) < ID_MAX for r in new)))
    msgs.append(("既存の行は列 w3_class／w3_side／mix が空（土台は触らない）", all(not r.get(c) for r in base for c in NEW_COLS)))
    n_pair = sum(1 for m in manifest if m["size"] == 2); n_tri = sum(1 for m in manifest if m["size"] == 3)
    msgs.append((f"組の数: 2 つ組 {n_pair}（params {d['n_patterns_pair']}）・3 つ組 {n_tri}（params {d['n_patterns_triple']}）",
                 n_pair == int(d["n_patterns_pair"]) and n_tri == int(d["n_patterns_triple"])))
    mix_rows = [r for r in new if r["scenario"] == SCENARIO]; moto_rows = [r for r in new if r["scenario"] == SCEN_MOTOFAR]
    n_exp = 2 * (n_pair * int(d["n_scenes_pair"]) + n_tri * int(d["n_scenes_triple"]))
    msgs.append((f"v19mix の本数 {len(mix_rows):,}（= 2 × ({n_pair} × {d['n_scenes_pair']} + {n_tri} × {d['n_scenes_triple']}) = {n_exp:,}・params n_clips {d['n_clips']:,}）",
                 len(mix_rows) == n_exp == int(d["n_clips"])))
    dm = P.get("D32_MOTOFAR") or {}
    n_moto_exp = 2 * int(dm["n_scenes"]) if dm.get("adopt") else 0
    msgs.append((f"v19motofar の本数 {len(moto_rows):,}（params {n_moto_exp:,}・車 0 台・警告音なし・mix 空・clip {IDM1:05d}〜）",
                 len(moto_rows) == n_moto_exp == int(dm.get("n_clips", 0)) and all(r["n_car"] == "0" and r["n_warnings"] == "0" and r["mix"] == "" and IDM1 <= int(r["clip_id"][-5:]) < ID_MAX for r in moto_rows)))
    msgs.append(("new は v19mix と v19motofar だけ", len(mix_rows) + len(moto_rows) == len(new)))
    tot = int(P.get("N_ROWS_TOTAL_EXPECTED_V27", -1))
    msgs.append((f"総行数 {len(base) + len(new):,} = N_ROWS_TOTAL_EXPECTED_V27 {tot:,}", len(base) + len(new) == tot))
    good = True
    for r in mix_rows:
        comp = v27.parse_mix(r["mix"])
        nw = sum(comp[w] for w in v27.WARN)
        good &= (not v27.mix_ok(comp)) and r["scenario"] == SCENARIO and int(r["n_car"]) == comp["car"] and int(r["n_warnings"]) == nw
        ws = [r[f"w{k}_class"] for k in (1, 2, 3) if r[f"w{k}_class"]]
        good &= ws == [w for w in v27.WARN for _ in range(comp[w])] and all(r[f"w{k}_side"] in ("L", "R") for k in range(1, nw + 1))
        good &= r["motion"] in ("walk", "static") and r["danger_tier"] in TIERS and r["car_side"] in ("L", "R")
    msgs.append(("新しい行: 規則に合う mix・scenario・n_car・n_warnings・w1〜w3 の列が mix と一致・側は L/R", good))
    by_seed = {}
    for r in new:
        by_seed.setdefault(r["seed"], []).append(r)
    msgs.append(("同じ種の 2 本（高さ 1・2）が同じ split・同じ scenario・同じ mix（mic_z は独立に引くので 3 桁丸めで同じ値になることがある・v19／v22 と同じ規則）",
                 all(len(v) == 2 and v[0]["split"] == v[1]["split"] and v[0]["scenario"] == v[1]["scenario"] and v[0]["mix"] == v[1]["mix"] and {v[0]["h_copy"], v[1]["h_copy"]} == {"1", "2"} for v in by_seed.values())))
    c = Counter(r["split"] for r in new)
    msgs.append((f"split: train {c['train']:,} / val {c['val']:,}（val 20% ± 1%）", abs(c["val"] / max(len(new), 1) - 0.2) < 0.01))
    walk = sum(r["motion"] == "walk" for r in new) / max(len(new), 1); rain = sum(bool(r["rain"]) for r in new) / max(len(new), 1)
    msgs.append((f"歩行 {walk:.1%}（0.70 ± 0.02）・雨 {rain:.1%}（0.20 ± 0.02）", abs(walk - float(P["D4_WALK_FRAC"])) < 0.02 and abs(rain - float(P["D3_RAIN_FRAC"])) < 0.02))
    mixes = Counter(r["mix"] for r in mix_rows)
    msgs.append(("組ごとの本数が manifest と一致", all(mixes[m["mix"]] == m["n_clips"] for m in manifest if m["size"] > 1) and len(mixes) == sum(1 for m in manifest if m["size"] > 1)))
    return msgs


def main() -> int:
    a = sys.argv[1:]
    dry = "--dry-run" in a
    known = {"--dry-run", "--sample", "--force-overwrite-superseded"}
    bad_args = [x for x in a if x.startswith("-") and x not in known] + [x for i, x in enumerate(a) if not x.startswith("-") and (i == 0 or a[i - 1] != "--sample")]
    if bad_args:
        print(f"見覚えのない引数 {bad_args}。何もしない"); return 2
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    # 2026-10-19 #70（v29）の門: v29 の plan（N_ROWS_TOTAL_EXPECTED_V29）が正になったあとは、この作成器で正本を書き直さない（旧の雨の dba で上書きしてしまう）。
    #   12:2x に「selftest」を付けて呼んだら引数が無視されて正本を書き直した事故の再発防止。見覚えのない引数でも止める。
    if not dry and P.get("N_ROWS_TOTAL_EXPECTED_V29") and "--force-overwrite-superseded" not in a:
        print("v29 の plan（step10_v29_plan.py）が正。この作成器では正本を書かない（--dry-run か --force-overwrite-superseded）"); return 2
    assert (P.get("D31_MIX") or {}).get("adopt"), "params の D31_MIX.adopt が true でない"
    base = read_rows(PLAN_PRE if PLAN_PRE.exists() else PLAN)
    assert len(base) == N_EXISTING, f"元の plan が {len(base)} 行（31,274 でない）"
    assert all(r.get("scenario") != SCENARIO for r in base), "元の plan に v19mix の行がある"
    new, manifest = build(base, P)
    msgs = check(base, new, manifest, P)
    for name, ok in msgs:
        print(("OK " if ok else "NG ") + name)
    import step11_v27_render as v27
    lim = int(a[a.index("--sample") + 1]) if "--sample" in a else None
    bad = 0; n = 0
    for r in new:
        if lim is not None and n >= lim:
            break
        n += 1
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            v27.check_row(v27.sample_scene_v19(rr), rr)
        except AssertionError as e:
            bad += 1
            if bad <= 5:
                print("NG 場面", r["clip_id"], e)
    print(f"場面の検査（サンプラで引いて check_scene／check_motofar）: {n:,} 本中 NG {bad}")
    n_moto = sum(1 for r in new if r["scenario"] == SCEN_MOTOFAR)
    print(f"- 2 つ組 {sum(1 for m in manifest if m['size'] == 2)} 種 × {P['D31_MIX']['n_scenes_pair']} 場面・3 つ組 {sum(1 for m in manifest if m['size'] == 3)} 種 × {P['D31_MIX']['n_scenes_triple']} 場面 → v19mix {len(new) - n_moto:,} 本・v19motofar {n_moto:,} 本・総行数 {len(base) + len(new):,}")
    if not all(ok for _, ok in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    if not PLAN_PRE.exists():
        PLAN_PRE.write_bytes(PLAN.read_bytes())
    write_rows(PLAN, base + new, COLS)
    write_rows(MANIFEST, manifest, MANIFEST_COLS)
    c = Counter(r["split"] for r in new)
    lines = [f"\n## v27 追加（2026-10-13・変更票 #64 D31 組み合わせの網羅）: {len(new):,} 行を末尾に足した → 総行数 {len(base) + len(new):,}",
             f"- 元の plan（31,274 行）= assignment_v2final_pre_v27.csv sha256 {sha256(PLAN_PRE)}",
             f"- 新しい plan sha256 {sha256(PLAN)}（列 w3_class・w3_side・mix を追加・既存行は空）",
             f"- v19mix: 2 つ組 {sum(1 for m in manifest if m['size'] == 2)} 種 × {P['D31_MIX']['n_scenes_pair']} 場面 ＋ 3 つ組 {sum(1 for m in manifest if m['size'] == 3)} 種 × {P['D31_MIX']['n_scenes_triple']} 場面・高さ 2 本・clip {ID1:05d}〜（高さ 1）・{ID2:05d}〜（高さ 2）",
             f"- v19motofar（D32・遠いバイク 4〜12 m）: {n_moto:,} 本・clip {IDM1:05d}〜（高さ 1）・{IDM2:05d}〜（高さ 2）",
             f"- 足した行の split: train {c['train']:,} / val {c['val']:,}",
             f"- 組ごとの一覧 = manifest_v27_mix.csv sha256 {sha256(MANIFEST)}",
             f"- 場面の検査（step11_v27_render.check_scene）: {n:,} 本 NG 0"]
    README.write_text(README.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    print("書いた:", PLAN, "/ 前の plan:", PLAN_PRE, "/ manifest:", MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
