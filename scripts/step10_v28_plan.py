# -*- coding: utf-8 -*-
"""v28 plan = 組み合わせ（v19mix）の積み増し（本人決定 2026-10-13 14:45・Opus 敵対的レビュー R-1／M-1 への対応・変更票 #64 の追記）。

何をするか: v27 の plan（50,574 行）の末尾に v19mix の行を足す。足す場面の数は 2 つの規則で決める。
  (a) ゲート後の積み増し: 描き直した束の H 検査（_v26_regen_check.py が書く REGEN_H_counts.csv = mix, allocated, all_labeled）から、
      組ごとに「全部の音のラベルが残る本」の実数 n と割当 a を読み、残る率 r = n/a で、目標 T（20 本）に届くまでの追加の本数 ≈ (T − n) / max(r, r_min) × 余裕 を
      場面数（÷ 2・切り上げ）にする。1 組の上限 cap（400 場面）。r が r_min 以下（ほぼ必ず埋もれる組）は上限で止め、届かない見込みとして manifest に印を付ける。
  (b) 床の積み増し（M-1）: 規則に合う 2 つ組のうち、今の plan に同じ構成の場面が pair_target_scenes（150）未満のもの（v27 で床 20 本＝10 場面以上として足さなかった
      車なしの警告音 2 つ 13 組など）を 150 場面まで足す。
行の作り方は step10_v27_plan と同じ（同じ列・seed から歩行・雨・高さ・側）。scene_type = "v28" で見分ける。clip 番号 h1 72001+i / h2 86001+i（i < 14,000）・seed 9202000000+i。
既存 50,574 行は値を変えない（前の plan を assignment_v2final_pre_v28.csv に残す）。組ごとの一覧 = manifest_v28_topup.csv。params には D33_TOPUP の実績（n_clips・組の数）と
N_ROWS_TOTAL_EXPECTED_V28 を書く（--dry-run は書かない）。

使い方: PYTHONPATH=scripts:src python scripts/step10_v28_plan.py [--h-csv <REGEN_H_counts.csv>] [--dry-run] [--sample N] [--table]
  --table = 本数の表だけ出す（本人に見せる用・書かない）。--selftest <tmp> = 偽の H で計算の検査。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v27_plan as p27  # noqa: E402

PLAN_DIR = p27.PLAN_DIR
PLAN = p27.PLAN
PLAN_PRE = PLAN_DIR / "assignment_v2final_pre_v28.csv"
MANIFEST = PLAN_DIR / "manifest_v28_topup.csv"
H_CSV_DEFAULT = PLAN_DIR / "REGEN_H_counts_v27.csv"
PARAMS = p27.PARAMS
README = p27.README
COLS = p27.COLS
MANIFEST_COLS = ["mix", "size", "reason", "allocated_before", "all_labeled_before", "retention", "scenes_before", "add_scenes", "add_clips",
                 "clip_h1_first", "clip_h1_last", "clip_h2_first", "clip_h2_last", "seed_first", "expected_all_labeled_after", "shortfall"]
SCENARIO = p27.SCENARIO
SCENE_TYPE = "v28"
N_EXISTING_V27 = 50574
ID1, ID2 = 72001, 86001
ID_MAX = 100000
SEED0 = 9202000000
DEFAULTS = {"target_all_labeled": 20, "safety": 1.3, "r_min": 0.02, "cap_scenes_per_mix": 400, "pair_target_scenes": 150}


def read_h(path: Path) -> dict:
    """REGEN_H_counts.csv（mix, allocated, all_labeled, …source 列があれば v19mix／existing）→ {mix: (allocated, all_labeled)}。既存の場面だけの組も入る（Astra MIX13-04）。"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["mix"]] = (int(r["allocated"]), int(r["all_labeled"]))
    return out


def read_h_sources(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["mix"]: r.get("source", "") for r in csv.DictReader(f)}


CAP_SCENES_TOTAL = ID2 - ID1 - 1      # 番号枠: 場面の通し番号 i < 14,000（Astra MIX13-06・黙って縮めない＝超えたら止めて本人へ）


def capacity_check(adds: list) -> tuple[int, bool]:
    tot = sum(int(d["add_scenes"]) for d in adds)
    return tot, tot <= CAP_SCENES_TOTAL


def plan_additions(base: list, H: dict, D: dict) -> list:
    """足す組の一覧 [dict(mix, size, reason, a, n, r, scenes_before, add_scenes, shortfall)]（決定論的な並び）。"""
    import step11_v27_render as v27
    T = int(D.get("target_all_labeled", DEFAULTS["target_all_labeled"])); safety = float(D.get("safety", DEFAULTS["safety"]))
    r_min = float(D.get("r_min", DEFAULTS["r_min"])); cap = int(D.get("cap_scenes_per_mix", DEFAULTS["cap_scenes_per_mix"]))
    pair_target = int(D.get("pair_target_scenes", DEFAULTS["pair_target_scenes"]))
    scenes_of = Counter()
    for r in base:
        scenes_of[p27.composition(r)] += 1          # 本数（高さ 2 本で 1 場面）
    out = []
    # (a) ゲート後
    for mix in sorted(H, key=lambda m: (len(v27.parse_mix(m)), v27.mix_string(v27.parse_mix(m)))):
        a, n = H[mix]
        comp = v27.parse_mix(mix)
        if v27.mix_ok(comp):                                   # H は全部の場面を数えるので、既存の 4 音源以上の組（v11core の車 2 台＋警告音 2 つ など）も入る → 積み増しの対象（2〜3 音源・規則に合う組）だけ
            continue
        if n >= T or a <= 0:
            continue
        r = n / a
        need_clips = (T - n) / max(r, r_min) * safety
        add = min(cap, int(math.ceil(need_clips / 2.0)))
        short = r <= r_min or add >= cap and (n + 2 * add * r) < T
        out.append(dict(mix=v27.mix_string(comp), size=sum(comp.values()), reason="gate", a=a, n=n, r=r,
                        scenes_before=scenes_of[tuple(sorted(comp.elements(), key=v27.TOKENS.index))] // 2, add_scenes=add,
                        expected_after=n + 2 * add * r, shortfall=bool(short)))
    # (b) 床（2 つ組を 150 場面に）
    import itertools
    done = {d["mix"]: d for d in out}
    for m in itertools.combinations_with_replacement(v27.TOKENS, 2):
        m = tuple(sorted(m, key=v27.TOKENS.index))
        if v27.mix_ok(m):
            continue
        mix = v27.mix_string(m)
        sc = scenes_of[m] // 2
        if mix in done:                                        # ゲート後で少しだけ足された 2 つ組にも床を効かせる（例: backup_beep+crossing 10 場面 +1 → 150 場面まで）
            d = done[mix]
            if sc + d["add_scenes"] < pair_target:
                d["add_scenes"] = pair_target - sc; d["reason"] = "gate+floor"
                d["expected_after"] = d["n"] + 2 * d["add_scenes"] * d["r"]; d["shortfall"] = False
            continue
        if sc >= pair_target:
            continue
        a, n = H.get(mix, (0, 0))
        out.append(dict(mix=mix, size=2, reason="floor", a=a, n=n, r=(n / a if a else float("nan")), scenes_before=sc,
                        add_scenes=pair_target - sc, expected_after=float("nan"), shortfall=False))
    return out


def build(base: list, adds: list, P: dict) -> tuple[list, list]:
    import step11_v27_render as v27
    p13 = p27._p13(P)
    walk = float(P["D4_WALK_FRAC"]); zr = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    rows, manifest = [], []
    i = 0
    for d in adds:
        comp = v27.parse_mix(d["mix"]); warns = [w for w in v27.WARN for _ in range(comp[w])]
        first = i
        for _ in range(int(d["add_scenes"])):
            seed = SEED0 + i
            split = "val" if i % 5 == 4 else "train"
            motion = "walk" if np.random.default_rng([seed, 7]).random() < walk else "static"
            rs = np.random.default_rng([seed, p27.SALT_SIDE])
            r0 = {"split": split, "motion": motion, "n_warnings": str(len(warns)), "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
                  "w3_class": "", "w3_side": "", "danger_tier": p27.TIERS[i % 3], "car_side": "L" if i % 2 else "R", "scenario": SCENARIO,
                  "seed": str(seed), "scene_type": SCENE_TYPE, "n_car": str(comp["car"]), "grammar": "v19", "close_slow": "", "cs_cpa_m": "",
                  "cs_speed_kmh": "", "cs_level_adj_db": "", "twin_of": "", "mix": d["mix"]}
            for k, w in enumerate(warns):
                r0[f"w{k + 1}_class"] = w
                r0[f"w{k + 1}_side"] = "L" if rs.random() < 0.5 else "R"
            for h, salt, idb in ((1, p27.SALT_H1, ID1), (2, p27.SALT_H2, ID2)):
                r = dict(r0)
                r["clip_id"] = f"{split}_syn_clip{idb + i:05d}"
                r["mic_z"] = f"{float(np.random.default_rng([seed, salt]).uniform(*zr)):.3f}"
                r["h_copy"] = str(h)
                p27._rain(r, P, p13)
                rows.append(r)
            i += 1
        last = i - 1
        manifest.append({"mix": d["mix"], "size": d["size"], "reason": d["reason"], "allocated_before": d["a"], "all_labeled_before": d["n"],
                         "retention": f"{d['r']:.3f}" if d["r"] == d["r"] else "", "scenes_before": d["scenes_before"], "add_scenes": d["add_scenes"],
                         "add_clips": 2 * d["add_scenes"], "clip_h1_first": f"{ID1 + first:05d}", "clip_h1_last": f"{ID1 + last:05d}",
                         "clip_h2_first": f"{ID2 + first:05d}", "clip_h2_last": f"{ID2 + last:05d}", "seed_first": SEED0 + first,
                         "expected_all_labeled_after": f"{d['expected_after']:.1f}" if d["expected_after"] == d["expected_after"] else "",
                         "shortfall": int(d["shortfall"])})
    assert i < ID2 - ID1, f"場面 {i} 本は番号の枡（{ID2 - ID1}）に入らない"
    return rows, manifest


def table(adds: list) -> str:
    L = [f"組 {len(adds)}（ゲート後 {sum(d['reason'] == 'gate' for d in adds)}・床 {sum(d['reason'] == 'floor' for d in adds)}）・追加 {sum(d['add_scenes'] for d in adds):,} 場面 = {2 * sum(d['add_scenes'] for d in adds):,} 本・届かない見込み {sum(d['shortfall'] for d in adds)} 組", ""]
    for d in sorted(adds, key=lambda d: (-d["add_scenes"], d["mix"])):
        r = f"{d['r']:.2f}" if d["r"] == d["r"] else "-"
        L.append(f"- {d['mix']}: {d['reason']}・割当 {d['a']}・全部残る {d['n']}・残る率 {r}・場面 {d['scenes_before']} → +{d['add_scenes']}（{2 * d['add_scenes']} 本）" + ("・**上限でも届かない見込み**" if d["shortfall"] else ""))
    return "\n".join(L)


def check(base: list, new: list, manifest: list, adds: list) -> list:
    import step11_v27_render as v27
    msgs = []
    ids = [r["clip_id"] for r in base + new]
    msgs.append(("clip_id が全部ユニーク", len(ids) == len(set(ids))))
    msgs.append((f"新しい番号は 71750 より大きく {ID_MAX} 未満", all(71750 < int(r["clip_id"][-5:]) < ID_MAX for r in new)))
    msgs.append(("既存の行に scene_type=v28 は無い・新しい行は全部 v19mix／v28", all(r.get("scene_type") != SCENE_TYPE for r in base) and all(r["scenario"] == SCENARIO and r["scene_type"] == SCENE_TYPE for r in new)))
    msgs.append((f"本数 {len(new):,} = 2 × 追加の場面 {sum(d['add_scenes'] for d in adds):,}", len(new) == 2 * sum(d["add_scenes"] for d in adds) and len(new) == sum(m["add_clips"] for m in manifest)))
    good = True
    for r in new:
        comp = v27.parse_mix(r["mix"]); nw = sum(comp[w] for w in v27.WARN)
        good &= (not v27.mix_ok(comp)) and int(r["n_car"]) == comp["car"] and int(r["n_warnings"]) == nw
        good &= [r[f"w{k}_class"] for k in (1, 2, 3) if r[f"w{k}_class"]] == [w for w in v27.WARN for _ in range(comp[w])]
    msgs.append(("新しい行: 規則に合う mix・n_car・n_warnings・w1〜w3 が mix と一致", good))
    by_seed = {}
    for r in new:
        by_seed.setdefault(r["seed"], []).append(r)
    msgs.append(("同じ種の 2 本が同じ split・同じ mix・高さ 1 と 2", all(len(v) == 2 and v[0]["split"] == v[1]["split"] and v[0]["mix"] == v[1]["mix"] and {v[0]["h_copy"], v[1]["h_copy"]} == {"1", "2"} for v in by_seed.values())))
    mixes = Counter(r["mix"] for r in new)
    msgs.append(("組ごとの本数が manifest と一致", all(mixes[m["mix"]] == m["add_clips"] for m in manifest) and len(mixes) == len(manifest)))
    return msgs


def selftest(tmp: Path) -> int:
    ok = []
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    D = dict(DEFAULTS)
    base = p27.read_rows(PLAN_PRE if PLAN_PRE.exists() else PLAN)
    H = {"car+bicycle+kick": (40, 3), "bicycle+bicycle+train": (40, 0), "car+bicycle": (300, 250), "kick+kick+kick": (40, 4), "car+kick": (300, 19)}
    adds = plan_additions(base, H, D)
    by = {d["mix"]: d for d in adds}
    ok.append(("3/40（r 0.075）→ (17/0.075)×1.3 = 294.7 本 → 148 場面", by["car+bicycle+kick"]["add_scenes"] == 148 and not by["car+bicycle+kick"]["shortfall"]))
    ok.append(("0/40（r 0 → r_min 0.02）→ 上限 400 場面・届かない印", by["bicycle+bicycle+train"]["add_scenes"] == 400 and by["bicycle+bicycle+train"]["shortfall"]))
    ok.append(("250/300（目標 20 を超える）→ 足さない", "car+bicycle" not in by))
    ok.append(("19/300（r 0.063）→ (1/0.063)×1.3 = 20.5 本 → 11 場面", by["car+kick"]["add_scenes"] == 11))
    floors = [d for d in adds if d["reason"] == "floor"]
    ok.append((f"床: 2 つ組で 150 場面未満の組 {len(floors)}（v27 で足さなかった車なしの警告音 2 つ = 13 のはず）・合計 +{sum(d['add_scenes'] for d in floors)} 場面", len(floors) == 13 and all(d["scenes_before"] + d["add_scenes"] == 150 for d in floors)))
    rows, manifest = build(base, adds, P)
    msgs = check(base, rows, manifest, adds)
    ok.append((f"build と check（{len(rows):,} 本）: " + " / ".join(("OK" if g else "NG") + " " + n for n, g in msgs), all(g for _, g in msgs)))
    tot, fits = capacity_check(adds); ok.append((f"番号枠の確認: 追加 {tot:,} 場面 ≤ {CAP_SCENES_TOTAL:,} → {fits}", fits))
    Hbig = {m: (40, 0) for m in H} | {f"mixbig{i}": (40, 0) for i in range(230)}
    try:
        tot_big, fits_big = capacity_check(plan_additions(base, {m: v for m, v in H.items()}, D) + [dict(mix=f"x{i}", size=3, reason="gate", a=40, n=0, r=0.0, scenes_before=0, add_scenes=400, expected_after=0.0, shortfall=True) for i in range(230)])
        ok.append((f"番号枠: 230 組 × 400 場面の境界入力は枠を超えると判定（{tot_big:,} 場面）", not fits_big))
    except Exception as e:  # noqa: BLE001
        ok.append((f"番号枠の境界入力: {e}", False))
    drv = (ROOT / "scripts/_run_v2final_gen.py").read_text(encoding="utf-8")
    ok.append(("生成器の入口が N_ROWS_TOTAL_EXPECTED_V28 を V27 より先に読む（Astra MIX13-01）", drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V28")') != -1 and drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V28")') < drv.find('P.get("N_ROWS_TOTAL_EXPECTED_V27")')))
    import step11_v27_render as v27
    bad = 0
    for r in rows[:: max(1, len(rows) // 40)][:40]:
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            v27.check_row(v27.sample_scene_v19(rr), rr)
        except AssertionError as e:
            bad += 1; print("NG", r["clip_id"], e)
    ok.append(("足した行 40 本をサンプラで引いて check_row: NG 0", bad == 0))
    print(table(adds)[:600])
    for n, g in ok:
        print(("PASS " if g else "FAIL ") + n)
    nb = sum(not g for _, g in ok); print(f"selftest v28: {len(ok) - nb}/{len(ok)}")
    return 1 if nb else 0


def main() -> int:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest(Path(a[a.index("--selftest") + 1]))
    dry = "--dry-run" in a or "--table" in a
    known = {"--h-csv", "--dry-run", "--sample", "--table", "--selftest", "--force-overwrite-superseded"}
    bad_args = [x for x in a if x.startswith("-") and x not in known] + [x for i, x in enumerate(a) if not x.startswith("-") and (i == 0 or a[i - 1] not in ("--h-csv", "--sample", "--selftest"))]
    if bad_args:
        print(f"見覚えのない引数 {bad_args}（selftest は --selftest <tmp>）。何もしない"); return 2
    P = json.loads(PARAMS.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    # 2026-10-19 #70（v29）の門: v29 の plan（N_ROWS_TOTAL_EXPECTED_V29）が正になったあとは、この作成器で正本を書き直さない（旧の雨の dba で上書きしてしまう）。
    #   12:2x に「selftest」を付けて呼んだら引数が無視されて正本を書き直した事故の再発防止。見覚えのない引数でも止める。
    if not dry and P.get("N_ROWS_TOTAL_EXPECTED_V29") and "--force-overwrite-superseded" not in a:
        print("v29 の plan（step10_v29_plan.py）が正。この作成器では正本を書かない（--dry-run か --force-overwrite-superseded）"); return 2
    D = P.get("D33_TOPUP") or {}
    assert D.get("adopt"), "params の D33_TOPUP.adopt が true でない"
    h_csv = Path(a[a.index("--h-csv") + 1]) if "--h-csv" in a else H_CSV_DEFAULT
    H = read_h(h_csv)
    base = p27.read_rows(PLAN_PRE if PLAN_PRE.exists() else PLAN)
    assert len(base) == N_EXISTING_V27, f"元の plan が {len(base)} 行（50,574 でない）"
    assert all(r.get("scene_type") != SCENE_TYPE for r in base), "元の plan に v28 の行がある"
    # H と plan の整合（Astra MIX13-04 のあと H は全部の場面を数え、電車だけの組は対象外）: plan の v19mix の組ごとの本数が H の割当以下で、v19mix だけの組は一致
    import step11_v27_render as v27
    src_of = read_h_sources(h_csv)
    c_plan = Counter(r["mix"] for r in base if r["scenario"] == SCENARIO)
    bad = []
    for m, c in c_plan.items():
        if all(t == "train" for t in v27.parse_mix(m).elements()):
            continue                                                   # 教師音源ゼロの組（train+train）は H に無い
        alloc = H.get(m, (0, 0))[0]
        if alloc < c or (src_of.get(m, "v19mix") == "v19mix" and alloc != c):
            bad.append((m, c, alloc, src_of.get(m, "")))
    assert not bad, f"H と plan の本数が合わない組: {bad[:5]}"
    adds = plan_additions(base, H, D)
    print(table(adds))
    tot, fits = capacity_check(adds)
    print(f"- 番号枠: 追加 {tot:,} 場面 / 枠 {CAP_SCENES_TOTAL:,} 場面 → {'収まる' if fits else '**超える（黙って縮めない・番号の体系を広げるか上限を決め直すかを本人に）**'}")
    if not fits:
        return 2
    if "--table" in a:
        return 0
    new, manifest = build(base, adds, P)
    msgs = check(base, new, manifest, adds)
    for name, ok in msgs:
        print(("OK " if ok else "NG ") + name)
    import step11_v27_render as v27
    lim = int(a[a.index("--sample") + 1]) if "--sample" in a else None
    bad = n = 0
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
    print(f"場面の検査: {n:,} 本中 NG {bad}")
    print(f"- 追加 {len(new):,} 本 → 総行数 {len(base) + len(new):,}")
    if not all(ok for _, ok in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    if not PLAN_PRE.exists():
        PLAN_PRE.write_bytes(PLAN.read_bytes())
    p27.write_rows(PLAN, base + new, COLS)
    p27.write_rows(MANIFEST, manifest, MANIFEST_COLS)
    c = Counter(r["split"] for r in new)
    D["n_mixes"] = len(adds); D["n_mixes_gate"] = sum(d["reason"] == "gate" for d in adds); D["n_mixes_floor"] = sum(d["reason"] == "floor" for d in adds)
    D["n_scenes"] = sum(d["add_scenes"] for d in adds); D["n_clips"] = len(new); D["n_shortfall"] = sum(d["shortfall"] for d in adds)
    D["h_csv_sha256"] = p27.sha256(h_csv); D["clip_ids"] = f"高さ 1 = {ID1}+i・高さ 2 = {ID2}+i（i < {D['n_scenes']}）・seed {SEED0}+i"
    P["N_ROWS_TOTAL_EXPECTED_V28"] = len(base) + len(new)
    P["_note_v28"] = f"2026-10-13: v28 の積み増し {len(new):,} 本（step10_v28_plan が書く）を足した総数 = N_ROWS_TOTAL_EXPECTED_V28 = {len(base) + len(new):,}。V27 の 50,574 は据え置き"
    PARAMS.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = [f"\n## v28 積み増し（2026-10-13・#64 追記・Opus R-1／M-1）: {len(new):,} 行を末尾に足した → 総行数 {len(base) + len(new):,}",
             f"- 元の plan（50,574 行）= assignment_v2final_pre_v28.csv sha256 {p27.sha256(PLAN_PRE)}", f"- 新しい plan sha256 {p27.sha256(PLAN)}",
             f"- H の実数 = {h_csv.name} sha256 {p27.sha256(h_csv)}・組 {len(adds)}（ゲート後 {D['n_mixes_gate']}・床 {D['n_mixes_floor']}）・届かない見込み {D['n_shortfall']} 組・train {c['train']:,} / val {c['val']:,}・clip {ID1:05d}〜（高さ 1）・{ID2:05d}〜（高さ 2）",
             f"- 組ごとの一覧 = manifest_v28_topup.csv sha256 {p27.sha256(MANIFEST)}", f"- 場面の検査: {n:,} 本 NG 0"]
    README.write_text(README.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    print("書いた:", PLAN, "/ 前の plan:", PLAN_PRE, "/ manifest:", MANIFEST, "/ params N_ROWS_TOTAL_EXPECTED_V28")
    return 0


if __name__ == "__main__":
    sys.exit(main())
