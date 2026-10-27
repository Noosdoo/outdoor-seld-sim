# -*- coding: utf-8 -*-
"""eval v3 plan = 確定評価セットの作り直し（本人決定 2026-10-13 15:06〜15:13・変更票 #65・凍結 r11）。

決めたこと: 旧 core 1,200 本（8 月以前の生成器の音・8 クラス）は測らない。新しいセット = **最新の学習データ（v27＋v28）と同じ場面の種類・同じ割合**を、
別の seed・別の番号で 6,000 本。マイクの高さは学習と同じ U(1.4, 2.1) m（1 本 1 高さ・高さの表を推論に渡す）。#54 の自転車 400 本は統合（自転車の表はこのセットの自転車の本から）。

作り方（v19bellpin の双子・#54 と同じ「同じ割当条件から新しい種で引き直す」）:
  1. 元 = 最終の学習 plan（train＋val・out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv）。層 = (scenario, mix)。層ごとの本数は元の割合に比例
     （最大剰余法で合計を N にそろえる・各層 最少 min_per_stratum）。層の中でどの行の条件を写すかは固定 seed のくじ（重複なし）。
  2. 写す条件 = scenario・mix・n_car・n_warnings・w1〜w3_class・danger_tier・scene_type・grammar。**新しい seed から引き直す** = 側（car_side・w*_side）・歩行（70%）・雨（v13 の規則）・
     高さ（U(1.4, 2.1)・1 本）・徐行（v11core の critical・step10_v14_plan.assign と同じ規則・横距離 cs_lat_m は step10_v29_plan の規則）。twin_of は空。
     2026-10-19 #70: 元は v29 の plan（62,102 行）。D19 の行は圏 × 帯（8 層）として別の層にし、速さと横距離を別の seed で帯の中から引く。層の最少本数は params EVAL_V3.d19_per_stratum（本人の判断・無ければ比例）。
  3. 番号 = eval_syn_clip00001〜（split = eval）。seed = 9300000000 + j（学習・検証の seed 9.10e9〜9.21e9・1.242e10 と重ならない・自動検査）。
  4. 出力 = out/dataset_outdoor_siren_v2final_eval_v3/plan/assignment_v2final.csv（生成器が読む名前）＋ assignment_eval_v3.csv（同じ中身）＋ manifest_eval_v3_strata.csv ＋ README_eval_v3.md。
     params に EVAL_V3（n_clips・seed・層の数・元の plan の sha256）を書く（--dry-run は書かない）。
封印: 生成（server_sde/eval_v3_gen.sbatch）のあと CONTRACT_eval_v3.json（全ファイルの sha256）を作り、確定評価の 1 回まで誰も開かない。学習の置き場に入れない（学習の入口の門が eval_ を弾く）。

使い方: PYTHONPATH=scripts:src python scripts/step10_eval_v3_plan.py [--src-plan <csv>] [--n 6000] [--dry-run] [--sample N]
       python scripts/step10_eval_v3_plan.py --selftest
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v27_plan as p27  # noqa: E402
import step10_v29_plan as p29  # noqa: E402  #70（2026-10-19）: 列（cs_lat_m・d19_*）・D10 の横距離・D19 の帯と圏の引き方

SRC_PLAN = p27.PLAN
PARAMS = p27.PARAMS
OUT_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final_eval_v3" / "plan"
PLAN_GEN = OUT_DIR / "assignment_v2final.csv"
PLAN_EVAL = OUT_DIR / "assignment_eval_v3.csv"
MANIFEST = OUT_DIR / "manifest_eval_v3_strata.csv"
README = OUT_DIR / "README_eval_v3.md"
COLS = p29.COLS
N_DEFAULT = 6000
MIN_PER_STRATUM = 2
SEED0 = 9300000000
SALT_PICK = 20261013            # 層の中で条件を写す行を選ぶくじ
SALT_SIDE = 20261015            # 側（car_side・w*_side）
SALT_Z = 20261016               # 高さ
RE_ID = re.compile(r"^eval_syn_clip\d{5}$")
COPY_COLS = ("scenario", "mix", "n_car", "n_warnings", "w1_class", "w2_class", "w3_class", "danger_tier", "scene_type", "grammar", "d19", "d19_tier", "d19_band")


def stratum(r: dict) -> tuple:
    """層 = (scenario, mix)。D19（注意・安全の徐行・#69 r3「eval v3 も同じ帯の定義」）は圏 × 帯ごとに別の層にして、帯の中を別の seed で引く。"""
    if r.get("d19") == "1":
        return (r["scenario"], f"d19:{r['d19_tier']}:b{r['d19_band']}")
    return (r["scenario"], r.get("mix", ""))


def read_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def allocate(strata: Counter, n: int, min_per: int, floors: dict | None = None) -> dict:
    """最大剰余法。各層 最少 min_per（元が min_per 未満の層は元の本数まで）。floors = 層ごとの最少本数（D19 の層・本人の判断）。"""
    total = sum(strata.values())
    floor = {k: int(n * v / total) for k, v in strata.items()}
    mins = {k: min(max(min_per, int((floors or {}).get(k, 0))), v) for k, v in strata.items()}
    base = {k: max(mins[k], floor[k]) for k in strata}
    frac = sorted(strata, key=lambda k: (-(n * strata[k] / total - floor[k]), k))
    rem = n - sum(base.values())
    if rem > 0:                                       # 端数を大きい層から 1 本ずつ
        for k in [k for k in frac if base[k] == floor[k]][:rem]:
            base[k] += 1
        for k in frac[: n - sum(base.values())]:
            base[k] += 1
    while sum(base.values()) > n:                     # 最少を置いた分だけ超えたら、一番大きい層（最少より上）から 1 本ずつ戻す
        cand = [k for k in base if base[k] > mins[k]]
        if not cand:                                  # 最少すら置けない小さい N → 最少なしの比例
            base = dict(floor)
            for k in frac[: n - sum(base.values())]:
                base[k] += 1
            break
        k = max(cand, key=lambda k: (base[k], k)); base[k] -= 1
    assert sum(base.values()) == n, (sum(base.values()), n)
    return base


def build(src: list, P: dict, n: int) -> tuple[list, list]:
    import step10_v14_plan as p14
    p13 = p27._p13(P)
    p14.CLOSE_SLOW_FRAC = float(P["D10_CLOSE_SLOW_FRAC"]); p14.SLOW_TIERS = ("critical",)
    walk = float(P["D4_WALK_FRAC"]); zr = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    by = defaultdict(list)
    for r in src:
        by[stratum(r)].append(r)
    strata = Counter({k: len(v) for k, v in by.items()})
    d19_min = (P.get("EVAL_V3") or {}).get("d19_per_stratum")                 # D19 の層（圏 × 帯 = 8）の最少本数（本人の判断・無ければ比例）
    floors = {k: int(d19_min) for k in strata if k[1].startswith("d19:")} if d19_min else None
    alloc = allocate(strata, n, MIN_PER_STRATUM, floors)
    rows, manifest = [], []
    j = 0
    for key in sorted(by):                              # 決定論的な並び（scenario, mix）
        pool = sorted(by[key], key=lambda r: r["clip_id"])
        rng = np.random.default_rng([SALT_PICK, hash_int(key[0] + "|" + key[1])])
        picks = rng.choice(len(pool), size=alloc[key], replace=False)
        first = j
        for idx in sorted(int(i) for i in picks):
            s = pool[idx]
            seed = SEED0 + j
            r = {c: s.get(c, "") for c in COPY_COLS}
            r.update({"clip_id": f"eval_syn_clip{j + 1:05d}", "split": "eval", "seed": str(seed), "twin_of": "", "h_copy": "1",
                      "motion": "walk" if np.random.default_rng([seed, 7]).random() < walk else "static",
                      "mic_z": f"{float(np.random.default_rng([seed, SALT_Z]).uniform(*zr)):.3f}",
                      "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "", **{c: "" for c in p29.NEW_COLS}})
            rs = np.random.default_rng([seed, SALT_SIDE])
            r["car_side"] = "L" if rs.random() < 0.5 else "R"
            nw = int(r["n_warnings"] or 0)
            for k in (1, 2, 3):
                r[f"w{k}_side"] = ("L" if rs.random() < 0.5 else "R") if k <= nw and r[f"w{k}_class"] else ""
            p27._rain(r, P, p13)                                        # 雨: 種類は params の割合・dba は params の D2_RAIN_DBA（#70 r13 の範囲）
            if s.get("d19") == "1":                                     # D19: 圏と帯は写し、速さと横距離を別の seed で帯の中から引く（#69 r3）
                rd = np.random.default_rng([seed, p29.SALT29, p29.SALT_D19])
                r.update({"d19": "1", "d19_tier": s["d19_tier"], "d19_band": s["d19_band"], "danger_tier": s["d19_tier"],
                          "d19_speed_kmh": p29._draw_in_band(rd, int(s["d19_band"])), "d19_lat_m": p29._draw_lat(rd, s["d19_tier"]), "grammar": "v29"})
            elif r["scenario"] == "v11core":
                r = p14.assign(r)                         # 徐行（critical の 50%）を新しい seed で引き直す（grammar は v14 に）
                if r["close_slow"] == "1":
                    r["cs_lat_m"] = p29.cs_lat_v29(seed)  # #70 r7: 至近の徐行の横距離 1.0〜1.5 m（cs_cpa_m は記録だけ・描画器は読まない）
            rows.append(r)
            j += 1
        manifest.append({"scenario": key[0], "mix": key[1], "n_src": strata[key], "n_eval": alloc[key], "clip_first": f"{first + 1:05d}", "clip_last": f"{j:05d}",
                         "share_src": f"{strata[key] / len(src):.4f}", "share_eval": f"{alloc[key] / n:.4f}"})
    return rows, manifest


def hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


P_CHECK: dict = {}


def check(src: list, rows: list, manifest: list, n: int) -> list:
    import step11_v27_render as v27
    global P_CHECK
    if not P_CHECK:
        P_CHECK = json.loads(PARAMS.read_text(encoding="utf-8"))
    msgs = []
    msgs.append((f"本数 {len(rows):,} = {n:,}", len(rows) == n))
    ids = [r["clip_id"] for r in rows]
    msgs.append(("番号 eval_syn_clip00001〜 がユニークで連番", len(set(ids)) == len(ids) and all(RE_ID.match(i) for i in ids) and ids == [f"eval_syn_clip{k + 1:05d}" for k in range(len(ids))]))
    src_seeds = {int(r["seed"]) for r in src}; ev_seeds = [int(r["seed"]) for r in rows]
    msgs.append((f"seed {SEED0}〜{SEED0 + n - 1} は学習・検証の seed（{min(src_seeds)}〜{max(src_seeds)}）と 1 つも重ならない", not (set(ev_seeds) & src_seeds) and len(set(ev_seeds)) == n))
    msgs.append(("split は全部 eval・h_copy 1・twin_of 空", all(r["split"] == "eval" and r["h_copy"] == "1" and r["twin_of"] == "" for r in rows)))
    c_src = Counter(stratum(r) for r in src); c_ev = Counter(stratum(r) for r in rows)
    # Astra RMIX14-03（2026-10-14・Opus）: 表示どおり「差は 1 本以内」を本当に見る。最少本数を置いた層（理想が最少未満）は別に「最少ちょうど」を見る
    ideal = {k: n * c_src[k] / len(src) for k in c_src}; floor_k = [k for k in c_src if ideal[k] < min(MIN_PER_STRATUM, c_src[k])]
    d19_min = (P_CHECK.get("EVAL_V3") or {}).get("d19_per_stratum")
    if d19_min:
        floor_k += [k for k in c_src if k[1].startswith("d19:") and ideal[k] < int(d19_min)]
    max_dev = max((abs(c_ev[k] - ideal[k]) for k in c_src if k not in floor_k), default=0.0)
    msgs.append((f"層の数 {len(c_ev)} = 元 {len(c_src)}・最少を置いていない層の差は 1 本以内（最大 {max_dev:.3f} 本・最大剰余法）", set(c_ev) == set(c_src) and max_dev <= 1.0))
    msgs.append((f"最少（{MIN_PER_STRATUM} 本・D19 の層は {d19_min or '比例'}）を置いた層 {len(floor_k)} 個は最少ちょうど（元がそれより少ない層は元の本数）",
                 all(c_ev[k] == min(int(d19_min) if (d19_min and k[1].startswith("d19:")) else MIN_PER_STRATUM, c_src[k]) for k in floor_k)))
    zs = np.array([float(r["mic_z"]) for r in rows])
    msgs.append((f"高さ 1.4〜2.1 m（中央値 {np.median(zs):.2f}）・歩行 {np.mean([r['motion'] == 'walk' for r in rows]):.1%}・雨 {np.mean([bool(r['rain']) for r in rows]):.1%}",
                 zs.min() >= 1.4 and zs.max() <= 2.1 and abs(np.mean([r["motion"] == "walk" for r in rows]) - 0.7) < 0.03 and abs(np.mean([bool(r["rain"]) for r in rows]) - 0.2) < 0.03))
    good = True
    for r in rows:
        if r["scenario"] == "v19mix":
            comp = v27.parse_mix(r["mix"]); nw = sum(comp[w] for w in v27.WARN)
            good &= (not v27.mix_ok(comp)) and int(r["n_car"]) == comp["car"] and int(r["n_warnings"]) == nw and all(r[f"w{k}_side"] in ("L", "R") for k in range(1, nw + 1))
        else:
            good &= r["mix"] == "" and all((r[f"w{k}_side"] in ("L", "R")) == bool(r[f"w{k}_class"]) for k in (1, 2, 3))
        good &= r["car_side"] in ("L", "R") and r["motion"] in ("walk", "static")
    msgs.append(("条件の整合（mix と n_car／n_warnings／w*・側は L/R）", good))
    cs = [r for r in rows if r["scenario"] == "v11core" and r["danger_tier"] == "critical" and r["scene_type"] in ("residential", "daily")]
    frac = sum(r["close_slow"] == "1" for r in cs) / max(len(cs), 1)
    msgs.append((f"徐行: v11core の critical（住宅街・日常）{len(cs)} 本のうち {frac:.1%}（規則 50% ± 8pt）", abs(frac - 0.5) < 0.08))
    cs1 = [r for r in rows if r["close_slow"] == "1"]
    msgs.append((f"D10 の {len(cs1)} 本は cs_lat_m 1.0〜1.5 m（#70 r7）・徐行でない行は空",
                 all(r["cs_lat_m"] != "" and 1.0 <= float(r["cs_lat_m"]) <= 1.5 for r in cs1) and all(r["cs_lat_m"] == "" for r in rows if r["close_slow"] != "1")))
    d19r = [r for r in rows if r["d19"] == "1"]; cells = Counter((r["d19_tier"], r["d19_band"]) for r in d19r)
    msgs.append((f"D19: {len(d19r)} 本・層 {dict(sorted(cells.items()))}・帯 = 速さの帯・圏 = 横距離の圏 = danger_tier・grammar v29",
                 len(cells) == 8 and all(p29.band_of(float(r["d19_speed_kmh"])) == int(r["d19_band"]) and p29.tier_of_lateral(float(r["d19_lat_m"])) == r["d19_tier"] == r["danger_tier"]
                             and r["grammar"] == "v29" and r["close_slow"] == "" for r in d19r)))
    D2 = {k: (float(v[0]), float(v[1])) for k, v in P_CHECK["D2_RAIN_DBA"].items()}
    msgs.append((f"雨の dba は params の範囲 {D2}", all(D2[r["rain"]][0] <= float(r["rain_dba"]) <= D2[r["rain"]][1] for r in rows if r["rain"])))
    msgs.append(("manifest と本数が一致", sum(m["n_eval"] for m in manifest) == n and all(c_ev[(m["scenario"], m["mix"])] == m["n_eval"] for m in manifest)))
    return msgs


def sample_check(rows: list, limit: int | None) -> int:
    """全部の行をサンプラで引けること（例外なし）。v19mix／v19motofar は check_row も通す。"""
    import step11_v27_render as v27
    import step11_v29_render as v29                                     # #70: 高さ・横距離・D10／D19 の差し替え（無いと cs_lat_m の無い旧の読み方になる）
    import step11_v25_render as v25
    v25.set_bellbike_rows_override([r for r in rows if r["scenario"] == "v19bellbike"], "eval_v3")   # D26 のベルの時刻（後 4.4%）をこの plan の行で割り当てる
    bad = n = 0
    for r in rows:
        if limit is not None and n >= limit:
            break
        n += 1
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            s = v27.sample_scene_v19(rr)
            if rr["scenario"] in (v27.SCENARIO, v27.SCEN_MOTOFAR):
                v27.check_row(s, rr)
            assert s["sources"] or rr["scenario"] == "v11core", "音源が無い"
            bad29 = v29.check_scene_v29(s, rr)
            assert not bad29, f"#70 の受け入れ: {bad29[:2]}"
        except Exception as e:  # noqa: BLE001
            bad += 1
            if bad <= 5:
                print("NG 場面", r["clip_id"], r["scenario"], e)
    print(f"場面の検査: {n:,} 本中 NG {bad}")
    return bad


def selftest() -> int:
    ok = []
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    a = allocate(Counter({"a": 1000, "b": 100, "c": 10, "d": 1}), 100, 2)
    ok.append((f"配分（最大剰余法・最少 2・超えた分は一番大きい層から戻す）: {a}", sum(a.values()) == 100 and a["d"] == 1 and a["c"] == 2 and a["a"] == 88 and a["b"] == 9))
    a2 = allocate(Counter({"a": 1000, "b": 100, "c": 10, "d": 1}), 5, 2)
    ok.append((f"配分（N が小さい → 最少なし）: {a2}", sum(a2.values()) == 5))
    src = read_rows(SRC_PLAN)
    rows, manifest = build(src, P, N_DEFAULT)
    msgs = check(src, rows, manifest, N_DEFAULT)
    for name, g in msgs:
        ok.append((name, g))
    rows2, _ = build(src, P, N_DEFAULT)
    ok.append(("決定論: 同じ入力で同じ plan", rows == rows2))
    ok.append(("選んだ条件は元の行から写している（scenario・mix・danger_tier が元に存在）", all(any(s["scenario"] == r["scenario"] and s.get("mix", "") == r["mix"] and s["danger_tier"] == r["danger_tier"] for s in src) for r in rows[:50])))
    ok.append(("場面の検査 120 本 NG 0", sample_check(rows, 120) == 0))
    for name, g in ok:
        print(("PASS " if g else "FAIL ") + name)
    nb = sum(not g for _, g in ok); print(f"selftest eval_v3: {len(ok) - nb}/{len(ok)}")
    return 1 if nb else 0


def main() -> int:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    dry = "--dry-run" in a
    n = int(a[a.index("--n") + 1]) if "--n" in a else N_DEFAULT
    src_plan = Path(a[a.index("--src-plan") + 1]) if "--src-plan" in a else SRC_PLAN
    P = json.loads(PARAMS.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    src = read_rows(src_plan)
    assert all(r["split"] in ("train", "val") for r in src), "元の plan に train/val 以外の行がある"
    rows, manifest = build(src, P, n)
    msgs = check(src, rows, manifest, n)
    for name, g in msgs:
        print(("OK " if g else "NG ") + name)
    lim = int(a[a.index("--sample") + 1]) if "--sample" in a else None
    bad = sample_check(rows, lim)
    c = Counter(r["scenario"] for r in rows)
    print("- 層（scenario）:", dict(sorted(c.items())))
    if not all(g for _, g in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert not PLAN_GEN.exists(), f"{PLAN_GEN} が既にある（作り直すなら名前を変える・封印した plan は上書きしない）"
    p27.write_rows(PLAN_GEN, rows, COLS)
    PLAN_EVAL.write_bytes(PLAN_GEN.read_bytes())
    with open(MANIFEST, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), lineterminator="\n"); w.writeheader()
        for m in manifest:
            w.writerow(m)
    E = OrderedDict([("adopt", True),
                     ("decided", "2026-10-13 15:06〜15:13 本人（確定評価も最新版に合わせて作り直す・6,000 本・高さ 1.4〜2.1 m・旧 core は測らない・#54 は統合）。変更票 #65・凍結 r11。2026-10-19 #70（r0〜r16）の値で作り直し（元 = v29 の plan 62,102 行・雨の dba は D2_RAIN_DBA・D10 は cs_lat_m・D19 は圏 × 帯の 8 層）"),
                     ("n_clips", n), ("seed_base", SEED0), ("clip_ids", f"eval_syn_clip00001〜{n:05d}（split = eval・1 本 1 高さ）"),
                     ("src_plan", src_plan.relative_to(ROOT).as_posix() if src_plan.is_relative_to(ROOT) else str(src_plan)), ("src_plan_sha256", p27.sha256(src_plan)), ("src_rows", len(src)),
                     ("n_strata", len(manifest)), ("min_per_stratum", MIN_PER_STRATUM), ("d19_per_stratum", (P.get("EVAL_V3") or {}).get("d19_per_stratum")), ("d19_eval", {f"{m['mix']}": m["n_eval"] for m in manifest if m["mix"].startswith("d19:")}), ("salts", {"pick": SALT_PICK, "side": SALT_SIDE, "z": SALT_Z, "motion": 7, "rain": "v13 の規則（params の SALTS.v13 と 19）", "close_slow": "step10_v14_plan.SALT14"}),
                     ("plan_sha256", p27.sha256(PLAN_GEN)), ("manifest_sha256", p27.sha256(MANIFEST)),
                     ("note", "層 = (scenario, mix) の割合を最終の学習 plan（train＋val）に比例。写す条件 = scenario・mix・n_car・n_warnings・w*_class・danger_tier・scene_type・grammar。引き直す = 側・歩行・雨・高さ・徐行。生成は server_sde/eval_v3_gen.sbatch（学習の置き場に入れない・CONTRACT で封印・開くのは確定評価の 1 回）")])
    import step11_v25_render as v25
    bb = v25.write_bellbike_manifest(OUT_DIR / "manifest_v26_bellbike.csv", PLAN_GEN)          # D26: この plan のベルつき走行の時刻（後 = 4.4% に比例）
    E["bellbike_manifest"] = f"manifest_v26_bellbike.csv（{bb['n']} 本・後 {bb['after']}・前 {bb['before']}・sha256 {p27.sha256(OUT_DIR / 'manifest_v26_bellbike.csv')[:16]}）"
    P["EVAL_V3"] = E
    PARAMS.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    L = [f"# eval v3 plan（確定評価セットの作り直し・2026-10-13・変更票 #65・凍結 r11）: {n:,} 本", "",
         f"- 元の plan = {E['src_plan']}（{len(src):,} 行）sha256 {E['src_plan_sha256']}", f"- plan sha256 {E['plan_sha256']}（assignment_v2final.csv = assignment_eval_v3.csv）", f"- 層 {len(manifest)}・manifest sha256 {E['manifest_sha256']}",
         f"- seed {SEED0}〜{SEED0 + n - 1}・番号 eval_syn_clip00001〜{n:05d}・高さ U(1.4, 2.1)・歩行 70%・雨 20%（v13 の規則）・徐行 critical 50%（v14 の規則・横距離 cs_lat_m 1.0〜1.5 m）・D19 は圏 × 帯の 8 層（#69 r3・#70）・雨の dba は D2_RAIN_DBA（#70 r13）", f"- 場面の検査 {'全数' if lim is None else lim} NG 0", ""]
    L += [f"- {m['scenario']} {m['mix'] or '-'}: 元 {m['n_src']:,} → {m['n_eval']}（clip {m['clip_first']}〜{m['clip_last']}）" for m in manifest]
    README.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("書いた:", PLAN_GEN, "/", PLAN_EVAL, "/", MANIFEST, "/ params EVAL_V3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
