# -*- coding: utf-8 -*-
"""step10_v11_eval_plan.py — v11評価拡張（実装③）の割当表ジェネレータ＋検算。

仕様の正= md/design/v11評価拡張_実装仕様_2026-07-28.md（事前登録）。
増量9セット（既存レシピ流用）＋新種N1〜N7（各150、新サンプラ）＝計3,246本。
スキーマはv11の14列（scene_type="eval"固定、n_car=実台数）。scenarioトークン:
  既存流用: carfree_siren / v11core / crossing_wait / bell_overtake / backup_reverse /
            siren_worstnoise / intersection_siren / traffic2,3 / probe_*
  新規:     n1_blind / n2_ev / n3_parking / n4_fast_siren / n5_downtown /
            n6_overtake / n7_pullout（幾何・レベルの正= step11_v11_eval_render.py）
N3の行エンコード: n_warnings∈{2,3}=バック音の本数（w1=w2=backup_beep、3本目は
サンプラがseedから引く。同一側=w1_side=w2_side=car_side）。

シード: GLOBAL_SEED=20260727（coreと同基底）、予約帯 +210000〜（セット別下表）。
既存全plan（core含む）との衝突ゼロを機械検証。

出力: out/dataset_outdoor_siren_v11_eval/plan/assignment_{set}.csv ＋ 検算レポート
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v9_plan as v9  # noqa: E402
import step10_v11_plan as v11  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v11_eval" / "plan"
SEED = v11.GLOBAL_SEED  # 20260727
BASE = SEED * 613

OFFSETS = {"halluc600": 210000, "safe600": 211000, "s1_200": 212000,
           "s2_100": 213000, "s3_100": 214000, "s5_200": 215000,
           "cross100": 216000, "multi200": 217000, "probe96": 218000,
           "n1": 220000, "n2": 221000, "n3": 222000, "n4": 223000,
           "n5": 224000, "n6": 225000, "n7": 226000}


def row(clip_id, motion, n_warn, w1, s1, w2, s2, tier, car, scen, n_car, seed):
    return {"clip_id": clip_id, "split": clip_id.split("_")[0], "motion": motion,
            "n_warnings": n_warn, "w1_class": w1, "w1_side": s1,
            "w2_class": w2, "w2_side": s2, "danger_tier": tier,
            "car_side": car, "scenario": scen, "seed": seed,
            "scene_type": "eval", "n_car": n_car}


def build_halluc600():
    b = BASE + OFFSETS["halluc600"]
    return [row(f"fold4_room1_mix{i+1:04d}",
                "static" if i % 2 == 0 else "walk", 1,
                "siren", v9.SIDES[i % 2], "", "", "na", "",
                "carfree_siren", 0, b + i) for i in range(600)]


def build_safe600():
    b = BASE + OFFSETS["safe600"]
    return [row(f"fold4_room2_mix{i+1:04d}",
                "static" if i % 2 == 0 else "walk", 0,
                "", "", "", "", "safe", v9.SIDES[(i // 2) % 2],
                "v11core", 1, b + i) for i in range(600)]


def build_scn2_more():
    """S1/S2/S3/S5の増量（build_scenario2と同式の行、本数と部屋のみ変更）。"""
    out = {}
    spec = [("s1_200", "fold5_room1", "crossing_wait", "walk", 1, "crossing",
             "caution", True, 200),
            ("s2_100", "fold5_room2", "bell_overtake", "walk", 1, "bike_bell",
             "na", False, 100),
            ("s3_100", "fold5_room3", "backup_reverse", "half", 1, "backup_beep",
             "tier2", True, 100),
            ("s5_200", "fold5_room5", "siren_worstnoise", "half", 1, "siren",
             "na", False, 200)]
    for key, room, scen, motion_mode, n_warn, wcls, tier_mode, has_car, n in spec:
        b = BASE + OFFSETS[key]
        rows = []
        for i in range(n):
            motion = ("walk" if motion_mode == "walk"
                      else ("static" if i < n // 2 else "walk"))
            tier = ("na" if tier_mode == "na"
                    else ("caution" if tier_mode == "caution"
                          else ("critical" if i % 2 == 0 else "caution")))
            rows.append(row(f"{room}_mix{i+1:04d}", motion, n_warn,
                            wcls, v9.SIDES[i % 2] if wcls else "", "", "",
                            tier, v9.SIDES[(i // 2) % 2] if has_car else "",
                            scen, 1 if has_car else 0, b + i))
        out[key] = rows
    return out


def build_cross100():
    b = BASE + OFFSETS["cross100"]
    return [row(f"fold5_room9_mix{i+1:04d}", "walk", 1,
                "siren", v9.SIDES[i % 2], "", "", "na", "",
                "intersection_siren", 0, b + i) for i in range(100)]


def build_multi200():
    """複数車増量（build_v10aと同式、2台100/3台100）。"""
    b = BASE + OFFSETS["multi200"]
    rows, warn_k = [], 0
    for i in range(200):
        ncars = 2 if i < 100 else 3
        motion = "static" if i % 2 == 0 else "walk"
        has_warn = (i // 2) % 2 == 0
        w = v9.WARN_CLASSES[warn_k % 5] if has_warn else ""
        if w:
            warn_k += 1
        rows.append(row(f"fold8_room2_mix{i+1:04d}", motion,
                        1 if w else 0, w, v9.SIDES[i % 2] if w else "", "", "",
                        v9.TIERS[i % 3], v9.SIDES[(i // 3) % 2],
                        f"traffic{ncars}", ncars, b + i))
    return rows


def build_probe96():
    """プローブ増量（build_probeと同式、6クラス×16=静止8/歩行8）。"""
    b = BASE + OFFSETS["probe96"]
    rows = []
    classes = v9.WARN_CLASSES + ["car_drive"]
    per = 16
    for ci, cls in enumerate(classes):
        for j in range(per):
            k = ci * per + j
            rows.append(row(
                f"fold9_room2_mix{k+1:04d}",
                "static" if j < per // 2 else "walk",
                0 if cls == "car_drive" else 1,
                "" if cls == "car_drive" else cls, v9.SIDES[j % 2],
                "", "",
                "safe" if cls == "car_drive" else "na",
                v9.SIDES[j % 2] if cls == "car_drive" else "",
                f"probe_{cls}", 1 if cls == "car_drive" else 0, b + k))
    return rows


def build_n_sets():
    """N1〜N7（各150本）。幾何・レベルの連続値はレンダラがseedから引く。"""
    out = {}
    N = 150
    pairs = [(a, c) for i, a in enumerate(v9.WARN_CLASSES)
             for c in v9.WARN_CLASSES[i + 1:]]  # N5用の異クラス10ペア循環
    for k, (key, room, scen) in enumerate([
            ("n1", "fold6_room1", "n1_blind"),
            ("n2", "fold6_room2", "n2_ev"),
            ("n3", "fold6_room3", "n3_parking"),
            ("n4", "fold6_room4", "n4_fast_siren"),
            ("n5", "fold6_room5", "n5_downtown"),
            ("n6", "fold6_room6", "n6_overtake"),
            ("n7", "fold6_room7", "n7_pullout")]):
        b = BASE + OFFSETS[key]
        rows = []
        for i in range(N):
            motion = "static" if i % 2 == 0 else "walk"
            side = v9.SIDES[i % 2]
            car2 = v9.SIDES[(i // 2) % 2]
            if key in ("n1", "n2"):        # 車1台・警告なし・tier3層循環
                r = row(f"{room}_mix{i+1:04d}", motion, 0, "", "", "", "",
                        v9.TIERS[i % 3], car2, scen, 1, b + i)
            elif key == "n3":              # バック音2-3本(同一側)＋徐行車1台
                nbeep = 2 if i % 2 == 0 else 3
                r = row(f"{room}_mix{i+1:04d}", motion, nbeep,
                        "backup_beep", side, "backup_beep", side,
                        v9.TIERS[i % 3], side, scen, 1, b + i)
            elif key in ("n4", "n6"):      # 警告1・車なし
                wcls = "siren" if key == "n4" else "bike_bell"
                r = row(f"{room}_mix{i+1:04d}", motion, 1, wcls, side,
                        "", "", "na", "", scen, 0, b + i)
            elif key == "n5":              # 車3台＋異クラス警告2＋高騒音
                w1, w2 = pairs[i % len(pairs)]
                r = row(f"{room}_mix{i+1:04d}", motion, 2, w1, side,
                        w2, v9.SIDES[(i + 1) % 2], v9.TIERS[i % 3], car2,
                        scen, 3, b + i)
            else:                          # n7: 停車→発進の車1台（幾何はサンプラ）
                r = row(f"{room}_mix{i+1:04d}", motion, 0, "", "", "", "",
                        "na", car2, scen, 1, b + i)
            rows.append(r)
        out[key] = rows
    return out


def build_all() -> dict:
    sets = {"halluc600": build_halluc600(), "safe600": build_safe600()}
    sets.update(build_scn2_more())
    sets["cross100"] = build_cross100()
    sets["multi200"] = build_multi200()
    sets["probe96"] = build_probe96()
    sets.update(build_n_sets())
    return sets


def load_existing_seeds() -> tuple:
    seeds, files = set(), 0
    for f in sorted((ROOT / "out").glob("dataset_outdoor_siren_*/plan/assignment_*.csv")):
        if "v11_eval" in f.as_posix():
            continue
        files += 1
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                seeds.add(int(r["seed"]))
    return seeds, files


def main() -> int:
    s1, s2 = build_all(), build_all()
    csv1 = {k: v11.to_csv(v) for k, v in s1.items()}
    assert {k: v11.to_csv(v) for k, v in s2.items()} == csv1, "determinism NG"

    allrows = [r for v in s1.values() for r in v]
    assert len(allrows) == 3246, len(allrows)
    ids = [r["clip_id"] for r in allrows]
    seeds = [r["seed"] for r in allrows]
    assert len(set(ids)) == len(ids) and len(set(seeds)) == len(seeds)
    existing, nfiles = load_existing_seeds()
    hit = set(seeds) & existing
    assert not hit, f"seed collision: {sorted(hit)[:3]}"
    assert min(seeds) >= BASE + 210000, "予約帯(+210000〜)の外"

    # セット別検算
    want_n = {"halluc600": 600, "safe600": 600, "s1_200": 200, "s2_100": 100,
              "s3_100": 100, "s5_200": 200, "cross100": 100, "multi200": 200,
              "probe96": 96, "n1": 150, "n2": 150, "n3": 150, "n4": 150,
              "n5": 150, "n6": 150, "n7": 150}
    for k, n in want_n.items():
        assert len(s1[k]) == n, (k, len(s1[k]))
    assert all(r["w1_class"] == "siren" and r["n_car"] == 0
               for r in s1["halluc600"])
    assert all(r["danger_tier"] == "safe" and r["n_car"] == 1
               and r["n_warnings"] == 0 for r in s1["safe600"])
    assert sum(1 for r in s1["multi200"] if r["scenario"] == "traffic2") == 100
    ev = Counter()
    for r in s1["probe96"]:
        ev[r["scenario"]] += 1
    assert all(v == 16 for v in ev.values()) and len(ev) == 6
    assert all(r["w1_side"] == r["w2_side"] == r["car_side"]
               for r in s1["n3"])                      # 駐車場=同一側
    n3_beeps = Counter(r["n_warnings"] for r in s1["n3"])
    assert n3_beeps == {2: 75, 3: 75}
    assert all(r["n_car"] == 3 and r["w1_class"] != r["w2_class"]
               for r in s1["n5"])

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# v11評価拡張 割当表 検算レポート（step10_v11_eval_plan.py 自動生成）", "",
             f"- GLOBAL_SEED={SEED} / 予約帯+210000〜 / 総行数 {len(allrows)}",
             f"- 既存{len(existing)}シード（{nfiles}ファイル、v11 core含む）との衝突ゼロ",
             "- 決定論: 2回構築で全セットCSV一致 / ID・seed一意", ""]
    for k in want_n:
        (PLAN_DIR / f"assignment_{k}.csv").write_text(csv1[k], encoding="utf-8")
        md5 = hashlib.md5(csv1[k].encode()).hexdigest()
        lines.append(f"- {k}: {want_n[k]}行 md5={md5}")
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print("\n".join(lines))
    print("\nALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
