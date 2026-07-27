# -*- coding: utf-8 -*-
"""step10_v11_plan.py — v11 core 割当表ジェネレータ＋全数検算
（設計= md/design/v11データセット拡張_設計書_2026-07-27.md §1、引き継ぎ= 同 v11_Fable引き継ぎ）

v11 core = 7,200本（20h、DCASE2022 Task3公式合成相当）。fold1 4800 / fold2 1200 / fold3 1200。
学習分布=「自然頻度寄せ＋重要セルのフロア」（本人決定 2026-07-27、設計書§8）:
  - シーン種別 住宅40/生活40/幹線20 が車の台数 n_car を決める（自然頻度の骨）。
    種別ごとの台数分布は下記 NCAR_DIST。混合後の周辺分布は厳密に
    {0:18%, 1:47%, 2:23%, 3:12%}（設計書§1.2(1)の目標表と一致）
  - 警告音 n_warn = {0:45%, 1:40%, 2:15%}（検出対象ゆえ意図的オーバーサンプル=卒論明記）
  - フロア（自然頻度で薄くなっても割り込まない絶対数、check_coreでassert）:
    警告のみ・車なし(car0,warn>=1) fold1>=400 / 純静穏(car0,warn0)>=300 / 複数車(car>=2)>=1500
    （fold2/3 は規模比0.25でスケール: >=100 / >=75 / >=375）
  - 同一クラス警告×2（siren/backup_beep/bike_bell、v9.2/v10.2のC枠と同じ3クラス）を
    2音源ペア循環に組み込み（13ペア型）→ v10.2 C枠の役割をcoreに吸収（設計書§3）
  - 複数車(n_car=2/3)と警告のみ(n_car=0)をcoreに統合 → v10.2 A/B枠の役割を吸収

スキーマ: v9の12列 + scene_type + n_car。レンダラはcsv.DictReaderで読むため追加列は
後方互換。scenarioトークンは全行 'v11core'（step11_v11_render.py が専用サンプラで解釈。
乱数消費順は ①マイク ②車n_car台 ③w1 ④w2 ⑤暗騒音。車のt_cpaは n_car==1 は
CAR_TCPA(coreの「接近の頭を収める」規約)、n_car>=2 は U(4,9)（v10.2 A枠=traffic実証済み
regime）とする——ここはstep11_v11_render.py実装時の仕様として本ヘッダを正とする）。

シード: GLOBAL_SEED*613 + 200000 + 通し番号。既存全plan（v9系20260717/v10系20260721/
v10.2系20260722/対照系20260723）のシードをv11のオフセット空間へ写像すると最大でも
約9.0万に収まるため、+200000以上をv11専用帯域として予約する（v11の今後の評価枠は
+210000から刻む）。念のためディスク上の全assignment_*.csvと突合して衝突ゼロを機械検証。

出力: out/dataset_outdoor_siren_v11/plan/
  - assignment_core.csv (7200行)
  - plan_check_report.md
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step10_v9_plan as v9  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v11" / "plan"

GLOBAL_SEED = 20260727
CORE_OFFSET = 200000
SPLITS = {"fold1": 4800, "fold2": 1200, "fold3": 1200}

SCENE_TYPES = ["residential", "daily", "arterial"]  # 住宅街 / 生活道路 / 幹線沿い
FRAC_SCENE = {"residential": 0.40, "daily": 0.40, "arterial": 0.20}
NCAR_DIST = {
    "residential": {0: 0.40, 1: 0.50, 2: 0.10, 3: 0.00},
    "daily":       {0: 0.05, 1: 0.55, 2: 0.30, 3: 0.10},
    "arterial":    {0: 0.00, 1: 0.25, 2: 0.35, 3: 0.40},
}
FRAC_W = {0: 0.45, 1: 0.40, 2: 0.15}

SAME_CLASSES = ["siren", "backup_beep", "bike_bell"]
PAIR_TYPES = ([(a, b) for i, a in enumerate(v9.WARN_CLASSES)
               for b in v9.WARN_CLASSES[i + 1:]]
              + [(c, c) for c in SAME_CLASSES])  # 10異クラス + 3同一 = 13型

FLOORS = {"fold1": (400, 300, 1500), "fold2": (100, 75, 375), "fold3": (100, 75, 375)}

COLUMNS = v9.COLUMNS + ["scene_type", "n_car"]


def to_csv(rows: list) -> str:
    lines = [",".join(COLUMNS)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in COLUMNS))
    return "\n".join(lines) + "\n"


def spread_sequence(counts: dict, order: list) -> list:
    """各要素を全長に均等散布した並び。どの連続区間を切り出しても各要素の数が
    比例±1に収まる（低不一致インターリーブ）。層順に消費することで
    クラス×(scene,n_car)の交絡を機械的に抑える。"""
    items = []
    for ci, c in enumerate(order):
        k = counts.get(c, 0)
        for j in range(k):
            items.append(((j + 0.5) / k, ci))
    items.sort()
    return [order[ci] for _, ci in items]


def alloc_levels(strata: list, sizes: list, totals: dict) -> dict:
    """警告レベル{0,1,2}を層別に整数割付。層内は比例（floor基準+端数順の+1）、
    列合計は totals(=exact_counts) と厳密一致。決定論。
    貪欲は必ず完了する: 行残と列残が同時に残る限りその交点セルに常に+1できる。"""
    levels = sorted(totals)
    size_of = dict(zip(strata, sizes))
    a = {(s, l): int(np.floor(size_of[s] * FRAC_W[l])) for s in strata for l in levels}
    row_def = {s: size_of[s] - sum(a[(s, l)] for l in levels) for s in strata}
    col_def = {l: totals[l] - sum(a[(s, l)] for s in strata) for l in levels}
    assert all(v >= 0 for v in row_def.values()) and all(v >= 0 for v in col_def.values())
    frac = {(s, l): size_of[s] * FRAC_W[l] - a[(s, l)] for s in strata for l in levels}
    order = sorted(frac, key=lambda k: (-frac[k], strata.index(k[0]), k[1]))
    guard = 0
    while sum(row_def.values()) > 0:
        guard += 1
        assert guard < 100, "alloc_levels: no convergence"
        for s, l in order:
            if row_def[s] > 0 and col_def[l] > 0:
                a[(s, l)] += 1
                row_def[s] -= 1
                col_def[l] -= 1
    return a


def build_motion_group(split: str, motion: str, m: int, rng: np.random.Generator,
                       extra_classes: list) -> list:
    """1つの 分割×歩行 グループ（m本）の行を構築（clip_id/seedは未定）。"""
    # 1) シーン種別 → n_car の層（厳密数）
    n_scene = v9.exact_counts(m, FRAC_SCENE)
    strata, sizes = [], []
    for sc in SCENE_TYPES:
        ncar_counts = v9.exact_counts(n_scene[sc], NCAR_DIST[sc])
        for nc in sorted(ncar_counts):
            if ncar_counts[nc] > 0:
                strata.append((sc, nc))
                sizes.append(ncar_counts[nc])
    assert sum(sizes) == m

    # 2) 警告レベルの層別割付（層内比例・列合計厳密）
    C = v9.exact_counts(m, FRAC_W)
    alloc = alloc_levels(strata, sizes, C)

    # 3) 2音源ペア列（13型±1）と1音源クラス列（クラス別イベント合計が±1になる充当）
    pair_counts = v9.exact_counts(C[2], {pt: 1.0 / len(PAIR_TYPES) for pt in PAIR_TYPES})
    pair_seq = spread_sequence(pair_counts, PAIR_TYPES)
    pair_class_cnt = Counter()
    for x, y in pair_seq:
        pair_class_cnt[x] += 1
        pair_class_cnt[y] += 1
    total_events = C[1] + 2 * C[2]
    base_t, r = divmod(total_events, 5)
    assert len(extra_classes) == r, (split, motion, r, extra_classes)
    target = {c: base_t for c in range(5)}
    for c in extra_classes:
        target[c] += 1
    one_counts = {v9.WARN_CLASSES[c]: target[c] - pair_class_cnt[v9.WARN_CLASSES[c]]
                  for c in range(5)}
    assert all(v >= 0 for v in one_counts.values()), (split, motion, one_counts)
    assert sum(one_counts.values()) == C[1]
    one_seq = spread_sequence(one_counts, v9.WARN_CLASSES)

    # 4) 層順に配布（層内のレベル位置はシャッフル、組成列は層順の連続区間を消費）
    rows, p1, p2 = [], 0, 0
    for (sc, nc), size in zip(strata, sizes):
        levels = ([0] * alloc[((sc, nc), 0)] + [1] * alloc[((sc, nc), 1)]
                  + [2] * alloc[((sc, nc), 2)])
        assert len(levels) == size
        levels = [levels[i] for i in rng.permutation(size)]
        for lv in levels:
            if lv == 0:
                w1 = w2 = ""
            elif lv == 1:
                w1, w2 = one_seq[p1], ""
                p1 += 1
            else:
                w1, w2 = pair_seq[p2]
                p2 += 1
            rows.append({"split": split, "motion": motion, "scenario": "v11core",
                         "scene_type": sc, "n_car": nc, "n_warnings": lv,
                         "w1_class": w1, "w2_class": w2})
    assert p1 == C[1] and p2 == C[2]

    # 5) 危険層: 車あり行のみ、(scene,n_car,組成)層順のサイクリック → 層内±1
    def stratum_key(row):
        return (row["scene_type"], row["n_car"], row["n_warnings"],
                row["w1_class"], row["w2_class"])

    car_rows = sorted([r for r in rows if r["n_car"] >= 1], key=stratum_key)
    off = int(rng.integers(3))
    for i, row in enumerate(car_rows):
        row["danger_tier"] = v9.TIERS[(off + i) % 3]

    # 6) 車側（1台目）: (危険層,層)順でL/R交互 → 側⊥層・側⊥組成とも±1
    off = int(rng.integers(2))
    for i, row in enumerate(sorted(car_rows, key=lambda r: (r["danger_tier"],) + stratum_key(r))):
        row["car_side"] = v9.SIDES[(off + i) % 2]
    for r in rows:
        if r["n_car"] == 0:
            r["danger_tier"] = "na"
            r["car_side"] = ""

    # 7) 警告音の側: クラスごとにイベントを集めて交互 → クラス別 |L-R|<=1
    ev_by_class = defaultdict(list)
    for row in rows:
        if row["n_warnings"] >= 1:
            ev_by_class[row["w1_class"]].append((row, "w1_side"))
        if row["n_warnings"] == 2:
            ev_by_class[row["w2_class"]].append((row, "w2_side"))
    for c, evs in sorted(ev_by_class.items()):
        off = int(rng.integers(2))
        for i, (row, key) in enumerate(evs):
            row[key] = v9.SIDES[(off + i) % 2]
    for row in rows:
        row.setdefault("w1_side", "")
        row.setdefault("w2_side", "")
    return rows


def build_core(splits: dict = SPLITS, seed: int = GLOBAL_SEED) -> list:
    rows = []
    for split, n in splits.items():
        rng = np.random.default_rng(seed * 2741 + {"fold1": 1, "fold2": 2, "fold3": 3}[split])
        m = n // 2
        C = v9.exact_counts(m, FRAC_W)
        r = (C[1] + 2 * C[2]) % 5
        class_order = list(rng.permutation(5))
        extras = {"static": [class_order[i % 5] for i in range(r)],
                  "walk": [class_order[(r + i) % 5] for i in range(r)]}
        for motion in ["static", "walk"]:
            rows += build_motion_group(split, motion, m, rng, extras[motion])
        # グループ構築順とmix番号の相関を断つ最終シャッフル
        split_rows = [x for x in rows if x["split"] == split]
        perm = rng.permutation(len(split_rows))
        rows = [x for x in rows if x["split"] != split] + [split_rows[i] for i in perm]
    idx = 0
    for split in splits:
        for i, row in enumerate([x for x in rows if x["split"] == split]):
            row["clip_id"] = f"{split}_room1_mix{i + 1:04d}"
            row["seed"] = seed * 613 + CORE_OFFSET + idx
            idx += 1
    return rows


# ---------------------------------------------------------------- 検算 ----

def check_core(rows: list, splits: dict = SPLITS) -> list:
    rep = []
    for split, n in splits.items():
        sr = [r for r in rows if r["split"] == split]
        assert len(sr) == n, (split, len(sr))
        m = n // 2
        for motion in ["static", "walk"]:
            mr = [r for r in sr if r["motion"] == motion]
            assert len(mr) == m, (split, motion, len(mr))
            # シーン種別・n_car の厳密数
            n_scene = v9.exact_counts(m, FRAC_SCENE)
            got_sc = Counter(r["scene_type"] for r in mr)
            assert dict(got_sc) == {k: v for k, v in n_scene.items() if v}, (split, motion, got_sc)
            for sc in SCENE_TYPES:
                want_nc = v9.exact_counts(n_scene[sc], NCAR_DIST[sc])
                got_nc = Counter(r["n_car"] for r in mr if r["scene_type"] == sc)
                assert dict(got_nc) == {k: v for k, v in want_nc.items() if v}, \
                    (split, motion, sc, got_nc, want_nc)
            # 警告レベルの厳密数（グループ合計）＋ 層内比例（±2）
            got_w = Counter(r["n_warnings"] for r in mr)
            want_w = v9.exact_counts(m, FRAC_W)
            assert dict(got_w) == want_w, (split, motion, got_w, want_w)
            worst_dev = 0.0
            for (sc, nc), grp in _group_by(mr, lambda r: (r["scene_type"], r["n_car"])).items():
                for lv in (0, 1, 2):
                    got = sum(1 for r in grp if r["n_warnings"] == lv)
                    dev = abs(got - len(grp) * FRAC_W[lv])
                    worst_dev = max(worst_dev, dev)
                    assert dev <= 2.0, (split, motion, sc, nc, lv, got, len(grp))
            # クラス別イベント: グループ合計±1（extras設計）＋ n_car層内で偏らない（±4）
            ev = v9.class_events(mr)
            vals = [ev[c] for c in v9.WARN_CLASSES]
            assert max(vals) - min(vals) <= 1, (split, motion, ev)
            worst_cls = 0.0
            for nc, grp in _group_by(mr, lambda r: r["n_car"]).items():
                ev_g = v9.class_events(grp)
                n_ev_g = sum(ev_g.values())
                for c in v9.WARN_CLASSES:
                    dev = abs(ev_g[c] - n_ev_g / 5.0)
                    worst_cls = max(worst_cls, dev)
                    assert dev <= 4.0, (split, motion, nc, c, ev_g)
            # 危険層: 車あり行の(scene,n_car)層内±1、車なし行はna
            for (sc, nc), grp in _group_by(
                    [r for r in mr if r["n_car"] >= 1],
                    lambda r: (r["scene_type"], r["n_car"])).items():
                tc = Counter(r["danger_tier"] for r in grp)
                tv = [tc[t] for t in v9.TIERS]
                assert max(tv) - min(tv) <= 1, (split, motion, sc, nc, tc)
            # 車側: 車あり行で|L-R|<=1、危険層×側も±1
            car_mr = [r for r in mr if r["n_car"] >= 1]
            scnt = Counter(r["car_side"] for r in car_mr)
            assert abs(scnt["L"] - scnt["R"]) <= 1, (split, motion, scnt)
            for t, grp in _group_by(car_mr, lambda r: r["danger_tier"]).items():
                sc2 = Counter(r["car_side"] for r in grp)
                assert abs(sc2["L"] - sc2["R"]) <= 1, (split, motion, t, sc2)
            # tier×警告レベル: 層順サイクリックの残差が層数分だけ蓄積する項。
            # v9のcross-spreadと同じsqrt(規模)スケール許容（基準=v9本来の640/240/240）
            tol_tw = math.ceil(3 * math.sqrt(n / {"fold1": 640, "fold2": 240,
                                                  "fold3": 240}[split]))
            worst_tw = 0
            for lv in (0, 1, 2):
                tc = Counter(r["danger_tier"] for r in car_mr if r["n_warnings"] == lv)
                tv = [tc[t] for t in v9.TIERS]
                worst_tw = max(worst_tw, max(tv) - min(tv))
            assert worst_tw <= tol_tw, (split, motion, worst_tw, tol_tw)
            # 警告音の側: クラス別|L-R|<=1
            evt = Counter()
            for r in mr:
                for ck, sk in (("w1_class", "w1_side"), ("w2_class", "w2_side")):
                    if r[ck]:
                        evt[(r[ck], r[sk])] += 1
            for c in v9.WARN_CLASSES:
                assert abs(evt[(c, "L")] - evt[(c, "R")]) <= 1, (split, motion, c, evt)
            rep.append(f"- {split}/{motion}: clips={len(mr)} n_warn={dict(sorted(got_w.items()))} "
                       f"class_ev={{{', '.join(f'{c}:{ev[c]}' for c in v9.WARN_CLASSES)}}} "
                       f"warnレベル層内偏差max={worst_dev:.1f} クラス×n_car偏差max={worst_cls:.1f} "
                       f"tier×warn差={worst_tw}(許容{tol_tw})")
        # フロア（split単位、設計書§1.2(3)）
        f_warnonly = sum(1 for r in sr if r["n_car"] == 0 and r["n_warnings"] >= 1)
        f_quiet = sum(1 for r in sr if r["n_car"] == 0 and r["n_warnings"] == 0)
        f_multi = sum(1 for r in sr if r["n_car"] >= 2)
        lo1, lo2, lo3 = FLOORS[split]
        assert f_warnonly >= lo1 and f_quiet >= lo2 and f_multi >= lo3, \
            (split, f_warnonly, f_quiet, f_multi, FLOORS[split])
        ncar_cnt = Counter(r["n_car"] for r in sr)
        rep.append(f"- {split}: n_car={dict(sorted(ncar_cnt.items()))} "
                   f"({'/'.join(f'{100*ncar_cnt[k]/n:.0f}%' for k in range(4))}) | "
                   f"フロア実測: 警告のみ{f_warnonly}(>={lo1}) 純静穏{f_quiet}(>={lo2}) "
                   f"複数車{f_multi}(>={lo3}) ✓")
        # 分割合計でもクラス±1
        ev = v9.class_events(sr)
        vals = [ev[c] for c in v9.WARN_CLASSES]
        assert max(vals) - min(vals) <= 1, (split, ev)
    # 整合: 2音源の正準順（同一クラスはsiren/beep/bellのみ許可）・car0行の空欄・ID一意
    same_cnt = Counter()
    for r in rows:
        if r["n_warnings"] == 2:
            if r["w1_class"] == r["w2_class"]:
                assert r["w1_class"] in SAME_CLASSES, r
                same_cnt[r["w1_class"]] += 1
            else:
                assert (v9.WARN_CLASSES.index(r["w1_class"])
                        < v9.WARN_CLASSES.index(r["w2_class"])), r
        if r["n_car"] == 0:
            assert r["car_side"] == "" and r["danger_tier"] == "na", r
        else:
            assert r["car_side"] in v9.SIDES and r["danger_tier"] in v9.TIERS, r
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    rep.append(f"- 同一クラス警告×2: {dict(same_cnt)}（計{sum(same_cnt.values())}本、"
               f"v10.2 C枠112本の役割をcoreに吸収）")
    return rep


def _group_by(rows, keyfn):
    d = defaultdict(list)
    for r in rows:
        d[keyfn(r)].append(r)
    return d


def load_existing_seeds() -> set:
    """ディスク上の既存全plan（v9系/v10系/対照系）のシードを収集。"""
    seeds = set()
    files = []
    for f in sorted((ROOT / "out").glob("dataset_outdoor_siren_*/plan/assignment_*.csv")):
        if "dataset_outdoor_siren_v11" in f.as_posix():
            continue
        files.append(f)
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seeds.add(int(row["seed"]))
    return seeds, files


def main() -> int:
    core1, core2 = build_core(), build_core()
    csv1, csv2 = to_csv(core1), to_csv(core2)
    assert hashlib.md5(csv1.encode()).hexdigest() == hashlib.md5(csv2.encode()).hexdigest(), \
        "determinism check failed"
    rep = check_core(core1)

    seeds = [r["seed"] for r in core1]
    assert len(set(seeds)) == len(seeds), "seed collision (internal)"
    existing, files = load_existing_seeds()
    overlap = set(seeds) & existing
    assert not overlap, f"seed collision vs existing plans: {sorted(overlap)[:5]}"

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / "assignment_core.csv").write_text(csv1, encoding="utf-8")

    md5 = hashlib.md5(csv1.encode()).hexdigest()
    lines = ["# v11 core 割当表 検算レポート（step10_v11_plan.py 自動生成）", "",
             f"- GLOBAL_SEED={GLOBAL_SEED} / CORE_OFFSET={CORE_OFFSET} / core md5={md5}",
             f"- core={len(core1)}（fold1 {SPLITS['fold1']}/fold2 {SPLITS['fold2']}/"
             f"fold3 {SPLITS['fold3']} = 20h、DCASE2022 Task3公式合成相当・v10比2.0倍）",
             "- 分布: シーン種別 住宅40/生活40/幹線20 → n_car周辺分布 18/47/23/12% "
             "(厳密) / n_warn 45/40/15%",
             f"- シード帯域: {min(seeds)}..{max(seeds)}（既存{len(existing)}シード"
             f"[{len(files)}ファイル]との衝突ゼロを機械検証済み）",
             "- 決定論: 2回構築でmd5一致 / ID・seed一意",
             "- レンダラ仕様（step11_v11_render.py への申し送り）: scenario='v11core'、"
             "乱数消費順 ①マイク ②車n_car台 ③w1 ④w2 ⑤暗騒音、"
             "車t_cpa: n_car==1はCAR_TCPA / n_car>=2はU(4,9)", ""]
    lines += rep
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
