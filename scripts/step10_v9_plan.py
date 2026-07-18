# -*- coding: utf-8 -*-
"""step10_v9_plan.py — v9 割当表ジェネレータ＋全数検算（設計= out/v9_design_v2_2026-07-16.md 5節・9節1項）

生成前に「1120本それぞれに何を割り当てるか」の名簿(CSV)を確定し、交絡ゼロを検算する。
音は一切作らない。レンダラ(step6 v9経路)はこの表だけを入力に読む。

出力: out/dataset_outdoor_siren_v9/plan/
  - assignment_core.csv      本体1120行 (fold1 640 / fold2 240 / fold3 240)
  - assignment_scenario.csv  交差点サイレン20行 (fold2 mix241-260、均衡表の外側の追加評価枠)
  - assignment_probe.csv     レベル正規化プローブ48行 (6クラス×8、評価専用)
  - plan_check_report.md     検算結果とクロス表

均衡の作り方（idx算術は使わない）:
  分割×歩行ごとに 警告音0/1/2個=30/55/15% を厳密数で確定 → 2音源はペア循環、
  1音源はクラス別目標数(合計をクラス間±1に均す)への充当 → 危険層と車側は
  層内サイクリック配布（クラス×層×側の全ペアが±1均衡になる）→ 最後にシャッフルして
  mix番号を振る（番号と条件の相関を断つ）。
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v9" / "plan"

GLOBAL_SEED = 20260717  # v9計画専用（生成側step6は別シードを使う）

WARN_CLASSES = ["siren", "horn", "backup_beep", "bike_bell", "crossing"]
TIERS = ["critical", "caution", "safe"]  # マイク相対CPA ≤1.5 / 1.5-3.0 / >3.0 m
SIDES = ["L", "R"]

SPLITS = {"fold1": 640, "fold2": 240, "fold3": 240}
FRAC_W = {0: 0.30, 1: 0.55, 2: 0.15}
N_SCENARIO = 20  # fold2 追加枠 (mix241-260)
N_PROBE_PER_CLASS = 8  # 6クラス×8=48（静止4・歩行4）

COLUMNS = ["clip_id", "split", "motion", "n_warnings", "w1_class", "w1_side",
           "w2_class", "w2_side", "danger_tier", "car_side", "scenario", "seed"]


def exact_counts(total: int, fracs: dict) -> dict:
    """比率→整数数。端数は大きい順に+1（決定論）。"""
    raw = {k: total * v for k, v in fracs.items()}
    base = {k: int(np.floor(r)) for k, r in raw.items()}
    rem = total - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    return base


def split_thirds(n: int, offset: int) -> list:
    """nをTIERSへ±1で分配。余りの行き先はoffsetで回す（層の系統的偏り防止）。"""
    base = n // 3
    counts = [base] * 3
    for i in range(n - base * 3):
        counts[(offset + i) % 3] += 1
    return counts


def build_motion_group(split: str, motion: str, n_clips: int, rng: np.random.Generator,
                       extra_classes: list) -> list:
    """1つの 分割×歩行 グループの行を構築（クリップID・seedは未定）。
    extra_classes: クラス別目標数で+1を貰うクラス（分割単位で重複なく配られる）。"""
    n_by_w = exact_counts(n_clips, FRAC_W)
    rows = []  # dict per clip

    # --- 警告クラスの割当 -------------------------------------------------
    # 2音源: 10ペアを循環（シャッフル順）。同一クラス重複なし（設計5節）。
    pairs_all = list(itertools.combinations(range(5), 2))
    order = rng.permutation(len(pairs_all))
    pair_seq = [pairs_all[i] for i in order]
    pair_count = Counter()
    two_rows = []
    for i in range(n_by_w[2]):
        a, b = pair_seq[i % len(pair_seq)]
        pair_count[a] += 1
        pair_count[b] += 1
        two_rows.append({"n_warnings": 2, "w1_class": WARN_CLASSES[a],
                         "w2_class": WARN_CLASSES[b]})

    # 1音源: クラス別イベント総数(1音源+2音源)がクラス間±1になるよう充当。
    # 余り(+1)を貰うクラスは extra_classes（分割単位でmotion間の重複なし→分割合計も±1）。
    total_events = n_by_w[1] + 2 * n_by_w[2]
    base_t = total_events // 5
    r = total_events - base_t * 5
    assert len(extra_classes) == r, (split, motion, r, extra_classes)
    target = {c: base_t for c in range(5)}
    for c in extra_classes:
        target[c] += 1
    one_counts = {c: target[c] - pair_count[c] for c in range(5)}
    assert all(v >= 0 for v in one_counts.values()), (split, motion, one_counts)
    assert sum(one_counts.values()) == n_by_w[1]
    one_rows = []
    for c in range(5):
        one_rows += [{"n_warnings": 1, "w1_class": WARN_CLASSES[c], "w2_class": ""}
                     for _ in range(one_counts[c])]

    zero_rows = [{"n_warnings": 0, "w1_class": "", "w2_class": ""} for _ in range(n_by_w[0])]

    # --- 危険層と車側: 層別サイクリック配布 --------------------------------
    # クリップを「クラス構成の層」順に並べ、危険層を circular に配る
    # → どの層内でも±1、グループ合計も±1、クラス×層クロスも±1で均衡する。
    def stratum_key(row):
        return (row["n_warnings"], row["w1_class"], row["w2_class"])

    all_rows = sorted(zero_rows + one_rows + two_rows, key=stratum_key)
    tier_cycle_off = int(rng.integers(3))
    for i, row in enumerate(all_rows):
        row["danger_tier"] = TIERS[(tier_cycle_off + i) % 3]

    # 車側: (危険層, クラス層) 順で並べ替えて L/R を交互 → 側⊥層・側⊥クラスとも±1
    side_off = int(rng.integers(2))
    for i, row in enumerate(sorted(all_rows, key=lambda r: (r["danger_tier"],) + stratum_key(r))):
        row["car_side"] = SIDES[(side_off + i) % 2]

    # 警告音の側: クラスごとにイベントを集めて交互 → クラス別 |L-R|≤1
    ev_by_class = defaultdict(list)
    for row in all_rows:
        if row["n_warnings"] >= 1:
            ev_by_class[row["w1_class"]].append((row, "w1_side"))
        if row["n_warnings"] == 2:
            ev_by_class[row["w2_class"]].append((row, "w2_side"))
    for c, evs in sorted(ev_by_class.items()):
        off = int(rng.integers(2))
        for i, (row, key) in enumerate(evs):
            row[key] = SIDES[(off + i) % 2]
    for row in all_rows:
        row.setdefault("w1_side", "")
        row.setdefault("w2_side", "")
        row.update(split=split, motion=motion, scenario="normal")
    return all_rows


def build_core() -> list:
    rows = []
    for split, n in SPLITS.items():
        rng = np.random.default_rng(GLOBAL_SEED * 2741 + {"fold1": 1, "fold2": 2, "fold3": 3}[split])
        # +1を貰うクラスの並びを分割ごとに1回抽選し、静止が先頭r個・歩行が次のr個を取る
        n_by_w_half = exact_counts(n // 2, FRAC_W)
        r = (n_by_w_half[1] + 2 * n_by_w_half[2]) % 5
        class_order = list(rng.permutation(5))
        extras = {"static": [class_order[i % 5] for i in range(r)],
                  "walk": [class_order[(r + i) % 5] for i in range(r)]}
        for motion in ["static", "walk"]:
            rows += build_motion_group(split, motion, n // 2, rng, extras[motion])
        # グループ構築順とmix番号の相関を断つ最終シャッフル
        split_rows = [r for r in rows if r["split"] == split]
        perm = rng.permutation(len(split_rows))
        rows = [r for r in rows if r["split"] != split] + [split_rows[i] for i in perm]
    # クリップID・seed付与（split内連番）
    idx_global = 0
    for split in SPLITS:
        for i, row in enumerate([r for r in rows if r["split"] == split]):
            row["clip_id"] = f"{split}_room1_mix{i + 1:03d}"
            row["seed"] = GLOBAL_SEED * 613 + idx_global
            idx_global += 1
    return rows


def build_scenario() -> list:
    """交差点サイレン: 歩行(横断方向)×サイレン連続吹鳴。fold2の追加枠。

    room9トークンにする理由: PSELDNetsのroomsフィルタ([fold2_room1])から自動的に
    外れ、val主指標を汚さない。車は入れない（サイレン通知のリードタイムだけを
    採点する枠。横断歩行×並走車の危険幾何を統制するのは本枠の目的外）。
    """
    rows = []
    for i in range(N_SCENARIO):
        rows.append({
            "clip_id": f"fold2_room9_mix{i + 1:03d}", "split": "fold2",
            "motion": "walk", "n_warnings": 1,
            "w1_class": "siren", "w1_side": SIDES[i % 2], "w2_class": "", "w2_side": "",
            "danger_tier": "na", "car_side": "", "scenario": "intersection_siren",
            "seed": GLOBAL_SEED * 613 + 10000 + i,
        })
    return rows


def build_scenario2() -> list:
    """追加5シナリオ（2026-07-17設計、out/scenario_design_2026-07-17.md）。

    各20本・評価専用。room4〜8トークンでPSELDNetsの主評価から自動分離。
    幾何の連続値はレンダラ側がseedから引く（本表は離散条件のみ）。
    """
    rows = []
    spec = [
        ("fold2_room8", "crossing_wait", "walk", 1, "crossing", "caution", True),
        ("fold2_room7", "bell_overtake", "walk", 1, "bike_bell", "na", False),
        ("fold2_room6", "backup_reverse", "half", 1, "backup_beep", "tier2", True),
        ("fold2_room5", "silent_negative", "half", 0, "", "na", False),
        ("fold2_room4", "siren_worstnoise", "half", 1, "siren", "na", False),
    ]
    base = GLOBAL_SEED * 613 + 30000
    k = 0
    for room, scen, motion_mode, n_warn, wcls, tier_mode, has_car in spec:
        for i in range(N_SCENARIO):
            motion = ("walk" if motion_mode == "walk"
                      else ("static" if i < N_SCENARIO // 2 else "walk"))
            tier = ("na" if tier_mode == "na"
                    else ("caution" if tier_mode == "caution"
                          else ("critical" if i % 2 == 0 else "caution")))
            rows.append({
                "clip_id": f"{room}_mix{i + 1:03d}", "split": "fold2",
                "motion": motion, "n_warnings": n_warn,
                "w1_class": wcls, "w1_side": SIDES[i % 2] if wcls else "",
                "w2_class": "", "w2_side": "",
                "danger_tier": tier,
                "car_side": SIDES[(i // 2) % 2] if has_car else "",
                "scenario": scen, "seed": base + k,
            })
            k += 1
    return rows


def build_v10a() -> list:
    """v10a 交通量・複数車 60本（2026-07-18設計、out/v10_plan_2026-07-17.md トラックA）。

    fold8_room1=評価専用。scenario='traffic2'/'traffic3' が車の台数を運ぶ。
    car_side/danger_tier は1台目のみ（2台目以降はレンダラがseedから引く）。
    警告音あり30:なし30 × 静止30:歩行30 を直交させ、警告クラスは5種均等(6本ずつ)。
    """
    rows = []
    base = GLOBAL_SEED * 613 + 40000
    warn_k = 0
    for i in range(60):
        ncars = 2 if i < 30 else 3
        motion = "static" if i % 2 == 0 else "walk"
        has_warn = (i // 2) % 2 == 0
        if has_warn:
            wcls = WARN_CLASSES[warn_k % 5]
            warn_k += 1
        else:
            wcls = ""
        rows.append({
            "clip_id": f"fold8_room1_mix{i + 1:03d}", "split": "fold8",
            "motion": motion, "n_warnings": 1 if has_warn else 0,
            "w1_class": wcls, "w1_side": SIDES[i % 2] if wcls else "",
            "w2_class": "", "w2_side": "",
            "danger_tier": TIERS[i % 3],
            "car_side": SIDES[(i // 3) % 2],
            "scenario": f"traffic{ncars}", "seed": base + i,
        })
    return rows


def build_probe() -> list:
    """レベル正規化プローブ: 6クラス×8本（静止4/歩行4）、単一音源を同一受聴レベルで提示。

    fold9トークン=どの学習/評価yamlにも掛からない別枠（専用inferで使う）。
    車以外のプローブに車は入れない（単一音源で音色だけを問う）。
    """
    rows = []
    classes = WARN_CLASSES + ["car_drive"]
    for ci, cls in enumerate(classes):
        for j in range(N_PROBE_PER_CLASS):
            rows.append({
                "clip_id": f"fold9_room1_mix{ci * N_PROBE_PER_CLASS + j + 1:03d}", "split": "probe",
                "motion": "static" if j < N_PROBE_PER_CLASS // 2 else "walk",
                "n_warnings": 0 if cls == "car_drive" else 1,
                "w1_class": "" if cls == "car_drive" else cls,
                "w1_side": SIDES[j % 2], "w2_class": "", "w2_side": "",
                "danger_tier": "safe" if cls == "car_drive" else "na",
                "car_side": SIDES[j % 2] if cls == "car_drive" else "",
                "scenario": f"probe_{cls}",
                "seed": GLOBAL_SEED * 613 + 20000 + ci * N_PROBE_PER_CLASS + j,
            })
    return rows


def to_csv(rows: list) -> str:
    lines = [",".join(COLUMNS)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in COLUMNS))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 検算 ----

def crosstab(rows, key_a, key_b, filt=None):
    tab = Counter()
    for r in rows:
        if filt and not filt(r):
            continue
        tab[(r[key_a], r[key_b])] += 1
    return tab


def class_events(rows, filt=None):
    """警告イベント（クリップでなく音源単位）のクラス計数。"""
    cnt = Counter()
    for r in rows:
        if filt and not filt(r):
            continue
        for k in ("w1_class", "w2_class"):
            if r[k]:
                cnt[r[k]] += 1
    return cnt


def check_core(rows: list) -> list:
    """全数検算。返り値=レポート行。assert失敗=計画不合格。"""
    rep = []
    for split, n in SPLITS.items():
        sr = [r for r in rows if r["split"] == split]
        assert len(sr) == n, (split, len(sr))
        for motion in ["static", "walk"]:
            mr = [r for r in sr if r["motion"] == motion]
            assert len(mr) == n // 2, (split, motion, len(mr))
            # 音源数の厳密数
            got = Counter(r["n_warnings"] for r in mr)
            want = exact_counts(n // 2, FRAC_W)
            assert dict(got) == want, (split, motion, got, want)
            # クラス別イベント: ±1、val/testは各マス>=20（設計5節）
            ev = class_events(mr)
            vals = [ev[c] for c in WARN_CLASSES]
            assert max(vals) - min(vals) <= 1, (split, motion, ev)
            if split in ("fold2", "fold3"):
                assert min(vals) >= 20, (split, motion, ev)
            # 危険層 ±1
            tc = Counter(r["danger_tier"] for r in mr)
            tv = [tc[t] for t in TIERS]
            assert max(tv) - min(tv) <= 1, (split, motion, tc)
            # 車側 ±1
            sc = Counter(r["car_side"] for r in mr)
            assert abs(sc["L"] - sc["R"]) <= 1, (split, motion, sc)
            rep.append(f"- {split}/{motion}: clips={len(mr)} n_warn={dict(sorted(got.items()))} "
                       f"class_ev={ {c: ev[c] for c in WARN_CLASSES} } tier={ {t: tc[t] for t in TIERS} }")
        # 分割合計でもクラス±1
        ev = class_events(sr)
        vals = [ev[c] for c in WARN_CLASSES]
        assert max(vals) - min(vals) <= 1, (split, ev)

        # 交絡チェック（クロス表の行内 max-min）: クラス×側 / クラス×層 / 層×側 / 層×音源数
        def spread(tab, a_vals, b_vals, tol, name):
            worst = 0
            for a in a_vals:
                row = [tab[(a, b)] for b in b_vals]
                worst = max(worst, max(row) - min(row))
            assert worst <= tol, (split, name, worst, tab)
            return worst

        # 警告イベント単位のクラス×側
        evt = Counter()
        for r in sr:
            for ck, sk in (("w1_class", "w1_side"), ("w2_class", "w2_side")):
                if r[ck]:
                    evt[(r[ck], r[sk])] += 1
        w1 = spread(evt, WARN_CLASSES, SIDES, 2, "class_x_wside")
        # クリップ単位のクラス(w1基準)×危険層・×車側、危険層×車側・×音源数
        w2 = spread(crosstab(sr, "w1_class", "danger_tier", lambda r: r["n_warnings"] >= 1),
                    WARN_CLASSES, TIERS, 3, "class_x_tier")
        w3 = spread(crosstab(sr, "w1_class", "car_side", lambda r: r["n_warnings"] >= 1),
                    WARN_CLASSES, SIDES, 3, "class_x_carside")
        w4 = spread(crosstab(sr, "danger_tier", "car_side"), TIERS, SIDES, 2, "tier_x_carside")
        w5 = spread(crosstab(sr, "danger_tier", "motion"), TIERS, ["static", "walk"], 2, "tier_x_motion")
        rep.append(f"- {split}: cross-spread class_x_wside={w1} class_x_tier={w2} "
                   f"class_x_carside={w3} tier_x_carside={w4} tier_x_motion={w5} (all within tol)")
    # 2音源の同一クラス重複なし・正準順
    for r in rows:
        if r["n_warnings"] == 2:
            assert r["w1_class"] != r["w2_class"], r
            assert WARN_CLASSES.index(r["w1_class"]) < WARN_CLASSES.index(r["w2_class"]), r
    # ID・seed一意
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    return rep


def main() -> int:
    core1, scen, probe = build_core(), build_scenario(), build_probe()
    scen2 = build_scenario2()
    v10a = build_v10a()
    core2 = build_core()  # 決定論チェック: 2回構築して一致
    csv1, csv2 = to_csv(core1), to_csv(core2)
    assert hashlib.md5(csv1.encode()).hexdigest() == hashlib.md5(csv2.encode()).hexdigest(), \
        "determinism check failed"

    rep = check_core(core1)
    seeds = [r["seed"] for r in core1 + scen + probe + scen2 + v10a]
    assert len(set(seeds)) == len(seeds), "seed collision"
    ids = [r["clip_id"] for r in core1 + scen + probe + scen2 + v10a]
    assert len(set(ids)) == len(ids), "clip_id collision"

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / "assignment_core.csv").write_text(csv1, encoding="utf-8")
    (PLAN_DIR / "assignment_scenario.csv").write_text(to_csv(scen), encoding="utf-8")
    (PLAN_DIR / "assignment_probe.csv").write_text(to_csv(probe), encoding="utf-8")
    (PLAN_DIR / "assignment_scenario2.csv").write_text(to_csv(scen2), encoding="utf-8")
    (PLAN_DIR / "assignment_v10a.csv").write_text(to_csv(v10a), encoding="utf-8")

    md5 = hashlib.md5(csv1.encode()).hexdigest()
    lines = ["# v9 割当表 検算レポート（step10_v9_plan.py 自動生成）", "",
             f"- GLOBAL_SEED={GLOBAL_SEED} / core md5={md5}",
             f"- core={len(core1)} scenario={len(scen)} probe={len(probe)}",
             "- 決定論: 2回構築でmd5一致 / ID・seed一意 / 2音源の同一クラス重複ゼロ", ""]
    lines += rep
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
