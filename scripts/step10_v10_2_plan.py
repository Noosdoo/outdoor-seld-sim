# -*- coding: utf-8 -*-
"""step10_v10_2_plan.py — v10.2 割当表ジェネレータ（複数車対応の学習追加、v10規模）。

設計= md/design/v10_2_design_2026-07-21.md。v9.2（md/design/v9_2_design_2026-07-18.md）の
学習追加3系統を、v10のcore拡大率と同じ3.75倍にスケールする（本人方針=ゼミ骨子
md/seminar/seminar_20260804_outline.md の「v9.2の180本相当をv10規模に拡大」）:

  A 複数車      100 -> 375（2台225/3台150、60:40比を維持。v10aと同じtraffic経路）
  B 車なし       50 -> 188（siren入り保証 20->75。共起幻覚の負例）
  C 同クラス×2   30 -> 112（siren38/backup_beep37/bike_bell37。車1台同乗）
  合計          180 -> 675（fold1_room2、学習専用トークン）

さらに幻覚評価30本（fold2_room3、車なし×連続サイレン）は**v9.2の行をシード込みで
そのまま再録**する（「同一評価セットで比較」原則。v10物理での再生成だが行は同一）。
v9.2のctrl armは再生成しない（増量効果との分離はv9.2 vs v9.2ctrlで実証済み、
v10.2では機構が既知のため対照は不要=Colab学習1本分の節約）。

行の構造式（motion/tier/side/警告の巡回パターン）はv9.2のbuild_v92と同一式のまま
本数だけ拡大——v9.2で検証済みの均衡パターンを崩さないため。

出力: out/dataset_outdoor_siren_v10_2_add/plan/
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step10_v9_plan as v9  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v10_2_add" / "plan"
V10_PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v10" / "plan"

# シード空間: v9系(20260717)・v10 core(20260721)と衝突しない別日付。
# build_v92と同じ +50000 オフセット方式（下でv10全planとの重複を実測assertする）
GLOBAL_SEED = 20260722

N_A, N_A2CAR = 375, 225          # 複数車（うち2台。残り150が3台）
N_B, N_B_SIREN = 188, 75         # 車なし（うちsiren保証）
N_C_SPLIT = (38, 37, 37)         # siren / backup_beep / bike_bell


def build_v10_2add() -> list:
    base = GLOBAL_SEED * 613 + 50000
    add, k = [], 0

    def row(clip_id, motion, n_warn, w1, s1, w2, s2, tier, car, scen):
        nonlocal k
        r = {"clip_id": clip_id, "split": clip_id.split("_")[0],
             "motion": motion, "n_warnings": n_warn,
             "w1_class": w1, "w1_side": s1, "w2_class": w2, "w2_side": s2,
             "danger_tier": tier, "car_side": car, "scenario": scen,
             "seed": base + k}
        k += 1
        return r

    # A: 複数車375（v9.2のA枠と同一式、本数のみ拡大。警告0/1を交互）
    warn_k = 0
    for i in range(N_A):
        ncars = 2 if i < N_A2CAR else 3
        w = v9.WARN_CLASSES[warn_k % 5] if i % 2 == 0 else ""
        if w:
            warn_k += 1
        add.append(row(f"fold1_room2_mix{i+1:03d}",
                       "static" if (i // 2) % 2 == 0 else "walk",
                       1 if w else 0, w, v9.SIDES[i % 2] if w else "", "", "",
                       v9.TIERS[i % 3], v9.SIDES[(i // 3) % 2], f"traffic{ncars}"))
    # B: 車なし188（siren保証75本、残りは他4クラス循環。5本に1本は警告2個）
    for i in range(N_B):
        w1 = "siren" if i < N_B_SIREN else v9.WARN_CLASSES[1 + (i - N_B_SIREN) % 4]
        two = i % 5 == 4
        w2 = v9.WARN_CLASSES[(i + 2) % 5] if two else ""
        if w2 == w1:
            w2 = v9.WARN_CLASSES[(i + 3) % 5]
        add.append(row(f"fold1_room2_mix{N_A+i+1:03d}",
                       "static" if i % 2 == 0 else "walk",
                       2 if w2 else 1, w1, v9.SIDES[i % 2],
                       w2, v9.SIDES[(i + 1) % 2] if w2 else "", "na", "", "normal"))
    # C: 同一クラス警告×2 112（siren38/beep37/bell37。車1台同乗）
    c_classes = (["siren"] * N_C_SPLIT[0] + ["backup_beep"] * N_C_SPLIT[1]
                 + ["bike_bell"] * N_C_SPLIT[2])
    for i, cls in enumerate(c_classes):
        add.append(row(f"fold1_room2_mix{N_A+N_B+i+1:03d}",
                       "static" if i % 2 == 0 else "walk", 2,
                       cls, v9.SIDES[i % 2], cls, v9.SIDES[(i + 1) % 2],
                       v9.TIERS[i % 3], v9.SIDES[(i // 2) % 2], "normal"))
    return add


def load_v10_seeds() -> set:
    seeds = set()
    for f in sorted(V10_PLAN_DIR.glob("assignment_*.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                seeds.add(int(r["seed"]))
    return seeds


def main() -> int:
    add1, add2 = build_v10_2add(), build_v10_2add()
    csv1, csv2 = v9.to_csv(add1), v9.to_csv(add2)
    assert hashlib.md5(csv1.encode()).hexdigest() == \
        hashlib.md5(csv2.encode()).hexdigest(), "determinism check failed"

    # 幻覚評価30本はv9.2の行をそのまま再録（シード込み同一=同一評価セット原則）
    hal = v9.build_v92()[2]
    hal_csv = v9.to_csv(hal)
    assert len(hal) == 30 and all(r["scenario"] == "carfree_siren" for r in hal)

    # 検算: 構成
    assert len(add1) == N_A + N_B + sum(N_C_SPLIT) == 675
    assert sum(1 for r in add1 if r["scenario"] == "traffic2") == N_A2CAR
    assert sum(1 for r in add1 if r["scenario"] == "traffic3") == N_A - N_A2CAR
    assert sum(1 for r in add1 if not r["car_side"] and r["scenario"] == "normal") == N_B
    assert sum(1 for r in add1 if r["w1_class"] == r["w2_class"] != "") == sum(N_C_SPLIT)
    b_rows = add1[N_A:N_A + N_B]
    assert sum(1 for r in b_rows if r["w1_class"] == "siren") == N_B_SIREN
    # 検算: ID・シード一意＋v10全plan（core3600+評価枠228）との衝突ゼロ
    ids = [r["clip_id"] for r in add1 + hal]
    assert len(set(ids)) == len(ids), "clip_id collision"
    seeds = [r["seed"] for r in add1 + hal]
    assert len(set(seeds)) == len(seeds), "seed collision (internal)"
    v10_seeds = load_v10_seeds()
    overlap = set(r["seed"] for r in add1) & v10_seeds
    assert not overlap, f"seed collision vs v10: {sorted(overlap)[:5]}"
    # halはv9.2と同一シード（意図的再録）だが、v10のどの行とも重複しないこと
    assert not (set(r["seed"] for r in hal) & v10_seeds)

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / "assignment_v10_2add.csv").write_text(csv1, encoding="utf-8")
    (PLAN_DIR / "assignment_halluc.csv").write_text(hal_csv, encoding="utf-8")

    md5 = hashlib.md5(csv1.encode()).hexdigest()
    lines = ["# v10.2 割当表 検算レポート（step10_v10_2_plan.py 自動生成）", "",
             f"- GLOBAL_SEED={GLOBAL_SEED} / v10_2add md5={md5}",
             f"- v10_2add={len(add1)}本 = A複数車{N_A}(2台{N_A2CAR}/3台{N_A-N_A2CAR})"
             f" + B車なし{N_B}(siren保証{N_B_SIREN}) + C同クラス×2 {sum(N_C_SPLIT)}"
             f"(siren{N_C_SPLIT[0]}/beep{N_C_SPLIT[1]}/bell{N_C_SPLIT[2]})",
             "- halluc=30本（fold2_room3、v9.2の行をシード込み再録=同一評価セット原則）",
             "- 決定論: 2回構築でmd5一致 / ID・seed一意 / "
             f"v10全plan({len(v10_seeds)}シード)との衝突ゼロ", ""]
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print("\n".join(lines))
    print("ALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
