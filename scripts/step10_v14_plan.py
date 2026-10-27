# -*- coding: utf-8 -*-
"""v14 plan = v13 plan ＋ D10「至近・低速」層（2026-09-03・本人判断「実装してください」）。

背景（束の設計案 D10）: 学習データで GT<1 m のフレームは全体の 0.6%（<1.5 m で 2.5%）しかなく、
0.5〜1 m の距離が 2〜3 割遠く出る。原因はデータ不足。critical 層（最接近 0.6〜1.5 m）の車は
今は 11〜36 km/h（v10 で生活道路 30 km/h に合わせた範囲） で通るので 1.5 m 以内にいるのは 0.2 秒（2 フレーム）だけ。

変更（データだけ。学習レシピは変えない＝本人判断 B と両立）:
  critical 層の車の一部（CLOSE_SLOW_FRAC）を「徐行で至近を通る」に置き換える
    - 速度 5〜15 km/h（狭い生活道路で歩行者の脇を徐行する車）
    - 最接近 0.6〜1.2 m（3D。横距離はそれより少し小さい）
    - 音量 −6〜−10 dB（徐行はタイヤ音が小さくエンジン音主体。仮置き＝出典なし、実録で見直す）
  → 1.5 m 以内にいる時間が 0.9 秒前後（9 フレーム）になり、至近の例が約 3 倍になる見込み。
  実際の ≤1.5 m の通過は徐行が大半なので、分布としてもこの方が現実に近い（主張は控えめに）。

対象: scenario=v11core・danger_tier=critical・scene_type ∈ {residential, daily}（幹線は除く）。
seed × SALT14 で決定論的に選ぶ。列: close_slow(1/空), cs_cpa_m, cs_speed_kmh, cs_level_adj_db。
それ以外の列は v13 と同一（clip_id・seed・歩行/静止・雨・文法は不変）。

使い方: python scripts/step10_v14_plan.py   → out/dataset_outdoor_siren_v14/plan/assignment_v14.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_V13 = ROOT / "out/dataset_outdoor_siren_v13/plan/assignment_v13.csv"
OUT_DIR = ROOT / "out/dataset_outdoor_siren_v14/plan"

CLOSE_SLOW_FRAC = 0.50           # critical（住宅街・日常）のうち徐行に置き換える割合
CS_SPEED_KMH = (5.0, 15.0)       # 徐行
CS_CPA_M = (0.6, 1.2)            # 3D 最接近
CS_LEVEL_ADJ_DB = (-10.0, -6.0)  # 徐行の音量補正（仮置き）
ELIGIBLE_SCENE = {"residential", "daily"}
SALT14 = 20260914

# ---- D19（2026-09-21・変更票 #36 の (a)・既定 OFF）: 徐行を注意帯・安全帯にも広げる ----
# 既定 SLOW_TIERS=("critical",) のとき出力は従来と 1 バイトも変わらない（回帰 --check-v16 が成立する）。
# caution/safe を足すときは別の塩 SALT14_SLOW を使うので、critical の行（seed×SALT14）は変わらない（監査 9/21 §9「RNG」）。
# 3D 最接近の範囲は層ごと（m9.TIER_CPA と同じ値。critical は CS_CPA_M=(0.6,1.2) を保つ＝監査 M11）。
SLOW_TIERS = ("critical",)
SLOW_CPA = {"caution": (1.55, 3.0), "safe": (3.2, 15.0)}
SLOW_FRAC = {"caution": 0.50, "safe": 0.50}     # 層ごとの置換割合（本人の決定で変える。step10_v2final_plan が params から入れる）
SALT14_SLOW = 20260921


def assign(row: dict) -> dict:
    r = dict(row)
    r["close_slow"], r["cs_cpa_m"], r["cs_speed_kmh"], r["cs_level_adj_db"] = "", "", "", ""
    tier = row["danger_tier"]
    ok = (row["scenario"] == "v11core" and tier in SLOW_TIERS
          and row["scene_type"] in ELIGIBLE_SCENE and row["car_side"] in ("L", "R"))
    if not ok:
        return r
    if tier == "critical":
        rng = np.random.default_rng([int(row["seed"]), SALT14])
        frac, cpa = CLOSE_SLOW_FRAC, CS_CPA_M
    else:
        rng = np.random.default_rng([int(row["seed"]), SALT14_SLOW])
        frac, cpa = float(SLOW_FRAC[tier]), SLOW_CPA[tier]
    if rng.random() < frac:
        r["close_slow"] = "1"
        r["cs_cpa_m"] = f"{rng.uniform(*cpa):.2f}"
        r["cs_speed_kmh"] = f"{rng.uniform(*CS_SPEED_KMH):.1f}"
        r["cs_level_adj_db"] = f"{rng.uniform(*CS_LEVEL_ADJ_DB):.1f}"
    r["grammar"] = "v14"
    return r


def main() -> int:
    with open(SRC_V13, newline="", encoding="utf-8") as f:
        rows = [assign(r) for r in csv.DictReader(f)]
    cols = list(rows[0].keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "assignment_v14.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_el = sum(1 for r in rows if r["grammar"] == "v14")
    n_cs = sum(1 for r in rows if r["close_slow"] == "1")
    L = ["# v14 plan台帳（自動生成: step10_v14_plan.py）", "",
         "土台= v13 plan（clip_id・seed・歩行/静止・雨・文法 S1〜S4 は不変）。",
         f"D10 至近・低速: 対象 {n_el:,} 本（v11core × critical × 住宅街/日常）のうち "
         f"{n_cs:,} 本（{CLOSE_SLOW_FRAC:.0%}）を 徐行 {CS_SPEED_KMH[0]:.0f}〜{CS_SPEED_KMH[1]:.0f} km/h・"
         f"最接近 {CS_CPA_M[0]}〜{CS_CPA_M[1]} m・音量 {CS_LEVEL_ADJ_DB[0]:.0f}〜{CS_LEVEL_ADJ_DB[1]:.0f} dB に置換"
         f"（seed×{SALT14} で決定論的）。", ""]
    for split in ("fold1", "fold2"):
        rs = [r for r in rows if r["split"] == split]
        L.append(f"- {split}: {len(rs):,} 本、うち至近・低速 {sum(1 for r in rs if r['close_slow'] == '1'):,} 本")
    (OUT_DIR / "README_plan_v14.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("->", OUT_DIR / "assignment_v14.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
