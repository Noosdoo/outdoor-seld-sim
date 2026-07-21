# -*- coding: utf-8 -*-
"""step10_v10_plan.py — v10 割当表ジェネレータ（設計= out/v10_design_2026-07-21.md）。

規模を「論文レベル」= TAU-NIGENS Spatial Sound Events 2021の train:val:test=4:1:1比を
core本数に適用して拡大する（train2400/val600/test600、v9.1の3.75倍）。
バランス設計（クラス×危険層×側の均衡アルゴリズム）はstep10_v9_plan.pyのbuild_core/
check_coreをそのまま再利用（引数化により後方互換を保ったまま拡張済み）。
評価専用の追加枠（scenario/probe/scenario2/v10a）は本数はv9.1のまま据え置き
（既にn>=20/セルで評価には十分。拡大は将来のv11以降の課題として明記）。

出力: out/dataset_outdoor_siren_v10/plan/
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step10_v9_plan as v9  # noqa: E402

PLAN_DIR = ROOT / "out" / "dataset_outdoor_siren_v10" / "plan"

# v9計画のシード空間（20260717系列）と衝突しないよう別日付をシードに使う
GLOBAL_SEED = 20260721

# TAU-NIGENS 2021 dev-train:val:test = 400:100:100 = 4:1:1 の比をcoreに適用。
# train2400 = 現行640本の3.75倍（論文レベル、本人合意 2026-07-21）
SPLITS = {"fold1": 2400, "fold2": 600, "fold3": 600}


def main() -> int:
    core1 = v9.build_core(splits=SPLITS, seed=GLOBAL_SEED)
    core2 = v9.build_core(splits=SPLITS, seed=GLOBAL_SEED)  # 決定論チェック
    csv1, csv2 = v9.to_csv(core1), v9.to_csv(core2)
    assert hashlib.md5(csv1.encode()).hexdigest() == hashlib.md5(csv2.encode()).hexdigest(), \
        "determinism check failed"

    rep = v9.check_core(core1, splits=SPLITS)

    # 評価専用の追加枠はv9.1のものをそのまま踏襲（本数据え置き、room/seedも同一）。
    # v10のレンダラ（step11_v10_render.py）がJP精査済みのLAW/SPEEDで生成するため、
    # 対象がhorn/backup_beep/car_driveを含む枠（scenario2のS1/S3/S5・probe・v10a）も
    # 新しい物理定数の恩恵を受ける（行の構成自体はv9.1と同一でよい）。
    scen, probe = v9.build_scenario(), v9.build_probe()
    scen2 = v9.build_scenario2()
    v10a_rows = v9.build_v10a()

    allrows = core1 + scen + probe + scen2 + v10a_rows
    seeds = [r["seed"] for r in allrows]
    assert len(set(seeds)) == len(seeds), "seed collision"
    ids = [r["clip_id"] for r in allrows]
    assert len(set(ids)) == len(ids), "clip_id collision"

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / "assignment_core.csv").write_text(csv1, encoding="utf-8")
    (PLAN_DIR / "assignment_scenario.csv").write_text(v9.to_csv(scen), encoding="utf-8")
    (PLAN_DIR / "assignment_probe.csv").write_text(v9.to_csv(probe), encoding="utf-8")
    (PLAN_DIR / "assignment_scenario2.csv").write_text(v9.to_csv(scen2), encoding="utf-8")
    (PLAN_DIR / "assignment_v10a.csv").write_text(v9.to_csv(v10a_rows), encoding="utf-8")

    md5 = hashlib.md5(csv1.encode()).hexdigest()
    lines = ["# v10 割当表 検算レポート（step10_v10_plan.py 自動生成）", "",
             f"- GLOBAL_SEED={GLOBAL_SEED} / core md5={md5}",
             f"- core={len(core1)}（train{SPLITS['fold1']}/val{SPLITS['fold2']}/"
             f"test{SPLITS['fold3']}、TAU-NIGENS train:val:test=4:1:1比・"
             f"v9.1比 {SPLITS['fold1']/640:.2f}倍）",
             f"- scenario={len(scen)} probe={len(probe)} scenario2={len(scen2)} "
             f"v10a={len(v10a_rows)}（評価専用枠、v9.1本数のまま据え置き）",
             "- 決定論: 2回構築でmd5一致 / ID・seed一意 / 2音源の同一クラス重複ゼロ", ""]
    lines += rep
    (PLAN_DIR / "plan_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nALL CHECKS PASSED ->", PLAN_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
