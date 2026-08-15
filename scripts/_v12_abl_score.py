# -*- coding: utf-8 -*-
"""ablation arm の採点ラッパ（2026-08-16、確認runで必要になり新規作成）。

`_v12_score.py` は正解ラベルの参照先 `DS12` を**基準データセットに固定**している。
armの予測をそのまま流すと、**armの予測を基準の世界の正解で採点**してしまう。
これは破壊的ではないが**黙って誤った比較**になるため、参照先を明示的に切り替える。

  --gt self : armの世界の正解で採点（自条件val。「その世界で学習できたか」）
  --gt full : 基準の世界の正解で採点（フル物理val＝転移ギャップ。**主要評価**）

使い方:
  ABLATE=no_1r python scripts/_v12_abl_score.py <pred_val_all.csv> <出力dir> --gt self
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ARM = os.environ.get("ABLATE", "").strip()
if not ARM:
    raise SystemExit("ABLATE が空です。arm名を指定してください。")

GT = "self"
if "--gt" in sys.argv:
    i = sys.argv.index("--gt")
    GT = sys.argv[i + 1]
    del sys.argv[i:i + 2]
assert GT in ("self", "full"), f"--gt は self|full: {GT}"

BASE_DS = ROOT / "out" / "dataset_outdoor_siren_v12"
ARM_DS = ROOT / "out" / f"dataset_outdoor_siren_v12_abl_{ARM}"
GT_DS = ARM_DS if GT == "self" else BASE_DS
assert (GT_DS / "metadata").is_dir(), f"正解ラベルが無い: {GT_DS}"
print(f"[abl_score] arm={ARM} 正解={GT}（{GT_DS.name}）", flush=True)

spec = importlib.util.spec_from_file_location(
    "v12score", ROOT / "scripts" / "_v12_score.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.DS12 = GT_DS                       # モジュール定数を差し替えてから main() を呼ぶ
m.main()
print(f"[abl_score] DONE arm={ARM} gt={GT}")
