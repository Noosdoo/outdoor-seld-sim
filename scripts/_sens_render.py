# -*- coding: utf-8 -*-
"""_sens_render.py — 感度の変種の束を描く入口（2026-10-21 Fable）。_run_v2final_gen.py をそのまま使い（コードは変えない）、描く前に 2 つだけ足す:
  1. 置き場 out/<V2FINAL_DS_NAME>/SENS_OVERRIDES.json の定数の上書きを当てる（_sens_overrides.apply_from_env・無い定数や種類違いは止まる）
     JSON に記録した「元の値」と今のコードの値が違えば止まる（上書きを書いた後にコードが変わった = 変種と基準の差が定数 1 つの差でなくなる）
  2. 総数 N を params（EVAL_V3 = 6,000）ではなく plan の行数にする（名前に _eval_ を含むので _run_v2final_gen は確定評価の総数を仮定する）
置き場の名前は dataset_outdoor_siren_v2final_eval_sens_<tag> だけ（封印の eval v3・学習の束・補正表の組には書かない）。
使い方（sens_variant.sbatch から）: V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_eval_sens_<tag> python scripts/_sens_render.py [--list | --shard i/n | --rows lo-hi]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "scripts"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DS_PREFIX = "dataset_outdoor_siren_v2final_eval_sens_"


def main() -> int:
    ds_name = os.environ.get("V2FINAL_DS_NAME", "")
    if not ds_name.startswith(DS_PREFIX) or len(ds_name) == len(DS_PREFIX):
        print(f"NG V2FINAL_DS_NAME は {DS_PREFIX}<tag> の形（{ds_name!r}）"); return 3
    ds = ROOT / "out" / ds_name
    ovj = ds / "SENS_OVERRIDES.json"
    if not ovj.is_file():
        print(f"NG 上書きの記録が無い（{ovj}・基準でも空の overrides で置く = sens_variant.py overrides --tag <tag>）"); return 3
    if not (ds / "plan" / "assignment_v2final.csv").is_file():
        print(f"NG plan が無い（{ds / 'plan' / 'assignment_v2final.csv'}）"); return 3
    os.environ["SENS_OVERRIDES_JSON"] = str(ovj)
    import _run_v2final_gen as G  # noqa: E402  置き場を向け・描画の連なりを読む（module-global の定数はこのあと差し替える）
    import _sens_overrides as S  # noqa: E402
    J = json.loads(ovj.read_text(encoding="utf-8"))
    try:
        rec = S.apply_from_env()
    except S.OverrideError as e:
        print(f"NG 上書きが当てられない: {e}"); return 3
    want = J.get("checked_old_values") or {}
    bad = [k for k, r in rec.items() if k in want and json.dumps(want[k], default=str) != json.dumps(r["old"], default=str)]
    if bad:
        print("NG 上書きの記録にある『元の値』と今のコードの値が違う（記録のあとにコードが変わった・変種を作り直す）: " + "・".join(f"{k}: 記録 {want[k]!r} 今 {rec[k]['old']!r}" for k in bad)); return 3
    rows = G.load_plan()
    if not rows or not all(r["clip_id"].startswith("eval_syn_clip9") for r in rows):
        print("NG plan の番号が全部 eval_syn_clip9NNNN でない（封印の eval v3 の番号と混ざる）"); return 3
    G.N_TOTAL = len(rows)                               # 総数 = plan の行数（params の EVAL_V3 ではない）
    print(f"[sens] {ds_name}: N={len(rows)} overrides={len(rec)} tag={J.get('tag', '')}", flush=True)
    G.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
