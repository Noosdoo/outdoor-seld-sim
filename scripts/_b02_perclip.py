# -*- coding: utf-8 -*-
"""B02 の切り分け（2026-09-24）: 統一採点器の clip_stats をクリップごとに JSON に書き出し、手元とサーバ（環境違い）で突き合わせる。
使い方: python scripts/_b02_perclip.py <出力json> --plan <manifest> --meta <GT dir> --pred <pred.csv> [--split val] [--minframe 40]
出力: {clip: {"n_fire":…, "crit":[hit,n], "caut":…, "safe":…, "n_strong":…, "n_pairs":…}} と環境（numpy の版）
"""
from __future__ import annotations

import csv
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_score_unified", ROOT / "scripts" / "_score_unified.py")
S = importlib.util.module_from_spec(_spec); sys.modules["_score_unified"] = S; _spec.loader.exec_module(S)


def arg(argv, k, d=None):
    if k in argv:
        i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
    return d


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[0])
    plan = Path(arg(argv, "--plan")); meta = Path(arg(argv, "--meta")); pred_p = Path(arg(argv, "--pred"))
    split = arg(argv, "--split", "val"); minframe = int(arg(argv, "--minframe", "40"))
    sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import normalize, split_to_new
    clips = [normalize(r["clip_id"]) for r in csv.DictReader(open(plan, encoding="utf-8")) if split_to_new(r.get("split", split)) == split]
    pred = S.load_pred_rows(pred_p, horiz=False, strict=False)
    res = {}
    for c in clips:
        s = S.clip_stats(pred, meta, c, minframe)
        res[c] = {"n_fire": s["n_fire"], "crit": s["crit"], "caut": s["caut"], "safe": s["safe"], "n_strong": s["n_strong"],
                  "n_pairs": s["n_pairs"], "n_close_det": s["n_close_det"], "n_close_cap": s["n_close_cap"]}
    env = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "n_clips": len(clips)}
    out.write_text(json.dumps({"env": env, "clips": res}, ensure_ascii=False, indent=0), encoding="utf-8")
    tot = {k: sum(v[k] for v in res.values()) for k in ("n_fire", "n_strong", "n_pairs")}
    print(env, "totals", tot, "crit_hit", sum(v["crit"][0] for v in res.values()), "/", sum(v["crit"][1] for v in res.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
