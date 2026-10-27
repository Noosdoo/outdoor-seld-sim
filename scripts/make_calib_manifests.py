# -*- coding: utf-8 -*-
"""make_calib_manifests.py — 補正表の組（#72）の名簿 2 本と、確かめる組だけの plan を calib の plan から作る（2026-10-20・Fable）。
  - manifest_calib_fit.csv  = tune_syn_clip00001〜02000（表を作る・_dist_calib_v2.py fit の --manifest）
  - manifest_calib_check.csv = tune_syn_clip02001〜04000（条件 (1)・統一採点器の名簿の確認用）
  - plan_calib_check.csv     = check の行だけの plan（統一採点器の --plan・split = tune）。--clip-max では範囲を切れないため
使い方: python scripts/make_calib_manifests.py [--check]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "out/dataset_outdoor_siren_v2final_calib/plan"
PLAN = PLAN_DIR / "assignment_v2final.csv"
CFG = ROOT / "md/design/calib_sets_v29.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    assert sha(PLAN) == cfg["plan_sha256"], "calib の plan の sha が calib_sets_v29.json と違う"
    n_fit = int(cfg["sets"]["calib_fit"]["n"])
    with open(PLAN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f)); cols = list(rows[0].keys())
    fit = [r for r in rows if int(r["clip_id"][-5:]) <= n_fit]; chk = [r for r in rows if int(r["clip_id"][-5:]) > n_fit]
    assert len(fit) == n_fit and len(chk) == int(cfg["sets"]["calib_check"]["n"]) and all(r["split"] == "tune" for r in rows)
    outs = {}
    for name, sub in (("manifest_calib_fit.csv", fit), ("manifest_calib_check.csv", chk)):
        p = PLAN_DIR / name
        if "--check" not in sys.argv:
            p.write_text("clip_id\n" + "".join(r["clip_id"] + "\n" for r in sub), encoding="utf-8", newline="\n")
        outs[name] = (len(sub), sha(p)[:16])
    p = PLAN_DIR / "plan_calib_check.csv"
    if "--check" not in sys.argv:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(chk)
    outs["plan_calib_check.csv"] = (len(chk), sha(p)[:16])
    for k, (n, s) in outs.items():
        print(f"{k}: {n} 本・sha256 {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
