# -*- coding: utf-8 -*-
"""相手のゲインと閾値を選ぶための val の部分集合の名簿（変更票 #73・Q4・R73-05）。
規則（固定）: val の名簿（manifest_val_syn_v2final.csv・12,420 本）のうち clip 番号を 12 で割った余りが 0 の本 = 選択用（manifest_val_select.csv）、
余りが 6 の本 = 確認用（manifest_val_check.csv）。成績は見ずに番号だけで決める。
使い方: python scripts/make_val_select_manifest.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _baseline_select_gain import subsets  # noqa: E402

SRC = ROOT / "out/dataset_outdoor_siren_v2final/plan/manifest_val_syn_v2final.csv"


def main() -> int:
    with open(SRC, encoding="utf-8", newline="") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    sel, chk = subsets(clips)
    for name, ids in (("manifest_val_select.csv", sel), ("manifest_val_check.csv", chk)):
        p = SRC.parent / name
        p.write_text("clip_id\n" + "".join(f"{c}\n" for c in ids), encoding="utf-8", newline="\n")
        print(f"{name}: {len(ids)} 本・sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
