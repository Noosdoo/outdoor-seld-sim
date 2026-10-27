# -*- coding: utf-8 -*-
"""最新の val の名簿（manifest）を plan から作る（凍結 r11・変更票 #65・選択器 2.1 の入力・2026-10-13）。

選択器 2.0 は v12 の val（manifest_val_syn.csv・1,800 本・自転車の正解 0・高さ 1.5 m 固定）で時点を選んでいた。r11 で「最新の val」に替える:
  入力 = 最終の学習 plan（out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv）の split=val の行（v27：10,116 本・v28 の積み増し後は増える）
  出力 = manifest_val_syn_v2final.csv（先頭列 clip_id・plan の順・重複なし）＋ 同名 .sha256（plan の sha256 と本数）
使い方: python scripts/make_val_manifest_v2final.py [--plan <csv>] [--out <csv>] [--check]
  --check = 書かずに、既存の名簿が今の plan の val と一致するかだけ見る（終了コード 0/1）
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
OUT = ROOT / "out/dataset_outdoor_siren_v2final/plan/manifest_val_syn_v2final.csv"


def val_clips(plan: Path) -> list[str]:
    with open(plan, newline="", encoding="utf-8") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f) if r["split"] == "val"]
    assert len(clips) == len(set(clips)) and all(c.startswith("val_syn_clip") for c in clips), "val の番号が重複か val_syn_ でない"
    return clips


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    a = sys.argv[1:]
    plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else PLAN
    out = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
    clips = val_clips(plan)
    text = "clip_id\n" + "".join(c + "\n" for c in clips)
    if "--check" in a:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print(f"{'一致' if same else '不一致'}: {out.name}（plan の val {len(clips):,} 本）")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    out.with_suffix(".sha256").write_text(f"{sha256(out)}  {out.name}\n{sha256(plan)}  {plan.name}\nn_val {len(clips)}\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}（{len(clips):,} 本・plan {plan.name} sha256 {sha256(plan)[:16]}…）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
