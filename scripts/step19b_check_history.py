# -*- coding: utf-8 -*-
"""切り出しが事象の前に 4 秒以上の音を残しているかの確かめ（本体 §2-6 の L3・2026-10-26）。切り出し（step19b_realsmoke_cut.py）は変えない。一覧を出すだけ。

入力 = step19b の --ann-out（切り出し後の注釈・時刻はクリップの先頭からの秒）。class ≠ none・scored ≠ 0 の事象ごとに:
  - margin_s = t_start（クリップの先頭から事象の始まりまで）。4.0 s 未満なら short_margin = 1（clamped = 1 は t_start が 0.00 = 事象の始まりがクリップより前）
  - v30 の暖機で主から外れるか（step20 --warmup-v30 と同じ規則・warmup_reason）: 通過・発進 = 最接近 < 4.0 s（warmup_cpa）・警告音 = 鳴り終わり ≤ 4.0 s（warmup_end）
報告 = 本数・足りない本の一覧（clip・event・class・t_start・t_cpa・理由）。--out で csv。
使い方: python scripts/step19b_check_history.py <ann_clips.csv> [<ann_clips2.csv> ...] [--out out/realsmoke/history_check.csv]
試験: scripts/test_step20_warmup_v30.py の W8（架空の注釈だけ）
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("s20v3_hist", ROOT / "scripts/step20_realsmoke_score_v3.py")
S20 = importlib.util.module_from_spec(_spec); sys.modules["s20v3_hist"] = S20; _spec.loader.exec_module(S20)

MIN_MARGIN_S = S20.WARMUP_S
FIELDS = ["ann", "clip_id", "event_id", "class", "t_start", "t_cpa", "t_end", "margin_s", "short_margin", "clamped", "warmup_reason"]


def check_rows(rows: list[dict], ann_name: str = "") -> list[dict]:
    out = []
    for r in rows:
        cls = (r.get("class") or "").strip()
        if cls in ("", "none") or (r.get("scored") or "1").strip() == "0":
            continue
        ci = S20.CLS_IDX.get(cls)
        try:
            t0 = float(r["t_start"])
        except (KeyError, ValueError):
            continue
        hs, why = S20.warmup_v30(r, ci) if ci is not None else (0, "知らないクラス")
        out.append({"ann": ann_name, "clip_id": r.get("clip_id", ""), "event_id": r.get("event_id", ""), "class": cls, "t_start": r.get("t_start", ""),
                    "t_cpa": r.get("t_cpa", ""), "t_end": r.get("t_end", ""), "margin_s": round(t0, 3), "short_margin": int(t0 < MIN_MARGIN_S),
                    "clamped": int(abs(t0) < 1e-9), "warmup_reason": why})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ann", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = []
    for p in a.ann:
        res += check_rows(list(csv.DictReader(open(p, encoding="utf-8-sig"))), p.name)
    short = [x for x in res if x["short_margin"]]
    excl = [x for x in res if x["warmup_reason"] in ("warmup_cpa", "warmup_end")]
    print(f"事象 {len(res)} 件・事象の前が {MIN_MARGIN_S:.1f} s 未満 {len(short)} 件（うち始まりがクリップより前 {sum(x['clamped'] for x in short)} 件）・"
          f"v30 の暖機で主から外れる {len(excl)} 件（最接近 < {MIN_MARGIN_S:.1f} s {sum(1 for x in excl if x['warmup_reason'] == 'warmup_cpa')}・"
          f"鳴り終わり ≤ {MIN_MARGIN_S:.1f} s {sum(1 for x in excl if x['warmup_reason'] == 'warmup_end')}）")
    for x in short:
        print(f"  {x['ann']} {x['clip_id']}/{x['event_id']} {x['class']} t_start={x['t_start']} t_cpa={x['t_cpa']}"
              + ("（始まりがクリップより前）" if x["clamped"] else "") + (f"・{x['warmup_reason']}" if x["warmup_reason"] else ""))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(res)
        print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
