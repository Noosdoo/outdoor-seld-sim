# -*- coding: utf-8 -*-
"""v19 の実現値の台帳（2026-09-08・監査の補足「plan の n_car/n_warnings は v19static では仮の 0」への対応）。
生成済みの work/<clip>/scene.json から、場面ごとに実際の構成を 1 行 1 クリップで書き出す:
  clip_id, scenario, split, motion, mic_z, n_car, n_siren, n_horn, n_sources, car_speed_max_mps, car_min_dist_m,
  warn_dist_m（静止警告音の距離）, kind（v19 の種別）, turn_rev, headway_s, level_adj_db, has_car
使い方: python scripts/_v19_realized_ledger.py --scene fast|static|turn|platoon|all [--ds out/dataset_outdoor_siren_v19]
出力: <ds>/plan/realized_v19_<scene>.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
COLS = ["clip_id", "scenario", "split", "motion", "mic_z", "n_car", "n_siren", "n_horn", "n_sources", "car_speed_max_mps",
        "car_min_dist_m", "warn_dist_m", "kind", "turn_rev", "headway_s", "level_adj_db", "has_car"]


def _arg(name, default):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def row_of(s: dict) -> dict:
    row = s.get("row", {})
    srcs = s.get("sources", [])
    cars = [x for x in srcs if x.get("class") == "car_drive"]
    v19 = s.get("v19", {}) or {}
    warn = [x for x in srcs if x.get("class") in ("siren", "horn")]
    return {
        "clip_id": row.get("clip_id", ""), "scenario": row.get("scenario", ""), "split": row.get("split", ""),
        "motion": (s.get("mic") or {}).get("motion", ""), "mic_z": (s.get("mic") or {}).get("mic_z", ""),
        "n_car": len(cars), "n_siren": sum(x.get("class") == "siren" for x in srcs), "n_horn": sum(x.get("class") == "horn" for x in srcs),
        "n_sources": len(srcs),
        "car_speed_max_mps": (f"{max(float(x.get('speed_mps', 0)) for x in cars):.2f}" if cars else ""),
        "car_min_dist_m": (f"{min(float(x.get('min_dist_m', 99)) for x in cars):.2f}" if cars else ""),
        "warn_dist_m": (f"{float(warn[0].get('dist_draw_m')):.1f}" if warn and warn[0].get("dist_draw_m") is not None else ""),
        "kind": v19.get("kind", ""), "turn_rev": v19.get("turn_rev", ""), "headway_s": v19.get("headway_s", ""),
        "level_adj_db": v19.get("level_adj_db", ""), "has_car": int(bool(cars)),
    }


def main() -> int:
    scene = _arg("--scene", "all")
    ds = ROOT / _arg("--ds", "out/dataset_outdoor_siren_v19")
    scenes = ["fast", "static", "turn", "platoon"] if scene == "all" else [scene]
    for sc in scenes:
        plan = ds / "plan" / f"assignment_v19_{sc}.csv"
        ids = [r["clip_id"] for r in csv.DictReader(open(plan, encoding="utf-8"))]
        rows, missing = [], 0
        for cid in ids:
            sj = ds / "work" / cid / "scene.json"
            if not sj.exists():
                missing += 1
                continue
            rows.append(row_of(json.loads(sj.read_text(encoding="utf-8"))))
        out = ds / "plan" / f"realized_v19_{sc}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        n_car = sum(int(r["has_car"]) for r in rows)
        print(f"v19{sc}: {len(rows)} 本（scene.json 無し {missing}）→ {out.name}・車あり {n_car}・"
              f"siren {sum(r['n_siren'] > 0 for r in rows)}・horn {sum(r['n_horn'] > 0 for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
