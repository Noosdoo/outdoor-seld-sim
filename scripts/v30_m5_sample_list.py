# -*- coding: utf-8 -*-
"""−5 dB の小標本の本の一覧を決める（本体 §9-1 の 8・§9-4 の 6・2026-10-26）。実行（描く・数える）の前に一覧を固定してコミットする。描かない。

選び方（決まった手順・乱数の種 2026・評価の本と val は使わない＝plan の train_syn だけ・高さの双子〔twin_of〕は外す）:
  1. plan（out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv・v30 も同じ plan）の train の行を random.Random(2026) で並べ替えて順に見る。
  2. 各行の v30 の場面（step11_v30_render.sample_scene・描かない）から層を 1 つ決める（上が優先）。**対象 = 正解のある音源のうち電車の車両
     （class crossing の params.train_car・先頭の車両などに正解が付くものも）を除いたもの**（仕様 §9-1 の 8: 電車の車両は対象から外し背景に残す）:
       S4 電車＋警報機 = 対象に踏切の警報機（crossing・train_car でない）があり、同じ本に電車の車両がある（v19mix に約 1.2%）
       S3 同じクラスの複数音源 = 対象のうち同じクラスが 2 つ以上（警報機の左右 2 つもここ）
       S2 雨あり = plan の rain が空でない
       S5 弱い音のクラス = 対象のクラスに bicycle・kickboard・bike_bell のどれかがある
       S1 ふつう = 上のどれでもない（対象が 1 つ以上）
     対象が無い本（背景だけ・電車の車両だけ）は選ばない。
  3. 背景の形（step11_v30_render.bg_shape_of・本ごとの別の乱数・描く前に決まる）の折れ点で 3 群: low = 折れ点 ≤ 100 Hz・mid = 200〜400 Hz・high = ≥ 630 Hz
     （折れ点は 63〜1,000 Hz の対数一様・両端と中間）。どれにも入らない本は選ばない。
  4. 枠: S1 40（low 14・mid 13・high 13）・S2〜S5 各 20（low 7・mid 7・high 6）＝ 計 120 本。埋まった枠は飛ばす。
出力（列）: clip_id,stratum,shape_group,knee_hz,slope_db_oct,rain,scenario,labeled_classes（対象のクラス）,n_labeled（対象の数）,n_nolabel,n_train_car,seed
使い方: python scripts/v30_m5_sample_list.py --out md/design/v30_m5_小標本_本の一覧_2026-10-26.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 2026
QUOTA = {"S1": {"low": 14, "mid": 13, "high": 13}, "S2": {"low": 7, "mid": 7, "high": 6}, "S3": {"low": 7, "mid": 7, "high": 6},
         "S4": {"low": 7, "mid": 7, "high": 6}, "S5": {"low": 7, "mid": 7, "high": 6}}
WEAK = {"bicycle", "kickboard", "bike_bell"}   # 場面のクラス名（kick は kickboard）
FIELDS = ["clip_id", "stratum", "shape_group", "knee_hz", "slope_db_oct", "rain", "scenario", "labeled_classes", "n_labeled", "n_nolabel", "n_train_car", "seed"]


def shape_group(knee: float) -> str | None:
    if knee <= 100.0:
        return "low"
    if 200.0 <= knee <= 400.0:
        return "mid"
    if knee >= 630.0:
        return "high"
    return None


def is_train_car(s: dict) -> bool:
    return bool((s.get("params") or {}).get("train_car"))


def targets(scene: dict) -> list[dict]:
    """対象 = 正解のある音源のうち電車の車両を除いたもの。"""
    return [s for s in scene["sources"] if not s.get("no_label") and not is_train_car(s)]


def stratum_of(row: dict, scene: dict) -> str | None:
    lab = targets(scene)
    if not lab:
        return None
    cls = Counter(s["class"] for s in lab)
    if "crossing" in cls and any(is_train_car(s) for s in scene["sources"]):
        return "S4"
    if any(n >= 2 for n in cls.values()):
        return "S3"
    if (row.get("rain") or "").strip():
        return "S2"
    if WEAK & set(cls):
        return "S5"
    return "S1"


def pick(rows: list[dict], v30) -> list[dict]:
    cand = [r for r in rows if r["clip_id"].startswith("train_syn") and r.get("split") == "train" and not (r.get("twin_of") or "").strip()]
    random.Random(SEED).shuffle(cand)
    left = {s: dict(q) for s, q in QUOTA.items()}
    out = []
    for r in cand:
        if not any(v for q in left.values() for v in q.values()):
            break
        shp = v30.bg_shape_of({}, r)
        g = shape_group(float(shp["knee_hz"]))
        if g is None:
            continue
        sc = v30.sample_scene(dict(r))
        st = stratum_of(r, sc)
        if st is None or left[st][g] <= 0:
            continue
        left[st][g] -= 1
        lab = targets(sc)
        out.append({"clip_id": r["clip_id"], "stratum": st, "shape_group": g, "knee_hz": round(float(shp["knee_hz"]), 3),
                    "slope_db_oct": round(float(shp["slope_db_oct"]), 4), "rain": (r.get("rain") or "").strip(), "scenario": r.get("scenario", ""),
                    "labeled_classes": "+".join(sorted(s["class"] for s in lab)), "n_labeled": len(lab),
                    "n_nolabel": sum(1 for s in sc["sources"] if s.get("no_label")), "n_train_car": sum(1 for s in sc["sources"] if is_train_car(s)), "seed": r["seed"]})
    short = {s: {g: n for g, n in q.items() if n > 0} for s, q in left.items() if any(n > 0 for n in q.values())}
    if short:
        raise SystemExit(f"枠が埋まらない: {short}（候補 {len(cand)} 本を全部見た）")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"一覧が既にある: {a.out}（固定した一覧は上書きしない）")
    import _run_v2final_gen as g
    import step11_v30_render as v30
    rows = g.load_plan()
    out = pick(rows, v30)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)
    c = Counter((r["stratum"], r["shape_group"]) for r in out)
    print(f"{len(out)} 本 → {a.out}: " + "・".join(f"{s}/{gg} {c[(s, gg)]}" for s in QUOTA for gg in ("low", "mid", "high")))
    print("クラス（正解のある音源の数）: " + "・".join(f"{k} {v}" for k, v in sorted(Counter(x for r in out for x in r["labeled_classes"].split("+")).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
