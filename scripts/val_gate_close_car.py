"""近くを通る（critical）車 1 台の本で、ラベルが 1 フレームも無い本（41 本）と、ある本の違いを plan の列で並べる（2026-10-25 Opus・Astra／Fable の問い）。
val だけ・eval は読まない。選び方は scripts/val_gate_by_tier.py と同じ（n_car = 1・警告音 0・scenario が v19fast／v19turn／v19launch／v11core）。
使い方: python scripts/val_gate_close_car.py --masks <置き場> --meta <置き場>
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median

E2E = Path(__file__).resolve().parents[1]
PLAN = E2E / "out" / "dataset_outdoor_siren_v2final" / "plan" / "assignment_v2final.csv"
CAR_SC = {"v19fast", "v19turn", "v19launch", "v11core"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(PLAN, encoding="utf-8-sig")) if r["clip_id"].startswith("val_syn_")
            and r["scenario"] in CAR_SC and r["n_car"] == "1" and r["n_warnings"] in ("0", "") and r["danger_tier"] == "critical"]
    zero, some = [], []
    for r in rows:
        mp, lp = os.path.join(a.masks, r["clip_id"] + ".csv"), os.path.join(a.meta, r["clip_id"] + ".csv")
        snr = [float(m["snr_a_db"]) for m in csv.DictReader(open(mp, encoding="utf-8")) if int(m["class"]) == 4]
        snr = [s for s in snr if s > -100]
        if not snr:
            continue
        lab = any(l.strip() and l.split(",")[1].strip() == "4" for l in open(lp, encoding="utf-8")) if os.path.exists(lp) else False
        r = dict(r)
        r["_max_snr"] = max(snr)
        (some if lab else zero).append(r)
    print(f"近い車 1 台の本: {len(zero) + len(some)}（ラベル 0 = {len(zero)}・あり = {len(some)}）")
    print(f"  ラベル 0 の本の seed の数: {len({r['seed'] for r in zero})}（同じ seed の高さ違いを別々に数えている・Astra HB25-07）")

    def show(key, f=lambda r, k: r.get(k) or "なし"):
        cz, cs = Counter(f(r, key) for r in zero), Counter(f(r, key) for r in some)
        keys = sorted(set(cz) | set(cs), key=str)
        print(f"  {key}: " + "・".join(f"{k} = {cz[k]}/{cz[k] + cs[k]}" for k in keys) + "（ラベル 0 の本／全部）")

    for k in ("scenario", "scene_type", "rain", "motion", "close_slow", "grammar"):
        show(k)
    rd = lambda r, k: ("雨なし" if not r.get("rain_dba") else ("雨 <50" if float(r["rain_dba"]) < 50 else ("雨 50〜60" if float(r["rain_dba"]) < 60 else "雨 ≥60")))
    show("rain_dba", rd)
    print(f"  車の最大の SN 比の中央値: ラベル 0 = {median(r['_max_snr'] for r in zero):.1f} dB・あり = {median(r['_max_snr'] for r in some):.1f} dB")
    sp = [float(r["cs_speed_kmh"]) for r in zero if r.get("cs_speed_kmh")]
    if sp:
        print(f"  ラベル 0 の本の cs_speed_kmh（ある本だけ）: {sorted(sp)[:10]}…")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
