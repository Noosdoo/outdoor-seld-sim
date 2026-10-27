"""可聴ゲートで消えたラベルが、近い（critical）・中くらい（caution）・遠い（safe）のどこで多いかを数える（2026-10-25 Opus・本人 21:39 の続き）。
音源が 1 つだけの本（plan の n_car と警告音の数で選ぶ・走行音だけの scenario）に限り、本の danger_tier をその音源の圏とみなす。val だけ・eval は読まない。
  車: n_car = 1・警告音 0・scenario が v19fast / v19turn / v19launch / v11core（車 1 台の本）
  キック: v12kick・v19kickfar ／ バイク: v12bike・v19motofar ／ 自転車: v19bicycle（どれも警告音 0）
使い方: python scripts/val_gate_by_tier.py --masks <置き場> --meta <置き場>
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
PLAN = E2E / "out" / "dataset_outdoor_siren_v2final" / "plan" / "assignment_v2final.csv"
SETS = {4: ("車", {"v19fast", "v19turn", "v19launch", "v11core"}), 6: ("キック", {"v12kick", "v19kickfar"}),
        7: ("バイク", {"v12bike", "v19motofar"}), 8: ("自転車", {"v19bicycle"})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    plan = [r for r in csv.DictReader(open(PLAN, encoding="utf-8-sig")) if r["clip_id"].startswith("val_syn_")]
    stat = defaultdict(Counter)
    for r in plan:
        if r["n_warnings"] not in ("0", ""):
            continue
        for c, (name, scs) in SETS.items():
            if r["scenario"] not in scs:
                continue
            if c == 4 and r["n_car"] != "1":
                continue
            if c != 4 and r["n_car"] not in ("0", ""):
                continue
            mp, lp = os.path.join(a.masks, r["clip_id"] + ".csv"), os.path.join(a.meta, r["clip_id"] + ".csv")
            if not os.path.exists(mp):
                continue
            best = {}
            for m in csv.DictReader(open(mp, encoding="utf-8")):
                if int(m["class"]) == c:
                    k = int(m["frame"])
                    best[k] = max(best.get(k, -1e9), float(m["snr_a_db"]))
            lab = {int(l.split(",")[0]) for l in open(lp, encoding="utf-8") if l.strip() and l.split(",")[1].strip() == str(c)} if os.path.exists(lp) else set()
            pres = [k for k, s in best.items() if s > -100]
            if not pres:
                continue
            key = (c, r["danger_tier"] or ("空（" + r["scenario"] + "）"))   # Astra HB25-07: 圏が空の組（v19kickfar・v19motofar）も別に出す
            stat[key]["clips"] += 1
            stat[key]["present"] += len(pres)
            stat[key]["kept"] += sum(1 for k in pres if k in lab)
            stat[key]["zero"] += not any(k in lab for k in pres)
            stat[key]["m10_0"] += sum(1 for k in pres if k not in lab and -10 <= best[k] < 0)
    for c, (name, _) in SETS.items():
        for tier in ["critical", "caution", "safe"] + sorted({k[1] for k in stat if k[0] == c and k[1] not in ("critical", "caution", "safe")}):
            t = stat[(c, tier)]
            if not t["clips"]:
                continue
            print(f"  {name}・{tier}: {t['clips']} 本・鳴っているフレームのうち残った {100 * t['kept'] / t['present']:.0f}%・消えた −10〜0 dB {100 * t['m10_0'] / t['present']:.0f}%・"
                  f"ラベルが全部消えた本 {t['zero']}/{t['clips']}（{100 * t['zero'] / t['clips']:.0f}%）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
