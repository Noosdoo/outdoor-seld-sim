"""切り分け 3 の採点: 合成の自転車（val）だけのときと、家の背景を足したときで、今のモデルが自転車（クラス 8）を見つける割合が変わるか（2026-10-25 Opus）。
正解 = val の正解（out/_v29_val_gt/metadata_dist・可聴ゲートのあと）。ラベルの無い本でも、自転車を出した行の数は数える（モデルが聞いているかを見るため）。
使い方: python scripts/demo_bike_check_valmix_score.py --pred <val_all_causal.csv> [--dir out/_demo_bike_check_valmix]
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
VAL_GT = E2E / "out" / "_v29_val_gt" / "metadata_dist"
NAMES = {0: "サイレン", 1: "クラクション", 2: "バック音", 3: "ベル", 4: "車", 5: "踏切", 6: "キック", 7: "バイク", 8: "自転車"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, nargs="+", help="予測の csv（複数可・そのままの本と混ぜた本を別の推論で）")
    ap.add_argument("--dir", default=str(E2E / "out" / "_demo_bike_check_valmix"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(Path(a.dir) / "map.csv", encoding="utf-8")))
    pred = defaultdict(set)
    cls_rows = defaultdict(Counter)
    for pf in a.pred:
        for r in csv.reader(open(pf, encoding="utf-8")):
            if len(r) < 7:
                continue
            pred[(r[0], int(r[2]))].add(int(r[1]))
            cls_rows[r[0]][int(r[2])] += 1
    res = {"orig": Counter(), "mix": Counter()}
    other = {"orig": Counter(), "mix": Counter()}
    for m in rows:
        gt = {int(l.split(",")[0]) for l in open(VAL_GT / f"{m['val_id']}.csv", encoding="utf-8") if l.strip() and l.split(",")[1].strip() == "8"}
        for ver, cid in (("orig", m["clip_orig"]), ("mix", m["clip_mix"])):
            det = pred.get((cid, 8), set())
            res[ver]["gt"] += len(gt)
            res[ver]["hit"] += len(gt & det)
            res[ver]["bike_rows_unlabeled_clips"] += len(det) if not gt else 0
            res[ver]["bike_rows_all"] += len(det)
            res[ver]["clips_with_bike_output"] += bool(det)
            res[ver]["labeled_clips"] += bool(gt)
            res[ver]["labeled_clips_hit"] += bool(gt & det)
            for c, n in cls_rows[cid].items():
                if c != 8:
                    other[ver][c] += n
    for ver, name in (("orig", "合成のまま"), ("mix", "合成＋家の背景")):
        r = res[ver]
        rec = r["hit"] / r["gt"] if r["gt"] else float("nan")
        print(f"{name}: ラベルのあるフレームで自転車を見つけた {r['hit']}/{r['gt']}（{100 * rec:.0f}%）・ラベルのある本 {r['labeled_clips_hit']}/{r['labeled_clips']} 本で 1 回以上・"
              f"自転車を出した本 {r['clips_with_bike_output']}/30（ラベルの無い本での自転車の行 {r['bike_rows_unlabeled_clips']}）・ほかのクラスの行 {dict((NAMES[c], n) for c, n in sorted(other[ver].items()))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
