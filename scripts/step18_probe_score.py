# -*- coding: utf-8 -*-
"""Step 18: レベル正規化プローブの採点（音量なしで音色識別できるかの証明、案A''）。

v9.1時代のアドホック集計を正式スクリプト化（2026-07-22、Fable）。
プローブ48本（6クラス×8、fold9_room1）は受聴A特性レベルを共通値に正規化してあり、
「音量という手がかりなしでクラスを当てられるか」を問う。採点はプローブ窓
（PROBE_WIN=3.0-7.0s）内の予測フレームで行う:
  - 正解 = 窓内の最頻予測クラスがGTクラスに一致
  - 純度 = 窓内予測フレームのうちGTクラスの割合

使い方: python scripts/step18_probe_score.py \
    [--ds out/dataset_outdoor_siren_v10] [--pred out/predictions_v10_2] \
    [--out out/step12_notify_v10_2] [--title v10.2 run1]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


DS = ROOT / _arg("--ds", "out/dataset_outdoor_siren_v10")
PRED = ROOT / _arg("--pred", "out/predictions_v10_2")
OUT = ROOT / _arg("--out", "out/step12_notify_v10_2")
TITLE = _arg("--title", "v10.2 run1")

CLS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
           "car_drive": 4, "crossing": 5}
WIN = (30, 70)   # PROBE_WIN 3.0-7.0s → フレーム30-70


def main():
    pred = defaultdict(lambda: defaultdict(list))
    for line in open(PRED / "probe_all.csv"):
        p = line.strip().split(",")
        if len(p) >= 5:
            pred[p[0]][int(p[1])].append(int(p[2]))

    rows = list(csv.DictReader(open(DS / "plan" / "assignment_probe.csv")))
    n_ok = 0
    purities = []
    misses = []
    by_class = defaultdict(lambda: [0, 0])
    for row in rows:
        clip = row["clip_id"]
        scene = json.loads((DS / "work" / clip / "scene.json").read_text())
        assert len(scene["sources"]) == 1, f"{clip}: プローブは単一音源のはず"
        gt = CLS_IDX[scene["sources"][0]["class"]]
        votes = Counter()
        for k in range(WIN[0], WIN[1]):
            for c in pred[clip].get(k, []):
                votes[c] += 1
        total = sum(votes.values())
        dom = votes.most_common(1)[0][0] if votes else None
        ok = dom == gt
        n_ok += int(ok)
        by_class[scene["sources"][0]["class"]][1] += 1
        by_class[scene["sources"][0]["class"]][0] += int(ok)
        if total:
            purities.append(votes.get(gt, 0) / total)
        if not ok:
            misses.append(f"{clip}: GT={scene['sources'][0]['class']} "
                          f"pred={dom} votes={dict(votes)}")

    n = len(rows)
    rep = [f"# プローブ採点（レベル正規化・音色識別の証明。{TITLE}）", "",
           f"- 正解（窓内最頻クラス=GT）: **{n_ok}/{n}**",
           f"- 純度（窓内予測のGTクラス率）: 中央値 {np.median(purities):.1%} / "
           f"最小 {min(purities):.1%}" if purities else "- 純度: 予測なし",
           "- クラス別: " + " / ".join(
               f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_class.items())), ""]
    if misses:
        rep += ["## 不正解の内訳"] + [f"- {m}" for m in misses] + [""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_summary.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
