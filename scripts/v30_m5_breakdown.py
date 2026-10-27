# -*- coding: utf-8 -*-
"""−5 dB の小標本の内訳（本体 §9-1 の 8・2026-10-26）: out/v30_m5_small_sample/result.json と本の一覧から、層別・単音源の本・背景の形の 3 群・一番有利な帯・区切りを出す（結果の md の数の出どころ）。
帯域ゲートの列（Fable 10/26 18:3x・描き直さずに stem から数えた keep_band_0／keep_band_3）が JSON にあれば、クラス別・層別・単音源の本で 旧 0／新 −5／帯 0／帯 +3 を並べる。
使い方: python scripts/v30_m5_breakdown.py [<result json>（既定 out/v30_m5_small_sample/result.json）] > out/v30_m5_small_sample/breakdown.txt
"""
import csv, json, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)))); sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pathlib import Path as _P
R = str(_P(__file__).resolve().parents[1])
JP = sys.argv[1] if len(sys.argv) > 1 else R + r"\out\v30_m5_small_sample\result.json"
J = json.load(open(JP, encoding="utf-8"))
L = {r["clip_id"]: r for r in csv.DictReader(open(R + r"\md\design\v30_m5_小標本_本の一覧_2026-10-26.csv", encoding="utf-8"))}
import step11_v13_render as v13, step11_v30_render as v30
print("json", JP)
print("GATED_V30", sorted(v30.GATED_V30))
print("error" in J, J.get("error"))
KEYS = ("keep_old", "keep_new", "keep_band_0", "keep_band_3")       # 旧 0／新 −5／帯 0／帯 +3


def fmt(v):
    """[枠, Σ旧, Σ新, Σ帯0, Σ帯3] → 旧/新/帯0/帯3（%）。帯の列が無ければ 旧/新 だけ。"""
    n = v[0]
    parts = [f"{100 * v[1] / n:.1f}", f"{100 * v[2] / n:.1f}"]
    if v[3] is not None:
        parts += [f"{100 * v[3] / n:.1f}", f"{100 * v[4] / n:.1f}"]
    return "/".join(parts)


for arm in ("off", "on"):
    A = J["arms"][arm]
    band = all("keep_band_0" in p for p in A["per_source"])
    print(f"== {arm} diff", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in A["diff_vs_gate"].items()})
    print(f" 並び = 旧 0 以上／新 −5 以上" + ("／帯 0 以上／帯 +3 以上" if band else "") + "（%・枠で重みづけ）")

    def new_acc():
        return [0, 0.0, 0.0, (0.0 if band else None), (0.0 if band else None)]
    agg = defaultdict(new_acc); agg1 = defaultdict(new_acc); aggs = defaultdict(new_acc); aggst = defaultdict(new_acc); aggc = defaultdict(new_acc)
    for p in A["per_source"]:
        li = L[p["clip"]]
        n = p["n_active"]
        single = int(li["n_labeled"]) == 1 and int(li["n_nolabel"]) == 0
        for d, key, use in ((agg, (li["stratum"], p["class"]), True), (agg1, p["class"], single), (aggs, (li["shape_group"], p["class"]), True),
                            (aggst, li["stratum"], True), (aggc, p["class"], True)):
            if not use:
                continue
            v = d[key]
            v[0] += n
            for j, k in enumerate(KEYS):
                if j < 2 or band:
                    v[j + 1] += n * p[k]
    print(" クラス別: " + "・".join(f"{c} {fmt(v)}" for c, v in sorted(aggc.items())))
    print(" 層別:")
    for st in ("S1", "S2", "S3", "S4", "S5"):
        print(f"  {st}: 枠 {aggst[st][0]}・{fmt(aggst[st])}  " + "・".join(f"{c} {fmt(v)}" for (s, c), v in sorted(agg.items()) if s == st))
    print(" 単音源の本（対象 1・ほかの音源なし）: " + "・".join(f"{c} 枠{v[0]} {fmt(v)}" for c, v in sorted(agg1.items())))
    print(" 形の 3 群（全クラス・新 −5）: " + "・".join(f"{g} {100 * sum(v[2] for (gg, c), v in aggs.items() if gg == g) / sum(v[0] for (gg, c), v in aggs.items() if gg == g):.1f}" for g in ("low", "mid", "high")))
    print(" 一番有利な帯")
    for cls, r in A["classes"].items():
        for lo in ("-10", "-5"):
            b = r.get(f"best_band_{lo}")
            if b:
                print(f"  {cls:12s} [{lo},{int(lo) + 5}) n {b['n']:4d}・全部 中央 {b['all_med']:+.1f} [{b['all_p10']:+.1f},{b['all_p90']:+.1f}]・暗騒音のみ 中央 {b['noise_med']:+.1f}・帯 {b['top_fc']}")
    print(" 区切り（新）: " + " | ".join(f"{cls} " + ",".join(f"{100 * v:.0f}" for v in r["bins_new"].values()) for cls, r in A["classes"].items()))
