"""学習データの計画（31,274 行）から場面を作り直し（音は作らない）、ベルが「最接近（通り過ぎ）の後」に鳴り始める本を数える。
最接近はベルの音源の軌道とマイクの位置（歩くマイクも）から 0.01 秒きざみで求める。リポジトリは変更しない（結果は表示だけ）。
使い方: DynamicSound/.venv の python で `python scripts/_count_bell_after_cpa.py`（2 分ほど）。2026-10-08 Opus・変更票 #53 の根拠。
2026-10-08 の結果: v11core 68/1,548（4.4%）・v19bellpin 32/774（4.1%）・v19bellbike 218/500（43.6%）・全体 318/2,822（11.3%）。"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "scripts"), str(R / "src")]
import step11_v22_render as v

m9 = v.m9
P = R / "out/dataset_outdoor_siren_v2final/plan"
plan = list(csv.DictReader((P / "assignment_v2final.csv").open(encoding="utf8")))
tgrid = np.arange(0.0, 10.0 + 1e-9, 0.01)

clips = Counter(); clips_after = Counter(); srcs = Counter(); srcs_after = Counter()
for j, r in enumerate(plan):
    r["seed"] = int(r["seed"]); r["n_warnings"] = int(r["n_warnings"] or 0)
    sc = v.sample_scene_v19(r)
    bells = [s for s in sc["sources"] if s["class"] == "bike_bell"]
    if not bells:
        continue
    mic_arr = sc["_mic_arr"]
    mic = np.array([m9._mic_pos_at(mic_arr, t) for t in tgrid])
    any_after = False
    for s in bells:
        (t0, *p0), (t1, *p1) = s["wp"][0], s["wp"][-1]
        p0, p1 = np.array(p0, float), np.array(p1, float)
        pos = p0[None, :] + (tgrid[:, None] - t0) / (t1 - t0) * (p1 - p0)[None, :]
        t_cpa = float(tgrid[np.argmin(np.linalg.norm(pos - mic, axis=1))])
        after = float(s["t_on"]) > t_cpa + 1e-6
        key = (r["scenario"], s.get("grammar", ""))
        srcs[key] += 1; srcs_after[key] += after
        any_after |= after
    clips[r["scenario"]] += 1; clips_after[r["scenario"]] += any_after
    if (j + 1) % 5000 == 0:
        print("sampled", j + 1, flush=True)

out = {"clips": {k: [clips_after[k], clips[k]] for k in clips},
       "sources": {f"{a}|{b}": [srcs_after[(a, b)], srcs[(a, b)]] for (a, b) in srcs},
       "total_clips": [sum(clips_after.values()), sum(clips.values())],
       "total_sources": [sum(srcs_after.values()), sum(srcs.values())]}
print(json.dumps(out, ensure_ascii=False, indent=1))
