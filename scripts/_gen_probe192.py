# -*- coding: utf-8 -*-
"""probe192（6クラス×32=192）を独立データセットに生成。
既存 build_probe96(per=16) と同式で per=32・room=fold9_room3・seed offset=219000。
使い方: python gen_probe192.py [N]   # N指定でスモーク(先頭N本)
"""
import sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v11_eval_render as mev
import step10_v11_eval_plan as pv
m9 = mev.m9
v9 = pv.v9

# --- 独立データセットへ出力先を切替（本体v11_evalに触れない） ---
DSN = "outdoor_siren_probe192"
m9.DS_NAME = DSN
m9.DS = ROOT / "out" / f"dataset_{DSN}"
m9.WORK = m9.DS / "work"
m9.PLAN = m9.DS / "plan"
for sub in ("foa", "metadata", "masks", "work", "plan"):
    (m9.DS / sub).mkdir(parents=True, exist_ok=True)

# --- 192行を build_probe96 と同式で構築（per=32） ---
classes = v9.WARN_CLASSES + ["car_drive"]
per = 32
b = pv.BASE + 219000
rows = []
for ci, cls in enumerate(classes):
    for j in range(per):
        k = ci * per + j
        rows.append(pv.row(
            f"fold9_room3_mix{k+1:04d}",
            "static" if j < per // 2 else "walk",
            0 if cls == "car_drive" else 1,
            "" if cls == "car_drive" else cls, v9.SIDES[j % 2],
            "", "",
            "safe" if cls == "car_drive" else "na",
            v9.SIDES[j % 2] if cls == "car_drive" else "",
            f"probe_{cls}", 1 if cls == "car_drive" else 0, b + k))
assert len(rows) == 192, len(rows)

# --- 割当CSV（step18採点が読む） ---
with open(m9.PLAN / "assignment_probe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# --- 生成 ---
N = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
import time
t0 = time.time()
for i, r in enumerate(rows[:N]):
    m9.generate_clip(r)
    if (i + 1) % 16 == 0 or i + 1 == N:
        el = time.time() - t0
        print(f"{i+1}/{N}  {el:.0f}s  {el/(i+1):.2f}s/clip", flush=True)
print(f"=== probe192 gen done: {N} clips ===")
