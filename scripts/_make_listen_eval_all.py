# -*- coding: utf-8 -*-
"""評価専用3,504本（新設3,246＋v10共用258）を全部、試聴用モノラルWAVに変換。

- 音=Wチャンネル（無指向成分）を試聴用に正規化（物理較正の正はflac側）
- セット別サブフォルダ＋場面情報入りファイル名
出力: out/listen_v11_eval_all/<セット名>/mixNNNN_<詳細>.wav（約1.7GB）
使い方: python scripts/_make_listen_eval_all.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "listen_v11_eval_all"
DSE = ROOT / "out" / "dataset_outdoor_siren_v11_eval"
DS10 = ROOT / "out" / "dataset_outdoor_siren_v10"
DSADD = ROOT / "out" / "dataset_outdoor_siren_v10_2_add"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JP = {"siren": "サイレン", "horn": "クラクション", "backup_beep": "バック音",
      "bike_bell": "ベル", "crossing": "踏切", "car_drive": "車",
      "critical": "重大", "caution": "注意", "safe": "安全", "na": "－",
      "static": "静止", "walk": "歩行", "half": "混在"}

# (planセット名, planのあるDS, foaのあるDS, 出力フォルダ名)
NEW_SETS = [
    ("halluc600", "01_幻覚600_車なしサイレン"), ("safe600", "02_safe過剰通知600"),
    ("s1_200", "03_S1踏切200"), ("s2_100", "04_S2背後ベル100"),
    ("s3_100", "05_S3バック車100"), ("s5_200", "06_S5悪条件200"),
    ("cross100", "07_交差点100"), ("multi200", "08_複数車200"),
    ("probe96", "09_プローブ96"),
    ("n1", "10_N1突然出現"), ("n2", "11_N2静音EV"), ("n3", "12_N3駐車場多重"),
    ("n4", "13_N4高速サイレン"), ("n5", "14_N5繁華街"),
    ("n6", "15_N6至近追い越し"), ("n7", "16_N7停車発進"),
]
V10_SETS = [  # (planファイル, planのDS, foaのDS, フォルダ名)
    ("assignment_scenario", DS10, DS10, "20_従来_交差点20"),
    ("assignment_scenario2", DS10, DS10, "21_従来_6シナリオ100"),
    ("assignment_v10a", DS10, DS10, "22_従来_交通量60"),
    ("assignment_probe", DS10, DS10, "23_従来_プローブ48"),
    ("assignment_halluc", DSADD / "plan", DSADD, "24_従来_幻覚30"),
]


def detail(r):
    warn = "警告なし"
    nw = int(r["n_warnings"])
    if nw >= 1:
        warn = JP.get(r["w1_class"], r["w1_class"])
        if nw >= 2:
            warn += "＋" + JP.get(r["w2_class"], r["w2_class"]) if r["w2_class"] \
                else f"×{nw}"
    if r.get("scenario", "").startswith("n3"):
        warn = f"バック音×{nw}"
    ncar = r.get("n_car", "")
    car = (f"車{ncar}台" if ncar != "" else
           ("車あり" if r.get("car_side") else "車なし"))
    tier = JP.get(r.get("danger_tier", "na"), "－")
    return f"{car}_{warn}_{tier}_{JP.get(r['motion'], r['motion'])}"


def convert(rows, foa_ds, folder):
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        stem = r["clip_id"]
        dst = d / f"{stem}_{detail(r)}.wav"
        if not dst.exists():
            x = np.asarray(sf.read(foa_ds / "foa" / f"{stem}.flac")[0], np.float64)
            w = x[:, 0]
            pk = float(np.max(np.abs(w)))
            if pk > 0:
                w = w / pk * 0.85
            sf.write(dst, w.astype(np.float32), 24000, subtype="PCM_16")
        n += 1
    return n


def main():
    total = 0
    for which, folder in NEW_SETS:
        rows = list(csv.DictReader(open(DSE / "plan" / f"assignment_{which}.csv",
                                        encoding="utf-8")))
        total += convert(rows, DSE, folder)
        print(f"{folder}: done ({total}累計)", flush=True)
    for fname, plan_dir, foa_ds, folder in V10_SETS:
        plan_path = (plan_dir / "plan" / f"{fname}.csv"
                     if (plan_dir / "plan").exists() else plan_dir / f"{fname}.csv")
        rows = list(csv.DictReader(open(plan_path, encoding="utf-8")))
        total += convert(rows, foa_ds, folder)
        print(f"{folder}: done ({total}累計)", flush=True)
    print(f"ALL DONE: {total}本 -> {OUT}")
    assert total == 3504, total


if __name__ == "__main__":
    main()
