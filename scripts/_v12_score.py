# -*- coding: utf-8 -*-
"""v12採点: 8クラス可聴recall/方向・車バリアント別（EV/大型/通常）・距離。

使い方: python scripts/_v12_score.py <pred_val_all.csv> <出力dir>
- クラス表: v12のmetadata/masksをGTに、_e0_probe_scoreの集計をDS差替で流用
- 車バリアント別: 単独車クリップ（曖昧性なし）に限定して car recall を分解
- 距離: _score_sde_dist を META差替（v12 metadata_dist）で流用
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS12 = ROOT / "out" / "dataset_outdoor_siren_v12"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e0s = _load("e0s", ROOT / "scripts" / "_e0_probe_score.py")
e0s.DS = DS12
e0s.CLASSES_JA = ["サイレン", "クラクション", "バック音", "自転車ベル", "車", "踏切/列車",
                  "キックボード", "バイク"]


def _manifest(pred):
    """【Sol再監査・条件2】予測ゼロのクリップも分母に含める全クリップ巡回。"""
    prefixes = {c.rsplit("_mix", 1)[0] for c in pred}
    return sorted({p.stem for p in (DS12 / "metadata").glob("*.csv")
                   if p.stem.rsplit("_mix", 1)[0] in prefixes} | set(pred))


def class_table8(pred):
    stats = {c: {"tp": 0, "n": 0, "errs": []} for c in range(8)}
    for clip in _manifest(pred):
        gt = defaultdict(dict)
        for line in open(DS12 / "metadata" / f"{clip}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                gt[int(q[0])][int(q[1])] = (float(q[3]), float(q[4]))
        mask = {}
        with open(DS12 / "masks" / f"{clip}.csv") as f:
            next(f)
            for line in f:
                q = line.strip().split(",")
                mask[(int(q[0]), int(q[1]))] = float(q[2])
        for k, evs in gt.items():
            pk = pred.get(clip, {}).get(k, {})
            for c, (gaz, gel) in evs.items():
                if mask.get((k, c), 99.0) < 0.0:
                    continue
                stats[c]["n"] += 1
                if c in pk:
                    stats[c]["tp"] += 1
                    stats[c]["errs"].append(e0s.ang_err(gaz, gel, *pk[c]))
    return stats


def car_variant_recall(pred):
    """単独車クリップ限定の車recallをバリアント別に分解。"""
    out = {v: {"tp": 0, "n": 0} for v in ("normal", "heavy", "ev")}
    for clip in _manifest(pred):
        sj = DS12 / "work" / clip / "scene.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        cars = [x for x in s["sources"] if x["class"] == "car_drive"]
        if len(cars) != 1:
            continue
        var = cars[0].get("car_variant", "normal")
        mask = {}
        with open(DS12 / "masks" / f"{clip}.csv") as f:
            next(f)
            for line in f:
                q = line.strip().split(",")
                mask[(int(q[0]), int(q[1]))] = float(q[2])
        for line in open(DS12 / "metadata" / f"{clip}.csv"):
            q = line.strip().split(",")
            if len(q) == 5 and int(q[1]) == 4:
                k = int(q[0])
                if mask.get((k, 4), 99.0) < 0.0:
                    continue
                out[var]["n"] += 1
                if 4 in pred.get(clip, {}).get(k, {}):
                    out[var]["tp"] += 1
    return out


def main():
    pred_path, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    pred = e0s.load_pred(pred_path)
    st = class_table8(pred)
    lines = [f"# v12採点: {pred_path}", "", "## クラス別 可聴(SNR≥0dB)検出率 / 方向誤差中央値",
             "", "| クラス | 検出率 | n | 方向誤差 |", "| --- | --- | --- | --- |"]
    for c in range(8):
        s = st[c]
        rec = f"{s['tp']/s['n']:.2%}" if s["n"] else "n/a"
        le = f"{np.median(s['errs']):.2f}°" if s["errs"] else "n/a"
        lines.append(f"| {e0s.CLASSES_JA[c]} | {rec} | {s['n']:,} | {le} |")
    cv = car_variant_recall(pred)
    lines += ["", "## 車バリアント別recall（単独車クリップ限定）", "",
              "| バリアント | 検出率 | n |", "| --- | --- | --- |"]
    for v, d in cv.items():
        rec = f"{d['tp']/d['n']:.2%}" if d["n"] else "n/a"
        lines.append(f"| {v} | {rec} | {d['n']:,} |")
    (outdir / "class_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 距離採点（_score_sde_dist を META差替で流用）
    sd = _load("sd", ROOT / "scripts" / "_score_sde_dist.py")
    sd.PRED = pred_path
    sd.META = DS12 / "metadata_dist"
    sd.OUT = outdir
    sd.main()


if __name__ == "__main__":
    main()
