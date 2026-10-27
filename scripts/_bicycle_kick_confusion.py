# -*- coding: utf-8 -*-
"""キックボード（6）と自転車（8）の取り違えを数える（2026-10-08・変更票 #48 の開発一周の合格線「取り違え 5% 以下」用）。

数え方（知覚層・通知の閾値を通さない・方位 20° 以内の 1 対 1 対応・frame ≥ 40）:
  - 正解が kick のフレームで、20° 以内に bicycle の予測があり kick の予測が無い → 「kick → bicycle の取り違え」
  - 正解が bicycle のフレームで、20° 以内に kick の予測があり bicycle の予測が無い → 「bicycle → kick の取り違え」
  - 分母 = そのクラスの正解フレーム数（frame ≥ 40）。率 = 取り違え／分母。参考に「正しいクラスの予測が 20° 以内にある率（検出率）」も出す。
入力: 予測 csv（7 列 clip,frame,class,track,az,el,dist）・正解の metadata_dist ディレクトリ（6 列 frame,class,track,az,el,dist）・名簿（plan の csv・clip_id 列・--scenario で絞れる）
使い方: python scripts/_bicycle_kick_confusion.py --pred <csv> --meta <dir> --plan <csv> [--scenario v19bicycle,v12kick,v19kickfar] [--split val] [--json <out>]
        python scripts/_bicycle_kick_confusion.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    from outdoor_seld.clip_ids import normalize as _normalize
except Exception:  # pragma: no cover
    _normalize = lambda s: s  # noqa: E731

KICK, BICYCLE = 6, 8
AZ_PAIR = 20.0
FRAME_MIN = 40


def _ang(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def load_pred(path: Path) -> dict:
    out = defaultdict(lambda: defaultdict(list))     # clip -> frame -> [(cls, az)]
    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 5 or not p[1].strip().isdigit():
                continue
            out[_normalize(p[0].strip())][int(p[1])].append((int(float(p[2])), float(p[4])))
    return out


def load_gt(meta: Path, clip: str) -> dict:
    out = defaultdict(list)                              # frame -> [(cls, az)]
    p = meta / f"{clip}.csv"
    if not p.is_file():
        return out
    with open(p, newline="", encoding="utf-8") as f:
        for ln in f:
            q = ln.strip().split(",")
            if len(q) < 5 or not q[0].strip().isdigit():
                continue
            out[int(q[0])].append((int(float(q[1])), float(q[3])))
    return out


def count(pred: dict, gt_of, clips: list) -> dict:
    res = {c: {"gt": 0, "confused": 0, "detected": 0} for c in (KICK, BICYCLE)}
    other = {KICK: BICYCLE, BICYCLE: KICK}
    for clip in clips:
        gt = gt_of(clip); pr = pred.get(clip, {})
        for fr, rows in gt.items():
            if fr < FRAME_MIN:
                continue
            for cls, az in rows:
                if cls not in res:
                    continue
                res[cls]["gt"] += 1
                near = [(c, a) for c, a in pr.get(fr, []) if _ang(a, az) <= AZ_PAIR]
                if any(c == cls for c, _ in near):
                    res[cls]["detected"] += 1
                elif any(c == other[cls] for c, _ in near):
                    res[cls]["confused"] += 1
    for c in res:
        n = res[c]["gt"]
        res[c]["confusion_rate"] = (res[c]["confused"] / n) if n else None
        res[c]["detection_rate"] = (res[c]["detected"] / n) if n else None
    return {"kick_to_bicycle": res[KICK], "bicycle_to_kick": res[BICYCLE], "az_pair_deg": AZ_PAIR, "frame_min": FRAME_MIN}


def selftest() -> int:
    pred = {"c1": {50: [(BICYCLE, 10.0)], 51: [(KICK, 10.0)], 52: [(BICYCLE, 100.0)], 53: []},
            "c2": {50: [(KICK, -30.0)], 51: [(BICYCLE, -30.0)], 52: [(KICK, -30.0), (BICYCLE, -30.0)]}}
    gts = {"c1": {50: [(KICK, 12.0)], 51: [(KICK, 12.0)], 52: [(KICK, 12.0)], 53: [(KICK, 12.0)], 10: [(KICK, 12.0)]},
           "c2": {50: [(BICYCLE, -28.0)], 51: [(BICYCLE, -28.0)], 52: [(BICYCLE, -28.0)]}}
    r = count(pred, lambda c: gts[c], ["c1", "c2"])
    k, b = r["kick_to_bicycle"], r["bicycle_to_kick"]
    ok = [("kick: 分母 4（frame<40 は除く）", k["gt"] == 4), ("kick: 取り違え 1（50 のみ・52 は 90° 離れ・53 は予測なし）", k["confused"] == 1),
          ("kick: 検出 1（51）", k["detected"] == 1), ("bicycle: 分母 3・取り違え 1（50）・検出 2（51・52 は両方あるので検出）", b["gt"] == 3 and b["confused"] == 1 and b["detected"] == 2),
          ("率", abs(k["confusion_rate"] - 0.25) < 1e-9 and abs(b["confusion_rate"] - 1 / 3) < 1e-9)]
    n = 0
    for name, passed in ok:
        print(("PASS " if passed else "FAIL ") + name); n += int(passed)
    print(f"{n}/{len(ok)} " + ("ALL PASS" if n == len(ok) else "NG"))
    return 0 if n == len(ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--meta"); ap.add_argument("--plan"); ap.add_argument("--scenario", default="")
    ap.add_argument("--split", default=""); ap.add_argument("--json", default=""); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    scen = {s for s in a.scenario.split(",") if s}
    clips = []
    with open(ROOT / a.plan if not Path(a.plan).is_absolute() else a.plan, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if scen and r.get("scenario") not in scen:
                continue
            if a.split and r.get("split") != a.split:
                continue
            clips.append(_normalize(r["clip_id"]))
    meta = Path(a.meta) if Path(a.meta).is_absolute() else ROOT / a.meta
    pred = load_pred(Path(a.pred) if Path(a.pred).is_absolute() else ROOT / a.pred)
    r = count(pred, lambda c: load_gt(meta, c), clips)
    r["n_clips"] = len(clips); r["scenario"] = sorted(scen); r["split"] = a.split
    for key, lab in (("kick_to_bicycle", "kick → bicycle"), ("bicycle_to_kick", "bicycle → kick")):
        d = r[key]
        print(f"{lab}: 正解フレーム {d['gt']:,}・取り違え {d['confused']:,}（{(d['confusion_rate'] or 0) * 100:.2f}%）・正しいクラスの検出 {d['detected']:,}（{(d['detection_rate'] or 0) * 100:.1f}%）")
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
