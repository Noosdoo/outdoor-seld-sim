# -*- coding: utf-8 -*-
"""v30 の学習データの正解が空の本の内訳と、クラス別の残存率（2026-10-27 Opus・Fable 18:5x の門 B）。読み取りだけ（描かない・書き換えない）。

入力: <ds>/work/<clip>/scene.json（予定の音源・t_on／t_off・gate_band_max_db）・<ds>/metadata/<clip>.csv（最終の正解・D14 と隙間埋めのあと）・
      <ds>/plan/assignment_v2final.csv（scenario・rain）。
数えるもの:
  1. 空の本（metadata の行が 0）の内訳: (a) 設計上の負例 =〔a1〕予定の音源（9 クラス）が 1 つも無い本・〔a2〕D14 で空にした電車だけの本（scenario v12train）、
     (b) 予定の音源があるのに全部落ちた本（予定の音源のクラス別・層別〔電車あり > 雨 > 弱い音 > ふつう〕）。
     ゲートの不具合の疑い = 空の本で gate_band_max_db ≥ 0 の音源（電車の車両と D14 の本を除く）の数（0 が期待）。
  2. クラス別の枠の残存率 = 正解の付いた枠 ÷ 鳴っている枠（音源の [t_on, t_off) の 0.1 s 枠・小標本 §8-1 と同じ数え方）。
     警報機は電車の車両（params.train_car）を除く。電車の車両は参考に別。
  3. クラス別の事象の残存率 = 正解の枠が 1 つ以上ある音源 ÷ 予定の音源。
使い方（サーバ）: python scripts/v30_empty_label_breakdown.py <ds> [--out <json>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

CLS = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3, "car_drive": 4, "crossing": 5, "kickboard": 6, "motorcycle": 7, "bicycle": 8}
NAMES = ["サイレン", "クラクション", "バック音", "ベル", "車", "警報機", "キック", "バイク", "自転車"]
WEAK = {"bicycle", "kickboard", "bike_bell"}
FRAMES = 100


def frames_of(t_on: float, t_off: float) -> range:
    a = max(0, int(math.floor(t_on / 0.1 + 1e-9)))
    b = min(FRAMES, int(math.ceil(t_off / 0.1 - 1e-9)))
    return range(a, max(a, b))


def is_train_car(s: dict) -> bool:
    p = s.get("params") or {}
    return s.get("class") == "crossing" and bool(p.get("train_car"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    plan = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(a.ds, "plan", "assignment_v2final.csv"), encoding="utf-8"))}
    sound, kept, ev_all, ev_kept = Counter(), Counter(), Counter(), Counter()
    tc_sound = tc_kept = tc_ev = tc_ev_kept = 0
    outside = Counter()
    n_clip = n_empty = 0
    a1 = a2 = 0
    b_clips, b_cls, b_layer = 0, Counter(), Counter()
    suspect, suspect_ex = 0, []
    mism = 0
    for clip in sorted(plan):
        sp = os.path.join(a.ds, "work", clip, "scene.json")
        mp = os.path.join(a.ds, "metadata", clip + ".csv")
        if not os.path.exists(sp):
            continue
        n_clip += 1
        sc = json.load(open(sp, encoding="utf-8"))
        lab = set()
        if os.path.exists(mp):
            for line in open(mp, encoding="utf-8"):
                p = line.strip().split(",")
                if len(p) >= 3 and p[0].isdigit():
                    lab.add((int(p[0]), int(p[1]), int(p[2])))
        srcs = [s for s in sc.get("sources", []) if s.get("class") in CLS]
        has_train = any(is_train_car(s) for s in srcs)
        targets = [s for s in srcs if not is_train_car(s)]
        scen = plan[clip].get("scenario", "")
        for s in srcs:
            c, trk = CLS[s["class"]], int(s.get("track", 0))
            fr = frames_of(float(s.get("t_on", 0.0)), float(s.get("t_off", 10.0)))
            k = sum(1 for f in fr if (f, c, trk) in lab)
            if is_train_car(s):
                tc_sound += len(fr); tc_kept += k; tc_ev += 1; tc_ev_kept += k > 0
                continue
            sound[c] += len(fr); kept[c] += k; ev_all[c] += 1; ev_kept[c] += k > 0
            if "n_label_frames" in s and scen != "v12train" and int(s["n_label_frames"]) > 0 and k == 0:
                mism += 1
        lab_cls_trk = {(c, t) for (_, c, t) in lab}
        for (c, t) in lab_cls_trk:
            fr_all = set()
            for s in srcs:
                if CLS[s["class"]] == c and int(s.get("track", 0)) == t:
                    fr_all |= set(frames_of(float(s.get("t_on", 0.0)), float(s.get("t_off", 10.0))))
            outside[c] += sum(1 for (f, cc, tt) in lab if cc == c and tt == t and f not in fr_all)
        if lab:
            continue
        n_empty += 1
        if not srcs:
            a1 += 1; continue
        if scen == "v12train":
            a2 += 1; continue
        if not targets:                                   # 電車の車両だけ（v12train 以外）
            a2 += 1; continue
        b_clips += 1
        for s in targets:
            b_cls[CLS[s["class"]]] += 1
            g = s.get("gate_band_max_db")
            if g is not None and float(g) >= 0.0:
                suspect += 1
                if len(suspect_ex) < 10:
                    suspect_ex.append((clip, s["class"], s.get("track"), g, s.get("n_label_frames")))
        layer = ("電車あり" if has_train else "雨" if (plan[clip].get("rain") or "").strip() else
                 "弱い音" if any(s["class"] in WEAK for s in targets) else "ふつう")
        b_layer[layer] += 1
    res = {"n_clip": n_clip, "n_empty": n_empty, "a1_no_sources": a1, "a2_train_only_or_v12train": a2, "b_all_gated": b_clips,
           "b_by_class": {NAMES[c]: b_cls[c] for c in range(9)}, "b_by_layer": dict(b_layer),
           "suspect_band_ge0_in_empty": suspect, "suspect_examples": suspect_ex, "label_count_mismatch": mism,
           "frame_survival": {NAMES[c]: (round(100 * kept[c] / sound[c], 1) if sound[c] else None) for c in range(9)},
           "frames_sounding": {NAMES[c]: sound[c] for c in range(9)},
           "event_survival": {NAMES[c]: (round(100 * ev_kept[c] / ev_all[c], 1) if ev_all[c] else None) for c in range(9)},
           "events": {NAMES[c]: ev_all[c] for c in range(9)},
           "train_car": {"frame_survival": round(100 * tc_kept / tc_sound, 1) if tc_sound else None, "event_survival": round(100 * tc_ev_kept / tc_ev, 1) if tc_ev else None, "events": tc_ev},
           "labels_outside_sounding": {NAMES[c]: outside[c] for c in range(9)}}
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
