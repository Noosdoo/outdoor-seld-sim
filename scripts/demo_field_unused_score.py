# -*- coding: utf-8 -*-
"""開発用の学内の本（デモの組）の採点（2026-10-27 Opus・一覧 md §10・決め直し 6）。家の 13 本（demo_bike_check_score.py）とは別の表で、足し合わせない。

入力: --infer（demo_field_unused_chain.py の出力: gt/<clip>.csv〔step19k・6 列 frame,class,track,az,el,dist〕・ann_all.csv・exclusions.csv）と
      --pred（因果推論の val_all_causal.csv・7 列 clip,frame,class,track,az,el,dist）。
採点（評価の本と同じ考え方・家の 13 本と同じ指標）:
  - 距離のクラス（自転車 8 など）: 採点するフレーム（暖機 4.0 s のあと・40〜99）から除外区間のフレームを引いた差集合で、正解のあるフレームのうち
    同じクラスの予測があるフレームの割合（見つけた割合）・同じクラスの予測のうち一番近い行の向きの誤差（度）・距離の誤差（m）。正解の残りが 0 なら欠測。
  - 警告音（ベル 3）: 鳴り始め（注釈の t_start − cut_offset）で、成功の窓 [−1, +4] s と、その前 4.0 s の履歴を全部観察できる本だけ「届いた／届かない」を数える。
    除外区間が重なれば「部分観察」（成功にも失敗にも混ぜない）。除外中の発火は数えない。向きは鳴り始めから 1 s の行。
  - ほかのクラスの発火（その本の正解に無いクラス）は、除外区間の外で数える（参考）。
  - 参考（2026-10-27・本人 12:35 (a)(b)・scripts/_class_confusion.py）: 取り違えの表（フレーム・正解の種類 × 出した種類＋なし）と、
    本ごとの種類の正答率（成功の窓の中で発火の一番多い種類・発火なしは未検出）。主の指標は変えない。
出力: --out の json（本ごとの行・表の頭の数・区分ごとの集計・confusion・take_accuracy）と、画面の表（入れる 15 と除外区間つき 4 は行に印）。
使い方: python scripts/demo_field_unused_score.py --infer out/_demo_field_unused_149/infer --pred <val_all_causal.csv> --out <json> [--label 名前]
試験: scripts/test_demo_field_unused_score.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import field_exclusion as XC  # noqa: E402
import _class_confusion as CC  # noqa: E402

CLASS_IDX = {"bicycle": 8, "bike_bell": 3, "car_drive": 4, "car": 4, "horn": 1, "backup_beep": 2, "crossing": 5, "kick": 6, "bike": 7}


def take_window(r: dict, scored: set, ex_f: set) -> set:
    """本ごとの種類の窓（§6 の成功の窓と同じ）: 警告音 = [鳴り始め − 1, 鳴り始め + 4] s・距離のクラス = [t_start − 1, t_cpa + 1] s（切り出しの秒）。暖機のあと・除外区間を引く。"""
    off = float(r["cut_offset_s"])
    on = float(r["t_start"]) - off if (r.get("t_start") or "").strip() else None
    cpa = float(r["t_cpa"]) - off if (r.get("t_cpa") or "").strip() else None
    if CLASS_IDX.get(r.get("class", "")) in WARN:
        a, b = (on - 1.0, on + 4.0) if on is not None else (None, None)
    else:
        a = (on if on is not None else (cpa - 6.0 if cpa is not None else None))
        a, b = ((a - 1.0), (cpa + 1.0)) if (a is not None and cpa is not None) else (None, None)
    if a is None:
        return set()
    return {k for k in scored if a - 1e-9 <= (k + 0.5) * RES <= b + 1e-9} - ex_f

RES = XC.RES
NAMES = {0: "サイレン", 1: "クラクション", 2: "バック音", 3: "ベル", 4: "車", 5: "踏切", 6: "キック", 7: "バイク", 8: "自転車"}
WARN = {0, 1, 2, 3}                 # 警告音（通知の成否で見る）


def ang_err(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def read_csv(p) -> list:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_gt(p: Path) -> dict:
    """{class: {frame: (az, el, dist)}}"""
    g = defaultdict(dict)
    if p.is_file():
        for row in csv.reader(open(p, encoding="utf-8")):
            if len(row) >= 6 and row[0].strip().lstrip("-").isdigit():
                g[int(row[1])][int(row[0])] = (float(row[3]), float(row[4]), float(row[5]))
    return g


def load_pred(p: Path) -> dict:
    """{clip: {frame: [(class, az, el, dist)]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for row in csv.reader(open(p, encoding="utf-8")):
        if len(row) >= 7 and row[1].strip().lstrip("-").isdigit():
            out[row[0]][int(row[1])].append((int(row[2]), float(row[4]), float(row[5]), float(row[6]) if row[6] not in ("", "nan") else float("nan")))
    return out


def score_clip(clip: str, gt: dict, pred: dict, ann_rows: list, excl_iv: list) -> dict:
    ex_f = XC.frames_in(excl_iv)
    scored = XC.scored_frames()
    rec = {"clip_id": clip, "区分": ann_rows[0].get("区分", ""), "原本": Path(ann_rows[0].get("orig_file", "")).name, "exclusion": excl_iv, "classes": {}}
    for c, fr in sorted(gt.items()):
        det = {k for k, rows in pred.items() if any(r[0] == c for r in rows)}
        m = XC.frame_metric(set(fr), det, ex_f, scored)
        valid = (scored - ex_f) & set(fr)
        errs, derr = [], []
        for k in sorted(valid):
            same = [r for r in pred.get(k, []) if r[0] == c]
            if not same:
                continue
            az, el, dd = fr[k]
            best = min(same, key=lambda r: ang_err(r[1], r[2], az, el))
            errs.append(ang_err(best[1], best[2], az, el))
            if math.isfinite(best[3]):
                derr.append(abs(best[3] - dd))
        cr = dict(m, ang_med=(median(errs) if errs else None), dist_err_med=(median(derr) if derr else None))
        if c in WARN:
            on = None
            for r in ann_rows:
                if (r.get("t_start") or "").strip():
                    on = float(r["t_start"]) - float(r["cut_offset_s"])
                    break
            if on is not None:
                fires = [(k + 0.5) * RES for k, rows in pred.items() if any(r[0] == c for r in rows)]
                n = XC.notification(on, fires, excl_iv)
                e1 = []
                for k in sorted(fr):
                    t = (k + 0.5) * RES
                    if on - 1e-9 <= t <= on + 1.0 and k not in ex_f:
                        same = [r for r in pred.get(k, []) if r[0] == c]
                        if same:
                            az, el, _ = fr[k]
                            e1.append(min(ang_err(r[1], r[2], az, el) for r in same))
                cr["notification"] = {"status": n["status"], "onset_s": round(on, 3), "first_fire_s": n.get("first_fire"),
                                      "latency_s": (round(n["first_fire"] - on, 3) if n.get("first_fire") is not None else None),
                                      "ang_med_first1s": (median(e1) if e1 else None)}
        rec["classes"][str(c)] = cr
    other = defaultdict(int)
    for k, rows in pred.items():
        if k in ex_f or k not in scored:
            continue
        for r in rows:
            if r[0] not in gt:
                other[NAMES.get(r[0], str(r[0]))] += 1
    rec["other_class_frames"] = dict(other)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer", type=Path, default=E2E / "out" / "_demo_field_unused_149" / "infer")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    ann = defaultdict(list)
    for r in read_csv(a.infer / "ann_all.csv"):
        ann[r["clip_id"]].append(r)
    excl = defaultdict(list)
    if (a.infer / "exclusions.csv").is_file():
        for r in read_csv(a.infer / "exclusions.csv"):
            excl[r["clip_id"]].append((float(r["入り_クリップの秒"]), float(r["出_クリップの秒"])))
    pred = load_pred(a.pred)
    clips = sorted(ann)
    recs = []
    CM = CC.new_matrix()                 # 取り違えの表（参考・本人 12:35 (a)）
    tk_events = []                       # 本ごとの種類（参考・(b)）
    for c in clips:
        g = load_gt(a.infer / "gt" / f"{c}.csv")
        recs.append(score_clip(c, g, pred.get(c, {}), ann[c], excl.get(c, [])))
        ex_f = XC.frames_in(excl.get(c, []))
        scored = XC.scored_frames() - ex_f
        g_sets, p_sets = defaultdict(set), defaultdict(set)
        for ci, fr in g.items():
            for k in fr:
                g_sets[k].add(ci)
        for k, rows in pred.get(c, {}).items():
            for r in rows:
                p_sets[k].add(r[0])
        CC.add_clip(CM, g_sets, p_sets, sorted(scored))
        for r in ann[c]:
            gi = CLASS_IDX.get(r.get("class", ""))
            if gi is None:
                continue
            win = take_window(r, XC.scored_frames(), ex_f)
            other = {x for k in win for x in g_sets.get(k, ()) if x != gi}        # その窓の正解にある別の種類は取り違えに数えない（本人 14:13）
            tk_events.append({"clip_id": c, "gt": gi, "pred": CC.take_class(p_sets, win, ignore=other), "n_window_frames": len(win)})
    # 表の頭（決め直し 6 の 5）と集計（区分ごと・クラスごと）
    tr = [{"frame": next(iter(r["classes"].values()), {}) if r["classes"] else {"recall": None},
           "notifications": [cr["notification"] for cr in r["classes"].values() if "notification" in cr]} for r in recs]
    head = XC.summarize(tr)
    agg = defaultdict(lambda: {"n_gt": 0, "n_hit": 0, "ang": [], "dist": [], "n_clips": 0, "n_missing": 0})
    for r in recs:
        for c, cr in r["classes"].items():
            g = agg[c]
            g["n_clips"] += 1
            g["n_missing"] += cr["recall"] is None
            g["n_gt"] += cr["n_gt"]; g["n_hit"] += cr["n_hit"]
            if cr["ang_med"] is not None:
                g["ang"].append(cr["ang_med"])
            if cr["dist_err_med"] is not None:
                g["dist"].append(cr["dist_err_med"])
    summary = {NAMES.get(int(c), c): {"clips": g["n_clips"], "missing": g["n_missing"], "recall": (g["n_hit"] / g["n_gt"]) if g["n_gt"] else None,
                                      "frames": f"{g['n_hit']}/{g['n_gt']}", "ang_med_of_clips": (median(g["ang"]) if g["ang"] else None),
                                      "dist_err_med_of_clips": (median(g["dist"]) if g["dist"] else None)} for c, g in sorted(agg.items())}
    TA = CC.take_accuracy(tk_events)
    out = {"label": a.label, "pred": str(a.pred), "infer": str(a.infer), "head": head, "summary": summary, "clips": recs,
           "confusion": CC.to_json(CM), "take_accuracy": dict(TA, events=tk_events),
           "rule": "学内の開発用の本（評価には使わない）。家の 13 本とは別の表・足し合わせない。除外区間はどのクラスも採点しない（決め直し 6）"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"== 学内の開発用の本 {a.label}: 本 {head['n_takes']}・欠測 {head['n_missing_takes']}・警告音の事象 {head['n_events']}（通常 {head['n_scored_events']}・部分観察 {head['n_partial']}・届いた {head['n_success']}）")
    for k, s in summary.items():
        print(f"  {k}: 本 {s['clips']}（欠測 {s['missing']}）・見つけたフレーム {s['frames']}" + (f"（{100 * s['recall']:.0f}%）" if s["recall"] is not None else "")
              + (f"・向きの誤差の中央値 {s['ang_med_of_clips']:.0f}°" if s["ang_med_of_clips"] is not None else "")
              + (f"・距離の誤差の中央値 {s['dist_err_med_of_clips']:.2f} m" if s["dist_err_med_of_clips"] is not None else ""))
    for r in recs:
        mark = "＊" if r["exclusion"] else "　"
        parts = []
        for c, cr in r["classes"].items():
            p = f"{NAMES.get(int(c), c)} {cr['n_hit']}/{cr['n_gt']}（有効 {cr['valid_s']} s）"
            if "notification" in cr:
                p += f"・届いた={cr['notification']['status']}"
            parts.append(p)
        print(f"  {mark}{r['clip_id']}（{r['原本']}・{r['区分']}）: " + "・".join(parts) + (f"・ほかのクラス {r['other_class_frames']}" if r["other_class_frames"] else ""))
    print("  ＊ = 除外区間あり（その時間はどのクラスも採点しない）")
    for s in CC.md_lines(CM) + [""] + CC.md_lines_take(TA):
        print("  " + s if s else "")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
