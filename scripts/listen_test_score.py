# -*- coding: utf-8 -*-
"""人と比べる測定の採点（2026-10-27 Opus）。人の「気づいた」時刻と、モデルが最初に鳴った時刻を、同じ本・同じ時間軸で並べる。

入力: --listen（_key.csv・results_<人>.csv）・--infer（gt/<clip>.csv〔frame,class,track,az,el,dist〕・exclusions.csv）・--pred（モデルの val_all_causal.csv・7 列）
時刻はすべて「切り出しの秒」（モデルの 10 秒の切り出しの頭が 0・最接近は 8.0 s）。人の押しは聞いた wav の秒 + 切り詰めで直す。
  - 反応の速さ: 「ピッ」の本の (押し − 鳴った時刻) の中央値（0.1〜1.5 s の外は外す）。人の時刻は「そのまま」と「反応の速さを引いた値」の両方を出す。
  - 本人は全部の本で「自転車の走る音が聞こえたら押す」で聞いた（10/27 14:15）。比べるのは自転車の走る音に気づいた時刻:
    人 = 最初の押し・モデル = frame 40（暖機 4.0 s）以降で自転車（8）が最初に出たフレーム（判断できる時刻 = (frame + 1) × 0.1 s・ベルの本も自転車の正解がある〔本人 13:09〕）。
    自転車の最接近（自転車の正解の距離が最小の枠）からの秒と、その時刻の距離（正解の枠が無ければ「10 m より遠い」か「なし」）。主の比べは自転車の本、ベルの本は参考。
  - ベルの本の参考: 押しがベルの鳴り始めより前か後か（後ならベルが手がかりになった可能性）・鳴り始めからの遅れ（人・モデルのベル〔3〕の [鳴り始め − 1, + 4] s の最初の発火）。
  - 通った側: 人の答え（L/R/?）と、自転車の正解の最接近 ±1 s の方位の符号（左が正）と、モデルの目的の種類の方位の平均の符号。
  - 除外区間（決め直し 6）: 除外中のモデルの発火は数えない・人の最初の押しが除外中なら印を付けて比べない。
  - 人が 4.0 s より前に押した本は、モデルがまだ動けない時間なので印を付けて別に数える。
使い方: python scripts/listen_test_score.py --reader A --pred out/_demo_field_unused_149/pred_e139_before.csv --label e139 --out <json>
試験: scripts/test_listen_test.py
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEMO = E2E / "out" / "_demo_field_unused_149"
RES = 0.1
WARM_FRAME = 40
CLASS_OF = {"bicycle": 8, "bike_bell": 3}


def rcsv(p: Path) -> list:
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load_gt(p: Path) -> dict:
    g = defaultdict(dict)
    if p.is_file():
        for row in csv.reader(open(p, encoding="utf-8")):
            if len(row) >= 6 and row[0].strip().lstrip("-").isdigit():
                g[int(row[1])][int(row[0])] = (float(row[3]), float(row[4]), float(row[5]))
    return g


def load_pred(p: Path) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    for row in csv.reader(open(p, encoding="utf-8")):
        if len(row) >= 7 and row[1].strip().lstrip("-").isdigit():
            out[row[0]][int(row[1])].append((int(row[2]), float(row[4])))
    return out


def in_excl(t: float, iv: list) -> bool:
    return any(a <= t < b for a, b in iv)


def side_of(az: float | None) -> str:
    if az is None:
        return ""
    return "L" if az > 0 else "R"


def circ_mean(azs: list) -> float | None:
    if not azs:
        return None
    s = sum(math.sin(math.radians(a)) for a in azs); c = sum(math.cos(math.radians(a)) for a in azs)
    return math.degrees(math.atan2(s, c))


def score(listen: Path, reader: str, infer: Path, pred_p: Path) -> dict:
    key = {(r["block"], r["file"]): r for r in rcsv(listen / "_key.csv")}
    res = rcsv(listen / f"results_{reader}.csv")
    excl = defaultdict(list)
    if (infer / "exclusions.csv").is_file():
        for r in rcsv(infer / "exclusions.csv"):
            excl[r["clip_id"]].append((float(r["入り_クリップの秒"]), float(r["出_クリップの秒"])))
    rts = []
    for r in res:
        if r["block"] == "rt" and r["first_press_s"]:
            d = float(r["first_press_s"]) - float(key[("rt", r["file"])]["bell_onset_file_s"])
            if 0.1 <= d <= 1.5:
                rts.append(d)
    rt = median(rts) if rts else 0.0
    pred = load_pred(pred_p)
    trials = []
    for r in res:
        if r["block"] != "main":
            continue
        k = key[("main", r["file"])]
        cid = k["clip_id"]
        is_bell = k["class"] == "bike_bell"
        # 本人 10/27 14:15「すべて自転車の走行音が聞こえたら押す形にした」→ 比べるのは自転車の走る音に気づいた時刻。ベルの本でも自転車は走っていて正解がある
        # （本人 13:09）ので、どの本もモデルの自転車（8）の最初の発火と比べる。ベルの本で押しが鳴り始めより後なら、ベルが気づく手がかりになった可能性がある（印を付ける）。
        targets = {8}
        trim = float(k["trim_s"])
        G = load_gt(infer / "gt" / f"{cid}.csv")
        gt = G.get(8, {})                                       # 距離と最接近と通った側は自転車の正解から（ベルの本も自転車の正解を付けた）
        gk = sorted(gt)
        fin = {f: v[2] for f, v in gt.items()}
        cpa = ((min(fin, key=lambda f: (fin[f], f)) + 0.5) * RES) if fin else float(k["cpa_file_s"]) + trim
        iv = excl.get(cid, [])
        ex_frames = {f for a, b in iv for f in range(int(math.floor(a / RES)), int(math.ceil(b / RES)))}
        h_raw = (float(r["first_press_s"]) + trim) if r["first_press_s"] else None
        h = (h_raw - rt) if h_raw is not None else None
        fires = sorted(f for f, rows in pred.get(cid, {}).items() if f >= WARM_FRAME and f not in ex_frames and any(c in targets for c, _ in rows))
        t_m = (fires[0] + 1) * RES if fires else None
        rec = {"file": r["file"], "clip_id": cid, "class": k["class"], "trim_s": trim, "human_raw_s": h_raw, "human_s": h,
               "human_before_warm": (h_raw is not None and h_raw < WARM_FRAME * RES), "human_in_exclusion": (h_raw is not None and in_excl(h_raw, iv)),
               "human_side": r["side"], "targets": sorted(targets)}

        def dist_at(t):
            """t までに聞いた最後のフレーム（[f·0.1, (f+1)·0.1) で t を含む・境目ちょうどは前のフレーム）の自転車の正解の距離。"""
            if t is None:
                return None
            f = int(math.floor(t / RES - 1e-9))
            if f in gt:
                return gt[f][2]
            return ">10" if (gk and f < gk[0]) else None
        rec.update(ref="最接近", ref_s=cpa, human_rel_s=(h - cpa if h is not None else None), human_raw_rel_s=(h_raw - cpa if h_raw is not None else None),
                   model_s=t_m, model_rel_s=(t_m - cpa if t_m is not None else None), human_dist_m=dist_at(h), model_dist_m=dist_at(t_m))
        if is_bell:                                             # 参考: ベルの鳴り始めからの遅れ（人が鳴り始めより前に押した本は自転車で気づいた）
            on = float(k["bell_onset_file_s"]) + trim
            fb = [f for f in sorted(pred.get(cid, {})) if f >= WARM_FRAME and f not in ex_frames and on - 1.0 <= (f + 1) * RES <= on + 4.0
                  and any(c == 3 for c, _ in pred[cid][f])]
            rec.update(bell_onset_s=on, bell_human_rel_s=(h - on if h is not None else None),
                       bell_model_rel_s=(((fb[0] + 1) * RES - on) if fb else None), human_before_bell=(h_raw is not None and h_raw < on),
                       human_after_bell=(h_raw is not None and h_raw >= on))
        # 通った側: 最接近 ±1 s（自転車の正解の方位の符号・モデルは目的の種類の方位の平均の符号）
        ref_f = min(gk, key=lambda f: abs((f + 0.5) * RES - cpa)) if gk else None
        if ref_f is None and is_bell:                           # 自転車の正解が無い古い置き場ではベルの正解で
            gb = G.get(3, {})
            ref_f = min(gb, key=lambda f: abs((f + 0.5) * RES - cpa)) if gb else None
            rec["gt_side"] = side_of(gb[ref_f][0]) if ref_f is not None else ""
        else:
            rec["gt_side"] = side_of(gt[ref_f][0]) if ref_f is not None else ""
        maz = [az for f in range(int(math.floor((cpa - 1.0) / RES)), int(math.floor((cpa + 1.0) / RES)) + 1) if f not in ex_frames
               for c, az in pred.get(cid, {}).get(f, []) if c in targets]
        rec["model_side"] = side_of(circ_mean(maz))
        if rec["human_rel_s"] is not None and rec["model_rel_s"] is not None and not rec["human_in_exclusion"]:
            rec["human_minus_model_s"] = rec["human_rel_s"] - rec["model_rel_s"]   # 人の最初の押しが除外区間（車・声）の中なら比べない
        trials.append(rec)
    summ = {}
    for cname in ("bicycle", "bike_bell"):
        t = [x for x in trials if x["class"] == cname]
        both = [x["human_minus_model_s"] for x in t if "human_minus_model_s" in x]
        summ[cname] = {"n": len(t), "human_pressed": sum(x["human_s"] is not None for x in t), "model_detected": sum(x["model_s"] is not None for x in t),
                       "both": len(both), "human_earlier": sum(d < 0 for d in both), "model_earlier": sum(d > 0 for d in both),
                       "median_human_minus_model_s": (median(both) if both else None),
                       "human_before_warm": sum(x["human_before_warm"] for x in t),
                       "human_in_exclusion": sum(x["human_in_exclusion"] for x in t),
                       "human_before_bell": sum(bool(x.get("human_before_bell")) for x in t),
                       "human_after_bell": sum(bool(x.get("human_after_bell")) for x in t),
                       "human_side_correct": sum(x["human_side"] == x["gt_side"] for x in t if x["gt_side"]),
                       "model_side_correct": sum(x["model_side"] == x["gt_side"] for x in t if x["gt_side"]),
                       "n_side_scored": sum(bool(x["gt_side"]) for x in t)}
    return {"reader": reader, "rt_median_s": rt, "n_rt": len(rts), "trials": trials, "summary": summ, "pred": str(pred_p)}


def fmt(v, nd=2):
    return "—" if v is None else (v if isinstance(v, str) else f"{v:+.{nd}f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=Path, default=DEMO / "listen")
    ap.add_argument("--infer", type=Path, default=DEMO / "infer")
    ap.add_argument("--reader", default="A")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    r = score(a.listen, a.reader, a.infer, a.pred)
    r["label"] = a.label
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"== 人（{a.reader}）とモデル（{a.label}）: 反応の速さの中央値 {r['rt_median_s']:.2f} s（{r['n_rt']} 本）")
    for c, s in r["summary"].items():
        print(f"  {'自転車の本' if c == 'bicycle' else 'ベルの本（自転車の走る音・参考）'}: 本 {s['n']}・人が押した {s['human_pressed']}・モデルが鳴った {s['model_detected']}・比べた {s['both']}"
              f"（人が早い {s['human_earlier']}・モデルが早い {s['model_earlier']}・差の中央値 人 − モデル {fmt(s['median_human_minus_model_s'])} s）"
              f"・通った側の正解 人 {s['human_side_correct']}/{s['n_side_scored']}・モデル {s['model_side_correct']}/{s['n_side_scored']}"
              + (f"・人が 4 秒より前に押した {s['human_before_warm']}" if s["human_before_warm"] else "")
              + (f"・人の押しが除外区間の中 {s['human_in_exclusion']}（比べない）" if s["human_in_exclusion"] else "")
              + (f"・押しがベルの鳴り始めより前 {s['human_before_bell']}・後 {s['human_after_bell']}（後はベルが手がかりになった可能性）" if c == "bike_bell" else ""))
    dm = lambda v: "—" if v is None else (v if isinstance(v, str) else f"{v:.1f}")   # noqa: E731
    for t in r["trials"]:
        bell = (f"・参考 ベルの鳴り始めから 人 {fmt(t.get('bell_human_rel_s'))}／モデル {fmt(t.get('bell_model_rel_s'))} s" if t["class"] == "bike_bell" else "")
        print(f"  {t['file']}（{t['clip_id']}・{t['class']}）: {t['ref']}から 人 {fmt(t['human_rel_s'])} s（そのまま {fmt(t['human_raw_rel_s'])}）・モデル {fmt(t['model_rel_s'])} s"
              f"・距離 人 {dm(t.get('human_dist_m'))} m／モデル {dm(t.get('model_dist_m'))} m{bell}"
              f"・側 人 {t['human_side'] or '—'}／モデル {t['model_side'] or '—'}／正解 {t['gt_side'] or '—'}"
              + ("・＊人が 4 秒より前" if t["human_before_warm"] else "") + ("・＊除外区間の中（比べない）" if t["human_in_exclusion"] else ""))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
