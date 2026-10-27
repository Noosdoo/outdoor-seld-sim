"""家で録った 10/7 の自転車（デモ 3 本・走行音の測定 10 本）の手応えの採点（2026-10-25 Opus・本人 20:25）。評価には使わない（10/7 の決まり ⑥）。

正解（モデルは使わない）:
  - 自転車の走行音（クラス 8）: 動画の読み（前輪が目印を通った秒）を直線でつなぎ、読みの範囲の中だけ（外へは伸ばさない）。
    音の位置は前輪と後輪の真ん中（前輪から進む向きの後ろへ 0.5 m・自転車の軸の間 約 1 m の半分の仮定）。--front なら前輪のまま（感度）。
    横は線の値（1.0 m・本人の申告は 1.0〜1.2 m）・音源の高さ 0.25 m（step19k の SRC_Z）・マイク 1.5 m・距離は水平。
    読み → マイクの座標は takes.csv の sx・sy（demo_bike_check_prepare.py が三脚の向き〔デモ〕か音の向き〔測定〕で決めた）。
    方位はマイク基準で左が +（x 前・y 左・atan2(y, x)）。フレーム k の時刻は (k + 0.5) × 0.1 s（step19k と同じ）。
  - ベル（クラス 3・デモ 3 本だけ）: 鳴り始め = 注釈の t_start（録音のベル開始・5.00 s）。見つけたか = 警告音の成功の窓 [鳴り始め − 1, + 4] s に
    クラス 3 の行があるか（step20 と同じ窓）。向きは鳴り始めから 1 s の行を、ベルを鳴らした目印（注釈の ベル目印_m・線の上・高さ 1.0 m）と比べる。
    （10/7 の step19k の正解 perception_h0703/gt はベルの札で「動画で位置の分かるフレーム」に付いていて、鳴っている間ではないので使わない。）
予測: 7 列（clip, frame, class, track, az, el, dist）。正解のフレームに同じクラスの行があれば「見つけた」。向きは同じクラスの行のうち誤差の一番小さい行。
自転車は、クラスを問わず向きが 30° 以内の行があるかも数える（音には気づいているが別のクラスと言っているかを見るため）。

使い方: python scripts/demo_bike_check_score.py --pred <val_all_causal.csv> [--dir out/_demo_bike_check] [--front] [--label 名前]
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

E2E = Path(__file__).resolve().parents[1]
MEAS_READS = E2E / "raw_bicycle_measure" / "reads" / "reads_A.csv"
DEMO_READS = E2E / "out" / "field_demo" / "reads" / "reads_A.csv"
DEMO_ANN = E2E / "out" / "field_demo" / "ann_all.csv"
BELL_Z = 1.0
RES = 0.1
NF = 100
HALF_WB = 0.5
SRC_Z = 0.25
MIC_Z = 1.5
NAMES = {0: "サイレン", 1: "クラクション", 2: "バック音", 3: "ベル", 4: "車", 5: "踏切", 6: "キック", 7: "バイク", 8: "自転車"}


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ang_err(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def az_err(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def bike_gt(take_row, reads, front=False):
    rr = sorted([r for r in reads if r["take"] == take_row["take"]], key=lambda r: float(r["t"]))
    off = float(take_row["cut_offset_s"])
    ts = [float(r["t"]) - off for r in rr]
    xs = [float(r["x"]) for r in rr]
    y_read = float(take_row["line_m"])
    sx, sy = int(take_row["sx"]), int(take_row["sy"])
    d = 1 if xs[-1] > xs[0] else -1
    gt = {}
    for k in range(NF):
        tc = (k + 0.5) * RES
        if not (ts[0] <= tc <= ts[-1]):
            continue
        j = max(i for i in range(len(ts) - 1) if ts[i] <= tc) if tc < ts[-1] else len(ts) - 2
        xf = xs[j] + (xs[j + 1] - xs[j]) * (tc - ts[j]) / (ts[j + 1] - ts[j])
        xm = xf if front else xf - HALF_WB * d
        x, y = sx * xm, sy * y_read
        dist = math.hypot(x, y)
        gt[k] = (math.degrees(math.atan2(y, x)), math.degrees(math.atan2(SRC_Z - MIC_Z, dist)), dist, abs(xm))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dir", default=str(E2E / "out" / "_demo_bike_check"))
    ap.add_argument("--front", action="store_true", help="音の位置を前輪のまま（感度）")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    takes = read_csv(Path(a.dir) / "takes.csv")
    reads_m, reads_d = read_csv(MEAS_READS), read_csv(DEMO_READS)
    pred = defaultdict(list)
    with open(a.pred, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[1].strip().lstrip("-").isdigit():
                continue
            pred[(row[0], int(row[1]))].append((int(row[2]), int(row[3]), float(row[4]), float(row[5]), float(row[6])))
    clips_in_pred = {c for c, _ in pred}
    out = {"label": a.label, "pred": str(a.pred), "front": a.front, "bike": [], "bell": []}
    tot = Counter()
    errs_ang, errs_az, errs_dist, near_zone = [], [], [], []
    conf = Counter()
    any30 = Counter()
    lines = []
    for t in takes:
        cid = t["clip_id"]
        gt = bike_gt(t, reads_m if t["source"] == "measure" else reads_d, a.front)
        det = 0
        e_a, e_d = [], []
        for k, (gaz, gel, gd, axm) in gt.items():
            rows = pred.get((cid, k), [])
            b = [r for r in rows if r[0] == 8]
            for r in rows:
                if r[0] not in (8, 3):
                    conf[r[0]] += 1
            near = [r for r in rows if ang_err(r[2], r[3], gaz, gel) <= 30.0]
            if near:
                any30["frames"] += 1
                for c in {r[0] for r in near}:
                    any30[c] += 1
            if not b:
                continue
            det += 1
            r = min(b, key=lambda r: ang_err(r[2], r[3], gaz, gel))
            ea = ang_err(r[2], r[3], gaz, gel)
            e_a.append(ea)
            errs_ang.append(ea)
            errs_az.append(az_err(r[2], gaz))
            if math.isfinite(r[4]):
                e_d.append(abs(r[4] - gd))
                errs_dist.append(abs(r[4] - gd))
                if axm <= 1.0:
                    near_zone.append(r[4] <= 1.5)
        tot["gt"] += len(gt)
        tot["det"] += det
        tot["pass"] += 1
        tot["pass_det"] += det > 0
        other = Counter(r[0] for (c, _), rs in pred.items() if c == cid for r in rs if r[0] not in (8, 3))
        rec = {"clip": cid, "take": t["take"], "source": t["source"], "n_gt": len(gt), "n_det": det,
               "recall": det / len(gt) if gt else None, "ang_med": median(e_a) if e_a else None,
               "dist_err_med": median(e_d) if e_d else None, "other_classes_in_clip": {NAMES[c]: n for c, n in sorted(other.items())},
               "in_pred": cid in clips_in_pred}
        out["bike"].append(rec)
        lines.append(f"  {cid}（{t['source']} {t['take']}）: 見つけた {det}/{len(gt)}"
                     + (f"・向きの誤差の中央値 {median(e_a):.0f}°" if e_a else "")
                     + (f"・距離の誤差の中央値 {median(e_d):.2f} m" if e_d else "")
                     + (f"・ほかのクラス {dict((NAMES[c], n) for c, n in sorted(other.items()))}" if other else ""))
    # ベル（デモ 3 本）: 警告音の成功の窓 [鳴り始め − 1, + 4] s・向きは鳴り始めから 1 s をベルを鳴らした目印と比べる
    bell_lines = []
    ann = {r["clip_id"]: r for r in read_csv(DEMO_ANN)}
    bells_ok = 0
    for t in [t for t in takes if t["source"] == "demo"]:
        cid = t["clip_id"]
        r0 = ann[cid]
        on = float(r0["t_start"])
        sx, sy = int(t["sx"]), int(t["sy"])
        bx, by = sx * float(r0["ベル目印_m"]), sy * float(t["line_m"])
        bd = math.hypot(bx, by)
        baz, bel = math.degrees(math.atan2(by, bx)), math.degrees(math.atan2(BELL_Z - MIC_Z, bd))
        rows3 = sorted((k, r) for (c, k), rs in pred.items() if c == cid for r in rs if r[0] == 3)
        inwin = [(k, r) for k, r in rows3 if on - 1.0 <= (k + 0.5) * RES <= on + 4.0]
        outwin = [(k, r) for k, r in rows3 if not (on - 1.0 <= (k + 0.5) * RES <= on + 4.0)]
        first = inwin[0][0] if inwin else None
        e1 = [ang_err(r[2], r[3], baz, bel) for k, r in inwin if on <= (k + 0.5) * RES <= on + 1.0]   # 鳴り始めから 1 秒（下限も・Astra HB25-05）
        bells_ok += bool(inwin)
        rec = {"clip": cid, "onset_s": on, "bell_marker_mic_xy": [bx, by], "bell_az": baz, "detected": bool(inwin),
               "latency_s": ((first + 0.5) * RES - on) if first is not None else None,
               "ang_med_first1s": median(e1) if e1 else None, "bell_rows_outside_window": len(outwin)}
        out["bell"].append(rec)
        bell_lines.append(f"  {cid}: ベル {'○' if inwin else '×'}"
                          + (f"・鳴り始めから {rec['latency_s']:+.2f} s" if first is not None else "")
                          + (f"・向きの誤差の中央値 {median(e1):.0f}°（正解の方位 {baz:.0f}°）" if e1 else "")
                          + (f"・窓の外のベル {len(outwin)} 行" if outwin else ""))
    out["bell_detected"] = f"{bells_ok}/{len(out['bell'])}"
    s = {
        "bike_recall": tot["det"] / tot["gt"] if tot["gt"] else None, "bike_gt_frames": tot["gt"], "bike_det_frames": tot["det"],
        "bike_pass_detected": f"{tot['pass_det']}/{tot['pass']}",
        "ang_med": median(errs_ang) if errs_ang else None, "az_med": median(errs_az) if errs_az else None,
        "within20": (sum(e <= 20 for e in errs_ang) / len(errs_ang)) if errs_ang else None,
        "dist_err_med": median(errs_dist) if errs_dist else None,
        "near_cpa_in_shikin": (sum(near_zone) / len(near_zone)) if near_zone else None, "near_cpa_n": len(near_zone),
        "other_classes_in_bike_frames": {NAMES[c]: n for c, n in sorted(conf.items())},
        "any_class_within30_frames": any30["frames"],
        "any_class_within30_by_class": {NAMES[c]: n for c, n in sorted((c, n) for c, n in any30.items() if c != "frames")},
    }
    out["summary"] = s
    print(f"== {a.label or a.pred}（音の位置 = {'前輪' if a.front else '前輪と後輪の真ん中'}）")
    print(f"自転車の走行音: 見つけたフレーム {tot['det']}/{tot['gt']}（{100 * (s['bike_recall'] or 0):.0f}%）・見つけた通過 {s['bike_pass_detected']}")
    if errs_ang:
        print(f"  向きの誤差の中央値 {s['ang_med']:.0f}°（方位だけ {s['az_med']:.0f}°）・20° 以内 {100 * s['within20']:.0f}%・距離の誤差の中央値 {s['dist_err_med']:.2f} m"
              + (f"・通過の前後 1 m で「至近（1.5 m 以下）」と出した割合 {100 * s['near_cpa_in_shikin']:.0f}%（{s['near_cpa_n']} フレーム）" if near_zone else ""))
    print(f"  自転車がいるフレームで出したほかのクラス: {s['other_classes_in_bike_frames']}")
    print(f"  クラスを問わず自転車の向き（30° 以内）に何か出したフレーム: {s['any_class_within30_frames']}/{tot['gt']}・内訳 {s['any_class_within30_by_class']}")
    print("\n".join(lines))
    print(f"ベル（デモ 3 本・見つけた {out['bell_detected']}）:")
    print("\n".join(bell_lines))
    tag = (a.label or Path(a.pred).stem) + ("_front" if a.front else "")
    (Path(a.dir) / f"score_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
