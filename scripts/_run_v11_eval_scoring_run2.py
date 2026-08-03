# -*- coding: utf-8 -*-
"""v11評価拡張（16セット3,246本）の採点。仕様§3/§4の事前登録に従う。
出力: out/step12_notify_v11eval/v11eval_summary.md"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.calibration import frame_spl_a  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "step12", ROOT / "scripts" / "step12_notify_v9.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

DS = ROOT / "out" / "dataset_outdoor_siren_v11_eval"
PRED = ROOT / "out" / "predictions_v11eval_run2"
OUT = ROOT / "out" / "step12_notify_v11eval_run2"
CLS = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
       "car_drive": 4, "crossing": 5}
CAR = 4


def load_pred(path):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def plan(which):
    return list(csv.DictReader(open(DS / "plan" / f"assignment_{which}.csv")))


_scene_cache = {}


def scene(clip):
    if clip not in _scene_cache:
        _scene_cache[clip] = json.loads((DS / "work" / clip / "scene.json").read_text())
    return _scene_cache[clip]


def fires(clip, pred):
    mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
    return m12.fire_events(pred.get(clip, {}), frame_spl_a(mix[0], 24000))


def first(fs, ci):
    ks = [k for k, c, _ in fs if c == ci]
    return m12.emit_time(ks[0]) if ks else None


def pct(k, n):
    return f"{k}/{n} ({100*k/max(n,1):.1f}%)"


def med(a):
    return f"{np.median(a):.2f}s" if len(a) else "-"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = ["# v11評価拡張 採点（v11 run1、2026-07-28。仕様§4の事前登録に従う）", ""]
    false_total, exposure_h = 0, 3246 * 10.0 / 3600.0

    def acc_false(clip, fs):
        nonlocal false_total
        present = {CLS[s["class"]] for s in scene(clip)["sources"]}
        false_total += sum(1 for _, c, _ in fs if c not in present)

    # ---- 幻覚600 ----
    p = load_pred(PRED / "evhalluc_all.csv")
    n_clip = car_fr = car_fire = s_ok = 0
    for r in plan("halluc600"):
        c = r["clip_id"]
        fs = fires(c, p)
        acc_false(c, fs)
        cf = sum(len([e for e in evs if e[0] == CAR]) for evs in p.get(c, {}).values())
        car_fr += cf
        n_clip += int(cf > 0)
        car_fire += sum(1 for _, cc, _ in fs if cc == CAR)
        s_ok += int(any(cc == 0 for _, cc, _ in fs))
    up = m12.poisson_upper95(car_fire, 600 * 10 / 3600.0)
    R += ["## 幻覚フロア600（車なし×サイレン。旧n=50: 幻覚1/50・誤通知0）",
          f"- 幻覚ありクリップ(車predフレーム>0): **{pct(n_clip, 600)}**、車フレーム計{car_fr}",
          f"- 車の誤通知発火: **{car_fire}件/600本(1.67h)** → Poisson95%上限 **{up:.1f}回/時**",
          f"- サイレン通知: {pct(s_ok, 600)}", ""]

    # ---- safe600 距離カーブ ----
    p = load_pred(PRED / "evsafe_all.csv")
    bins = [(3.2, 5), (5, 7), (7, 9), (9, 12), (12, 15.1)]
    bc = {b: [0, 0] for b in bins}
    over = 0
    for r in plan("safe600"):
        c = r["clip_id"]
        fs = fires(c, p)
        acc_false(c, fs)
        n_car_f = sum(1 for _, cc, _ in fs if cc == CAR)
        cpa = scene(c).get("cpa_rel_dist_m", 99)
        for b in bins:
            if b[0] <= cpa < b[1]:
                bc[b][1] += 1
                bc[b][0] += int(n_car_f > 0)
        over += int(n_car_f > 0)
    R += ["## safe過剰通知600（車1台・警告なし・CPA3.2-15m。v11 valでは89.9%）",
          f"- 全体: 過剰通知 **{pct(over, 600)}**", "- 距離分解（通知層再設計の入力）:"]
    for b in bins:
        k, n = bc[b]
        R.append(f"    CPA {b[0]:.1f}-{b[1]:.0f}m: {pct(k, n)}")
    R.append("")

    # ---- 増量S系 ----
    p = load_pred(PRED / "evscn2_all.csv")
    for which, name in [("s1_200", "S1踏切200"), ("s2_100", "S2背後ベル100"),
                        ("s3_100", "S3バック車100"), ("s5_200", "S5悪条件200")]:
        rows = plan(which)
        wn = wo = cn = co = 0
        leads, tiers = [], defaultdict(lambda: [0, 0])
        for r in rows:
            c = r["clip_id"]
            fs = fires(c, p)
            acc_false(c, fs)
            sc = scene(c)
            if r["w1_class"]:
                wn += 1
                wo += int(any(cc == CLS[r["w1_class"]] for _, cc, _ in fs))
            car = next((s for s in sc["sources"] if s["class"] == "car_drive"), None)
            if car is not None:
                cn += 1
                t = first(fs, CAR)
                tiers[r["danger_tier"]][1] += 1
                if t is not None:
                    co += 1
                    tiers[r["danger_tier"]][0] += 1
                    leads.append(sc.get("cpa_rel_time_s", 99) - t)
            elif r["w1_class"] and r["w1_class"] != "crossing":
                t = first(fs, CLS[r["w1_class"]])
                src = sc["sources"][0]
                if t is not None and src.get("t_cpa_rel_s"):
                    leads.append(src["t_cpa_rel_s"] - t)
        line = f"- **{name}**: 警告音通知 {pct(wo, wn)}" if wn else f"- **{name}**:"
        if cn:
            line += f" / 車通知 {pct(co, cn)}（リード中央 {med(leads)}）"
            if len(tiers) > 1:
                line += " " + " ".join(f"{t}:{v[0]}/{v[1]}" for t, v in sorted(tiers.items()))
        elif leads:
            line += f"（リード中央 {med(leads)}）"
        R.append(line)
    R.append("")

    # ---- 交差点100 ----
    p = load_pred(PRED / "evcross_all.csv")
    ok, leads = 0, []
    for r in plan("cross100"):
        c = r["clip_id"]
        fs = fires(c, p)
        acc_false(c, fs)
        t = first(fs, 0)
        if t is not None:
            ok += 1
            tp = next(s["t_pass_s"] for s in scene(c)["sources"] if s["class"] == "siren")
            leads.append(tp - t)
    R += [f"## 交差点サイレン100: 通知 {pct(ok, 100)}、通過の{med(leads)}前"
          f"（最小 {min(leads):.1f}s、≥2.5s: {np.mean(np.array(leads) >= 2.5):.0%}）", ""]

    # ---- 複数車200 ----
    p = load_pred(PRED / "evmulti_all.csv")
    simu = {2: [0, 0], 3: [0, 0, 0]}
    for r in plan("multi200"):
        c = r["clip_id"]
        acc_false(c, fires(c, p))
        labels = defaultdict(list)
        for line in open(DS / "metadata" / f"{c}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                labels[int(q[0])].append(int(q[1]))
        for k, evs in labels.items():
            ncar = sum(1 for e in evs if e == CAR)
            npred = len([e for e in p.get(c, {}).get(k, []) if e[0] == CAR])
            if ncar == 2:
                simu[2][1] += 1
                simu[2][0] += int(npred >= 2)
            elif ncar == 3:
                simu[3][2] += 1
                simu[3][0] += int(npred >= 2)
                simu[3][1] += int(npred >= 3)
    R += [f"## 複数車200（n=200での再測。v10a60では2台69.1%/3台全29.9%）",
          f"- 2台両方: **{pct(simu[2][0], simu[2][1])}** / "
          f"3台で2+: {pct(simu[3][0], simu[3][2])} / 3台全部: {pct(simu[3][1], simu[3][2])}", ""]

    # ---- プローブ96 ----
    p = load_pred(PRED / "evprobe_all.csv")
    okc, purs = 0, []
    for r in plan("probe96"):
        c = r["clip_id"]
        gt = CLS["car_drive" if r["scenario"] == "probe_car_drive"
                 else r["scenario"].replace("probe_", "")]
        votes = [e[0] for k, evs in p.get(c, {}).items() if 30 <= k < 70 for e in evs]
        if votes:
            maj = max(set(votes), key=votes.count)
            okc += int(maj == gt)
            purs.append(votes.count(gt) / len(votes))
    R += [f"## プローブ96（音量なし音色識別、n倍増）: 正解 **{okc}/96**、"
          f"純度中央値 {np.median(purs):.0%}", ""]

    # ---- N1-N7 ----
    p = load_pred(PRED / "evn_all.csv")
    R.append("## 新種N1-N7ベースライン（各150本。合格線なし=弱点マップ）")
    for which, name in [("n1", "N1 突然出現"), ("n2", "N2 静音EV"), ("n3", "N3 駐車場多重"),
                        ("n4", "N4 高速サイレン"), ("n5", "N5 繁華街"),
                        ("n6", "N6 至近追い越し"), ("n7", "N7 停車→発進")]:
        rows = plan(which)
        stats = defaultdict(list)
        for r in rows:
            c = r["clip_id"]
            fs = fires(c, p)
            acc_false(c, fs)
            sc = scene(c)
            if which in ("n1", "n2", "n7"):
                t = first(fs, CAR)
                stats["ok"].append(int(t is not None))
                if which == "n2":
                    stats["aud"].append(int((DS / "metadata" / f"{c}.csv").stat().st_size > 0))
                    if stats["aud"][-1]:
                        stats["ok_aud"].append(int(t is not None))
                if t is not None:
                    src = sc["sources"][0]
                    if which == "n7":
                        stats["lead"].append(t - src["t_start_s"])      # 反応時間
                    elif src.get("t_cpa_rel_s"):
                        stats["lead"].append(src["t_cpa_rel_s"] - t)    # リード
            elif which in ("n4", "n6"):
                ci = 0 if which == "n4" else 3
                t = first(fs, ci)
                stats["ok"].append(int(t is not None))
                if t is not None:
                    stats["lead"].append(sc["sources"][0]["t_cpa_rel_s"] - t)
            elif which == "n3":
                stats["ok"].append(int(any(cc == 2 for _, cc, _ in fs)))
                stats["car"].append(int(any(cc == CAR for _, cc, _ in fs)))
            elif which == "n5":
                for key in ("w1", "w2"):
                    stats["ok"].append(int(any(cc == CLS[r[f"{key}_class"]]
                                               for _, cc, _ in fs)))
                stats["car"].append(int(any(cc == CAR for _, cc, _ in fs)))
        ok = sum(stats["ok"])
        line = f"- **{name}**: 通知 {pct(ok, len(stats['ok']))}"
        if stats.get("lead"):
            tag = "反応中央" if which == "n7" else "リード中央"
            line += f"（{tag} {med(stats['lead'])}）"
        if which == "n2":
            line += (f" / 可聴クリップ{sum(stats['aud'])}本に限ると "
                     f"{pct(sum(stats['ok_aud']), len(stats['ok_aud']))}"
                     f"（完全不可聴{150-sum(stats['aud'])}本=物理限界）")
        if stats.get("car"):
            line += f" / 車通知 {pct(sum(stats['car']), len(stats['car']))}"
        R.append(line)
    up_all = m12.poisson_upper95(false_total, exposure_h)
    R += ["", f"## 誤通知（全3,246本=9.02hの露出、シーンに無いクラスの発火）",
          f"- **{false_total}件/9.02h** → Poisson95%上限 **{up_all:.2f}回/時**", ""]

    (OUT / "v11eval_summary.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
