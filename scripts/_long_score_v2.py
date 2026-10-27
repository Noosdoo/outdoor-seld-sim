# -*- coding: utf-8 -*-
"""長尺セット v1 の採点（2026-09-03・宣言 §4・**2026-10-20 scripts_archive から scripts/ に戻した版 _long_score_v2**: 変更票 #72 の条件 (2a)〔中/分 +10% 以内〕を表あり・なしで比べる。中身の数字は同じ・--dist-table あり・場面ごとの合計行〔(scene, all)〕を足した。(2b)〔安全な車への強〕は _long_breakdown_v2.py で見る）: 注意/分・強/分・至近の強到達・流れモードの害・警告音の不変。

入力: 予測 csv（clip,frame,class,track,az,el,dist。_causal_infer_long.py の val_all_causal.csv）、
      GT out/dataset_outdoor_long_v1/metadata_dist/<clip>.csv、計画表 plan/assignment_long_v1.csv
通知: v4.3（winner.json）＋警告音 hold を nframes=600 で当てる。帰属は方位帰属（±0.5 s・≤30°）。
流れモード（v4.5 候補・宣言 §5）: --flow K,W,C → 同じ側（方位の左右）で「中」が K 回 W 秒以内に続いたら以後の「中」を抑える。
                                     解除 = その側で強 ／ C 秒その側で中が無い。強・警告音は抑えない。

使い方:
  python scripts/_long_score.py --pred out/long_v1/pred.csv [--split fold40] [--flow 2,20,20] [--out md]
  python scripts/_long_score.py --oracle [--flow 2,20,20]          # GT を予測の代わりに（自己検査・規則だけの上限）
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _sc(name):
    """2026-09-12 の整理で scripts_archive/ へ移した分の道しるべ。

    探す順: ①フォルダまで書いてあればその通り ②自分の隣 ③scripts/。
    ②と③の両方に同じ名前があって、どちらか指定が無いときは**黙って選ばず例外**にする
    （独立監査 R9・2026-09-13。以前は隣を優先していたので、scripts/ を明示しても
    退避版が返っていた）。
    """
    _t = str(name).replace("\\", "/")
    _root = ROOT
    if "/" in _t and (_root / _t).exists():
        return _root / _t
    _base = _t.split("/")[-1]
    _here = Path(__file__).resolve().parent / _base
    _there = _root / "scripts" / _base
    if _here.resolve() == _there.resolve():
        return _there                                   # scripts/ に戻した版（2026-10-20）
    if _here.exists() and _there.exists():
        raise FileNotFoundError(
            "%s が scripts/ と %s/ の両方にあります。どちらか指定してください"
            "（例: _sc(\'scripts/%s\')）" % (_base, _here.parent.name, _base))
    return _here if _here.exists() else _there

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 2026-09-12: 退避先の隣も見る
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _sc(fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V42 = _load("nv42", "step12_notify_v42_bearing.py")
V43 = _load("nv43", "step12_notify_v43.py")
H = _load("nhold", "step12_notify_v9b_hold.py")
v4 = sys.modules["nv42"].v4
C43 = V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))

DS = ROOT / "out/dataset_outdoor_long_v1"
NFR = 600
DIST_CLASSES = {4, 6, 7}
WARN_CLASSES = {0, 1, 2, 3, 5}
CLOSE_M = 1.5
ATTR_FR, ATTR_DEG = 5, 30.0


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def load_gt(clip):
    """GT: {(cls,track): {frame: (az, dist)}}"""
    g = defaultdict(dict)
    p = DS / "metadata_dist" / f"{clip}.csv"
    if not p.exists():
        return g
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split(",")
        if len(q) < 6:
            continue
        g[(int(q[1]), int(q[2]))][int(q[0])] = (float(q[3]), float(q[5]))
    return g


def gt_as_pred(gt):
    """GT → 予測と同じ形（frames_dist: frame→[(cls,az,dist)], frames_warn: frame→[(cls,az,el)]）"""
    fd, fw = defaultdict(list), defaultdict(list)
    for (cls, _tr), fr in gt.items():
        for k, (az, d) in fr.items():
            if cls in DIST_CLASSES:
                fd[k].append((cls, az, d))
            else:
                fw[k].append((cls, az, 0.0))
    return dict(fd), dict(fw)


DIST_TABLE = None      # --dist-table <json>（_dist_calib_proto.py の表）: 予測距離を規則の前で補正する（v2 束 §6.1 判断 1）


def load_dist_table(path):
    import numpy as np
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    return (np.array(t["centers"], dtype=float), np.array(t["values"], dtype=float))


DIST_TABLE_MAX = 5.0   # --dist-table-max <m>: この予測距離以下だけ補正する（既定 5 m = 表の全域）


def corr_d(d):
    if DIST_TABLE is None or not (d == d):
        return d
    import numpy as np
    if d >= min(5.0, DIST_TABLE_MAX):
        return d
    return float(np.interp(d, DIST_TABLE[0], DIST_TABLE[1]))


def load_pred_long(path):
    """予測 csv → clip → (frames_dist, frames_warn)。DIST_TABLE があれば距離を補正して読む。"""
    out = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        q = line.strip().split(",")
        if len(q) < 6:
            continue
        clip, k, cls, az = q[0], int(q[1]), int(q[2]), float(q[4])
        if cls in DIST_CLASSES:
            d = float(q[6]) if len(q) > 6 and q[6] != "" else float("nan")
            d = corr_d(d)
            out[clip][0][k].append((cls, az, d))
        else:
            out[clip][1][k].append((cls, az, float(q[5]) if len(q) > 5 else 0.0))
    return out


def episodes(frames_dist, frames_warn):
    """v4.3＋hold の発火 [(frame, az, tier, cls)]（tier: 強/中/警告）"""
    eps = []
    res = V43.run_rule3({"x": frames_dist}, C43, nframes=NFR).get("x", {})
    for cls, lst in res.items():
        for j, az, tier, _d in lst:
            eps.append((j, az, tier, cls))
    for k, cls, az in H.warn_fires(frames_warn, hold=True, nframes=NFR):
        eps.append((k, az, "警告", cls))
    return sorted(eps)


def flow_filter(eps, K, W, C):
    """流れモード候補: 側（az>0=左）ごとに、直近 W 秒に届けた中が K 回以上なら中を抑える。解除は強／C 秒無音。
    返り値: [(frame, az, tier, cls, delivered)]"""
    out = []
    delivered_mid = {"L": [], "R": []}      # 届けた中のフレーム
    suppress_until = {"L": -1, "R": -1}
    last_mid = {"L": -10**9, "R": -10**9}
    for j, az, tier, cls in eps:
        side = "L" if az > 0 else "R"
        if tier != "中":
            if tier == "強":
                suppress_until[side] = -1
                delivered_mid[side] = []
            out.append((j, az, tier, cls, True))
            continue
        if j - last_mid[side] > C * 10:            # C 秒その側で中が無かった → 解除
            suppress_until[side] = -1
            delivered_mid[side] = []
        last_mid[side] = j
        recent = [f for f in delivered_mid[side] if j - f <= W * 10]
        delivered_mid[side] = recent
        if len(recent) >= K:
            out.append((j, az, tier, cls, False))
        else:
            delivered_mid[side].append(j)
            out.append((j, az, tier, cls, True))
    return out


def attribute(ep, gt):
    """発火 (frame, az) → GT の (cls, track)（±ATTR_FR フレーム・≤ATTR_DEG°、最も近い方位）"""
    j, az = ep[0], ep[1]
    best, bestd = None, 999.0
    for key, fr in gt.items():
        if key[0] not in DIST_CLASSES:
            continue
        for k in range(j - ATTR_FR, j + ATTR_FR + 1):
            if k in fr:
                d = dang(az, fr[k][0])
                if d <= ATTR_DEG and d < bestd:
                    best, bestd = key, d
    return best


def score_clip(clip, frames_dist, frames_warn, gt, flow):
    eps = episodes(frames_dist, frames_warn)
    rows = flow_filter(eps, *flow) if flow else [(j, az, t, c, True) for j, az, t, c in eps]
    mid_deliv = sum(1 for r in rows if r[2] == "中" and r[4])
    mid_supp = sum(1 for r in rows if r[2] == "中" and not r[4])
    strong = sum(1 for r in rows if r[2] == "強")
    warn = sum(1 for r in rows if r[2] == "警告")
    # 害: 抑えた中の車が、その後に至近になるか
    harm = 0
    for r in rows:
        if r[2] == "中" and not r[4]:
            key = attribute(r, gt)
            if key is not None:
                fut = [d for k, (_a, d) in gt[key].items() if k >= r[0]]
                if fut and min(fut) <= CLOSE_M:
                    harm += 1
    # 至近の強到達: 至近になる車に、至近になる前に強（方位帰属）が出たか
    close_cars = reached = 0
    for key, fr in gt.items():
        if key[0] not in DIST_CLASSES:
            continue
        ks = [k for k, (_a, d) in fr.items() if d <= CLOSE_M]
        if not ks:
            continue
        close_cars += 1
        k0 = min(ks)
        ok = False
        for r in rows:
            if r[2] == "強" and r[0] <= k0 + ATTR_FR and attribute(r, gt) == key:
                ok = True
                break
        reached += int(ok)
    return {"mid": mid_deliv, "mid_supp": mid_supp, "strong": strong, "warn": warn, "harm": harm,
            "close_cars": close_cars, "reached": reached}


def main() -> int:
    a = sys.argv
    flow = tuple(float(x) for x in a[a.index("--flow") + 1].split(",")) if "--flow" in a else None
    if flow:
        flow = (int(flow[0]), flow[1], flow[2])
    split = a[a.index("--split") + 1] if "--split" in a else None
    global DIST_TABLE, DIST_TABLE_MAX
    if "--dist-table" in a:
        DIST_TABLE = load_dist_table(a[a.index("--dist-table") + 1])
        if "--dist-table-max" in a:
            DIST_TABLE_MAX = float(a[a.index("--dist-table-max") + 1])
        print(f"# 距離の補正表を適用: {a[a.index('--dist-table') + 1]}（予測 ≤{DIST_TABLE_MAX:g} m だけ）")
    plan = {r["clip_id"]: r for r in csv.DictReader(open(DS / "plan/assignment_long_v1.csv", encoding="utf-8"))}
    clips = [c for c in plan if (DS / "metadata_dist" / f"{c}.csv").exists() and (split is None or plan[c]["split"] == split)]
    preds = None if "--oracle" in a else load_pred_long(a[a.index("--pred") + 1])
    agg = defaultdict(lambda: defaultdict(float))
    for clip in clips:
        gt = load_gt(clip)
        if preds is None:
            fd, fw = gt_as_pred(gt)
        else:
            fd, fw = preds.get(clip, ({}, {}))
        s = score_clip(clip, fd, fw, gt, flow)
        key = (plan[clip]["scene"], plan[clip]["stratum"])
        for k, v in s.items():
            agg[key][k] += v
        agg[key]["n"] += 1
        for k, v in s.items():
            agg[("all", "all")][k] += v
        agg[("all", "all")]["n"] += 1
    label = "オラクル（GT→規則）" if preds is None else "モデル予測"
    L = [f"# 長尺セット v1 採点 — {label}" + (f" / 流れモード K={flow[0]} W={flow[1]}s C={flow[2]}s" if flow else " / 流れモードなし")
         + (f" / {split}" if split else ""), "",
         "| 場面 | 交通量 | 本数 | 中/分 | 抑えた中/分 | 害（抑えた中のうち後で至近） | 強/分 | 至近の車 | 至近の強到達 | 警告/本 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (scene, st), g in list(agg.items()):                                    # 場面ごとの合計（#72 (2a) は幹線歩道 arterial_walk の行でも見る）
        if scene != "all":
            for k, v in g.items():
                agg[(scene, "all")][k] += v
    order = sorted(agg.keys(), key=lambda k: (k[0] == "all", k[0], {"low": 0, "mid": 1, "high": 2, "all": 3}.get(k[1], 9)))
    for key in order:
        g = agg[key]; n = max(g["n"], 1)
        harm_txt = f"{g['harm']:.0f}/{g['mid_supp']:.0f}" + (f"（{100*g['harm']/g['mid_supp']:.1f}%）" if g["mid_supp"] else "")
        reach_txt = f"{100*g['reached']/g['close_cars']:.1f}%" if g["close_cars"] else "—"
        L.append(f"| {key[0]} | {key[1]} | {g['n']:.0f} | {g['mid']/n:.2f} | {g['mid_supp']/n:.2f} | {harm_txt} | {g['strong']/n:.2f} | {g['close_cars']:.0f} | {reach_txt} | {g['warn']/n:.2f} |")
    txt = "\n".join(L)
    print(txt)
    if "--out" in a:
        Path(a[a.index("--out") + 1]).write_text(txt + "\n", encoding="utf-8")
    if "--json" in a:                                                            # R72-08: 丸め前の分子・分母（中/分 = mid / 分数）
        J = {f"{k[0]}/{k[1]}": {kk: float(vv) for kk, vv in g.items()} for k, g in agg.items()}
        for k, g in J.items():
            g["minutes"] = g["n"] * NFR / 10.0 / 60.0; g["mid_per_min"] = g["mid"] / g["minutes"] if g["minutes"] else None
        Path(a[a.index("--json") + 1]).write_text(json.dumps({"split": split, "pred": a[a.index("--pred") + 1] if "--pred" in a else "oracle", "dist_table": a[a.index("--dist-table") + 1] if "--dist-table" in a else "", "groups": J}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
