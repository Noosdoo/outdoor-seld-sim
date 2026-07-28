# -*- coding: utf-8 -*-
"""通知ルールv2候補の掃引: 車の発火条件に「雑音床からの持ち上がり >= Δ dB」を追加。

背景: v11でsafe層(CPA3.2-15m)の過剰通知87-92.7%（距離カーブ: 3-5m 97%→12-15m 78%）。
ルールv1は接近(レベル上昇)しか見ないため遠い車でも鳴る。役割②=弱通知の思想に合わせ、
「十分近い（=雑音床より十分持ち上がった）車だけ役割②を発火」させるΔを設計する。
警告音（役割①）の規則は一切変更しない。

掃引: Δ ∈ {0(=v1相当), 3, 6, 9, 12} dB、雑音床=クリップのフレームA特性レベル10パーセンタイル
評価: ①safe600の過剰通知率（↓させたい） ②v11 val危険層の通知率・リード（維持したい）
      ③N1突然出現の通知率（維持したい）
出力: out/step12_notify_v11/notify_v2_sweep.md
"""
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

DS11 = ROOT / "out" / "dataset_outdoor_siren_v11"
DSEV = ROOT / "out" / "dataset_outdoor_siren_v11_eval"
CAR = 4
DELTAS = [0.0, 3.0, 6.0, 9.0, 12.0]


def load_pred(path):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def car_fires_v2(pred_clip, lv, delta):
    """ルールv1の車条件＋「lv[k] - 雑音床 >= delta」。v1のロジックを踏襲（車のみ）。"""
    floor = float(np.percentile(lv, 10))
    byframe = {k: {c: (a, e) for c, a, e in evs} for k, evs in pred_clip.items()}
    fires, last = [], []
    for k in range(100):
        ks = list(range(k - m12.CAR_WIN + 1, k + 1))
        if ks[0] < 0:
            continue
        hits = [kk for kk in ks if CAR in byframe.get(kk, {})]
        if len(hits) < m12.CAR_MIN_HITS:
            continue
        azs = [byframe[kk][CAR][0] for kk in hits]
        if m12.circ_drift(azs) > m12.CAR_AZ_DRIFT_MAX:
            continue
        seg = lv[ks[0]:k + 1]
        if np.polyfit(np.arange(len(seg)), seg, 1)[0] <= 0.0:
            continue
        if lv[k] - floor < delta:                       # ← v2の追加条件
            continue
        if any(k - kp < m12.REFRACTORY and
               abs((azs[-1] - ap + 180) % 360 - 180) <= m12.DIR_REFRACT_DEG
               for kp, ap in last):
            continue
        fires.append(k)
        last.append((k, azs[-1]))
    return fires


def eval_set(rows, ds, pred, tag):
    """各Δで (通知数, リード中央のためのリスト) を返す。"""
    res = {d: {"n": 0, "ok": 0, "leads": []} for d in DELTAS}
    for r in rows:
        c = r["clip_id"]
        mix = np.asarray(sf.read(ds / "foa" / f"{c}.flac")[0], np.float64).T
        lv = frame_spl_a(mix[0], 24000)
        sc = json.loads((ds / "work" / c / "scene.json").read_text())
        t_cpa = sc.get("cpa_rel_time_s")
        for d in DELTAS:
            fs = car_fires_v2(pred.get(c, {}), lv, d)
            res[d]["n"] += 1
            if fs:
                res[d]["ok"] += 1
                if t_cpa is not None:
                    res[d]["leads"].append(t_cpa - m12.emit_time(fs[0]))
    return res


def main():
    R = ["# 通知ルールv2掃引（車の発火に「雑音床+ΔdB」条件を追加。警告音は不変）", ""]
    # ① safe600（過剰通知=下げたい）
    p = load_pred(ROOT / "out" / "predictions_v11eval" / "evsafe_all.csv")
    rows = list(csv.DictReader(open(DSEV / "plan" / "assignment_safe600.csv")))
    safe = eval_set(rows, DSEV, p, "safe")
    # ② v11 val 危険層（維持したい）
    p = load_pred(ROOT / "out" / "predictions_v11" / "val_all.csv")
    core = list(csv.DictReader(open(DS11 / "plan" / "assignment_core.csv")))
    dang = [r for r in core if r["split"] == "fold2" and r["n_car"] != "0"
            and r["danger_tier"] in ("critical", "caution")]
    danger = eval_set(dang, DS11, p, "danger")
    # ③ N1突然出現（維持したい）
    p = load_pred(ROOT / "out" / "predictions_v11eval" / "evn_all.csv")
    rows = list(csv.DictReader(open(DSEV / "plan" / "assignment_n1.csv")))
    n1 = eval_set(rows, DSEV, p, "n1")

    R.append("| Δ[dB] | safe過剰通知↓ | 危険層通知↑ | 危険層リード中央 | N1通知↑ |")
    R.append("| --- | --- | --- | --- | --- |")
    for d in DELTAS:
        s, g, n = safe[d], danger[d], n1[d]
        lead = np.median(g["leads"]) if g["leads"] else float("nan")
        R.append(f"| {d:.0f} | {s['ok']}/{s['n']} ({100*s['ok']/s['n']:.1f}%) "
                 f"| {g['ok']}/{g['n']} ({100*g['ok']/g['n']:.1f}%) "
                 f"| {lead:.2f}s | {n['ok']}/{n['n']} ({100*n['ok']/n['n']:.1f}%) |")
    R += ["", "読み方: Δ=0がルールv1相当。膝=「危険層とN1をほぼ落とさずsafeを最大限削るΔ」。",
          "採用Δは本表の膝で決定し、警告音（役割①）の規則・数値は一切不変。"]
    out = ROOT / "out" / "step12_notify_v11" / "notify_v2_sweep.md"
    out.write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
