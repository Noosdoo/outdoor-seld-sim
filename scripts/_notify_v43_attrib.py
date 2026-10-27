# -*- coding: utf-8 -*-
"""v4.3 段階1 — fire を「方位」で車に帰属する型別表と採点（2026-09-03）。

宣言= md/design/通知v4.3_検討の事前宣言_2026-09-03.md §1 段階1。

従来（窓帰属）: イベント窓 [f0−1s, cpa+1s] に入った同クラスの fire を全部そのイベントの結果と数える
（型別表 outcome()）。複数車クリップでは、近い車に正しく出た強が、同時刻の安全な車にも付く。
本版（方位帰属）: fire (j, az) を、同クラスで j±0.5s に存在するGTイベントのうち方位差最小（≤30°）の
ものに帰属する。該当なしは「幻影」。イベントの結果はその車に帰属した fire の最良（強>中>無）。

出力: <outdir>/q2_table_attrib.md（窓帰属と方位帰属の並記）／score_attrib.md（主要指標の両方式）
使い方: python scripts/_notify_v43_attrib.py <pred.csv> <meta_dir> <outdir> [--old42 --winner <winner.json>]
  --old42 --winner: 旧= v4.2採用構成 / 新= v4.3 勝ち構成（winner.json）で同じ表を出す
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V42 = _load("nv42", "step12_notify_v42_bearing.py")
DG = _load("nv42diag", "_notify_v42_diag.py")
Q2 = _load("nv42q2", "_notify_v42_q2_table.py")
EV = _load("nv42ev", "_notify_v42_eval.py")

FPS = v4.FPS
ADOPTED = V42.Cfg(route_c=True, adot_th=0.10, dn=15.0, link_pred=True, cpa_strong=1.3, cpa_mid=1.6)
# 方位帰属の許容（確定評価の既定＝30.0° / ±5 frame = ±0.5 s）。
# 環境変数での上書きは **感度の掃引専用**（2026-09-21・監査の軽微「20° や 45° で結論が動くか調べていない」への対応）。
# 変数を置かなければ従来と同一の値・同一の出力。確定評価では絶対に設定しない。
import os as _os

ATTRIB_DEFAULTS = (30.0, 5)                                       # 既定（確定評価の値）。ガードにも使う（監査 9/21 M05・9/22 R03）
AZ_TOL = float(_os.environ.get("NOTIFY_ATTRIB_AZ_TOL", str(ATTRIB_DEFAULTS[0])))   # [deg] 方位帰属の許容
T_TOL = int(_os.environ.get("NOTIFY_ATTRIB_T_TOL", str(ATTRIB_DEFAULTS[1])))       # [frame] ±0.5s
# 混入防止: 既定と違う値は、掃引の道具（_attrib_sweep.py）が NOTIFY_ATTRIB_SWEEP=1 を同時に置いたときだけ受け付ける。
# 置き忘れた環境変数で本番の採点が静かに変わるのを防ぐ（本番経路の検査 = test_score_unified_strict.py K/L）。
if (AZ_TOL, T_TOL) != ATTRIB_DEFAULTS and _os.environ.get("NOTIFY_ATTRIB_SWEEP") != "1":
    raise RuntimeError(f"帰属の許容が既定 {ATTRIB_DEFAULTS} と違う（AZ_TOL={AZ_TOL}, T_TOL={T_TOL}）のに NOTIFY_ATTRIB_SWEEP=1 が無い。"
                       "掃引以外では NOTIFY_ATTRIB_AZ_TOL / NOTIFY_ATTRIB_T_TOL を置かない（確定評価の混入防止）")
IS_SWEEP = (AZ_TOL, T_TOL) != ATTRIB_DEFAULTS


def settings() -> dict:
    """実効設定（採点 md の見出しと run の記録に写す）。default は呼ぶたびに計算する
    （読み込み後にプロセス内で AZ_TOL/T_TOL を書き換えた場合も見出しに「掃引の設定」が出るように。Opus 5.5 監査 9/23 F14）。"""
    return {"az_tol_deg": AZ_TOL, "t_tol_frame": T_TOL, "t_tol_s": T_TOL / FPS, "default": (AZ_TOL, T_TOL) == ATTRIB_DEFAULTS}


def attribute(fires, events):
    """fire ごとに帰属イベントの index（無ければ None）を返す。"""
    out = []
    for (j, az, tier, d, cls) in fires:
        best, best_e = None, None
        for i, ev in enumerate(events):
            if ev["cls"] != cls:
                continue
            ks = [k for k in range(j - T_TOL, j + T_TOL + 1) if k in ev["fr"]]
            if not ks:
                continue
            e = min(v4.cdiff(az, ev["fr"][k][0]) for k in ks)
            if e <= AZ_TOL and (best_e is None or e < best_e):
                best, best_e = i, e
        out.append(best)
    return out


def type_table(pred, meta_dir, res_old, res_new):
    rows_w = defaultdict(lambda: defaultdict(int))     # 窓帰属
    rows_a = defaultdict(lambda: defaultdict(int))     # 方位帰属
    ghost = {"旧": 0, "新": 0}
    for clip in sorted(pred):
        evs = [DG.mk_event(c, t, fr) for c, t, fr in DG.gt_tracks(meta_dir, clip)]
        typ = {}
        for i, ev in enumerate(evs):
            ad = Q2.gt_adot_before_cpa(ev["fr"], ev["cpa"])
            typ[i] = None if ad is None else ("対向型" if ad < Q2.ADOT_SPLIT else "すり抜け型")
        for tag, res in (("旧", res_old), ("新", res_new)):
            fires = [(f[0], f[1], f[2], f[3], cls)
                     for cls, eps in res.get(clip, {}).items() for f in eps]
            att = attribute(fires, evs)
            best = defaultdict(lambda: "無")
            for f, i in zip(fires, att):
                if i is None:
                    ghost[tag] += 1
                    continue
                if f[2] == "強" or (f[2] == "中" and best[i] == "無"):
                    best[i] = f[2]
            for i, ev in enumerate(evs):
                if typ[i] is None:
                    continue
                key = (ev["tier"], typ[i])
                rows_w[key][(tag, Q2.outcome(fires, ev))] += 1
                rows_a[key][(tag, best[i])] += 1
    return rows_w, rows_a, ghost


def score_attrib(pred, meta_dir, res):
    """方位帰属版の主要指標（強到達・至近到達・注意到達・安全抑制）。"""
    stat = defaultdict(lambda: [0, 0])
    n_strong_hit = 0
    for clip in sorted(pred):
        evs = [DG.mk_event(c, t, fr) for c, t, fr in DG.gt_tracks(meta_dir, clip)]
        fires = [(f[0], f[1], f[2], f[3], cls)
                 for cls, eps in res.get(clip, {}).items() for f in eps]
        att = attribute(fires, evs)
        got = defaultdict(set)
        for f, i in zip(fires, att):
            if i is not None and f[0] <= evs[i]["cpa"] + FPS:
                got[i].add(f[2])
        for i, ev in enumerate(evs):
            hit = bool(got[i])
            if ev["tier"] == "safe":
                stat["safe"][1] += 1
                stat["safe"][0] += int(not hit)
                continue
            stat[ev["tier"]][1] += 1
            stat[ev["tier"]][0] += int(hit)
            if ev["tier"] == "critical" and "強" in got[i]:
                n_strong_hit += 1
    g = lambda t: 100 * stat[t][0] / max(stat[t][1], 1)
    return dict(crit=g("critical"), strong=100 * n_strong_hit / max(stat["critical"][1], 1),
                caut=g("caution"), safe=g("safe"))


def main() -> int:
    pred_path, meta_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    pred = v4.load_pred(pred_path)
    if "--winner" in sys.argv:
        import json
        V43 = _load("nv43", "step12_notify_v43.py")
        C43 = V43.Cfg43(**json.loads(Path(sys.argv[sys.argv.index("--winner") + 1]).read_text()))
        res_old = V42.run_rule2(pred, ADOPTED)          # 旧= v4.2
        res_new = V43.run_rule3(pred, C43)              # 新= v4.3
        names = ("v4.2", "v4.3")
    else:
        res_old = v4.run_rule(pred, "cpa")
        res_new = V42.run_rule2(pred, ADOPTED)
        names = ("v4.1", "v4.2")
    rows_w, rows_a, ghost = type_table(pred, meta_dir, res_old, res_new)

    R = [f"# 型別表 — 窓帰属 vs 方位帰属 pred={pred_path.name}", "",
         f"- 旧= {names[0]} / 新= {names[1]}。方位帰属= fire を同クラス・j±0.5s・方位差≤{AZ_TOL:.0f}°で最も近い車へ。",
         f"- どの車にも帰属できなかった fire（幻影）: 旧 {ghost['旧']} / 新 {ghost['新']}", "",
         "| GT区分 | 型 | n | 窓帰属 旧:強/中/無 | 窓帰属 新:強/中/無 | **方位帰属 旧:強/中/無** | **方位帰属 新:強/中/無** |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda r, tag: "/".join(str(r.get((tag, x), 0)) for x in ("強", "中", "無"))
    for tier, tjp in (("critical", "重大(≤1.5m)"), ("caution", "注意(≤3.2m)"), ("safe", "安全(>3.2m)")):
        for typ in ("対向型", "すり抜け型"):
            key = (tier, typ)
            if key not in rows_w:
                continue
            n = sum(v for (tag, _), v in rows_w[key].items() if tag == "旧")
            R.append(f"| {tjp} | {typ} | {n:,} | {fmt(rows_w[key], '旧')} | {fmt(rows_w[key], '新')} "
                     f"| **{fmt(rows_a[key], '旧')}** | **{fmt(rows_a[key], '新')}** |")
    s_old_w = EV.score(pred, meta_dir, res_old)
    s_new_w = EV.score(pred, meta_dir, res_new)
    s_old_a = score_attrib(pred, meta_dir, res_old)
    s_new_a = score_attrib(pred, meta_dir, res_new)
    R += ["", "## 主要指標（窓帰属=採用済みの採点器 vs 方位帰属）", "",
          "| 規則 | 帰属 | 至近到達 | 強到達 | 注意到達 | 安全抑制 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, s_w, s_a in ((names[0], s_old_w, s_old_a), (names[1], s_new_w, s_new_a)):
        R.append(f"| {name} | 窓（採用済み） | {s_w['crit']:.1f}% | {s_w['strong']:.1f}% | {s_w['caut']:.1f}% | {s_w['safe']:.1f}% |")
        R.append(f"| {name} | 方位 | {s_a['crit']:.1f}% | {s_a['strong']:.1f}% | {s_a['caut']:.1f}% | {s_a['safe']:.1f}% |")
    (outdir / "q2_table_attrib.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))
    return 0


if __name__ == "__main__":
    sys.exit(main())
