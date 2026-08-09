# -*- coding: utf-8 -*-
"""通知層v12b: 新クラス通知規則のSol再監査対応版（v12初版からの差分は3点）。

①【条件1】距離トリガに同一物体の方位連続性（前フレーム候補と±25°連結）を要求
②【条件2】metadata_dist manifest基準の全クリップ巡回（予測ゼロクリップも分母に）
③【条件4】帰属不能トリガのepisode分類と完全誤報KPI併記

規則自体はv12初版と同じ（車v3.3の距離規則を全クラス共通流用・クラス別調整なし）。
入力: out/predictions_v12_val系CSV(7列)・out/dataset_outdoor_siren_v12/metadata_dist
使い方: python scripts/step12_notify_v12b.py [pred_csv] [out_dir]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "out" / "predictions_v12_val" / "val_all.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "out" / "v12_notify_newcls_v2"
DS = ROOT / "out" / "dataset_outdoor_siren_v12"
CLASSES = {4: "車", 6: "キックボード", 7: "バイク"}
AZ_MATCH = 25.0
LINK_DEG = float(os.environ.get("NOTIFY_LINK_DEG", "25.0"))  # v3.4=60(設計定数由来)
T3, T2, SUPP = 1.5, 3.0, 3.2
ORDER = {"中": 1, "強": 2}


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def gt_tier(dmin):
    if dmin <= T3:
        return "重大"
    if dmin <= SUPP:
        return "注意"
    return "安全"


def dist_triggers(dseq, thresh):
    """≤threshの2フレーム連続＋前フレーム候補との方位±25°連結（v3.3と同一）。"""
    hits, prev = [], []
    for j in range(100):
        close = [(a, d) for a, d in dseq.get(j, []) if d <= thresh]
        if close and prev:
            linked = [(a, d) for a, d in close
                      if any(cdiff(a, pa) <= LINK_DEG for pa, _ in prev)]
            if linked:
                a, _ = min(linked, key=lambda x: x[1])
                hits.append((j, a))
        prev = close
    return hits


def group_episodes(hits):
    eps = []
    for j, a in hits:
        if eps and j - eps[-1][-1][0] <= 1 and cdiff(a, eps[-1][-1][1]) <= AZ_MATCH:
            eps[-1].append((j, a))
        else:
            eps.append([(j, a)])
    return eps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dists = {c: defaultdict(lambda: defaultdict(list)) for c in CLASSES}
    pred_keys = set()
    for line in open(PRED, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        pred_keys.add(p[0])
        c = int(p[2])
        if c in CLASSES:
            dists[c][p[0]][int(p[1])].append(
                (float(p[4]), max(float(p[6]), 0.0)))

    prefixes = {k.rsplit("_mix", 1)[0] for k in pred_keys}
    clips = sorted({p.stem for p in (DS / "metadata_dist").glob("*.csv")
                    if p.stem.rsplit("_mix", 1)[0] in prefixes} | pred_keys)
    n_nopred = sum(1 for c in clips if c not in pred_keys)

    R = ["# 通知層v12b: 新クラス通知（Sol再監査対応版）", "",
         f"対象クリップ: {len(clips):,}本（manifest基準・予測ゼロ{n_nopred}本含む）。",
         f"規則=車v3.3系の距離規則を全クラス共通流用（連結幅±{LINK_DEG:.0f}°）。"
         "強: ≤1.5m×2フレーム連続＋方位連結 /",
         "中: ≤3.0m×同 / 帰属±25°。v1知覚ゲートは適用外（車の行もこの土俵）。", ""]

    for cls, name in CLASSES.items():
        conf = defaultdict(int)
        n_unnotified = defaultdict(int)
        ep_cls = {"強": defaultdict(int), "中": defaultdict(int)}
        ep_false_clips = {"強": set(), "中": set()}
        n_false_clips = {"強": 0, "中": 0}
        n_absent = 0
        for clip in clips:
            gt_tracks = defaultdict(dict)
            mp = DS / "metadata_dist" / f"{clip}.csv"
            if mp.exists():
                for line in open(mp, encoding="utf-8"):
                    g = line.strip().split(",")
                    if len(g) == 6 and int(g[1]) == cls:
                        gt_tracks[int(g[2])][int(g[0])] = (float(g[3]),
                                                           float(g[5]))
            dv = dists[cls].get(clip, {})
            trig = {"強": dist_triggers(dv, T3), "中": dist_triggers(dv, T2)}
            if not gt_tracks:
                n_absent += 1
                for lv in ("強", "中"):
                    n = len(trig[lv]) if lv == "強" else \
                        len(trig["中"]) - len(trig["強"])
                    n_false_clips[lv] += int(n > 0)
                continue

            best = {tr: None for tr in gt_tracks}

            def attribute(k, az):
                cands = [(tr, cdiff(az, fr[k][0]))
                         for tr, fr in gt_tracks.items() if k in fr]
                if not cands:
                    return None
                tr, dd = min(cands, key=lambda x: x[1])
                return tr if dd <= AZ_MATCH else None

            unattr = {"強": [], "中": []}
            for lv in ("中", "強"):
                for j, a in trig[lv]:
                    tr = attribute(j, a)
                    if tr is None:
                        unattr[lv].append((j, a))
                        continue
                    if best[tr] is None or ORDER[lv] > ORDER.get(best[tr], 0):
                        best[tr] = lv

            for lv in ("強", "中"):
                for ep in group_episodes(unattr[lv]):
                    diffs = [cdiff(a, fr[j][0]) for (j, a) in ep
                             for fr in gt_tracks.values() if j in fr]
                    if not diffs:
                        c_ = "GT行なし(完全誤報)"
                        ep_false_clips[lv].add(clip)
                    elif min(diffs) <= 45.0:
                        c_ = "実車スイープ疑い(≤45°)"
                    else:
                        c_ = "方位乖離(>45°)"
                    ep_cls[lv][c_] += 1

            for tr, gt in gt_tracks.items():
                t = gt_tier(min(v[1] for v in gt.values()))
                if best[tr] is None:
                    n_unnotified[t] += 1
                else:
                    conf[(t, best[tr])] += 1

        def tot(t):
            return conf.get((t, "強"), 0) + conf.get((t, "中"), 0) \
                + n_unnotified.get(t, 0)

        n_sev, n_safe = tot("重大"), tot("安全")
        strong_ok = conf.get(("重大", "強"), 0)
        reach2 = strong_ok + conf.get(("重大", "中"), 0)
        safe_ok = n_unnotified.get("安全", 0)
        R += [f"## {name}（GTトラック単位）",
              "| GT＼出力 | 強(③) | 中(②) | 通知なし |",
              "| --- | --- | --- | --- |"]
        for t in ("重大", "注意", "安全"):
            R.append(f"| {t} (n={tot(t)}) | {conf.get((t, '強'), 0)} "
                     f"| {conf.get((t, '中'), 0)} | {n_unnotified.get(t, 0)} |")
        R += [f"- GT重大の強到達: **{strong_ok}/{n_sev} "
              f"({100*strong_ok/max(n_sev,1):.1f}%)** / 強or中到達: "
              f"{reach2}/{n_sev} ({100*reach2/max(n_sev,1):.1f}%)",
              f"- GT安全の非通知(抑制): {safe_ok}/{n_safe} "
              f"({100*safe_ok/max(n_safe,1):.1f}%) / 誤・強通知 "
              f"{conf.get(('安全', '強'), 0)}/{n_safe}",
              f"- {name}なし{n_absent}本での誤発火clip: 強{n_false_clips['強']} "
              f"/ 中{n_false_clips['中']}",
              f"- 帰属不能episode: 強[スイープ疑い"
              f"{ep_cls['強'].get('実車スイープ疑い(≤45°)', 0)}/方位乖離"
              f"{ep_cls['強'].get('方位乖離(>45°)', 0)}/完全誤報"
              f"{ep_cls['強'].get('GT行なし(完全誤報)', 0)}"
              f"({len(ep_false_clips['強'])}clip)] 中[同"
              f"{ep_cls['中'].get('実車スイープ疑い(≤45°)', 0)}/"
              f"{ep_cls['中'].get('方位乖離(>45°)', 0)}/"
              f"{ep_cls['中'].get('GT行なし(完全誤報)', 0)}"
              f"({len(ep_false_clips['中'])}clip)]",
              ""]

    (OUT / "notify_newcls_v2.md").write_text("\n".join(R) + "\n",
                                             encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
