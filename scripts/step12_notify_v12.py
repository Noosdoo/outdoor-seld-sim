# -*- coding: utf-8 -*-
"""通知層v12: 新クラス（キックボード・バイク）への通知規則の初設計＋採点。

規則（宣言=新自由度なし。第8回監査の事後変更指摘を踏まえ、車v3.2の距離規則を
そのまま流用し、クラス別の閾値調整は行わない）:
- 対象クラス: 車(4)・キックボード(6)・バイク(7) = 歩行者に接近しうる移動体
- 強(役割③): 当該クラスの推定距離≤1.5mが2フレーム連続
- 中(役割②): 同≤3.0mが2フレーム連続
- 帰属: トリガ方位とGT方位の最近傍±25°（同クラス内）
- GT tier: トラックの最小GT距離で 重大≤1.5 / 注意≤3.2 / 安全>3.2
- v1知覚ゲート（音量ベースの弱通知）は6クラス車道音向け設計のため新クラスには
  適用しない=距離トリガのみの成績。車も同じ土俵（距離トリガのみ）で併記する
  （v3.2フル規則の車成績とは別物。比較の土俵を明記）

入力: out/predictions_v12_val/val_all.csv（7列）・out/dataset_outdoor_siren_v12/metadata_dist
出力: out/v12_notify_newcls/notify_newcls_val.md
使い方: python scripts/step12_notify_v12.py [pred_csv] [out_dir]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "out" / "predictions_v12_val" / "val_all.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "out" / "v12_notify_newcls"
DS = ROOT / "out" / "dataset_outdoor_siren_v12"
CLASSES = {4: "車", 6: "キックボード", 7: "バイク"}
AZ_MATCH = 25.0
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
    hits, prev = [], False
    for j in range(100):
        close = [(a, d) for a, d in dseq.get(j, []) if d <= thresh]
        if close and prev:
            a, _ = min(close, key=lambda x: x[1])
            hits.append((j, a))
        prev = bool(close)
    return hits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 予測: dists[cls][clip][frame] = [(az, dist), ...]
    dists = {c: defaultdict(lambda: defaultdict(list)) for c in CLASSES}
    clips = set()
    for line in open(PRED, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        clips.add(p[0])
        c = int(p[2])
        if c in CLASSES:
            dists[c][p[0]][int(p[1])].append(
                (float(p[4]), max(float(p[6]), 0.0)))

    R = ["# 通知層v12: 新クラス通知規則（距離トリガのみ）val採点", "",
         "規則=車v3.2の距離規則を全クラス共通で流用（クラス別調整なし・新自由度なし）。",
         "強: 推定距離≤1.5m×2フレーム連続 / 中: ≤3.0m×2フレーム連続 / 帰属±25°。",
         "v1知覚ゲートは適用外＝車の行もこの土俵（v3.2フル規則の車成績とは別物）。", ""]

    for cls, name in CLASSES.items():
        conf = defaultdict(int)
        n_unnotified = defaultdict(int)
        n_trig_unattr = {"強": 0, "中": 0}
        n_false_clips = {"強": 0, "中": 0}
        n_absent_clips = 0
        for clip in sorted(clips):
            gt_tracks = defaultdict(dict)
            mp = DS / "metadata_dist" / f"{clip}.csv"
            if mp.exists():
                for line in open(mp, encoding="utf-8"):
                    g = line.strip().split(",")
                    if len(g) == 6 and int(g[1]) == cls:
                        gt_tracks[int(g[2])][int(g[0])] = (float(g[3]),
                                                           float(g[5]))
            trig = {"強": dist_triggers(dists[cls][clip], T3),
                    "中": dist_triggers(dists[cls][clip], T2)}
            if not gt_tracks:
                n_absent_clips += 1
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

            for lv in ("中", "強"):
                for j, a in trig[lv]:
                    tr = attribute(j, a)
                    if tr is None:
                        n_trig_unattr[lv] += 1
                        continue
                    if best[tr] is None or ORDER[lv] > ORDER.get(best[tr], 0):
                        best[tr] = lv

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
              f"- {name}なし{n_absent_clips}本での誤発火clip: "
              f"強{n_false_clips['強']} / 中{n_false_clips['中']} / "
              f"帰属不能トリガ 強{n_trig_unattr['強']}件・中{n_trig_unattr['中']}件",
              ""]

    (OUT / "notify_newcls_val.md").write_text("\n".join(R) + "\n",
                                              encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
