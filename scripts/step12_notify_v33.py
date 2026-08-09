# -*- coding: utf-8 -*-
"""通知層v3.3: Sol第8回再監査（2026-08-09・不合格）対応版。

v3.2からの変更（ルール変更は①のみ。結果を見る前に凍結コミットする）:
①【条件1】距離トリガの「2フレーム連続」に同一物体の方位連続性を要求。
  v3.2は前フレームを真偽値でしか見ておらず、車A→車Bの乗り継ぎでも
  2連続が成立した（強到達の主経路の欠陥）。v3.3は現フレーム候補のうち
  前フレーム候補と方位±25°で連結するものだけを採用する。
②【条件2】採点の分母をmanifest基準に: 予測CSVに1行も無いクリップも
  metadata_distから列挙して巡回し、GT車トラックを未通知として計上する
  （v3.2は予測キー巡回のため予測ゼロクリップが分母から脱落していた）。
③【条件4】帰属不能トリガ（±25°超）を件数でなくepisode単位でまとめ、
  「実車スイープ疑い(≤45°)／方位乖離(>45°)／GT行なし(完全誤報)」に分類し、
  完全誤報episodeを誤・強通知KPIに併記する。

発火タイミング: 弱通知=v1ルール（変更なし）。強/中=距離トリガ（独立）。
入力: 予測CSV(7列 or 旧6列)・metadata_dist・v11 foa
使い方: python scripts/step12_notify_v33.py [pred_csv] [out_dir]
出力: <out_dir>/notify_v33.md
"""
from __future__ import annotations

import importlib.util
import os
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

DS = ROOT / "out" / "dataset_outdoor_siren_v11"
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "out" / "predictions_v11sde" / "val_all.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "out" / "step12_notify_v11sde_v33"
MANIFEST_FILTER = None   # サーバwrapper等がクリップ絞り込み関数を注入する
CAR = 4
AZ_MATCH = 25.0
# 連結幅(deg)。既定25=v3.3。v3.4=60: 事前凍結済みの設計定数のみから導出
# （車速上限15m/s・強トリガ域d=1.5m・フレーム0.1s → 最大方位変化
#   atanに拠らず上限 v/d = 10rad/s = 57.3°/フレーム → 60°に切上げ）。
# 選択の経緯はv3.3の結果を見た後だが、値自体は設計定数由来であることを
# 対応報告に明記し、以後の連結幅変更は行わない（停止規則）。
LINK_DEG = float(os.environ.get("NOTIFY_LINK_DEG", "25.0"))
T3, T2, SUPP = 1.5, 3.0, 3.2
ORDER = {"抑制": 0, "中": 1, "強": 2}


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def role_of(dmin):
    if dmin <= T3:
        return "強"
    if dmin <= SUPP:
        return "中"
    return "抑制"


def load_preds():
    pred_ev = defaultdict(lambda: defaultdict(list))
    dists = defaultdict(lambda: defaultdict(list))
    for line in open(PRED, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            clip, k, c, az, el, d = (p[0], int(p[1]), int(p[2]),
                                     float(p[4]), float(p[5]), float(p[6]))
        elif len(p) == 6:
            clip, k, c, az, el, d = (p[0], int(p[1]), int(p[2]),
                                     float(p[3]), float(p[4]), float(p[5]))
        else:
            continue
        pred_ev[clip][k].append((c, az, el))
        if c == CAR:
            dists[clip][k].append((az, max(d, 0.0)))
    return dict(pred_ev), dict(dists)


def manifest_clips(pred_keys):
    """【条件2】予測キーのfold接頭辞に一致する全metadata_distクリップ。"""
    prefixes = {k.rsplit("_mix", 1)[0] for k in pred_keys}
    clips = {p.stem for p in (DS / "metadata_dist").glob("*.csv")
             if p.stem.rsplit("_mix", 1)[0] in prefixes}
    clips |= set(pred_keys)
    if MANIFEST_FILTER is not None:
        clips = {c for c in clips if MANIFEST_FILTER(c)}
    return sorted(clips)


def dist_triggers(dseq, thresh):
    """【条件1】≤threshが2フレーム連続、かつ前フレーム候補と方位±25°で
    連結する候補のみ採用（同一物体の連続性）。"""
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
    """【条件4】(frame, az) を連続フレーム＆方位±25°連結でepisodeにまとめる。"""
    eps = []
    for j, a in hits:
        if eps and j - eps[-1][-1][0] <= 1 and cdiff(a, eps[-1][-1][1]) <= AZ_MATCH:
            eps[-1].append((j, a))
        else:
            eps.append([(j, a)])
    return eps


def classify_episode(ep, gt_tracks):
    """episodeの分類: 実車スイープ疑い/方位乖離/GT行なし(完全誤報)。"""
    diffs = [cdiff(a, fr[j][0]) for (j, a) in ep
             for fr in gt_tracks.values() if j in fr]
    if not diffs:
        return "GT行なし(完全誤報)"
    return "実車スイープ疑い(≤45°)" if min(diffs) <= 45.0 else "方位乖離(>45°)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred_ev, dists = load_preds()
    clips = manifest_clips(pred_ev.keys())

    conf = defaultdict(int)
    n_unnotified = defaultdict(int)
    v1_notified = defaultdict(int)
    v1_to_silent = defaultdict(int)
    esc_lead = []
    n_fire_unattr = 0
    n_trig_false = {"強": 0, "中": 0}      # 車なしクリップの誤発火(clip数)
    ep_cls = {"強": defaultdict(int), "中": defaultdict(int)}  # 帰属不能episode分類
    ep_false_clips = {"強": set(), "中": set()}
    n_carless_clips = 0
    n_nopred_clips = 0

    for clip in clips:
        pv = pred_ev.get(clip, {})
        dv = dists.get(clip, {})
        if not pv:
            n_nopred_clips += 1

        gt_tracks = defaultdict(dict)
        mp = DS / "metadata_dist" / f"{clip}.csv"
        if mp.exists():
            for line in open(mp, encoding="utf-8"):
                g = line.strip().split(",")
                if len(g) == 6 and int(g[1]) == CAR:
                    gt_tracks[int(g[2])][int(g[0])] = (float(g[3]), float(g[5]))

        trig = {"強": dist_triggers(dv, T3), "中": dist_triggers(dv, T2)}
        if not gt_tracks:
            n_carless_clips += 1
            for lv in ("強", "中"):
                n = len(trig[lv]) if lv == "強" else \
                    len(trig["中"]) - len(trig["強"])
                n_trig_false[lv] += int(n > 0)
                # 【Sol検証第2R】GT車なしクリップのトリガもepisode分類に含める
                # （定義上すべて完全誤報。従来はここで漏れて表が非網羅だった）
                for ep in group_episodes(trig[lv]):
                    ep_cls[lv]["GT行なし(完全誤報)"] += 1
                    ep_false_clips[lv].add(clip)
            continue

        best = {tr: None for tr in gt_tracks}
        fired_v1 = {tr: False for tr in gt_tracks}

        def attribute(k, az):
            cands = [(tr, cdiff(az, fr[k][0]))
                     for tr, fr in gt_tracks.items() if k in fr]
            if not cands:
                return None
            tr, dd = min(cands, key=lambda x: x[1])
            return tr if dd <= AZ_MATCH else None

        if pv:
            mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0],
                             np.float64).T
            fires = m12.fire_events(pv, frame_spl_a(mix[0], 24000))
        else:
            fires = []

        for k, c, az in fires:
            if c != CAR:
                continue
            tr = attribute(k, az)
            if tr is None:
                n_fire_unattr += 1
                continue
            fired_v1[tr] = True
            dmin = min((d for j in range(max(0, k - 9), k + 1)
                        for a, d in dv.get(j, [])
                        if cdiff(a, az) <= AZ_MATCH), default=99.0)
            role = role_of(dmin)
            if best[tr] is None or ORDER[role] > ORDER[best[tr]]:
                best[tr] = role

        unattr = {"強": [], "中": []}
        for lv, upgrade in (("中", "中"), ("強", "強")):
            for j, a in trig[lv]:
                tr = attribute(j, a)
                if tr is None:
                    unattr[lv].append((j, a))
                    continue
                if best[tr] is None or ORDER[upgrade] > ORDER[best[tr]]:
                    best[tr] = upgrade
                    if upgrade == "強":
                        gt = gt_tracks[tr]
                        if min(v[1] for v in gt.values()) <= T3:
                            k_cpa = min(gt, key=lambda kk: gt[kk][1])
                            esc_lead.append((k_cpa - j) * 0.1)

        for lv in ("強", "中"):
            for ep in group_episodes(unattr[lv]):
                c = classify_episode(ep, gt_tracks)
                ep_cls[lv][c] += 1
                if c == "GT行なし(完全誤報)":
                    ep_false_clips[lv].add(clip)

        for tr, gt in gt_tracks.items():
            gt_tier = {"強": "重大", "中": "注意", "抑制": "安全"}[
                role_of(min(v[1] for v in gt.values()))]
            if fired_v1[tr]:
                v1_notified[gt_tier] += 1
            if best[tr] is None:
                n_unnotified[gt_tier] += 1
            else:
                conf[(gt_tier, best[tr])] += 1
                if fired_v1[tr] and best[tr] == "抑制":
                    v1_to_silent[gt_tier] += 1

    R = ["# 通知層v3.3（Sol再監査対応版）採点", "",
         f"対象クリップ: {len(clips):,}本（manifest基準・予測ゼロ{n_nopred_clips}本を含む）。",
         f"弱通知=v1ルール。強/中=距離トリガ(2フレーム連続＋方位±{LINK_DEG:.0f}°の"
         "同一物体連続性、±25°帰属)。", "",
         "## 車単位の混同行列（GT tier × 最終出力）＋v1比較",
         "| GT＼出力 | 強(③) | 中(②) | 抑制 | 未通知 | v1なら通知 | **v1通知→v3無音化** |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for t in ("重大", "注意", "安全"):
        row = [conf.get((t, r), 0) for r in ("強", "中", "抑制")]
        n = sum(row) + n_unnotified.get(t, 0)
        R.append(f"| {t} (n={n}) | {row[0]} | {row[1]} | {row[2]} "
                 f"| {n_unnotified.get(t, 0)} | {v1_notified.get(t, 0)} "
                 f"| **{v1_to_silent.get(t, 0)}** |")

    def tot(t):
        return sum(conf.get((t, r), 0) for r in ("強", "中", "抑制")) \
            + n_unnotified.get(t, 0)

    n_sev, n_safe = tot("重大"), tot("安全")
    strong_ok = conf.get(("重大", "強"), 0)
    safe_supp = conf.get(("安全", "抑制"), 0) + n_unnotified.get("安全", 0)
    el = np.array(esc_lead) if esc_lead else np.array([np.nan])
    fs = conf.get(("安全", "強"), 0)
    R += ["",
          f"- GT重大車の役割③(強)到達率: **{strong_ok}/{n_sev} "
          f"({100*strong_ok/max(n_sev,1):.1f}%)**",
          f"- GT安全車の抑制率(未通知含む): **{safe_supp}/{n_safe} "
          f"({100*safe_supp/max(n_safe,1):.1f}%)**",
          f"- GT安全車への誤・強通知: {fs}/{n_safe}",
          f"- **誤・強通知(拡張KPI)**: 車単位{fs}/{n_safe} ＋ 完全誤報episode "
          f"強{ep_cls['強'].get('GT行なし(完全誤報)', 0)}件"
          f"({len(ep_false_clips['強'])}clip) / 中"
          f"{ep_cls['中'].get('GT行なし(完全誤報)', 0)}件"
          f"({len(ep_false_clips['中'])}clip)",
          f"- 安全側の後退(v1通知→v3無音化): 重大 {v1_to_silent.get('重大',0)}台 / "
          f"注意 {v1_to_silent.get('注意',0)}台",
          f"- ③到達リード(GT重大, n={len(esc_lead)}): 中央{np.nanmedian(el):.2f}s / "
          f"p90 {np.nanpercentile(el, 90):.2f}s / ≥0.5s "
          f"{100*np.nanmean(el >= 0.5):.1f}%",
          f"- 帰属不能のv1発火: {n_fire_unattr}件（±25°ゲート超）",
          f"- GT車なしクリップ {n_carless_clips}本での距離トリガ誤発火: "
          f"強{n_trig_false['強']}本 / 中{n_trig_false['中']}本", "",
          "## 帰属不能トリガのepisode分類（Sol条件4）",
          "| レベル | 実車スイープ疑い(≤45°) | 方位乖離(>45°) | GT行なし(完全誤報) |",
          "| --- | --- | --- | --- |"]
    for lv in ("強", "中"):
        R.append(f"| {lv} | {ep_cls[lv].get('実車スイープ疑い(≤45°)', 0)} "
                 f"| {ep_cls[lv].get('方位乖離(>45°)', 0)} "
                 f"| {ep_cls[lv].get('GT行なし(完全誤報)', 0)} |")
    R += ["", "※「GT行なし」はその時間帯に可聴GT車行が無いこと（不可聴車の実在は"
          "否定できない=ラベル宇宙の定義に依存）。",
          "※分母は車(トラック)単位・manifest基準。v3.2以前とは分母が異なる。"]
    (OUT / "notify_v33.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
