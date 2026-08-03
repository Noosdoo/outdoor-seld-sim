# -*- coding: utf-8 -*-
"""通知層v3: 推定距離による3役割のリアルタイム出し分け（役割③の実装）。

v9からの変更点（v9は不変更・本ファイルは採点シミュレータ）:
- 発火タイミングはv1ルールそのまま（9/10・ドリフト15°・音量上昇・不応期5s）
  → リードタイムはv11 valと同一（選別で早さを犠牲にしない=第7回監査の教訓）
- 発火時に役割を判定: 発火窓[k-9..k]の発火方位±25°にある車の**推定距離**の最小値
  dmin → 役割③強(≤1.5m) / 役割②中(≤3.0m) / 抑制(>3.2m、3.0-3.2は緩衝=②扱い)
- 発火後も因果トラッカで追跡し、推定距離≤1.5m到達で役割③へ格上げ

採点（第7回監査の要件を継承: 車単位・全数分母・境界層別）:
- GT対応: 発火方位 ↔ metadata_dist の車トラック方位の最近傍マッチ
- 車(クリップ×GTトラック)ごとに GT tier(GT最小距離) × 最終出力強度 の混同行列
- GT安全車の抑制率（=safe過剰通知の削減）、GT重大車の③到達率と到達時刻(CPA比)

入力: out/predictions_v11sde/val_all.csv（6列）・metadata_dist・v11 foa
出力: out/step12_notify_v11sde/notify_v3_val.md
"""
from __future__ import annotations

import importlib.util
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
PRED = ROOT / "out" / "predictions_v11sde" / "val_all.csv"
OUT = ROOT / "out" / "step12_notify_v11sde"
CAR = 4
AZ_MATCH = 25.0          # 発火→予測距離/GTトラックの方位マッチ幅[deg]
TRACK_JUMP = 30.0        # 因果トラッカの1フレーム最大方位ジャンプ[deg]
T3, T2, SUPP = 1.5, 3.0, 3.2


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def role_of(dmin):
    if dmin <= T3:
        return "強"
    if dmin <= SUPP:      # 3.0-3.2の緩衝帯も②扱い
        return "中"
    return "抑制"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 予測: pred_ev[clip][frame] = [(class, az, el)] / dists[clip][frame] = [(az, dist)]
    pred_ev = defaultdict(lambda: defaultdict(list))
    dists = defaultdict(lambda: defaultdict(list))
    for line in open(PRED, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        clip, k, c = p[0], int(p[1]), int(p[2])
        pred_ev[clip][k].append((c, float(p[3]), float(p[4])))
        if c == CAR:
            dists[clip][k].append((float(p[3]), float(p[5])))

    conf = defaultdict(int)            # (GT tier, 最終強度) -> 台数
    n_unnotified = defaultdict(int)    # GT tier -> 未通知台数
    esc_lead = []                      # GT重大車: ③到達がCPAの何秒前か
    for clip in sorted(pred_ev.keys()):
        # GT車トラック: track -> {frame: (az, dist)}
        gt_tracks = defaultdict(dict)
        for line in open(DS / "metadata_dist" / f"{clip}.csv", encoding="utf-8"):
            g = line.strip().split(",")
            if len(g) == 6 and int(g[1]) == CAR:
                gt_tracks[int(g[2])][int(g[0])] = (float(g[3]), float(g[5]))
        if not gt_tracks:
            continue
        mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
        fires = m12.fire_events(pred_ev[clip], frame_spl_a(mix[0], 24000))

        # 車ごとの最終強度を集計
        best = {tr: None for tr in gt_tracks}          # None=未通知
        for k, c, az in fires:
            if c != CAR:
                continue
            # 発火→GTトラック対応(方位最近傍)
            cands = [(tr, cdiff(az, fr[k][0])) for tr, fr in gt_tracks.items()
                     if k in fr]
            if not cands:
                continue
            tr = min(cands, key=lambda x: x[1])[0]
            # 発火窓の推定距離最小
            dmin = min((d for j in range(max(0, k - 9), k + 1)
                        for a, d in dists[clip].get(j, [])
                        if cdiff(a, az) <= AZ_MATCH), default=99.0)
            role = role_of(dmin)
            order = {"抑制": 0, "中": 1, "強": 2}
            if best[tr] is None or order[role] > order[best[tr]]:
                best[tr] = role

        # 役割③の独立トリガ: 推定距離≤1.5mが2フレーム連続 → 強
        # （追跡不要=至近通過の高速な方位スイープでも取りこぼさない。
        #   帰属は当該フレームのGT方位への最近傍）
        prev_close = False
        for j in range(100):
            close = [(a, d) for a, d in dists[clip].get(j, []) if d <= T3]
            if close and prev_close:
                a = close[0][0]
                cands = [(tr, cdiff(a, fr[j][0]))
                         for tr, fr in gt_tracks.items() if j in fr]
                if cands:
                    tr = min(cands, key=lambda x: x[1])[0]
                    if best[tr] != "強":
                        best[tr] = "強"
                        gt = gt_tracks[tr]
                        if min(v[1] for v in gt.values()) <= T3:
                            k_cpa = min(gt, key=lambda kk: gt[kk][1])
                            esc_lead.append((k_cpa - j) * 0.1)
            prev_close = bool(close)

        for tr, gt in gt_tracks.items():
            gt_tier = {"強": "重大", "中": "注意", "抑制": "安全"}[
                role_of(min(v[1] for v in gt.values()))]
            if best[tr] is None:
                n_unnotified[gt_tier] += 1
            else:
                conf[(gt_tier, best[tr])] += 1

    R = ["# 通知層v3（推定距離による3役割出し分け）val採点", "",
         "発火タイミングはv1と同一（リード維持）。役割のみ推定距離で決定。", "",
         "## 車単位の混同行列（GT tier × 最終出力）",
         "| GT＼出力 | 強(役割③) | 中(役割②) | 抑制 | 未通知 |",
         "| --- | --- | --- | --- | --- |"]
    for t in ("重大", "注意", "安全"):
        row = [conf.get((t, r), 0) for r in ("強", "中", "抑制")]
        n = sum(row) + n_unnotified.get(t, 0)
        R.append(f"| {t} (n={n}) | {row[0]} | {row[1]} | {row[2]} "
                 f"| {n_unnotified.get(t, 0)} |")
    n_sev = sum(conf.get(("重大", r), 0) for r in ("強", "中", "抑制")) \
        + n_unnotified.get("重大", 0)
    n_safe = sum(conf.get(("安全", r), 0) for r in ("強", "中", "抑制")) \
        + n_unnotified.get("安全", 0)
    strong_ok = conf.get(("重大", "強"), 0)
    safe_supp = conf.get(("安全", "抑制"), 0) + n_unnotified.get("安全", 0)
    safe_strong = conf.get(("安全", "強"), 0)
    R += ["",
          f"- **GT重大車の役割③(強)到達率: {strong_ok}/{n_sev} "
          f"({100*strong_ok/max(n_sev,1):.1f}%)**",
          f"- **GT安全車の抑制率(未通知含む): {safe_supp}/{n_safe} "
          f"({100*safe_supp/max(n_safe,1):.1f}%)** ← v1では強弱の区別なく"
          f"90-92%が通知されていた",
          f"- GT安全車への誤・強通知: {safe_strong}/{n_safe} "
          f"({100*safe_strong/max(n_safe,1):.1f}%)",
          f"- 重大車の③到達タイミング: CPAの中央{np.median(esc_lead):.2f}s前 "
          f"(n={len(esc_lead)}, p10 {np.percentile(esc_lead, 10):.2f}s)"
          if esc_lead else "- ③到達タイミング: データなし"]
    (OUT / "notify_v3_val.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
