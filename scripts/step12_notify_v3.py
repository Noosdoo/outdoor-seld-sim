# -*- coding: utf-8 -*-
"""通知層v3.2: 推定距離による3役割のリアルタイム出し分け（内部監査対応版）。

v3.1からの変更（内部監査=Opus 2026-08-03 の指摘対応）:
- 帰属に±25°ゲートを実装（自白と実装の不一致を解消。ゲート超は帰属不能として計上）
- 中トリガを追加: 推定距離≤3.0mが2フレーム連続→役割②(中)（3段構成を機能させる）
- 強/中トリガは複数候補から**最小距離**を採用（並び順依存を解消）
- GT車なしクリップも採点対象にし、トリガ誤発火を誤報として計上（分母を閉じる）
- **v1比較列**を追加: v1なら通知されていた車の数と、v3で無音化した数を明示
  （安全側の後退を隠さない）
- 役割③到達リードの分布(p50/p90/≥0.5s/≥2.0s)を報告
- 【2026-08-06 第8回監査NOTIFY-3対応】車ありクリップ内の帰属不能トリガ
  （±25°超）を件数計上（旧v3.2はno-opで消えていた=誤報の過小評価穴）。
  規則(T3/T2/SUPP/2フレーム/±25°)は不変=fold3追試向け凍結対象。

発火タイミング: 役割②の弱通知はv1ルールのまま。強(③)/中(②格上げ)は距離トリガ
（独立トリガ。v1発火ゲートは通らない=この点は仕様として明記）。
入力: out/predictions_v11sde/val_all.csv（7列 or 旧6列）・metadata_dist・v11 foa
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
# 引数で予測CSVと出力先を指定可（既定=run1のまま。新しいランは新フォルダに出す=上書き禁止運用）
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "out" / "predictions_v11sde" / "val_all.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "out" / "step12_notify_v11sde"
CAR = 4
AZ_MATCH = 25.0          # 帰属・距離窓の方位マッチ幅[deg]
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
    """7列 [clip,frame,class,track,az,el,dist] / 旧6列 の両対応。"""
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
    return pred_ev, dists


def dist_triggers(dseq, thresh):
    """推定距離≤threshが2フレーム連続した (frame, az) を返す（最小距離の予測を採用）。"""
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
    pred_ev, dists = load_preds()

    conf = defaultdict(int)
    n_unnotified = defaultdict(int)
    v1_notified = defaultdict(int)         # GT tier -> v1発火が帰属した台数
    v1_to_silent = defaultdict(int)        # GT tier -> v1通知→v3抑制の台数
    esc_lead = []
    n_fire_unattr = 0                      # 帰属不能のv1発火
    n_trig_false = {"強": 0, "中": 0}      # 車なしクリップの距離トリガ誤発火(本数)
    n_trig_unattr = {"強": 0, "中": 0}     # 車ありクリップ内の±25°超トリガ(件数)
    n_carless_clips = 0

    for clip in sorted(pred_ev.keys()):
        gt_tracks = defaultdict(dict)
        mp = DS / "metadata_dist" / f"{clip}.csv"
        if mp.exists():
            for line in open(mp, encoding="utf-8"):
                g = line.strip().split(",")
                if len(g) == 6 and int(g[1]) == CAR:
                    gt_tracks[int(g[2])][int(g[0])] = (float(g[3]), float(g[5]))

        # 距離トリガ（強/中）。GT車なしクリップでも誤報として数える
        trig = {"強": dist_triggers(dists[clip], T3),
                "中": dist_triggers(dists[clip], T2)}
        if not gt_tracks:
            n_carless_clips += 1
            for lv in ("強", "中"):
                # 強は中の部分集合なので中側は強を除いた件数だけ数える
                n = len(trig[lv]) if lv == "強" else \
                    len(trig["中"]) - len(trig["強"])
                n_trig_false[lv] += int(n > 0)
            continue

        mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
        fires = m12.fire_events(pred_ev[clip], frame_spl_a(mix[0], 24000))

        best = {tr: None for tr in gt_tracks}
        fired_v1 = {tr: False for tr in gt_tracks}

        def attribute(k, az):
            cands = [(tr, cdiff(az, fr[k][0]))
                     for tr, fr in gt_tracks.items() if k in fr]
            if not cands:
                return None
            tr, dd = min(cands, key=lambda x: x[1])
            return tr if dd <= AZ_MATCH else None   # ±25°ゲート(監査対応)

        for k, c, az in fires:
            if c != CAR:
                continue
            tr = attribute(k, az)
            if tr is None:
                n_fire_unattr += 1
                continue
            fired_v1[tr] = True
            dmin = min((d for j in range(max(0, k - 9), k + 1)
                        for a, d in dists[clip].get(j, [])
                        if cdiff(a, az) <= AZ_MATCH), default=99.0)
            role = role_of(dmin)
            if best[tr] is None or ORDER[role] > ORDER[best[tr]]:
                best[tr] = role

        for lv, upgrade in (("中", "中"), ("強", "強")):
            for j, a in trig[lv]:
                tr = attribute(j, a)
                if tr is None:
                    n_trig_unattr[lv] += 1
                    continue
                if best[tr] is None or ORDER[upgrade] > ORDER[best[tr]]:
                    best[tr] = upgrade
                    if upgrade == "強":
                        gt = gt_tracks[tr]
                        if min(v[1] for v in gt.values()) <= T3:
                            k_cpa = min(gt, key=lambda kk: gt[kk][1])
                            esc_lead.append((k_cpa - j) * 0.1)

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

    R = ["# 通知層v3.2（監査対応版）val採点", "",
         "弱通知の発火=v1ルール。強/中=距離トリガ(独立、2フレーム連続、±25°帰属)。", "",
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
    R += ["",
          f"- GT重大車の役割③(強)到達率: **{strong_ok}/{n_sev} "
          f"({100*strong_ok/max(n_sev,1):.1f}%)**",
          f"- GT安全車の抑制率(未通知含む): **{safe_supp}/{n_safe} "
          f"({100*safe_supp/max(n_safe,1):.1f}%)**",
          f"- GT安全車への誤・強通知: {conf.get(('安全','強'),0)}/{n_safe}",
          f"- **安全側の後退(v1通知→v3無音化)**: 重大 {v1_to_silent.get('重大',0)}台 / "
          f"注意 {v1_to_silent.get('注意',0)}台（この数を隠さないことが本表の目的）",
          f"- ③到達リード(GT重大, n={len(esc_lead)}): 中央{np.nanmedian(el):.2f}s / "
          f"p90 {np.nanpercentile(el, 90):.2f}s / ≥0.5s "
          f"{100*np.nanmean(el >= 0.5):.1f}% / ≥2.0s {100*np.nanmean(el >= 2.0):.1f}%",
          f"- 帰属不能のv1発火: {n_fire_unattr}件（±25°ゲート超）",
          f"- GT車なしクリップ {n_carless_clips}本での距離トリガ誤発火: "
          f"強{n_trig_false['強']}本 / 中{n_trig_false['中']}本",
          f"- 車ありクリップ内の帰属不能トリガ(±25°超・件数・強は中と重複計上): "
          f"強{n_trig_unattr['強']}件 / 中{n_trig_unattr['中']}件"
          "（誤報か至近スイープの実車かは個別監査対象=第8回監査NOTIFY-3）",
          "", "※分母は車(トラック)単位。v9系の「クリップ単位97.3%」とは別物。",
          "※強/中は距離トリガ独立発火＝v1の発火ゲートを通らない（仕様として明記）。"]
    (OUT / "notify_v3_val.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
