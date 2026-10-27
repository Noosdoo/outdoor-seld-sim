# -*- coding: utf-8 -*-
"""⑦の着手前診断 — v4.1 が重大イベントを逃すとき、どのゲートで落ちているか（2026-08-30）。

「作り替える前に測る」（準備の軌跡§6の教訓）。v4.2（方位主導の判定経路）の設計を
数字で決めるため、現行 v4.1 (fires_cpa) の見逃し1件ずつに、落ちた場所のラベルを付ける:

  A 系列なし    … 知覚/追跡: イベント窓内でこのGT車へ帰属できる予測フレームが窓長未満
  B0 予測不能   … 帰属フレームはあるが、連続窓が埋まらず d_cpa/t_cpa が一度も出ない
  B d_cpa膨張   … 予測最接近 dc が中しきい値(CPA_MID=2.0m)を一度も下回らない
  C t_cpa失敗   … dc は下回るが、その場で到達 tc がしきい値(TTC_CAUTION=4.0s)超
  D confirm不足 … 条件成立が CONFIRM_CPA(4) フレーム連続しない
  E 成立するが未帰属 … 条件は連続成立（発火はしたはずだが、窓の外/他車に消費/帰属外れ）

あわせて「方位主導経路」（|dθ/dt| ≤ TH かつ 接近中 が連続）の実現可能性を測る:
  - 見逃した重大イベントのうち、この条件なら拾えた件数（THと方位窓のグリッド）
  - 安全イベントで、この条件が誤って立つ件数（安全抑制を壊さないか）

⚠️ ここでするのは設計の当たり付けだけ。しきい値の**選定**は既存valでは行わない
（v4.1選定で使用済み。3回目は過適合）。選定はチューニング専用の新valで行う。

使い方:
  python scripts/_notify_v42_diag.py [pred_val_all.csv] [metadata_distディレクトリ] [出力dir]
  （省略時: predictions_v12_w3/val_all.csv / dataset_outdoor_siren_v12/metadata_dist /
    out/notify_v42_diag）
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "nv4", ROOT / "scripts" / "step12_notify_v4_ttc.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

FPS = v4.FPS
NFR = 100                              # 10秒クリップ × 10fps
WIN_PRE, WIN_POST = 1.0, 1.0           # _notify_v4_eval と同じイベント窓
ATTR_MATCH = 60.0                      # GT車への帰属幅（オラクル実験と同じ）
V_CLOSE = 0.3                          # [m/s] 接近判定に使う頑健傾きのしきい値（診断用）
TH_GRID = (0.05, 0.10, 0.20, 0.30)     # [rad/s] |dθ/dt| のグリッド
WB_GRID = (5, 9)                       # 方位窓のグリッド（5=現行, 9=長窓）
DN_GRID = (8.0, 15.0, 30.0)            # [m] 方位主導経路の距離ゲート。
                                       # dθ/dt は d² に反比例するので、固定THだと
                                       # 「遠くをまっすぐ通る安全車」も誤成立する。
                                       # 近距離に限定してどこまで割り切れるかを見る
CONFIRM_C = v4.CONFIRM_CPA             # 方位主導経路の確認フレーム（現行cpaと同じ4）


def gt_tracks(meta_dir: Path, clip):
    """[(cls, trk, {frame:(az,dist)})] を連続ラン単位で返す（1ラン=1イベント）。"""
    f = meta_dir / f"{clip}.csv"
    if not f.exists():
        return []
    per = defaultdict(dict)
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) < 6:
            continue
        c = int(g[1])
        if c not in v4.DIST_CLASSES:
            continue
        per[(c, int(g[2]))][int(g[0])] = (float(g[3]), float(g[5]))
    out = []
    for (c, t), fr in per.items():
        js = sorted(fr)
        run = [js[0]]
        for j in js[1:]:
            if j == run[-1] + 1:
                run.append(j)
            else:
                out.append((c, t, {k: fr[k] for k in run}))
                run = [j]
        out.append((c, t, {k: fr[k] for k in run}))
    return out


def mk_event(cls, trk, fr):
    js = sorted(fr)
    d = np.array([fr[j][1] for j in js])
    i = int(np.argmin(d))
    dmin = float(d[i])
    tier = ("critical" if dmin <= v4.T3
            else "caution" if dmin <= v4.SUPP else "safe")
    return dict(cls=cls, trk=trk, fr=fr, f0=js[0], f1=js[-1],
                cpa=js[i], dmin=dmin, tier=tier)


def theil_sen(d_at, j, win=v4.VEL_WIN):
    """ペアごとの傾きの中央値[m/s]。外れ値1つで壊れない接近判定のために使う。"""
    ks = list(range(j - win + 1, j + 1))
    if any(d_at.get(k) is None for k in ks):
        return None
    sl = [(d_at[b] - d_at[a]) / ((b - a) / FPS)
          for i, a in enumerate(ks) for b in ks[i + 1:]]
    return float(np.median(sl))


def max_run(flags_at, js):
    """js（整数フレーム列）上で、連続フレームかつ True の最長ラン長。"""
    best = run = 0
    prev = None
    for j in js:
        if flags_at.get(j) and prev is not None and j == prev + 1:
            run += 1
        elif flags_at.get(j):
            run = 1
        else:
            run = 0
        best = max(best, run)
        prev = j
    return best


def main() -> int:
    pred_path = (Path(sys.argv[1]) if len(sys.argv) > 1
                 else ROOT / "out/predictions_v12_w3/val_all.csv")
    meta_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
                else ROOT / "out/dataset_outdoor_siren_v12/metadata_dist")
    outdir = (Path(sys.argv[3]) if len(sys.argv) > 3
              else ROOT / "out/notify_v42_diag")
    outdir.mkdir(parents=True, exist_ok=True)

    pred = v4.load_pred(pred_path)
    res = v4.run_rule(pred, "cpa")

    # ---- 1) v4.1 の再現採点（_notify_v4_eval と同じ窓・同じ貪欲マッチ）----
    #      加えて「強到達」（重大イベントに 強tier の発火が付いたか）を別に数える。
    #      確定評価の至近到達73.0%は**役割③=強**の到達率なので、⑦の標的はこちら。
    stat = defaultdict(lambda: [0, 0])
    n_strong_hit = 0
    leads, missed_crit, safe_events = [], [], []
    missed_strong = []                 # (clip, ev, 中では鳴ったか)
    n_events = 0
    for clip in sorted(pred):
        events = [mk_event(c, t, fr) for c, t, fr in gt_tracks(meta_dir, clip)]
        n_events += len(events)
        fires = [(f[0], f[1], f[2], f[3], cls)
                 for cls, eps in res.get(clip, {}).items() for f in eps]
        used = [False] * len(fires)
        used_s = [False] * len(fires)
        for ev in events:
            a, b = ev["f0"] - WIN_PRE * FPS, ev["cpa"] + WIN_POST * FPS
            hit = None
            for i, fr_ in enumerate(fires):
                if not used[i] and fr_[4] == ev["cls"] and a <= fr_[0] <= b:
                    hit = i
                    break
            if ev["tier"] == "safe":
                stat["safe"][1] += 1
                stat["safe"][0] += int(hit is None)
                if hit is not None:
                    used[hit] = True
                safe_events.append((clip, ev))
                continue
            stat[ev["tier"]][1] += 1
            if hit is not None:
                used[hit] = True
                stat[ev["tier"]][0] += 1
                leads.append((ev["cpa"] - fires[hit][0]) / FPS)
            elif ev["tier"] == "critical":
                missed_crit.append((clip, ev))
            if ev["tier"] == "critical":
                hs = None
                for i, fr_ in enumerate(fires):
                    if (not used_s[i] and fr_[4] == ev["cls"] and fr_[2] == "強"
                            and a <= fr_[0] <= b):
                        hs = i
                        break
                if hs is not None:
                    used_s[hs] = True
                    n_strong_hit += 1
                else:
                    missed_strong.append((clip, ev, hit is not None))

    # ---- 2) 見逃した重大イベントのゲート帰属 ----
    attr = Counter()
    attr_by_band = defaultdict(Counter)      # GT最小距離の帯 × 帰属
    dc_ratio = []                            # dc最小値 / GT dmin（膨張の度合い）
    series_cache = {}

    def series(clip, cls):
        key = (clip, cls)
        if key not in series_cache:
            series_cache[key] = v4.track_series(pred[clip], cls, NFR)
        return series_cache[key]

    def per_frame(clip, ev):
        """イベント窓内の per-frame 判定材料を、本番と同じ系列の上で計算する。"""
        d_at, az_at = series(clip, ev["cls"])
        hi = min(NFR - 1, int(ev["cpa"] + WIN_POST * FPS))
        js = list(range(ev["f0"], hi + 1))
        usable = [j for j in js if j in d_at and j in ev["fr"]
                  and v4.cdiff(az_at[j], ev["fr"][j][0]) <= ATTR_MATCH]
        vals = {}
        for j in usable:
            v = v4.closing_speed(d_at, j)
            adot = v4.azimuth_rate(az_at, j)
            dc, tc = v4.cpa_of(d_at[j], None if v is None else -v, adot)
            vals[j] = (dc, tc)
        return d_at, az_at, js, usable, vals

    for clip, ev in missed_crit:
        d_at, az_at, js, usable, vals = per_frame(clip, ev)
        band = ("≤0.5m" if ev["dmin"] <= 0.5
                else "0.5–1.0m" if ev["dmin"] <= 1.0 else "1.0–1.5m")
        if len(usable) < v4.VEL_WIN:
            lab = "A 系列なし"
        else:
            dcs = [vals[j][0] for j in usable if vals[j][0] is not None]
            if not dcs:
                lab = "B0 予測不能(窓が埋まらない)"
            else:
                dc_ratio.append(min(dcs) / max(ev["dmin"], 0.05))
                ok_dc = {j: (vals[j][0] is not None and vals[j][0] <= v4.CPA_MID_M)
                         for j in usable}
                ok_both = {j: (ok_dc[j] and vals[j][1] <= v4.TTC_CAUTION)
                           for j in usable}
                if not any(ok_dc.values()):
                    lab = "B d_cpa膨張"
                elif not any(ok_both.values()):
                    lab = "C t_cpa失敗"
                elif max_run(ok_both, usable) < v4.CONFIRM_CPA:
                    lab = "D confirm不足"
                else:
                    lab = "E 成立するが未帰属"
        attr[lab] += 1
        attr_by_band[band][lab] += 1

    # ---- 2b) 「強」に届かなかった重大イベントのゲート帰属（⑦の本丸）----
    attr_s = Counter()
    n_mid_only = sum(1 for _c, _e, had in missed_strong if had)
    for clip, ev, _had in missed_strong:
        d_at, az_at, js, usable, vals = per_frame(clip, ev)
        if len(usable) < v4.VEL_WIN:
            lab = "A 系列なし"
        else:
            dcs = [vals[j][0] for j in usable if vals[j][0] is not None]
            ok_dc = {j: (vals[j][0] is not None
                         and vals[j][0] <= v4.CPA_STRONG_M) for j in usable}
            ok_both = {j: (ok_dc[j] and vals[j][1] <= v4.TTC_WARN) for j in usable}
            ins = {j: d_at[j] <= v4.T3 for j in usable}    # 距離保険（強側）
            if (max_run(ok_both, usable) >= v4.CONFIRM_CPA
                    or max_run(ins, usable) >= v4.CONFIRM):
                lab = "E 成立するが未帰属"
            elif not dcs:
                lab = "B0 予測不能(窓が埋まらない)"
            elif not any(ok_dc.values()):
                lab = "B d_cpa>1.0m"
            elif not any(ok_both.values()):
                lab = "C t_cpa>2.5s"
            else:
                lab = "D confirm不足"
        attr_s[lab] += 1

    # ---- 3) 方位主導経路（定方位接近）の実現可能性 ----
    #   条件: |dθ/dt(長窓)| ≤ TH かつ Theil–Sen傾き < −V_CLOSE かつ d ≤ DN
    #   が CONFIRM_C フレーム連続 → この経路だけで「強候補」になったとみなす。
    #   救済の対象は「強に届かなかった重大イベント」。
    def route_c_hits(clip, ev, wb, th, dn):
        d_at, az_at = series(clip, ev["cls"])
        hi = min(NFR - 1, int(ev["cpa"] + WIN_POST * FPS))
        usable = [j for j in range(ev["f0"], hi + 1)
                  if j in d_at and j in ev["fr"]
                  and v4.cdiff(az_at[j], ev["fr"][j][0]) <= ATTR_MATCH]
        flags = {}
        for j in usable:
            adot = v4.azimuth_rate(az_at, j, win=wb)
            ts = theil_sen(d_at, j)
            flags[j] = (adot is not None and abs(adot) <= th
                        and ts is not None and ts < -V_CLOSE
                        and d_at[j] <= dn)
        return max_run(flags, usable) >= CONFIRM_C

    feas = {}
    for wb in WB_GRID:
        for th in TH_GRID:
            for dn in DN_GRID:
                catch = sum(route_c_hits(c, e, wb, th, dn)
                            for c, e, _h in missed_strong)
                false_ = sum(route_c_hits(c, e, wb, th, dn)
                             for c, e in safe_events)
                feas[(wb, th, dn)] = (catch, false_)

    # ---- 出力 ----
    n_crit, n_caut = stat["critical"][1], stat["caution"][1]
    n_safe = stat["safe"][1]
    L = np.array(leads) if leads else np.array([0.0])
    R = [f"# ⑦着手前診断 — v4.1の見逃しはどのゲートで落ちているか",
         f"", f"- 予測: `{pred_path}`", f"- GT: `{meta_dir}`",
         f"- クリップ {len(pred):,} / GTイベント {n_events:,}"
         f"（重大 {n_crit:,} / 注意 {n_caut:,} / 安全 {n_safe:,}）", "",
         "## 1. v4.1 (fires_cpa) の再現採点", "",
         f"- 至近到達（重大・中でも可）: {100*stat['critical'][0]/max(n_crit,1):.1f}%"
         f"（{stat['critical'][0]:,}/{n_crit:,}）",
         f"- **強到達（重大・強のみ＝役割③相当）: "
         f"{100*n_strong_hit/max(n_crit,1):.1f}%（{n_strong_hit:,}/{n_crit:,}）**",
         f"- 注意到達: {100*stat['caution'][0]/max(n_caut,1):.1f}%",
         f"- 安全抑制: {100*stat['safe'][0]/max(n_safe,1):.1f}%",
         f"- リード中央 {np.median(L):.2f}s / ≥2.5s {100*np.mean(L>=2.5):.1f}%", "",
         f"## 2. 完全な見逃し（中も強も無し）{len(missed_crit):,} 件のゲート帰属", ""]
    for lab, n in attr.most_common():
        R.append(f"- {lab}: **{n}件** ({100*n/max(len(missed_crit),1):.1f}%)")
    R += ["", "GT最小距離の帯別（≤0.5mほど「対向・真正面」寄り）:", "",
          "| 帯 | " + " | ".join(k for k, _ in attr.most_common()) + " |",
          "| --- |" + " --- |" * len(attr)]
    for band in ("≤0.5m", "0.5–1.0m", "1.0–1.5m"):
        row = attr_by_band.get(band, Counter())
        R.append(f"| {band} | " + " | ".join(str(row.get(k, 0))
                 for k, _ in attr.most_common()) + " |")
    if dc_ratio:
        q = np.percentile(dc_ratio, [25, 50, 75])
        R += ["", f"- 予測最接近の膨張度 dc_min/GT真値: "
              f"中央 {q[1]:.1f}倍（四分位 {q[0]:.1f}〜{q[2]:.1f}倍, n={len(dc_ratio)}）"]
    R += ["", f"## 2b. 強に届かなかった重大イベント {len(missed_strong):,} 件"
          f"（うち中では鳴った {n_mid_only:,} 件）のゲート帰属", "",
          "ゲートは強側（dc≤1.0m ∧ tc≤2.5s ×4連続、保険 d≤1.5m ×2連続）。", ""]
    for lab, n in attr_s.most_common():
        R.append(f"- {lab}: **{n}件** ({100*n/max(len(missed_strong),1):.1f}%)")
    R += ["", "## 3. 方位主導経路（|dθ/dt|≤TH ∧ 接近中 ∧ d≤DN が4フレーム連続）の当たり",
          "",
          "しきい値の**選定はしない**（それは新valの仕事）。桁の当たりだけ見る。",
          "救済対象は 2b（強に届かなかった重大）。", "",
          "| 方位窓 | TH[rad/s] | DN[m] | 強未達の救済 | 安全イベントでの誤成立 |",
          "| --- | --- | --- | --- | --- |"]
    for (wb, th, dn), (catch, false_) in feas.items():
        R.append(f"| {wb}fr | {th:.2f} | {dn:.0f} | {catch}/{len(missed_strong)} "
                 f"({100*catch/max(len(missed_strong),1):.0f}%) | "
                 f"{false_}/{n_safe} ({100*false_/max(n_safe,1):.1f}%) |")
    out_md = outdir / "diag.md"
    out_md.write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
