# -*- coding: utf-8 -*-
"""Step 20 v2: 実録採点の全面改修版（第11回監査 高5・高6・中7への対応。
再監査指摘（2026-08-13 19:02）反映済み: 負例マスク・v1警告定数の正確な踏襲・
中通知(≤3.0m)・比較キーへのclip_id追加・クリップ単位paired bootstrap）。

旧 step20_realsmoke_score.py の欠陥と本版の修正:
  A) 全区間走査 — 旧は通知関数が先頭100フレーム(10秒)固定。本版はクリップ実長で走査。
  B) event_id単位のイベント窓 — 各イベントの[t_start−1s, t_cpa＋1s]窓へ発火を
     貪欲割当（1発火は最大1イベント）。さらに**負例窓では、注釈済みイベント窓に
     入る発火をマスク**して誤警告に二重計上しない（未注釈イベントは注釈が規約）。
  C) 現行通知規則 —
     ・距離クラス(car/kick/bike): v3.4（距離≤閾が2フレーム連続＋前フレーム候補と
       方位連結±LINK_DEG）。強=≤1.5m、中=≤3.0m の2段をエピソード単位で判定
       （定数はstep12_notify_v33と同一。可変長に一般化）。
     ・警告音クラス(siren等): ルールv1と同一定数（WARN_CONFIRM=3フレーム連続・
       不応期5s・方向別±45°。step12_notify_v9の事前決定値を踏襲）。
  D) 事前登録の対応あり統計 — McNemar厳密検定・**クリップ単位**paired bootstrap・
     Poisson厳密区間（実録ハンドブック§9-3）。

入力:
  --pred   予測CSV。6/7列 [stem,frame,class(,track),az,el,dist]（v12形式）。
           5列（距離なし旧形式）は距離クラス採点をスキップし警告クラスのみ。
  --ann    注釈CSV。列: clip_id,event_id,trial,class,quadrant,t_start,t_cpa[,...]
           class=none 行は負例窓。
  --pred-b 対応あり比較のもう1系統。通知有無=McNemar厳密検定 /
           リード差=クリップ単位paired bootstrap。
  --out    出力md（既定 out/realsmoke/score_v2.md）
  --link-deg 方位連結幅。既定60（=v3.4。停止規則により変更しない）

クラス番号（v12・8クラス）: 0=siren 1=horn 2=backup_beep 3=bike_bell 4=car_drive
                            5=crossing 6=kick 7=bike
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FPS = 10.0                       # 0.1sフレーム
T3, T2, SUPP = 1.5, 3.0, 3.2     # v3.3/v3.4と同一（step12_notify_v33）
# 警告音クラス: ルールv1の事前決定値（step12_notify_v9 L110-118）をそのまま使う
WARN_CONFIRM = 3                 # 0.3s 連続
REFRACT_FRAMES = 50              # 不応期5s
REFRACT_DEG = 45.0               # 不応期は「クラス×方向」単位（±45°）
EP_GAP = 10                      # 距離トリガ発火列→エピソード化のフレームギャップ
WIN_PRE, WIN_POST = 1.0, 1.0     # イベント窓 [t_start−1s, t_cpa+1s]（注釈±1s精度）

CLS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
           "car_drive": 4, "crossing": 5, "kick": 6, "bike": 7}
DIST_CLS = {4, 6, 7}             # 距離トリガで採点するクラス


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def quadrant_of(az_deg: float) -> str:
    a = (az_deg + 180.0) % 360.0 - 180.0
    if abs(a) <= 45:
        return "F"
    if abs(a) > 135:
        return "B"
    return "L" if a > 0 else "R"


# ---------------------------------------------------------------- 予測の読込
def load_pred(path: Path):
    """clip -> frame -> [(cls, az, dist|None)]。5/6/7列を自動判別。"""
    out = defaultdict(lambda: defaultdict(list))
    has_dist = True
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:      # stem,frame,class,track,az,el,dist
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[4]), float(p[6])
        elif len(p) == 6:    # stem,frame,class,az,el,dist
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[3]), float(p[5])
        elif len(p) == 5:    # 旧: stem,frame,class,az,el（距離なし）
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[3]), None
            has_dist = False
        else:
            continue
        out[clip][k].append((c, az, d))
    return dict(out), has_dist


# ------------------------------------------------------------ 通知発火（可変長）
def dist_triggers_var(dseq, thresh, nframes, link_deg):
    """v3.4距離トリガの可変長版（規則はstep12_notify_v33.dist_triggersと同一。
    旧実装のrange(100)固定をnframes引数に一般化し、採用候補の距離も返す）。"""
    hits, prev = [], []
    for j in range(nframes):
        close = [(a, d) for a, d in dseq.get(j, []) if d is not None and d <= thresh]
        if close and prev:
            linked = [(a, d) for a, d in close
                      if any(cdiff(a, pa) <= link_deg for pa, _ in prev)]
            if linked:
                a, d = min(linked, key=lambda x: x[1])
                hits.append((j, a, d))
        prev = close
    return hits


def episodes_of(hits, link_deg):
    """発火列→エピソード（gap>EP_GAPまたは方位乖離で分割）。
    返り値: [(t_first_frame, az_first, tier)]。tier=エピソード内最小距離で
    強(≤T3)/中(≤T2)を判定（step12_notify_v33.role_ofと同じ割当）。"""
    eps = []
    for j, a, d in hits:
        if eps and j - eps[-1][-1][0] <= EP_GAP and cdiff(a, eps[-1][-1][1]) <= link_deg:
            eps[-1].append((j, a, d))
        else:
            eps.append([(j, a, d)])
    out = []
    for ep in eps:
        dmin = min(d for _, _, d in ep)
        tier = "強" if dmin <= T3 else "中"
        out.append((ep[0][0], ep[0][1], tier))
    return out


def warn_fires(frames_of_cls, az_of, nframes):
    """警告音クラス: v1定数（3フレーム連続・不応期5s・方向±45°）で発火。"""
    fires, last = [], []
    for k in range(nframes):
        if all((k - i) in frames_of_cls for i in range(WARN_CONFIRM)):
            az = az_of[k]
            if any(k - kp < REFRACT_FRAMES and cdiff(az, ap) <= REFRACT_DEG
                   for kp, ap in last):
                continue
            fires.append((k, az))
            last.append((k, az))
    return fires


def fires_for_clip(pred_clip, nframes, link_deg):
    """clsごとの発火 [(t_sec, az, tier)]。距離クラス=強/中エピソード、
    警告クラス=tier「警告」。"""
    by_cls_frames = defaultdict(set)
    az_at = defaultdict(dict)
    dseq = defaultdict(lambda: defaultdict(list))
    for k, evs in pred_clip.items():
        for c, az, d in evs:
            by_cls_frames[c].add(k)
            az_at[c][k] = az
            if c in DIST_CLS:
                dseq[c][k].append((az, d))
    out = defaultdict(list)
    for c in sorted(by_cls_frames):
        if c in DIST_CLS:
            hits = dist_triggers_var(dseq[c], T2, nframes, link_deg)
            for j, a, tier in episodes_of(hits, link_deg):
                out[c].append((j / FPS, a, tier))
        else:
            for k, a in warn_fires(by_cls_frames[c], az_at[c], nframes):
                out[c].append((k / FPS, a, "警告"))
    return dict(out)


# ------------------------------------------------------------------ 採点
def evaluate(rows, pred, link_deg, has_dist):
    """注釈行を採点。返り値: (events, negatives)
    events: dict(clip,trial,event_id,class,notified(強/警告),mid_reach,lead,quad_ok)
    negatives: dict(n_false, exposure_s)。負例窓では注釈済みイベント窓
    [t_start−1, t_cpa+1] に入る発火をマスクし、露出も重なり分を控除する。"""
    need = defaultdict(float)
    for r in rows:
        need[r["clip_id"]] = max(need[r["clip_id"]], float(r["t_cpa"]) + WIN_POST + 1.0)
    fires_by_clip = {}
    for clip in {r["clip_id"] for r in rows}:
        pc = pred.get(clip, {})
        nf = int(max(need[clip] * FPS,
                     (max(pc.keys()) + 1) if pc else 0)) + 1
        fires_by_clip[clip] = fires_for_clip(pc, nf, link_deg)

    # クリップごとの注釈済みイベント窓（負例マスク用）
    pos_windows = defaultdict(list)
    for r in rows:
        if r["class"].strip() != "none":
            pos_windows[r["clip_id"]].append(
                (float(r["t_start"]) - WIN_PRE, float(r["t_cpa"]) + WIN_POST))

    def masked(clip, t):
        return any(a <= t <= b for a, b in pos_windows[clip])

    def overlap(t0, t1, wins):
        """[t0,t1]と窓集合の重なり秒数（窓同士の重複は逐次マージで近似排除）。"""
        segs = sorted((max(t0, a), min(t1, b)) for a, b in wins if b > t0 and a < t1)
        total, cur = 0.0, None
        for a, b in segs:
            if cur is None or a > cur[1]:
                if cur:
                    total += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur:
            total += cur[1] - cur[0]
        return total

    events, n_false, exposure_s = [], 0, 0.0
    by_key = defaultdict(list)
    for r in rows:
        clip = r["clip_id"]
        if r["class"].strip() == "none":
            t0, t1 = float(r["t_start"]), float(r["t_cpa"])
            exposure_s += (t1 - t0) - overlap(t0, t1, pos_windows[clip])
            fires_all = [t for fl in fires_by_clip[clip].values()
                         for t, _, _ in fl]
            n_false += sum(1 for t in fires_all
                           if t0 <= t <= t1 and not masked(clip, t))
            continue
        by_key[(clip, r["class"].strip())].append(r)

    for (clip, cls), evrows in sorted(by_key.items()):
        ci = CLS_IDX[cls]
        if ci in DIST_CLS and not has_dist:
            for r in evrows:
                events.append({"clip": clip, "trial": r["trial"],
                               "event_id": r.get("event_id", "1"), "class": cls,
                               "notified": None, "mid_reach": None,
                               "lead": None, "quad_ok": None})
            continue
        fl = fires_by_clip[clip].get(ci, [])
        # 至近警告の対象: 距離クラス=強のみ / 警告クラス=警告
        cand = sorted([(t, az) for t, az, tier in fl
                       if tier == ("強" if ci in DIST_CLS else "警告")])
        used = [False] * len(cand)
        for r in sorted(evrows, key=lambda x: float(x["t_cpa"])):
            t0 = float(r["t_start"]) - WIN_PRE
            t1 = float(r["t_cpa"]) + WIN_POST
            pick = None
            for i, (t, az) in enumerate(cand):
                if not used[i] and t0 <= t <= t1:
                    pick = (i, t, az)
                    break
            mid_reach = any(t0 <= t <= t1 for t, _, tier in fl
                            if tier in ("強", "中")) if ci in DIST_CLS else None
            base = {"clip": clip, "trial": r["trial"],
                    "event_id": r.get("event_id", "1"), "class": cls,
                    "mid_reach": mid_reach}
            if pick is not None:
                used[pick[0]] = True
                events.append({**base, "notified": True,
                               "lead": round(float(r["t_cpa"]) - pick[1], 2),
                               "quad_ok": quadrant_of(pick[2]) == r["quadrant"].strip()})
            else:
                events.append({**base, "notified": False,
                               "lead": None, "quad_ok": None})
    return events, {"n_false": n_false, "exposure_s": exposure_s}


# ------------------------------------------------------------------ 統計
def mcnemar_exact(b: int, c: int) -> float:
    """対応あり2値の不一致対(b,c)に対する厳密両側p（二項検定）。"""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))


def poisson_rate_ci(k: int, hours: float, alpha=0.05):
    """件数kと露出時間[h]から誤警告率[回/h]の厳密CI（chi2法）。"""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k) / hours
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / hours
    return lo, hi


def paired_bootstrap_median_diff_by_clip(diffs_by_clip, n_boot=10000, seed=0):
    """クリップ単位のpaired bootstrap（事前規定=ハンドブック§9-3）。
    diffs_by_clip: {clip: [lead_A−lead_B, ...]}。クリップを復元抽出し、
    採択クリップのイベント差をプールした中央値の分布から95%CIを出す。"""
    clips = sorted(diffs_by_clip)
    if not clips:
        return None
    rng = np.random.default_rng(seed)
    all_d = [d for c in clips for d in diffs_by_clip[c]]
    meds = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(clips), len(clips))
        pool = [d for j in pick for d in diffs_by_clip[clips[j]]]
        meds[i] = np.median(pool)
    return (float(np.median(all_d)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


# ------------------------------------------------------------------ main
def main() -> int:
    pred_path = Path(_arg("--pred"))
    ann_path = Path(_arg("--ann"))
    out_md = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "score_v2.md")))
    link_deg = float(_arg("--link-deg", "60"))
    pred, has_dist = load_pred(pred_path)
    rows = list(csv.DictReader(open(ann_path, encoding="utf-8-sig")))
    events, neg = evaluate(rows, pred, link_deg, has_dist)

    n = len(events)
    scored = [e for e in events if e["notified"] is not None]
    n_notif = sum(1 for e in scored if e["notified"])
    mids = [e for e in scored if e["mid_reach"] is not None]
    n_mid = sum(1 for e in mids if e["mid_reach"])
    leads = [e["lead"] for e in scored if e["lead"] is not None]
    quads = [e["quad_ok"] for e in scored if e["quad_ok"] is not None]
    hours = neg["exposure_s"] / 3600.0
    rep = [f"# 実録採点 v2（pred={pred_path.name}, v3.4 link=±{link_deg:.0f}°）", "",
           f"- イベント {n}件（event_id単位）: 至近警告/警告の到達 {n_notif}/{len(scored)}"
           + ("" if has_dist else "（距離なし予測のため距離クラスは未採点）")]
    if mids:
        rep.append(f"- 中通知（≤{T2}m）以上の到達（距離クラス）: {n_mid}/{len(mids)}")
    rep += [(f"- リード中央値 {np.median(leads):.1f}s（範囲 {min(leads):.1f}〜"
             f"{max(leads):.1f}s、注釈±1s精度）" if leads else "- リード: n/a"),
            (f"- 方向4象限一致 {sum(quads)}/{len(quads)}" if quads else "- 象限: n/a")]
    if hours > 0:
        lo, hi = poisson_rate_ci(neg["n_false"], hours)
        rep.append(f"- 誤警告 {neg['n_false']}件 / {hours:.2f}h（注釈イベント窓は"
                   f"マスク済み）= {neg['n_false']/hours:.2f}回/h"
                   f"（Poisson95%CI {lo:.2f}〜{hi:.2f}回/h）")
    else:
        rep.append("- 負例露出なし")

    pred_b = _arg("--pred-b")
    if pred_b:
        pb, hd_b = load_pred(Path(pred_b))
        ev_b, _ = evaluate(rows, pb, link_deg, hd_b)
        key = lambda e: (e["clip"], e["trial"], e["event_id"], e["class"])
        da = {key(e): e for e in events if e["notified"] is not None}
        db = {key(e): e for e in ev_b if e["notified"] is not None}
        common = sorted(set(da) & set(db))
        b_ = sum(1 for k in common if da[k]["notified"] and not db[k]["notified"])
        c_ = sum(1 for k in common if not da[k]["notified"] and db[k]["notified"])
        rep += ["", "## 対応あり比較（A=--pred, B=--pred-b）",
                f"- 通知の不一致対: A○B×={b_} / A×B○={c_} → "
                f"McNemar厳密p={mcnemar_exact(b_, c_):.4f}"]
        diffs = defaultdict(list)
        for k in common:
            if da[k]["lead"] is not None and db[k]["lead"] is not None:
                diffs[k[0]].append(da[k]["lead"] - db[k]["lead"])
        bs = paired_bootstrap_median_diff_by_clip(diffs)
        if bs:
            n_pair = sum(len(v) for v in diffs.values())
            rep.append(f"- リード差(A−B) 中央値 {bs[0]:+.2f}s（クリップ単位paired "
                       f"bootstrap95%CI {bs[1]:+.2f}〜{bs[2]:+.2f}s, "
                       f"クリップ{len(diffs)}件/対{n_pair}件）")

    rep += ["", "| clip | trial | event | クラス | 通知 | リード[s] | 象限 |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in events:
        mark = ("—" if e["notified"] is None else ("○" if e["notified"] else "×"))
        rep.append(f"| {e['clip']} | {e['trial']} | {e['event_id']} | {e['class']} | "
                   f"{mark} | {e['lead'] if e['lead'] is not None else '—'} | "
                   f"{'○' if e['quad_ok'] else ('×' if e['quad_ok'] is not None else '—')} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:10]))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
