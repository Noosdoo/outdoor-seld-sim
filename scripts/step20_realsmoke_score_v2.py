# -*- coding: utf-8 -*-
"""Step 20 v2: 実録採点の全面改修版（第11回監査 高5・高6・中7への対応）。

旧 step20_realsmoke_score.py の欠陥と本版の修正:
  A) 全区間走査 — 旧は通知関数が先頭100フレーム(10秒)固定で、長尺負例の発火を
     見逃したまま露出分母だけ全秒数を計上していた。本版はクリップ実長で走査する。
  B) event_id単位のイベント窓 — 旧はクリップ内の同一クラス発火を全注釈行で共有し、
     1発火で複数イベントが全部「通知成功」になった。本版は各イベントの
     [t_start−1s, t_cpa＋1s] 窓内の発火だけを貪欲割当（1発火は最大1イベント）する。
  C) 現行通知規則 — 旧はstep12_notify_v9のルールv1を読み込んでいた。本版は
     v3.4（距離≤閾が2フレーム連続＋前フレーム候補と方位連結±LINK_DEG）を
     step12_notify_v33と同一定数で可変長に再実装する（距離クラス）。
     警告音クラス（サイレン等）は従来どおり2フレーム連続検出＋不応期。
  D) 事前登録の対応あり統計 — McNemar厳密検定・paired bootstrap・Poisson厳密区間
     （実録ハンドブック§9-3、2026-08-13改訂）。

入力:
  --pred   予測CSV。6/7列 [stem,frame,class(,track),az,el,dist] を推奨（v12形式）。
           5列 [stem,frame,class,az,el] の旧形式は距離が無いため距離クラスの
           採点をスキップし警告クラスのみ採点（警告を表示）。
  --ann    注釈CSV。列: clip_id,event_id,trial,class,quadrant,t_start,t_cpa[,...]
           class=none 行は負例窓（誤警告率の露出時間として計上）。
  --pred-b 対応あり比較のもう1系統（ablation arm等）。同一注釈で採点し、
           通知有無=McNemar厳密検定 / リード差=クリップ単位paired bootstrap を出力。
  --out    出力md（既定 out/realsmoke/score_v2.md）
  --link-deg 方位連結幅。既定60（=v3.4。停止規則により変更しない）

クラス番号（v12・8クラス）: 0=siren 1=horn 2=backup_beep 3=bike_bell 4=car_drive
                            5=crossing 6=kick 7=bike
距離トリガ対象: car_drive / kick / bike（v12b系）。それ以外は警告クラス。
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
WARN_CONFIRM = 2                 # 警告音クラス: 2フレーム連続で発火
REFRACT_FRAMES = 30              # 不応期3s（v1踏襲・多重カウント防止）
REFRACT_DEG = 60.0
EP_GAP = 10                      # 距離トリガの発火列→エピソード化のフレームギャップ
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
    """v3.4距離トリガの可変長版（step12_notify_v33.dist_triggersと同一規則。
    旧実装のrange(100)固定をnframes引数に一般化）。"""
    hits, prev = [], []
    for j in range(nframes):
        close = [(a, d) for a, d in dseq.get(j, []) if d is not None and d <= thresh]
        if close and prev:
            linked = [(a, d) for a, d in close
                      if any(cdiff(a, pa) <= link_deg for pa, _ in prev)]
            if linked:
                a, _ = min(linked, key=lambda x: x[1])
                hits.append((j, a))
        prev = close
    return hits


def episodes_of(hits, link_deg):
    """発火フレーム列→エピソード（gap>EP_GAP or 方位乖離で分割）。先頭を発火時刻とする。"""
    eps = []
    for j, a in hits:
        if eps and j - eps[-1][-1][0] <= EP_GAP and cdiff(a, eps[-1][-1][1]) <= link_deg:
            eps[-1].append((j, a))
        else:
            eps.append([(j, a)])
    return [(ep[0][0], ep[0][1]) for ep in eps]


def warn_fires(frames_of_cls, az_of, nframes):
    """警告音クラス: WARN_CONFIRM連続検出で発火＋不応期。"""
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
    """clsごとの発火 [(t_sec, az)] を返す。距離クラス=強トリガ(≤T3)のエピソード、
    警告クラス=2連続検出。"""
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
            hits = dist_triggers_var(dseq[c], T3, nframes, link_deg)
            for j, a in episodes_of(hits, link_deg):
                out[c].append((j / FPS, a))
        else:
            for k, a in warn_fires(by_cls_frames[c], az_at[c], nframes):
                out[c].append((k / FPS, a))
    return dict(out)


# ------------------------------------------------------------------ 採点
def evaluate(rows, pred, link_deg, has_dist):
    """注釈行を採点。返り値: (events, negatives)
    events: dict(trial,event_id,class,notified,lead,quad_ok)
    negatives: dict(n_false, exposure_s)"""
    # クリップごとの必要走査長 = 注釈窓の最大端 と 予測最大フレームの大きい方
    need = defaultdict(float)
    for r in rows:
        need[r["clip_id"]] = max(need[r["clip_id"]], float(r["t_cpa"]) + WIN_POST + 1.0)
    fires_by_clip = {}
    for clip in {r["clip_id"] for r in rows}:
        pc = pred.get(clip, {})
        nf = int(max(need[clip] * FPS,
                     (max(pc.keys()) + 1) if pc else 0)) + 1
        fires_by_clip[clip] = fires_for_clip(pc, nf, link_deg)

    events, n_false, exposure_s = [], 0, 0.0
    # 割当: クリップ×クラスごとに、t_cpa昇順のイベントへ窓内最先発火を貪欲割当
    by_key = defaultdict(list)
    for r in rows:
        if r["class"].strip() == "none":
            t0, t1 = float(r["t_start"]), float(r["t_cpa"])
            exposure_s += (t1 - t0)
            fires_all = [t for fl in fires_by_clip[r["clip_id"]].values()
                         for t, _ in fl]
            n_false += sum(1 for t in fires_all if t0 <= t <= t1)
            continue
        by_key[(r["clip_id"], r["class"].strip())].append(r)

    for (clip, cls), evrows in sorted(by_key.items()):
        ci = CLS_IDX[cls]
        if ci in DIST_CLS and not has_dist:
            for r in evrows:
                events.append({"trial": r["trial"], "event_id": r.get("event_id", "1"),
                               "class": cls, "notified": None, "lead": None,
                               "quad_ok": None})
            continue
        cand = sorted(fires_by_clip[clip].get(ci, []))
        used = [False] * len(cand)
        for r in sorted(evrows, key=lambda x: float(x["t_cpa"])):
            t0 = float(r["t_start"]) - WIN_PRE
            t1 = float(r["t_cpa"]) + WIN_POST
            pick = None
            for i, (t, az) in enumerate(cand):
                if not used[i] and t0 <= t <= t1:
                    pick = (i, t, az)
                    break
            if pick is not None:
                used[pick[0]] = True
                events.append({"trial": r["trial"], "event_id": r.get("event_id", "1"),
                               "class": cls, "notified": True,
                               "lead": round(float(r["t_cpa"]) - pick[1], 2),
                               "quad_ok": quadrant_of(pick[2]) == r["quadrant"].strip()})
            else:
                events.append({"trial": r["trial"], "event_id": r.get("event_id", "1"),
                               "class": cls, "notified": False,
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
    """件数kと露出時間[h]から誤警告率[回/h]の厳密95%CI（chi2法）。"""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k) / hours
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / hours
    return lo, hi


def paired_bootstrap_median_diff(xa, xb, n_boot=10000, seed=0):
    """対応ありリード差（A−B）の中央値と95%CI。"""
    rng = np.random.default_rng(seed)
    d = np.asarray(xa, float) - np.asarray(xb, float)
    if len(d) == 0:
        return None
    meds = np.empty(n_boot)
    for i in range(n_boot):
        meds[i] = np.median(d[rng.integers(0, len(d), len(d))])
    return (float(np.median(d)), float(np.percentile(meds, 2.5)),
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
    leads = [e["lead"] for e in scored if e["lead"] is not None]
    quads = [e["quad_ok"] for e in scored if e["quad_ok"] is not None]
    hours = neg["exposure_s"] / 3600.0
    rep = [f"# 実録採点 v2（pred={pred_path.name}, v3.4 link=±{link_deg:.0f}°）", "",
           f"- イベント {n}件（event_id単位）: 通知 {n_notif}/{len(scored)}"
           + ("" if has_dist else "（距離なし予測のため距離クラスは未採点）"),
           (f"- リード中央値 {np.median(leads):.1f}s（範囲 {min(leads):.1f}〜"
            f"{max(leads):.1f}s、注釈±1s精度）" if leads else "- リード: n/a"),
           (f"- 方向4象限一致 {sum(quads)}/{len(quads)}" if quads else "- 象限: n/a")]
    if hours > 0:
        lo, hi = poisson_rate_ci(neg["n_false"], hours)
        rep.append(f"- 誤警告 {neg['n_false']}件 / {hours:.2f}h = "
                   f"{neg['n_false']/hours:.2f}回/h（Poisson95%CI {lo:.2f}〜{hi:.2f}回/h）")
    else:
        rep.append("- 負例露出なし")

    pred_b = _arg("--pred-b")
    if pred_b:
        pb, hd_b = load_pred(Path(pred_b))
        ev_b, _ = evaluate(rows, pb, link_deg, hd_b)
        key = lambda e: (e["trial"], e["event_id"], e["class"])
        da = {key(e): e for e in events if e["notified"] is not None}
        db = {key(e): e for e in ev_b if e["notified"] is not None}
        common = sorted(set(da) & set(db))
        b_ = sum(1 for k in common if da[k]["notified"] and not db[k]["notified"])
        c_ = sum(1 for k in common if not da[k]["notified"] and db[k]["notified"])
        rep += ["", "## 対応あり比較（A=--pred, B=--pred-b）",
                f"- 通知の不一致対: A○B×={b_} / A×B○={c_} → "
                f"McNemar厳密p={mcnemar_exact(b_, c_):.4f}"]
        la = [da[k]["lead"] for k in common
              if da[k]["lead"] is not None and db[k]["lead"] is not None]
        lb = [db[k]["lead"] for k in common
              if da[k]["lead"] is not None and db[k]["lead"] is not None]
        bs = paired_bootstrap_median_diff(la, lb)
        if bs:
            rep.append(f"- リード差(A−B) 中央値 {bs[0]:+.2f}s"
                       f"（paired bootstrap95%CI {bs[1]:+.2f}〜{bs[2]:+.2f}s, n={len(la)}）")

    rep += ["", "| trial | event | クラス | 通知 | リード[s] | 象限 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for e in events:
        mark = ("—" if e["notified"] is None else ("○" if e["notified"] else "×"))
        rep.append(f"| {e['trial']} | {e['event_id']} | {e['class']} | {mark} | "
                   f"{e['lead'] if e['lead'] is not None else '—'} | "
                   f"{'○' if e['quad_ok'] else ('×' if e['quad_ok'] is not None else '—')} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:10]))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
