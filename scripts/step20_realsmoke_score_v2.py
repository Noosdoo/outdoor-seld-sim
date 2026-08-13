# -*- coding: utf-8 -*-
"""Step 20 v2: 実録採点（第11回監査系の対応版。2026-08-13 19:18指摘まで反映）。

規則の正: 距離クラスは step12_notify_v33/v12b と同じく **T3(1.5m)とT2(3.0m)で
独立に** 2フレーム連続＋方位連結±LINK_DEG のトリガを生成する（T2列の
エピソード最小値で強へ昇格させる旧v2実装は誤りとして撤回）。
警告音クラスは v1 の事前決定値（3フレーム連続・不応期5s・方向±45°）。
発火時刻は因果時刻 (k+1)/FPS（step12_notify_v9.emit_time と同一）。

GT区分（実録の主評価）: 注釈の**横距離m**から
  critical: ≤1.5m ／ caution: 1.5〜3.2m ／ safe: >3.2m
を定義（3.0–3.2mはグレーだが抑制境界SUPP=3.2でcaution側に含める）。
実録はcritical車を安全上収録しないため、車系の主評価は caution/safe。
  - critical: 窓内の強発火=成功（リード・象限は強から）
  - caution : 窓内の中or強発火=成功（リード・象限はその発火から。
              強での過剰通知は n_strong_on_caution として別掲）
  - safe    : 窓内に強発火が**無い**こと=成功（安全車への誤・強通知=失敗。
              中発火は n_mid_on_safe として別掲）
  - 警告音クラス: 窓内の警告発火=成功
横距離m列が無い距離クラス行はcritical扱いで採点し、警告を表示する。

負例(class=none): 注釈済みイベント窓に入る発火をマスクし、露出も重なりを控除。
誤警告率は両側95%CIに加えて**片側95%上限**（事前登録の「<1.8回/h」判定用）を出力。

統計: McNemar厳密（イベント成功のA/B不一致対）・クリップ単位paired bootstrap
（リード差）・Poisson厳密区間。比較キーは (clip, trial, event_id, class)。

入力列: clip_id,event_id,trial,class,quadrant,t_start,t_cpa[,横距離m,...]
予測CSV: 6/7列 [stem,frame,class(,track),az,el,dist]（5列旧形式は警告クラスのみ）
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

FPS = 10.0
T3, T2, SUPP = 1.5, 3.0, 3.2     # step12_notify_v33と同一
WARN_CONFIRM = 3                 # v1: 0.3s連続
REFRACT_FRAMES = 50              # v1: 不応期5s
REFRACT_DEG = 45.0               # v1: 方向別±45°
EP_GAP = 10
WIN_PRE, WIN_POST = 1.0, 1.0

CLS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
           "car_drive": 4, "crossing": 5, "kick": 6, "bike": 7}
DIST_CLS = {4, 6, 7}
LATERAL_KEYS = ("横距離m", "横距離", "lateral_m")


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


def gt_tier_of(lateral_m):
    if lateral_m is None:
        return "critical"
    if lateral_m <= T3:
        return "critical"
    if lateral_m <= SUPP:
        return "caution"
    return "safe"


def lateral_of(row):
    for k in LATERAL_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------- 予測の読込
def load_pred(path: Path):
    out = defaultdict(lambda: defaultdict(list))
    has_dist = True
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[4]), float(p[6])
        elif len(p) == 6:
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[3]), float(p[5])
        elif len(p) == 5:
            clip, k, c = p[0], int(p[1]), int(p[2])
            az, d = float(p[3]), None
            has_dist = False
        else:
            continue
        out[clip][k].append((c, az, d))
    return dict(out), has_dist


# ------------------------------------------------------------ 通知発火（可変長）
def dist_triggers_var(dseq, thresh, nframes, link_deg):
    """v3.4距離トリガの可変長版（≤threshが2フレーム連続＋方位連結）。
    強(T3)・中(T2)は**この関数を閾値別に呼んで独立に**生成する。"""
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
    """発火列→エピソード先頭 [(t_causal, az)]。因果時刻=(j+1)/FPS。"""
    eps = []
    for j, a, _ in hits:
        if eps and j - eps[-1][-1][0] <= EP_GAP and cdiff(a, eps[-1][-1][1]) <= link_deg:
            eps[-1].append((j, a))
        else:
            eps.append([(j, a)])
    return [((ep[0][0] + 1) / FPS, ep[0][1]) for ep in eps]


def warn_fires(frames_of_cls, az_of, nframes):
    """警告音クラス: v1定数。発火時刻は因果時刻(k+1)/FPS。"""
    fires, last = [], []
    for k in range(nframes):
        if all((k - i) in frames_of_cls for i in range(WARN_CONFIRM)):
            az = az_of[k]
            if any(k - kp < REFRACT_FRAMES and cdiff(az, ap) <= REFRACT_DEG
                   for kp, ap in last):
                continue
            fires.append(((k + 1) / FPS, az))
            last.append((k, az))
    return fires


def fires_for_clip(pred_clip, nframes, link_deg):
    """cls -> {"strong":[(t,az)], "mid":[(t,az)]}（距離クラス）
              {"warn":[(t,az)]}（警告クラス）"""
    by_cls_frames = defaultdict(set)
    az_at = defaultdict(dict)
    dseq = defaultdict(lambda: defaultdict(list))
    for k, evs in pred_clip.items():
        for c, az, d in evs:
            by_cls_frames[c].add(k)
            az_at[c][k] = az
            if c in DIST_CLS:
                dseq[c][k].append((az, d))
    out = {}
    for c in sorted(by_cls_frames):
        if c in DIST_CLS:
            strong = episodes_of(dist_triggers_var(dseq[c], T3, nframes, link_deg),
                                 link_deg)
            mid = episodes_of(dist_triggers_var(dseq[c], T2, nframes, link_deg),
                              link_deg)
            out[c] = {"strong": strong, "mid": mid}
        else:
            out[c] = {"warn": warn_fires(by_cls_frames[c], az_at[c], nframes)}
    return out


# ------------------------------------------------------------------ 採点
def evaluate(rows, pred, link_deg, has_dist):
    """返り値: (events, negatives, extras)
    events[i]: clip,trial,event_id,class,gt_tier,notified(=GT区分に応じた成功),
               fired_tier,lead,quad_ok
    negatives: n_false, exposure_s（イベント窓マスク・露出控除済み）
    extras: n_strong_on_caution, n_mid_on_safe, n_no_lateral"""
    need = defaultdict(float)
    for r in rows:
        need[r["clip_id"]] = max(need[r["clip_id"]], float(r["t_cpa"]) + WIN_POST + 1.0)
    fires_by_clip = {}
    for clip in {r["clip_id"] for r in rows}:
        pc = pred.get(clip, {})
        nf = int(max(need[clip] * FPS,
                     (max(pc.keys()) + 1) if pc else 0)) + 1
        fires_by_clip[clip] = fires_for_clip(pc, nf, link_deg)

    pos_windows = defaultdict(list)
    for r in rows:
        if r["class"].strip() != "none":
            pos_windows[r["clip_id"]].append(
                (float(r["t_start"]) - WIN_PRE, float(r["t_cpa"]) + WIN_POST))

    def masked(clip, t):
        return any(a <= t <= b for a, b in pos_windows[clip])

    def overlap(t0, t1, wins):
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
    extras = {"n_strong_on_caution": 0, "n_mid_on_safe": 0, "n_no_lateral": 0}
    by_key = defaultdict(list)
    for r in rows:
        clip = r["clip_id"]
        if r["class"].strip() == "none":
            t0, t1 = float(r["t_start"]), float(r["t_cpa"])
            exposure_s += (t1 - t0) - overlap(t0, t1, pos_windows[clip])
            # 誤警告は通知エピソード単位: 同一クラスで強と時間的に重なる中は
            # 同じ通知の段階違いなので1件に統合（強を代表にする）
            fires_all = []
            for grp in fires_by_clip[clip].values():
                strong_t = [t for t, _ in grp.get("strong", [])]
                fires_all += strong_t
                fires_all += [t for t, _ in grp.get("mid", [])
                              if all(abs(t - ts) > EP_GAP / FPS for ts in strong_t)]
                fires_all += [t for t, _ in grp.get("warn", [])]
            n_false += sum(1 for t in fires_all
                           if t0 <= t <= t1 and not masked(clip, t))
            continue
        by_key[(clip, r["class"].strip())].append(r)

    for (clip, cls), evrows in sorted(by_key.items()):
        ci = CLS_IDX[cls]
        grp = fires_by_clip[clip].get(ci, {})
        if ci in DIST_CLS and not has_dist:
            for r in evrows:
                events.append({"clip": clip, "trial": r["trial"],
                               "event_id": r.get("event_id", "1"), "class": cls,
                               "gt_tier": "-", "notified": None, "fired_tier": None,
                               "lead": None, "quad_ok": None})
            continue
        if ci in DIST_CLS:
            strong = sorted(grp.get("strong", []))
            mid = sorted(grp.get("mid", []))
            used_s = [False] * len(strong)
            used_m = [False] * len(mid)
            for r in sorted(evrows, key=lambda x: float(x["t_cpa"])):
                lat = lateral_of(r)
                if lat is None:
                    extras["n_no_lateral"] += 1
                tier = gt_tier_of(lat)
                t0 = float(r["t_start"]) - WIN_PRE
                t1 = float(r["t_cpa"]) + WIN_POST
                base = {"clip": clip, "trial": r["trial"],
                        "event_id": r.get("event_id", "1"), "class": cls,
                        "gt_tier": tier}
                s_in = [(i, t, az) for i, (t, az) in enumerate(strong)
                        if t0 <= t <= t1]
                m_in = [(i, t, az) for i, (t, az) in enumerate(mid)
                        if t0 <= t <= t1]
                if tier == "safe":
                    # 安全車: 強発火が1件でもあれば失敗（誤・強通知）
                    if m_in:
                        extras["n_mid_on_safe"] += len(m_in)
                    ok = not s_in
                    events.append({**base, "notified": ok,
                                   "fired_tier": ("強" if s_in else
                                                  ("中" if m_in else None)),
                                   "lead": None, "quad_ok": None})
                    continue
                if tier == "critical":
                    pool = [("強", i, t, az) for i, t, az in s_in
                            if not used_s[i]]
                else:  # caution: 中or強の早い方
                    pool = sorted(
                        [("強", i, t, az) for i, t, az in s_in if not used_s[i]]
                        + [("中", i, t, az) for i, t, az in m_in if not used_m[i]],
                        key=lambda x: x[2])
                if pool:
                    ft, i, t, az = pool[0]
                    (used_s if ft == "強" else used_m)[i] = True
                    if tier == "caution" and ft == "強":
                        extras["n_strong_on_caution"] += 1
                    events.append({**base, "notified": True, "fired_tier": ft,
                                   "lead": round(float(r["t_cpa"]) - t, 2),
                                   "quad_ok": quadrant_of(az) == r["quadrant"].strip()})
                else:
                    events.append({**base, "notified": False, "fired_tier": None,
                                   "lead": None, "quad_ok": None})
        else:
            cand = sorted(grp.get("warn", []))
            used = [False] * len(cand)
            for r in sorted(evrows, key=lambda x: float(x["t_cpa"])):
                t0 = float(r["t_start"]) - WIN_PRE
                t1 = float(r["t_cpa"]) + WIN_POST
                base = {"clip": clip, "trial": r["trial"],
                        "event_id": r.get("event_id", "1"), "class": cls,
                        "gt_tier": "warn"}
                pick = None
                for i, (t, az) in enumerate(cand):
                    if not used[i] and t0 <= t <= t1:
                        pick = (i, t, az)
                        break
                if pick is not None:
                    used[pick[0]] = True
                    events.append({**base, "notified": True, "fired_tier": "警告",
                                   "lead": round(float(r["t_cpa"]) - pick[1], 2),
                                   "quad_ok": quadrant_of(pick[2]) == r["quadrant"].strip()})
                else:
                    events.append({**base, "notified": False, "fired_tier": None,
                                   "lead": None, "quad_ok": None})
    return events, {"n_false": n_false, "exposure_s": exposure_s}, extras


# ------------------------------------------------------------------ 統計
def mcnemar_exact(b: int, c: int) -> float:
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


def poisson_rate_ci(k: int, hours: float, alpha=0.05):
    """両側CI（既定alpha=0.05）。片側95%上限は alpha=0.10 の上side。"""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k) / hours
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / hours
    return lo, hi


def poisson_upper95_one_sided(k: int, hours: float) -> float:
    """事前登録の判定基準（<1.8回/h）用の片側95%上限。"""
    return poisson_rate_ci(k, hours, alpha=0.10)[1]


def paired_bootstrap_median_diff_by_clip(diffs_by_clip, n_boot=10000, seed=0):
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
    events, neg, extras = evaluate(rows, pred, link_deg, has_dist)

    scored = [e for e in events if e["notified"] is not None]
    by_tier = defaultdict(lambda: [0, 0])
    for e in scored:
        by_tier[e["gt_tier"]][0] += int(bool(e["notified"]))
        by_tier[e["gt_tier"]][1] += 1
    leads = [e["lead"] for e in scored if e["lead"] is not None]
    quads = [e["quad_ok"] for e in scored if e["quad_ok"] is not None]
    hours = neg["exposure_s"] / 3600.0
    rep = [f"# 実録採点 v2（pred={pred_path.name}, v3.4 link=±{link_deg:.0f}°）", "",
           f"- イベント {len(events)}件（event_id単位）"
           + ("" if has_dist else "（距離なし予測のため距離クラスは未採点）")]
    label = {"critical": "critical到達（強）", "caution": "caution到達（中以上）",
             "safe": "safe抑制（強なし）", "warn": "警告音到達"}
    for t in ("critical", "caution", "safe", "warn"):
        if by_tier[t][1]:
            rep.append(f"- {label[t]}: {by_tier[t][0]}/{by_tier[t][1]}")
    if extras["n_strong_on_caution"]:
        rep.append(f"- caution車への強通知（過剰・別掲）: {extras['n_strong_on_caution']}件")
    if extras["n_mid_on_safe"]:
        rep.append(f"- safe車への中通知（別掲）: {extras['n_mid_on_safe']}件")
    if extras["n_no_lateral"]:
        rep.append(f"- ⚠️横距離m欠落でcritical扱いにした行: {extras['n_no_lateral']}件")
    rep += [(f"- リード中央値 {np.median(leads):.1f}s（範囲 {min(leads):.1f}〜"
             f"{max(leads):.1f}s、注釈±1s精度）" if leads else "- リード: n/a"),
            (f"- 方向4象限一致 {sum(quads)}/{len(quads)}" if quads else "- 象限: n/a")]
    if hours > 0:
        lo, hi = poisson_rate_ci(neg["n_false"], hours)
        up1 = poisson_upper95_one_sided(neg["n_false"], hours)
        rep.append(f"- 誤警告 {neg['n_false']}件 / {hours:.2f}h（注釈イベント窓マスク済み）"
                   f"= {neg['n_false']/hours:.2f}回/h"
                   f"（両側95%CI {lo:.2f}〜{hi:.2f}／**片側95%上限 {up1:.2f}回/h**"
                   f"=事前登録の<1.8回/h判定用）")
    else:
        rep.append("- 負例露出なし")

    pred_b = _arg("--pred-b")
    if pred_b:
        pb, hd_b = load_pred(Path(pred_b))
        ev_b, _, _ = evaluate(rows, pb, link_deg, hd_b)
        key = lambda e: (e["clip"], e["trial"], e["event_id"], e["class"])
        da = {key(e): e for e in events if e["notified"] is not None}
        db = {key(e): e for e in ev_b if e["notified"] is not None}
        common = sorted(set(da) & set(db))
        b_ = sum(1 for k in common if da[k]["notified"] and not db[k]["notified"])
        c_ = sum(1 for k in common if not da[k]["notified"] and db[k]["notified"])
        rep += ["", "## 対応あり比較（A=--pred, B=--pred-b。成功はGT区分に応じた定義）",
                f"- 成功の不一致対: A○B×={b_} / A×B○={c_} → "
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

    rep += ["", "| clip | trial | event | クラス | GT区分 | 成功 | 発火tier | リード[s] | 象限 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in events:
        mark = ("—" if e["notified"] is None else ("○" if e["notified"] else "×"))
        rep.append(f"| {e['clip']} | {e['trial']} | {e['event_id']} | {e['class']} | "
                   f"{e['gt_tier']} | {mark} | {e['fired_tier'] or '—'} | "
                   f"{e['lead'] if e['lead'] is not None else '—'} | "
                   f"{'○' if e['quad_ok'] else ('×' if e['quad_ok'] is not None else '—')} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:12]))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
