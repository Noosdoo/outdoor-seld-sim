# -*- coding: utf-8 -*-
"""Step 20 v2: 実録採点（2026-08-13 19:53指摘まで反映）。

設計原則: 通知規則・エピソード統合は**正規実装(step12_notify_v33/v12b, v9)の定数と
規則をそのまま使う**。本スクリプト独自の規則発明はしない。

距離クラス（car/kick/bike）:
  - トリガ: v3.4（≤閾が2フレーム連続＋前フレーム候補と方位連結±LINK_DEG=60）を
    強(T3=1.5)と中(T2=3.0)で**独立に**生成
  - エピソード統合: 正規group_episodesと同一の「**フレーム差≤1かつ方位差≤AZ_MATCH=25°**」
    （旧v2のEP_GAP=10・±60°統合は誤りとして撤回）
  - **1つの物理エピソードは1つ**: 中トリガ列からエピソードを作り、エピソード内に
    強トリガフレームが含まれればtier=強（発火時刻は強成立の因果時刻）、無ければ中。
    消費フラグはエピソード単位で共通（強/中の二重割当は起きない）
警告音クラス: v1定数（3フレーム連続・不応期5s・方向±45°）。
発火時刻は因果時刻 (k+1)/FPS。

GT区分（注釈の横距離mから）: critical ≤1.5 / caution ≤3.2 / safe >3.2
  - critical: 窓内の強エピソード=成功（リードは強成立時刻から）
  - caution : 窓内のエピソード（中/強）=成功（リードは中成立時刻から。
              強での過剰通知はn_strong_on_cautionに別掲）
  - safe    : 窓内に**強・中いずれの通知も無い**こと=成功
              （安全車への通知は段階を問わず失敗。正規の抑制定義と同一）
  - 横距離m欠落/不正の距離クラス行は**未採点**（分母から除外・件数を警告表示）
  - scored=0 の行も**未採点**（step19bの切り出しで最接近がクリップ外になった行。
    誤警告マスクには使うが到達判定の分母には入れない）
負例(class=none): エピソード単位で計数・注釈イベント窓をマスク・露出は重なり控除。
  負例窓は**半開区間 [t_start, t_cpa)**（2026-08-14）。step19b --mode negative が
  重複分割したクリップの担当区間は原録音上で隙間なくタイルするため、境界時刻
  ちょうどの発火も半開区間により**ちょうど1回**だけ数えられる。
誤警告率は両側95%CIに加え片側95%上限（事前登録<1.8回/hの判定用）。
統計: McNemar厳密・クリップ単位paired bootstrap・Poisson厳密区間。

入力列: clip_id,event_id,trial,class,quadrant,t_start,t_cpa[,横距離m,scored,...]
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
AZ_MATCH = 25.0                  # v33のエピソード統合幅（LINK_DEGとは別物）
WARN_CONFIRM = 3                 # v1: 0.3s連続
REFRACT_FRAMES = 50              # v1: 不応期5s
REFRACT_DEG = 45.0               # v1: 方向別±45°
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
    """横距離m→GT区分。Noneは呼び出し側で未採点にする。"""
    if lateral_m <= T3:
        return "critical"
    if lateral_m <= SUPP:
        return "caution"
    return "safe"


def is_scored(row) -> bool:
    """scored列（step19bが付与）。0/false/noは未採点=マスク専用行。"""
    v = row.get("scored")
    if v is None or str(v).strip() == "":
        return True
    return str(v).strip().lower() not in ("0", "false", "no")


def lateral_of(row):
    for k in LATERAL_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            try:
                lateral_m = float(v)
            except (TypeError, ValueError):
                return None
            # NaN/Infinityは比較演算をすり抜けてsafeへ分類され得るため、
            # 横距離として有効な有限・非負値だけを採用する。
            return lateral_m if np.isfinite(lateral_m) and lateral_m >= 0.0 else None
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
    """v3.4距離トリガ（≤threshが2フレーム連続＋方位連結±link_deg）の可変長版。"""
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


def build_episodes(dseq, nframes, link_deg):
    """中(T2)トリガ列を正規group_episodesと同一規則（フレーム差≤1かつ
    方位差≤AZ_MATCH=25°）でエピソード化し、エピソード内に強(T3)トリガが
    含まれればtier=強とする。返り値: [{"t_mid","az_mid","t_strong","az_strong",
    "tier"}]（時刻は因果時刻(j+1)/FPS）。"""
    trig_m = dist_triggers_var(dseq, T2, nframes, link_deg)
    strong = {j: a for j, a, _ in dist_triggers_var(dseq, T3, nframes, link_deg)}
    groups = []
    for j, a, d in trig_m:
        if groups and j - groups[-1][-1][0] == 1 and cdiff(a, groups[-1][-1][1]) <= AZ_MATCH:
            groups[-1].append((j, a))
        else:
            groups.append([(j, a)])
    eps = []
    for g in groups:
        s = [(j, a) for j, a in g if j in strong]
        eps.append({"t_mid": (g[0][0] + 1) / FPS, "az_mid": g[0][1],
                    "t_strong": (s[0][0] + 1) / FPS if s else None,
                    "az_strong": s[0][1] if s else None,
                    "tier": "強" if s else "中"})
    return eps


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
    """cls -> {"episodes":[...]}（距離クラス） / {"warn":[(t,az)]}（警告クラス）"""
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
            out[c] = {"episodes": build_episodes(dseq[c], nframes, link_deg)}
        else:
            out[c] = {"warn": warn_fires(by_cls_frames[c], az_at[c], nframes)}
    return out


# ------------------------------------------------------------------ 採点
def evaluate(rows, pred, link_deg, has_dist):
    """返り値: (events, negatives, extras)。
    events[i]: clip,trial,event_id,class,gt_tier,notified,fired_tier,lead,quad_ok
    （横距離欠落の距離クラス行は notified=None=未採点で分母除外）"""
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
    extras = {"n_strong_on_caution": 0, "n_mid_on_safe": 0, "n_unscored": 0,
              "n_maskonly": 0}
    by_key = defaultdict(list)
    for r in rows:
        clip = r["clip_id"]
        if r["class"].strip() == "none":
            t0, t1 = float(r["t_start"]), float(r["t_cpa"])
            exposure_s += (t1 - t0) - overlap(t0, t1, pos_windows[clip])
            fires_all = []
            for grp in fires_by_clip[clip].values():
                fires_all += [ep["t_mid"] for ep in grp.get("episodes", [])]
                fires_all += [t for t, _ in grp.get("warn", [])]
            # 半開区間 [t0, t1): 重複分割した隣接クリップの担当が原録音上で
            # 隙間なくタイルするため、境界時刻の発火も一意に1回だけ数えられる
            n_false += sum(1 for t in fires_all
                           if t0 <= t < t1 and not masked(clip, t))
            continue
        if not is_scored(r):
            # 最接近がクリップ外＝到達判定はできないが、pos_windowsには既に
            # 入っているので誤警告マスクとしては働く（本物のイベントを
            # 誤警告に数えないため行を消してはいけない）
            extras["n_maskonly"] += 1
            events.append({"clip": clip, "trial": r.get("trial", ""),
                           "event_id": r.get("event_id", "1"),
                           "class": r["class"].strip(), "gt_tier": "mask",
                           "notified": None, "fired_tier": None,
                           "lead": None, "quad_ok": None})
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
            eps = grp.get("episodes", [])
            used = [False] * len(eps)
            for r in sorted(evrows, key=lambda x: float(x["t_cpa"])):
                lat = lateral_of(r)
                base = {"clip": clip, "trial": r["trial"],
                        "event_id": r.get("event_id", "1"), "class": cls}
                if lat is None:
                    extras["n_unscored"] += 1
                    events.append({**base, "gt_tier": "-", "notified": None,
                                   "fired_tier": None, "lead": None,
                                   "quad_ok": None})
                    continue
                tier = gt_tier_of(lat)
                t0 = float(r["t_start"]) - WIN_PRE
                t1 = float(r["t_cpa"]) + WIN_POST
                base["gt_tier"] = tier
                if tier == "safe":
                    # 安全車: 強・中いずれの通知も無いこと=成功（消費はしない）
                    in_w = [ep for ep in eps if t0 <= ep["t_mid"] <= t1
                            or (ep["t_strong"] is not None
                                and t0 <= ep["t_strong"] <= t1)]
                    extras["n_mid_on_safe"] += sum(1 for ep in in_w
                                                   if ep["tier"] == "中")
                    events.append({**base, "notified": not in_w,
                                   "fired_tier": (in_w[0]["tier"] if in_w else None),
                                   "lead": None, "quad_ok": None})
                    continue
                pick = None
                for i, ep in enumerate(eps):
                    if used[i]:
                        continue
                    if tier == "critical":
                        if ep["tier"] == "強" and t0 <= ep["t_strong"] <= t1:
                            pick = (i, ep["t_strong"], ep["az_strong"], ep["tier"])
                            break
                    else:  # caution: 中/強どちらのエピソードでも、中成立時刻で判定
                        if t0 <= ep["t_mid"] <= t1:
                            pick = (i, ep["t_mid"], ep["az_mid"], ep["tier"])
                            break
                if pick is not None:
                    i, t, az, ft = pick
                    used[i] = True
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
           f"- イベント {len(events)}件（event_id単位・採点対象 {len(scored)}件）"
           + ("" if has_dist else "（距離なし予測のため距離クラスは未採点）")]
    label = {"critical": "critical到達（強）", "caution": "caution到達（中以上）",
             "safe": "safe抑制（強・中とも通知なし）", "warn": "警告音到達"}
    for t in ("critical", "caution", "safe", "warn"):
        if by_tier[t][1]:
            rep.append(f"- {label[t]}: {by_tier[t][0]}/{by_tier[t][1]}")
    if extras["n_strong_on_caution"]:
        rep.append(f"- caution車への強通知（過剰・別掲）: {extras['n_strong_on_caution']}件")
    if extras["n_mid_on_safe"]:
        rep.append(f"- safe車への中通知（失敗の内訳・別掲）: {extras['n_mid_on_safe']}件")
    if extras["n_unscored"]:
        rep.append(f"- ⚠️横距離m欠落で未採点（分母除外）: {extras['n_unscored']}件")
    if extras["n_maskonly"]:
        rep.append(f"- scored=0（最接近がクリップ外・マスク専用）: "
                   f"{extras['n_maskonly']}件")
    rep += [(f"- リード中央値 {np.median(leads):.1f}s（範囲 {min(leads):.1f}〜"
             f"{max(leads):.1f}s、注釈±1s精度）" if leads else "- リード: n/a"),
            (f"- 方向4象限一致 {sum(quads)}/{len(quads)}" if quads else "- 象限: n/a")]
    if hours > 0:
        lo, hi = poisson_rate_ci(neg["n_false"], hours)
        up1 = poisson_upper95_one_sided(neg["n_false"], hours)
        rep.append(f"- 誤警告 {neg['n_false']}件 / {hours:.2f}h（エピソード単位・"
                   f"注釈イベント窓マスク済み）= {neg['n_false']/hours:.2f}回/h"
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
