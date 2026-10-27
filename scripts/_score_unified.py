# -*- coding: utf-8 -*-
"""統一採点器（開発用 v2dev・2026-09-05）— 中立監査 §1〜§3 への対応。

宣言どおりの定義に揃える:
  1. 通知規則 = **v4.3（out/notify_v43_sweep/winner.json）＋警告音 v9c（名札方式・2026-09-28 変更票 #41 で hold から置換）**、帰属 = **車ごとの方位帰属**
     （同クラス・±0.5 s・方位差 ≤30°、`_notify_v43_attrib.attribute`）。旧 `_hp_score.py` は v4.2＋時間窓帰属だった。
     旧 hold（前回検出から 5 s・アンカー方位 ±45°）は `--warn-rule hold` で並べて採点する（新旧併記の規約・凍結文書 §5 改訂 r5）
  2. 評価対象 = **plan（manifest）で固定**。予測が無いクリップも「発火ゼロ」として数える（旧: 予測があるクリップだけ走査）
  3. 距離の対応付け = **1 対 1**（同クラス・同フレーム・方位差 ≤20°、方位差の小さい順に貪欲）。旧: 予測ごとに最近傍 GT（GT を消費しない）
  4. 至近捕捉率 = **全至近 GT（≤1.5 m）を分母**（未検出も分母に入る）。旧「条件付き」（成立ペアだけが分母）も併記
  5. 3D 距離を出すモデル（ft2・v15c）は **予測側を d×cos(el) で水平に変換**してから水平 GT で採点（`ラベル=csv@h`）

使い方:
  python scripts/_score_unified.py <出力md> --plan <assignment.csv> [--split fold2] [--clip-max N] --meta <GTディレクトリ>
      [--minframe 40] [--bands 1.4,1.6,1.85,2.1] <ラベル>=<val_all_causal.csv>[@h] ...
  --minframe 0 で「起動直後を含む」値を出す（既定 40 = 因果推論の暖機 4 秒を除く。宣言と同じ）。
  --bands を付けると plan の mic_z で帯ごとの行も出す（クリップごとの集計を 1 回だけ計算して帯で足し合わせる）。
  --events <台帳.csv>（2026-10-26 本体 §2-5・§9-1・§9-4・変更票 #75 ⑭⑮⑯⑰）: v30 の採点。事象 = `_event_ledger_v30.py` の物理の通過（class・track ごとに 1 事象・
    音の途切れで割らない・分母 = 事前に定めた除外〔暖機〕のあとの物理の事象すべて・ラベル全消失も分母）。通知の規則（v4.3／v9c）は **frame 40 以降の予測だけ**に通す
    （名札の記憶も 40 から・--rule-start で変えられる）。警告音の「届いた」行（窓 [採点上の鳴り始め − 1, ＋4]・遅れも採点上の鳴り始めから）・距離の誤差の帯
    （≤1.5／1.5〜3.2／≤3.2 と「いつも同じ値を出す行」）・自転車の横距離別（lateral_m）を足し、「2.0 秒以上前」の割合は参考の節へ移す。--events なしは今までどおり。
  --confusion <json>（2026-10-27・本人 12:35 (a)(b)・scripts/_class_confusion.py）: 取り違えの表（フレーム・9×9＋なし）と本ごとの種類の正答率（参考・
    --events があれば台帳の事象の窓・無ければ正解から窓）を別の json と同じ名前の .md に書く。主の md は 1 バイトも変えない。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V43 = _load("nv43", "step12_notify_v43.py")
DG = _load("nv42diag", "_notify_v42_diag.py")
A = _load("nv43attrib", "_notify_v43_attrib.py")
H = _load("nhold", "step12_notify_v9b_hold.py")
V9C = _load("nv9c", "step12_notify_v9c_multitrack.py")   # 警告音の規則 v9c（名札で 1 台ずつ追跡・変更票 #41・2026-09-28 16:13 本人採用）

C43 = V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
# 警告音の規則（変更票 #41・凍結文書 改訂 r5）: 既定 = v9c（門 45°・同一方位 30° の二重出力を 1 つに・名札ごとに 1 回・寿命 5 s）。
# 旧 hold（前回検出から 5 s・アンカー方位 ±45°）は --warn-rule hold で並べて採点する（新旧併記の規約）。
WARN_RULE = "v9c"
WARN_RULES = ("v9c", "hold")
WARN_GATE_DEG, WARN_DEDUP_DEG = 45.0, 30.0
FPS = v4.FPS
DIST_CLS = {4, 6, 7, 8}       # car / kick / bike / bicycle（距離クラス。8 = bicycle は 2026-10-08 変更票 #48）
AZ_PAIR = 20.0                # 距離対応の方位ゲート（旧と同じ）
TH_CLOSE, TH_SAFE = 1.5, 3.2
EVENTS = None          # --events のときだけ {clip: [台帳の行]}（本体 §2-5）。None = 旧（metadata の連続ラン）
RULE_START = 0         # 通知の規則に通す最初の枠（--events の既定 40・旧は 0 = 全部）
WARN_WIN = (1.0, 4.0)  # 警告音の届いた窓 [採点上の鳴り始め − 1, ＋4]（step20 と同じ・§9-4 の 1）
CONST_CLOSE_M = 1.39   # 「いつも同じ値を出す行」の至近の値（選択器 2.1 の議論の値・⑬）


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


DIST_TABLE = None       # --dist-table <json>: 予測距離を規則・対応付けの前で補正する（v2 束 §6.1 判断 1・_dist_calib_proto.py の表）
DIST_TABLE_MAX = 5.0    # --dist-table-max <m>: この予測距離以下だけ補正
DIST_TABLE_FOR = None   # --dist-table-for <label,...>（Astra R72-01・2026-10-20）: 表を掛ける入力の名前。main では必須（v1・オラクルに掛けない）。外の道具が DIST_TABLE だけを置く旧の使い方は従来どおり全入力
DIST_TABLE_CUR = None   # いま読んでいる入力に掛ける表（pred_for が入力ごとに置く）


def corr_d(d: float) -> float:
    tbl = DIST_TABLE_CUR if DIST_TABLE_FOR is not None else DIST_TABLE
    if tbl is None or not math.isfinite(d) or d >= min(5.0, DIST_TABLE_MAX):
        return d
    return float(np.interp(d, tbl[0], tbl[1]))


def pred_for(label: str, path: Path, horiz: bool, strict: bool = False):
    """入力 label の予測を読む。--dist-table-for に label があるときだけ表を掛ける（R72-01: v1・オラクルには掛けない）。"""
    global DIST_TABLE_CUR
    DIST_TABLE_CUR = DIST_TABLE if (DIST_TABLE_FOR is not None and label in DIST_TABLE_FOR) else None
    try:
        return load_pred_rows(path, horiz, strict)
    finally:
        DIST_TABLE_CUR = None


def load_pred_rows(path: Path, horiz: bool, strict: bool = False):
    """clip -> frame -> [(cls, az, el, d)]。horiz なら d を d×cos(el) に。DIST_TABLE があれば（水平化の後で）補正。
    strict（依頼 2-5・Astra R1・2026-09-14）: 7 列（clip,frame,cls,track,az,el,dist）以外の行が 1 つでもあれば拒否（6 列の混在・欠けた行）。"""
    out = defaultdict(lambda: defaultdict(list))
    import sys as _sys; _sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import normalize as _norm     # 2026-09-22 改名: 旧名の予測（過去の走行）も新名の plan・GT と突き合わせる
    origins = {}                                              # 正規化 ID -> 元 ID の集合（旧名と新名が同じ本を指す衝突の検出・再監査 9/22 R01）
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.strip().split(",")
        if strict and not (len(p) == 7 or (len(p) == 1 and not p[0])):
            raise ValueError(f"{path.name}:{ln}: 7 列でない行（{len(p)} 列）— --strict")
        if p and p[0]:
            raw = p[0]
            p[0] = _norm(raw)
            seen = origins.setdefault(p[0], set())
            seen.add(raw)
            if len(seen) > 1 and strict:
                raise ValueError(f"{path.name}:{ln}: 予測 csv に同じ本を指す旧名と新名が混在（{sorted(seen)}）— --strict")
        if len(p) >= 7:
            clip, k, c, az, el = p[0], int(p[1]), int(p[2]), float(p[4]), float(p[5])
            d = float(p[6]) if p[6] not in ("", "nan") else float("nan")
        elif len(p) == 6:
            clip, k, c, az, el = p[0], int(p[1]), int(p[2]), float(p[3]), float(p[4])
            d = float(p[5])
        else:
            continue
        if horiz and math.isfinite(d):
            d = d * math.cos(math.radians(el))
        if c in DIST_CLS:
            d = corr_d(d)
        out[clip][k].append((c, az, el, d))
    return out


def load_gt(meta: Path, clip: str):
    """frame -> [(cls, trk, az, d)]（距離クラスのみ）"""
    f = meta / f"{clip}.csv"
    out = defaultdict(list)
    if not f.exists():
        return out
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) < 6:
            continue
        c = int(g[1])
        if c in DIST_CLS:
            out[int(g[0])].append((c, int(g[2]), float(g[3]), float(g[5])))
    return out


LAT_GATE = None   # --lat-gate "A,2.0,10,3.0" = (案 A/B, 強の門 T[m], 窓[frame], 中の門 T_mid[m] or '-')（v4.7 候補・宣言 2026-09-07）


def _lat_est(frames_dist, j, az0, cls, win):
    vals = [d * abs(math.sin(math.radians(az))) for k in range(j - win + 1, j + 1)
            for c, az, d in frames_dist.get(k, []) if c == cls and math.isfinite(d) and dang(az, az0) <= 30.0]
    return float(np.median(vals)) if vals else float("nan")


def apply_lat_gate(fires, frames_dist):
    """v4.7 横距離の門: 強の発火で見積もり ≥ T なら 中に下げる（A）／出さない（B）。中は T_mid（任意）で出さない。"""
    if LAT_GATE is None:
        return fires
    mode, T, win, tmid = LAT_GATE
    out = []
    for j, az, tier, d, cls in fires:
        if tier == "強":
            v = _lat_est(frames_dist, j, az, cls, win)
            if math.isfinite(v) and v >= T:
                if mode == "B":
                    continue
                tier = "中"
        if tier == "中" and tmid is not None:
            v = _lat_est(frames_dist, j, az, cls, win)
            if math.isfinite(v) and v >= tmid:
                continue
        out.append((j, az, tier, d, cls))
    return out


NOTIFY_RULE = "v43"                 # 変更票 #73（Q2）: 既定 = この研究の規則 v4.3。--notify-rule detect = 「対象音を検出したら知らせる（距離を見ない）」（_notify_detect.py・比較用）
NOTIFY_RULES = ("v43", "detect")
DET = _load("ndetect", "_notify_detect.py")


def fires_of(frames):
    """v4.3＋hold の発火 [(frame, az, tier, d, cls)]（tier: 強/中/警告）。frames: frame -> [(cls, az, el, d)]。LAT_GATE があれば後段で門を掛ける。
    NOTIFY_RULE = detect のときは距離の規則の部分だけ _notify_detect.run_rule_detect（追跡・確認・エピソード化は同じ部品・全部 強）に替える。警告音の規則は同じ。"""
    frames_dist = {k: [(c, az, d) for c, az, el, d in v if math.isfinite(d)] for k, v in frames.items()}
    frames_warn = {k: [(c, az, el) for c, az, el, d in v] for k, v in frames.items()}
    fires = []
    if NOTIFY_RULE == "detect":
        res = DET.run_rule_detect({"x": frames_dist}, C43, nframes=100).get("x", {})
    else:
        res = V43.run_rule3({"x": frames_dist}, C43, nframes=100).get("x", {})
    for cls, lst in res.items():
        for j, az, tier, d in lst:
            fires.append((j, az, tier, d, cls))
    fires = apply_lat_gate(fires, frames_dist)
    for k, cls, az in warn_fires(frames_warn, nframes=100):
        fires.append((k, az, "警告", 0.0, cls))
    return sorted(fires)


def warn_fires(frames_warn, nframes: int = 100):
    """警告音の発火 [(frame, cls, az)]。WARN_RULE = v9c（既定・名札方式）か hold（旧・併記用）。"""
    if WARN_RULE == "hold":
        return H.warn_fires(frames_warn, hold=True, nframes=nframes)
    if WARN_RULE == "v9c":
        return V9C.warn_fires_v9c(frames_warn, WARN_GATE_DEG, nframes=nframes, dedup_deg=WARN_DEDUP_DEG)
    raise ValueError(f"未知の警告音の規則: {WARN_RULE}（{WARN_RULES}）")


def ledger_events(meta: Path, clip: str) -> tuple[list, list]:
    """--events: (距離クラスの事象 = DG.mk_event と同じ形, 警告音の事象)。除外の行は入れない。
    fr = 通知の帰属に使う枠 → (方位, 距離): 方位は台帳の az_series（scene の軌道の受音時刻の見かけの方位・歩く向きで回したもの・Fable 10/26 16:00）、
    距離は正解の枠の値（全消失の枠は nan）。ラベルのある枠では今と同じ方位・全消失の事象でも方向の合う発火があれば通知と数える。"""
    rows = [r for r in EVENTS.get(clip, []) if not r["excluded"]]
    per = defaultdict(dict)
    f = meta / f"{clip}.csv"
    if f.exists():
        for line in open(f, encoding="utf-8"):
            g = line.strip().split(",")
            if len(g) >= 6:
                per[(int(g[1]), int(g[2]))][int(g[0])] = (float(g[3]), float(g[5]))
    dist, warn = [], []
    for r in rows:
        ci, trk = int(r["cls"]), int(r["track"])
        lab = per.get((ci, trk), {})
        azs = (r.get("az_series") or "").split(";") if r.get("az_series") else []
        fr = {k: (float(a), lab.get(k, (None, float("nan")))[1]) for k, a in enumerate(azs) if a != ""} if azs else lab
        if r["kind"] == "warn":
            warn.append(dict(cls=ci, trk=trk, onset=float(r["score_onset_s"]), name=r["class"]))
            continue
        js = sorted(lab)
        cpa = min(int(float(r["phys_cpa_s"]) * FPS), 99)
        dist.append(dict(cls=ci, trk=trk, fr=fr, f0=(js[0] if js else cpa), f1=(js[-1] if js else cpa), cpa=cpa, dmin=float(r["phys_cpa_h_m"]),
                         tier=r["tier"], name=r["class"], lateral=float(r["lateral_m"]), all_lost=int(r["all_lost"])))
    return dist, warn


def clip_stats(pred, meta: Path, clip: str, minframe: int) -> dict:
    """1 クリップ分の集計（数える・リストに溜める）。帯や全体はこれを足し合わせるだけ。"""
    s = dict(n_pred=0, n_gt=0, crit=[0, 0], caut=[0, 0], safe=[0, 0], n_strong=0, leads=[], n_fire=0, n_fire_nogt=0,
             n_close_gt=0, n_close_det=0, n_close_cap=0, n_safe_pairs=0, n_safe_fp=0, rel=[], close_est=[], n_pairs=0)
    frames = pred.get(clip, {})
    if frames:
        s["n_pred"] = 1
    gt = load_gt(meta, clip)
    if gt:
        s["n_gt"] = 1      # 正解が 1 行でもある本（敵対レビュー D・R2: 「予測あり」と「正解あり」は別物）
    # --- 通知（v4.3＋hold・方位帰属） ---
    if EVENTS is None:
        evs = [DG.mk_event(c, t, fr) for c, t, fr in DG.gt_tracks(meta, clip)]
        wevs = []
    else:                                                     # 本体 §2-5: 物理の通過の台帳（事前の除外のあと・全消失も分母）
        evs, wevs = ledger_events(meta, clip)
        s.update(rel_b=[], gd_b=[], warn=[], bike=[])
    if RULE_START > 0:                                        # §9-4 の 1: 通知の規則は frame RULE_START 以降の予測だけ（名札の記憶もそこから）
        frames = {k: v for k, v in frames.items() if k >= RULE_START}
    fires = fires_of(frames) if frames else []
    s["n_fire"] = len(fires)
    if not gt:
        s["n_fire_nogt"] = sum(1 for f in fires if f[2] != "警告")      # 距離クラスの正解が無い本での車／キック／バイクの発火（警告音の発火は除く。監査 A・F9）
    att = A.attribute(fires, evs) if fires else []
    got = defaultdict(set); first = {}
    for f, i in zip(fires, att):
        if i is not None and f[0] <= evs[i]["cpa"] + FPS:
            got[i].add(f[2]); first[i] = min(first.get(i, 10**9), f[0])
    for i, ev in enumerate(evs):
        hit = bool(got[i])
        if ev["tier"] == "safe":
            s["safe"][1] += 1; s["safe"][0] += int(not hit); continue
        key = "crit" if ev["tier"] == "critical" else "caut"
        s[key][1] += 1; s[key][0] += int(hit)
        if hit:
            s["leads"].append((ev["cpa"] - first[i]) / FPS)
        if ev["tier"] == "critical" and "強" in got[i]:
            s["n_strong"] += 1
    if EVENTS is not None:
        for i, ev in enumerate(evs):
            if ev["cls"] == 8:                                # M9: 自転車は lateral_m で層別（圏の列ではない）
                s["bike"].append((ev["lateral"], bool(got[i])))
        for ev in wevs:                                       # ⑭ 警告音の届いた行（同じクラスの警告の発火が窓 [鳴り始め − 1, ＋4] に入れば届いた）
            lo, hi = (ev["onset"] - WARN_WIN[0]) * FPS, (ev["onset"] + WARN_WIN[1]) * FPS
            ks = [f[0] for f in fires if f[2] == "警告" and f[4] == ev["cls"] and lo - 1e-9 <= f[0] <= hi + 1e-9]
            s["warn"].append((ev["name"], bool(ks), (min(ks) / FPS - ev["onset"]) if ks else None))
    # --- 距離（1 対 1・全 GT 分母） ---
    for k, gts in gt.items():
        if k < minframe:
            continue
        preds = [(c, az, d) for c, az, el, d in frames.get(k, []) if c in DIST_CLS and math.isfinite(d)]
        cand = []
        for gi, (gc, _t, gaz, gd) in enumerate(gts):
            for pi, (pc, paz, pd) in enumerate(preds):
                if pc == gc:
                    e = dang(paz, gaz)
                    if e <= AZ_PAIR:
                        cand.append((e, gi, pi))
        cand.sort()
        used_g, used_p, match = set(), set(), {}
        for e, gi, pi in cand:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi); match[gi] = pi
        for gi, (gc, _t, gaz, gd) in enumerate(gts):
            if gd <= TH_CLOSE:
                s["n_close_gt"] += 1
            if EVENTS is not None and gd <= TH_SAFE:
                s["gd_b"].append(gd)                          # 「いつも同じ値を出す行」の材料（帯の中の全正解の距離）
            if gi in match:
                pd = preds[match[gi]][2]
                s["n_pairs"] += 1
                s["rel"].append(abs(pd - gd) / max(gd, 0.1))
                if EVENTS is not None:
                    s["rel_b"].append((gd, abs(pd - gd) / max(gd, 0.1)))
                if gd <= TH_CLOSE:
                    s["n_close_det"] += 1; s["close_est"].append(pd)
                    if pd <= TH_CLOSE:
                        s["n_close_cap"] += 1
                elif gd > TH_SAFE:
                    s["n_safe_pairs"] += 1
                    if pd <= TH_CLOSE:
                        s["n_safe_fp"] += 1
    return s


def N_(s: dict, key: str):
    return s.get(key, "?")


def aggregate(stats: list) -> dict:
    T = lambda key, idx: sum(x[key][idx] for x in stats)
    N = lambda key: sum(x.get(key, 0) for x in stats)      # 掃引など別の道具が作る stats には新しい鍵（n_gt・n_fire_nogt）が無いので 0 扱い
    leads = np.array([v for x in stats for v in x["leads"]]) if any(x["leads"] for x in stats) else np.array([np.nan])
    rel = [v for x in stats for v in x["rel"]]
    close_est = [v for x in stats for v in x["close_est"]]
    n_crit, n_caut, n_safe = T("crit", 1), T("caut", 1), T("safe", 1)
    return dict(
        crit=100 * T("crit", 0) / max(n_crit, 1), strong=100 * N("n_strong") / max(n_crit, 1),
        caut=100 * T("caut", 0) / max(n_caut, 1), safe=100 * T("safe", 0) / max(n_safe, 1),
        n_crit=n_crit, n_caut=n_caut, n_safe=n_safe,
        lead=(float(np.nanmedian(leads)) if np.isfinite(leads).any() else float("nan")),
        lead25=(float(100 * np.nanmean(leads >= 2.5)) if np.isfinite(leads).any() else float("nan")),   # 旧（知らせが出た相手だけの分母・2.5 s）。表には出さない（#70 r7）
        # #70 r8／r12／r14（2026-10-19）: 「設定した 2 秒の基準を満たす早さ」= 最接近の 2.0 秒以上前に最初の知らせ（強さを問わない）が出た事象 ÷ 評価できる至近・注意の事象全体
        #   見逃し（知らせなし）は「間に合わなかった」に数える・2.0 ちょうどは成功・E = 0 なら未定義。leads は知らせが出た事象の値（first = 最初の発火）
        lead20=(float(100 * np.sum(leads[np.isfinite(leads)] >= 2.0) / (n_crit + n_caut)) if (n_crit + n_caut) > 0 else float("nan")),
        n_lead_ev=n_crit + n_caut, n_fire=N("n_fire"),
        det_close=100 * N("n_close_det") / max(N("n_close_gt"), 1), cap_all=100 * N("n_close_cap") / max(N("n_close_gt"), 1),
        cap_cond=100 * N("n_close_cap") / max(N("n_close_det"), 1), n_close_gt=N("n_close_gt"),
        # 丸め前の判定用の整数（依頼 3-3・2026-09-14）。表示は変えない
        n_strong=N("n_strong"), n_crit_hit=T("crit", 0), n_close_det=N("n_close_det"), n_close_cap=N("n_close_cap"),
        n_caut_hit=T("caut", 0), n_safe_quiet=T("safe", 0), n_gt_clips=N("n_gt"), n_fire_nogt=N("n_fire_nogt"),   # 注意・安全の分子分母と正解のある本数（敵対レビュー D・R1/R2・2026-10-01）
        fp=100 * N("n_safe_fp") / max(N("n_safe_pairs"), 1), n_safe_fp=N("n_safe_fp"), n_safe_pairs=N("n_safe_pairs"), close_med=float(np.median(close_est)) if close_est else float("nan"),
        dist_err=float(100 * np.median(rel)) if rel else float("nan"), n_pairs=N("n_pairs"),
        n_clips=len(stats), n_pred_clips=N("n_pred"))


HEADER = ("| 予測 | 本数(予測あり) | 至近到達 | **強到達** | 注意到達 | 安全抑制 | リード中央 | ≥2.0s前(全事象) | 発火数 "
          "| 検出率(GT≤1.5m) | **至近捕捉(全GT)** | 至近捕捉(条件付) | 誤捕捉 | 至近推定距離 | **距離誤差(1対1)** |")
SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def fmt(label, s):
    return (f"| {label} | {s['n_clips']:,}({s['n_pred_clips']:,}) | {s['crit']:.1f}% | **{s['strong']:.1f}%** | {s['caut']:.1f}% "
            f"| {s['safe']:.1f}% | {s['lead']:.2f}s | {s['lead20']:.1f}% | {s['n_fire']:,} "
            f"| {s['det_close']:.1f}% | **{s['cap_all']:.1f}%** | {s['cap_cond']:.1f}% | {s['fp']:.2f}% | {s['close_med']:.2f}m "
            f"| **{s['dist_err']:.1f}%** ({s['n_pairs']:,}) |")


MISSING = []      # 未着だった予測のラベル（--strict で終了コード 2・監査 F4）


HEADER_V30 = ("| 予測 | 本数(予測あり) | 至近到達 | **強到達** | 注意到達 | 安全抑制 | リード中央 | 発火数 "
              "| 検出率(GT≤1.5m) | **至近捕捉(全GT)** | 至近捕捉(条件付) | 誤捕捉 | 至近推定距離 | 距離誤差(1対1・全帯) |")
SEP_V30 = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def fmt_v30(label, s):
    """--events の主表（⑯: 「2.0 秒以上前」の列は参考の節へ）。"""
    return (f"| {label} | {s['n_clips']:,}({s['n_pred_clips']:,}) | {s['crit']:.1f}% | **{s['strong']:.1f}%** | {s['caut']:.1f}% "
            f"| {s['safe']:.1f}% | {s['lead']:.2f}s | {s['n_fire']:,} "
            f"| {s['det_close']:.1f}% | **{s['cap_all']:.1f}%** | {s['cap_cond']:.1f}% | {s['fp']:.2f}% | {s['close_med']:.2f}m "
            f"| {s['dist_err']:.1f}% ({s['n_pairs']:,}) |")


def v30_tables(label: str, stats: list) -> dict:
    """--events の追加の表（⑬ 帯ごとの距離誤差と定数の行・⑭ 警告音の届いた行・M9 自転車の横距離別）の行。"""
    rb = [x for st in stats for x in st.get("rel_b", [])]
    gd = np.array([x for st in stats for x in st.get("gd_b", [])])

    def med(sel):
        v = [r for g, r in rb if sel(g)]
        return (float(100 * np.median(v)) if v else float("nan")), len(v)

    def const(c, sel):
        g = gd[[sel(x) for x in gd]] if len(gd) else gd
        return float(100 * np.median(np.abs(c - g) / np.maximum(g, 0.1))) if len(g) else float("nan")
    close, mid, comb = (lambda g: g <= TH_CLOSE), (lambda g: TH_CLOSE < g <= TH_SAFE), (lambda g: g <= TH_SAFE)
    mid_c = float(np.median(gd[(gd > TH_CLOSE) & (gd <= TH_SAFE)])) if len(gd) and np.any((gd > TH_CLOSE) & (gd <= TH_SAFE)) else float("nan")
    comb_c = float(np.median(gd)) if len(gd) else float("nan")
    (ec, nc), (em, nm), (eb, nb) = med(close), med(mid), med(comb)
    dist_row = (f"| {label} | {ec:.1f}% ({nc:,}) | {em:.1f}% ({nm:,}) | **{eb:.1f}%** ({nb:,}) | {const(CONST_CLOSE_M, close):.1f}%（{CONST_CLOSE_M:g} m） "
                f"| {const(mid_c, mid):.1f}%（{mid_c:.2f} m） | {const(comb_c, comb):.1f}%（{comb_c:.2f} m） |")
    warn = defaultdict(list)
    for st in stats:
        for name, ok, dl in st.get("warn", []):
            warn[name].append((ok, dl))
    warn_rows = []
    for name in sorted(warn):
        v = warn[name]; dls = [d for ok, d in v if ok and d is not None]
        warn_rows.append(f"| {label} | {name} | {sum(ok for ok, _ in v)}/{len(v)} | {np.median(dls):.2f}s | {np.percentile(dls, 90):.2f}s |" if dls
                         else f"| {label} | {name} | 0/{len(v)} | — | — |")
    bike = [x for st in stats for x in st.get("bike", [])]
    bins = (("≤1.5 m", lambda x: x <= 1.5), ("1.5〜3.2 m", lambda x: 1.5 < x <= 3.2), ("3.2 m 超", lambda x: x > 3.2))
    bike_row = f"| {label} | " + " | ".join(f"{sum(ok for l_, ok in bike if f_(l_))}/{sum(1 for l_, _ in bike if f_(l_))}" for _, f_ in bins) + " |"
    return dict(dist=dist_row, warn=warn_rows, bike=bike_row)


def strict_check_pred(p: Path, pred, clips) -> None:
    """--strict の中身の検査（依頼 2-5・Astra R1・2026-09-14）。2 つの集合を**別々に**見る:
      (1) 評価 manifest（plan で選んだ clips）⊆ 推論 manifest（csv の隣の .run_meta.json の clips）
      (2) csv に現れた予測の clip ⊆ 推論 manifest（未知 ID を拒否）
    4,500 本の予測から 300 本を採点する（評価 ⊂ 推論）のは正当なので、「予測 ⊆ 評価 manifest」は要求しない。
    空 csv は、完了記録（.done.json）が n_rows=0 で全 clip 処理済みと示すときだけ「無検出」として通し、未完了の空 csv は止める。
    .run_meta.json が無い旧い csv は推論 manifest が分からないので、評価 manifest に 1 本も合わなければ止める（それ以外は注記だけ）。"""
    meta_p = p.with_suffix(".run_meta.json")
    if not meta_p.is_file() and (p.parent / "run_meta.json").is_file():
        meta_p = p.parent / "run_meta.json"      # サーバの推論フォルダの置き方（<run>/run_meta.json）も読む。Opus 5.5 監査 B06（2026-09-24）: 手元の <stem>.run_meta.json だけ探していたので、サーバでは推論 manifest の検査が素通りしていた
    done_p = p.with_suffix(".done.json")
    if not done_p.is_file() and (p.parent / "val_all_causal.done.json").is_file() and p.name == "val_all_causal.csv":
        done_p = p.parent / "val_all_causal.done.json"
    observed = set(pred.keys())
    if not observed:
        if not done_p.is_file():
            raise ValueError(f"{p.name}: 空の予測 csv だが完了記録（{done_p.name}）が無い（未完了の空 csv）— --strict")
        d = json.loads(done_p.read_text(encoding="utf-8"))
        if d.get("n_rows") != 0 or not d.get("n_clips"):
            raise ValueError(f"{p.name}: 空の予測 csv と完了記録が食い違う（n_rows={d.get('n_rows')}, n_clips={d.get('n_clips')}）— --strict")
    # 2026-09-22 改名（監査 9/22 R01）: 評価 manifest・推論 manifest・予測の ID を同じ規約（新名）に正規化してから比べる。
    # 旧名と新名が同じ本を指して混在する推論 manifest（衝突）と、同じ ID の重複は拒む。生データも run_meta も書き換えない。
    import sys as _sys; _sys.path.insert(0, str(ROOT / "src"))
    from collections import Counter as _Counter
    from outdoor_seld.clip_ids import normalize as _norm
    clips_n = [_norm(c) for c in clips]
    if meta_p.is_file():
        raw = [Path(c).stem for c in json.loads(meta_p.read_text(encoding="utf-8"))["clips"]]
        inf_list = [_norm(c) for c in raw]
        dup = sorted(c for c, k in _Counter(inf_list).items() if k > 1)
        if dup:
            raise ValueError(f"{p.name}: 推論 manifest に同じ本を指す ID が重複（旧名と新名の衝突か重複・{len(dup)} 本・例 {dup[:3]}）— --strict")
        inf = set(inf_list)
        miss = [c for c in clips_n if c not in inf]
        if miss:
            raise ValueError(f"{p.name}: 評価 manifest の {len(miss)} 本が推論 manifest に無い（例 {miss[:3]}）— --strict")
        unknown = sorted(observed - inf)
        if unknown:
            raise ValueError(f"{p.name}: 推論 manifest に無い clip の予測が {len(unknown)} 本（例 {unknown[:3]}）— --strict")
    elif observed and not (observed & set(clips_n)):
        raise ValueError(f"{p.name}: 評価 manifest に 1 本も合わない予測（run_meta 無し）— --strict")
    elif not meta_p.is_file():
        print(f"note: {p.name} に run_meta.json が無い（<stem>.run_meta.json も同じフォルダの run_meta.json も無い。推論 manifest の検査は省略。2026-09-08 より前の推論か、run_meta を写していない）", file=sys.stderr)


def main() -> int:
    MISSING.clear()
    argv = list(sys.argv[1:])
    strict = "--strict" in argv
    if strict:
        argv.remove("--strict")
    t3_sens = "--no-t3-sensitivity" not in argv   # 変更票 #69 r20（B11・2026-10-20）: 至近の境目 1.3／1.7 m の感度を既定で並べる（掃引や試験では --no-t3-sensitivity で省く）
    if not t3_sens:
        argv.remove("--no-t3-sensitivity")
    raw = "--raw" in argv            # 依頼 3-3: 丸め前の値（分子/分母・小数 4 桁）を末尾に足す。無ければ出力は従来どおり
    if raw:
        argv.remove("--raw")
    split_col = arg(argv, "--split-by", "")   # 依頼 3-1: plan の列（例 motion）で層別した行を末尾に足す。無ければ出力は従来どおり
    mode = arg(argv, "--mode", "")            # 2 設定（2026-09-16・変更票 #32）: 控えめ（既定と同一）／厳格。規則は同じで距離の閾値だけ変わる
    # 変更票 #37（2026-09-23）で 2 設定は撤回。--mode は掃引の道具のためだけに残し、既定と違う設定は NOTIFY_MODE_SWEEP=1 が無ければ拒む
    # （置き忘れで本番の採点が静かに変わるのを防ぐ・帰属の許容の NOTIFY_ATTRIB_SWEEP と同じ形。Opus 5.5 監査 9/23 F03）
    global C43, WARN_RULE
    if mode:
        import os as _os
        if mode != "控えめ" and _os.environ.get("NOTIFY_MODE_SWEEP") != "1":
            raise RuntimeError(f"--mode {mode} は撤回した設定（変更票 #37）。掃引で使うときだけ NOTIFY_MODE_SWEEP=1 を置く。本番の採点では --mode を渡さない")
        C43 = V43.apply_mode(mode)
    wr = arg(argv, "--warn-rule", WARN_RULE)     # 変更票 #41: 既定 v9c。旧 hold は併記用（--warn-rule hold）
    if wr not in WARN_RULES:
        raise ValueError(f"--warn-rule は {WARN_RULES} のどれか: {wr}")
    WARN_RULE = wr
    global NOTIFY_RULE
    nr = arg(argv, "--notify-rule", NOTIFY_RULE)   # 変更票 #73（Q2）: v43（既定）か detect（検出したら知らせる・比較用）
    if nr not in NOTIFY_RULES:
        raise ValueError(f"--notify-rule は {NOTIFY_RULES} のどれか: {nr}")
    NOTIFY_RULE = nr
    plan = ROOT / arg(argv, "--plan")
    import sys as _sys; _sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import normalize as _normalize_id, number as _clip_number, split_to_new as _split_to_new   # 2026-09-22 改名（旧 fold2 = val）
    split = _split_to_new(arg(argv, "--split", "val"))
    clip_max = int(arg(argv, "--clip-max", "0"))
    meta = ROOT / arg(argv, "--meta")
    minframe = int(arg(argv, "--minframe", "40"))
    bands_s = arg(argv, "--bands", "")
    out_conf = arg(argv, "--confusion", "")     # 2026-10-27（本人 12:35 (a)(b)）: 取り違えの表と本ごとの種類の正答率を別の json と .md に（参考・主の md は変えない）
    conf = {}
    global EVENTS, RULE_START
    ev_path = arg(argv, "--events", "")                   # 本体 §2-5: v30 の事象台帳（無ければ旧の連続ラン）
    rs = arg(argv, "--rule-start", "")
    ev_summary = None
    if ev_path:
        EVENTS = defaultdict(list)
        with open(ROOT / ev_path if not Path(ev_path).is_absolute() else Path(ev_path), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                EVENTS[r["clip"]].append(r)
        RULE_START = int(rs) if rs else 40                   # §9-4 の 1: 既定は frame 40 から（--rule-start で明示もできる）
    elif rs:
        RULE_START = int(rs)
    global DIST_TABLE, DIST_TABLE_MAX, LAT_GATE
    lg = arg(argv, "--lat-gate", "")
    if lg:
        p = lg.split(",")
        LAT_GATE = (p[0], float(p[1]), int(p[2]), None if len(p) < 4 or p[3] in ("-", "") else float(p[3]))
    dt = arg(argv, "--dist-table", "")
    DIST_TABLE_MAX = float(arg(argv, "--dist-table-max", "5.0"))
    global DIST_TABLE_FOR
    dtf = arg(argv, "--dist-table-for", "")
    if dt:
        if not dtf:
            raise ValueError("--dist-table には --dist-table-for <label,...>（表を掛ける入力の名前・例 v2final）が要る。v1 とオラクルには掛けない（Astra R72-01・2026-10-20）")
        DIST_TABLE_FOR = {x.strip() for x in dtf.split(",") if x.strip()}
        t = json.loads((ROOT / dt if not Path(dt).is_absolute() else Path(dt)).read_text(encoding="utf-8"))
        DIST_TABLE = (np.array(t["centers"], dtype=float), np.array(t["values"], dtype=float))
    elif dtf:
        raise ValueError("--dist-table-for は --dist-table と一緒に渡す")
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]

    rows = []
    with open(plan, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if _split_to_new(r.get("split", split)) != split:
                continue
            if clip_max and _clip_number(r["clip_id"]) > clip_max:
                continue
            r["clip_id"] = _normalize_id(r["clip_id"])      # 評価 manifest も新名に正規化してから一意性・GT 参照・集計に使う（再監査 9/22 R01）
            rows.append(r)
    clips = [r["clip_id"] for r in rows]
    if not strict and len(set(clips)) != len(clips):
        # --strict なしでも、同じ本（旧名と新名の混在か重複）を帯・層別の表で 2 回数えない（主表は辞書で 1 回だった。Opus 5.5 監査 9/23 F15）
        print(f"note: 評価 manifest に同じ本が {len(clips) - len(set(clips))} 件重複（旧名と新名の混在か重複）→ 1 本として数える（--strict なら拒否）", file=sys.stderr)
        clips = list(dict.fromkeys(clips))
    if strict:
        if not clips or not items:
            raise ValueError("nonempty unique manifest and prediction inputs required")
        if len(set(clips)) != len(clips):
            from collections import Counter as _Ctr
            dup = sorted(c for c, k in _Ctr(clips).items() if k > 1)
            raise ValueError(f"評価 manifest に同じ本が重複（旧名と新名の混在か重複・{len(dup)} 本・例 {dup[:3]}）— --strict")
        missing_gt = [c for c in clips if not (meta / f"{c}.csv").is_file()]
        if missing_gt:
            raise FileNotFoundError(f"missing GT ({len(missing_gt)}): {missing_gt[:3]}")
    mic_z = {r["clip_id"]: float(r["mic_z"]) for r in rows if r.get("mic_z")}
    bands = [float(x) for x in bands_s.split(",")] if bands_s else []

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    warn_desc = {"v9c": f"v9c 名札方式（門 {WARN_GATE_DEG:g}°・同一方位 {WARN_DEDUP_DEG:g}° を 1 つに・名札ごとに 1 回・寿命 5 s・変更票 #41）",
                 "hold": "hold（旧・前回検出から 5 s・アンカー方位 ±45°・併記用）"}[WARN_RULE]
    R = [f"# 統一採点（{'v4.3' if NOTIFY_RULE == 'v43' else '規則 = 検出したら知らせる（距離を見ない・比較用・#73 Q2）'}＋警告音 {WARN_RULE}・方位帰属・manifest 固定・1対1・全GT分母） — {out_md.stem}", "",
         f"plan= {_rel(plan)} ({split}{f', mix≤{clip_max}' if clip_max else ''}) = {len(clips):,} 本 / GT= {_rel(meta)} / "
         f"frame≥{minframe} / v4.3 = {V43.label43(C43)} / 警告音 = {warn_desc}" + (f" / 距離の補正表 {dt}（掛ける入力 = {'・'.join(sorted(DIST_TABLE_FOR))} だけ・ほかの入力とオラクルは表なし・予測 ≤{DIST_TABLE_MAX:g} m）" if dt else "") + (f" / v4.7 横距離の門 {lg}" if lg else "")
         + (f" / **--mode {mode}（撤回した 2 設定の掃引・本番の値ではない・変更票 #37）**" if mode and mode != "控えめ" else ""), "",   # 見出しに mode を写す（Opus 5.5 監査 9/23 F03）
         "本数(予測あり)= manifest の本数（うち予測ファイルにあった本数。無い本は発火ゼロとして採点）。到達・抑制= 車ごとの方位帰属"
         + (lambda s: f"（±{s['t_tol_s']:.1f} s・≤{s['az_tol_deg']:g}°{'' if s['default'] else '・**掃引の設定（既定ではない）**'}）。")(A.settings()),   # 実効値を写す（監査 M05/R03）
         "検出率= GT≤1.5 m のフレームのうち 1 対 1 で対応した割合。至近捕捉(全GT)= GT≤1.5 m のうち対応した予測が ≤1.5 m の割合（未検出は不捕捉）。",
         "至近捕捉(条件付)= 旧定義相当（対応したペアだけが分母）。誤捕捉= GT>3.2 m の対応ペアで予測 ≤1.5 m。距離誤差= 1 対 1 ペアの相対誤差中央値(ペア数)。", ""]
    if EVENTS is not None:                                   # 本体 §2-5: 台帳の数（除外前・理由・分母・全消失）を報告の頭に
        ev_rows = [r for c in clips for r in EVENTS.get(c, [])]
        by = defaultdict(lambda: [0, defaultdict(int), 0, 0])
        for r in ev_rows:
            b = by[r["class"]]; b[0] += 1
            if r["excluded"]:
                b[1][r["excluded"]] += 1
            else:
                b[2] += 1; b[3] += int(r["all_lost"])
        R += [f"**事象 = 物理の通過の台帳（{_rel(Path(ev_path))}・class・track ごとに 1 事象・本体 §2-5）**。通知の規則は frame {RULE_START} 以降の予測だけ。"
              "分母 = 事前に定めた除外（通過・発進は物理の最接近 < 4.0 s・警告音は鳴り終わり ≤ 4.0 s）のあとの物理の事象すべて（ラベル全消失も分母・未通知に数える）。",
              "", "| クラス | 物理の事象 | 除外（理由） | 分母 | うち全消失 |", "| --- | --- | --- | --- | --- |"]
        R += [f"| {k} | {v[0]} | {'・'.join(f'{a} {b}' for a, b in sorted(v[1].items())) or '0'} | {v[2]} | {v[3]} |" for k, v in sorted(by.items())] + [""]
    print("\n".join(R[:3]), flush=True)
    R += [HEADER_V30, SEP_V30] if EVENTS is not None else [HEADER, SEP]
    v30_rows = []
    per_band_rows = []
    per_split_rows, raw_rows = [], []
    t3_rows = []
    groups = {r["clip_id"]: (r.get(split_col) or "") for r in rows} if split_col else {}
    for label, src in items:
        horiz = src.endswith("@h")
        p = Path(src[:-2] if horiz else src)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            R.append(f"| {label} | （未着: {p.name}） |"); print(R[-1], flush=True); MISSING.append(label); continue
        pred = pred_for(label, p, horiz, strict)                 # R72-01: 表は --dist-table-for の入力だけ
        if strict:
            strict_check_pred(p, pred, clips)
        per_clip = {c: clip_stats(pred, meta, c, minframe) for c in clips}
        if out_conf:
            CC = _load("class_confusion", "_class_confusion.py")
            conf[label] = CC.score_set(clips, meta, pred, minframe, ledger=(dict(EVENTS) if EVENTS is not None else None))
        s = aggregate(list(per_clip.values()))
        R.append((fmt_v30 if EVENTS is not None else fmt)(label + (" (予測を水平変換)" if horiz else ""), s)); print(R[-1], flush=True)
        if EVENTS is not None:
            v30_rows.append((label, s, v30_tables(label, list(per_clip.values()))))
        if t3_sens:
            # #69 r20: GT 区分の至近の境目（DG.mk_event の v4.T3 = 1.5）を 1.3／1.7 にして数え直す（主の値は上の表のまま・距離の指標は変えない）
            def _t3row(t3_, s_):
                return (f"| {t3_:.1f} m{'（主）' if t3_ == 1.5 else ''} | {label} | {s_['n_crit_hit']}/{s_['n_crit']} | {s_['n_strong']}/{s_['n_crit']} "
                        f"| {s_['n_caut_hit']}/{s_['n_caut']} | {s_['n_safe_quiet']}/{s_['n_safe']} |")
            for t3_ in (1.3, 1.5, 1.7):
                if t3_ == 1.5:
                    t3_rows.append(_t3row(t3_, s)); continue
                _old = DG.v4.T3; DG.v4.T3 = t3_
                try:
                    s_t3 = aggregate([clip_stats(pred, meta, c, minframe) for c in clips])
                finally:
                    DG.v4.T3 = _old
                t3_rows.append(_t3row(t3_, s_t3))
        for i in range(len(bands) - 1):
            lo, hi = bands[i], bands[i + 1]
            sub = [c for c in clips if c in mic_z and (lo <= mic_z[c] < hi or (i == len(bands) - 2 and mic_z[c] == hi))]
            if not sub:
                continue
            per_band_rows.append(fmt(f"{label} {lo}〜{hi} m", aggregate([per_clip[c] for c in sub]))); print(per_band_rows[-1], flush=True)
        if split_col:
            for val in sorted({groups.get(c, "") for c in clips}):
                sub = [c for c in clips if groups.get(c, "") == val]
                per_split_rows.append(fmt(f"{label} {split_col}={val or '(空)'}", aggregate([per_clip[c] for c in sub]))); print(per_split_rows[-1], flush=True)
        if raw:
            raw_rows.append(f"| {label} | {s['n_strong']}/{s['n_crit']} = {s['strong']:.4f}% | {s['n_crit_hit']}/{s['n_crit']} = {s['crit']:.4f}% "
                            f"| {s['n_close_cap']}/{s['n_close_gt']} = {s['cap_all']:.4f}% | {s['n_close_det']}/{s['n_close_gt']} = {s['det_close']:.4f}% "
                            f"| {s['dist_err']:.4f}% | {s['n_pairs']:,} | {N_(s, 'n_safe_fp')}/{N_(s, 'n_safe_pairs')} = {s['fp']:.4f}% "
                            f"| {s['n_caut_hit']}/{s['n_caut']} = {s['caut']:.4f}% | {s['n_safe_quiet']}/{s['n_safe']} = {s['safe']:.4f}% "
                            f"| {s['n_gt_clips']}/{s['n_pred_clips']}/{s['n_clips']} | {s['n_fire_nogt']} |")
    if v30_rows:
        R += ["", "## 距離の相対誤差の帯（⑬・選択器 2.2 と同じ帯・1 対 1 の対応・「いつも同じ値を出す行」は帯の中の全正解に同じ値を出したときの値）", "",
              "| 予測 | 至近 ≤1.5 m | 注意 1.5〜3.2 m | **合わせ ≤3.2 m** | 定数（至近） | 定数（注意） | 定数（合わせ） |", "| --- | --- | --- | --- | --- | --- | --- |"]
        R += [t["dist"] for _, _, t in v30_rows]
        R += ["", "## 警告音の届いた行（⑭・同じクラスの警告の発火が [採点上の鳴り始め − 1, ＋4] に入れば届いた・遅れは採点上の鳴り始めから・遅い順 1 割目 = 90 パーセンタイル）", "",
              "| 予測 | クラス | 届いた/事象 | 遅れの中央値 | 遅い順 1 割目 |", "| --- | --- | --- | --- | --- |"]
        R += [w for _, _, t in v30_rows for w in t["warn"]]
        R += ["", "## 自転車の横距離別（M9・scene の lateral_m・知らせが出た/事象）", "", "| 予測 | ≤1.5 m | 1.5〜3.2 m | 3.2 m 超 |", "| --- | --- | --- | --- |"]
        R += [t["bike"] for _, _, t in v30_rows]
        R += ["", "## 参考（⑯・主の表から外した）: 最接近の 2.0 秒以上前に最初の知らせが出た事象の割合（分母 = 評価できる至近・注意の事象全体）", ""]
        R += [f"- {lb}: {s_['lead20']:.1f}%（分母 {s_['n_lead_ev']}）" for lb, s_, _ in v30_rows]
    if per_band_rows:
        R += ["", "## 高さの帯ごと（plan の mic_z）", "", HEADER, SEP] + per_band_rows
    if per_split_rows:
        R += ["", f"## {split_col} ごと（plan の {split_col} 列・依頼 3-1）", "", HEADER, SEP] + per_split_rows
    if raw_rows:
        R += ["", "## 丸め前（判定用・依頼 3-3。判定は表示値ではなくここの整数と 4 桁で行う）", "",
              "| 予測 | 強到達 n_strong/n_crit | 至近到達 hit/n_crit | 至近捕捉(全GT) cap/n_gt | 検出率 det/n_gt | 距離誤差(1対1) | ペア | 誤捕捉 "
              "| 注意到達 hit/n_caut | 安全抑制 quiet/n_safe | 本数 正解あり/予測あり/名簿 | 距離クラスの正解が無い本での車クラスの発火 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + raw_rows
    if t3_rows:
        # 行の先頭は「境目」（| 1.3 m | 予測 | …）: 「| <予測の名前> |」で丸め前の行を拾う検査・道具（test_v2final_conf_scoring S7 など）と衝突しない
        R += ["", "## 至近の境目の感度（変更票 #69 r20・主は 1.5 m・区分の本数と到達・抑制の動きを見るだけ・結果に関係なく出す）", "",
              "| 境目 | 予測 | 至近到達 hit/n_crit | 強到達 n_strong/n_crit | 注意到達 hit/n_caut | 安全抑制 quiet/n_safe |", "| --- | --- | --- | --- | --- | --- |"] + t3_rows
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(R) + "\n", encoding="utf-8")
    print("->", out_md)
    if out_conf:
        _load("class_confusion", "_class_confusion.py").write_outputs(out_conf, conf)
        print("->", out_conf, "（取り違えの表と本ごとの種類の正答率・参考）")
    if strict and MISSING:
        print(f"未着の予測: {MISSING}（--strict → 終了コード 2）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
