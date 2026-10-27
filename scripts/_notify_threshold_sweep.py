# -*- coding: utf-8 -*-
"""通知層の「閾値を動かした」評価（2026-09-15・ゼミ相談で教員が求めた「閾値に依らない評価」と「厳しい／楽観的の 2 設定」）。

A. 閾値なし（イベント単位）
   正解の車ごとの区間（イベント = 統一採点器と同じ gt_tracks/mk_event）に、同じクラス・方位 ±30° の予測を帰属し、
   イベント開始から最接近の 1 秒後までの「予測距離の最小値」を点数にする（小さいほど「鳴らすべき」）。
   鳴らすべき = 重大（正解の最接近 ≤1.5 m）、鳴らすべきでない = 安全（>3.2 m）。注意（1.5〜3.2 m）は主表では除く。
   点数の閾値を掃引して PR 曲線の面積（AP）・ROC 曲線の面積（AUC）・F1 の最大値とその閾値・固定閾値での適合率と再現率を出す。
   補助として「重大＋注意 対 安全」も出す。予測が帰属しないイベントは点数 = ∞（どの閾値でも鳴らない）。
B. 実際の規則（v4.3＋hold）の距離閾値を一斉に s 倍
   強 1.5 m→1.5s・中 3.2 m→3.2s・予測最接近の強 1.3→1.3s・中 1.6→1.6s。s<1 = 楽観的（鳴らしにくい）、s>1 = 厳しい（鳴らしやすい）。
   採用は s=1（標準・主）。**「控えめ／厳格」という設定名は変更票 #37（2026-09-23）で撤回**した（9/15 の「控えめ ×0.8・厳格 ×1.2 を宣言値として固定」も、9/16 の「控えめ ×1.0・厳格 ×1.2」も無効）。
   表の札は倍率だけにし、設定名を付けない（Opus 5.5 監査 9/23 F02）。正解の危険度の定義（1.5/3.2 m）は動かさない。統一採点器と同じ帰属（車ごとの方位帰属）で 至近到達・強到達・注意到達・安全抑制・発火数・リードを出す。

使い方:
  python scripts/_notify_threshold_sweep.py <出力md> --plan <assignment.csv> [--split fold2] [--clip-max N] --meta <GTディレクトリ>
      [--scales 0.6,0.8,0.9,1.0,1.1,1.2,1.5,2.0] [--fig <png>] [--csv <曲線のcsv>] [--strict] <ラベル>=<pred.csv>[@h] ...
"""
from __future__ import annotations

import csv
import importlib.util
import math
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("score_unified", ROOT / "scripts/_score_unified.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FPS = int(S.FPS)
AZ_ATTR = 30.0
FIXED_TH = (1.0, 1.3, 1.5, 2.0, 2.5, 3.2)
CLASS_NAME = {4: "car", 6: "kick", 7: "bike"}
BASE = dict(T3=1.5, SUPP=3.2, cpa_strong=1.3, cpa_mid=1.6)


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def events_of(meta: Path, clip: str):
    return [S.DG.mk_event(c, t, fr) for c, t, fr in S.DG.gt_tracks(meta, clip)]


def event_score(ev, frames) -> float:
    """イベント開始〜最接近の 1 秒後までの、帰属した予測距離の最小値（帰属なし = inf）"""
    best = float("inf")
    last = min(ev["f1"], ev["cpa"] + FPS)
    for j in range(ev["f0"], last + 1):
        if j not in ev["fr"]:
            continue
        gaz = ev["fr"][j][0]
        for c, az, el, d in frames.get(j, []):
            if c == ev["cls"] and math.isfinite(d) and dang(az, gaz) <= AZ_ATTR and d < best:
                best = d
    return best


def curve_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """y = 1 なら鳴らすべき。score = 帰属した予測距離の最小値（小さいほど鳴らすべき）。
    Astra A2（2026-09-16）: 帰属する予測が無いイベント（score = inf）は**どの閾値でも鳴らせない**ので順位に入れない。
    鳴らすべき側の帰属なしは分母（再現率の分母）にだけ入れ、鳴らすべきでない側の帰属なしは常に正しく鳴らない（順位にも分母にも入れない）。
    AP = 到達可能な範囲（再現率の上限 < 1）の PR 曲線の面積（全点補間）。同点の閾値はまとめて処理する。参考として、帰属なしを最下位に置いた
    従来の値（sklearn・inf → 1e6）も ap_all/auc_all として残す。"""
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    reach = np.isfinite(score)
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    out = dict(n_pos=n_pos, n_neg=n_neg, n_pos_unreach=int(((y == 1) & ~reach).sum()), n_neg_unreach=int(((y == 0) & ~reach).sum()))
    if n_pos == 0 or n_neg == 0:
        return out
    # 従来の値（参考）
    dec_all = -np.where(reach, score, 1e6)
    out["ap_all"] = float(average_precision_score(y, dec_all)); out["auc_all"] = float(roc_auc_score(y, dec_all))
    # 到達可能な範囲の PR 曲線（同点をまとめる）
    ys, ss = y[reach], score[reach]
    order = np.argsort(ss, kind="stable"); ys, ss = ys[order], ss[order]
    uniq = np.unique(ss)
    tp_c = np.cumsum(ys); fp_c = np.cumsum(1 - ys)
    idx = np.searchsorted(ss, uniq, side="right") - 1        # 同点の最後の位置 = その閾値で鳴る全部
    tp = tp_c[idx]; fp = fp_c[idx]
    rec = tp / n_pos; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec]); mpre = np.concatenate([[1.0], prec])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    out["ap"] = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    out["recall_max"] = float(rec[-1]) if len(rec) else 0.0
    # ROC（到達可能な範囲。到達不能な正例は最後まで見つからない = TPR の上限 <1、到達不能な負例は FPR に入らない）
    fpr = fp / max(n_neg - out["n_neg_unreach"], 1); tpr = rec
    mf = np.concatenate([[0.0], fpr]); mt = np.concatenate([[0.0], tpr])
    out["auc"] = float(np.sum((mf[1:] - mf[:-1]) * (mt[1:] + mt[:-1]) / 2))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1)) if len(f1) else 0
    out["f1_max"] = float(f1[i]); out["f1_max_th"] = float(uniq[i]); out["f1_max_p"] = float(prec[i]); out["f1_max_r"] = float(rec[i])
    fixed = {}
    for th in FIXED_TH:
        pred = reach & (score <= th)
        tpn = int((pred & (y == 1)).sum()); fpn = int((pred & (y == 0)).sum()); fnn = int((~pred & (y == 1)).sum())
        pp = tpn / max(tpn + fpn, 1); rr = tpn / max(tpn + fnn, 1)
        fixed[th] = dict(p=100 * pp, r=100 * rr, f1=(2 * pp * rr / max(pp + rr, 1e-12)), tp=tpn, fp=fpn, fn=fnn)
    out["fixed"] = fixed
    out["_pr"] = (rec, prec); out["_roc"] = (fpr, tpr)
    return out


def notify_stats(clips, pred, evs_of, scale: float) -> dict:
    """規則の距離閾値を scale 倍して統一採点器と同じ帰属で数える（正解の危険度は evs_of で固定済み）"""
    v4r = S.V43.v4
    keep = (v4r.T3, v4r.SUPP, S.C43)
    v4r.T3, v4r.SUPP = BASE["T3"] * scale, BASE["SUPP"] * scale
    S.C43 = replace(S.C43, cpa_strong=BASE["cpa_strong"] * scale, cpa_mid=BASE["cpa_mid"] * scale)
    try:
        stats = []
        for c in clips:
            evs = evs_of[c]
            frames = pred.get(c, {})
            fires = S.fires_of(frames) if frames else []
            att = S.A.attribute(fires, evs) if fires else []
            got = defaultdict(set); first = {}
            for fr, i in zip(fires, att):
                if i is not None and fr[0] <= evs[i]["cpa"] + FPS:
                    got[i].add(fr[2]); first[i] = min(first.get(i, 10 ** 9), fr[0])
            s = dict(n_pred=int(bool(frames)), crit=[0, 0], caut=[0, 0], safe=[0, 0], n_strong=0, leads=[], n_fire=len(fires),
                     n_close_gt=0, n_close_det=0, n_close_cap=0, n_safe_pairs=0, n_safe_fp=0, rel=[], close_est=[], n_pairs=0)
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
            stats.append(s)
        return S.aggregate(stats)
    finally:
        v4r.T3, v4r.SUPP, S.C43 = keep


def pct(v):
    return "-" if v is None or not math.isfinite(v) else f"{v:.1f}%"


def main() -> int:
    argv = list(sys.argv[1:])
    strict = "--strict" in argv
    if strict:
        argv.remove("--strict")
    plan = ROOT / arg(argv, "--plan")
    import sys as _sys; _sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import normalize as _normalize_id, number as _clip_number, split_to_new as _split_to_new   # 2026-09-22 改名（旧 fold2 = val）
    split = _split_to_new(arg(argv, "--split", "val"))
    clip_max = int(arg(argv, "--clip-max", "0"))
    meta = ROOT / arg(argv, "--meta")
    scales = [float(x) for x in arg(argv, "--scales", "0.6,0.8,0.9,1.0,1.1,1.2,1.5,2.0").split(",")]
    fig = arg(argv, "--fig", "")
    out_csv = arg(argv, "--csv", "")
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]

    rows = []
    with open(plan, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if _split_to_new(r.get("split", split)) != split:
                continue
            if clip_max and _clip_number(r["clip_id"]) > clip_max:
                continue
            r["clip_id"] = _normalize_id(r["clip_id"])      # 評価 manifest を新名に正規化（統一採点器と同じ規約・Opus 5.5 監査 9/23 F01）
            rows.append(r)
    clips = [r["clip_id"] for r in rows]
    motion = {r["clip_id"]: r.get("motion", "") or "-" for r in rows}
    if not clips or not items:
        raise ValueError("nonempty unique manifest and prediction inputs required")
    if len(set(clips)) != len(clips):
        from collections import Counter as _Ctr
        dup = sorted(c for c, k in _Ctr(clips).items() if k > 1)
        raise ValueError(f"評価 manifest に同じ本が重複（旧名と新名の混在か重複・{len(dup)} 本・例 {dup[:3]}）")
    missing_gt = [c for c in clips if not (meta / f"{c}.csv").is_file()]
    if missing_gt:
        raise FileNotFoundError(f"missing GT ({len(missing_gt)}): {missing_gt[:3]}")
    # 正解の危険度はここで固定（規則の閾値を動かしても変わらない）
    assert S.DG.v4 is not S.V43.v4, "GT 側と規則側の定数モジュールが同一 — 掃引で正解の定義が動く"
    evs_of = {c: events_of(meta, c) for c in clips}
    n_tier = defaultdict(int)
    for evs in evs_of.values():
        for ev in evs:
            n_tier[ev["tier"]] += 1

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    lines = [f"# 通知層: 閾値なしの評価と閾値の掃引 — {out_md.stem}", "",
             f"plan= {_rel(plan)} ({split}{', mix≤' + str(clip_max) if clip_max else ''}) = {len(clips):,} 本 / GT= {_rel(meta)} / "
             f"イベント = 正解の車ごとの区間: 重大 {n_tier['critical']:,}・注意 {n_tier['caution']:,}・安全 {n_tier['safe']:,} / "
             f"採用の規則 = v4.3（{S.V43.label43(S.C43)}）＋hold", ""]
    A_rows, A_fixed_rows, A_sub_rows, B_rows = [], [], [], []
    curves, sweep_pts = {}, {}
    for label, spec in items:
        horiz = spec.endswith("@h")
        p = ROOT / (spec[:-2] if horiz else spec)
        if not p.is_file():
            raise FileNotFoundError(f"missing prediction: {p}")
        pred = S.load_pred_rows(p, horiz=horiz, strict=strict)
        if strict:
            S.strict_check_pred(p, pred, clips)     # 推論 manifest・未知 ID・「1 本も合わない」の検査を統一採点器と同じにする（Opus 5.5 監査 9/23 F01）
        # ---- A. 閾値なし ----
        recs = []   # (tier, cls, motion, score)
        for c in clips:
            frames = pred.get(c, {})
            for ev in evs_of[c]:
                recs.append((ev["tier"], ev["cls"], motion[c], event_score(ev, frames)))
        tiers = np.array([r[0] for r in recs]); clss = np.array([r[1] for r in recs]); mots = np.array([r[2] for r in recs])
        sc = np.array([r[3] for r in recs], dtype=float)

        def _sub(mask, pos_tiers, name):
            m = mask & np.isin(tiers, list(pos_tiers) + ["safe"])
            y = np.isin(tiers[m], list(pos_tiers)).astype(int)
            return name, curve_metrics(y, sc[m])

        main_name, cm = _sub(np.ones(len(recs), dtype=bool), ("critical",), "重大 対 安全")
        curves[label] = cm
        A_rows.append(f"| {label} | 全体 | {cm['n_pos']:,} / {cm['n_neg']:,}（帰属なし {cm['n_pos_unreach']:,} / {cm['n_neg_unreach']:,}） | {cm.get('ap', float('nan')):.3f}（再現率の上限 {100 * cm.get('recall_max', float('nan')):.1f}%） | {cm.get('auc', float('nan')):.3f} "
                      f"| {cm.get('f1_max', float('nan')):.3f}（閾値 {cm.get('f1_max_th', float('nan')):.2f} m・適合 {pct(100 * cm.get('f1_max_p', float('nan')))}・再現 {pct(100 * cm.get('f1_max_r', float('nan')))}） | {cm.get('ap_all', float('nan')):.3f} / {cm.get('auc_all', float('nan')):.3f} |")
        for cls in sorted(CLASS_NAME):
            _, m = _sub(clss == cls, ("critical",), "")
            if m.get("ap") is not None:
                A_rows.append(f"| {label} | {CLASS_NAME[cls]} | {m['n_pos']:,} / {m['n_neg']:,}（帰属なし {m['n_pos_unreach']:,} / {m['n_neg_unreach']:,}） | {m['ap']:.3f}（上限 {100 * m['recall_max']:.1f}%） | {m['auc']:.3f} "
                              f"| {m['f1_max']:.3f}（閾値 {m['f1_max_th']:.2f} m） | {m['ap_all']:.3f} / {m['auc_all']:.3f} |")
        for mo in sorted(set(mots)):
            _, m = _sub(mots == mo, ("critical",), "")
            if m.get("ap") is not None:
                A_rows.append(f"| {label} | motion={mo} | {m['n_pos']:,} / {m['n_neg']:,}（帰属なし {m['n_pos_unreach']:,} / {m['n_neg_unreach']:,}） | {m['ap']:.3f}（上限 {100 * m['recall_max']:.1f}%） | {m['auc']:.3f} "
                              f"| {m['f1_max']:.3f}（閾値 {m['f1_max_th']:.2f} m） | {m['ap_all']:.3f} / {m['auc_all']:.3f} |")
        _, sub = _sub(np.ones(len(recs), dtype=bool), ("critical", "caution"), "重大＋注意 対 安全")
        if sub.get("ap") is not None:
            A_sub_rows.append(f"| {label} | 重大＋注意 対 安全 | {sub['n_pos']:,} / {sub['n_neg']:,}（帰属なし {sub['n_pos_unreach']:,} / {sub['n_neg_unreach']:,}） | {sub['ap']:.3f}（上限 {100 * sub['recall_max']:.1f}%） | {sub['auc']:.3f} "
                              f"| {sub['f1_max']:.3f}（閾値 {sub['f1_max_th']:.2f} m） | {sub['ap_all']:.3f} / {sub['auc_all']:.3f} |")
        if cm.get("fixed"):
            for th, v in cm["fixed"].items():
                A_fixed_rows.append(f"| {label} | {th:.1f} m | {pct(v['p'])} | {pct(v['r'])} | {v['f1']:.3f} | {v['tp']:,} / {v['fp']:,} / {v['fn']:,} |")
        # ---- B. 規則の閾値を s 倍 ----
        pts = []
        for sc_ in scales:
            g = notify_stats(clips, pred, evs_of, sc_)
            # 札は倍率の向きだけ。「控えめ／厳格」「宣言値」の設定名は変更票 #37（2026-09-23）で撤回（Opus 5.5 監査 9/23 F02）
            tag = "標準（採用・主）" if abs(sc_ - 1.0) < 1e-9 else ("鳴らしやすい側（参考）" if sc_ > 1 else "鳴らしにくい側（参考）")
            B_rows.append(f"| {label} | ×{sc_:.1f}（強 {BASE['T3'] * sc_:.2f} m・中 {BASE['SUPP'] * sc_:.2f} m・最接近 {BASE['cpa_strong'] * sc_:.2f}/{BASE['cpa_mid'] * sc_:.2f} m） | {tag} "
                          f"| {g['crit']:.1f}% | {g['strong']:.1f}% | {g['caut']:.1f}% | {g['safe']:.1f}% | {g['lead']:.2f}s | {g['n_fire']:,} |")
            pts.append((sc_, g["crit"], g["strong"], g["caut"], g["safe"], g["n_fire"], g["lead"]))
        sweep_pts[label] = pts
        print(f"{label}: AP {cm.get('ap', float('nan')):.3f} AUC {cm.get('auc', float('nan')):.3f}; 掃引 {len(pts)} 点")

    lines += ["## A. 閾値なし（イベント単位・点数 = 帰属した予測距離の最小値）", "",
              "| 予測 | 層 | 鳴らすべき / べきでない（帰属なし） | AP（到達可能な範囲・再現率の上限） | AUC（到達可能な範囲） | F1 の最大値 | 参考: 帰属なしを最下位に置いた AP / AUC |",
              "| --- | --- | --- | --- | --- | --- | --- |"] + A_rows + [""]
    if A_sub_rows:
        lines += ["補助（注意も「鳴らすべき」に含める）:", "", "| 予測 | 層 | 鳴らすべき / べきでない（帰属なし） | AP（上限） | AUC | F1 の最大値 | 参考 AP / AUC |",
                  "| --- | --- | --- | --- | --- | --- | --- |"] + A_sub_rows + [""]
    lines += ["固定の閾値での値（重大 対 安全・点数 ≤ 閾値 なら「鳴らす」）:", "",
              "| 予測 | 閾値 | 適合率 | 再現率 | F1 | 当たり / 無駄鳴り / 見逃し |", "| --- | --- | --- | --- | --- | --- |"] + A_fixed_rows + [""]
    lines += ["## B. 実際の規則（v4.3＋hold）の距離閾値を一斉に s 倍（正解の危険度 1.5/3.2 m は固定）", "",
              "| 予測 | 閾値 | 設定 | 至近到達 | 強到達 | 注意到達 | 安全抑制 | リード中央 | 発火数 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + B_rows + [""]
    # 「宣言した 2 設定の差（控えめ ×0.8・厳格 ×1.2）」の表は、変更票 #37（2026-09-23）で 2 設定を撤回したので出さない（Opus 5.5 監査 9/23 F02）。
    # 倍率ごとの値は上の B 表にそのまま残る（差が要るなら B 表から引く）。
    lines += ["## 定義", "",
              "- イベント = 正解の (クラス, track) の連続区間。危険度 = 区間内の正解の最接近距離で 重大(≤1.5 m)／注意(≤3.2 m)／安全(>3.2 m)（統一採点器と同じ）",
              "- A の点数 = イベント開始〜最接近の 1 秒後までに、同じクラスで正解の方位から ±30° 以内にある予測の距離の最小値。帰属する予測が無ければ ∞（鳴らない）。"
              "暖機（frame<40）も含める（規則も frame 0 から動くため）",
              "- AP・AUC・F1 は**到達可能な範囲**で計算する（Astra A2・2026-09-16）: 帰属する予測が無いイベントはどの閾値でも鳴らせないので順位に入れず、"
              "鳴らすべき側の帰属なしは再現率の分母にだけ入れる（再現率の上限 < 1）。同点の閾値はまとめて処理。「参考」の列は帰属なしを最下位に置いた従来の値（sklearn）",
              "- B の「s 倍」= 規則の距離の閾値 4 つ（強 1.5・中 3.2・予測最接近の強 1.3・中 1.6 m）を同じ倍率で動かす。方位・時間の閾値と hold は動かさない。"
              "至近到達・強到達・注意到達・安全抑制・発火数・リードは統一採点器と同じ定義（車ごとの方位帰属・最接近の 1 秒後まで）"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "scale", "crit", "strong", "caut", "safe", "n_fire", "lead"])
            for label, pts in sweep_pts.items():
                for p in pts:
                    w.writerow([label] + [f"{v:.4f}" if isinstance(v, float) else v for v in p])
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
        fg, ax = plt.subplots(1, 3, figsize=(13, 4))
        for label, cm in curves.items():
            if "_pr" in cm:
                ax[0].plot(cm["_pr"][0], cm["_pr"][1], label=f"{label} AP={cm['ap']:.2f}")
                ax[1].plot(cm["_roc"][0], cm["_roc"][1], label=f"{label} AUC={cm['auc']:.2f}")
        ax[0].set_xlabel("再現率（重大を鳴らした割合）"); ax[0].set_ylabel("適合率（鳴らした中の重大）"); ax[0].set_title("PR 曲線（重大 対 安全）")
        ax[1].set_xlabel("偽陽性率（安全なのに鳴った）"); ax[1].set_ylabel("真陽性率（重大で鳴った）"); ax[1].set_title("ROC 曲線")
        for label, pts in sweep_pts.items():
            ax[2].plot([p[4] for p in pts], [p[1] for p in pts], "o-", label=label)
            for p in pts:
                ax[2].annotate(f"×{p[0]:.1f}", (p[4], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[2].set_xlabel("安全抑制（安全な車を鳴らさなかった割合 %）"); ax[2].set_ylabel("至近到達（重大な車に届いた割合 %）")
        ax[2].set_title("規則の閾値を s 倍したときの釣り合い")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fg.tight_layout(); fg.savefig(fig, dpi=150); plt.close(fg)
    print(f"-> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
