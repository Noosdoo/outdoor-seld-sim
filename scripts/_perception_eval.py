# -*- coding: utf-8 -*-
"""知覚層だけの精度評価（2026-09-15・ゼミ相談で教員が求めた「位置を当てる部分だけの評価」）。

通知の閾値を通さず、「音から相手の位置（クラス・方位・距離・動き）をどれだけ当てているか」を正解の座標と直接比べる。
  対応付け  フレームごと・クラスごとに正解と予測を 1 対 1 で対応付け（方位差の合計が最小になる組み合わせ・門なし）
  検出      再現率・適合率・F1。DCASE 流に「方位 20° 以内なら正解」とする版と、方位を問わない版
  方位誤差  対応した組の 中央値・平均・RMSE（二乗平均平方根＝大きいずれを重く見る）・90 百分位・10/20/30° 以内の割合
  距離誤差  距離クラス（car/kick/bike）の対応した組の 絶対誤差の中央値・平均・RMSE、相対誤差の中央値
  位置誤差  方位＋距離を平面の xy に直したときの正解との距離（中央値・平均・RMSE）
  動きの向き 1 秒窓（10 フレーム）で正解が接近／遠ざかり（|傾き| ≥ 0.3 m/s）のとき、予測の傾きの符号が一致する割合。
            方位の回り方（左へ／右へ・|傾き| ≥ 5°/s）も同様
  層別      クラス、正解の方向（前／左／右／後・±45° で 4 分割）、正解の距離帯、歩行／静止（plan の motion）、同クリップの距離クラスの台数（単独／多重）、暖機の扱い

使い方:
  python scripts/_perception_eval.py <出力md> --plan <assignment.csv> [--split fold2] [--clip-max N] --meta <GTディレクトリ>
      [--minframe 40] [--json <出力json>] [--strict] [--confusion <json>] <ラベル>=<pred.csv>[@h] ...
  --confusion（2026-10-27・本人 12:35 (a)(b)・scripts/_class_confusion.py）: 取り違えの表（フレーム・9×9＋なし）と本ごとの種類の正答率（参考）を
  別の json と同じ名前の .md に書く。主の md・--json は 1 バイトも変えない。
方位の規約 = az=atan2(y,x)・反時計回り（0=前・+90=左・±180=後・−90=右）。距離は水平距離（metadata_dist）。
予測の読み込み・距離クラス・水平化（@h）は統一採点器 _score_unified.py と同じ関数を使う。
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
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("score_unified", ROOT / "scripts/_score_unified.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FPS = 10
CLASS_NAME = {0: "siren", 1: "horn", 2: "backup_beep", 3: "bike_bell", 4: "car", 5: "crossing", 6: "kick", 7: "bike", 8: "bicycle"}   # 8 = bicycle（2026-10-08 変更票 #48・Astra D08-08）
DIST_CLS = S.DIST_CLS                      # {4, 6, 7}
AZ_TP = 20.0                               # DCASE 流の「方位 20° 以内で正解」
DIST_BANDS = [(0.0, 1.5, "≤1.5 m"), (1.5, 3.2, "1.5〜3.2 m"), (3.2, 10.0, "3.2〜10 m"), (10.0, 30.0, "10〜30 m"), (30.0, 1e9, ">30 m")]
SECTORS = ("前", "左", "後", "右")
WIN = 10                                   # 動きの向きの窓（1 秒）
MIN_MATCHED = 8                            # 窓の中で対応した組がこれ未満なら判定しない
V_FLOOR = 0.3                              # [m/s] 正解の接近／遠ざかりの床
AZ_RATE_FLOOR = 5.0                        # [deg/s] 正解の方位の回り方の床


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def sector(az: float) -> str:
    a = (az + 180.0) % 360.0 - 180.0
    if abs(a) <= 45.0:
        return "前"
    if abs(a) >= 135.0:
        return "後"
    return "左" if a > 0 else "右"


def band_of(d: float) -> str:
    """≤1.5 / 1.5〜3.2 / 3.2〜10 / 10〜30 / >30 m（上端を含む・危険度の定義 ≤1.5・≤3.2 と同じ切り方）"""
    if not math.isfinite(d):
        return DIST_BANDS[-1][2]
    if d <= DIST_BANDS[0][1]:
        return DIST_BANDS[0][2]
    for lo, hi, name in DIST_BANDS[1:]:
        if lo < d <= hi:
            return name
    return DIST_BANDS[-1][2]


def load_gt_all(meta: Path, clip: str):
    """frame -> [(cls, trk, az, el, d)]（全クラス。d は距離クラス以外 nan）"""
    f = meta / f"{clip}.csv"
    out = defaultdict(list)
    if not f.exists():
        return out
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) < 5:
            continue
        c = int(g[1])
        d = float(g[5]) if (len(g) >= 6 and c in DIST_CLS and g[5] not in ("", "nan")) else float("nan")
        out[int(g[0])].append((c, int(g[2]), float(g[3]), float(g[4]), d))
    return out


def match_frame(gts, preds):
    """同じクラスの正解と予測を方位差の合計が最小になるよう 1 対 1 に。戻り [(gi, pi, err_az)]"""
    if not gts or not preds:
        return []
    if len(gts) == 1 and len(preds) == 1:
        return [(0, 0, dang(preds[0][1], gts[0][2]))]
    cost = np.array([[dang(p[1], g[2]) for p in preds] for g in gts])
    gi, pi = linear_sum_assignment(cost)
    return [(int(a), int(b), float(cost[a, b])) for a, b in zip(gi, pi)]


def xy(az_deg: float, d: float):
    a = math.radians(az_deg)
    return d * math.cos(a), d * math.sin(a)


class Acc:
    """対応した組の記録（配列にして層別集計する）"""

    def __init__(self):
        self.rec = defaultdict(list)     # key -> list
        self.n_gt = defaultdict(int)     # stratum -> 正解フレーム数
        self.n_tp20 = defaultdict(int)   # stratum -> 方位 20° 以内で対応
        self.n_match = defaultdict(int)  # stratum -> 対応（方位を問わず）
        self.n_pred = defaultdict(int)   # cls -> 予測数
        self.n_fp20 = defaultdict(int)   # cls -> 予測のうち 20° 以内の対応が無い数
        self.n_unmatched_pred = defaultdict(int)
        self.motion = dict(app=[0, 0], rec=[0, 0], left=[0, 0], right=[0, 0])   # [一致, 窓数]

    def add_pair(self, strata, cls, gaz, gel, gd, paz, pel, pd, err):
        for st in strata:
            self.n_gt[st] += 1
            self.n_match[st] += 1
            if err <= AZ_TP:
                self.n_tp20[st] += 1
        r = self.rec
        r["cls"].append(cls); r["err_az"].append(err); r["err_el"].append(abs(pel - gel))
        r["gd"].append(gd); r["pd"].append(pd)
        if cls in DIST_CLS and math.isfinite(gd) and math.isfinite(pd):
            gx, gy = xy(gaz, gd); px, py = xy(paz, pd)
            r["err_d"].append(abs(pd - gd)); r["rel_d"].append(abs(pd - gd) / max(gd, 0.1))
            r["err_xy"].append(math.hypot(px - gx, py - gy))
        else:
            r["err_d"].append(float("nan")); r["rel_d"].append(float("nan")); r["err_xy"].append(float("nan"))
        r["strata"].append(tuple(strata))

    def add_miss(self, strata):
        for st in strata:
            self.n_gt[st] += 1


def strata_of(cls, gaz, gd, motion, warm, multi=""):
    st = ["全体", f"cls:{CLASS_NAME.get(cls, cls)}"]
    if cls in DIST_CLS:
        st += ["距離クラス", f"方向:{sector(gaz)}", f"距離帯:{band_of(gd)}", f"motion:{motion}"]
        if multi:
            st.append(f"台数:{multi}")
        if warm:
            st.append("距離クラス(暖機込み)")
    return st


def clip_eval(acc: Acc, acc0: Acc, frames, gt, motion: str, minframe: int):
    """1 クリップ。acc = frame ≥ minframe、acc0 = 全フレーム（暖機込み）"""
    tracks = defaultdict(dict)   # (cls, trk) -> frame -> (gaz, gd, paz, pd or None)
    n_dist_tracks = len({(g[0], g[1]) for k in gt for g in gt[k] if g[0] in DIST_CLS})
    multi = "多重" if n_dist_tracks >= 2 else "単独"   # 同じクリップに距離クラスの相手が 2 つ以上いるか
    for k in sorted(set(gt) | set(frames)):
        gts_all = gt.get(k, [])
        preds_all = [(c, az, el, d) for c, az, el, d in frames.get(k, [])]
        for cls in sorted(set(g[0] for g in gts_all) | set(p[0] for p in preds_all)):
            gts = [g for g in gts_all if g[0] == cls]
            preds = [p for p in preds_all if p[0] == cls]
            pairs = match_frame(gts, preds)
            used_g = {gi for gi, _, _ in pairs}
            used_p = {pi for _, pi, _ in pairs}
            for a in ((acc, k >= minframe), (acc0, True)):
                A, on = a
                if not on:
                    continue
                for gi, pi, err in pairs:
                    c, trk, gaz, gel, gd = gts[gi]
                    _c, paz, pel, pd = preds[pi]
                    A.add_pair(strata_of(c, gaz, gd, motion, A is acc0, multi), c, gaz, gel, gd, paz, pel, pd, err)
                for gi, g in enumerate(gts):
                    if gi not in used_g:
                        A.add_miss(strata_of(g[0], g[2], g[4], motion, A is acc0, multi))
                A.n_pred[cls] += len(preds)
                A.n_unmatched_pred[cls] += len(preds) - len(pairs)
                A.n_fp20[cls] += len(preds) - sum(1 for _, _, e in pairs if e <= AZ_TP)
            for gi, g in enumerate(gts):
                if g[0] in DIST_CLS:
                    pi = next((p for gg, p, _ in pairs if gg == gi), None)
                    pd = preds[pi][3] if pi is not None else None
                    paz = preds[pi][1] if pi is not None else None
                    tracks[(g[0], g[1])][k] = (g[2], g[4], paz, pd)
    # 動きの向き（1 秒窓・重ならない窓）
    for (cls, trk), fr in tracks.items():
        ks = sorted(fr)
        if not ks:
            continue
        for start in range(ks[0], ks[-1] - WIN + 2, WIN):
            win = [j for j in range(start, start + WIN) if j in fr]
            if len(win) < WIN:
                continue
            g_d = np.array([fr[j][1] for j in win]); t = np.arange(WIN) / FPS
            g_az = np.degrees(np.unwrap(np.radians([fr[j][0] for j in win])))
            got = [(j, fr[j][2], fr[j][3]) for j in win if fr[j][3] is not None and math.isfinite(fr[j][3])]
            if len(got) < MIN_MATCHED:
                continue
            tp = np.array([(j - start) / FPS for j, _, _ in got])
            p_d = np.array([d for _, _, d in got])
            p_az = np.degrees(np.unwrap(np.radians([a for _, a, _ in got])))
            gs, ps = np.polyfit(t, g_d, 1)[0], np.polyfit(tp, p_d, 1)[0]
            ga, pa = np.polyfit(t, g_az, 1)[0], np.polyfit(tp, p_az, 1)[0]
            for A in (acc, acc0):
                if A is acc and start < minframe:
                    continue
                if gs <= -V_FLOOR:
                    A.motion["app"][1] += 1; A.motion["app"][0] += int(ps < 0)
                elif gs >= V_FLOOR:
                    A.motion["rec"][1] += 1; A.motion["rec"][0] += int(ps > 0)
                if ga >= AZ_RATE_FLOOR:
                    A.motion["left"][1] += 1; A.motion["left"][0] += int(pa > 0)
                elif ga <= -AZ_RATE_FLOOR:
                    A.motion["right"][1] += 1; A.motion["right"][0] += int(pa < 0)


def stats(acc: Acc, st: str) -> dict:
    r = acc.rec
    idx = np.array([st in s for s in r["strata"]], dtype=bool) if r["strata"] else np.zeros(0, dtype=bool)
    n_gt = acc.n_gt.get(st, 0)
    out = dict(n_gt=n_gt, n_match=acc.n_match.get(st, 0), n_tp20=acc.n_tp20.get(st, 0))
    out["recall_any"] = 100.0 * out["n_match"] / max(n_gt, 1)
    out["recall20"] = 100.0 * out["n_tp20"] / max(n_gt, 1)
    if st.startswith("cls:"):
        cls = next(c for c, n in CLASS_NAME.items() if f"cls:{n}" == st)
        n_pred = acc.n_pred.get(cls, 0)
        out["n_pred"] = n_pred
        out["prec20"] = 100.0 * out["n_tp20"] / max(n_pred, 1)
        out["f1_20"] = (2 * out["prec20"] * out["recall20"] / max(out["prec20"] + out["recall20"], 1e-9))
        out["prec_any"] = 100.0 * out["n_match"] / max(n_pred, 1)
        out["f1_any"] = (2 * out["prec_any"] * out["recall_any"] / max(out["prec_any"] + out["recall_any"], 1e-9))
    if idx.any():
        e = np.array(r["err_az"])[idx]
        out.update(az_med=float(np.median(e)), az_mean=float(np.mean(e)), az_rmse=float(np.sqrt(np.mean(e ** 2))),
                   az_p90=float(np.percentile(e, 90)), az_le10=float(100 * np.mean(e <= 10)), az_le20=float(100 * np.mean(e <= 20)),
                   az_le30=float(100 * np.mean(e <= 30)), az_ge150=float(100 * np.mean(e >= 150)))
        ee = np.array(r["err_el"])[idx]
        out.update(el_med=float(np.median(ee)), el_rmse=float(np.sqrt(np.mean(ee ** 2))), el_le10=float(100 * np.mean(ee <= 10)), el_le20=float(100 * np.mean(ee <= 20)))   # #70 r16: 仰角を表に
        ed = np.array(r["err_d"])[idx]; ed = ed[np.isfinite(ed)]
        if ed.size:
            rel = np.array(r["rel_d"])[idx]; rel = rel[np.isfinite(rel)]
            exy = np.array(r["err_xy"])[idx]; exy = exy[np.isfinite(exy)]
            out.update(n_dist=int(ed.size), d_med=float(np.median(ed)), d_mean=float(np.mean(ed)), d_rmse=float(np.sqrt(np.mean(ed ** 2))),
                       rel_med=float(100 * np.median(rel)), xy_med=float(np.median(exy)), xy_mean=float(np.mean(exy)),
                       xy_rmse=float(np.sqrt(np.mean(exy ** 2))))
    return out


def f(v, fmt="{:.1f}"):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v)


HEAD_MAIN = ("| 予測 | 層 | 正解フレーム | 検出率(方位問わず) | 検出率(20°以内) | 方位誤差 中央/平均/RMSE[°] | 90%点[°] | ≤20° | 前後取り違え(≥150°) | 仰角誤差 中央/RMSE[°] | 仰角≤10° "
             "| 距離誤差 中央/平均/RMSE[m] | 相対誤差中央 | 位置誤差 中央/平均/RMSE[m] | 距離の組数 |")
SEP_MAIN = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
HEAD_CLS = ("| 予測 | クラス | 正解フレーム | 予測数 | 再現率(20°) | 適合率(20°) | F1(20°) | 再現率(方位問わず) | 適合率(方位問わず) | F1(方位問わず) "
            "| 方位誤差 中央/RMSE[°] | ≤20° | 仰角誤差 中央/RMSE[°] | 距離誤差 中央/RMSE[m] | 位置誤差 中央/RMSE[m] |")
SEP_CLS = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def row_main(label, st, s):
    az = f"{f(s.get('az_med'))} / {f(s.get('az_mean'))} / {f(s.get('az_rmse'))}"
    dd = (f"{f(s.get('d_med'), '{:.2f}')} / {f(s.get('d_mean'), '{:.2f}')} / {f(s.get('d_rmse'), '{:.2f}')}" if s.get("n_dist") else "-")
    xy_ = (f"{f(s.get('xy_med'), '{:.2f}')} / {f(s.get('xy_mean'), '{:.2f}')} / {f(s.get('xy_rmse'), '{:.2f}')}" if s.get("n_dist") else "-")
    el = f"{f(s.get('el_med'))} / {f(s.get('el_rmse'))}"
    return (f"| {label} | {st} | {s['n_gt']:,} | {s['recall_any']:.1f}% | {s['recall20']:.1f}% | {az} | {f(s.get('az_p90'))} "
            f"| {f(s.get('az_le20'))}% | {f(s.get('az_ge150'), '{:.2f}')}% | {el} | {f(s.get('el_le10'))}% | {dd} | {f(s.get('rel_med'))}% | {xy_} | {s.get('n_dist', 0):,} |")


def row_cls(label, st, s):
    az = f"{f(s.get('az_med'))} / {f(s.get('az_rmse'))}"
    dd = f"{f(s.get('d_med'), '{:.2f}')} / {f(s.get('d_rmse'), '{:.2f}')}" if s.get("n_dist") else "-"
    xy_ = f"{f(s.get('xy_med'), '{:.2f}')} / {f(s.get('xy_rmse'), '{:.2f}')}" if s.get("n_dist") else "-"
    el = f"{f(s.get('el_med'))} / {f(s.get('el_rmse'))}"
    return (f"| {label} | {st[4:]} | {s['n_gt']:,} | {s.get('n_pred', 0):,} | {s['recall20']:.1f}% | {f(s.get('prec20'))}% | {f(s.get('f1_20'))} "
            f"| {s['recall_any']:.1f}% | {f(s.get('prec_any'))}% | {f(s.get('f1_any'))} | {az} | {f(s.get('az_le20'))}% | {el} | {dd} | {xy_} |")


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
    minframe = int(arg(argv, "--minframe", "40"))
    out_json = arg(argv, "--json", "")
    out_conf = arg(argv, "--confusion", "")       # 2026-10-27（本人 12:35 (a)(b)）: 取り違えの表と本ごとの種類の正答率を別の json と .md に（参考・主の md と json は変えない）
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
    gts = {c: load_gt_all(meta, c) for c in clips}

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    lines = [f"# 知覚層だけの精度（通知の閾値を通さない・1 対 1 対応・門なし） — {out_md.stem}", "",
             f"plan= {_rel(plan)} ({split}{', mix≤' + str(clip_max) if clip_max else ''}) = {len(clips):,} 本 / GT= {_rel(meta)} / "
             f"frame≥{minframe}（「暖機込み」の行は frame≥0） / 方位 20° 以内 = DCASE 流の正解 / 距離クラス = car・kick・bike / 位置誤差 = 方位＋水平距離を xy に直した平面距離", ""]
    result = {}
    conf = {}
    tab_main, tab_cls, tab_motion = [], [], []
    for label, spec in items:
        horiz = spec.endswith("@h")
        p = ROOT / (spec[:-2] if horiz else spec)
        if not p.is_file():
            raise FileNotFoundError(f"missing prediction: {p}")
        pred = S.load_pred_rows(p, horiz=horiz, strict=strict)
        if strict:
            S.strict_check_pred(p, pred, clips)     # 推論 manifest・未知 ID・「1 本も合わない」の検査を統一採点器と同じにする（Opus 5.5 監査 9/23 F01）
        acc, acc0 = Acc(), Acc()
        for c in clips:
            clip_eval(acc, acc0, pred.get(c, {}), gts[c], motion[c], minframe)
        if out_conf:
            import _class_confusion as CC
            conf[label] = CC.score_set(clips, meta, pred, minframe)
        res = {}
        order = (["距離クラス", "距離クラス(暖機込み)"] + [f"方向:{s}" for s in SECTORS] + [f"距離帯:{b[2]}" for b in DIST_BANDS]
                 + [f"motion:{m}" for m in sorted(set(motion.values()))] + ["台数:単独", "台数:多重"])
        for st in order:
            A = acc0 if st == "距離クラス(暖機込み)" else acc
            s = stats(A, st)
            if s["n_gt"]:
                res[st] = s
                tab_main.append(row_main(label, st, s))
        for cls in sorted(CLASS_NAME):
            st = f"cls:{CLASS_NAME[cls]}"
            s = stats(acc, st)
            if s["n_gt"] or s.get("n_pred"):
                res[st] = s
                tab_cls.append(row_cls(label, st, s))
        mo = {k: dict(agree=v[0], n=v[1], rate=(100.0 * v[0] / v[1] if v[1] else float("nan"))) for k, v in acc.motion.items()}
        res["動きの向き"] = mo
        tab_motion.append(f"| {label} | {mo['app']['n']:,} | {f(mo['app']['rate'])}% | {mo['rec']['n']:,} | {f(mo['rec']['rate'])}% "
                          f"| {mo['left']['n']:,} | {f(mo['left']['rate'])}% | {mo['right']['n']:,} | {f(mo['right']['rate'])}% |")
        result[label] = res
        print(f"{label}: 距離クラス {res.get('距離クラス', {}).get('n_gt', 0):,} 正解フレーム・対応 {res.get('距離クラス', {}).get('n_match', 0):,}")

    lines += ["## 1. 距離クラス（car・kick・bike）— 全体と層別", "",
              "- 仰角の誤差（#70 r16・2026-10-19）: 正解の仰角は音源の高さの仮定で決まる（合成 = 描いた高さ・実録 = step19k の SRC_Z〔車 0.325・自転車 0.25・ベルは実測〕）。実録は車 0.05／0.325／0.6 m の感度も並べる",
              "", HEAD_MAIN, SEP_MAIN] + tab_main + [""]
    lines += ["## 2. クラス別（警告クラスは方位と検出だけ・距離は無い）", "", HEAD_CLS, SEP_CLS] + tab_cls + [""]
    lines += ["## 3. 動きの向き（1 秒窓・正解の傾きが床を超える窓で、予測の傾きの符号が一致した割合）", "",
              "| 予測 | 接近の窓 | 一致 | 遠ざかりの窓 | 一致 | 左へ回る窓 | 一致 | 右へ回る窓 | 一致 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + tab_motion + [""]
    lines += ["## 定義", "",
              "- 対応付け: フレームごと・クラスごとに、正解と予測を方位差の合計が最小になるよう 1 対 1 に組む（ハンガリアン法・方位の門なし）。組めなかった正解 = 見逃し、組めなかった予測 = 余分",
              "- 検出率(方位問わず) = 組めた正解フレーム / 正解フレーム。検出率(20°以内) = 方位差 20° 以内で組めた正解フレーム / 正解フレーム（DCASE 2020 以降の位置つき F の流儀）",
              "- 方位誤差・距離誤差・位置誤差は組めた組だけで計算（見逃しは検出率に出る）。RMSE = 二乗平均平方根（大きいずれを重く見る）。"
              "前後取り違え = 組めた組のうち方位誤差 150° 以上の割合（前の相手を後ろと言う、またはその逆）",
              "- 方向 = 正解の方位で 前(|az|≤45°)／左(45°<az<135°)／後(|az|≥135°)／右(−135°<az<−45°)。距離帯 = 正解の水平距離。台数 = 同じクリップに距離クラスの相手（track）が 2 つ以上なら多重",
              "- 動きの向き: 正解の (クラス, track) ごとに 1 秒の重ならない窓で、正解の距離の傾きが ≤−0.3 m/s（接近）／≥+0.3 m/s（遠ざかり）の窓だけを数え、"
              "組めた予測が 8 フレーム以上あるとき、予測の距離の傾きの符号が同じなら一致。方位の回り方は正解の方位の傾き ≥5°/s（左へ）／≤−5°/s（右へ）で同様",
              "- 暖機: 因果推論の最初の 4 秒（frame<40）は宣言どおり主表から外す。「暖機込み」の行だけ frame≥0"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_json:
        Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if out_conf:
        import _class_confusion as CC
        CC.write_outputs(out_conf, conf)
        print(f"-> {out_conf}（取り違えの表と本ごとの種類の正答率・参考）")
    print(f"-> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
