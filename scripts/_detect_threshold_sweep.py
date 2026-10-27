# -*- coding: utf-8 -*-
"""検出層の閾値を動かした評価（2026-09-15 深夜・ゼミ相談の「厳しい閾値と楽観的な閾値」「閾値なしの PR 曲線」を**検出層**に適用）。

検出層 = モデルが 0.1 秒ごとに出す「そこにその種類の相手がいる度合い」（ACCDOA ベクトルの長さ・0〜1 くらい）。本番は 0.5 を超えたら「いる」。
材料 = server_sde/_causal_infer_scores.py が保存した scores/<clip>.csv（frame,track,cls,長さ,x,y,z,距離[m]・長さ > 0.05 の行だけ）。

  A. 閾値なし: 全候補（3 track × クラス・長さ > 0.05・同一クラスで 15° 以内の track は本番と同じ規則で 1 つに統合）を長さの降順に並べ、
     正解（同じ種類・方位 20° 以内・1 対 1）に当たれば当たり、外れれば無駄。PR 曲線の面積（AP）と F1 の最大値を出す。
  B. 閾値ごと: 長さ > t で「いる」とした予測を本番と同じ形（統合・方位/仰角を整数に丸め）にして、
     (1) 検出（再現率・適合率・F1・方位誤差・距離誤差）と (2) そのまま通知層（v4.3＋hold・標準の閾値）まで通した 至近到達・強到達・注意到達・安全抑制・発火数 を出す。
     t = 0.5 の行は本番の予測 CSV での値と一致するはず（自己検査・丸めの差だけ）。
  宣言案: 控えめ = 0.7（いると言いにくい）／標準 = 0.5（本番）／厳格 = 0.3（いると言いやすい）。

使い方:
  python scripts/_detect_threshold_sweep.py <出力md> --plan <assignment.csv> [--split fold2] [--clip-max N] --meta <GTディレクトリ>
      [--minframe 40] [--thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8] [--declare 0.7,0.5,0.3] [--fig <png>] [--strict] [--verify <本番の val_all_causal.csv>] [--legacy8] <ラベル>=<scores ディレクトリ> ...
  --strict = scores_manifest.json と照合（欠損・改変で停止）。--verify = 0.5 で組み立て直した予測が本番と全行一致するか（1 ラベルのとき）。
"""
from __future__ import annotations

import csv
import importlib.util
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


S = _load("score_unified", "_score_unified.py")
PE = _load("perception_eval", "_perception_eval.py")
FPS = int(S.FPS)
UNIFY_DEG = 15.0
AZ_TP = 20.0
DIST_CLS = S.DIST_CLS
SCORE_MIN = 0.05


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


LOG_K = float(np.log1p(100.0))   # server_sde/sde_patch.py の dist_decode（log モード）と同じ
LEGACY8 = False
DUP_AS_FP = False                # --dup-as-fp: 同じ正解を指す 2 本目以降の track を無駄に数える（厳しい版・既定は無視）


def dist_decode(y: float) -> float:
    y = max(float(y), 0.0)
    return float(np.expm1(y * LOG_K))


def load_scores(path: Path):
    """frame -> [(track, cls, norm, x, y, z, d, sed05)]。9 列（Astra A4 の可逆版: dist_raw を復号・sed05 = 本番の 0.5 判定）。
    旧 8 列（4 桁丸め・dist は復号済み）は --legacy8 で明示したときだけ受ける。"""
    out = defaultdict(list)
    if not path.is_file():
        return out
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.strip().split(",")
        if len(p) == 9:
            out[int(p[0])].append((int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), dist_decode(float(p[7])), int(p[8])))
        elif len(p) == 8 and LEGACY8:
            out[int(p[0])].append((int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]), int(float(p[3]) > 0.5)))
        elif p == [""]:
            continue
        else:
            raise ValueError(f"{path.name}:{ln}: scores の列数が {len(p)}（9 列でない）")
    return out


def check_manifest(d: Path, clips) -> dict:
    """Astra A5: scores_manifest.json（推論側が書く）と実ファイルを照合。欠損・sha256 不一致・行数不一致で停止"""
    import hashlib
    import json
    mf = d.parent / "scores_manifest.json"
    if not mf.is_file():
        raise FileNotFoundError(f"scores_manifest.json が無い: {mf}（推論側の来歴・完了検査の対象外の scores は使わない）")
    man = json.loads(mf.read_text(encoding="utf-8"))
    missing = [c for c in clips if c not in man["files"]]
    if missing:
        raise ValueError(f"manifest に無い clip {len(missing)}: {missing[:3]}")
    bad = []
    for c in clips:
        f = d / f"{c}.csv"
        if not f.is_file():
            bad.append((c, "file missing")); continue
        data = f.read_bytes()
        if hashlib.sha256(data).hexdigest() != man["files"][c]["sha256"]:
            bad.append((c, "sha256 mismatch")); continue
        if data.count(b"\n") != man["files"][c]["rows"]:
            bad.append((c, "row count mismatch"))
    if bad:
        raise ValueError(f"scores の照合に失敗 {len(bad)}: {bad[:3]}")
    return man


def _angle(a, b):
    """2 つの xyz ベクトルのなす角[deg]。本番（PSELDNets distance_between_cartesian_coordinates）と同じ式: 正規化に 1e-10 を足し、arccos を [-1,1] に切る"""
    x1, y1, z1 = (float(v) for v in a); x2, y2, z2 = (float(v) for v in b)
    n1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2 + 1e-10); n2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1 / n1, y1 / n1, z1 / n1, x2 / n2, y2 / n2, z2 / n2
    dist = np.clip(x1 * x2 + y1 * y2 + z1 * z2, -1, 1)
    return float(np.arccos(dist) * 180 / np.pi)


def decode(frames_scores, th: float):
    """長さ > th の候補を本番と同じ規則で統合し、frame -> [(cls, az, el, d, score)] を返す（方位・仰角は 0 へ切り捨て）。
    Astra A4: 統合の平均は本番（numpy float32）と同じ精度で計算し、th == 0.5 のときは本番の判定そのもの（sed05）を使う。"""
    f32 = np.float32
    out = {}
    for k, rows in frames_scores.items():
        act = [r for r in rows if (r[7] == 1 if th == 0.5 else r[2] > th)]
        if not act:
            continue
        by_cls = defaultdict(list)
        for trk, cls, nrm, x, y, z, d, _s in act:
            by_cls[cls].append(((f32(x), f32(y), f32(z)), d, nrm))
        res = []
        for cls, grp in by_cls.items():
            def merge(items):
                n = f32(len(items))
                v = tuple(sum((it[0][i] for it in items), f32(0)) / n for i in range(3))
                return v, sum(it[1] for it in items) / len(items), sum(it[2] for it in items) / len(items)
            if len(grp) == 1:
                units = [grp[0]]
            elif len(grp) == 2:
                units = [merge(grp)] if _angle(grp[0][0], grp[1][0]) < UNIFY_DEG else grp
            else:
                f01 = _angle(grp[0][0], grp[1][0]) < UNIFY_DEG
                f12 = _angle(grp[1][0], grp[2][0]) < UNIFY_DEG
                f20 = _angle(grp[2][0], grp[0][0]) < UNIFY_DEG
                s = int(f01) + int(f12) + int(f20)
                if s == 0:
                    units = grp
                elif s == 1:
                    if f01:
                        units = [merge(grp[:2]), grp[2]]
                    elif f12:
                        units = [grp[0], merge(grp[1:])]          # 本番: [grp[0], merge(grp[1], grp[2])] の順
                    else:
                        # 本番（sde_patch の multi_accdoa_to_dcase_format_sde・上流のバグをビット互換で踏襲）: grp[1] を捨て grp[0] を二重計上
                        units = [grp[0], merge([grp[0], grp[2]])]
                else:
                    units = [merge(grp)]
            for (x, y, z), d, sc in units:
                az = int(np.arctan2(y, x) * 180 / np.pi)                   # 本番: np.arctan2 → int()（0 へ切り捨て）
                el = int(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi)
                res.append((cls, float(az), float(el), float(f"{float(d):.2f}"), sc))
        res.sort(key=lambda r: r[0])            # 本番はクラス順に並べて書く（同じ距離の候補があるとき規則の min の選び方が順序に依存するので合わせる）
        out[k] = res
    return out


def verify_against(clips, scores, prod_csv: Path) -> int:
    """Astra A4: 0.5（sed05）で組み立て直した予測と本番の予測 CSV を**全行**照合。不一致のフレーム数を返す（0 を要求）"""
    prod = defaultdict(lambda: defaultdict(list))
    for line in open(prod_csv, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            prod[p[0]][int(p[1])].append((int(p[2]), float(p[4]), float(p[5]), float(p[6])))
    n_bad = 0; shown = 0
    for c in clips:
        mine = decode(scores[c], 0.5)
        for k in sorted(set(prod[c]) | set(mine)):
            a = prod[c].get(k, []); b = [(cls, az, el, d) for cls, az, el, d, _ in mine.get(k, [])]   # 並び順も含めて一致を要求
            if a != b:
                n_bad += 1
                if shown < 5:
                    shown += 1; print(f"  不一致 {c} frame {k}: 本番 {a} / 再構成 {b}")
    print(f"[verify] 本番との照合: 不一致フレーム {n_bad:,}（{len(clips):,} 本）")
    return n_bad


def detection_ap(clips, scores, gts, minframe: int, cls_filter=None):
    """閾値なしの検出 AP（Astra A3・2026-09-16 で定義を変更）。
    候補 = **統合前の track 単位**（長さ > SCORE_MIN の (frame, track, cls)）。統合は閾値に依存するので AP には使わない（B の表とは候補の単位が違う）。
    各候補を長さの順に正解（同種・方位 20° 以内・1 対 1）に当て、**同点は同じ閾値としてまとめて**（並び順に依らない）PR 曲線を作り、
    全点補間の面積 = AP。右端の再現率 = 「長さ > SCORE_MIN の track を全部出したときの再現率」であり、「どの閾値でも取れない」の意味ではない。"""
    cands = []   # (score, tp)
    n_gt = 0; n_dup = 0
    for c in clips:
        sc = scores[c]
        gt = gts[c]
        for k in set(gt) | set(sc):
            if k < minframe:
                continue
            g_all = [g for g in gt.get(k, []) if cls_filter is None or g[0] in cls_filter]
            p_all = [(cls, math.degrees(math.atan2(y, x)), nrm) for trk, cls, nrm, x, y, z, d, _s in sc.get(k, [])
                     if nrm > SCORE_MIN and (cls_filter is None or cls in cls_filter)]
            n_gt += len(g_all)
            for cls in set(g[0] for g in g_all) | set(p[0] for p in p_all):
                g = [x for x in g_all if x[0] == cls]
                p = sorted([x for x in p_all if x[0] == cls], key=lambda x: (-x[2], x[1]))   # 長さ降順・同点は方位で固定
                used = set()
                for pr in p:
                    best, bi = None, None
                    for gi, gg in enumerate(g):                 # 全ての正解の中で最も近いもの（使用済みも含む）
                        e = PE.dang(pr[1], gg[2])
                        if e <= AZ_TP and (best is None or e < best):
                            best, bi = e, gi
                    if bi is None:
                        cands.append((pr[2], 0))                # どの正解からも 20° 以上 = 無駄
                    elif bi in used:
                        n_dup += 1                              # 既に上位の track が当てた正解を指す 2 本目以降 = 重複（無駄とも当たりとも数えない。
                        if DUP_AS_FP:                           #   本番の統合（15° 以内を 1 つに）に相当する扱い。--dup-as-fp で無駄に数える厳しい版）
                            cands.append((pr[2], 0))
                    else:
                        used.add(bi); cands.append((pr[2], 1))
    if not cands or n_gt == 0:
        return dict(ap=float("nan"), f1_max=float("nan"), f1_th=float("nan"), n_gt=n_gt, curve=([], []))
    sc_arr = np.array([s for s, _ in cands]); tp_arr = np.array([t for _, t in cands])
    order = np.argsort(-sc_arr, kind="stable"); sc_arr, tp_arr = sc_arr[order], tp_arr[order]
    uniq = np.unique(-sc_arr)                       # 降順の同点をまとめる
    idx = np.searchsorted(-sc_arr, uniq, side="right") - 1
    tp = np.cumsum(tp_arr)[idx]; fp = np.cumsum(1 - tp_arr)[idx]
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec]); mpre = np.concatenate([[1.0], prec])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return dict(ap=ap, f1_max=float(f1[i]), f1_th=float(-uniq[i]), n_gt=n_gt, n_cand=len(cands), n_dup=n_dup, recall_max=float(rec[-1]), curve=(rec, prec))


def eval_at(clips, scores, gts, meta: Path, motion, minframe: int, th: float):
    """閾値 t で (1) 検出 (2) 通知 を出す"""
    acc, acc0 = PE.Acc(), PE.Acc()
    pred = {}
    for c in clips:
        dec = decode(scores[c], th)
        frames = {k: [(cls, az, el, d) for cls, az, el, d, _ in v] for k, v in dec.items()}
        pred[c] = frames
        PE.clip_eval(acc, acc0, frames, gts[c], motion[c], minframe)
    det_all = PE.stats(acc, "全体")
    det_dist = PE.stats(acc, "距離クラス")
    n_pred_all = sum(acc.n_pred.values()); n_pred_dist = sum(v for k, v in acc.n_pred.items() if k in DIST_CLS)
    for st, n_pred in ((det_all, n_pred_all), (det_dist, n_pred_dist)):
        st["n_pred"] = n_pred
        st["prec20"] = 100.0 * st["n_tp20"] / max(n_pred, 1)
        st["f1_20"] = 2 * st["prec20"] * st["recall20"] / max(st["prec20"] + st["recall20"], 1e-9)
    stats = [S.clip_stats(pred, meta, c, minframe) for c in clips]
    return det_all, det_dist, S.aggregate(stats)


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    split = arg(argv, "--split", "fold2")
    clip_max = int(arg(argv, "--clip-max", "0"))
    meta = ROOT / arg(argv, "--meta")
    minframe = int(arg(argv, "--minframe", "40"))
    ths = [float(x) for x in arg(argv, "--thresholds", "0.2,0.3,0.4,0.5,0.6,0.7,0.8").split(",")]
    declare = [float(x) for x in arg(argv, "--declare", "0.7,0.5,0.3").split(",")]
    fig = arg(argv, "--fig", "")
    verify_csv = arg(argv, "--verify", "")
    global LEGACY8, DUP_AS_FP
    LEGACY8 = "--legacy8" in argv
    if LEGACY8:
        argv.remove("--legacy8")
    DUP_AS_FP = "--dup-as-fp" in argv
    if DUP_AS_FP:
        argv.remove("--dup-as-fp")
    strict = "--strict" in argv
    if strict:
        argv.remove("--strict")
    out_md = Path(argv[0])
    items = [a.split("=", 1) for a in argv[1:]]

    rows = []
    with open(plan, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("split", split) != split:
                continue
            if clip_max:
                m = re.search(r"mix(\d+)$", r["clip_id"])
                if m and int(m.group(1)) > clip_max:
                    continue
            rows.append(r)
    clips = [r["clip_id"] for r in rows]
    motion = {r["clip_id"]: r.get("motion", "") or "-" for r in rows}
    if not clips or len(set(clips)) != len(clips) or not items:
        raise ValueError("nonempty unique manifest and score inputs required")
    missing_gt = [c for c in clips if not (meta / f"{c}.csv").is_file()]
    if missing_gt:
        raise FileNotFoundError(f"missing GT ({len(missing_gt)}): {missing_gt[:3]}")
    gts = {c: PE.load_gt_all(meta, c) for c in clips}

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    names = {}
    for t in declare:
        names[round(t, 3)] = "控えめ" if t == max(declare) else "厳格" if t == min(declare) else "標準"
    lines = [f"# 検出層: 閾値なしの評価と閾値の掃引 — {out_md.stem}", "",
             f"plan= {_rel(plan)} ({split}{', mix≤' + str(clip_max) if clip_max else ''}) = {len(clips):,} 本 / GT= {_rel(meta)} / frame≥{minframe} / "
             f"検出の度合い = ACCDOA ベクトルの長さ（本番の閾値 0.5）/ 宣言案: " + "・".join(f"{names[round(t, 3)]} {t}" for t in declare) + " / 通知層は標準の閾値（v4.3＋hold）で固定", ""]
    A_rows, B_rows, C_rows, D_rows = [], [], [], []
    curves, sweep = {}, {}
    for label, sdir in items:
        d = ROOT / sdir if not Path(sdir).is_absolute() else Path(sdir)
        if not d.is_dir():
            raise FileNotFoundError(f"missing scores dir: {d}")
        if strict:
            man = check_manifest(d, clips)
            print(f"[manifest] {label}: {len(man['files']):,} clips 照合 OK（ckpt {man.get('ckpt_sha256', '')[:12]}）")
        scores = {c: load_scores(d / f"{c}.csv") for c in clips}
        n_missing = sum(1 for c in clips if not (d / f"{c}.csv").is_file())
        if strict and n_missing:
            raise FileNotFoundError(f"scores が {n_missing} 本欠けている")
        if verify_csv:
            vp = ROOT / verify_csv if not Path(verify_csv).is_absolute() else Path(verify_csv)
            n_bad = verify_against(clips, scores, vp)
            if n_bad:
                raise RuntimeError(f"本番の予測との照合で不一致 {n_bad} フレーム（--verify）")
        ap_all = detection_ap(clips, scores, gts, minframe, None)
        ap_dist = detection_ap(clips, scores, gts, minframe, DIST_CLS)
        curves[label] = ap_dist
        A_rows.append(f"| {label} | 全クラス | {ap_all['n_gt']:,} | {ap_all['ap']:.3f} | {100 * ap_all.get('recall_max', float('nan')):.1f}% | {ap_all['f1_max']:.3f}（長さ {ap_all['f1_th']:.2f}） | {ap_all.get('n_cand', 0):,} / {ap_all.get('n_dup', 0):,} |")
        A_rows.append(f"| {label} | 距離クラス | {ap_dist['n_gt']:,} | {ap_dist['ap']:.3f} | {100 * ap_dist.get('recall_max', float('nan')):.1f}% | {ap_dist['f1_max']:.3f}（長さ {ap_dist['f1_th']:.2f}） | {ap_dist.get('n_cand', 0):,} / {ap_dist.get('n_dup', 0):,} |")
        pts = []
        for th in sorted(set(ths) | set(declare)):
            det_all, det_dist, g = eval_at(clips, scores, gts, meta, motion, minframe, th)
            tag = names.get(round(th, 3), "")
            tag = f"**{tag}**" if tag else ""
            B_rows.append(f"| {label} | {th:.2f} | {tag} | {det_dist['recall_any']:.1f}% | {det_dist['recall20']:.1f}% | {det_dist['prec20']:.1f}% | {det_dist['f1_20']:.1f} "
                          f"| {PE.f(det_dist.get('az_med'))}° | {PE.f(det_dist.get('rel_med'))}% | {det_dist['n_pred']:,} | {det_all['recall20']:.1f}% | {det_all['prec20']:.1f}% |")
            C_rows.append(f"| {label} | {th:.2f} | {tag} | {g['crit']:.1f}% | {g['strong']:.1f}% | {g['caut']:.1f}% | {g['safe']:.1f}% | {g['lead']:.2f}s | {g['n_fire']:,} "
                          f"| {g['det_close']:.1f}% | {g['cap_all']:.1f}% | {g['dist_err']:.1f}% |")
            pts.append((th, det_dist['recall20'], det_dist['prec20'], g['crit'], g['strong'], g['caut'], g['safe'], g['n_fire']))
        sweep[label] = pts
        by = {round(p[0], 3): p for p in pts}
        hi, mid, lo = max(declare), sorted(declare)[len(declare) // 2], min(declare)
        if all(round(t, 3) in by for t in (hi, mid, lo)):
            a, m, b = by[round(hi, 3)], by[round(mid, 3)], by[round(lo, 3)]
            D_rows.append(f"| {label} | 厳格 − 控えめ（{lo} − {hi}） | {b[1] - a[1]:+.1f}pt | {b[2] - a[2]:+.1f}pt | {b[3] - a[3]:+.1f}pt | {b[4] - a[4]:+.1f}pt | {b[5] - a[5]:+.1f}pt | {b[6] - a[6]:+.1f}pt | {int(b[7] - a[7]):+,} |")
            D_rows.append(f"| {label} | 厳格 − 標準（{lo} − {mid}） | {b[1] - m[1]:+.1f}pt | {b[2] - m[2]:+.1f}pt | {b[3] - m[3]:+.1f}pt | {b[4] - m[4]:+.1f}pt | {b[5] - m[5]:+.1f}pt | {b[6] - m[6]:+.1f}pt | {int(b[7] - m[7]):+,} |")
            D_rows.append(f"| {label} | 控えめ − 標準（{hi} − {mid}） | {a[1] - m[1]:+.1f}pt | {a[2] - m[2]:+.1f}pt | {a[3] - m[3]:+.1f}pt | {a[4] - m[4]:+.1f}pt | {a[5] - m[5]:+.1f}pt | {a[6] - m[6]:+.1f}pt | {int(a[7] - m[7]):+,} |")
        print(f"{label}: AP(距離クラス) {ap_dist['ap']:.3f}・欠けた scores {n_missing}・{len(pts)} 点")

    lines += ["## A. 閾値なし（候補を度合いの順に並べた PR 曲線の面積）", "",
              "| 予測 | 対象 | 正解フレーム | AP（track 単位・同点まとめ・重複は無視） | 右端の再現率（長さ > 0.05 の track を全部出したとき） | F1 の最大値（そのときの長さ） | 候補数 / 重複（無視した数） |", "| --- | --- | --- | --- | --- | --- | --- |"] + A_rows + [""]
    lines += ["## B. 閾値ごとの検出（距離クラス = 車・キックボード・バイク。右 2 列は全クラス）", "",
              "| 予測 | 閾値 | 設定 | 検出率(方位問わず) | 再現率(20°) | 適合率(20°) | F1(20°) | 方位誤差中央 | 相対距離誤差中央 | 予測数 | 全クラス再現率(20°) | 全クラス適合率(20°) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + B_rows + [""]
    lines += ["## C. 閾値ごとに通知層（標準の閾値）まで通したとき", "",
              "| 予測 | 閾値 | 設定 | 至近到達 | 強到達 | 注意到達 | 安全抑制 | リード中央 | 発火数 | 検出率(GT≤1.5m) | 至近捕捉(全GT) | 距離誤差(1対1) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + C_rows + [""]
    if D_rows:
        lines += ["宣言案の 3 設定の差:", "",
                  "| 予測 | 差 | 再現率(20°) | 適合率(20°) | 至近到達 | 強到達 | 注意到達 | 安全抑制 | 発火数 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + D_rows + [""]
    lines += ["## 定義", "",
              "- 検出の度合い = モデルが 0.1 秒ごとに出す ACCDOA ベクトルの長さ（3 track × クラス）。本番は 0.5 を超えた track を「いる」とし、同じクラスで 15° 以内の track を 1 つに統合する。ここでも同じ規則",
              "- A の AP（Astra A3・2026-09-16 で定義変更）= **統合前の track 単位**の候補（長さ > 0.05）を長さの降順に並べ、正解（同じ種類・方位 20° 以内・1 対 1）に当たれば当たり、外れれば無駄として、"
              "**同点は同じ閾値としてまとめ**、再現率–適合率曲線の下の面積（全点補間）。統合は閾値に依存するので AP には使わない（B の表は統合後）。右端の再現率は「長さ > 0.05 の track を全部出したときの再現率」で、どの閾値でも取れない量ではない",
              "- B の検出は _perception_eval.py と同じ対応付け（方位差最小の 1 対 1・門なし）。C は _score_unified.py の clip_stats（v4.3＋hold・方位帰属・全 GT 分母）",
              "- 0.5 の行は本番の予測 CSV での値と一致するはず（方位・仰角の丸めの差だけ）"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
        fg, ax = plt.subplots(1, 3, figsize=(13, 4))
        for label, ap in curves.items():
            rec, prec = ap["curve"]
            if len(rec):
                ax[0].plot(rec, prec, label=f"{label} AP={ap['ap']:.2f}")
        ax[0].set_xlabel("再現率（正解フレームのうち当てた割合）"); ax[0].set_ylabel("適合率（出した中の当たり）"); ax[0].set_title("検出の PR 曲線（距離クラス・20° 以内）")
        for label, pts in sweep.items():
            ax[1].plot([p[2] for p in pts], [p[1] for p in pts], "o-", label=label)
            ax[2].plot([p[6] for p in pts], [p[3] for p in pts], "o-", label=label)
            for p in pts:
                ax[1].annotate(f"{p[0]:.1f}", (p[2], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
                ax[2].annotate(f"{p[0]:.1f}", (p[6], p[3]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].set_xlabel("適合率(20°) %"); ax[1].set_ylabel("再現率(20°) %"); ax[1].set_title("検出の閾値を動かしたとき")
        ax[2].set_xlabel("安全抑制 %"); ax[2].set_ylabel("至近到達 %"); ax[2].set_title("検出の閾値を動かして通知まで通したとき")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fg.tight_layout(); fg.savefig(fig, dpi=150); plt.close(fg)
    print(f"-> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
