# -*- coding: utf-8 -*-
"""_dist_calib_v2.py — 距離の補正表（凍結 §5・変更票 #72・2026-10-20・Fable）。scripts_archive/_dist_calib_proto.py（2026-09-03）の手順を 1 字も変えず、今の CSV の形（因果推論の 7 列・水平距離の GT・選択器と同じ対応）で動くようにしたもの。

表 = 「モデルが言った距離 → 実際の距離の中央値」の単調な対応（0〜5 m を 0.25 m 刻み・ビンのペアが 30 未満なら恒等・5 m 超は恒等・隣のビンと順が逆なら平均で併合 = PAVA）。
対応ペア = 選択器 2.1 と同じ（frame 40〜99・距離クラス {4,6,7,8}・同じクラス・方位差 ≤ 20°・貪欲 1 対 1・_v2_ckpt_selector.match_frame）。

使い方:
  fit:   python scripts/_dist_calib_v2.py fit --pred <calib の因果推論 csv（4,000 本）> --gt <metadata_dist dir> --manifest <calib_fit の名簿 csv> --manifest-all <calib 全体の名簿（plan）> [--ckpt-sha <hex>] --out <json>
         --manifest-all を渡すと予測は全体の名簿で厳密に読み、作表は fit の本だけ（RN20-05）。渡さなければ --pred は fit の本だけの csv であること
  apply: python scripts/_dist_calib_v2.py apply --table <json> --pred <csv> --out <csv>      # 距離の列だけ読み替えた予測 csv（統一採点器は --dist-table で直接読めるので通常は不要）
  decide: python scripts/_dist_calib_v2.py decide --unified-off <表なし md> --unified-on <表あり md> --long-off40 <json> --long-on40 <json> --long-off41 <json> --long-on41 <json> --break-off40 <json> --break-on40 <json> --break-off41 <json> --break-on41 <json> [--label v2final] [--table <dist_table.json>] [--ckpt-sha <hex>] [--pred-calib <csv>] [--pred-long <csv>] --out <json>
          採否を丸め前の分子・分母で判定して JSON に残す（R72-08）。分母 0・表なしの基準 0 は判定不能 = 採用しない。fold40 と fold41 は別々に見て両方が通ること。
          RN20-06: 表あり・なしの集計が同じ fold・同じ予測・同じ分母（本数・分数）で表の有無だけ違うことを確かめ、全入力・表・ckpt・予測の sha256 を JSON に残す（確定評価の実行の契約が読む）
  selftest: python scripts/_dist_calib_v2.py --selftest
条件 (1)(2) の判定は、統一採点器（--dist-table）と長尺の採点を表あり・なしで回して比べる（calib_check・長尺 v1）。この道具は表を作る所まで。
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EDGES = np.arange(0.0, 5.0 + 1e-9, 0.25)
MIN_PAIRS = 30
IDENTITY_ABOVE = 5.0


def _sel():
    spec = importlib.util.spec_from_file_location("v2sel_calib", ROOT / "scripts/_v2_ckpt_selector.py")
    m = importlib.util.module_from_spec(spec); sys.modules["v2sel_calib"] = m; spec.loader.exec_module(m)
    return m


def pava(y, w):
    """単調非減少になるよう隣接ビンを平均で併合（pool-adjacent-violators・試作と同じ）。"""
    y = list(map(float, y)); w = list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            m = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            blocks[i] = [m, a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, _, n in blocks:
        out += [m] * n
    return np.array(out)


def fit_table(P: np.ndarray):
    """P[:,0] = 推定距離・P[:,1] = GT 距離 → (centers, values, counts)。ビンの GT 中央値を単調化。ペア < MIN_PAIRS のビンは恒等。"""
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    med, cnt = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (P[:, 0] >= lo) & (P[:, 0] < hi)
        cnt.append(int(sel.sum()))
        med.append(float(np.median(P[sel, 1])) if sel.sum() >= MIN_PAIRS else float("nan"))
    med = np.array(med); cnt = np.array(cnt)
    ok = np.isfinite(med)
    vals = med.copy()
    if ok.any():
        vals[ok] = pava(med[ok], cnt[ok])
    vals[~ok] = centers[~ok]
    return centers, vals, cnt


def apply_table(d, centers, vals):
    d = np.asarray(d, dtype=float)
    out = np.interp(d, centers, vals)
    return np.where(d >= IDENTITY_ABOVE, d, out)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs_from(sel, pred_csv: Path, gt_dir: Path, clips: list[str], pred: dict | None = None) -> np.ndarray:
    """選択器と同じ対応で (推定距離, GT 距離) のペアを全 clip から集める。pred を渡せばそれを使う（全体の名簿で読んで fit の本に絞ったもの・RN20-05）。"""
    gt = sel.read_gt(gt_dir, clips)
    if pred is None:
        pred, _ = sel.read_pred(pred_csv, clips)
    P = []
    for c in clips:
        gts = sel._norm(r for r in gt.get(c, []) if sel.FRAME_MIN <= r[0] <= sel.FRAME_MAX and r[1] in sel.DIST_CLS)
        prs = sel._norm(r for r in pred.get(c, []) if sel.FRAME_MIN <= r[0] <= sel.FRAME_MAX and r[1] in sel.DIST_CLS and np.isfinite(r[5]))
        by_g, by_p = defaultdict(list), defaultdict(list)
        for r in gts:
            by_g[r[0]].append(r)
        for r in prs:
            by_p[r[0]].append(r)
        for k in sorted(by_g):
            for g, p in sel.match_frame(by_g[k], by_p.get(k, [])):
                P.append((float(p[5]), float(g[5])))
    return np.array(P, dtype=float).reshape(-1, 2)


def fit(pred_csv: Path, gt_dir: Path, manifest: Path, out_json: Path, manifest_all: Path | None = None, ckpt_sha: str | None = None) -> dict:
    """表を作る。manifest = fit の名簿（calib_fit 2,000 本）。manifest_all を渡すと（RN20-05）、予測 csv は全体の名簿（calib 4,000 本）で厳密に読み
    （名簿外の行は拒否）、作表には fit の本だけを使う（check の行は混ぜない）。抽出元・名簿・抽出後の本数と sha を JSON に残す。"""
    sel = _sel()
    clips = sel.read_manifest(manifest)
    pred = None; sub = {}
    if manifest_all is not None:
        clips_all = sel.read_manifest(manifest_all)
        missing = sorted(set(clips) - set(clips_all))
        if missing:
            raise SystemExit(f"fit の名簿に全体の名簿に無い本がある: {missing[:3]}（{len(missing)} 本）")
        pred_all, _ = sel.read_pred(pred_csv, clips_all)                      # 全体の名簿で読む（名簿外の clip は read_pred が拒否）
        pred = {c: pred_all.get(c, []) for c in clips}                       # fit の本だけ（check の本は作表に入れない）
        sub = {"manifest_all": str(manifest_all), "manifest_all_sha256": _sha(manifest_all), "n_clips_all": len(clips_all),
               "n_pred_rows_all": int(sum(len(v) for v in pred_all.values())), "n_pred_rows_fit": int(sum(len(v) for v in pred.values())),
               "n_clips_fit_with_pred": int(sum(1 for v in pred.values() if v))}
    P = pairs_from(sel, pred_csv, gt_dir, clips, pred=pred)
    if len(P) == 0:
        raise SystemExit("対応ペアが 0（予測と GT の clip が合っていない）")
    centers, vals, cnt = fit_table(P)
    close = P[P[:, 1] <= 1.5]
    out = {"version": "dist_calib/2.0", "edges": EDGES.tolist(), "centers": centers.tolist(), "values": vals.tolist(), "counts": cnt.tolist(),
           "min_pairs": MIN_PAIRS, "identity_above_m": IDENTITY_ABOVE, "n_pairs": int(len(P)), "n_clips": len(clips),
           "fit_pred": str(pred_csv), "fit_pred_sha256": _sha(pred_csv), "manifest": str(manifest), "manifest_sha256": _sha(manifest), "gt_dir": str(gt_dir),
           "ckpt_sha256": ckpt_sha, **sub,
           "close_pred_median_before_m": (float(np.median(close[:, 0])) if len(close) else None),
           "close_pred_median_after_m": (float(np.median(apply_table(close[:, 0], centers, vals))) if len(close) else None),
           "note": "0.25 m ビンの実距離中央値を単調化（単調化はペア 30 以上のビンの中・ペア<30 は恒等・5 m 超は恒等）。手順は 2026-09-03 の試作と同じ（凍結 §5）。採否は calib_check と長尺 v1 の条件 (1)(2) で決める（#72）"}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"作成: ペア {len(P):,}（clip {len(clips):,}）・至近（GT ≤ 1.5 m）の推定距離の中央値 {out['close_pred_median_before_m']} → {out['close_pred_median_after_m']} m")
    print("| 推定距離のビン | ペア数 | 実距離の中央値（補正後の値） | 補正量 |")
    for lo, hi, c, v, n in zip(EDGES[:-1], EDGES[1:], centers, vals, cnt):
        if hi <= 3.0:
            print(f"| {lo:.2f}–{hi:.2f} m | {n:,} | {v:.2f} m | {v - c:+.2f} m |")
    print("->", out_json)
    return out


THRESH = dict(cap_pt=5.0, fp_pt=0.05, mid_ratio=0.10, safe_pt=2.0)   # 採用条件（変更票 #72・凍結 §5）: (1a) 至近捕捉 +5 pt 以上・(1b) 誤捕捉 +0.05 pt 以内・(2a) 中/分 +10% 以内・(2b) 安全な車への強 +2 pt 以内


def _raw_row(md: Path, label: str) -> dict:
    """統一採点器（--raw）の「丸め前」の行から 至近捕捉 cap/n_gt と 誤捕捉 n_safe_fp/n_safe_pairs を読む（表示値ではなく分子・分母）。"""
    import re
    rows = [l for l in md.read_text(encoding="utf-8").splitlines() if l.startswith(f"| {label} |") and l.count("|") >= 13]
    if not rows:
        raise SystemExit(f"{md}: 丸め前の行（| {label} | … 13 列以上）が無い。--raw で採点したか")
    cells = [x.strip() for x in rows[-1].split("|")][1:]
    def frac(cell):
        m = re.match(r"(\d+)/(\d+)", cell)
        if not m:
            raise SystemExit(f"分子/分母が読めない: {cell!r}（{md}）")
        return int(m.group(1)), int(m.group(2))
    cap, n_gt = frac(cells[3]); fp, n_safe = frac(cells[7])
    return dict(cap=cap, n_gt=n_gt, fp=fp, n_safe=n_safe)


def _same_long_inputs(j0: dict, j1: dict, fold: str, kind: str, table: Path | None) -> None:
    """表なし（j0）と表あり（j1）の長尺の集計が、同じ fold・同じ予測・同じ分母（本数・分数）で、表の有無だけ違うことを確かめる（RN20-06）。"""
    want = f"fold{fold}"
    if j0.get("split") != want or j1.get("split") != want:
        raise SystemExit(f"{kind} fold{fold}: split が違う（表なし {j0.get('split')}・表あり {j1.get('split')}）")
    if j0.get("pred") != j1.get("pred"):
        raise SystemExit(f"{kind} fold{fold}: 表あり・なしで予測の csv が違う（{j0.get('pred')} ≠ {j1.get('pred')}）")
    if j0.get("dist_table") or not j1.get("dist_table"):
        raise SystemExit(f"{kind} fold{fold}: 表なし／表ありの取り違え（表なし側 {j0.get('dist_table')!r}・表あり側 {j1.get('dist_table')!r}）")
    if table is not None and Path(j1["dist_table"]).name != Path(table).name:
        raise SystemExit(f"{kind} fold{fold}: 表あり側の表（{j1['dist_table']}）が --table（{table}）と違う")
    if kind == "long":
        g0, g1 = j0["groups"]["all/all"], j1["groups"]["all/all"]
        if g0.get("n") != g1.get("n") or g0.get("minutes") != g1.get("minutes"):
            raise SystemExit(f"long fold{fold}: 表あり・なしで分母が違う（本数 {g0.get('n')}/{g1.get('n')}・分 {g0.get('minutes')}/{g1.get('minutes')}）")
    else:
        if j0.get("scene") != "arterial_walk" or j1.get("scene") != "arterial_walk":
            raise SystemExit(f"break fold{fold}: scene が arterial_walk でない（{j0.get('scene')}・{j1.get('scene')}）")
        if j0.get("n_clips") != j1.get("n_clips") or j0.get("minutes") != j1.get("minutes"):
            raise SystemExit(f"break fold{fold}: 表あり・なしで分母が違う（本数 {j0.get('n_clips')}/{j1.get('n_clips')}・分 {j0.get('minutes')}/{j1.get('minutes')}）")


def decide(args: dict, out_json: Path) -> dict:
    """採否を丸め前の数で判定して JSON に残す（R72-08）。分母 0 や表なしの基準 0 は「判定不能」= 採用しない。fold40 と fold41 は別々に見て両方が通ること。
    RN20-06: 表あり・なしの入力が同じ fold・同じ予測・同じ分母であることを確かめ、全入力・表・ckpt・予測の sha256 を JSON に残す。"""
    table = Path(args["table"]) if args.get("table") else None
    R = {"thresholds": THRESH, "conditions": {}, "undefined": [], "label": args["label"],
         "inputs": {k: {"path": str(args[k]), "sha256": _sha(args[k])} for k in ("unified_off", "unified_on", "long_off40", "long_on40", "long_off41", "long_on41", "break_off40", "break_on40", "break_off41", "break_on41")},
         "table_sha256": (_sha(table) if table else None), "fit_pred_sha256": (json.loads(table.read_text(encoding="utf-8")).get("fit_pred_sha256") if table else None),
         "ckpt_sha256": args.get("ckpt_sha"), "pred_calib_sha256": (_sha(args["pred_calib"]) if args.get("pred_calib") else None),
         "pred_long_sha256": (_sha(args["pred_long"]) if args.get("pred_long") else None)}
    if table and args.get("pred_calib") and R["fit_pred_sha256"] and R["fit_pred_sha256"] != R["pred_calib_sha256"]:
        raise SystemExit("表を作った予測（fit_pred_sha256）と条件 (1) の予測（--pred-calib）が違う")
    if table and args.get("ckpt_sha"):
        t_ck = json.loads(table.read_text(encoding="utf-8")).get("ckpt_sha256")
        if t_ck and t_ck != args["ckpt_sha"]:
            raise SystemExit("表の ckpt_sha256 と --ckpt-sha が違う")
    u0, u1 = _raw_row(Path(args["unified_off"]), args["label"]), _raw_row(Path(args["unified_on"]), args["label"])
    if u0["n_gt"] != u1["n_gt"] or u0["n_safe"] != u1["n_safe"]:
        raise SystemExit(f"表あり・なしで分母が違う（至近 GT {u0['n_gt']}/{u1['n_gt']}・安全なペア {u0['n_safe']}/{u1['n_safe']}）")
    if u0["n_gt"] == 0:
        R["undefined"].append("1a: 至近の GT が 0")
    else:
        d = 100.0 * (u1["cap"] - u0["cap"]) / u0["n_gt"]
        R["conditions"]["1a"] = dict(name="至近捕捉（全 GT 分母）の増加 pt", off=100.0 * u0["cap"] / u0["n_gt"], on=100.0 * u1["cap"] / u0["n_gt"], diff_pt=d, n=u0["n_gt"], passed=bool(d >= THRESH["cap_pt"] - 1e-12))
    if u0["n_safe"] == 0:
        R["undefined"].append("1b: 安全な対応ペアが 0")
    else:
        d = 100.0 * (u1["fp"] - u0["fp"]) / u0["n_safe"]
        R["conditions"]["1b"] = dict(name="誤捕捉（GT > 3.2 m の対応ペア分母）の増加 pt", off=100.0 * u0["fp"] / u0["n_safe"], on=100.0 * u1["fp"] / u0["n_safe"], diff_pt=d, n=u0["n_safe"], passed=bool(d <= THRESH["fp_pt"] + 1e-12))
    for fold in ("40", "41"):
        lo, l1 = json.loads(Path(args[f"long_off{fold}"]).read_text(encoding="utf-8")), json.loads(Path(args[f"long_on{fold}"]).read_text(encoding="utf-8"))
        _same_long_inputs(lo, l1, fold, "long", table)
        g0, g1 = lo["groups"]["all/all"], l1["groups"]["all/all"]
        if not g0["mid"]:
            R["undefined"].append(f"2a fold{fold}: 表なしの中が 0")
        else:
            ratio = (g1["mid"] - g0["mid"]) / g0["mid"]                       # 分母（分数）が同じことは _same_long_inputs で確かめた → 件数の比 = 率の比
            R["conditions"][f"2a_fold{fold}"] = dict(name="中/分の増加率", off=g0["mid_per_min"], on=g1["mid_per_min"], mid_off=g0["mid"], mid_on=g1["mid"], minutes=g0.get("minutes"), n=g0.get("n"), ratio=ratio, passed=bool(ratio <= THRESH["mid_ratio"] + 1e-12))
        b0, b1 = json.loads(Path(args[f"break_off{fold}"]).read_text(encoding="utf-8")), json.loads(Path(args[f"break_on{fold}"]).read_text(encoding="utf-8"))
        _same_long_inputs(b0, b1, fold, "break", table)
        s0, s1 = b0["tiers"]["強"], b1["tiers"]["強"]
        if not s0.get("合計") or not s1.get("合計"):
            R["undefined"].append(f"2b fold{fold}: 強の発火が 0（表なし {s0.get('合計')}・表あり {s1.get('合計')}）")
        else:
            d = 100.0 * (s1["安全"] / s1["合計"] - s0["安全"] / s0["合計"])
            R["conditions"][f"2b_fold{fold}"] = dict(name="幹線歩道の安全な車への強の割合（強の発火全体のうち）の増加 pt", off=100.0 * s0["安全"] / s0["合計"], on=100.0 * s1["安全"] / s1["合計"], n_off=s0["合計"], n_on=s1["合計"], diff_pt=d, passed=bool(d <= THRESH["safe_pt"] + 1e-12))
    allp = all(v["passed"] for v in R["conditions"].values())
    R["adopt"] = bool(allp and not R["undefined"] and len(R["conditions"]) == 6)
    R["verdict"] = "表あり（採用）" if R["adopt"] else ("判定不能 → 表なし" if R["undefined"] or len(R["conditions"]) < 6 else "表なし（不採用）")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(R["verdict"]); print(json.dumps(R["conditions"], ensure_ascii=False, indent=1)[:1500])
    if R["undefined"]:
        print("判定不能:", R["undefined"])
    return R


def apply_csv(table: Path, pred_csv: Path, out_csv: Path) -> int:
    t = json.loads(table.read_text(encoding="utf-8"))
    centers, vals = np.array(t["centers"]), np.array(t["values"])
    n = 0
    with open(pred_csv, encoding="utf-8") as f, open(out_csv, "w", encoding="utf-8") as g:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) >= 7 and p[6] not in ("", "nan"):
                p[6] = f"{float(apply_table(float(p[6]), centers, vals)):.3f}"; n += 1
            g.write(",".join(p) + "\n")
    print(f"読み替えた行 {n:,} -> {out_csv}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.3, 6.0, 20000)
    pred = np.where(gt < 2.0, gt + 0.4, gt) + rng.normal(0, 0.05, gt.size)        # 2 m 未満だけ 0.4 m 遠めに出る癖
    P = np.stack([pred, gt], axis=1)
    centers, vals, cnt = fit_table(P)
    # 単調化は「ペアが 30 以上のビンの中」で行う（試作と同じ）。ペアの少ないビンは恒等（ビンの中心）なので、全体では単調にならないことがある
    okb = cnt >= MIN_PAIRS
    ok.append(("ペアの多いビンの中で単調（非減少）", bool(np.all(np.diff(vals[okb]) >= -1e-9)) and okb.sum() >= 5))
    ok.append(("ペアの多いビンは実距離の中央値（1.5〜1.75 のビン → 約 1.2）", abs(vals[6] - (centers[6] - 0.4)) < 0.08))
    close = P[P[:, 1] <= 1.5]
    before = float(np.median(close[:, 0])); after = float(np.median(apply_table(close[:, 0], centers, vals)))
    ok.append((f"至近の推定の中央値が GT の中央値に寄る（{before:.2f} → {after:.2f}・GT {float(np.median(close[:, 1])):.2f}）", abs(after - float(np.median(close[:, 1]))) < 0.1 and before > after))
    ok.append(("5 m 超は恒等", float(apply_table(np.array([5.5]), centers, vals)[0]) == 5.5))
    few = np.stack([rng.uniform(0, 0.25, 10), rng.uniform(0, 0.25, 10)], axis=1)   # ペア < 30 のビンだけ → 全部恒等
    c2, v2, n2 = fit_table(few)
    ok.append(("ペア 30 未満のビンは恒等", bool(np.allclose(v2, c2))))
    # 選択器の対応で pairs_from が動く（架空の 1 clip・GT と予測が同じ方位）
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="dcal_"))
    (tmp / "gt").mkdir(); man = tmp / "man.csv"; man.write_text("clip_id\nval_syn_clip00001\n", encoding="utf-8")
    (tmp / "gt/val_syn_clip00001.csv").write_text("".join(f"{k},4,0,30,0,{1.2:.2f}\n" for k in range(40, 100)), encoding="utf-8")
    pred_csv = tmp / "pred.csv"; pred_csv.write_text("".join(f"val_syn_clip00001,{k},4,0,30,0,{1.7:.2f}\n" for k in range(40, 100)), encoding="utf-8")
    sel = _sel()
    Pp = pairs_from(sel, pred_csv, tmp / "gt", ["val_syn_clip00001"])
    ok.append((f"選択器と同じ対応でペアが取れる（{len(Pp)} ペア・推定 1.7・GT 1.2）", len(Pp) == 60 and abs(Pp[0, 0] - 1.7) < 1e-9 and abs(Pp[0, 1] - 1.2) < 1e-9))
    # decide（R72-08）: 架空の md／json で、丸め前の数で判定し、分母 0 は判定不能
    def umd(cap, fp, n_gt=1000, n_safe=2000):
        p = tmp / f"u_{cap}_{fp}.md"
        p.write_text(f"| v2final | 1/2 = 50.0000% | 1/2 = 50.0000% | {cap}/{n_gt} = x% | 1/2 = x% | 1.0000% | 9 | {fp}/{n_safe} = x% | 1/2 = x% | 1/2 = x% | 1/1/1 | 0 |\n", encoding="utf-8"); return p
    tbl = tmp / "dist_table.json"; tbl.write_text(json.dumps({"version": "dist_calib/2.0", "centers": [0.0] * 20, "values": [0.0] * 20, "fit_pred_sha256": "f" * 64}), encoding="utf-8")
    def lj(mid, tag, fold="40", on=False, n=200, minutes=200.0, pred="long.csv"):      # RN20-06: split・pred・dist_table・分母を持つ（本番の _long_score_v2 --json と同じ鍵）
        p = tmp / f"l_{tag}.json"; p.write_text(json.dumps({"split": f"fold{fold}", "pred": pred, "dist_table": (str(tbl) if on else None), "groups": {"all/all": {"mid": mid, "n": n, "minutes": minutes, "mid_per_min": mid / minutes}}}), encoding="utf-8"); return p
    def bj(safe, total, tag, fold="40", on=False, n_clips=200, minutes=200.0, pred="long.csv", scene="arterial_walk"):
        p = tmp / f"b_{tag}.json"; p.write_text(json.dumps({"split": f"fold{fold}", "scene": scene, "n_clips": n_clips, "minutes": minutes, "pred": pred, "dist_table": (str(tbl) if on else ""), "tiers": {"強": {"合計": total, "安全": safe}, "中": {"合計": 1, "安全": 0}}}), encoding="utf-8"); return p
    args = dict(label="v2final", unified_off=umd(600, 2), unified_on=umd(660, 3), long_off40=lj(276, "o40"), long_on40=lj(290, "n40", on=True), long_off41=lj(280, "o41", "41"), long_on41=lj(300, "n41", "41", on=True),
                break_off40=bj(56, 104, "bo40"), break_on40=bj(59, 108, "bn40", on=True), break_off41=bj(50, 100, "bo41", "41"), break_on41=bj(52, 100, "bn41", "41", on=True), table=tbl)
    Rj = decide({k: str(v) for k, v in args.items()}, tmp / "decide.json")
    ok.append((f"decide: +6.0 pt・誤捕捉 +0.05 pt・中/分 +5.1%／+7.1%・安全な車への強 +0.8 pt／+2.0 pt → 表あり（{Rj['verdict']}）", Rj["adopt"] is True))
    ok.append(("decide: 入力 10 本・表・fit の予測の sha256 を JSON に残す", len(Rj["inputs"]) == 10 and all(len(v["sha256"]) == 64 for v in Rj["inputs"].values()) and Rj["table_sha256"] == _sha(tbl) and Rj["fit_pred_sha256"] == "f" * 64))
    args2 = dict(args, break_on41=bj(53, 100, "bn41b", "41", on=True))                      # 41 で +3 pt → 不採用
    Rj2 = decide({k: str(v) for k, v in args2.items()}, tmp / "decide2.json")
    ok.append((f"decide: fold41 の安全な車への強が +3.0 pt → 表なし（{Rj2['verdict']}）", Rj2["adopt"] is False and "不採用" in Rj2["verdict"]))
    args3 = dict(args, break_off40=bj(0, 0, "bo40z"), break_on40=bj(0, 0, "bn40z", on=True))   # 分母 0 → 判定不能
    Rj3 = decide({k: str(v) for k, v in args3.items()}, tmp / "decide3.json")
    ok.append((f"decide: 強の発火 0 → 判定不能 = 表なし（{Rj3['verdict']}）", Rj3["adopt"] is False and "判定不能" in Rj3["verdict"]))
    # RN20-06: 表あり・なしで分母（分数）が違う・fold の取り違え・表なし／ありの取り違え → 拒否（判定しない）
    def refuses(a):
        try:
            decide({k: str(v) for k, v in a.items()}, tmp / "decide_x.json"); return False
        except SystemExit:
            return True
    ok.append(("decide: 表あり側の分数が違う（100 件/200 分 vs 100 件/100 分）→ 拒否", refuses(dict(args, long_off40=lj(100, "m0"), long_on40=lj(100, "m1", on=True, minutes=100.0)))))
    ok.append(("decide: fold41 の欄に fold40 の集計 → 拒否", refuses(dict(args, long_on41=lj(300, "f1", "40", on=True)))))
    ok.append(("decide: 表なし側に表ありの集計 → 拒否", refuses(dict(args, long_off40=lj(276, "s0", on=True)))))
    ok.append(("decide: 表あり・なしで予測の csv が違う → 拒否", refuses(dict(args, break_on40=bj(59, 108, "p1", on=True, pred="other.csv")))))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    def opt(k):
        return Path(a[a.index(k) + 1]) if k in a else None
    def sopt(k):
        return a[a.index(k) + 1] if k in a else None
    if a and a[0] == "fit":
        fit(opt("--pred"), opt("--gt"), opt("--manifest"), opt("--out"), manifest_all=opt("--manifest-all"), ckpt_sha=sopt("--ckpt-sha")); return 0
    if a and a[0] == "apply":
        return apply_csv(opt("--table"), opt("--pred"), opt("--out"))
    if a and a[0] == "decide":
        keys = ("unified_off", "unified_on", "long_off40", "long_on40", "long_off41", "long_on41", "break_off40", "break_on40", "break_off41", "break_on41")
        args = {k: a[a.index("--" + k.replace("_", "-")) + 1] for k in keys}
        args["label"] = a[a.index("--label") + 1] if "--label" in a else "v2final"
        args.update(table=sopt("--table"), ckpt_sha=sopt("--ckpt-sha"), pred_calib=sopt("--pred-calib"), pred_long=sopt("--pred-long"))   # RN20-06: 入力の sha を記録・表と ckpt の照合
        decide(args, opt("--out")); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
