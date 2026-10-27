# -*- coding: utf-8 -*-
"""検出のヒステリシス（入りの閾値と保持の閾値の 2 つ）を探す（2026-09-16 20:25・本人「教員の言っていたのは 2 つの閾値を同時に使う方（ヒステリシス）」）。

判定（フレーム順に処理・種類ごと）:
  候補 = 矢印の長さ > 保持の閾値 (stay) の track
  「いる」= 長さ > 入りの閾値 (enter)、または（前のフレームで「いる」だった同じ種類の相手が方位 ±assoc° 以内にいて、長さ > stay）
  「いる」となった track を本番と同じ規則で統合（15° 以内を 1 つに）→ 予測。enter == stay なら本番と同じ 1 閾値の判定
評価は層ごと:
  知覚層 = _perception_eval と同じ検出率・再現率(20°)・適合率(20°)・方位/距離誤差 ＋ **ちらつき**（正解の相手ごとに、検出が「あり→なし」に落ちた回数 / 正解フレーム数）
  通知層 = 統一採点器と同じ帰属で 至近到達・強到達・注意到達・安全抑制・発火数（通知の閾値は現行）

使い方:
  python scripts/_hysteresis_grid.py <出力md> --plan <assignment.csv> [--split fold2] --meta <GTディレクトリ> --scores <scores dir>
      [--enter 0.4,0.5,0.6,0.7] [--stay 0.1,0.2,0.3,0.4] [--assoc 20] [--minframe 40] [--csv <格子のcsv>]
"""
from __future__ import annotations

import csv
import importlib.util
import math
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


D = _load("detect_sweep", "_detect_threshold_sweep.py")
N = _load("notify_sweep", "_notify_threshold_sweep.py")
PE = D.PE
S = N.S


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def decode_hyst(frames_scores, enter: float, stay: float, assoc: float = 20.0):
    """ヒステリシス付きの組み立て直し。frame -> [(cls, az, el, d, score)]"""
    if stay >= enter:
        return D.decode(frames_scores, enter)
    out = {}
    prev = defaultdict(list)          # cls -> [az] 前のフレームで「いる」だった方位
    for k in sorted(frames_scores):
        rows = frames_scores[k]
        act = []
        for r in rows:
            trk, cls, nrm, x, y, z, d, s05 = r
            if nrm > enter:
                act.append(r)
            elif nrm > stay and prev.get(cls):
                az = math.degrees(math.atan2(y, x))
                if any(PE.dang(az, pa) <= assoc for pa in prev[cls]):
                    act.append(r)
        # 「いる」となった track だけを本番と同じ統合に通す（decode は th で選別するので、選別済みの行を長さ > -1 で全部通す）
        dec = D.decode({k: act}, -1.0) if act else {}
        res = dec.get(k, [])
        out[k] = res
        prev = defaultdict(list)
        for cls, az, el, d, sc in res:
            prev[cls].append(az)
    return {k: v for k, v in out.items() if v}


def flicker(frames, gt, minframe: int):
    """正解の (cls, trk) ごとに、対応する予測が「あり→なし」に落ちた回数と、正解フレーム数（距離クラス・frame≥minframe）"""
    drops = 0; n = 0
    tracks = defaultdict(list)
    for k in sorted(gt):
        for g in gt[k]:
            if g[0] in PE.DIST_CLS:
                tracks[(g[0], g[1])].append((k, g[2]))
    for (cls, trk), seq in tracks.items():
        was = None
        for k, gaz in seq:
            if k < minframe:
                continue
            n += 1
            has = any(c == cls and PE.dang(az, gaz) <= 20.0 for c, az, el, d in frames.get(k, []))
            if was is True and not has:
                drops += 1
            was = has
    return drops, n


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    split = arg(argv, "--split", "fold2")
    meta = ROOT / arg(argv, "--meta")
    sdir = ROOT / arg(argv, "--scores")
    enters = [float(x) for x in arg(argv, "--enter", "0.4,0.5,0.6,0.7").split(",")]
    stays = [float(x) for x in arg(argv, "--stay", "0.1,0.2,0.3,0.4").split(",")]
    assoc = float(arg(argv, "--assoc", "20"))
    minframe = int(arg(argv, "--minframe", "40"))
    out_csv = arg(argv, "--csv", "")
    out_md = Path(argv[0])

    rows = [r for r in csv.DictReader(open(plan, newline="", encoding="utf-8")) if r.get("split", split) == split]
    clips = [r["clip_id"] for r in rows]
    motion = {r["clip_id"]: r.get("motion", "") or "-" for r in rows}
    D.check_manifest(sdir, clips)
    scores = {c: D.load_scores(sdir / f"{c}.csv") for c in clips}
    gts = {c: PE.load_gt_all(meta, c) for c in clips}
    assert S.DG.v4 is not S.V43.v4
    evs_of = {c: N.events_of(meta, c) for c in clips}

    grid = []
    combos = [(e, s) for e in enters for s in stays if s <= e]
    if not any(abs(e - 0.5) < 1e-9 and abs(s - 0.5) < 1e-9 for e, s in combos):
        combos.insert(0, (0.5, 0.5))    # 現行（1 閾値）を基準として必ず入れる
    for e, s in combos:
        acc, acc0 = PE.Acc(), PE.Acc()
        pred = {}; drops = 0; nfr = 0
        for c in clips:
            dec = decode_hyst(scores[c], e, s, assoc)
            frames = {k: [(cls, az, el, d) for cls, az, el, d, _ in v] for k, v in dec.items()}
            pred[c] = frames
            PE.clip_eval(acc, acc0, frames, gts[c], motion[c], minframe)
            dd, nn = flicker(frames, gts[c], minframe); drops += dd; nfr += nn
        det = PE.stats(acc, "距離クラス")
        n_pred = sum(v for k, v in acc.n_pred.items() if k in PE.DIST_CLS)
        det["prec20"] = 100.0 * det["n_tp20"] / max(n_pred, 1)
        det["f1_20"] = 2 * det["prec20"] * det["recall20"] / max(det["prec20"] + det["recall20"], 1e-9)
        g = N.notify_stats(clips, pred, evs_of, 1.0)
        row = dict(enter=e, stay=s, recall_any=det["recall_any"], recall20=det["recall20"], prec20=det["prec20"], f1=det["f1_20"],
                   az_med=det.get("az_med", float("nan")), rel_med=det.get("rel_med", float("nan")), n_pred=n_pred,
                   drops_per_100=100.0 * drops / max(nfr, 1), crit=g["crit"], strong=g["strong"], caut=g["caut"], safe=g["safe"], n_fire=g["n_fire"])
        grid.append(row)
        print(f"入り {e:.2f} 保持 {s:.2f}: 再現率20° {row['recall20']:.1f} 適合率 {row['prec20']:.1f} ちらつき {row['drops_per_100']:.2f}/100fr | 至近到達 {g['crit']:.1f} 安全抑制 {g['safe']:.1f} 発火 {g['n_fire']:,}", flush=True)

    base = next(x for x in grid if abs(x["enter"] - 0.5) < 1e-9 and abs(x["stay"] - 0.5) < 1e-9)

    def row(x, tag=""):
        return (f"| {x['enter']:.2f} | {x['stay']:.2f} | {tag} | {x['recall_any']:.1f}% | {x['recall20']:.1f}% | {x['prec20']:.1f}% | {x['f1']:.1f} | {x['az_med']:.1f}° | {x['rel_med']:.1f}% | "
                f"{x['drops_per_100']:.2f} | {x['crit']:.1f}% | {x['strong']:.1f}% | {x['caut']:.1f}% | {x['safe']:.1f}% | {x['n_fire']:,} |")
    hdr = ["| 入り | 保持 | 印 | 検出率 | 再現率(20°) | 適合率(20°) | F1 | 方位誤差中央 | 相対距離誤差中央 | ちらつき(落ち/100fr) | 至近到達 | 強到達 | 注意到達 | 安全抑制 | 発火数 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    # 選び方（宣言）: 知覚層 = 再現率(20°) が現行以上かつ適合率が現行 −1pt 以内の中で、ちらつきが最小。通知は参考
    cand = [x for x in grid if x["recall20"] >= base["recall20"] - 1e-9 and x["prec20"] >= base["prec20"] - 1.0]
    best = min(cand, key=lambda x: (x["drops_per_100"], -x["recall20"])) if cand else base
    # 本人 9/16 20:29「最も検出層の精度が良くなるペア」: 主 = F1(20°) 最大（同点はちらつきが小さい方）。副 = 再現率最大（適合率 ≥ 現行 −1pt）・適合率最大（再現率 ≥ 現行 −1pt）
    best_f1 = max(grid, key=lambda x: (round(x["f1"], 2), -x["drops_per_100"]))
    c_r = [x for x in grid if x["prec20"] >= base["prec20"] - 1.0]
    best_rec = max(c_r, key=lambda x: (x["recall20"], -x["drops_per_100"])) if c_r else base
    c_p = [x for x in grid if x["recall20"] >= base["recall20"] - 1.0]
    best_prec = max(c_p, key=lambda x: (x["prec20"], -x["drops_per_100"])) if c_p else base
    lines = [f"# 検出のヒステリシス（入り × 保持）の探索 — {out_md.stem}", "",
             f"plan= {plan.relative_to(ROOT)} ({split}) = {len(clips):,} 本 / GT= {meta.relative_to(ROOT)} / scores= {sdir.relative_to(ROOT)} / frame≥{minframe} / 前フレームとの対応 ±{assoc:.0f}° / 通知の閾値は現行", "",
             "## 1. 現行と、選ばれた組（知覚層の指標で選ぶ・通知層は参考）", "", *hdr, row(base, "現行（1 閾値 0.5）"),
             row(best_f1, "**F1(20°) が最大**（主の選び方・同点はちらつきが小さい方）"),
             row(best_rec, "再現率が最大（適合率 ≥ 現行 −1pt）"),
             row(best_prec, "適合率が最大（再現率 ≥ 現行 −1pt）"),
             row(best, "ちらつきが最小（再現率 ≥ 現行・適合率 ≥ 現行 −1pt）"), "",
             "## 2. 全格子", "", *hdr, *[row(x) for x in grid], "",
             "## 定義", "",
             "- 入り = この長さを超えたら「いる」。保持 = いったん「いる」になった相手（前のフレームで同じ種類・±assoc° 以内）は、この長さを下回るまで「いる」を続ける。入り == 保持 なら本番と同じ 1 閾値",
             "- 知覚層の列は _perception_eval と同じ定義（1 対 1・門なし）。ちらつき = 正解の相手ごとに検出が「あり→なし」に落ちた回数を、正解フレーム 100 個あたりに直したもの（小さいほど安定）",
             "- 通知層の列は統一採点器と同じ帰属（v4.3＋hold・通知の閾値は現行）",
             "- 開発 val での調整。選んだ値は確定評価の前に固定する"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grid[0].keys())); w.writeheader(); w.writerows(grid)
    print(f"-> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
