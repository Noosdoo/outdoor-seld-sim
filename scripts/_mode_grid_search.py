# -*- coding: utf-8 -*-
"""2 設定（控えめ／厳格）の閾値の組を探す（2026-09-16・本人「最も精度が良くなる 2 つの閾値のペアを探して」）。

格子 = 検出の閾値（矢印の長さ）× 通知の距離閾値の倍率。各組で、度合いから予測を組み立て直し（統合は本番と同じ）、
通知規則（v4.3＋hold）の距離閾値 4 つを倍率で動かして、統一採点器と同じ帰属で 至近到達・強到達・注意到達・安全抑制・発火数 を出す。
選び方（宣言）:
  控えめ = 安全抑制が現行（検出 0.5・×1.0）以上の組の中で、至近到達が最大のもの（同点なら安全抑制が高い方）
  厳格   = 至近到達が最大の組の中で、安全抑制が最大のもの。ただし安全抑制の下限（--safe-floor・既定 80%）を割る組は除く
  参考   = 至近到達と安全抑制の調和平均が最大の組（釣り合いの 1 点）
これは開発 val での調整（確定評価の前に固定する値を探す作業）。確定評価では fold20 に一度だけ掛ける。

使い方:
  python scripts/_mode_grid_search.py <出力md> --plan <assignment.csv> [--split fold2] --meta <GTディレクトリ> --scores <scores dir>
      [--det 0.2,0.3,0.4,0.5,0.6,0.7] [--scales 0.8,0.9,1.0,1.1,1.2,1.3,1.5] [--safe-floor 80] [--minframe 40] [--csv <格子のcsv>]
"""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

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
S = N.S


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    split = arg(argv, "--split", "fold2")
    meta = ROOT / arg(argv, "--meta")
    sdir = ROOT / arg(argv, "--scores")
    dets = [float(x) for x in arg(argv, "--det", "0.2,0.3,0.4,0.5,0.6,0.7").split(",")]
    scales = [float(x) for x in arg(argv, "--scales", "0.8,0.9,1.0,1.1,1.2,1.3,1.5").split(",")]
    safe_floor = float(arg(argv, "--safe-floor", "80"))
    minframe = int(arg(argv, "--minframe", "40"))
    out_csv = arg(argv, "--csv", "")
    out_md = Path(argv[0])

    rows = [r for r in csv.DictReader(open(plan, newline="", encoding="utf-8")) if r.get("split", split) == split]
    clips = [r["clip_id"] for r in rows]
    D.check_manifest(sdir, clips)
    scores = {c: D.load_scores(sdir / f"{c}.csv") for c in clips}
    assert S.DG.v4 is not S.V43.v4
    evs_of = {c: N.events_of(meta, c) for c in clips}

    grid = []
    for th in dets:
        pred = {}
        for c in clips:
            dec = D.decode(scores[c], th)
            pred[c] = {k: [(cls, az, el, d) for cls, az, el, d, _ in v] for k, v in dec.items()}
        for sc in scales:
            g = N.notify_stats(clips, pred, evs_of, sc)
            grid.append(dict(det=th, scale=sc, crit=g["crit"], strong=g["strong"], caut=g["caut"], safe=g["safe"], n_fire=g["n_fire"], lead=g["lead"]))
            print(f"det {th:.1f} × {sc:.1f}: 至近到達 {g['crit']:.1f} 強 {g['strong']:.1f} 注意 {g['caut']:.1f} 安全抑制 {g['safe']:.1f} 発火 {g['n_fire']:,}", flush=True)

    base = next(x for x in grid if abs(x["det"] - 0.5) < 1e-9 and abs(x["scale"] - 1.0) < 1e-9)
    hm = lambda x: 2 * x["crit"] * x["safe"] / max(x["crit"] + x["safe"], 1e-9)
    cand_h = [x for x in grid if x["safe"] >= base["safe"] - 1e-9]
    hikaeme = max(cand_h, key=lambda x: (x["crit"], x["safe"])) if cand_h else base
    cand_g = [x for x in grid if x["safe"] >= safe_floor]
    best_crit = max(x["crit"] for x in cand_g) if cand_g else None
    genkaku = max([x for x in cand_g if abs(x["crit"] - best_crit) < 1e-9], key=lambda x: x["safe"]) if cand_g else None
    balance = max(grid, key=hm)
    # パレート前線（至近到達・安全抑制の両方で他に負けない組）
    front = [x for x in grid if not any((y["crit"] >= x["crit"] and y["safe"] >= x["safe"]) and (y["crit"] > x["crit"] or y["safe"] > x["safe"]) for y in grid)]
    front.sort(key=lambda x: -x["safe"])

    def row(x, tag=""):
        return f"| {x['det']:.1f} | ×{x['scale']:.1f} | {tag} | {x['crit']:.1f}% | {x['strong']:.1f}% | {x['caut']:.1f}% | {x['safe']:.1f}% | {x['n_fire']:,} | {x['lead']:.2f}s | {hm(x):.1f} |"
    hdr = ["| 検出 | 通知 | 印 | 至近到達 | 強到達 | 注意到達 | 安全抑制 | 発火数 | リード中央 | 調和平均 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines = [f"# 2 設定の閾値の組を探す — {out_md.stem}", "",
             f"plan= {plan.relative_to(ROOT)} ({split}) = {len(clips):,} 本 / GT= {meta.relative_to(ROOT)} / scores= {sdir.relative_to(ROOT)} / frame≥{minframe} / "
             f"格子 = 検出 {dets} × 通知 {scales} / 安全抑制の下限 {safe_floor}%", "",
             "## 1. 選ばれた組（宣言した選び方）", "", *hdr,
             row(base, "現行（検出 0.5・×1.0）"),
             row(hikaeme, "**控えめ**（安全抑制 ≥ 現行 の中で至近到達が最大）"),
             *( [row(genkaku, f"**厳格**（安全抑制 ≥ {safe_floor:.0f}% の中で至近到達が最大）")] if genkaku else []),
             row(balance, "参考: 調和平均が最大"), "",
             "## 2. パレート前線（至近到達と安全抑制の両方で他に負けない組・安全抑制の高い順）", "", *hdr, *[row(x) for x in front], "",
             "## 3. 全格子", "", *hdr, *[row(x) for x in grid], "",
             "## 定義", "",
             "- 各組: 保存した度合いから検出の閾値で予測を組み立て直し（統合は本番と同じ）、通知規則の距離閾値 4 つ（強 1.5・中 3.2・予測最接近 1.3/1.6 m）を倍率で動かして、統一採点器と同じ帰属で採点",
             "- 調和平均 = 至近到達と安全抑制の調和平均（釣り合いの 1 点の目安）",
             "- これは開発 val での調整。選んだ値は確定評価の前に固定し、確定評価（fold20）に一度だけ掛ける"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grid[0].keys())); w.writeheader(); w.writerows(grid)
    print(f"-> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
