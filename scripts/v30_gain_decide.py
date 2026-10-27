# -*- coding: utf-8 -*-
"""入力の倍率の決め方（本体 §3-2・§9-1 の 4・§9-4 の 2・変更票 #75 ②・決め直し 2 の 4・決め直し 3 の 2・2026-10-26）。本人の裁量なし（結果は報告だけ）。

F = 基準学習 20 epoch 目（epoch_019）の val のクラス平均 F: クラスごとのフレーム単位の検出 F（正解 = metadata_dist にそのクラスの行がある枠・
    予測 = 因果推論の csv にそのクラスの行がある枠〔入り 0.5 の固定・ヒステリシス無し〕）を、**フレーム 40〜99**・9 クラスで足し合わせ、
    正解の無いクラスは平均から外す（`scripts/_baseline_select_gain.py` の f_per_class／mean_f と同じ式）。F は 0〜1・1 pt = 0.01。
規則:
  1. seed 1 で最良の F を出し、**g1 = 最良 − 1 pt 以内で一番小さい倍率**。
  2. **g2 = g1 を除く候補のうち seed 1 の F が最良のもの**（同点は倍率の小さい方・1 pt の内外を問わない）。
  3. seed 2 で g1・g2 を回し、**2 seed の平均 F で g2 が g1 より 1 pt 超良いときだけ g2**、それ以外は g1。
  4. 全候補が同点なら最小の倍率（= 規則 1 で g1 になる）。
出力 decision.json: 規則の写し・候補数・各 seed・各倍率の未丸めの F とクラス別・分岐・最終の倍率。参考の列（決めには使わない）: 正解の無い本での発火の枠数。
使い方: python scripts/v30_gain_decide.py --gt <metadata_dist dir> --manifest <val の csv（clip_id 列）> --seed1 0=<csv> 20=<csv> 30=<csv> 40=<csv>
        [--seed2 <g1>=<csv> <g2>=<csv>] --out out/v30_gain_compare/decision.json
        （seed2 を渡さないと規則 1・2 までで止まり、seed 2 で回す {g1, g2} を出す）
止める: seed 1 の倍率が --candidates（既定 0,20,30,40）とそろわない（候補の欠損）・F が非有限（§9-3 の受け入れ）
試験: scripts/test_gain_patch_v30.py の G1〜G3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

N_CLS, NFR, F0 = 9, 100, 40
PT = 0.01
RULE = ("F = epoch_019 の val のクラス平均 F（フレーム 40〜99・9 クラス・正解の無いクラスは外す・入り 0.5・ヒステリシス無し）。"
        "g1 = seed 1 の最良 − 1 pt 以内で最小の倍率／g2 = g1 を除く seed 1 の最良（同点は小さい倍率）／最終 = 2 seed の平均 F で g2 が g1 より 1 pt 超良いときだけ g2・ほかは g1。"
        "全候補同点なら最小。本人の裁量なし（本体 §9-4 の 2）")


def gt_active(gt_dir: Path, clip: str) -> np.ndarray:
    a = np.zeros((NFR, N_CLS), dtype=bool)
    p = gt_dir / f"{clip}.csv"
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            q = ln.strip().split(",")
            if len(q) >= 2 and q[0]:
                k, c = int(float(q[0])), int(float(q[1]))
                if 0 <= k < NFR and 0 <= c < N_CLS:
                    a[k, c] = True
    return a


def pred_active(pred_csv: Path, clips: set) -> dict:
    """因果推論の csv（7 列 clip,frame,cls,track,az,el,dist）→ {clip: (NFR, 9) bool}。"""
    out = defaultdict(lambda: np.zeros((NFR, N_CLS), dtype=bool))
    with open(pred_csv, encoding="utf-8") as f:
        for ln in f:
            q = ln.strip().split(",")
            if len(q) >= 3 and q[0] in clips:
                k, c = int(q[1]), int(q[2])
                if 0 <= k < NFR and 0 <= c < N_CLS:
                    out[q[0]][k, c] = True
    return out


def class_mean_f(gt: dict, pr: dict, clips: list) -> tuple[float | None, dict, int]:
    tot = {c: [0, 0, 0] for c in range(N_CLS)}
    n_fire_nogt = 0
    for cl in clips:
        g = gt[cl][F0:]
        p = pr.get(cl, np.zeros((NFR, N_CLS), dtype=bool))[F0:]
        for c in range(N_CLS):
            tot[c][0] += int((g[:, c] & p[:, c]).sum()); tot[c][1] += int((~g[:, c] & p[:, c]).sum()); tot[c][2] += int((g[:, c] & ~p[:, c]).sum())
        if not g.any():
            n_fire_nogt += int(p.any(axis=1).sum())
    fs = {c: 2 * tp / (2 * tp + fp + fn) for c, (tp, fp, fn) in tot.items() if tp + fn > 0}
    return (float(np.mean(list(fs.values()))) if fs else None), fs, n_fire_nogt


def _check_finite(f: dict, name: str) -> None:
    bad = {g: v for g, v in f.items() if v is None or not np.isfinite(v)}
    if bad:
        raise ValueError(f"{name} の F が非有限: {bad}（止める・§9-3）")


def rule_g1_g2(f1: dict) -> tuple[float, float | None, str]:
    """seed 1 の {倍率: F} → (g1, g2, 分岐の説明)。"""
    if not f1:
        raise ValueError("seed 1 の候補が無い")
    _check_finite(f1, "seed 1")
    gains = sorted(f1)
    best = max(f1[g] for g in gains)
    if all(f1[g] == f1[gains[0]] for g in gains):
        return gains[0], None, "全候補同点 → 最小の倍率（g2 なし）"
    g1 = min(g for g in gains if f1[g] >= best - PT - 1e-12)
    rest = [g for g in gains if g != g1]
    g2 = max(rest, key=lambda g: (f1[g], -g))
    n_in = sum(1 for g in gains if f1[g] >= best - PT - 1e-12)
    return g1, g2, f"1 pt 以内の候補 {n_in} 個 → g1 = 最小 {g1:g}・g2 = g1 を除く最良 {g2:g}"


def decide(f1: dict, f2: dict | None) -> dict:
    g1, g2, why = rule_g1_g2(f1)
    out = {"g1": g1, "g2": g2, "branch_seed1": why}
    if g2 is None:
        out.update(final=g1, branch_final="全候補同点 → 最小（seed 2 は回さない）")
        return out
    if f2 is None:
        out.update(final=None, branch_final=f"seed 2 で {{g1 = {g1:g}, g2 = {g2:g}}} を回す（--seed2）")
        return out
    if not ({g1, g2} <= set(f2)):
        raise ValueError(f"seed 2 に g1 = {g1:g}・g2 = {g2:g} の両方が要る（渡されたのは {sorted(f2)}）")
    _check_finite({g: f2[g] for g in (g1, g2)}, "seed 2")
    m1, m2 = (f1[g1] + f2[g1]) / 2.0, (f1[g2] + f2[g2]) / 2.0
    better = (m2 - m1) > PT + 1e-12
    out.update(mean_f_g1=m1, mean_f_g2=m2, final=(g2 if better else g1),
               branch_final=(f"2 seed の平均で g2 が g1 より {100 * (m2 - m1):.3f} pt 良い（> 1 pt）→ g2" if better
                             else f"2 seed の平均の差 {100 * (m2 - m1):+.3f} pt（≤ 1 pt）→ g1"))
    return out


def _pairs(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        g, p = it.split("=", 1)
        out[float(g)] = Path(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="入力の倍率の決め方（本体 §9-4 の 2）")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed1", nargs="+", required=True)
    ap.add_argument("--seed2", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", default="0,20,30,40", help="seed 1 にそろうべき倍率（欠けていれば止める）")
    a = ap.parse_args(argv)
    cands = sorted(float(x) for x in a.candidates.split(","))
    if sorted(_pairs(a.seed1)) != cands:
        raise SystemExit(f"seed 1 の倍率 {sorted(_pairs(a.seed1))} が候補 {cands} とそろわない（候補の欠損・止める）")
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    gt = {c: gt_active(a.gt, c) for c in clips}
    res = {"rule": RULE, "n_clips": len(clips), "seed1": {}, "seed2": {}}
    f1, f2 = {}, {}
    for seed, items, store in (("seed1", a.seed1, f1), ("seed2", a.seed2, f2)):
        for g, p in sorted(_pairs(items).items()):
            mf, fs, nf = class_mean_f(gt, pred_active(p, set(clips)), clips)
            if mf is None:
                raise SystemExit(f"{seed} {g:g} dB: F が出ない（正解のあるクラスが無い）")
            store[g] = mf
            res[seed][f"{g:g}"] = {"pred": str(p), "mean_f": mf, "f_per_class": {str(k): v for k, v in fs.items()}, "n_fire_frames_nogt": nf}
    res["n_candidates"] = len(f1)
    res["candidates"] = cands
    try:
        dec = decide(f1, f2 if f2 else None)
    except ValueError as e:
        raise SystemExit(str(e))
    res["decision"] = dec
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dec, ensure_ascii=False))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
