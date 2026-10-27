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
  参考の列の追加（2026-10-27 Fable 22:1x の直す所 1・#75 ② の併記「−5〜0 dB の枠の再現率」を帯域ゲート〔決め直し 4〕の版に）: --masks と --scenes を渡すと、
  各倍率の reference に (1) 際の枠の再現率 = 正解のある枠のうち、そのクラスの正解を付ける音源（no_label・電車の車両を除く）の帯の SN 比
  〔masks の snr_band_max_db・band_valid = 1〕の最大が 0 以上 +3 dB 未満の枠を、予測が拾った割合（クラス別と全体・フレーム 40〜99）と
  (2) 正解の無い本で 1 回以上発火した本の数 を足す。どちらも決めには使わない（規則と F は変えない）。
使い方: python scripts/v30_gain_decide.py --gt <metadata_dist dir> --manifest <val の csv（clip_id 列）> --seed1 0=<csv> 20=<csv> 30=<csv> 40=<csv>
        [--seed2 <g1>=<csv> <g2>=<csv>] [--masks <ds>/masks --scenes <ds>/work] --out out/v30_gain_compare/decision.json
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


EDGE_LO, EDGE_HI = 0.0, 3.0           # 際の枠 = 帯の SN 比 [0, +3) dB（正解の線 0 dB のすぐ上）


def edge_frames(masks_dir: Path, scenes_dir: Path | None, clip: str, gt: np.ndarray) -> np.ndarray:
    """際の枠（参考）: 正解のある (枠, クラス) のうち、正解を付ける音源の帯の SN 比の最大が [EDGE_LO, EDGE_HI) のもの → (NFR, 9) bool。
    正解を付けない音源（scene の no_label・電車の車両 params.train_car）は除く（scenes が無ければ除けないので全部の行を使う）。"""
    skip = set()
    if scenes_dir is not None:
        sp = scenes_dir / clip / "scene.json"
        if sp.is_file():
            for i, s in enumerate(json.loads(sp.read_text(encoding="utf-8")).get("sources", [])):
                if s.get("no_label") or (s.get("params") or {}).get("train_car"):
                    skip.add(i)
    best = np.full((NFR, N_CLS), -np.inf)
    mp = masks_dir / f"{clip}.csv"
    if mp.is_file():
        with open(mp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "snr_band_max_db" not in r or str(r.get("band_valid", "1")) != "1":
                    continue
                if "src" in r and r["src"] != "" and int(r["src"]) in skip:
                    continue
                k, c, v = int(r["frame"]), int(r["class"]), float(r["snr_band_max_db"])
                if 0 <= k < NFR and 0 <= c < N_CLS and v > best[k, c]:
                    best[k, c] = v
    return gt & (best >= EDGE_LO) & (best < EDGE_HI)


def edge_recall(edges: dict, pr: dict, clips: list) -> dict:
    """際の枠の再現率（フレーム 40〜99）: クラス別 {c: [拾った, 際の枠の数, 再現率]} と全体。"""
    hit, tot = np.zeros(N_CLS, dtype=np.int64), np.zeros(N_CLS, dtype=np.int64)
    for cl in clips:
        e = edges[cl][F0:]
        p = pr.get(cl, np.zeros((NFR, N_CLS), dtype=bool))[F0:]
        hit += (e & p).sum(axis=0); tot += e.sum(axis=0)
    per = {str(c): {"hit": int(hit[c]), "n_edge": int(tot[c]), "recall": (float(hit[c] / tot[c]) if tot[c] else None)} for c in range(N_CLS)}
    return {"band_db": [EDGE_LO, EDGE_HI], "per_class": per, "all": {"hit": int(hit.sum()), "n_edge": int(tot.sum()),
            "recall": (float(hit.sum() / tot.sum()) if tot.sum() else None)}}


def fire_clips_nogt(gt: dict, pr: dict, clips: list) -> int:
    """正解の無い本（フレーム 40〜99 に正解が 1 つも無い）で、1 回以上発火した本の数（参考）。"""
    n = 0
    for cl in clips:
        if not gt[cl][F0:].any() and pr.get(cl, np.zeros((NFR, N_CLS), dtype=bool))[F0:].any():
            n += 1
    return n


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
    ap.add_argument("--masks", type=Path, default=None, help="参考: 際の枠の再現率に使う masks の置き場（決めには使わない）")
    ap.add_argument("--scenes", type=Path, default=None, help="参考: 正解を付けない音源を除くための work の置き場（--masks と一緒に）")
    a = ap.parse_args(argv)
    if a.masks is not None and a.scenes is None:
        raise SystemExit("--masks には --scenes も渡す（正解を付けない音源〔電車の車両など〕を除くため）")
    cands = sorted(float(x) for x in a.candidates.split(","))
    if sorted(_pairs(a.seed1)) != cands:
        raise SystemExit(f"seed 1 の倍率 {sorted(_pairs(a.seed1))} が候補 {cands} とそろわない（候補の欠損・止める）")
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    gt = {c: gt_active(a.gt, c) for c in clips}
    edges = {c: edge_frames(a.masks, a.scenes, c, gt[c]) for c in clips} if a.masks is not None else None
    res = {"rule": RULE, "n_clips": len(clips), "seed1": {}, "seed2": {}}
    if edges is not None:
        res["reference_note"] = ("reference は参考（決めには使わない・規則と F は変えない）: 際の枠 = 正解のある枠で、正解を付ける音源の帯の SN 比"
                                 f"（masks の snr_band_max_db の最大・band_valid = 1）が [{EDGE_LO:g}, {EDGE_HI:g}) dB の枠・フレーム 40〜99。"
                                 "#75 ② の「−5〜0 dB の枠の再現率」を帯域ゲート（決め直し 4）に合わせた版（Fable 10/27 22:1x）。"
                                 "n_fire_clips_nogt = 正解の無い本で 1 回以上発火した本の数")
        res["reference_source"] = {"masks": str(a.masks), "scenes": str(a.scenes)}
    f1, f2 = {}, {}
    for seed, items, store in (("seed1", a.seed1, f1), ("seed2", a.seed2, f2)):
        for g, p in sorted(_pairs(items).items()):
            pr = pred_active(p, set(clips))
            mf, fs, nf = class_mean_f(gt, pr, clips)
            if mf is None:
                raise SystemExit(f"{seed} {g:g} dB: F が出ない（正解のあるクラスが無い）")
            store[g] = mf
            res[seed][f"{g:g}"] = {"pred": str(p), "mean_f": mf, "f_per_class": {str(k): v for k, v in fs.items()}, "n_fire_frames_nogt": nf}
            if edges is not None:
                res[seed][f"{g:g}"]["reference"] = {"edge_recall": edge_recall(edges, pr, clips), "n_fire_clips_nogt": fire_clips_nogt(gt, pr, clips)}
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
