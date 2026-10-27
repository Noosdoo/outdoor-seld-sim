# -*- coding: utf-8 -*-
"""背景の形の小さな比べ（決め直し 5・本人 10/26 20:03・仕様 §9-7 の 4）の決め方。本人の裁量なし（結果は Fable が本人に報告）。

主 = val（(ii) の 12,420 本）のクラス平均 F（scripts/v30_gain_decide.py の class_mean_f と同じ式: フレーム 40〜99・9 クラス・正解 = metadata_dist の行がある枠・
     予測 = 因果推論の csv の行がある枠〔入り 0.5・ヒステリシス無し〕・正解の無いクラスは平均から外す）。F は 0〜1・1 pt = 0.01。
規則:
  1. F が最大の通り（off／i／ii）を採る
  2. 最大との差が 1 pt 以内（≤ 0.01・丸めない）の通りが複数あり、その中に ii があれば ii（文献の両端まで振る方を選ぶ）
  3. 1 pt 以内が複数でも ii が入っていなければ、最大の通り（同点は ii → i → off の順）
参考（選択には使わない・decision.json に併記）: クラス別の F・正解の無い本の発火の枠数・家の 13 本で鳴った本の数（--home13 off=<n> i=<n> ii=<n>）
出力 decision.json: 規則の写し・3 通りの未丸めの F とクラス別・分岐・最終
使い方: python scripts/v30_bgcmp_decide.py --gt <比べ用の val の metadata_dist（out/dataset_outdoor_siren_v2final_v30bg_val/metadata_dist）> --manifest <val の名簿>
        --pred off=<csv> i=<csv> ii=<csv> [--home13 off=7 i=8 ii=9] --out <json>
試験: scripts/test_v30_bgcmp.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import v30_gain_decide as GD  # noqa: E402

VARIANTS = ("off", "i", "ii")
PREF = ("ii", "i", "off")            # 同点のときの順
PT = 0.01
RULE = ("主 = val のクラス平均 F（フレーム 40〜99・9 クラス・正解の無いクラスは外す・入り 0.5・ヒステリシス無し・v30_gain_decide と同じ式）。"
        "最大を採る・最大との差 1 pt（0.01）以内が複数でその中に ii があれば ii・ii が入っていなければ最大（同点は ii → i → off）。"
        "クラス別 F・正解の無い本の発火・家の 13 本の鳴った本数は参考（選択に使わない）。本人の裁量なし（仕様 §9-7）")


def decide(f: dict) -> dict:
    """{通り: F} → 決定。非有限・欠けは止める。"""
    import math
    if set(f) != set(VARIANTS):
        raise ValueError(f"3 通り（{VARIANTS}）がそろっていない: {sorted(f)}")
    if not all(isinstance(v, float) and math.isfinite(v) for v in f.values()):
        raise ValueError(f"F が非有限: {f}")
    fmax = max(f.values())
    near = [v for v in VARIANTS if f[v] >= fmax - PT - 1e-12]
    best = min((v for v in VARIANTS if f[v] == fmax), key=PREF.index)
    if len(near) >= 2 and "ii" in near:
        return {"final": "ii", "near_1pt": near, "best": best, "branch": f"最大 {best}（{fmax:.5f}）から 1 pt 以内が {near} → ii"}
    return {"final": best, "near_1pt": near, "best": best,
            "branch": (f"1 pt 以内は {best} だけ → {best}" if len(near) == 1 else f"1 pt 以内が {near} だが ii が入っていない → 最大の {best}")}


def _pairs(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--home13", nargs="*", default=None, help="参考: 家の 13 本で鳴った本の数（off=<n> i=<n> ii=<n>）・選択には使わない")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    preds = {k: Path(v) for k, v in _pairs(a.pred).items()}
    if set(preds) != set(VARIANTS):
        raise SystemExit(f"--pred は off・i・ii の 3 つ（渡されたのは {sorted(preds)}）")
    with open(a.manifest, newline="", encoding="utf-8-sig") as fh:
        clips = [r["clip_id"] for r in csv.DictReader(fh)]
    gt = {c: GD.gt_active(a.gt, c) for c in clips}
    res = {"rule": RULE, "n_clips": len(clips), "variants": {}}
    f = {}
    for v in VARIANTS:
        mf, fs, nf = GD.class_mean_f(gt, GD.pred_active(preds[v], set(clips)), clips)
        if mf is None:
            raise SystemExit(f"{v}: F が出ない（正解のあるクラスが無い）")
        f[v] = mf
        res["variants"][v] = {"pred": str(preds[v]), "mean_f": mf, "f_per_class": {str(k): x for k, x in fs.items()}, "n_fire_frames_nogt": nf}
    try:
        res["decision"] = decide(f)
    except ValueError as e:
        raise SystemExit(str(e))
    if a.home13:
        res["reference_home13_rang"] = {k: int(v) for k, v in _pairs(a.home13).items()}
        res["reference_note"] = "家の 13 本で鳴った本の数は参考（選択に使わない・本人 20:03「家の 13 本は選択ではなく参考」）"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["decision"], ensure_ascii=False))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
