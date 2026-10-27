# -*- coding: utf-8 -*-
"""本体 §2-2 の測り（2026-10-26）: 踏切をゲートに入れると、踏切の正解の枠がどれだけ消えるか（電車の重なる本と電車の無い本で分けて）。
plan の学習の本（雨なし）から、電車＋踏切の本 n 本と踏切だけ（車は入る）の本 n 本を順に選び、v29 の道と v30 の入口で描いて、
踏切（class 5・正解のある音源だけ）の正解の枠の数を比べる。描くのは手元の CPU・一時フォルダ（結果だけ md に書く）。評価の本は使わない。

使い方: python scripts/v30_crossing_gate_count.py [--n 10] [--out <md>]（既定の md はスクラッチの外に書かない＝標準出力だけ）
--bells-only（Fable 10/26 17:4x）: **警報機だけ**で数える。電車の車両も class は crossing（params.train_car）で先頭の車両などに正解が付くので、既定の数え方
  （metadata の class 5 の行）は車両の行も含んでいた（10/26 の 78.0%／31.8% は「車両込み」）。この形は 対象 = 正解のある crossing のうち train_car でないもの、
  電車あり／なし = 同じ本に train_car の音源があるか、で本を選び（scenario は問わない・雨なし・train だけ）、scene.json の音源ごとの n_label_frames を足す。
  全部消えた本（v29 で正解のある警報機が v30 で 0 枠）も数える。
--v30-gate band|broadband（決め直し 4・本人 10/26 19:32）: v30 の既定は帯域ゲート（band・一番有利な 1/3 オクターブ帯の SN 比 ≥ 0 dB）。broadband は旧の広帯域 −5 の道
  （FEATURES["band_gate"] = False・10/26 の 93.3%／31.8% はこれ）。既定 = band。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pick(rows, v30, n):
    with_train, no_train = [], []
    for r in rows:
        if not r["clip_id"].startswith("train_syn") or r["rain"] or r["scenario"] not in ("v12train", "v11core"):
            continue
        s = v30.sample_scene(r)
        if not any(x["class"] == "crossing" and not x.get("no_label") for x in s["sources"]):
            continue
        has_train = any(x.get("no_label") for x in s["sources"])
        (with_train if has_train else no_train).append(r)
        if len(with_train) >= n and len(no_train) >= n:
            break
    return with_train[:n], no_train[:n]


def is_bell(x: dict) -> bool:
    return x["class"] == "crossing" and not x.get("no_label") and not (x.get("params") or {}).get("train_car")


def pick_bells(rows, v30, n):
    with_train, no_train = [], []
    for r in rows:
        if not r["clip_id"].startswith("train_syn") or r["rain"] or (r.get("twin_of") or "").strip():
            continue
        s = v30.sample_scene(r)
        if not any(is_bell(x) for x in s["sources"]):
            continue
        has_train = any((x.get("params") or {}).get("train_car") for x in s["sources"])
        grp = with_train if has_train else no_train
        if len(grp) < n:
            grp.append(r)
        if len(with_train) >= n and len(no_train) >= n:
            break
    return with_train[:n], no_train[:n]


def count_bells(ds: Path, clip: str) -> int:
    import json
    sc = json.loads((ds / "work" / clip / "scene.json").read_text(encoding="utf-8"))
    return sum(int(x.get("n_label_frames") or 0) for x in sc["sources"] if is_bell(x))


def count(ds: Path, clip: str) -> int:
    return sum(1 for ln in (ds / "metadata" / f"{clip}.csv").read_text().splitlines() if ln.strip() and int(ln.split(",")[1]) == 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bells-only", action="store_true", help="警報機だけで数える（電車の車両 params.train_car を外す・Fable 10/26 17:4x）")
    ap.add_argument("--v30-gate", choices=("band", "broadband"), default="band", help="v30 のゲート（既定 band = 帯域ゲート 0 dB・決め直し 4／broadband = 旧の広帯域 −5）")
    a = ap.parse_args()
    import _run_v2final_gen as g
    import step11_v30_render as v30
    rows = g.load_plan()
    v30.FEATURES["band_gate"] = (a.v30_gate == "band")
    groups = dict(zip(("電車あり", "電車なし"), (pick_bells if a.bells_only else pick)(rows, v30, a.n)))
    cnt = count_bells if a.bells_only else count
    tmp = Path(tempfile.mkdtemp(prefix="v30_crossing_count_"))
    lines = [f"# 踏切のゲートで消える正解の枠（本体 §2-2・v29 の道 → v30 の入口・試し描き {2 * a.n} 本・手元の CPU"
             + ("・**警報機だけ**〔電車の車両 train_car を外す〕" if a.bells_only else "・class 5 の行〔電車の車両の行も込み〕")
             + ("・v30 = **帯域ゲート 0 dB**（決め直し 4）" if a.v30_gate == "band" else "・v30 = 広帯域 −5") + "）", ""]
    try:
        for name, rs in groups.items():
            n_old = n_new = 0
            per = []
            n_all_lost = 0
            for r in rs:
                for ds, fn in ((tmp / "v29", g.v19r.generate_clip), (tmp / "v30", v30.generate_clip)):
                    g.point_to(ds)
                    fn(dict(r))
                o, nw = cnt(tmp / "v29", r["clip_id"]), cnt(tmp / "v30", r["clip_id"])
                n_old += o; n_new += nw
                n_all_lost += int(o > 0 and nw == 0)
                per.append(f"{r['clip_id']} {o}→{nw}")
            lost = n_old - n_new
            lines.append(f"- {name}（{len(rs)} 本）: 踏切の正解の枠 {n_old} → {n_new}・**消えた割合 {100.0 * lost / max(n_old, 1):.1f}%**（増えた本も差し引き後）"
                         + (f"・全部消えた本 {n_all_lost}/{len(rs)}" if a.bells_only else ""))
            lines.append("  - " + "・".join(per))
            print(lines[-2], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines += ["", "- 定義: v29 = 踏切はゲートなし（鳴っている区間を全部正解）・v30 = " + ("一番有利な 1/3 オクターブ帯の SN 比 ≥ 0 dB" if a.v30_gate == "band" else "広帯域の線 −5 dB")
              + "・背景 = 暗騒音＋雨＋同じ場面の自分以外の全 stem（電車を含む）・隙間埋め 2 枠。",
              "- 電車なしの本も車が背景に入るので、車の大きい枠では踏切の正解が消えうる（線 −5 で増える分と差し引き）。"]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
