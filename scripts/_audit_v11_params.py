# -*- coding: utf-8 -*-
"""データ精査 自動集計（読み取り専用）: 計画書A3/B1-B8の分布実測。

正= md/audit/データ精査計画_2026-08-05.md。スライドP15/P16の公称値と
scene.json実体（7,200本＋評価3,342本）を突き合わせる。出力は標準出力＋
out/audit_v11_params_2026-08-05.md（新規）。データは一切変更しない。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "out" / "dataset_outdoor_siren_v11"
DSE = ROOT / "out" / "dataset_outdoor_siren_v11_eval"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

L = []


def p(s=""):
    print(s)
    L.append(s)


def load_scenes(ds):
    out = {}
    for sj in sorted((ds / "work").glob("*mix*/scene.json")):
        out[sj.parent.name] = json.loads(sj.read_text())
    return out


def rng_str(vals, fmt="{:.1f}"):
    v = np.asarray(vals, float)
    return (fmt + "〜" + fmt + " (中央" + fmt + ")").format(v.min(), v.max(),
                                                          float(np.median(v)))


def main():
    scenes = load_scenes(DS)
    p(f"# v11パラメータ精査（学習系7,200本、実測 {len(scenes)}本）")

    # ---- B1 車台数 / B2 警告音個数 / B3 マイク ----
    carn = Counter(); warnn = Counter(); mic = Counter(); wspd = []
    tiers = Counter(); dbas = []; seeds = defaultdict(set)
    empty_meta = 0; bg_only = 0; zero_label_with_src = []
    WARN = {"siren", "horn", "backup_beep", "bike_bell", "crossing"}
    for name, s in scenes.items():
        srcs = s["sources"]
        cars = [x for x in srcs if x["class"] == "car_drive"]
        warns = [x for x in srcs if x["class"] in WARN]
        carn[len(cars)] += 1
        warnn[len(warns)] += 1
        mic[s["mic"]["motion"]] += 1
        if s["mic"]["motion"] == "walk":
            wspd.append(s["mic"]["walk_speed_mps"])
        if cars:
            t = cars[0].get("danger_tier")
            tiers[t] += 1
        dbas.append(s["noise"]["dba"])
        seeds[name.split("_")[0]].add(s["row"]["seed"])
        if not srcs:
            bg_only += 1
        n_rows = sum(x.get("n_label_frames", 0) for x in srcs)
        meta = DS / "metadata" / f"{name}.csv"
        if meta.stat().st_size == 0:
            empty_meta += 1
            if srcs:
                zero_label_with_src.append(name)

    tot = len(scenes)
    p("\n## B1 車の台数分布（公称 0/1/2/3台 = 18/47/23/12%）")
    for k in sorted(carn):
        p(f"- {k}台: {carn[k]}本 ({carn[k]/tot:.1%})")
    p("\n## B2 警告音の個数（公称 0/1/2個 = 45/40/15%）")
    for k in sorted(warnn):
        p(f"- {k}個: {warnn[k]}本 ({warnn[k]/tot:.1%})")
    p("\n## B3 マイク（公称 静止50%/歩行50%）")
    for k, v in mic.items():
        p(f"- {k}: {v}本 ({v/tot:.1%})")
    if wspd:
        p(f"- 歩行速度: {rng_str(wspd, '{:.2f}')} m/s")
    p("\n## B4 主要車の危険層（公称 近/中/遠 均等）")
    for k, v in tiers.most_common():
        p(f"- {k}: {v}本")
    p("\n## B5 背景騒音 dB(A)（公称 40〜65）")
    p(f"- 実測: {rng_str(dbas)}")
    p("\n## B6 最低本数保証")
    no_car_warn = sum(1 for s in scenes.values()
                      if not any(x['class'] == 'car_drive' for x in s['sources'])
                      and any(x['class'] in WARN for x in s['sources']))
    none_at_all = sum(1 for s in scenes.values() if not s["sources"])
    multi_car = sum(v for k, v in carn.items() if k >= 2)
    p(f"- 車なし＋警告音あり: {no_car_warn}本（公称≥400）")
    p(f"- 対象音なし＋背景のみ: {none_at_all}本（公称≥300）")
    p(f"- 車2台以上: {multi_car}本（公称≥1,500）")
    p("\n## B7 fold間シード独立")
    for f in sorted(seeds):
        p(f"- {f}: {len(seeds[f])}種")
    inter = set.intersection(*seeds.values()) if len(seeds) > 1 else set()
    p(f"- fold間で共有されるseed: {len(inter)}個 {'⚠️' if inter else '(独立)'}")
    p("\n## B8 空メタデータの員数")
    p(f"- 空メタ: {empty_meta}本 / 音源ゼロ(背景のみ): {bg_only}本 / "
      f"差分（音源はあるがラベル0行）: {len(zero_label_with_src)}本")
    for n in zero_label_with_src[:10]:
        s = scenes[n]
        cls = [(x['class'], x.get('audible_frac')) for x in s['sources']]
        p(f"  - {n}: {cls}")

    # ---- A3 クラス別パラメータ分布（P15突合） ----
    p("\n# A3 音源パラメータ実測（P15公称との突合）")
    by_cls = defaultdict(lambda: defaultdict(list))
    subtype = Counter()
    for s in scenes.values():
        for x in s["sources"]:
            c = x["class"]
            by_cls[c]["law_db"].append(x.get("law_db"))
            by_cls[c]["l1m_db"].append(x.get("l1m_db"))
            if "speed_mps" in x:
                by_cls[c]["speed"].append(x["speed_mps"])
            elif c == "car_drive":
                by_cls[c]["speed"].append(abs(x.get("speed_mps", np.nan))
                                          if "speed_mps" in x else np.nan)
            pr = x.get("params", {})
            if c == "siren":
                subtype["siren:" + pr.get("siren_type", "?")] += 1
            if c == "bike_bell":
                subtype["bell:" + ("ring" if pr.get("bell_type") == "ring"
                                   else "single")] += 1
            if c == "backup_beep":
                lv = x.get("law_db")
                subtype["backup:" + ("実勢85-95" if lv is not None and lv >= 80
                                     else "R165_60-75")] += 1
            if c == "car_drive":
                by_cls[c]["f0"].append(pr.get("f0"))
    for c in sorted(by_cls):
        d = by_cls[c]
        law = [v for v in d["law_db"] if v is not None]
        p(f"\n## {c}  (n={len(d['l1m_db'])})")
        if law:
            p(f"- 規定音量 law_db: {rng_str(law)}")
        if d.get("speed"):
            sp = [v for v in d["speed"] if v is not None and np.isfinite(v)]
            if sp:
                p(f"- 速度: {rng_str(np.abs(sp))} m/s")
        if d.get("f0"):
            p(f"- 車f0: {rng_str([v for v in d['f0'] if v], '{:.2f}')} Hz")
    p("\n## サブタイプ取り分")
    for k, v in sorted(subtype.items()):
        p(f"- {k}: {v}")

    # ---- C1 評価専用の員数 ----
    p("\n# C1 評価専用セットの員数")
    if DSE.exists():
        ev = load_scenes(DSE)
        scn = Counter(s["row"].get("scenario", "?") for s in ev.values())
        p(f"評価クリップ総数: {len(ev)}（公称3,342）")
        for k, v in sorted(scn.items()):
            p(f"- {k}: {v}本")
    else:
        p(f"⚠️ {DSE} が見つからない")

    (ROOT / "out" / "audit_v11_params_2026-08-05.md").write_text(
        "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
