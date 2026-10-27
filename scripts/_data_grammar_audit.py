# -*- coding: utf-8 -*-
"""学習データの場面（work/<clip>/scene.json）を総なめして、クラスごとの「動き・速さ・高さ・距離・同時数・組み合わせ」を集計し、
現実にあり得ない／疑わしい場面を数える（2026-09-08 本人依頼「踏切が動いている、などが含まれていないか」）。

使い方: python scripts/_data_grammar_audit.py --ds out/dataset_outdoor_siren_v16 [--limit N] [--out md]
  （サーバでは PS の venv で実行。scene.json だけ読む。音声は読まない）
出力: クラス別の表（本数・静止/移動の内訳・速さの分位・高さ・最接近距離）、シナリオ別の本数、同時発生の組み合わせ、
      疑わしい場面の一覧（規則は本文 §RULES）。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---- RULES: 「現実にあり得ない／疑わしい」の判定（クラスごとの許容範囲。外れたら数える） ----
# 速さ [m/s]: (下限, 上限)。静止クラスは上限 0.1。高さ [m]。
RULES = {
    "car_drive":   dict(speed=(0.5, 33.0), z=(0.3, 2.0), static_ok=False),
    "kickboard":   dict(speed=(0.5, 8.5), z=(0.5, 2.0), static_ok=False),
    "motorcycle":  dict(speed=(2.0, 25.0), z=(0.3, 2.0), static_ok=False),
    "siren":       dict(speed=(0.0, 33.0), z=(0.5, 3.5), static_ok=True),     # 停車中の緊急車両もありうる
    "horn":        dict(speed=(0.0, 33.0), z=(0.3, 2.0), static_ok=True),
    "backup_beep": dict(speed=(0.0, 3.0), z=(0.3, 2.0), static_ok=True),      # 後退は徐行
    "bike_bell":   dict(speed=(0.0, 9.0), z=(0.5, 1.5), static_ok=True),
    "crossing":    dict(speed=(0.0, 0.1), z=(0.5, 5.0), static_ok=True),      # 警報機は動かない（列車の通過音を crossing と呼ぶなら別扱い）
}
DIST_CLS = {"car_drive", "kickboard", "motorcycle"}


def speed_of(src):
    v = src.get("speed_mps")
    if v is not None:
        return float(v)
    wp = np.array(src.get("wp", []), dtype=float)
    if wp.ndim == 2 and len(wp) >= 2 and wp.shape[1] >= 4:
        dt = wp[-1][0] - wp[0][0]
        return float(np.linalg.norm(wp[-1][1:4] - wp[0][1:4]) / dt) if dt > 0 else 0.0
    return 0.0


def z_of(src):
    wp = np.array(src.get("wp", []), dtype=float)
    if wp.ndim == 2 and wp.shape[1] >= 4:
        return float(np.median(wp[:, 3]))
    if wp.ndim == 1 and len(wp) >= 3:
        return float(wp[2])
    return float("nan")


def main() -> int:
    a = sys.argv
    ds = Path(a[a.index("--ds") + 1])
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 0
    work = ds / "work"
    dirs = sorted(p for p in work.iterdir() if (p / "scene.json").exists())
    if limit:
        dirs = dirs[:limit]
    per_cls = defaultdict(lambda: dict(n=0, static=0, moving=0, speeds=[], zs=[], mind=[], nolabel=0, train=0, grammar=Counter()))
    scen = Counter(); mic = Counter(); n_src = Counter(); pairs = Counter(); rain = Counter()
    flags = Counter(); examples = defaultdict(list)
    n = 0
    for d in dirs:
        try:
            s = json.loads((d / "scene.json").read_text(encoding="utf-8"))
        except Exception as e:
            flags["scene.json 読めない"] += 1; continue
        n += 1
        row = s.get("row", {})
        scen[row.get("scenario", "?")] += 1
        mic[s.get("mic", {}).get("motion", "?")] += 1
        rain[(s.get("rain") or {}).get("kind", "なし") if s.get("rain") else "なし"] += 1
        srcs = s.get("sources", [])
        n_src[len([x for x in srcs if not x.get("no_label")])] += 1
        classes = sorted({x["class"] for x in srcs if not x.get("no_label")})
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                pairs[(classes[i], classes[j])] += 1
        by_cls = Counter(x["class"] for x in srcs if not x.get("no_label"))
        for c, k in by_cls.items():
            if k >= 2 and c in ("crossing", "siren"):
                flags[f"同じクリップに {c} が {k} 系列"] += 1; examples[f"同じクリップに {c} が複数"].append(d.name)
        for src in srcs:
            c = src.get("class", "?"); P = per_cls[c]
            P["n"] += 1
            if src.get("no_label"):
                P["nolabel"] += 1
            if src.get("params", {}).get("train_car"):
                P["train"] += 1
            P["grammar"][src.get("grammar", "-")] += 1
            v = speed_of(src); z = z_of(src)
            P["speeds"].append(v); P["zs"].append(z)
            if src.get("min_dist_m") is not None:
                P["mind"].append(float(src["min_dist_m"]))
            (P.__setitem__("static", P["static"] + 1) if v < 0.1 else P.__setitem__("moving", P["moving"] + 1))
            r = RULES.get(c)
            if r is None:
                flags[f"未知のクラス {c}"] += 1; continue
            if v >= 0.1 and not (r["speed"][0] <= v <= r["speed"][1]):
                key = f"{c}: 速さが範囲外 {r['speed']} m/s"
                flags[key] += 1; examples[key].append(f"{d.name} v={v:.1f}")
            if v < 0.1 and not r["static_ok"]:
                key = f"{c}: 静止している（動くはずのクラス）"; flags[key] += 1; examples[key].append(d.name)
            if c == "crossing" and v >= 0.1:
                key = "crossing: 動いている（列車の通過音を crossing と呼んでいる）"; flags[key] += 1; examples[key].append(f"{d.name} v={v:.1f}")
            if math.isfinite(z) and not (r["z"][0] <= z <= r["z"][1]):
                key = f"{c}: 高さが範囲外 {r['z']} m"; flags[key] += 1; examples[key].append(f"{d.name} z={z:.2f}")
            md = src.get("min_dist_m")
            if md is not None and float(md) < 0.3:
                key = f"{c}: 最接近 0.3 m 未満（マイクを突き抜ける）"; flags[key] += 1; examples[key].append(f"{d.name} d={float(md):.2f}")
            if src.get("t_off", 10) <= src.get("t_on", 0):
                flags[f"{c}: t_off ≤ t_on"] += 1
            # 折れ線軌道（3 点以上）は宣言値でなく実際の区間で検査する（監査 F1: 瞬時の 180 度反転を見逃していた）
            wp = np.array(src.get("wp", []), dtype=float)
            if wp.ndim == 2 and len(wp) >= 3 and wp.shape[1] >= 4:
                dt = np.diff(wp[:, 0])
                if np.any(dt <= 0):
                    key = f"{c}: 軌道の時刻が増加していない"; flags[key] += 1; examples[key].append(d.name)
                else:
                    seg = np.diff(wp[:, 1:3], axis=0)
                    spd = np.linalg.norm(seg, axis=1) / dt
                    if r is not None and np.any(spd > r["speed"][1] + 0.5):
                        key = f"{c}: 区間速度が範囲外 {r['speed']} m/s"; flags[key] += 1; examples[key].append(f"{d.name} vmax={spd.max():.1f}")
                    for i in range(1, len(seg)):
                        n0, n1 = np.linalg.norm(seg[i - 1]), np.linalg.norm(seg[i])
                        if n0 < 1e-9 or n1 < 1e-9:
                            continue
                        ang = float(np.degrees(np.arccos(np.clip(np.dot(seg[i - 1], seg[i]) / (n0 * n1), -1.0, 1.0))))
                        if ang > 120.0:
                            key = f"{c}: 軌道が瞬時に反転（区間の向きが 120 度超変化）"; flags[key] += 1
                            examples[key].append(f"{d.name} t={wp[i, 0]:.2f} {ang:.0f}度"); break
    L = [f"# 学習データの場面の点検 — {ds.name}（{n:,} 本）", "",
         "## クラス別", "", "| クラス | 音源数 | 静止 | 移動 | 速さ 5/50/95% [m/s] | 高さ 中央 [m] | 最接近 5/50% [m] | ラベル無し（音だけ） | 列車 | 文法 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, P in sorted(per_cls.items(), key=lambda kv: -kv[1]["n"]):
        sp = np.array(P["speeds"]); zs = np.array([z for z in P["zs"] if math.isfinite(z)]); md = np.array(P["mind"])
        q = lambda x, p: (f"{np.percentile(x, p):.1f}" if len(x) else "-")
        L.append(f"| {c} | {P['n']:,} | {P['static']:,} | {P['moving']:,} | {q(sp, 5)} / {q(sp, 50)} / {q(sp, 95)} | {q(zs, 50)} | {q(md, 5)} / {q(md, 50)} | {P['nolabel']:,} | {P['train']:,} | {dict(P['grammar'])} |")
    L += ["", "## シナリオ・マイク・雨・同時数", "", f"- シナリオ: {dict(scen)}", f"- マイク: {dict(mic)}", f"- 雨: {dict(rain)}", f"- ラベル付き音源の数/本: {dict(sorted(n_src.items()))}", "",
          "## 同時に出るクラスの組み合わせ（本数・上位 15）", ""]
    for (x, y), k in pairs.most_common(15):
        L.append(f"- {x} × {y}: {k:,}")
    L += ["", "## 疑わしい場面（規則 = スクリプト RULES）", ""]
    if not flags:
        L.append("- なし")
    for k, v in flags.most_common():
        ex = examples.get(k, [])[:3]
        L.append(f"- **{k}**: {v:,} 件" + (f"（例: {', '.join(ex)}）" if ex else ""))
    txt = "\n".join(L)
    print(txt)
    if "--out" in a:
        Path(a[a.index("--out") + 1]).write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
