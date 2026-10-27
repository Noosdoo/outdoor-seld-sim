# -*- coding: utf-8 -*-
"""Q2への回答表 — 「すり抜け」と「対向」を区別できているか（⑦ Stage 4、2026-08-30）。

中間発表の質疑Q2:
  「狭いところをすり抜けていく車と、対向してくる車が、同じ至近警告になって
   区別できないのでは」 → 「三角関数を使って区別する予定」と約束した。

この表はその約束への直接の回答になる。**選定には使わない（報告専用）**。

型の定義（GTの幾何から。三角関数の原理そのもの）:
  **判定時窓 = CPAの2.5〜1.5秒前**（通知を出すべき時点）のGT方位変化率の中央値で分ける。
  対向型     = |dθ/dt| < 0.10 rad/s（判定時点で方位がほぼ動かない＝衝突コース状）
  すり抜け型 = 同 ≥ 0.10 rad/s（判定時点ですでに方位が動いている＝横を通る）
⚠️ CPA「直前」の窓は不可: 通過車は最接近の瞬間に方位が最速で回るため、
直前窓だと全イベントがすり抜け型になる（2026-08-30の初版で確認して修正）。
0.10 rad/s は記述用の区切り（表には連続値の中央値も併記する）。

使い方:
  python scripts/_notify_v42_q2_table.py <pred.csv> <meta_dir> <出力dir> [--set k=v ...]
--set で v4.2 の勝ち構成を渡す（省略時は v4.1 相当 = 全部OFF。その場合は新旧が同じ）。
"""
from __future__ import annotations

import importlib.util
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


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V42 = _load("nv42", "step12_notify_v42_bearing.py")
DG = _load("nv42diag", "_notify_v42_diag.py")

FPS = v4.FPS
WIN_PRE = WIN_POST = 1.0
ADOT_SPLIT = 0.10          # [rad/s] 型の区切り（記述用）
CLS_JP = {4: "車", 6: "キックボード", 7: "バイク"}


def gt_adot_before_cpa(fr, cpa):
    """判定時窓（CPAの2.5〜1.5秒前）のGT方位変化率の中央値[rad/s]。不足なら None。"""
    js = [j for j in sorted(fr) if cpa - 25 <= j <= cpa - 15]
    if len(js) < 4:
        return None
    unw = np.unwrap(np.radians([fr[j][0] for j in js]))
    d = np.abs(np.diff(unw)) * FPS / np.diff(js)
    return float(np.median(d))


def outcome(fires, ev):
    """イベントに付いた最良の通知（強 > 中 > 無）。貪欲マッチは採点器と同じ窓。"""
    a, b = ev["f0"] - WIN_PRE * FPS, ev["cpa"] + WIN_POST * FPS
    tiers = [f[2] for f in fires if f[4] == ev["cls"] and a <= f[0] <= b]
    if "強" in tiers:
        return "強"
    if "中" in tiers:
        return "中"
    return "無"


def main() -> int:
    pred_path, meta_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    C_new = V42.cfg_from_args(sys.argv)
    pred = v4.load_pred(pred_path)

    res_old = v4.run_rule(pred, "cpa")
    res_new = V42.run_rule2(pred, C_new)

    rows = defaultdict(lambda: defaultdict(int))    # (tier帯,型) -> (規則,結果) -> n
    adot_stats = defaultdict(list)
    n_skip = 0
    for clip in sorted(pred):
        evs = [DG.mk_event(c, t, fr) for c, t, fr in DG.gt_tracks(meta_dir, clip)]
        f_old = [(f[0], f[1], f[2], f[3], cls)
                 for cls, eps in res_old.get(clip, {}).items() for f in eps]
        f_new = [(f[0], f[1], f[2], f[3], cls)
                 for cls, eps in res_new.get(clip, {}).items() for f in eps]
        for ev in evs:
            ad = gt_adot_before_cpa(ev["fr"], ev["cpa"])
            if ad is None:
                n_skip += 1
                continue
            typ = "対向型" if ad < ADOT_SPLIT else "すり抜け型"
            adot_stats[(ev["tier"], typ)].append(ad)
            key = (ev["tier"], typ)
            rows[key][("旧", outcome(f_old, ev))] += 1
            rows[key][("新", outcome(f_new, ev))] += 1

    lab = V42.__dict__  # noqa: F841 (cfg表示用に下で直接使う)
    R = [f"# Q2への回答表 — すり抜け／対向の型別通知 pred={pred_path.name}", "",
         f"- 旧= v4.1 / 新= v4.2 `{C_new}`",
         f"- 型: **判定時窓（CPAの2.5〜1.5秒前）**のGT|dθ/dt|中央値 {ADOT_SPLIT} rad/s "
         f"で区切り（窓不足の除外 {n_skip}件）",
         "- 注: GT方位は整数度のため変化率は0.1745 rad/s刻みに量子化される。"
         "実質「動かない(0) vs 動く(≥0.17)」の2値に近い", "",
         "| GT区分 | 型 | n | GT\\|dθ/dt\\|中央 | 旧:強/中/無 | 新:強/中/無 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for tier, tjp in (("critical", "重大(≤1.5m)"), ("caution", "注意(≤3.2m)"),
                      ("safe", "安全(>3.2m)")):
        for typ in ("対向型", "すり抜け型"):
            key = (tier, typ)
            if key not in rows:
                continue
            r = rows[key]
            n = sum(v for (rule, _), v in r.items() if rule == "旧")
            ad = np.median(adot_stats[key])
            o = "/".join(str(r.get(("旧", x), 0)) for x in ("強", "中", "無"))
            nw = "/".join(str(r.get(("新", x), 0)) for x in ("強", "中", "無"))
            R.append(f"| {tjp} | {typ} | {n:,} | {ad:.3f} rad/s | {o} | {nw} |")
    R += ["", "読み方: **重大×対向型の「強」が増え、安全×すり抜け型の「無」が保たれて",
          "いれば、約束どおり三角関数（方位変化率）で両者を区別できている。**"]
    out_md = outdir / "q2_table.md"
    out_md.write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
