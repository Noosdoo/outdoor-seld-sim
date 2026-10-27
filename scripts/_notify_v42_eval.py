# -*- coding: utf-8 -*-
"""v4.2 の回帰確認と部品別の煙試験（2026-08-30）。

3つのことをする:
  1. 回帰: v4.2 全部品OFF が v4.1 (fires_cpa) と**同一の発火**になることを確認
  2. 煙試験: 部品を1つずつONにして、指標が壊れず妥当な方向へ動くことを見る
  3. 参考: 全部ONの合わせ技

⚠️ これは**設計の確認**であって、しきい値・部品の選定ではない。
選定はチューニング専用の新val（新しい乱数）で、偶奇分割ホールドアウトで行う。

使い方:
  python scripts/_notify_v42_eval.py [pred_val_all.csv] [metadata_distディレクトリ] [出力dir]
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
    sys.modules[name] = m       # dataclass が sys.modules[__module__] を参照するため必須
    spec.loader.exec_module(m)
    return m


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V42 = _load("nv42", "step12_notify_v42_bearing.py")
DG = _load("nv42diag", "_notify_v42_diag.py")     # gt_tracks / mk_event を借りる

FPS = v4.FPS
WIN_PRE, WIN_POST = 1.0, 1.0


def score(pred, meta_dir, res):
    """診断と同じ貪欲マッチで {到達, 強到達, 注意, 抑制, lead...} を返す。"""
    stat = defaultdict(lambda: [0, 0])
    n_strong_hit, leads = 0, []
    n_fire = 0
    for clip in sorted(pred):
        events = [DG.mk_event(c, t, fr) for c, t, fr in DG.gt_tracks(meta_dir, clip)]
        fires = [(f[0], f[1], f[2], f[3], cls)
                 for cls, eps in res.get(clip, {}).items() for f in eps]
        n_fire += len(fires)
        used = [False] * len(fires)
        used_s = [False] * len(fires)
        for ev in events:
            a, b = ev["f0"] - WIN_PRE * FPS, ev["cpa"] + WIN_POST * FPS
            hit = None
            for i, fr_ in enumerate(fires):
                if not used[i] and fr_[4] == ev["cls"] and a <= fr_[0] <= b:
                    hit = i
                    break
            if ev["tier"] == "safe":
                stat["safe"][1] += 1
                stat["safe"][0] += int(hit is None)
                if hit is not None:
                    used[hit] = True
                continue
            stat[ev["tier"]][1] += 1
            if hit is not None:
                used[hit] = True
                stat[ev["tier"]][0] += 1
                leads.append((ev["cpa"] - fires[hit][0]) / FPS)
            if ev["tier"] == "critical":
                for i, fr_ in enumerate(fires):
                    if (not used_s[i] and fr_[4] == ev["cls"] and fr_[2] == "強"
                            and a <= fr_[0] <= b):
                        used_s[i] = True
                        n_strong_hit += 1
                        break
    L = np.array(leads) if leads else np.array([0.0])
    g = lambda t: 100 * stat[t][0] / max(stat[t][1], 1)
    return dict(crit=g("critical"), strong=100 * n_strong_hit / max(stat["critical"][1], 1),
                caut=g("caution"), safe=g("safe"),
                lead=float(np.median(L)), lead25=float(100 * np.mean(L >= 2.5)),
                n_fire=n_fire, n_crit=stat["critical"][1])


def main() -> int:
    pred_path = (Path(sys.argv[1]) if len(sys.argv) > 1
                 else ROOT / "out/predictions_v12_w3/val_all.csv")
    meta_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
                else ROOT / "out/dataset_outdoor_siren_v12/metadata_dist")
    outdir = (Path(sys.argv[3]) if len(sys.argv) > 3
              else ROOT / "out/notify_v42_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    pred = v4.load_pred(pred_path)

    Cfg = V42.Cfg
    variants = [
        ("v4.1 (現行)", None),
        ("v4.2 全OFF (回帰)", Cfg()),
        ("+ts傾き", Cfg(robust_slope=True)),
        ("+中央値スケール", Cfg(robust_scale=True)),
        ("+brg9", Cfg(brg_win=9)),
        ("+mn4/6", Cfg(confirm_m=4, confirm_n=6)),
        ("+routeC", Cfg(route_c=True)),
        ("+link", Cfg(link_pred=True)),
        # 中央値スケールは煙試験で有害と判明したため合わせ技から外す
        ("合わせ技(ts+brg9+mn+routeC+link)",
         Cfg(robust_slope=True, brg_win=9, confirm_m=4, confirm_n=6,
             route_c=True, link_pred=True)),
    ]
    R = [f"# v4.2 回帰確認と部品別煙試験 pred={pred_path.name}", "",
         "⚠️ 選定ではない（選定は新valで行う）。方向の確認だけ。", "",
         "| 構成 | 至近到達 | **強到達** | 注意到達 | 安全抑制 | リード中央 | ≥2.5s | 発火数 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    res41 = res42_default = None
    for name, C in variants:
        res = v4.run_rule(pred, "cpa") if C is None else V42.run_rule2(pred, C)
        if name.startswith("v4.1"):
            res41 = res
        if name.startswith("v4.2 全OFF"):
            res42_default = res
        s = score(pred, meta_dir, res)
        R.append(f"| {name} | {s['crit']:.1f}% | **{s['strong']:.1f}%** "
                 f"| {s['caut']:.1f}% | {s['safe']:.1f}% "
                 f"| {s['lead']:.2f}s | {s['lead25']:.1f}% | {s['n_fire']:,} |")
        print(R[-1], flush=True)

    same = res41 == res42_default
    R += ["", f"## 回帰確認: v4.2全OFF と v4.1 の発火は"
          f"{'**完全一致** ✅' if same else '**不一致** ❌（要調査）'}"]
    out_md = outdir / "v42_smoke.md"
    out_md.write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R[-2:]))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
