# -*- coding: utf-8 -*-
"""長尺セット v1: 通知（強・中）を「帰属した車の最接近距離」で分類し（**2026-10-20 scripts/ に戻した版 _long_breakdown_v2**: 変更票 #72 の条件 (2b)〔幹線歩道の安全な車への強 +2 pt 以内〕を表あり・なしで比べる。中身は同じ・_long_score_v2 を読む）、同じ車への再発火を数える（v4.5 結果 §2 の表を任意の予測で再現・2026-09-07）。

  段 | 合計（/分） | 至近の車（≤1.5 m） | 注意の車（1.5〜3.2 m） | 安全な車（>3.2 m） | 帰属なし | 同じ車への再発火
帰属 = _long_score.attribute（±0.5 s・≤30°・最も近い方位）。車の区分 = その車のクリップ内の最接近（水平距離）。
再発火 = 同じ車に、同段以上の通知が既に届いた後の通知。

使い方: python scripts/_long_breakdown.py --pred <val_all_causal.csv> [--split fold40] [--scene arterial_walk] [--out md]
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _sc(name):
    """2026-09-12 の整理で scripts_archive/ へ移した分の道しるべ。

    探す順: ①フォルダまで書いてあればその通り ②自分の隣 ③scripts/。
    ②と③の両方に同じ名前があって、どちらか指定が無いときは**黙って選ばず例外**にする
    （独立監査 R9・2026-09-13。以前は隣を優先していたので、scripts/ を明示しても
    退避版が返っていた）。
    """
    _t = str(name).replace("\\", "/")
    _root = ROOT
    if "/" in _t and (_root / _t).exists():
        return _root / _t
    _base = _t.split("/")[-1]
    _here = Path(__file__).resolve().parent / _base
    _there = _root / "scripts" / _base
    if _here.resolve() == _there.resolve():
        return _there                                   # scripts/ に戻した版（2026-10-20）
    if _here.exists() and _there.exists():
        raise FileNotFoundError(
            "%s が scripts/ と %s/ の両方にあります。どちらか指定してください"
            "（例: _sc(\'scripts/%s\')）" % (_base, _here.parent.name, _base))
    return _here if _here.exists() else _there

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("_ls", _sc("_long_score_v2.py"))
LS = importlib.util.module_from_spec(spec); sys.modules["_ls"] = LS; spec.loader.exec_module(LS)

RANK = {"中": 1, "強": 2}


def tier_of_car(fr):
    d = min(v[1] for v in fr.values())
    return "至近" if d <= 1.5 else ("注意" if d <= 3.2 else "安全")


def main() -> int:
    a = sys.argv
    if "--dist-table" in a:
        LS.DIST_TABLE = LS.load_dist_table(a[a.index("--dist-table") + 1])
        if "--dist-table-max" in a:
            LS.DIST_TABLE_MAX = float(a[a.index("--dist-table-max") + 1])
    pred = LS.load_pred_long(a[a.index("--pred") + 1])
    split = a[a.index("--split") + 1] if "--split" in a else None
    scene = a[a.index("--scene") + 1] if "--scene" in a else None
    plan = {r["clip_id"]: r for r in csv.DictReader(open(LS.DS / "plan/assignment_long_v1.csv", encoding="utf-8"))}
    clips = [c for c, r in plan.items() if (split is None or r.get("split") == split) and (scene is None or r.get("scene", r.get("場面", "")) == scene)]
    cnt = {t: defaultdict(int) for t in ("強", "中")}
    total_min = 0.0
    for clip in clips:
        gt = LS.load_gt(clip)
        fd, fw = pred.get(clip, (dict(), dict()))
        eps = LS.episodes(fd, fw) if fd or fw else []
        total_min += LS.NFR / 10.0 / 60.0
        best = {}                                    # car -> 届いた最高段
        for j, az, tier, cls in eps:
            if tier not in RANK:
                continue
            key = LS.attribute((j, az), gt)
            cnt[tier]["合計"] += 1
            if key is None:
                cnt[tier]["帰属なし"] += 1
                continue
            cnt[tier][tier_of_car(gt[key])] += 1
            if best.get(key, 0) >= RANK[tier]:
                cnt[tier]["再発火"] += 1
            best[key] = max(best.get(key, 0), RANK[tier])
    L = [f"# 長尺 v1 の通知の内訳（{Path(a[a.index('--pred') + 1]).name}・{split or 'all'}・{scene or '全場面'}・{len(clips)} 本）", "",
         "| 段 | 合計 | /分 | 至近の車（≤1.5 m） | 注意の車（1.5〜3.2 m） | 安全な車（>3.2 m） | 帰属なし | 同じ車への再発火 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in ("強", "中"):
        n = cnt[t]["合計"] or 1
        pct = lambda k: f"{100 * cnt[t][k] / n:.0f}%"
        L.append(f"| {t} | {cnt[t]['合計']} | {cnt[t]['合計'] / max(total_min, 1e-9):.2f} | {pct('至近')} | {pct('注意')} | **{pct('安全')}** | {pct('帰属なし')} | {pct('再発火')} |")
    txt = "\n".join(L)
    print(txt)
    if "--out" in a:
        Path(a[a.index("--out") + 1]).write_text(txt + "\n", encoding="utf-8")
    if "--json" in a:                                                            # R72-08: 丸め前の分子・分母（安全な車への強 = 安全 / 合計）
        import json as _json
        J = {t: {k: int(v) for k, v in cnt[t].items()} for t in ("強", "中")}
        for t in J:
            J[t]["share_safe"] = (J[t].get("安全", 0) / J[t]["合計"]) if J[t].get("合計") else None
        Path(a[a.index("--json") + 1]).write_text(_json.dumps({"split": split, "scene": scene, "n_clips": len(clips), "minutes": total_min, "pred": a[a.index("--pred") + 1], "dist_table": a[a.index("--dist-table") + 1] if "--dist-table" in a else "", "tiers": J}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
