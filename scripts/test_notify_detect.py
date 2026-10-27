# -*- coding: utf-8 -*-
"""「検出したら知らせる」規則（_notify_detect.py・変更票 #73 Q2）の検査。v4.3 と同じ追跡・確認・エピソード化を使い、距離を見ない。
使い方: python scripts/test_notify_detect.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


U = _load("unified_t", "_score_unified.py")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)[:200]) if detail and not cond else ''}")


def frames_of(d_list, cls=4, az=30.0):
    """frame -> [(cls, az, el, d)]。d_list: フレームごとの距離（None = 検出なし）。"""
    return {k: [(cls, az, 0.0, d)] for k, d in enumerate(d_list) if d is not None}


def run(frames, rule):
    U.NOTIFY_RULE = rule
    return [f for f in U.fires_of(frames) if f[2] != "警告"]


def main() -> int:
    # 1 遠い車（8 m で一定・100 フレーム）: v4.3 は鳴らさない（最接近も距離の閾値も満たさない）・detect は 強 で 1 回（エピソード先頭）
    far = frames_of([8.0] * 100)
    f43, fdt = run(far, "v43"), run(far, "detect")
    check("遠い車（8 m 一定）: v4.3 = 0 回・detect = 強 1 回", len(f43) == 0 and len(fdt) == 1 and fdt[0][2] == "強", (f43, fdt))
    check("detect の発火は確認（CONFIRM=2）の 2 フレーム目", fdt and fdt[0][0] == U.v4.CONFIRM - 1, fdt)
    # 2 近づく車（6 m → 0.5 m・一定速度）: 両方とも鳴る・detect のほうが早いか同じ
    appr = frames_of([max(0.5, 6.0 - 0.11 * k) for k in range(100)])
    a43, adt = run(appr, "v43"), run(appr, "detect")
    check("近づく車: v4.3 も detect も鳴る・detect は v4.3 より早いか同じ", len(a43) >= 1 and len(adt) >= 1 and adt[0][0] <= a43[0][0], (a43, adt))
    # 3 1 フレームだけの検出（確認に足りない）: detect も鳴らない
    blip = frames_of([None] * 50 + [3.0] + [None] * 49)
    check("1 フレームだけの検出は鳴らない（確認は v4.3 と同じ）", len(run(blip, "detect")) == 0)
    # 4 途切れた 2 回の検出（別エピソード）: detect は 2 回（繰り返しの抑えは group_episodes と同じ）
    two = frames_of([5.0] * 20 + [None] * 30 + [5.0] * 20 + [None] * 30)
    check("途切れた 2 回の検出 → detect は 2 回（別エピソード）", len(run(two, "detect")) == 2)
    # 5 距離が nan の検出は距離の規則には入らない（採点器の frames_dist が落とす）= 既存の動き（Astra R73-02 の記録）
    nan_fr = {k: [(4, 30.0, 0.0, float("nan"))] for k in range(100)}
    check("距離 nan の検出は detect でも鳴らない（距離なしの相手は知覚の表だけ・#73 3）", len(run(nan_fr, "detect")) == 0)
    # 6 規則の名前と CLI の検査
    check("規則の一覧に detect・既定は v43", U.NOTIFY_RULES == ("v43", "detect"))
    U.NOTIFY_RULE = "v43"
    k = sum(c for _, c in results); print(f"{k}/{len(results)} {'ALL PASS' if k == len(results) else 'FAILED'}")
    return 0 if k == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
