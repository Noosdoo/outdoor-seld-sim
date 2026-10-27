# -*- coding: utf-8 -*-
"""ベルの打音と鳴動窓の検査（2026-10-08・Astra 監査 D08-01 の回帰検査）。
  B1 Astra の再現例 train_syn_clip40002（ベルつき走行・ピン型・n_ring=1・窓 0.768〜1.268 s・周期 2.04 s）: 打音合わせ後は窓の頭に打音がある
  B2 同じ本で打音合わせを外す（align_strike を消す）と窓の頭に打音が無い（= 監査が見つけた状態の再現）
  B3 ピン型の双子（v19bellpin）の最初の 20 本: 全部 窓の頭に打音がある
  B4 土台（v11core）のベルの本: ALIGN_BASE_BELLS=False では打音が無い本がある／True にすると全部ある（変更票 D24 用の旗の動作）
  B5 ベルつき走行の 3 型（ding／ring／pin）で、窓の頭の打音がある
使い方: PYTHONPATH=scripts:src python scripts/test_v2final_bellstrike.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import step11_v22_render as v22  # noqa: E402

PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
PLAN_PRE = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final_pre_v22.csv"


def rows_of(path: Path, scen: str) -> list:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == scen:
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); out.append(r)
    return out


def bells(s: dict) -> list:
    return [src for src in s["sources"] if src["class"] == "bike_bell"]


def main() -> int:
    res = []
    bb = {r["clip_id"]: r for r in rows_of(PLAN, "v19bellbike")}
    r = bb["train_syn_clip40002"]; s = v22.sample_scene_v19(r); b = bells(s)[0]
    res.append(("B1 clip40002: 打音合わせ後は窓の頭に打音", b["params"].get("bell_type") == "pin" and v22.strike_at_window_start(b)))
    b2 = dict(b); b2["params"] = {k: v for k, v in b["params"].items() if k != "align_strike"}
    # align_strike を外しても pin は常に合わせる実装なので、監査の状態は「t_on を 0 にしない＝合わせない」で再現する: 合わせる前のドライを直接見る
    full = v22._bell_dry(b2, b2["params"], v22.m9._make_dry)      # 合わせる前（打音は 0, period, …）
    import numpy as np
    a0 = int(b["t_on"] * v22.m9.FS_SIM); w = int(0.06 * v22.m9.FS_SIM)
    start = float(np.sqrt(np.mean(full[a0:a0 + w] ** 2))); peak = float(np.sqrt(np.convolve(full ** 2, np.ones(w) / w, mode="valid")).max())
    res.append(("B2 clip40002: 合わせる前は窓の頭が尻尾だけ（最大の 0.3 倍未満）", start < 0.3 * peak))
    pins = rows_of(PLAN, "v19bellpin")[:20]
    res.append(("B3 双子 20 本: 全部 窓の頭に打音", all(all(v22.strike_at_window_start(x) for x in bells(v22.sample_scene_v19(q))) for q in pins)))
    base = [q for q in rows_of(PLAN_PRE, "v11core") if q["w1_class"] == "bike_bell" or q["w2_class"] == "bike_bell"][:40]
    v22.ALIGN_BASE_BELLS = False
    n_bad_off = sum(1 for q in base for x in bells(v22.sample_scene_v19(q)) if not v22.strike_at_window_start(x))
    v22.ALIGN_BASE_BELLS = True
    n_bad_on = sum(1 for q in base for x in bells(v22.sample_scene_v19(q)) if not v22.strike_at_window_start(x))
    v22.ALIGN_BASE_BELLS = False
    res.append((f"B4 土台 40 本: 旗なしで打音の無い音源 {n_bad_off} 個（>0）→ 旗ありで {n_bad_on} 個（=0）", n_bad_off > 0 and n_bad_on == 0))
    seen = {}
    for q in rows_of(PLAN, "v19bellbike")[:40]:
        s = v22.sample_scene_v19(q); x = bells(s)[0]; k = x["params"].get("bell_type", "ding")
        seen.setdefault(k, v22.strike_at_window_start(x))
    res.append((f"B5 ベルつき走行の 3 型で窓の頭に打音 {sorted(seen)}", len(seen) == 3 and all(seen.values())))
    ok = 0
    for name, passed in res:
        print(("PASS " if passed else "FAIL ") + name); ok += int(passed)
    print(f"{ok}/{len(res)} " + ("ALL PASS" if ok == len(res) else "NG"))
    return 0 if ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
