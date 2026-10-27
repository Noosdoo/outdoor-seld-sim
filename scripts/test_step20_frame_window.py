# -*- coding: utf-8 -*-
"""test_step20_frame_window.py — step20 v3 の検出フレーム率の窓（変更票 #69 r31 ①・2026-10-23 Fable）の検査。
  W1 t_in10／t_out10 がある本: 分母 = [t_in10, t_out10]（±10 に入っている時刻）・fr_window = in10
  W2 無い本: 旧の窓 [t_start, t_cpa]・fr_window = start_cpa
  W3 extras に本数（n_fr_in10・n_fr_start_cpa）が残る
使い方: python scripts/test_step20_frame_window.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s20v3_fw", ROOT / "scripts/step20_realsmoke_score_v3.py")
S = importlib.util.module_from_spec(spec); sys.modules["s20v3_fw"] = S; spec.loader.exec_module(S)

COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "区分", "状態", "横距離m", "n_car", "scored", "t_in10", "t_out10"]


def main() -> int:
    ok = []
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    ci = S.CLS_IDX["car_drive"]
    pred = {"eval_field_clip00201": {}, "eval_field_clip00202": {}}
    for k in range(50, 80):                                              # 車の推定 5.0〜8.0 s（30 フレーム）
        pred["eval_field_clip00201"].setdefault(k, []).append((ci, 90.0, 0.0, 2.0))
        pred["eval_field_clip00202"].setdefault(k, []).append((ci, 90.0, 0.0, 2.0))
    rows = [dict(zip(COLS, ["eval_field_clip00201", "1", "W01", "P1", "", "car_drive", "L", "3", "8", "", "歩行", "静止", "2.0", "1", "1", "5.0", "11.0"])),
            dict(zip(COLS, ["eval_field_clip00202", "1", "W02", "P1", "", "car_drive", "L", "3", "8", "", "歩行", "歩行", "2.0", "1", "1", "", ""]))]
    events, neg, extras = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 7.5, True)
    e1 = [e for e in events if e["clip"] == "eval_field_clip00201"][0]; e2 = [e for e in events if e["clip"] == "eval_field_clip00202"][0]
    ok.append((f"W1 ±10 の窓 [5.0, 11.0]（60 フレーム）で検出 30 → 0.5（{e1['frame_recall']}）・fr_window in10", e1["frame_recall"] == 0.5 and e1["fr_window"] == "in10"))
    ok.append((f"W2 時刻が無い本は旧の窓 [3, 8]（50 フレーム）で 30 → 0.6（{e2['frame_recall']}）・fr_window start_cpa", e2["frame_recall"] == 0.6 and e2["fr_window"] == "start_cpa"))
    ok.append(("W3 extras に本数（in10 1・start_cpa 1）", extras["n_fr_in10"] == 1 and extras["n_fr_start_cpa"] == 1))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
