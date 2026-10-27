# -*- coding: utf-8 -*-
"""test_step20_b2_engine.py — 発進（B2）の「エンジンをかける音で出た通知」が今の採点器でどう数えられるかを架空のデータで確かめる（変更票 #69 r9・R97-04・B6・2026-10-20）。

実録の B2 は止まっている間エンジンを切り、合図でかけてすぐ動き出す。かける音（セル・かかった直後）は合成（D20・止まっている間は無音）に無い音。
今の数え方（この試験で固定し、結果 md と変更票に事実として書く）:
  E1 かける音で出た通知が成功の窓 [t_start − 1, t_cpa + 1]（t_start = t_cpa − 6）の中なら、その事象の「成功」に数える（リードは最初の知らせから）
  E2 窓より前に出た通知は、成功にも誤警告にも数えない（正例の本では窓の外の発火は数えない。誤警告は負例の本だけ）
  E3 窓より前の通知しか無ければ「成功しなかった」になる
  E4（#69 r23・本人 10/20 03:34）動き出し t_move があれば始まり = max(t_cpa − 6, t_move)。かける音（13.5 s）は窓 [15, 21] の外 → 数えない。窓の中で動き出し前（15.5 s）の知らせは成功で before_move=1
使い方: python scripts/test_step20_b2_engine.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s20v3_b2", ROOT / "scripts/step20_realsmoke_score_v3.py")
S = importlib.util.module_from_spec(spec); sys.modules["s20v3_b2"] = S; spec.loader.exec_module(S)

COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "区分", "状態", "横距離m", "n_car", "scored"]


def main() -> int:
    ok = []
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    # B2-01: 線 2.0 m（caution）・最接近 20.0 s・t_start = 14.0（pre 6 s）→ 成功の窓 [13.0, 21.0]
    row = dict(zip(COLS, ["eval_field_clip00201", "1", "B2-01", "", "", "car_drive", "B", "14", "20", "", "B", "静止", "2.0", "1", "1"]))
    pred = {"eval_field_clip00201": {100: [(4, 180.0, 0.0, 9.0)]}}      # 予測は空でなければよい（発火は下で差し替える）
    orig = S.fires_of
    try:
        # E1: かける音の通知が 13.5 s（窓の中）＋ 走り出しの通知が 17.0 s → 成功・リードは最初の知らせ 20 − 13.5 = 6.5 s
        S.fires_of = lambda fr, cfg, nf: [(13.5, 180.0, "中", 4), (17.0, 180.0, "強", 4)]
        ev, neg, extras = S.evaluate([row], pred, cfg43, (2.5, 1.5), 7.5, True)
        e = ev[0]
        ok.append(("E1 窓の中のかける音の通知 → 成功・lead_first 6.5・誤警告 0", e["notified"] is True and abs(e["lead_first"] - 6.5) < 1e-9 and neg["n_false"] == 0))
        # E2: かける音の通知が 12.0 s（窓より前）＋ 走り出し 17.0 s → 成功は 17.0 の知らせから（lead 3.0）・12.0 は成功にも誤警告にも数えない
        S.fires_of = lambda fr, cfg, nf: [(12.0, 180.0, "中", 4), (17.0, 180.0, "強", 4)]
        ev, neg, extras = S.evaluate([row], pred, cfg43, (2.5, 1.5), 7.5, True)
        e = ev[0]
        ok.append(("E2 窓より前のかける音の通知は数えない（成功は 17.0 から・lead 3.0・誤警告 0）", e["notified"] is True and abs(e["lead"] - 3.0) < 1e-9 and abs(e["lead_first"] - 3.0) < 1e-9 and neg["n_false"] == 0))
        # E3: かける音の通知（12.0 s）しか無い → 成功しなかった・誤警告 0
        S.fires_of = lambda fr, cfg, nf: [(12.0, 180.0, "中", 4)]
        ev, neg, extras = S.evaluate([row], pred, cfg43, (2.5, 1.5), 7.5, True)
        e = ev[0]
        ok.append(("E3 窓より前の通知だけ → 成功しなかった・誤警告 0", e["notified"] is False and e["lead_first"] is None and neg["n_false"] == 0))
        # E4: 動き出し 16.0 s（step19d が t_start = 16.0 にしている）・かける音 13.5 は窓の外・15.5 は窓の中で動き出し前
        row4 = dict(row, t_start="16", t_move_s="16.0")
        S.fires_of = lambda fr, cfg, nf: [(13.5, 180.0, "中", 4), (15.5, 180.0, "中", 4), (17.0, 180.0, "強", 4)]
        ev, neg, extras = S.evaluate([row4], pred, cfg43, (2.5, 1.5), 7.5, True)
        e = ev[0]
        ok.append(("E4 動き出しから採点: 13.5 は数えない・15.5 で成功（lead_first 4.5）・before_move 1", e["notified"] is True and abs(e["lead_first"] - 4.5) < 1e-9 and e["before_move"] == 1 and neg["n_false"] == 0))
        S.fires_of = lambda fr, cfg, nf: [(13.5, 180.0, "中", 4), (17.0, 180.0, "強", 4)]
        ev, neg, extras = S.evaluate([row4], pred, cfg43, (2.5, 1.5), 7.5, True)
        ok.append(("E4' かける音だけ前・走り出しで成功 → before_move 0", ev[0]["notified"] is True and ev[0]["before_move"] == 0))
    finally:
        S.fires_of = orig
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
