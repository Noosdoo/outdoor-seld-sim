# -*- coding: utf-8 -*-
"""scripts/field_exclusion.py の受け入れ（変更票 #75 決め直し 6 の 5・架空例だけ）。期待する分母と成否を書いて確かめる。
  E0 時刻の直し: 全長の録音の秒 [12.1, 空) → 録音の長さ 19.04・cut_offset 5.34 で切り出しの [6.76, 13.70)／入り ≥ 出 は止める
  E1 窓の一部が除外 → 部分観察（成功にも失敗にも数えない）
  E2 窓の全部が除外 → 部分観察
  E3 除外中だけ通知（窓の一部が除外・発火は除外の中だけ）→ 部分観察で成功にしない／窓が全部見える本で発火が除外の中だけ → 数えず失敗
  E4 除外の直後に通知: 除外が履歴（窓の前 4 秒）に重なる → 部分観察／除外が履歴より前で終わる → 窓の中の発火で成功
  E5 有効フレーム 0（除外が採点のフレームを全部覆う）→ 正解の残り 0 で欠測（None）・分母 0
  E6 フレームの差集合: 除外中の予測は誤検出に数えず、正解も分母から外す（有効フレーム数を併記）
  E7 表の頭の数（summarize）: 欠測の本・部分観察の事象・通常の分母
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import field_exclusion as X  # noqa: E402


def main() -> int:
    ok = []
    iv = X.to_clip([(12.1, None)], 5.34, 19.04)
    ok.append(("E0 時刻の直し: [12.1, 空) → 切り出しの [6.76, 13.70)", len(iv) == 1 and abs(iv[0][0] - 6.76) < 1e-9 and abs(iv[0][1] - 13.70) < 1e-9))
    try:
        X.to_clip([(5.0, 4.0)], 0.0, 10.0); bad = False
    except ValueError:
        bad = True
    ok.append(("E0 入り ≥ 出 は止める", bad))
    # 鳴り始め 7.0 s → 窓 [6, 11]・観察の要る範囲 [2, 11]
    r1 = X.notification(7.0, [8.0], [(9.0, 12.0)])
    ok.append(("E1 窓の一部が除外 → 部分観察（8.0 s の発火があっても成功にしない）", r1["status"] == "partial"))
    r2 = X.notification(7.0, [8.0], [(5.5, 12.0)])
    ok.append(("E2 窓の全部が除外 → 部分観察", r2["status"] == "partial"))
    r3a = X.notification(7.0, [9.5], [(9.0, 10.0)])
    ok.append(("E3 除外中だけ通知（窓の一部が除外）→ 部分観察で成功にしない", r3a["status"] == "partial"))
    r3b = X.notification(3.0, [9.5], [(9.0, 10.0)], warmup_s=1.0)
    ok.append(("E3 窓が全部見える本で発火が除外の中（9.5 s）だけ → 数えず失敗", r3b["status"] == "fail" and r3b["window"] == (2.0, 7.0)))
    r4a = X.notification(7.0, [6.2], [(3.0, 5.9)])
    ok.append(("E4 除外の直後に通知・除外が履歴（[2, 6)）に重なる → 部分観察", r4a["status"] == "partial"))
    r4b = X.notification(7.0, [6.2], [(0.5, 1.9)])
    ok.append(("E4 除外が履歴より前で終わる（1.9 < 2.0）→ 直後の発火 6.2 s で成功", r4b["status"] == "success" and abs(r4b["first_fire"] - 6.2) < 1e-9))
    ex5 = X.frames_in(X.to_clip([(4.0, None)], 0.0, 10.0))
    m5 = X.frame_metric(set(range(40, 100)), set(range(50, 60)), ex5)
    ok.append(("E5 有効フレーム 0 → 分母 0・正解の残り 0 で欠測（None）・誤検出 0", m5["n_valid"] == 0 and m5["n_gt"] == 0 and m5["recall"] is None and m5["n_fp"] == 0 and m5["n_excluded"] == 60))
    ex6 = X.frames_in([(8.0, 9.0)])
    m6 = X.frame_metric(set(range(60, 95)), set(range(60, 70)) | set(range(80, 90)) | {97}, ex6)
    ok.append(("E6 差集合: 除外 [8, 9) の 10 フレームを外す・有効 50 フレーム（5.0 s）・正解 25・当たり 10・誤検出は 97 の 1 つだけ（除外中の 80〜89 は数えない）",
               m6["n_valid"] == 50 and m6["valid_s"] == 5.0 and m6["n_gt"] == 25 and m6["n_hit"] == 10 and m6["n_fp"] == 1 and m6["n_excluded"] == 10))
    ok.append(("E6 フレームの境目: [8.0, 9.0) は 80〜89 だけ（79・90 は入らない）", ex6 == set(range(80, 90))))
    s = X.summarize([{"frame": m5, "notifications": [r1, r4b]}, {"frame": m6, "notifications": [r3b]}])
    ok.append(("E7 表の頭: 本 2・欠測 1・事象 3・部分観察 1・通常の分母 2・成功 1", s == {"n_takes": 2, "n_missing_takes": 1, "n_events": 3, "n_partial": 1, "n_scored_events": 2, "n_success": 1}))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
