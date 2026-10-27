# -*- coding: utf-8 -*-
"""test_hysteresis_pool_min.py — Astra RI73-05（2026-10-21）: 保持 stay が候補の下限 POOL_MIN（0.05）より小さいとき（相手 D の 0.1／0.0429）、
保持以下の候補が先に落ちて「継続」できない不具合の検査。直し = 下限を min(pool_min, stay) に。提案法の 0.7／0.3 の出力は変わらない。
  H1 入り 0.1・保持 0.0429・同じ向きで長さ 0.11 → 0.045 → 0.045: 3 フレームとも「いる」（0.045 は保持より大きい）
  H2 同じ入力で長さ 0.11 → 0.04 → 0.04: 2・3 フレーム目は「いない」（保持 0.0429 以下）
  H3 入り 0.7・保持 0.3 では 0.75 → 0.045 → 0.045 は 2・3 フレーム目「いない」（下限は 0.05 のまま = 従来どおり）・0.75 → 0.31 は「いる」
  H4 1 閾値（stay >= enter）は長さ > enter だけ
使い方: python scripts/test_hysteresis_pool_min.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hysteresis_patch", ROOT / "server_sde/hysteresis_patch.py")
HP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(HP)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
C = 9


def doa_of(lengths, cls=4, az_deg=0.0):
    doa = np.zeros((3, len(lengths), 4 * C), dtype=np.float32)
    for t, n in enumerate(lengths):
        doa[0, t, cls] = n * np.cos(np.radians(az_deg)); doa[0, t, C + cls] = n * np.sin(np.radians(az_deg))
    return doa


def main() -> int:
    ok = []
    s = HP.apply(doa_of([0.11, 0.045, 0.045]), C, 0.1, 0.0429)
    ok.append(("H1 0.1／0.0429: 0.11 → 0.045 → 0.045 は 3 フレームとも いる", s[0, :, 4].tolist() == [True, True, True]))
    s = HP.apply(doa_of([0.11, 0.04, 0.04]), C, 0.1, 0.0429)
    ok.append(("H2 0.1／0.0429: 0.11 → 0.04 → 0.04 は 2・3 フレーム目 いない（保持以下）", s[0, :, 4].tolist() == [True, False, False]))
    s = HP.apply(doa_of([0.75, 0.045, 0.045]), C, 0.7, 0.3)
    s2 = HP.apply(doa_of([0.75, 0.31, 0.31]), C, 0.7, 0.3)
    ok.append(("H3 0.7／0.3 は従来どおり（0.045 は いない・0.31 は いる）", s[0, :, 4].tolist() == [True, False, False] and s2[0, :, 4].tolist() == [True, True, True]))
    s = HP.apply(doa_of([0.75, 0.69, 0.71]), C, 0.7, 0.7)
    ok.append(("H4 1 閾値: 長さ > 入り だけ", s[0, :, 4].tolist() == [True, False, True]))
    s = HP.apply(doa_of([0.11, 0.045, 0.045]), C, 0.1, 0.0429, assoc=20.0, pool_min=0.05)
    ok.append(("H1b pool_min を明示 0.05 で渡しても保持 0.0429 で継続（min を取る）", s[0, :, 4].tolist() == [True, True, True]))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
