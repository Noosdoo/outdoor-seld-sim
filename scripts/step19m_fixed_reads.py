# -*- coding: utf-8 -*-
"""step19m_fixed_reads.py — 止まった音源（クラクション E2）の「いつ鳴っていたか」を知覚層の読みにする（Fable 仕様（先行分）§4・決定 ⑳・2026-10-26）。

知覚層の正解（step19k_field_perception.py）は、固定音源（horn・backup_beep・crossing）の座標を ann（x_m・y_m・高さ_m）から読める（build の fixed）。
足りなかったのは「いつ鳴っていたか」の読み（take・cls・track・t）だけ。この道具は台帳の events の E2 の各押しの**鳴り始め**（ラップ秒）と
**鳴り終わり**（終わり秒）から、押しごとに track を分けた 2 点（始まり・終わり）の読みを書く。間は step19k の直線でつなぐ（固定なので動かない）。
押しと押しの間は「隠れ」ではない（音が無いだけ）ので、hidden の印は付けない（step19k は押しの外のフレームに正解を作らない）。

時刻の約束は step19d（`step19d_field_csv_to_ann.py`）と同じ: 録音の秒 = ラップ秒 ＋ --lap-offset ＋ 拍手のずれ（拍手_録音側秒 − 拍手_音側秒・両方あるときだけ）。
鳴り終わりにも同じずれを足す。座標は step19d と同じ現場の座標のまま（音源の x_m・y_m は現場・マイクは ann の mic_x_m・mic_y_m・差は step19k で 1 回だけ）。
三脚の向き（E2 の 90／270 も）は step19k の tripod_heading の道を通す（この道具は向きを触らない・音には足さない）。

止める（黙って 0 本にしない）: E2 の押しの鳴り始めか鳴り終わりが空・数値でない・0 以下・鳴り終わり ≤ 鳴り始め／E2 が 1 本も無い／
--ann を渡したとき、その本の固定の座標（x_m・y_m・高さ_m）が台帳と違う。止めたときは理由の一覧を出して終了コード 2（出力は書かない）。

E2 の見分け: class = horn で、台本ID が E2- で始まる行（台本ID が空の古い形は、x_m・y_m・高さ_m が 3 つとも埋まっている horn の行）。
出力 = step19k の reads と同じ列（take, frame, cls, track, x, y, z, t）。take = 原本・cls = horn・track = その本の押しの順番（1, 2, …）・
x, y, z = 台帳の x_m・y_m・高さ_m（step19k は ann の固定の座標で上書きする）・t = 録音の秒。

使い方:
  python scripts/step19m_fixed_reads.py --events raw/ledger/events_fixed_20261029.csv --out out/realsmoke/reads/reads_fixed_20261029.csv [--ann out/realsmoke/ann_all.csv] [--lap-offset 0]
  → field_score_members.py --reads-fixed out/realsmoke/reads/reads_fixed_20261029.csv（知覚層の読みに足す）
試験: scripts/test_step19m_fixed_reads.py
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_COLS = ["take", "frame", "cls", "track", "x", "y", "z", "t"]
FIXED_KEYS = ("x_m", "y_m", "高さ_m")


def _num(s) -> float | None:
    s = (s or "").strip() if isinstance(s, str) else s
    if s in (None, ""):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def is_e2(r: dict) -> bool:
    if (r.get("class") or "").strip() != "horn":
        return False
    sid = (r.get("台本ID") or "").strip()
    if sid:
        return sid.upper().startswith("E2-")
    return all((r.get(k) or "").strip() for k in FIXED_KEYS)


def fixed_reads(events: list[dict], lap_offset: float = 0.0, ann: list[dict] | None = None) -> tuple[list[dict], list[str], list[str]]:
    """返り値 (reads の行, 止める理由, 知らせ)。止める理由があれば呼び出し側は書かない。"""
    stops, notes = [], []
    e2 = [r for r in events if is_e2(r)]
    if not e2:
        return [], ["E2（class = horn・台本ID E2-…）の行が 1 つも無い（台帳の取り違えか、E2 の無い日）"], []
    fixed_ann = {}
    for a in ann or []:
        if (a.get("class") or "").strip() == "horn" and all((a.get(k) or "").strip() for k in FIXED_KEYS):
            fixed_ann[Path(a.get("orig_file", "")).name.lower()] = tuple(float(a[k]) for k in FIXED_KEYS)
    by_take = defaultdict(list)
    for r in e2:
        take = (r.get("原本") or "").strip()
        who = f"{r.get('session_id', '')}/take{r.get('take_id', '')} event {r.get('event_id', '')}（{take or '原本なし'}）"
        if not take:
            stops.append(f"{who}: 原本が空"); continue
        xyz = [_num(r.get(k)) for k in FIXED_KEYS]
        if any(v is None for v in xyz) or xyz[2] < 0:
            stops.append(f"{who}: 固定の座標 x_m・y_m・高さ_m が 3 つとも数値で要る（{[r.get(k) for k in FIXED_KEYS]}）"); continue
        clap = 0.0
        cf, ca = _num(r.get("拍手_録音側秒")), _num(r.get("拍手_音側秒"))
        if cf is not None and ca is not None:
            clap = cf - ca                                    # step19d と同じ拍手のずれ
        on, off = _num(r.get("ラップ秒")), _num(r.get("終わり秒"))
        if on is None or on <= 0:
            stops.append(f"{who}: 鳴り始め（ラップ秒）が空か不正 '{r.get('ラップ秒', '')}'（波形の鳴り始めを書いてから）"); continue
        if off is None or off <= 0:
            stops.append(f"{who}: 鳴り終わり（終わり秒）が空か不正 '{r.get('終わり秒', '')}'（波形の鳴り終わりを書いてから・黙って 0 本にしない）"); continue
        t_on, t_off = on + lap_offset + clap, off + lap_offset + clap
        if t_off <= t_on:
            stops.append(f"{who}: 鳴り終わり {t_off:.3f} s が鳴り始め {t_on:.3f} s 以前"); continue
        a_xyz = fixed_ann.get(take.lower())
        if ann is not None and a_xyz is not None and any(abs(p - q) > 1e-6 for p, q in zip(a_xyz, xyz)):
            stops.append(f"{who}: ann の固定の座標 {a_xyz} と台帳の {tuple(xyz)} が違う（どちらかの写し間違い）"); continue
        if ann is not None and a_xyz is None:
            notes.append(f"{who}: ann にこの本の固定の座標が無い（step19k は台帳の座標をそのまま使わず、この本の正解を作らない）")
        by_take[take].append((t_on, t_off, xyz, r))
    rows = []
    for take, presses in sorted(by_take.items()):
        presses.sort(key=lambda p: p[0])
        for i in range(1, len(presses)):
            if presses[i][0] < presses[i - 1][1]:
                stops.append(f"{take}: 押し {i} と {i + 1} が重なる（{presses[i - 1][0]:.3f}〜{presses[i - 1][1]:.3f} と {presses[i][0]:.3f}〜）")
        for trk, (t_on, t_off, xyz, _) in enumerate(presses, start=1):
            for t in (t_on, t_off):
                rows.append({"take": take, "frame": "", "cls": "horn", "track": str(trk), "x": f"{xyz[0]:g}", "y": f"{xyz[1]:g}", "z": f"{xyz[2]:g}", "t": f"{t:.3f}"})
    return rows, stops, notes


def _read(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="止まった音源（E2）の鳴り始め・鳴り終わりを知覚層の読みにする（先行分 §4）")
    ap.add_argument("--events", type=Path, required=True, help="台帳の events（例 raw/ledger/events_fixed_<日付>.csv）")
    ap.add_argument("--out", type=Path, required=True, help="reads_fixed_<日付>.csv")
    ap.add_argument("--ann", type=Path, default=None, help="step19b の ann_all.csv（任意・固定の座標の照合）")
    ap.add_argument("--lap-offset", type=float, default=0.0, help="step19d と同じ（既定 0）")
    a = ap.parse_args(argv)
    rows, stops, notes = fixed_reads(_read(a.events), a.lap_offset, _read(a.ann) if a.ann else None)
    for n in notes:
        print("[注意] " + n)
    if stops:
        print(f"止める（{len(stops)} 件・出力は書かない）:")
        for s in stops:
            print("  - " + s)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(rows)
    takes = sorted({r["take"] for r in rows})
    n_press = len(rows) // 2
    print(f"E2 {len(takes)} 本・押し {n_press} 回 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
