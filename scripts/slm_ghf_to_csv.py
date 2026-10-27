# -*- coding: utf-8 -*-
"""騒音計 CENTER 392（SE392 ソフト）の「Save single selected data set」= .GHF（独自の二進形式）を CSV に起こす（2026-10-07・Fable）。

読み方（10/7 の実物 slm_20261007_bicycle.GHF・111,234 バイトで確かめた並び）:
  - 先頭 4 バイト = ファイル長 − 4（little endian の u32）。続く 16 バイトは GUID・その後 u32 7015・u32 500・f32 10/25/75/125 と 0 詰め
  - オフセット 2028 に u32 の記録数 n、2032 から f32 が n 個 = 1 秒ごとの dB（A・FAST）の読み
  - その後に u32 の印の列（1/0/3 …・意味は未確認）→ 使わない
  - 日時は入っていない（ソフトの画面で読む・録音とは拍手で合わせる）
使い方: python scripts/slm_ghf_to_csv.py <in.GHF> <out.csv> [--start "2026-10-07 16:48:21"] [--interval 1.0]
出力: t_s,dB（t_s は記録の先頭からの秒・interval 刻み）。確認用に件数・平均・最大を表示する（ソフトの表示と突き合わせる）。
"""
from __future__ import annotations
import argparse, csv, math, struct, sys
from pathlib import Path

COUNT_OFFSET = 2028


def read_ghf(path: Path) -> list[float]:
    b = path.read_bytes()
    if len(b) < COUNT_OFFSET + 4 or struct.unpack_from("<I", b, 0)[0] != len(b) - 4:
        raise SystemExit(f"{path.name}: 先頭の長さが合わない（GHF ではない？ size={len(b)}）")
    n = struct.unpack_from("<I", b, COUNT_OFFSET)[0]
    if n <= 0 or COUNT_OFFSET + 4 + 4 * n > len(b):
        raise SystemExit(f"{path.name}: 記録数 {n} が長さと合わない")
    vals = list(struct.unpack_from(f"<{n}f", b, COUNT_OFFSET + 4))
    bad = [v for v in vals if not (0.0 <= v <= 140.0)]
    if bad:
        raise SystemExit(f"{path.name}: dB の範囲外の値がある {bad[:5]}（読み方が違う？）")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--start", default="", help="記録の開始日時（ソフトの表示を写す・任意・CSV の先頭にコメントで残す）")
    ap.add_argument("--interval", type=float, default=1.0, help="記録間隔 [s]（既定 1.0）")
    a = ap.parse_args()
    vals = read_ghf(Path(a.src))
    leq = 10 * math.log10(sum(10 ** (v / 10) for v in vals) / len(vals))
    dst = Path(a.dst); dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# source={Path(a.src).name} start={a.start or 'unknown'} interval_s={a.interval} n={len(vals)} weighting=A time=FAST\n")
        w = csv.writer(f); w.writerow(["t_s", "dB"])
        for i, v in enumerate(vals):
            w.writerow([f"{i * a.interval:.1f}", f"{v:.1f}"])
    print(f"{dst}: n={len(vals)} 平均 {sum(vals)/len(vals):.1f} 最大 {max(vals):.1f} 最小 {min(vals):.1f} Leq {leq:.1f} dB（ソフトの表示と合うか確かめる）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
