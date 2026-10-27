# -*- coding: utf-8 -*-
"""正解（metadata_dist）をそのまま「予測」の形にする = オラクル入力（変更票 #45・2026-10-02）。

統一採点器（_score_unified.py）に正解を予測として渡すと、「知覚が完璧ならこの通知規則はここまで届く」という規則の上限が出る。
モデルに依らない行なので、v1／v2 の 4 セルと一緒に並べて、知覚の誤りで失った分を読む。

入力: --plan <manifest.csv>（clip_id 列・--split で絞る・--clip-max で番号の上限）、--meta <metadata_dist フォルダ>（<clip_id>.csv = frame,class,track,az,el,dist の 6 列・0 バイトは正解なし）
出力: --out <csv>（予測と同じ 7 列 clip,frame,cls,track,az,el,dist・ヘッダなし・frame と class は整数・距離は小数のまま）
使い方:
    python scripts/_gt_as_pred.py --plan out/dataset_outdoor_siren_v12_conf/plan/assignment_v12conf.csv --split eval --clip-max 1200 \
        --meta out/dataset_outdoor_siren_v12_conf/metadata_dist --out out/v2final_conf_unified/gt_as_pred_core.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def clip_number(clip_id: str) -> int | None:
    m = re.search(r"(\d{4,5})$", clip_id)
    return int(m.group(1)) if m else None


def select_clips(plan: Path, split: str | None, clip_max: int | None) -> list[str]:
    with open(plan, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if split and (r.get("split") or "") != split:
            continue
        c = r["clip_id"]
        if clip_max is not None:
            n = clip_number(c)
            if n is None or n > clip_max:
                continue
        out.append(c)
    return out


def convert(clips: list[str], meta: Path, out: Path) -> tuple[int, int, int]:
    """戻り = (本数, 正解のあった本数, 書いた行数)"""
    n_gt = n_rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as w:
        for c in clips:
            p = meta / f"{c}.csv"
            if not p.is_file():
                raise FileNotFoundError(f"正解が無い: {p}")
            if p.stat().st_size == 0:
                continue
            n_gt += 1
            with open(p, encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip() == "" or not row[0].strip().lstrip("-").isdigit():
                        continue      # ヘッダや空行
                    if len(row) < 6:
                        raise ValueError(f"{p.name}: 6 列でない行 {row}")
                    frame, cls, track, az, el, dist = row[:6]
                    w.write(f"{c},{int(float(frame))},{int(float(cls))},{int(float(track))},{float(az):g},{float(el):g},{float(dist):g}\n")
                    n_rows += 1
    return len(clips), n_gt, n_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--clip-max", type=int, default=None)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    clips = select_clips(Path(a.plan), a.split, a.clip_max)
    if not clips:
        print("名簿が空", file=sys.stderr)
        return 2
    n, n_gt, n_rows = convert(clips, Path(a.meta), Path(a.out))
    print(f"[gt_as_pred] {n} 本（正解あり {n_gt}）→ {n_rows} 行 -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
