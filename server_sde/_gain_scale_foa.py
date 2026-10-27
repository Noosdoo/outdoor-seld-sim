# -*- coding: utf-8 -*-
"""較正の感度測定（実録後リスト 3b・2026-09-27）: FOA の flac を一律に ±g dB して別フォルダに写す。

推論スクリプト（_causal_infer_v17.py）は CAUSAL_FOA で音の置き場を上書きできるので、推論のコードは触らずに
「騒音計の誤差 ±1.4 dB（クラス 2）が装置の出力にどれだけ効くか」を、入力の音量を ±2 dB ずらして測る。

使い方: python _gain_scale_foa.py --src <foa dir> --dst <foa dir> --gain-db -2 [--glob 'val_syn_*.flac'] [--workers 8]
出力: <dst>/<同じ名前>.flac（PCM_24・同じ fs）と <dst>/../scale_report_<gain>.json（本数・クリップしたサンプル数・上位のファイル）
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf


def _one(args):
    src, dst, g = args
    x, fs = sf.read(src, dtype="float64", always_2d=True)
    y = x * (10.0 ** (g / 20.0))
    lim = 1.0 - 2.0 ** -23
    n_clip = int(np.sum(np.abs(y) > lim))
    y = np.clip(y, -lim, lim)
    sf.write(dst, y, fs, subtype="PCM_24", format="FLAC")
    return src.name, n_clip, float(np.max(np.abs(x)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--gain-db", type=float, required=True); ap.add_argument("--glob", default="val_syn_*.flac")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(a.glob))
    if not files:
        raise SystemExit(f"no files: {src}/{a.glob}")
    jobs = [(p, dst / p.name, a.gain_db) for p in files if not (dst / p.name).exists()]
    with Pool(a.workers) as pool:
        res = pool.map(_one, jobs, chunksize=16)
    clipped = sorted(((n, name) for name, n, _ in res if n), reverse=True)
    rep = {"src": str(src), "dst": str(dst), "gain_db": a.gain_db, "glob": a.glob, "n_files": len(files), "n_written": len(res),
           "n_files_clipped": len(clipped), "n_samples_clipped": int(sum(n for n, _ in clipped)),
           "top_clipped": clipped[:10], "max_abs_before_top": sorted(((m, name) for name, _, m in res), reverse=True)[:5]}
    tag = f"{a.gain_db:+g}".replace("+", "p").replace("-", "m")
    (dst.parent / f"scale_report_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst}: {len(res)} written (skipped {len(files) - len(res)} existing), clipped files {len(clipped)}, samples {rep['n_samples_clipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
