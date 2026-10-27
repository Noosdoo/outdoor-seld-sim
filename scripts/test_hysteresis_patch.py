# -*- coding: utf-8 -*-
"""server_sde/hysteresis_patch.py（推論の後処理）が、探索に使った scripts/_hysteresis_grid2.py の decode_general（型 amp）と同じ出力になるかの検査。
材料 = 可逆版の度合い（out/detect_scores_2026-09-16/v2/scores_*/scores）。度合いの行から (3, T, 4C) の doa を組み立て、
両方の規則で「いる」track を決めて本番と同じ統合（D.decode）に通し、フレームごとの出力が一致するかを見る。

使い方: python scripts/test_hysteresis_patch.py [--scores <dir>] [--n 300] [--pairs 0.7:0.3,0.5:0.5,0.9:0.5]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


H2 = _load("hyst2", ROOT / "scripts" / "_hysteresis_grid2.py")
HP = _load("hysteresis_patch", ROOT / "server_sde" / "hysteresis_patch.py")
D = H2.D
NCLS = 8


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def doa_from_scores(frames_scores, T=100):
    """度合いの行 (trk, cls, nrm, x, y, z, d, s05) → doa (3, T, 4C) float32（無い行は 0 = 長さ 0）と、行の引き当て表"""
    doa = np.zeros((3, T, 4 * NCLS), dtype=np.float32)
    rows = {}
    for k, rs in frames_scores.items():
        for r in rs:
            trk, cls, nrm, x, y, z, d, s05 = r
            doa[trk, k, cls] = np.float32(x); doa[trk, k, NCLS + cls] = np.float32(y); doa[trk, k, 2 * NCLS + cls] = np.float32(z)
            doa[trk, k, 3 * NCLS + cls] = np.float32(0.0)
            rows[(k, trk, cls)] = r
    return doa, rows


def main() -> int:
    argv = list(sys.argv[1:])
    sdir = ROOT / arg(argv, "--scores", "out/detect_scores_2026-09-16/v2/scores_v18ft_e149_v12causal/scores")
    n = int(arg(argv, "--n", "300"))
    pairs = [tuple(float(v) for v in p.split(":")) for p in arg(argv, "--pairs", "0.7:0.3,0.5:0.5,0.9:0.5,0.76:0.05").split(",")]
    files = sorted(sdir.glob("*.csv"))[:n]
    if not files:
        print(f"scores が無い: {sdir}"); return 2
    total_bad = 0
    for enter, stay in pairs:
        bad = 0; nfr = 0; nact = 0
        for f in files:
            fs = D.load_scores(f)
            T = max(fs) + 1 if fs else 100
            doa, rows = doa_from_scores(fs, T)
            sed = HP.apply(doa, NCLS, enter, stay)
            # 推論側の「いる」行を本番と同じ統合に通す
            act_by_frame = {}
            for trk, k, cls in zip(*np.where(sed)):
                r = rows.get((int(k), int(trk), int(cls)))
                assert r is not None, f"{f.name}: いる行が度合いの表に無い ({k},{trk},{cls})"
                act_by_frame.setdefault(int(k), []).append(r)
            got = {k: D.decode({k: v}, -1.0).get(k, []) for k, v in act_by_frame.items()}
            got = {k: v for k, v in got.items() if v}
            # 探索側
            ref = H2.decode_general(fs, enter, stay, HP.ASSOC_DEG)
            nfr += T; nact += sum(len(v) for v in ref.values())
            if got != ref:
                bad += 1
                if bad <= 3:
                    ks = sorted(set(got) | set(ref))
                    diff = [k for k in ks if got.get(k) != ref.get(k)]
                    print(f"  不一致 {f.name} 入り {enter} 保持 {stay}: フレーム {diff[:5]} ... got={got.get(diff[0])} ref={ref.get(diff[0])}")
        print(f"入り {enter:.2f} 保持 {stay:.2f}: {len(files)} 本 / {nfr:,} フレーム / いる {nact:,} 件 → 不一致 {bad} 本")
        total_bad += bad
    print("OK: 推論の後処理と探索の規則は同じ出力" if total_bad == 0 else f"NG: 不一致 {total_bad}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
