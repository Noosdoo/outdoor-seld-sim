# -*- coding: utf-8 -*-
"""height_patch の高さゆらぎの種（依頼 3-2・敵対レビュー F9・2026-09-14）の試験。

受け入れ: (1) 8 ワーカーで引いた高さゆらぎが互いに独立（同じ item でも列が違う）、(2) エポックが変わると変わる、
(3) 周辺分布が N(0, JITTER) のまま（平均 ≈ 0・標準偏差 ≈ JITTER）、(4) 同じ worker/epoch/item なら再現する（決定的）。
torch が無い環境でも走るよう、torch.initial_seed / get_worker_info は差し替えて模擬する。
使い方: python scripts/test_height_jitter_seed.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HEIGHT_JITTER", "0.10")
_spec = importlib.util.spec_from_file_location("height_patch", ROOT / "server_sde/height_patch.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

# torch の模擬（実物があっても DataLoader の外では worker_info が None なので、同じ模擬で決定的に試す）
fake = types.ModuleType("torch")
fake.utils = types.SimpleNamespace(data=types.SimpleNamespace(get_worker_info=lambda: None))
fake.initial_seed = lambda: 0
sys.modules["torch"] = fake


def set_worker(worker: int | None, epoch: int):
    """DataLoader が worker ごと・epoch ごとに振る base seed を模擬（初期 seed + epoch * 8 + worker）"""
    fake.utils.data.get_worker_info = (lambda: types.SimpleNamespace(id=worker)) if worker is not None else (lambda: None)
    fake.initial_seed = lambda: 123456789 + epoch * 100003 + (worker or 0) * 7919


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)) if detail and not cond else ''}")


def draws(worker, epoch, idxs):
    set_worker(worker, epoch)
    return np.array([H._jitter_rng(i).normal(0.0, H.JITTER) for i in idxs])


def main() -> int:
    idxs = list(range(2000))
    per_worker = [draws(w, 0, idxs) for w in range(8)]
    # (1) 8 ワーカーで同じ item の値が全部違う（旧実装は 8 本とも同じ列）
    same = sum(np.allclose(per_worker[0], per_worker[w]) for w in range(1, 8))
    check("1 8 ワーカーの列は互いに独立（同じ列が 0 本）", same == 0, same)
    corr = max(abs(np.corrcoef(per_worker[0], per_worker[w])[0, 1]) for w in range(1, 8))
    check("2 ワーカー間の相関が小さい（|r| < 0.1）", corr < 0.1, corr)
    # (2) エポックが変わると同じ worker/item でも変わる
    e1 = draws(0, 1, idxs)
    check("3 エポックが変わると変わる", not np.allclose(per_worker[0], e1))
    # (3) 周辺分布 N(0, JITTER)
    allv = np.concatenate(per_worker)
    check("4 平均 ≈ 0（|mean| < 0.005）", abs(allv.mean()) < 0.005, allv.mean())
    check("5 標準偏差 ≈ JITTER（0.10 ± 0.005）", abs(allv.std() - H.JITTER) < 0.005, allv.std())
    # (4) 決定的
    check("6 同じ worker/epoch/item なら再現", np.allclose(per_worker[3], draws(3, 0, idxs)))
    # (5) 主プロセス（worker_info None）でも動き、HEIGHT_SEED に依存
    check("7 DataLoader の外（worker 無し）でも引ける", np.isfinite(draws(None, 0, idxs[:10])).all())
    n_ok = sum(ok for _, ok in results)
    print(f"{n_ok}/{len(results)} {'ALL PASS' if n_ok == len(results) else 'FAILED'}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
