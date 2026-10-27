# -*- coding: utf-8 -*-
"""_warn_episode_eval の単体試験（監査 F2 の模擬 3 例・2026-09-08）。
  W1 持続サイレンを最初から正しく検出       → 到達 100%・遅延 ≤1 s・イベント外発火 0（旧採点器は「発火なし 0%」だった）
  W2 4 秒見逃してから正しく検出             → 到達 100%・遅延 ≈4 s（>1 s）・イベント外発火 0
  W3 正解 5.0〜5.9 s 正面 / 予測 6.0〜9.9 s 背面 → 到達 0%・イベント外発火 ≥1（旧採点器は「発火あり 100%」だった）
  W4 列車群（scenario=v12train）             → 正解を無視して発火は全部イベント外
使い方: python scripts/test_warn_episode_eval.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("warn_episode_eval", ROOT / "scripts/_warn_episode_eval.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)
CLS = 0   # siren


def write_gt(meta: Path, clip: str, frames, az: float):
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / f"{clip}.csv", "w", encoding="utf-8") as f:
        for k in frames:
            f.write(f"{k},{CLS},0,{az:.0f},0,10.00\n")


def write_pred(path: Path, clip: str, frames, az: float):
    with open(path, "w", encoding="utf-8") as f:
        for k in frames:
            f.write(f"{clip},{k},{CLS},0,{az:.0f},0,nan\n")


def run(tmp: Path, name: str, gt_frames, gt_az, pred_frames, pred_az, group="有"):
    clip = f"val_syn_clip{name}"
    meta = tmp / f"meta_{name}"
    write_gt(meta, clip, gt_frames, gt_az)
    pred = tmp / f"pred_{name}.csv"
    write_pred(pred, clip, pred_frames, pred_az)
    acc = W.evaluate(pred, [clip], meta, CLS, lambda c: group, minframe=40, tol=5, az_tol=30.0)
    return acc[group]


def main() -> int:
    checks = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = run(tmp, "0001", range(0, 100), 0.0, range(0, 100), 0.0)
        checks.append(("W1 到達 100%", a["ep"] == 1 and a["reached"] == 1))
        checks.append(("W1 遅延 ≤1 s", bool(a["delays"]) and a["delays"][0] <= 10))
        checks.append(("W1 イベント外発火 0", a["false"] == 0))
        b = run(tmp, "0002", range(0, 100), 0.0, range(40, 100), 0.0)
        checks.append(("W2 到達 100%", b["ep"] == 1 and b["reached"] == 1))
        checks.append(("W2 遅延 ≈4 s（>1 s）", bool(b["delays"]) and 35 <= b["delays"][0] <= 50))
        checks.append(("W2 イベント外発火 0", b["false"] == 0))
        checks.append(("W1 と W2 が遅延で区別できる", bool(a["delays"]) and bool(b["delays"]) and a["delays"][0] < b["delays"][0]))
        c = run(tmp, "0003", range(50, 60), 0.0, range(60, 100), 180.0)
        checks.append(("W3 到達 0%", c["ep"] == 1 and c["reached"] == 0))
        checks.append(("W3 イベント外発火 ≥1", c["false"] >= 1 and c["false_clips"] == 1))
        d = run(tmp, "0004", range(0, 100), 0.0, range(0, 100), 0.0, group="train")
        checks.append(("W4 列車群は発火が全部イベント外", d["fires"] >= 1 and d["false"] == d["fires"]))
    ok = all(x for _, x in checks)
    for name, x in checks:
        print(("PASS " if x else "FAIL ") + name)
    print(f"{sum(x for _, x in checks)}/{len(checks)} " + ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
