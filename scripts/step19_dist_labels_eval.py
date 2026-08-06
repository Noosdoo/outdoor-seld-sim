# -*- coding: utf-8 -*-
"""v11評価専用セット(3,246本)の距離付き6列ラベル生成（step19のeval移植）。

v11版との差分はマイク復元のみ: 評価シナリオは waypoints保存型・停止→歩行(t_stop)型を
含むため、step12_notify_v9._mic_wp と同一の復元規約を使う。
検証ゲートは同一: 既存metadata全行の az/el を同一コードパスで再計算し整数一致。
使い方: python scripts/step19_dist_labels_eval.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v11_render as m11  # noqa: E402
m9 = m11.m9
from outdoor_seld.labels import frame_label_rows  # noqa: E402

DS = HERE.parent / "out" / "dataset_outdoor_siren_v11_eval"
OUT = DS / "metadata_dist"
N_FRAMES = 100


def rebuild_mic(mic_meta: dict) -> np.ndarray:
    """step12_notify_v9._mic_wp と同一規約（waypoints/static/walk/t_stop）。"""
    if "waypoints" in mic_meta:
        return np.array(mic_meta["waypoints"], dtype=np.float64)
    if mic_meta["motion"] == "static":
        return np.array([0.0, 0.0, 1.5])
    if mic_meta["motion"] == "walk":
        v = mic_meta["walk_speed_mps"]
        d = mic_meta.get("walk_dir_x", 1.0)
        x0 = -d * v * 5.0
        return np.array([[0.0, x0, 0.0, 1.5], [10.0, x0 + d * v * 10.0, 0.0, 1.5]])
    v, ts = mic_meta["walk_speed_mps"], mic_meta["t_stop_s"]
    y0 = -0.5 - v * ts
    return np.array([[0.0, 0.0, y0, 1.5], [ts, 0.0, -0.5, 1.5],
                     [10.0, 0.0, -0.5, 1.5]])


def regen_clip(clip: str, c: float) -> int:
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    mic = rebuild_mic(scene["mic"])
    per_src = {}
    for src in scene["sources"]:
        cls_idx = m9.CLASS_IDX[src["class"]]
        _, dbg = frame_label_rows(np.array(src["wp"], float), mic,
                                  clip_len_sec=m9.CLIP, class_idx=cls_idx,
                                  track_idx=src.get("track", 0),
                                  source_active_from=src["t_on"],
                                  source_active_until=src["t_off"], c=c)
        key = (cls_idx, src.get("track", 0))
        assert key not in per_src, f"{clip}: (class,track)重複 {key}"
        per_src[key] = dbg

    out_lines = []
    for line in (DS / "metadata" / f"{clip}.csv").read_text().splitlines():
        k, ci, tr, az, el = (int(v) for v in line.split(","))
        dbg = per_src[(ci, tr)]
        az_i = int(np.rint(dbg["az"][k]))
        if az_i == 180:
            az_i = -180
        el_i = int(np.rint(dbg["el"][k]))
        assert (az_i, el_i) == (az, el), \
            f"{clip}: az/el不一致 (frame={k},cls={ci},tr={tr})"
        d = float(dbg["dist"][k])
        assert np.isfinite(d), f"{clip}: dist非有限 frame={k}"
        out_lines.append(f"{line},{d:.2f}")
    with open(OUT / f"{clip}.csv", "w", newline="\n") as f:
        f.write("".join(s + "\n" for s in out_lines))
    return len(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    c = m9.sound_speed(m9.TEMP_C)
    clips = sorted(p.stem for p in (DS / "metadata").glob("*.csv"))
    if args.limit:
        clips = clips[: args.limit]
    t0 = time.perf_counter()
    n_rows = n_empty = 0
    for i, clip in enumerate(clips):
        r = regen_clip(clip, c)
        n_rows += r
        n_empty += int(r == 0)
        if (i + 1) % 800 == 0:
            print(f"{i+1}/{len(clips)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"DONE: {len(clips)}本 / 総行 {n_rows:,} / 空 {n_empty}本 / "
          f"5列一致 全数PASS -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
