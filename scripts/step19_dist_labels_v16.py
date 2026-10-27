# -*- coding: utf-8 -*-
"""v16 の（v15 と同じ水平距離ラベル。plan だけ v16）距離付き 6 列ラベル生成 = v13 版 ＋ **距離を水平距離に**（2026-09-04 本人「至近警告は横距離で決めたい」）＋ マイク高さ mic_z。

差分:
  1. マイクは scene.json の mic["mic_z"]（v15 で行ごとに引いた高さ）で組み直す（v12/v13 は固定 1.5 m）
  2. 距離列 = 3D 距離 × cos(仰角) ＝ 水平距離（マイク直下からの水平距離。横距離の定義と一致）。
     3D 距離は書かない。通知規則の閾値（強 1.3 / 中 1.6 m）はこの列に対して働く＝横距離基準になる
検証ゲート（az/el が metadata と全行一致）は v13 と同じ。プリロール（S1）対応も同じ。

使い方: PYTHONPATH=scripts:src python scripts/step19_dist_labels_v16.py [--limit N]
        python scripts/step19_dist_labels_v16.py --v12val   # v12 fold2 val の水平距離ラベルを out/dataset_outdoor_siren_v12/metadata_dist_h に（交差評価用・元は触らない）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step19_dist_labels_v12 as s19  # noqa: E402
import step11_v16_render as v15  # noqa: E402  (m9.DS が v15 に切替わる)
from outdoor_seld.labels import frame_label_rows  # noqa: E402
from outdoor_seld.geometry import mic_heading_from_record  # noqa: E402

v13 = v15.v13
m9 = v15.m9
V12VAL = "--v12val" in sys.argv
DS = ROOT / "out" / ("dataset_outdoor_siren_v12" if V12VAL else "dataset_outdoor_siren_v16")
s19.DS = DS
s19.OUT = DS / ("metadata_dist_h" if V12VAL else "metadata_dist")


def rebuild_mic_z(mic_meta: dict) -> np.ndarray:
    z = float(mic_meta.get("mic_z", 1.5))
    if mic_meta.get("motion", "static") == "static":
        return np.array([0.0, 0.0, z])
    if mic_meta["motion"] == "walk_cross_y":
        v = float(mic_meta["walk_speed_mps"]); ts = float(mic_meta["t_stop_s"]); y0 = -0.5 - v * ts
        return np.array([[0.0, 0.0, y0, z], [ts, 0.0, -0.5, z], [10.0, 0.0, -0.5, z]])
    v = float(mic_meta["walk_speed_mps"])
    d = float(mic_meta.get("walk_dir_x", 1.0))
    x0 = -d * v * 5.0
    return np.array([[0.0, x0, 0.0, z], [10.0, x0 + d * v * 10.0, 0.0, z]])


def regen_clip_v15(clip: str, c: float) -> int:
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    mic = rebuild_mic_z(scene["mic"])
    # 2026-10-26 先行分 §1: 描画 v30 の本は metadata の方位がマイク座標（scene.json の mic_heading_deg）。照合の門で同じ向きを通す。
    # 鍵が無い本（v29 まで）は None = 今までどおり。距離・仰角は向きで変わらない
    hd = (mic_heading_from_record(scene["mic"]["mic_heading_deg"], (np.arange(int(round(m9.CLIP / 0.1))) + 0.5) * 0.1)
          if "mic_heading_deg" in scene["mic"] else None)
    per_src = {}
    for src in scene["sources"]:
        if src.get("no_label"):
            continue
        cls_idx = v13.v12.CLASS_IDX_V12[src["class"]]
        wp = np.array(src["wp"], float)
        pre = float(src.get("preroll_s", 0.0))
        if pre > 0:
            wp, act_from = v13._extend_back(wp, pre), -pre
        else:
            act_from = src["t_on"]
        _, dbg = frame_label_rows(wp, mic, clip_len_sec=m9.CLIP, class_idx=cls_idx,
                                  track_idx=src.get("track", 0),
                                  source_active_from=act_from,
                                  source_active_until=src["t_off"], c=c, heading_deg=hd)
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
        assert (az_i, el_i) == (az, el), f"{clip}: az/el不一致 (frame={k},cls={ci},tr={tr})"
        d3 = float(dbg["dist"][k])
        dh = d3 * float(np.cos(np.radians(dbg["el"][k])))      # 水平距離
        assert np.isfinite(dh), f"{clip}: dist非有限 frame={k}"
        out_lines.append(f"{line},{dh:.2f}")
    s19.OUT.mkdir(parents=True, exist_ok=True)
    with open(s19.OUT / f"{clip}.csv", "w", newline="\n") as f:
        f.write("".join(s + "\n" for s in out_lines))
    return len(out_lines)


s19.regen_clip = regen_clip_v15


def main_v12val() -> None:
    """v12 fold2 val（work/scene.json があるクリップだけ）の水平距離ラベル → metadata_dist_h（交差評価用）。"""
    import time
    c = m9.sound_speed(m9.TEMP_C)
    clips = sorted(d.name for d in (DS / "work").iterdir() if (d / "scene.json").exists() and (DS / "metadata" / f"{d.name}.csv").exists())
    t0 = time.perf_counter(); n = 0
    for i, clip in enumerate(clips):
        n += regen_clip_v15(clip, c)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(clips)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"DONE(v12val): {len(clips)}本 / 総行 {n:,} -> {s19.OUT}", flush=True)


if __name__ == "__main__":
    if V12VAL:
        main_v12val()
    else:
        sys.argv = [a for a in sys.argv if a != "--v12val"]
        s19.main()
