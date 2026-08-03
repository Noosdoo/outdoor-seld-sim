# -*- coding: utf-8 -*-
"""役割③(至近車のリアルタイム区別)用: 距離付き6列ラベルの再生成。

既存の凍結データ(音・metadata・masks)は一切変更せず、scene.json＋masksから
DCASE2024形式の6列 [frame, class, track, az, el, dist_m] を metadata_dist/ に生成する。

正確さの担保（このスクリプトの本体）:
- 角度・フレーム・可聴ゲートは生成時と同一のコードパスを使う
  （outdoor_seld.labels.frame_label_rows ＝ 音のFOAエンコードと同じ apparent_azel_deg。
   距離はその関数が既に計算してdebugで返している値＝新しい計算式はゼロ）
- 「どの行が残ったか」（車の可聴ゲートの結果）は既存 metadata/*.csv を正として
  そのまま使う＝先頭5列は定義上ビット一致（ゲートの再現はしない。masksのSNRは
  2桁丸め保存でしきい値際の判定が再現不能なため）
- 検証ゲート: 既存の全行について az/el を同一コードパスで再計算し、整数一致
  しなければ即エラー（=行と音源の対応・マイク復元・角度計算の正しさを全行で証明）

使い方（DynamicSound venv）:
    python scripts/step19_dist_labels.py            # v11 core 7,200本
    python scripts/step19_dist_labels.py --limit 50 # スモーク
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

import step11_v11_render as m11  # noqa: E402  (v11チェーン=定数がv11状態になる)
m9 = m11.m9
from outdoor_seld.labels import frame_label_rows  # noqa: E402

DS = HERE.parent / "out" / "dataset_outdoor_siren_v11"
OUT = DS / "metadata_dist"
N_FRAMES = 100


def rebuild_mic(mic_meta: dict) -> np.ndarray:
    """scene.jsonのmic情報から生成時と同一のマイク配置を復元する。"""
    if mic_meta["motion"] == "static":
        return np.array([0.0, 0.0, 1.5])
    v = mic_meta["walk_speed_mps"]
    d = mic_meta.get("walk_dir_x", 1.0)
    x0 = -d * v * 5.0
    return np.array([[0.0, x0, 0.0, 1.5], [10.0, x0 + d * v * 10.0, 0.0, 1.5]])


def regen_clip(clip: str, c: float) -> tuple[int, int]:
    """既存metadataの各行を正とし、距離列だけを追加する。

    可聴ゲート(SNR≥0dB)の再現はしない（masksのSNRは2桁丸めで保存されており、
    しきい値ぎりぎりのフレームで判定が反転しうるため）。代わりに
    「どの行が残ったか」は既存metadataをそのまま使い、行ごとに
    (class,track)→音源対応で距離を引く。検証は行ごとの az/el 再計算一致。
    """
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    mic = rebuild_mic(scene["mic"])

    # (class_idx, track) -> フレーム別の az/el/dist（生成時と同一の計算経路）
    per_src = {}
    for src in scene["sources"]:
        cls_idx = m9.CLASS_IDX[src["class"]]
        _, dbg = frame_label_rows(np.array(src["wp"], float), mic,
                                  clip_len_sec=m9.CLIP, class_idx=cls_idx,
                                  track_idx=src.get("track", 0),
                                  source_active_from=src["t_on"],
                                  source_active_until=src["t_off"], c=c)
        key = (cls_idx, src.get("track", 0))
        assert key not in per_src, f"{clip}: (class,track)が一意でない {key}"
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
            f"{clip}: 行(frame={k},cls={ci},tr={tr}) az/el再計算不一致 " \
            f"({az_i},{el_i}) vs ({az},{el})"
        d = float(dbg["dist"][k])
        assert np.isfinite(d), f"{clip}: dist非有限 frame={k}"
        out_lines.append(f"{line},{d:.2f}")

    with open(OUT / f"{clip}.csv", "w", newline="\n") as f:
        f.write("".join(s + "\n" for s in out_lines))
    return len(out_lines), len(scene["sources"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    c = m9.sound_speed(m9.TEMP_C)
    clips = sorted(p.stem for p in (DS / "metadata").glob("*.csv"))
    if args.limit:
        clips = clips[:args.limit]
    t0 = time.perf_counter()
    n_rows = n_empty = 0
    for i, clip in enumerate(clips):
        r, s = regen_clip(clip, c)
        n_rows += r
        n_empty += int(r == 0)
        if (i + 1) % 800 == 0:
            print(f"{i+1}/{len(clips)} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    print(f"DONE: {len(clips)}本 / 総ラベル行 {n_rows:,} / 空 {n_empty}本 / "
          f"5列一致 全数PASS -> {OUT}")


if __name__ == "__main__":
    main()
