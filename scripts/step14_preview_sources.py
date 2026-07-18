# -*- coding: utf-8 -*-
"""Step 14: 合成クリーン音源の試聴用WAVを書き出す（本人の耳での確認用）。

out/preview_sources_v9/ に:
  - 各クラスのドライ音（マイク・距離・反射なしの素の合成音）を6秒ずつ
  - 実際のデータセットからの「マイクで聞いた音」例（mixのWチャンネル）を3本
すべて試聴しやすいようにピーク0.7へ正規化（絶対較正の生レベルは小さすぎるため。
データセット本体は無変更）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.alert_sounds import (make_backup_beep, make_bike_bell,  # noqa: E402
                                       make_crossing, make_horn)
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.siren import make_peepo_siren, make_siren  # noqa: E402

OUT = ROOT / "out" / "preview_sources_v9"
DS = ROOT / "out" / "dataset_outdoor_siren_v9"
FS = 48000
DUR = 6.0


def norm(x, peak=0.7):
    m = np.max(np.abs(x))
    return (x * (peak / m) if m > 0 else x).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dries = {
        "01_siren_peepo": make_peepo_siren(DUR, FS),
        "02_siren_wail": make_siren(DUR, FS),
        "03_horn": make_horn(DUR, FS, np.random.default_rng(1)),
        "04_backup_beep": make_backup_beep(DUR, FS),
        "05_bike_bell": make_bike_bell(DUR, FS),
        "06_crossing": make_crossing(DUR, FS),
        "07_car_drive_v9": make_car_v9(DUR, FS, rng),
    }
    for name, x in dries.items():
        sf.write(OUT / f"{name}_dry.wav", norm(x), FS)
        print("wrote", f"{name}_dry.wav")

    # 実データからの「マイクで聞いた音」例（mix W ch）: 大音量サイレン・踏切入り・交差点
    examples = ["fold3_room1_mix212", "fold1_room1_mix038", "fold2_room9_mix001"]
    for name in examples:
        p = DS / "foa" / f"{name}.flac"
        if not p.exists():
            continue
        foa = np.asarray(sf.read(p)[0], np.float64).T
        s = json.loads((DS / "work" / name / "scene.json").read_text())
        srcs = "+".join(x["class"] for x in s["sources"])
        sf.write(OUT / f"10_mix_{name}_{srcs}.wav", norm(foa[0]), 24000)
        print("wrote", f"10_mix_{name}_{srcs}.wav")

    (OUT / "README.txt").write_text(
        "試聴メモ（2026-07-17）\n"
        "- *_dry.wav = 合成の素の音（距離・反射・雑音なし、ピーク0.7に正規化）\n"
        "- 10_mix_* = 実際のv9クリップをマイクで聞いた音（W ch、正規化済み）\n"
        "  mix212=大音量サイレン+車 / mix038=ベル+踏切+車(歩行) / room9=交差点サイレン\n"
        "- データセット本体は無変更（これは試聴用コピー）\n", encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
