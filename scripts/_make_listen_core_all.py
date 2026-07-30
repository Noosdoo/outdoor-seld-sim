# -*- coding: utf-8 -*-
"""v11学習コアtrain 4,800本を全部、試聴用モノラルWAVに変換する。

- 音=Wチャンネル（無指向成分=マイク位置で聞こえた音）。試聴用に音量正規化
  （物理較正レベルは保持しない。正はflac側）
- ファイル名に割当表の場面情報を埋め込む: mixNNNN_車N台_<警告>_<危険層>_<移動>.wav
  → エクスプローラーの名前順・検索で「場面別の抜き聞き」ができる
出力: out/listen_v11_core_train/（約2.3GB、10秒×4,800本=13.3時間ぶん）
使い方: python scripts/_make_listen_core_all.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FOA = ROOT / "out" / "dataset_outdoor_siren_v11" / "foa"
PLAN = ROOT / "out" / "dataset_outdoor_siren_v11" / "plan" / "assignment_core.csv"
OUT = ROOT / "out" / "listen_v11_core_train"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JP = {"siren": "サイレン", "horn": "クラクション", "backup_beep": "バック音",
      "bike_bell": "ベル", "crossing": "踏切",
      "critical": "重大", "caution": "注意", "safe": "安全", "na": "－",
      "static": "静止", "walk": "歩行",
      "residential": "住宅", "daily": "生活", "arterial": "幹線"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(PLAN, encoding="utf-8"))
            if r["split"] == "fold1"]
    assert len(rows) == 4800
    done = 0
    for r in rows:
        stem = r["clip_id"]
        warn = "警告なし"
        if int(r["n_warnings"]) >= 1:
            warn = JP[r["w1_class"]]
            if int(r["n_warnings"]) == 2:
                warn += "＋" + JP[r["w2_class"]]
        name = (f"{stem.split('_')[-1]}_{JP[r['scene_type']]}_車{r['n_car']}台_"
                f"{warn}_{JP[r['danger_tier']]}_{JP[r['motion']]}.wav")
        dst = OUT / name
        if dst.exists():
            done += 1
            continue
        x = np.asarray(sf.read(FOA / f"{stem}.flac")[0], np.float64)
        w = x[:, 0]
        peak = float(np.max(np.abs(w)))
        if peak > 0:
            w = w / peak * 0.85
        sf.write(dst, w.astype(np.float32), 24000, subtype="PCM_16")
        done += 1
        if done % 400 == 0:
            print(f"{done}/4800", flush=True)
    print(f"done: {done} -> {OUT}")


if __name__ == "__main__":
    main()
