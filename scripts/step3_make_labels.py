"""Step 3: DCASE 形式 SELD ラベル生成。

- 音側（Step 2）と同一の geometry.py で「放射時刻補正済みの見かけDOA」を
  100 ms フレームごとに計算し、5列 CSV [frame,class,track,az,el] を出力
- PSELDNets 用クラス辞書 cls_indices_train.tsv（SELD-Data-Generator の屋外10クラス
  と同一、Siren=class 4）も生成

出力: out/clip/fold0_room0_mix001.csv, out/clip/cls_indices_train.tsv
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out" / "clip"

from outdoor_seld.geometry import sound_speed  # noqa: E402
from outdoor_seld.labels import frame_label_rows, write_dcase_csv  # noqa: E402
from outdoor_seld.scene import SceneConfig  # noqa: E402

CLS_SRC = (ROOT.parent / "SELD-Data-Generator" / "database"
           / "seld_FSD50K_5_ov1_train" / "cls_indices.tsv")


def main():
    scene = SceneConfig(**json.loads((OUT / "scene_config.json").read_text()))
    c = sound_speed(scene.temperature_c)

    rows, debug = frame_label_rows(
        scene.waypoints_direct(), np.array(scene.mic_pos),
        clip_len_sec=scene.clip_len_sec, class_idx=scene.class_idx,
        track_idx=0, source_active_until=scene.clip_len_sec, c=c)
    csv_path = OUT / f"{scene.clip_name}.csv"
    write_dcase_csv(csv_path, rows)

    # クラス辞書（tsv 3列目=クラス名を行順で ID 化する PSELDNets 仕様）
    tsv_out = OUT / "cls_indices_train.tsv"
    shutil.copyfile(CLS_SRC, tsv_out)
    names = [l.split("\t")[2] for l in tsv_out.read_text().strip().split("\n")]
    assert names[scene.class_idx] == scene.class_name, \
        f"class_idx mismatch: {names[scene.class_idx]} != {scene.class_name}"

    azs = [r[3] for r in rows]
    els = [r[4] for r in rows]
    print(f"wrote {csv_path.name}: {len(rows)} frames "
          f"(frame {rows[0][0]}..{rows[-1][0]})")
    print(f"  class={scene.class_idx} ({scene.class_name}), track=0")
    print(f"  az {azs[0]} -> {azs[-1]} deg (CPA az={azs[len(azs)//2]}), "
          f"el {min(els)}..{max(els)} deg")
    print(f"wrote {tsv_out.name}: {len(names)} classes: {', '.join(names[:5])}, ...")

    # 可視化・照合用に連続値もダンプ
    np.savez(OUT / "label_debug.npz", **debug)


if __name__ == "__main__":
    main()
