"""Step 4a: 生成物を PSELDNets のデータセット構造へ配置し、index CSV のパス区切りを直す。

配置:
  PSELDNets/datasets/outdoor1clip/foa/fold0_room0_mix001.flac   (直接音のみ版FOA)
  PSELDNets/datasets/outdoor1clip/metadata/fold0_room0_mix001.csv
  PSELDNets/datasets/cls_indices_train.tsv

その後 PSELDNets 側で:
  .venv/Scripts/python.exe src/preproc.py dataset=outdoor1clip
を実行し、本スクリプトを --fix-index で再実行して index CSV の `\\` を `/` に正規化する
（`src/data/data.py` が `path.split('/')[-3]` を使う Windows 非互換の回避。リポジトリ無改変）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "out" / "clip"
PSELD = ROOT.parent / "PSELDNet" / "PSELDNets"
DS = PSELD / "datasets"


def place():
    scene = json.loads((CLIP / "scene_config.json").read_text())
    name = scene["clip_name"]
    (DS / "outdoor1clip" / "foa").mkdir(parents=True, exist_ok=True)
    (DS / "outdoor1clip" / "metadata").mkdir(parents=True, exist_ok=True)

    pairs = [
        (CLIP / "foa_direct_24k.flac", DS / "outdoor1clip" / "foa" / f"{name}.flac"),
        (CLIP / f"{name}.csv", DS / "outdoor1clip" / "metadata" / f"{name}.csv"),
        (CLIP / "cls_indices_train.tsv", DS / "cls_indices_train.tsv"),
    ]
    for src, dst in pairs:
        shutil.copyfile(src, dst)
        print(f"placed {dst.relative_to(PSELD)}")


def fix_index():
    idx_dir = PSELD / "_hdf5" / "data" / "24000fs" / "wav" / "dev"
    fixed = 0
    for csv in idx_dir.glob("outdoor1clip_*.csv"):
        text = csv.read_text()
        new = text.replace("\\", "/")
        if new != text:
            csv.write_text(new)
            fixed += 1
        print(f"index {csv.name}: {'fixed' if new != text else 'already ok'}")
        for line in new.strip().split("\n"):
            print("  " + line)
    if fixed == 0:
        print("no files needed fixing")


if __name__ == "__main__":
    if "--fix-index" in sys.argv:
        fix_index()
    else:
        place()
