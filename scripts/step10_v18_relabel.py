# -*- coding: utf-8 -*-
"""v18 = v16 の音声そのまま・ラベルだけ変更: 列車の通過音（v12train シナリオの crossing）を背景（ラベル無し）にする（2026-09-08・場面点検 §5 案 A・D14）。

音声（foa/）と場面（work/）は v16 へのシンボリックリンク。plan はコピー。metadata/ と metadata_dist/ は v16 から複写し、
scenario == v12train のクリップだけ crossing（cls=5）の行を落とす（そのクリップは負例になる。警報機の crossing は v11core にしか無いので影響なし）。
出力: out/dataset_outdoor_siren_v18/{foa->, work->, plan/, metadata/, metadata_dist/} と README_v18.md（件数の証跡）。v16 には書かない。

使い方: python scripts/step10_v18_relabel.py [--src out/dataset_outdoor_siren_v16] [--dst out/dataset_outdoor_siren_v18]
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CROSSING = 5


def _arg(name, default):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def main() -> int:
    src = Path(_arg("--src", str(ROOT / "out/dataset_outdoor_siren_v16")))
    dst = Path(_arg("--dst", str(ROOT / "out/dataset_outdoor_siren_v18")))
    assert src.name != dst.name and dst.name.endswith("v18"), dst
    plan_src = next(p for p in (src / "plan/assignment_v16.csv", src / "plan/assignment_v12ext.csv", src / "plan/assignment_core.csv") if p.exists())
    rows = list(csv.DictReader(open(plan_src, encoding="utf-8")))
    train_clips = {r["clip_id"] for r in rows if r.get("scenario") == "v12train"}
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("foa", "work", "masks"):
        link = dst / sub
        if (src / sub).exists() and not link.exists():
            try:
                os.symlink(os.path.relpath(src / sub, dst), link, target_is_directory=True)
            except OSError as e:   # Windows は権限が無いとリンクを作れない → ラベルだけ作る（採点には foa/work は不要）
                print(f"リンク省略 {sub}: {e}")
    (dst / "plan").mkdir(exist_ok=True)
    shutil.copyfile(plan_src, dst / "plan/assignment_v18.csv")
    shutil.copyfile(plan_src, dst / "plan/assignment_v16.csv")   # 高さ表（HEIGHT_TABLE）は v16 と同じ内容
    stats = {}
    for sub in ("metadata", "metadata_dist"):
        sdir, ddir = src / sub, dst / sub
        if not sdir.exists():
            print("skip（無い）:", sdir); continue
        ddir.mkdir(exist_ok=True)
        n_files = n_dropped = n_emptied = 0
        for f in sorted(sdir.glob("*.csv")):
            clip = f.stem
            lines = f.read_text(encoding="utf-8").splitlines()
            if clip in train_clips:
                keep = [l for l in lines if l.strip() and int(l.split(",")[1]) != CROSSING]
                n_dropped += len(lines) - len(keep)
                if not keep:
                    n_emptied += 1
                (ddir / f.name).write_text("".join(l + "\n" for l in keep), encoding="utf-8")
            else:
                shutil.copyfile(f, ddir / f.name)
            n_files += 1
        stats[sub] = (n_files, n_dropped, n_emptied)
        print(f"{sub}: {n_files} files, 落とした行 {n_dropped}, 空になったクリップ {n_emptied}")
    L = [f"# v18 = v16 のラベルだけ変更（列車の通過音 → 背景）", "",
         f"- 元: {src.name}（音声・場面はリンク）。plan は v16 と同一（{len(rows)} 行）。列車クリップ（scenario=v12train）: {len(train_clips)} 本",
         *[f"- {k}: {v[0]} files / crossing 行を {v[1]} 行落とした / 空になったクリップ {v[2]}" for k, v in stats.items()],
         "- 警報機の crossing（v11core・静止）はそのまま。fold2 の val も同じ規則（列車クリップは負例として採点される）"]
    (dst / "README_v18.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
