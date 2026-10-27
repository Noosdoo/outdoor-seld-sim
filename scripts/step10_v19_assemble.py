# -*- coding: utf-8 -*-
"""v19 の学習用データ束を組む（2026-09-08）: 土台（v18 = v16 の音声＋列車を背景にしたラベル）＋ 新場面 1 つ。
出力 out/dataset_outdoor_siren_v19_<scene>/
  plan/assignment_v19_<scene>_all.csv  = 土台の plan（18,000 行）＋ 新場面の plan（1,500 行）→ HEIGHT_TABLE に使う
  foa/            土台の音声（v16）と新場面の音声（v19 生成）へのリンク
  metadata/       土台のラベル（v18）と新場面のラベルへのリンク
  metadata_dist/  同上（距離付き）
土台・v16・v19 の元には書かない。リンクが作れない環境（Windows）はコピーにする。
使い方: python scripts/step10_v19_assemble.py --scene fast [--base out/dataset_outdoor_siren_v18] [--audio out/dataset_outdoor_siren_v16]
        [--gen out/dataset_outdoor_siren_v19] [--dst out/dataset_outdoor_siren_v19_<scene>]
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


def _arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def _link(src: Path, dst: Path, copy_mode: list) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy_mode[0]:
        shutil.copyfile(src, dst)
        return
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        copy_mode[0] = True
        shutil.copyfile(src, dst)


def main() -> int:
    scene = _arg("--scene")                       # 1 場面、または "fast,turn,platoon" のように複数（v20 の合成束・2026-09-09）
    scenes = [s for s in scene.split(",") if s]
    assert all(s in ("fast", "static", "turn", "platoon") for s in scenes), scenes
    base = ROOT / _arg("--base", "out/dataset_outdoor_siren_v18")
    audio = ROOT / _arg("--audio", "out/dataset_outdoor_siren_v16")
    gen = ROOT / _arg("--gen", "out/dataset_outdoor_siren_v19")
    name = _arg("--name", f"v19_{scene}" if len(scenes) == 1 else "v20")     # plan の名前 assignment_<name>_all.csv
    dst = ROOT / _arg("--dst", f"out/dataset_outdoor_siren_{name}")
    assert dst.name.startswith("dataset_outdoor_siren_v19_") or dst.name.startswith("dataset_outdoor_siren_v20"), dst
    plan_base = next(p for p in (base / "plan/assignment_v18.csv", base / "plan/assignment_v16.csv") if p.exists())
    rows_b = list(csv.DictReader(open(plan_base, encoding="utf-8")))
    rows_g = []
    for s in scenes:
        rows_g += list(csv.DictReader(open(gen / "plan" / f"assignment_v19_{s}.csv", encoding="utf-8")))
    cols = list(rows_b[0].keys())
    assert set(cols) == set(rows_g[0].keys()), (cols, list(rows_g[0].keys()))
    ids = [r["clip_id"] for r in rows_b] + [r["clip_id"] for r in rows_g]
    assert len(set(ids)) == len(ids), "clip_id が重複"
    (dst / "plan").mkdir(parents=True, exist_ok=True)
    with open(dst / "plan" / f"assignment_{name}_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_b)
        w.writerows(rows_g)
    copy_mode = [False]
    missing = []
    for sub, sb, sg in (("foa", audio / "foa", gen / "foa"), ("metadata", base / "metadata", gen / "metadata"),
                        ("metadata_dist", base / "metadata_dist", gen / "metadata_dist")):
        (dst / sub).mkdir(exist_ok=True)
        ext = ".flac" if sub == "foa" else ".csv"
        for r in rows_b:
            p = sb / f"{r['clip_id']}{ext}"
            if not p.exists():
                missing.append(str(p))
                continue
            _link(p, dst / sub / p.name, copy_mode)
        for r in rows_g:
            p = sg / f"{r['clip_id']}{ext}"
            if not p.exists():
                missing.append(str(p))
                continue
            _link(p, dst / sub / p.name, copy_mode)
    n = {sub: len(list((dst / sub).iterdir())) for sub in ("foa", "metadata", "metadata_dist")}
    L = [f"# {name} = 土台 {base.name}（音声 {audio.name}）＋ 新場面 {' + '.join('v19' + s for s in scenes)}", "",
         f"- plan: 土台 {len(rows_b):,} 行 ＋ 新場面 {len(rows_g):,} 行（{len(scenes)} 場面 × 1,500）= {len(rows_b) + len(rows_g):,} 行",
         f"- foa {n['foa']:,} / metadata {n['metadata']:,} / metadata_dist {n['metadata_dist']:,}（{'コピー' if copy_mode[0] else 'リンク'}）",
         f"- 見つからないファイル: {len(missing)}" + (f"（例 {missing[:3]}）" if missing else "")]
    (dst / f"README_{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
