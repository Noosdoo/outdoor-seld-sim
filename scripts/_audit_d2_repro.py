# -*- coding: utf-8 -*-
"""データ精査D2: 再現性spot check（1クリップ再生成→ビット一致確認）。

安全プロトコル:
  1. 対象クリップの成果物(foa/metadata/masks/scene.json)をバックアップ＆SHA256記録
  2. generate_clip(row) をその1本だけ再実行（v11は決定論を謳う）
  3. 再生成後のSHA256と比較。一致→PASS（バックアップ破棄）
     不一致→**即バックアップから復元**し、差分を記録（環境で再現不可という重要知見）
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLIP = sys.argv[1] if len(sys.argv) > 1 else "fold2_room1_mix0001"
DS = ROOT / "out" / "dataset_outdoor_siren_v11"
BK = ROOT / "out" / "audit_d2_backup"


def targets():
    return [DS / "foa" / f"{CLIP}.flac",
            DS / "metadata" / f"{CLIP}.csv",
            DS / "masks" / f"{CLIP}.csv",
            DS / "work" / CLIP / "scene.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    files = targets()
    assert all(p.exists() for p in files), files
    BK.mkdir(parents=True, exist_ok=True)
    before = {}
    for p in files:
        # ⚠️ metadataとmasksは同名CSVなので親ディレクトリ名を前置して衝突を防ぐ
        # （2026-08-05の初回実行でこの衝突によりmetadata1本を一時破損→git/serverから
        #   完全復旧済み。教訓として名前空間を分離）
        bk_name = f"{p.parent.name}__{p.name}"
        shutil.copy2(p, BK / bk_name)
        before[bk_name] = sha(p)
        print(f"backup {bk_name}  {before[bk_name][:16]}…")

    import step11_v11_render as m11
    m9 = m11.m9
    assert m9.DS_NAME == "outdoor_siren_v11"
    rows = [r for r in m9.load_plan("core") if r["clip_id"] == CLIP]
    assert len(rows) == 1, f"plan row for {CLIP} not found"
    print(f"re-generating {CLIP} …", flush=True)
    m9.generate_clip(rows[0])

    ok = True
    for p in files:
        bk_name = f"{p.parent.name}__{p.name}"
        after = sha(p)
        same = after == before[bk_name]
        ok &= same
        print(f"{'PASS' if same else 'DIFF'}  {bk_name}  {after[:16]}…")
    if not ok:
        print("⚠️ バイト不一致 → バックアップから復元します")
        print("   （注: テキスト系はCRLF/LFの差だけの可能性あり。内容一致は別途"
              "改行正規化して比較すること＝2026-08-05のD2はそれで全一致だった）")
        for p in files:
            bk_name = f"{p.parent.name}__{p.name}"
            shutil.copy2(BK / bk_name, p)
            assert sha(p) == before[bk_name]
        print("復元完了（データセットは元のまま）")
    else:
        print("D2 PASS: 1クリップ完全ビット一致（決定論の実証）")


if __name__ == "__main__":
    main()
