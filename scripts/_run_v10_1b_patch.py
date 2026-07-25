# -*- coding: utf-8 -*-
"""v10.1bパッチ適用ドライバ。v10.1パッチ（消防車サイレン）の完了後に実行する。

対象 = backup_beep を含む全クリップ。クラス割当はplan CSVの w1_class/w2_class
列で確定している（音量抽選より前の工程で、V10_1/V10_1Bの影響を受けない）ため、
ディスク状態にもシード再計算にも依存せず**plan行だけで対象集合が決まる**
（=中断・再開しても常に同じ集合、べき等）。
対象は無条件にv10.1bコード（V10_1+V10_1B、fire割当保存＋backup混合レンジ）で
再生成し、最後にinspect_all()で全体の整合性を確認する。
"""
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1b_render as m10_1b  # noqa: E402
m9 = m10_1b.m9

SETS = ["core", "scenario", "probe", "scenario2", "v10a"]


def find_backup_clips():
    targets = []
    for which in SETS:
        rows = m9.load_plan(which)
        for row in rows:
            if row.get("w1_class") == "backup_beep" or row.get("w2_class") == "backup_beep":
                targets.append(row)
    return targets


def main():
    assert m9.V10_1 and m9.V10_1B, "V10_1/V10_1Bフラグ継承が壊れている"
    t0 = time.time()
    targets = find_backup_clips()
    print(f"=== 対象クリップ(backup_beep含有): {len(targets)}本 ===", flush=True)
    for i, row in enumerate(targets):
        m9.generate_clip(row)
        if (i + 1) % 50 == 0 or i + 1 == len(targets):
            el = time.time() - t0
            print(f"{i + 1}/{len(targets)}  {el:.0f}s elapsed", flush=True)

    print(f"\n=== 再生成完了: {len(targets)}本, {time.time() - t0:.0f}s ===", flush=True)
    print("=== running inspect_all() (v10全体の再検品) ===", flush=True)
    fail = m9.inspect_all()
    print(f"\n=== v10.1b ALL DONE (inspect fail={fail}) total {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    main()
