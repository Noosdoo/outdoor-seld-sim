# -*- coding: utf-8 -*-
"""v10.1パッチ適用ドライバ。v10の全量生成が完了してから実行する。

手順: 5つのplanセット（core/scenario/probe/scenario2/v10a）全行について、
**シードから「v10.1なしなら本来wail型だったか」を再計算**して対象を決める
（ディスク上のscene.jsonを見て判定しない）。理由: 途中で中断したパッチ実行が
一部のクリップを既にfire型へ書き換えている可能性があり、その場合scene.json上の
siren_typeは既に"fire"になっていて「まだ処理していない」判定に使えないため。
シードは不変なので、V10_1フラグをOFFにした`_warn_params`の結果と比較すれば
中断・再開を何度挟んでも常に正しい対象集合が求まる（=べき等）。
対象は無条件に（現在の状態を問わず）v10.1コードで再生成する。
最後にinspect_all()を再実行して全体の整合性を確認する。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1_render as m10_1  # noqa: E402
m9 = m10_1.m9

SETS = ["core", "scenario", "probe", "scenario2", "v10a"]


def find_wail_clips():
    """行のseedから、v10（V10_1なし）の時点で本来wail型だった行を再計算する。
    w1/w2どちらかがsirenの行だけを候補にする（他の行はsiren自体を持たない）。
    """
    targets = []
    for which in SETS:
        rows = m9.load_plan(which)
        for row in rows:
            if row.get("w1_class") != "siren" and row.get("w2_class") != "siren":
                continue
            if trace_is_wail(row):
                targets.append(row)
    return targets


def trace_is_wail(row: dict) -> bool:
    """V10_1をOFFにした状態で_warn_paramsを呼び、そのsirenがwail型だったか判定する。
    mic/car/w1/w2の乱数消費順を正確に再現するため、sample_scene_v9本体と同じ
    手順をV10_1=Falseに一時的に切り替えて実行する。"""
    saved = m9.V10_1
    try:
        m9.V10_1 = False
        scene = m9.sample_scene_v9(row)
    finally:
        m9.V10_1 = saved
    for src in scene["sources"]:
        if src["class"] == "siren" and src["params"].get("siren_type") == "wail":
            return True
    return False


def main():
    t0 = time.time()
    targets = find_wail_clips()
    print(f"=== 対象クリップ(旧wail型): {len(targets)}本 ===", flush=True)
    for i, row in enumerate(targets):
        m9.generate_clip(row)
        if (i + 1) % 50 == 0 or i + 1 == len(targets):
            el = time.time() - t0
            print(f"{i + 1}/{len(targets)}  {el:.0f}s elapsed", flush=True)

    print(f"\n=== 再生成完了: {len(targets)}本, {time.time() - t0:.0f}s ===", flush=True)
    print("=== running inspect_all() (v10全体の再検品) ===", flush=True)
    fail = m9.inspect_all()
    print(f"\n=== v10.1 ALL DONE (inspect fail={fail}) total {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    main()
