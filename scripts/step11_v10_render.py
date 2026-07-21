# -*- coding: utf-8 -*-
"""step11_v10_render.py — v10 レンダラ（設計= out/v10_design_2026-07-21.md）。

step11_v9_render.py の物理・音源合成ロジックをモジュールとしてimportし、
モジュール変数を上書きする方式で再利用する（1000行規模の音源合成コードを
複製しない。関数はPython実行時にモジュールのグローバルを参照するため、
import後に属性を上書きすれば以後の呼び出し全てに反映される）。

v9.1からの差分はこの3つのみ:
  1. 規模: 割当表がstep10_v10_plan.py製（core3600=train2400/val600/test600、
     評価専用枠はv9.1本数のまま）
  2. 舞台=日本の適合性監査（out/japan_stage_audit_2026-07-21.md）: horn/backup_beepの
     dB法規レンジを日本の一次資料に、車速レンジを生活道路30km/h（2026-09-01施行）に修正
  3. 出力=独立フォルダ out/dataset_outdoor_siren_v10/（v9.1/v9.2/v9.3とは完全に別データ）

それ以外（クラス構成・絶対較正・歩行マイク・危険3層・音源の音色そのもの）はv9.1と同一。

使い方（dynamic-sound venv の python で）:
  python scripts/step11_v10_render.py precheck
  python scripts/step11_v10_render.py gen 0-9 --set core
  python scripts/step11_v10_render.py inspect
  python scripts/step11_v10_render.py pack
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v9_render as m9  # noqa: E402

# --- 1. 規模: v10専用の割当表を読む -----------------------------------------
m9.DS_NAME = "outdoor_siren_v10"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = ROOT / "out" / "dataset_outdoor_siren_v10" / "plan"
m9.WORK = m9.DS / "work"

# v9.1の音源修正（踏切v2・日本wail・ベル引き打ち）はv10でも当然使う（フラグ強制ON）
m9.V91 = True
m9.V92 = False
m9.V93 = False

# --- 2. 舞台=日本の適合性監査（out/japan_stage_audit_2026-07-21.md）の反映 ---
# horn: 旧ECE R28(93-112dB@7m) → 道路運送車両の保安基準第43条/告示219号別添74(87-112dB@7m)
# backup_beep: 旧OSHA(87-112dB@1.2m) → 保安基準第145条の6・UN R165準拠(60-92dB@1m)
m9.LAW["horn"] = (87.0, 112.0, 7.0)
m9.LAW["backup_beep"] = (60.0, 92.0, 1.0)
# car_drive: 生活道路の法定速度引下げ（60→30km/h、2026-09-01施行）に対応。
# 旧(5.0,15.0)=18-54km/hは新法下で上限が違法速度。新(3.0,10.0)=11-36km/hは
# 慎重運転〜軽微な超過（現実の違反ケースの余地は残す）
m9.SPEED["car_drive"] = (3.0, 10.0)


if __name__ == "__main__":
    # step11_v9_render.py の __main__ ディスパッチをそのまま再利用する
    # （sys.argv はこのプロセスのものなので --v91 等のフラグ文字列は不要、
    # 上の直接代入が優先される）。
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    which = "core"
    if "--set" in sys.argv:
        which = sys.argv[sys.argv.index("--set") + 1]
    if mode == "precheck":
        sys.exit(m9.precheck())
    elif mode == "gen":
        rows = m9.load_plan(which)
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        for i in range(int(a), int(b) + 1):
            m9.generate_clip(rows[i])
    elif mode == "inspect":
        sys.exit(1 if m9.inspect_all() else 0)
    elif mode == "pack":
        m9.pack()
    elif mode == "pack_scn2":
        m9.pack_scn2()
    elif mode == "pack_v10a":
        m9.pack_v10a()
    else:
        print(f"unknown mode: {mode}")
        sys.exit(1)
