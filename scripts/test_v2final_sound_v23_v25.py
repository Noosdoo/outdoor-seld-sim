# -*- coding: utf-8 -*-
"""合成の音の直し（2026-10-12・v23 警告音のドライ音 #60／v24 音の大きさ #61／v25 大型車 #52・ベルつき走行のベルの時刻 #53）の自己検査をまとめて回す。
  S1 outdoor_seld.alert_sounds_v23（押しの列・バック音の頭・ベルの高さの表）
  S2 outdoor_seld.level_v24（曲線・向きの差・LAFmax）
  S3 outdoor_seld.engine_heavy_v25（大型車のドライ音）
  S4 scripts/step11_v23_render.py selftest（plan の本で押しの列・先頭の音・基音・1 打・ベルの高さ・乱数の消費）
  S5 scripts/step11_v24_render.py selftest（曲線どおりの l1m・発進・バイク・サイレン・最大値合わせ・乱数の消費）※ md/design/level_match_v24.json が要る
  S6 scripts/step11_v25_render.py selftest（大型車の場面とドライ音・ベルの時刻 22/500・乱数の消費）※ 10 分ほど
  S7 scripts/step10_v25_plan.py --dry-run（大型車の plan の検査・D25 が不採用（2026-10-12 18:17）の間は飛ばす）
  S8 outdoor_seld.siren_v26・outdoor_seld.avas_v26（サイレンの形・接近通報音）
  S9 scripts/step11_v26_render.py selftest（v26 のサイレンが v24 を通る・EV の下地・乱数の消費）
  S10 scripts/test_sound_reference_v26.py（音の検査・出典の値と照らす・段階 1＋2）※ 10 分ほど
使い方: python scripts/test_v2final_sound_v23_v25.py [--quick]（--quick は S6・S7・S10 を飛ばす）
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = sys.executable
ENV = {"PYTHONPATH": str(ROOT / "src") + ";" + str(ROOT / "scripts"), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run(name: str, args: list, ok_mark: str) -> bool:
    import os
    r = subprocess.run([PY] + args, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **ENV})
    out = r.stdout + r.stderr
    good = r.returncode == 0 and ok_mark in out
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] if out.strip() else [""]
    print(("PASS " if good else "FAIL ") + name + f"（{tail[0][:120]}）")
    if not good:
        print(out[-1500:])
    return good


def main() -> int:
    quick = "--quick" in sys.argv[1:]
    res = [run("S1 alert_sounds_v23", ["-m", "outdoor_seld.alert_sounds_v23"], "ALL PASS"),
           run("S2 level_v24", ["-m", "outdoor_seld.level_v24"], "ALL PASS"),
           run("S3 engine_heavy_v25", ["-m", "outdoor_seld.engine_heavy_v25"], "ALL PASS"),
           run("S4 step11_v23_render selftest", [str(ROOT / "scripts/step11_v23_render.py"), "selftest"], "ALL PASS"),
           run("S5 step11_v24_render selftest", [str(ROOT / "scripts/step11_v24_render.py"), "selftest"], "ALL PASS")]
    if not quick:
        res.append(run("S6 step11_v25_render selftest", [str(ROOT / "scripts/step11_v25_render.py"), "selftest"], "ALL PASS"))
        import json as _json
        if (_json.loads((ROOT / "md/design/v2final_params.json").read_text(encoding="utf-8")).get("D25_HEAVY") or {}).get("adopt"):
            res.append(run("S7 step10_v25_plan --dry-run", [str(ROOT / "scripts/step10_v25_plan.py"), "--dry-run"], "dry-run"))
        else:
            print("S7 step10_v25_plan --dry-run: 飛ばす（D25 大型車は 2026-10-12 18:17 に不採用）")
    res.append(run("S8a siren_v26", ["-m", "outdoor_seld.siren_v26"], "ALL PASS"))
    res.append(run("S8b avas_v26", ["-m", "outdoor_seld.avas_v26"], "ALL PASS"))
    res.append(run("S9 step11_v26_render selftest", [str(ROOT / "scripts/step11_v26_render.py"), "selftest"], "ALL PASS"))
    if not quick:
        res.append(run("S10 test_sound_reference_v26", [str(ROOT / "scripts/test_sound_reference_v26.py")], "ALL PASS"))
    print(f"{sum(res)}/{len(res)} {'ALL PASS' if all(res) else 'NG'}")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
