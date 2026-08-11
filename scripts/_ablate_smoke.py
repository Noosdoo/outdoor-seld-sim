# -*- coding: utf-8 -*-
"""物理ablationスイッチのスモークテスト（第5版・2026-08-11）。

検証内容:
  1) ABLATE未設定の出力が既存の確定評価セット(conf)の同名クリップとビット一致
     （=スイッチ実装が既定経路を一切変えていない回帰確認）
  2) 4条件それぞれで生成が通り、音声がfullとも相互とも異なる
  3) ラベル規約: no_airabs/no_1r/no_ground では非ゲートクラス(サイレン)の行が不変、
     no_doppler では一定遅延規約により行が移動しうる（差分を報告）
使い方: python scripts/_ablate_smoke.py <出力ベースdir>
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["", "no_doppler", "no_airabs", "no_1r", "no_ground"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def rows_of(p: Path, cls=None):
    out = []
    for line in open(p, encoding="utf-8"):
        q = line.strip().split(",")
        if len(q) == 5 and (cls is None or int(q[1]) == cls):
            out.append(tuple(int(v) for v in q))
    return out


def child(base: Path):
    mode = os.environ.get("ABLATE", "")
    tag = mode or "full"
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import step11_v12_conf_render as cf
    ds = base / tag
    cf.m9.DS = ds
    cf.m9.WORK = ds / "work"
    for sub in ("foa", "metadata", "masks", "work"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    row = cf.load_plan_v12conf()[0]
    cf.m12.generate_clip_v12(row)
    clip = row["clip_id"]
    print("SMOKE_JSON:" + json.dumps({
        "mode": tag,
        "foa": sha(ds / "foa" / f"{clip}.flac"),
        "meta": str(ds / "metadata" / f"{clip}.csv"),
        "clip": clip}))


def main():
    base = Path(sys.argv[-1]).resolve()
    if "--child" in sys.argv:
        child(base)
        return
    base.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode in MODES:
        env = dict(os.environ, ABLATE=mode, PYTHONIOENCODING="utf-8")
        out = subprocess.run(
            [sys.executable, __file__, "--child", str(base)],
            env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if out.returncode != 0:
            print(out.stdout[-2000:])
            print(out.stderr[-2000:])
            raise SystemExit(f"child failed for mode={mode!r}")
        line = [l for l in out.stdout.splitlines()
                if l.startswith("SMOKE_JSON:")][0]
        res[mode or "full"] = json.loads(line[len("SMOKE_JSON:"):])

    clip = res["full"]["clip"]
    # 1) 回帰: full が確定評価セットの実物とビット一致
    conf = ROOT / "out" / "dataset_outdoor_siren_v12_conf" / "foa" / f"{clip}.flac"
    ok_regress = sha(conf) == res["full"]["foa"]
    print(f"[1] 既定経路の回帰(bit一致 vs conf実物): "
          f"{'PASS' if ok_regress else 'FAIL'} ({res['full']['foa']})")
    assert ok_regress

    # 2) 音声が全条件で異なる
    hashes = {m: r["foa"] for m, r in res.items()}
    print("[2] foa sha256(先頭16):")
    for m, h in hashes.items():
        print(f"    {m:>10}: {h}")
    assert len(set(hashes.values())) == len(hashes), "条件間で音声が同一のものがある"

    # 3) ラベル規約
    full_siren = rows_of(Path(res["full"]["meta"]), cls=0)
    for m in ("no_airabs", "no_1r", "no_ground"):
        same = rows_of(Path(res[m]["meta"]), cls=0) == full_siren
        print(f"[3] サイレン行不変({m}): {'PASS' if same else 'FAIL'}")
        assert same
    nd = rows_of(Path(res["no_doppler"]["meta"]), cls=0)
    moved = len(set(nd) ^ set(full_siren))
    f0 = full_siren[0][0] if full_siren else None
    n0 = nd[0][0] if nd else None
    print(f"[3] no_dopplerのサイレン行: full {len(full_siren)}行(先頭fr{f0}) → "
          f"nd {len(nd)}行(先頭fr{n0})・対称差{moved}行"
          "（一定遅延規約による移動。0でも可=遅延差が0.1s未満のケース）")
    print("ALL SMOKE CHECKS DONE")


if __name__ == "__main__":
    main()
