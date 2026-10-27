# -*- coding: utf-8 -*-
"""scripts/_atomic_write.py の試験（2026-10-27 Opus・job 9342 の競合の直し）。架空のファイルだけ。
  A1 48 プロセスが同時に同じ中身を書く（同じ合図で一斉に始める）→ 全部止まらない・中身は求める中身と同じ・一時ファイルが残らない
  A2 既にある写しの中身が違う → ContentMismatch で止める（取り違えの検出は今までどおり）
  A3 既にある写しの中身が同じ → 書き換えずに通る（戻り False）
  A4 _run_v2final_gen.write_gen_versions が写しをこの書き方で書く（コードの文字で確かめる・描画のコードの sha は変わる）
"""
from __future__ import annotations

import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _worker(args):
    path, data, start = args
    sys.path.insert(0, str(HERE))
    import _atomic_write as W
    while time.time() < start:                                            # 一斉に始める
        pass
    try:
        W.write_once_atomic(path, data)
        return "ok"
    except Exception as ex:                                               # noqa: BLE001
        return f"{type(ex).__name__}: {ex}"


def main() -> int:
    import _atomic_write as W
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="atomic_"))
    try:
        data = ("{" + ",".join(f'"k{i}": {i}' for i in range(20000)) + "}").encode("utf-8")   # 書きかけが起きやすいよう大きめ（約 0.2 MB）
        target = tmp / "params_snapshots" / "params_0123456789abcdef.json"
        start = time.time() + 2.0
        with mp.Pool(48) as pool:
            res = pool.map(_worker, [(str(target), data, start)] * 48)
        bad = [r for r in res if r != "ok"]
        left = list(target.parent.glob(".*.tmp"))
        ok.append((f"A1 48 プロセスが同時に書く → 全部止まらない（止まった {len(bad)} 本）・中身が同じ・一時ファイルが残らない（{len(left)}）",
                   not bad and target.read_bytes() == data and not left))
        other = tmp / "x" / "params_ffff.json"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"{}")
        try:
            W.write_once_atomic(other, data); mism = False
        except W.ContentMismatch:
            mism = True
        ok.append(("A2 既にある写しの中身が違う → ContentMismatch で止める", mism and other.read_bytes() == b"{}"))
        ok.append(("A3 既にある写しの中身が同じ → 書き換えずに通る（戻り False）", W.write_once_atomic(target, data) is False))
        src = (HERE / "_run_v2final_gen.py").read_text(encoding="utf-8")
        ok.append(("A4 write_gen_versions が写しを write_once_atomic で書く（直接の write_bytes は無い）",
                   "write_once_atomic(snap, pb)" in src and "snap.write_bytes(pb)" not in src))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
