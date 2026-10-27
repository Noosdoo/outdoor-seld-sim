# -*- coding: utf-8 -*-
"""同じ中身を多くのプロセスが同時に書くファイルの、書きかけを読ませない書き方（2026-10-27 Opus・Fable 10/26 21:5x の OK）。

job 9342（背景の形の比べの val の描画）で、48 シャードが同時に _run_v2final_gen.write_gen_versions の params の写しを
`write_bytes` で直接書き、書きかけを読んだ 2 シャードが「写しの中身が今の params と違う」で止まった。
ここでは、無いときだけ一時ファイル（プロセス番号つき）に書いてから os.replace で置き換える（置き換えは不可分＝読む側には
「無い」か「全部ある」しか見えない）。そのあと、置き場の中身が求める中身と同じかを必ず確かめ、違えば止める（取り違え・書き換えの検出は今までどおり）。
試験: scripts/test_atomic_write.py（48 プロセスが同時に書く）
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


class ContentMismatch(RuntimeError):
    """名前が合って中身が違う（取り違えか書き換え）。"""


def write_once_atomic(path, data: bytes) -> bool:
    """path が無ければ data を一時ファイル → 置き換えで書く。あれば中身を確かめる。戻り = この呼び出しで置き換えたか。
    中身が data と違えば ContentMismatch。"""
    path = Path(path)
    wrote = False
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        try:
            os.replace(tmp, path)
            wrote = True
        except OSError:                                                   # ほかのプロセスが先に置き換えた（Windows で開かれている等）
            tmp.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise ContentMismatch(f"{path.name} の中身が求める中身と違う（取り違えか書き換え）")
    return wrote
