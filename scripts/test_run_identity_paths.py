# -*- coding: utf-8 -*-
"""来歴の同一性がジョブ固有の写しの置き場所に依らないことの試験（2026-09-09・独立監査 R1/A3・A7 への対応）。

問題: 推論台本と来歴モジュールをジョブごとの写し（_job<ID>/…、または *_job<ID>.py）で実行すると、run_meta.json の
code_sha256 の鍵がジョブ固有のパスになり、別ジョブが同じ出力先を引き継いだ（または補助ジョブが仕上げた）ときに
「run identity changed」で止まった。鍵は論理名（写し・置き場所を消した名前）にする。

使い方: python scripts/test_run_identity_paths.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("prediction_integrity", ROOT / "server_sde/_prediction_integrity.py")
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")


def make_meta(code_keys, epoch=99):
    h = hashlib.sha256(b"ckpt").hexdigest()
    clips = ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]
    return {
        "schema": I.SCHEMA, "ckpt": "x/epoch_099.ckpt", "ckpt_sha256": h, "ckpt_epoch": epoch,
        "dataset": "d", "dsdir": "d", "foa": "d/foa", "glob": "", "prefix": "val_syn_",
        "clips": clips, "audio_sha256": {c: hashlib.sha256(c.encode()).hexdigest() for c in clips},
        "height": {"on": True}, "settings": {"batch": 1},
        "code_sha256": {k: hashlib.sha256(k.encode()).hexdigest() for k in code_keys},
    }


def main() -> int:
    ps = Path(tempfile.mkdtemp(prefix="ps_"))
    try:
        # 1. 論理名: ジョブ専用ディレクトリの写し・*_job<ID>.py の写し・共有パスの原本が同じ鍵になる
        for p in (ps / "_job4090/_causal_infer_v17.py", ps / "_job4111/_causal_infer_v17.py",
                  ps / "_causal_infer_v17_job4037.py", ps / "_causal_infer_v17.py"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# x\n", encoding="utf-8")
        names = {I.logical_code_name(p, ps) for p in (ps / "_job4090/_causal_infer_v17.py", ps / "_job4111/_causal_infer_v17.py",
                                                       ps / "_causal_infer_v17_job4037.py", ps / "_causal_infer_v17.py")}
        check("T1 推論台本の論理名は写しの場所に依らない", names == {"_causal_infer_v17.py"}, str(names))
        check("T2 来歴モジュールの論理名も同じ", I.logical_code_name(ps / "_job4090/_prediction_integrity.py", ps) == "_prediction_integrity.py")
        # 2. 置き場所の違いは残す（別のモジュールを同名にしない）
        sub = ps / "utils/height_patch.py"
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.write_text("# y\n", encoding="utf-8")
        check("T3 下位ディレクトリの名前は保つ", I.logical_code_name(sub, ps) == "utils/height_patch.py", I.logical_code_name(sub, ps))
        check("T4 root 外のファイルはファイル名だけ", I.logical_code_name(Path(tempfile.gettempdir()) / "z_job1.py", ps) == "z.py")
        check("T5 root 無しはファイル名だけ", I.logical_code_name(ps / "_job1/_causal_infer_v17.py") == "_causal_infer_v17.py")
        # 3. begin_run: 別ジョブの写しから作った同じ論理 meta は同じ run に受け入れられ、ジョブ固有の鍵は拒まれる
        run = ps / "infer_v20aft_e099_selfcausal"
        run.mkdir()
        keys = ["_causal_infer_v17.py", "_prediction_integrity.py", "height_patch.py"]
        m1 = make_meta(keys)
        I.begin_run(run, m1)
        m2 = copy.deepcopy(m1)                     # 別ジョブ（別の _job<ID>）で同じ論理名 → 同一
        I.begin_run(run, m2)
        check("T6 別ジョブの同じ論理 meta は受け入れ", (run / "run_meta.json").exists())
        m3 = make_meta(["_job4111/_causal_infer_v17.py", "_prediction_integrity.py", "height_patch.py"])
        try:
            I.begin_run(run, m3)
            check("T7 ジョブ固有の鍵は run identity changed", False, "受け入れてしまった")
        except ValueError as e:
            check("T7 ジョブ固有の鍵は run identity changed", "identity changed" in str(e), str(e))
        m4 = copy.deepcopy(m1)
        m4["code_sha256"]["_causal_infer_v17.py"] = hashlib.sha256(b"edited").hexdigest()
        try:
            I.begin_run(run, m4)
            check("T8 中身が違う写しは拒む", False, "受け入れてしまった")
        except ValueError as e:
            check("T8 中身が違う写しは拒む", "identity changed" in str(e), str(e))
        # 4. run 名の epoch と ckpt の epoch の不一致は拒む（既存の規則が残っていること）
        try:
            I.begin_run(ps / "infer_v20aft_e119_selfcausal", make_meta(keys, epoch=99))
            check("T9 名前と ckpt の epoch 不一致は拒む", False)
        except ValueError as e:
            check("T9 名前と ckpt の epoch 不一致は拒む", "epoch" in str(e), str(e))
    finally:
        shutil.rmtree(ps, ignore_errors=True)
    n_ok = sum(ok for _, ok in results)
    print(f"{n_ok}/{len(results)} {'ALL PASS' if n_ok == len(results) else 'FAILED'}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
