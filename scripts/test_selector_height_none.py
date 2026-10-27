# -*- coding: utf-8 -*-
"""選択器 2.1 の --height-none（変更票 #74・アブレーション noheight）の検査。高さの入力なし（height.on=false）の候補を受け、--height-table の候補は拒む。
使い方: python scripts/test_selector_height_none.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "server_sde")]
import _prediction_integrity as I  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("v2sel_hn", ROOT / "scripts/_v2_ckpt_selector.py")
S = importlib.util.module_from_spec(_spec); sys.modules["v2sel_hn"] = S; _spec.loader.exec_module(S)
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)[:200]) if detail and not cond else ''}")


def make_run(run: Path, foa: Path, clips, height, epoch=99):
    (run / "submissions").mkdir(parents=True, exist_ok=True)
    meta = {"schema": I.SCHEMA, "ckpt": "x.ckpt", "ckpt_sha256": hashlib.sha256(b"c").hexdigest(), "ckpt_epoch": epoch, "dataset": "d", "foa": str(foa), "glob": "*", "prefix": "",
            "clips": clips, "audio_sha256": I.audio_manifest(foa, clips), "height": height, "code_sha256": {"i": "c" * 64}, "settings": {"batch": 1}}
    I.begin_run(run, meta)
    for c in clips:
        (run / "submissions" / (Path(c).stem + ".csv")).write_text("".join(f"{k},4,0,30,0,1.3\n" for k in range(40, 100)), encoding="utf-8")
    I.publish(run, meta)
    (run / "val_all_causal.run_meta.json").write_text((run / "run_meta.json").read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    T = Path(tempfile.mkdtemp(prefix="selhn_"))
    foa = T / "foa"; foa.mkdir(); clips = ["val_syn_clip00001.flac"]
    (foa / clips[0]).write_bytes(b"a")
    ht = T / "ht.csv"; ht.write_text("clip_id,mic_z\nval_syn_clip00001,1.5\n", encoding="utf-8")
    integrity = I
    r_none = T / "infer_x_abl_noheight_e099_selfcausal"; make_run(r_none, foa, clips, {"on": False, "table": "", "table_sha256": None, "offset": None, "default": None, "values": {}})
    r_tab = T / "infer_x_e099_selfcausal"; make_run(r_tab, foa, clips, {"on": True, "table": str(ht), "table_sha256": I.sha256(ht), "offset": 0.0, "default": 1.5, "values": {"val_syn_clip00001": 1.5}})
    ids = ["val_syn_clip00001"]
    ok1 = False
    try:
        S.verify_candidate(integrity, r_none / "val_all_causal.csv", 99, ids, 1, None, True); ok1 = True
    except S.InputError as e:
        print("  ", e)
    check("--height-none: 高さなし（on=false）の候補を受ける", ok1)
    try:
        S.verify_candidate(integrity, r_tab / "val_all_causal.csv", 99, ids, 1, None, True); ok2 = False
    except S.InputError:
        ok2 = True
    check("--height-none: 高さの表つきの候補は拒む", ok2)
    try:
        S.verify_candidate(integrity, r_none / "val_all_causal.csv", 99, ids, 1, ht, False); ok3 = False
    except S.InputError:
        ok3 = True
    check("--height-table: 高さなしの候補は拒む（既存の契約のまま）", ok3)
    (T / "m.csv").write_text("clip_id\nval_syn_clip00001\n", encoding="utf-8")
    (T / "gt").mkdir(); (T / "gt/val_syn_clip00001.csv").write_text("".join(f"{k},4,0,30,0,1.2\n" for k in range(40, 100)), encoding="utf-8")
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = S.main(["--pattern", str(T / "x{epoch:03d}.csv"), "--manifest", str(T / "m.csv"), "--gt", str(T / "gt"), "--expected", "1", "--out", str(T / "o.json"), "--height-none", "--height-table", str(ht), "--label", "abl_test"])
    check("--height-none と --height-table は同時に渡せない（rc 1・その理由で止まる）", rc == 1 and "同時に渡せない" in err.getvalue(), err.getvalue()[-200:])
    k = sum(c for _, c in results); print(f"{k}/{len(results)} {'ALL PASS' if k == len(results) else 'FAILED'}")
    return 0 if k == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
