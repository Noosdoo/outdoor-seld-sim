# -*- coding: utf-8 -*-
"""test_v2final_relabel_append.py — step10_v2final_relabel.py が「追加生成された clip」を扱えるかの検査（2026-09-28・job 5074 の NG を受けて）。

一時フォルダに小さな置き場を作る: plan（v12train 1 本・v11core 1 本・v19kickfar 1 本）、metadata/ と metadata_dist/ に 3 本の csv。
  R1 初回の変換: _orig ができ、v12train の crossing 行だけ落ち、他は不変、--check が通る
  R2 追加生成: 新しい clip（kickfar）を metadata/ と metadata_dist/ にだけ足して plan にも足す → 変換で _orig に足され、NG にならない・既存の _orig は不変
  R3 もう一度通しても同じ結果（何度通しても同じ）・--check が通る
  R4 --check だけで _orig に無い clip があれば NG（足さない）
使い方: python scripts/test_v2final_relabel_append.py
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPT = ROOT / "scripts/step10_v2final_relabel.py"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(ds: Path, check=False):
    r = subprocess.run([PY, str(SCRIPT), "--ds", str(ds)] + (["--check"] if check else []), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def write_plan(ds: Path, clips: list[tuple[str, str]]):
    (ds / "plan").mkdir(parents=True, exist_ok=True)
    (ds / "plan/assignment_v2final.csv").write_text("clip_id,split,scenario\n" + "".join(f"{c},train,{s}\n" for c, s in clips), encoding="utf-8")


def write_labels(ds: Path, clip: str, lines: list[str]):
    for sub in ("metadata", "metadata_dist"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
        (ds / sub / f"{clip}.csv").write_bytes(("".join(l + "\n" for l in lines)).encode("utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="relabel_append_"))
    ok = []
    try:
        ds = tmp / "ds"
        train = ["0,5,0,10,0,20.5", "1,5,0,11,0,20.1", "1,4,0,30,0,12.0"]          # crossing 2 行 ＋ car 1 行
        core = ["0,5,0,90,0,15.0", "0,4,0,-40,0,5.0"]                                 # 警報機の crossing は保持
        write_plan(ds, [("train_syn_clip00001", "v12train"), ("train_syn_clip00002", "v11core")])
        write_labels(ds, "train_syn_clip00001", train)
        write_labels(ds, "train_syn_clip00002", core)
        rc, out = run(ds)
        m = (ds / "metadata/train_syn_clip00001.csv").read_text(encoding="utf-8").splitlines()
        ok.append(("R1 初回: rc 0・v12train は crossing 2 行が落ちて car 1 行だけ", rc == 0 and m == ["1,4,0,30,0,12.0"]))
        ok.append(("R1 初回: v11core は不変（警報機の crossing 保持）", (ds / "metadata/train_syn_clip00002.csv").read_text(encoding="utf-8").splitlines() == core))
        ok.append(("R1 初回: _orig に元が残る", (ds / "metadata_orig/train_syn_clip00001.csv").read_text(encoding="utf-8").splitlines() == train))
        rc_c, _ = run(ds, check=True)
        ok.append(("R1 初回: --check が通る", rc_c == 0))
        orig_sha_before = {p.name: sha(p) for p in (ds / "metadata_orig").glob("*.csv")}
        # R4: 追加生成の後に --check だけ → NG（足さない）
        kick = ["3,6,0,60,0,2.9", "4,6,0,58,0,2.8"]
        write_labels(ds, "train_syn_clip30001", kick)
        write_plan(ds, [("train_syn_clip00001", "v12train"), ("train_syn_clip00002", "v11core"), ("train_syn_clip30001", "v19kickfar")])
        rc_c2, out_c2 = run(ds, check=True)
        ok.append(("R4 追加後の --check だけは NG（_orig に無いファイル 1）", rc_c2 == 1 and "_orig に無いファイル 1" in out_c2 and not (ds / "metadata_orig/train_syn_clip30001.csv").exists()))
        # R2: 変換で _orig に足される
        rc2, out2 = run(ds)
        ok.append(("R2 追加生成: rc 0・_orig に無い 1 本を原本として追加", rc2 == 0 and "_orig に無い 1 本を原本として追加" in out2))
        ok.append(("R2 追加生成: 新 clip の _orig と metadata が同じ（規則の対象外）", (ds / "metadata_orig/train_syn_clip30001.csv").exists()
                   and sha(ds / "metadata_orig/train_syn_clip30001.csv") == sha(ds / "metadata/train_syn_clip30001.csv")))
        ok.append(("R2 追加生成: 既存の _orig は不変", all(sha(ds / "metadata_orig" / n) == h for n, h in orig_sha_before.items())))
        # R3: もう一度
        snap = {p.name: sha(p) for p in (ds / "metadata").glob("*.csv")}
        rc3, _ = run(ds)
        snap2 = {p.name: sha(p) for p in (ds / "metadata").glob("*.csv")}
        rc_c3, _ = run(ds, check=True)
        ok.append(("R3 もう一度通しても同じ・--check が通る", rc3 == 0 and snap == snap2 and rc_c3 == 0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
