# -*- coding: utf-8 -*-
"""感度の変種の道具の検査（2026-10-21 Fable）: _sens_overrides／sens_variant の selftest・描く入口 _sens_render の点検（--list・記録とコードの差・置き場の名前）・
server_sde/sens_variant.sbatch の構文と乾式（SENS_DRYRUN=1・手元の python と偽の置き場で点検だけ）。
  python scripts/test_sens_variant.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BASH = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else "bash"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(cmd, env=None, cwd=ROOT):
    e = dict(os.environ, PYTHONUTF8="1"); e.update(env or {})
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ok = []
    rc, out = run([PY, "scripts/_sens_overrides.py", "--selftest"])
    ok.append(("T1 _sens_overrides --selftest", rc == 0 and "ALL PASS" in out))
    rc, out = run([PY, "scripts/sens_variant.py", "--selftest"])
    ok.append(("T2 sens_variant --selftest", rc == 0 and "ALL PASS" in out))
    src_plan = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
    if not src_plan.is_file():
        ok.append(("T3〜T7 は学習の plan（out/dataset_outdoor_siren_v2final/plan）が手元に無いので飛ばす", True))
    else:
        tag = "selftest_render"; ds = ROOT / "out" / f"dataset_outdoor_siren_v2final_eval_sens_{tag}"
        shutil.rmtree(ds, ignore_errors=True)
        tmp = Path(tempfile.mkdtemp(prefix="sens_t_"))
        try:
            rc, out = run([PY, "scripts/sens_variant.py", "plan", "--tag", tag, "--src-plan", str(src_plan), "--n", "12", "--seed", "3"])
            rc2, out2 = run([PY, "scripts/sens_variant.py", "overrides", "--tag", tag, "--set", "step11_v22_render.BIC_SLOPE_HIGH=20", "--note", "検査"])
            ok.append(("T3a plan 12 本と上書き（自転車の傾き 30 → 20）が作れる", rc == 0 and rc2 == 0 and "30.0 → 20.0" in out2))
            env = {"V2FINAL_DS_NAME": ds.name}
            rc, out = run([PY, "scripts/_sens_render.py", "--list"], env=env)
            ok.append(("T3b _sens_render --list: 上書き 1 件を当て・N = plan の 12（params の 6,000 でない）", rc == 0 and "[sens] 上書き 1 件" in out and "N=12 overrides=1" in out and "total rows: 12" in out))
            rc, out = run([PY, "scripts/_sens_render.py", "--list"], env={"V2FINAL_DS_NAME": "dataset_outdoor_siren_v2final_eval_v3"})
            ok.append(("T4 置き場の名前が _eval_sens_ でなければ 3（封印の eval v3 に向けられない）", rc == 3 and "NG" in out))
            J = json.loads((ds / "SENS_OVERRIDES.json").read_text(encoding="utf-8")); J["checked_old_values"]["step11_v22_render.BIC_SLOPE_HIGH"] = 31.0
            bak = (ds / "SENS_OVERRIDES.json").read_text(encoding="utf-8")
            (ds / "SENS_OVERRIDES.json").write_text(json.dumps(J, ensure_ascii=False), encoding="utf-8")
            rc, out = run([PY, "scripts/_sens_render.py", "--list"], env=env)
            ok.append(("T5 記録の『元の値』と今のコードが違えば 3", rc == 3 and "元の値" in out))
            (ds / "SENS_OVERRIDES.json").write_text(bak, encoding="utf-8")
            # sbatch: 構文と乾式
            rc, out = run([BASH, "-n", "server_sde/sens_variant.sbatch"])
            ok.append(("T6a sbatch の構文", rc == 0))
            fake_ckpt = tmp / "fake.ckpt"; fake_ckpt.write_bytes(b"x"); (tmp / "ps").mkdir()
            senv = {"SENS_DRYRUN": "1", "SENS_TAG": tag, "SENS_E2E": ROOT.as_posix(), "SENS_PS": (tmp / "ps").as_posix(), "SENS_PY": Path(PY).as_posix(), "V2FINAL_CKPT": fake_ckpt.as_posix(), "GUARD_FILE": (tmp / "guard.txt").as_posix()}
            rc, out = run([BASH, "server_sde/sens_variant.sbatch"], env=senv)
            ok.append(("T6b sbatch 乾式: 点検 → --list → SENS_VARIANT_DRYRUN_OK（描かない）", rc == 0 and "SENS_VARIANT_DRYRUN_OK" in out and "N=12 overrides=1" in out and not (ds / "foa").exists()))
            rc, out = run([BASH, "server_sde/sens_variant.sbatch"], env=dict(senv, V2FINAL_CKPT=(tmp / "none.ckpt").as_posix()))
            ok.append(("T6c sbatch: ckpt が無ければ 3", rc == 3 and "V2FINAL_CKPT" in out))
            (ds / "SENS_OVERRIDES.json").rename(ds / "SENS_OVERRIDES.json.off")
            rc, out = run([BASH, "server_sde/sens_variant.sbatch"], env=senv)
            ok.append(("T7 sbatch: SENS_OVERRIDES.json が無ければ 3（基準でも空の記録が要る）", rc == 3 and "SENS_OVERRIDES.json" in out))
            (ds / "SENS_OVERRIDES.json.off").rename(ds / "SENS_OVERRIDES.json")
        finally:
            shutil.rmtree(ds, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
            for p in ROOT.glob(f"sens_{tag}_*jobdry*"):
                p.unlink()
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
