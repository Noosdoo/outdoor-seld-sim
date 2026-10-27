# -*- coding: utf-8 -*-
"""_paired_ci の単体試験（2026-09-09）。
  P1 同じ予測どうし → 全指標の差 0・区間 [0, 0]
  P2 B の距離を全部 1.3 倍にした予測 → 距離誤差の差は正で、区間が 0 を含まない
  P3 場面の鍵: v16/v18 の mix N と N+10000、v19 の mix 20001+i と 21001+i が同じ場面
使い方: python scripts/test_paired_ci.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("paired_ci", ROOT / "scripts/_paired_ci.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
PY = sys.executable
SRC = ROOT / "out/v17b/C/infer_v17bft_e139_v12causal.csv"
PLAN = ROOT / "out/dataset_outdoor_siren_v12/plan/manifest_val_syn.csv"     # 2026-09-22 改名（旧 manifest_fold2_val.csv）。SRC は旧名の予測（履歴）のまま = 正規化で突き合わせる
META = ROOT / "out/dataset_outdoor_siren_v12/metadata_dist_h"


def run_ci(out, spec, n_boot=60):
    r = subprocess.run([PY, str(ROOT / "scripts/_paired_ci.py"), str(out), "--plan", str(PLAN.relative_to(ROOT)), "--meta", str(META.relative_to(ROOT)),
                        "--n-boot", str(n_boot), "--seed", "1", spec], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]
    rows = [l for l in out.read_text(encoding="utf-8").splitlines() if l.startswith("| t")]
    return {l.split("|")[2].strip(): l for l in rows}


def main() -> int:
    checks = []
    checks.append(("P3 v16 の対", P.scene_key("val_syn_clip10007") == "val_syn_clip00007"))
    checks.append(("P3 v19 の対", P.scene_key("val_syn_clip21005") == P.scene_key("val_syn_clip20005") == "val_syn_clip20005"))
    checks.append(("P3 別の場面は別", P.scene_key("val_syn_clip22005") != P.scene_key("val_syn_clip20005")))
    checks.append(("P3 旧名でも同じ鍵", P.scene_key("fold2_room1_mix10007") == "val_syn_clip00007"))
    if not SRC.exists():
        print("SKIP P1/P2: 予測ファイルが無い", SRC)
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 300 本ぶんだけ抜いて速くする
            lines = [l for l in SRC.read_text(encoding="utf-8").splitlines() if l.split(",")[0] <= "fold2_room1_mix0300"]
            a = tmp / "a.csv"
            a.write_text("\n".join(lines) + "\n", encoding="utf-8")
            b = tmp / "b.csv"
            out_b = []
            for l in lines:
                p = l.split(",")
                if p[6] not in ("nan", ""):
                    p[6] = f"{float(p[6]) * 1.3:.2f}"
                out_b.append(",".join(p))
            b.write_text("\n".join(out_b) + "\n", encoding="utf-8")
            r1 = run_ci(tmp / "ci1.md", f"t1={a},{a}")
            checks.append(("P1 同一 → 差 0", all("| +0.0 |" in l and "[+0.0, +0.0]" in l for l in r1.values())))
            r2 = run_ci(tmp / "ci2.md", f"t2={a},{b}")
            d = r2.get("距離誤差", "")
            checks.append(("P2 距離 1.3 倍 → 距離誤差の差が正で 0 を含まない", "含まない" in d and "| +" in d.split("| 95%")[0]))
    ok = all(x for _, x in checks)
    for name, x in checks:
        print(("PASS " if x else "FAIL ") + name)
    print(f"{sum(x for _, x in checks)}/{len(checks)} " + ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
