# -*- coding: utf-8 -*-
"""server_sde/_v30_val_match_check.py の試験（2026-10-27 Opus）。架空の置き場だけ。
  V1 中身が全部同じ・版の記録の違いが許す 3 本だけ → 0
  V2 1 本の masks が 1 バイト違う → 8（その本の masks が一覧に）
  V3 scene.json が欠ける → 8
  V4 版の記録で許さないファイル（band_gate.py）の sha が違う → 8
  V5 版の記録の render: の行（幅）が違う → 8
  V6 scene.json の違いが描いた秒数（stats.gen_seconds）だけ → 0（2026-10-27・job 9388）
  V7 scene.json のほかの鍵が違う → 8
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_sde"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _v30_val_match_check as M  # noqa: E402

H = lambda c: c * 64  # noqa: E731
GV = ("render chain (import order): step11_v9_render.py -> step11_v30_render.py\n"
      "render: v30 FEATURES={\"bg_shape\": true} GATE_BAND_DB=0 BG_KNEE_RANGE=[63.0, 1000.0] BG_SLOPE_RANGE=[-6.5, -2.5]\nplan rows expected: 62102\n\n"
      f"{H('a')}  scripts/_run_v2final_gen.py\n{H('b')}  scripts/step11_v30_render.py\n{H('c')}  src/outdoor_seld/band_gate.py\n")


def make(root: Path, clips, gv: str):
    for c in clips:
        for d, ext in M.FILES:
            p = root / d / f"{c}{ext}"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(f"{d}:{c}".encode())
        s = root / "work" / c / "scene.json"; s.parent.mkdir(parents=True, exist_ok=True); s.write_text('{"x": 1}', encoding="utf-8")
    (root / "GEN_VERSIONS.txt").write_text(gv, encoding="utf-8")


def run(tmp, a, b):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = M.main(["--a", str(a), "--b", str(b), "--manifest", str(tmp / "m.csv"), "--out", str(tmp / "r.json")])
    return rc, buf.getvalue()


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="valmatch_"))
    try:
        clips = ["val_syn_clip00001", "val_syn_clip00002", "val_syn_clip00003"]
        (tmp / "m.csv").write_text("clip_id\n" + "\n".join(clips) + "\n", encoding="utf-8")
        a, b = tmp / "a", tmp / "b"
        gv_a = GV.replace(H("a") + "  scripts/_run_v2final_gen.py", H("d") + "  scripts/_run_v2final_gen.py").replace(H("b"), H("e")) + f"{H('f')}  scripts/_atomic_write.py\n"
        make(a, clips, gv_a); make(b, clips, GV)
        rc, out = run(tmp, a, b)
        ok.append(("V1 中身が同じ・版の記録の違いは許す 3 本だけ → 0", rc == 0 and "一致" in out))
        (a / "masks" / "val_syn_clip00002.csv").write_bytes(b"masks:val_syn_clip00002 ")
        rc, out = run(tmp, a, b)
        ok.append(("V2 1 本の masks が違う → 8", rc == 8 and "masks/val_syn_clip00002.csv" in (tmp / "r.json").read_text(encoding="utf-8")))
        (a / "masks" / "val_syn_clip00002.csv").write_bytes(b"masks:val_syn_clip00002")
        (a / "work" / "val_syn_clip00003" / "scene.json").unlink()
        rc, out = run(tmp, a, b)
        ok.append(("V3 scene.json が欠ける → 8", rc == 8 and "work/val_syn_clip00003/scene.json" in (tmp / "r.json").read_text(encoding="utf-8")))
        (a / "work" / "val_syn_clip00003" / "scene.json").write_text('{"x": 1}', encoding="utf-8")
        (a / "GEN_VERSIONS.txt").write_text(gv_a.replace(H("c"), H("9")), encoding="utf-8")
        rc, out = run(tmp, a, b)
        ok.append(("V4 許さないファイル（band_gate.py）の sha が違う → 8", rc == 8 and "src/outdoor_seld/band_gate.py" in out))
        (a / "GEN_VERSIONS.txt").write_text(gv_a.replace("[-6.5, -2.5]", "[-5.0, -3.0]"), encoding="utf-8")
        rc, out = run(tmp, a, b)
        ok.append(("V5 render: の行（幅）が違う → 8", rc == 8))
        # V6・V7（2026-10-27・job 9388）: scene.json の描いた秒数（stats.gen_seconds）だけの違いは中身が同じ → 0・ほかの鍵が違えば 8
        (a / "GEN_VERSIONS.txt").write_text(gv_a, encoding="utf-8")
        for root, sec in ((a, 12.2), (b, 25.4)):
            for c in clips:
                (root / "work" / c / "scene.json").write_text('{"x": 1, "stats": {"gen_seconds": %s, "n": 3}}' % sec, encoding="utf-8")
        rc, out = run(tmp, a, b)
        import json as _json
        r6 = _json.loads((tmp / "r.json").read_text(encoding="utf-8"))
        ok.append(("V6 scene.json の違いが描いた秒数だけ → 0（3 本を数える）", rc == 0 and r6["scene_timing_only"] == 3 and "一致" in out))
        (a / "work" / "val_syn_clip00001" / "scene.json").write_text('{"x": 1, "stats": {"gen_seconds": 12.2, "n": 4}}', encoding="utf-8")
        rc, out = run(tmp, a, b)
        ok.append(("V7 scene.json のほかの鍵（stats.n）が違う → 8", rc == 8 and "work/val_syn_clip00001/scene.json" in (tmp / "r.json").read_text(encoding="utf-8")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
