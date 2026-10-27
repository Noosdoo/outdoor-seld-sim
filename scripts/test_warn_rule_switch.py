# -*- coding: utf-8 -*-
"""test_warn_rule_switch.py — 変更票 #41（警告音の規則 hold → v9c）の採点器の切り替えの検査（2026-09-28・Fable）。

  W1 統一採点器 `_score_unified.fires_of`: WARN_RULE=hold の警告音の発火が `step12_notify_v9b_hold.warn_fires(hold=True)` と同一、
     WARN_RULE=v9c の発火が `step12_notify_v9c_multitrack.warn_fires_v9c(門 45°・同一方位 30°)` と同一。車の発火（強／中）は規則によらず同一。
  W2 実録の採点器 `step20_realsmoke_score_v3.fires_of` も同じ（時刻は (k+1)/FPS）。
  W3 回帰: `--warn-rule hold` で走らせた統一採点器の出力（本文）が、改訂前（git の基準コミット）の `_score_unified.py` の出力と一致する
     （開発用の予測 out/_b02_2026-09-24/dev_e149_v12causal.csv・v12 の val・clip 番号 ≤ CLIP_MAX）。
  W4 既定（v9c）で走らせた出力は本文の少なくとも 1 行が hold と違い（警告音の到達が変わる）、終了コード 0。
出力は out/_test_warn_rule_switch/（既存の out には書かない）。改訂前の写しは scripts/_tmp_score_unified_BASE.py に一時的に置き、終了時に消す。
使い方: python scripts/test_warn_rule_switch.py [--skip-regression]
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE_COMMIT = "80506bbe0"     # 回帰の基準 = 警告音の規則を採点器に入れる前の最後のコミット（2026-09-28 16:10）
PRED = ROOT / "out/_b02_2026-09-24/dev_e149_v12causal.csv"
PLAN = ROOT / "out/dataset_outdoor_siren_v12/plan/manifest_val_syn.csv"
META = ROOT / "out/dataset_outdoor_siren_v12/metadata_dist"
CLIP_MAX = 600
OUT = ROOT / "out/_test_warn_rule_switch"


def _load(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def synth_frames():
    """frame -> [(cls, az, el, d)]: サイレン（cls 0）が 0° → 100° へ動き、10〜30 フレームは +20° の二重出力つき。車（cls 4）は 20 m → 2 m に近づく"""
    fr = {}
    for k in range(100):
        v = [(0, 1.0 * k, 0.0, float("nan"))]
        if 10 <= k < 30:
            v.append((0, 1.0 * k + 20.0, 0.0, float("nan")))
        v.append((4, 30.0, 0.0, max(20.0 - 0.3 * k, 1.5)))
        fr[k] = v
    return fr


def main() -> int:
    a = sys.argv[1:]
    ok = []
    su = _load("su_test", ROOT / "scripts/_score_unified.py")
    s20 = _load("s20_test", ROOT / "scripts/step20_realsmoke_score_v3.py")
    H, V9C = su.H, su.V9C
    fr = synth_frames()
    fw = {k: [(c, az, el) for c, az, el, d in v] for k, v in fr.items()}
    exp_hold = H.warn_fires(fw, hold=True, nframes=100)
    exp_v9c = V9C.warn_fires_v9c(fw, 45.0, nframes=100, dedup_deg=30.0)
    ok.append(("合成: hold と v9c の発火が違う（検査が意味を持つ）", exp_hold != exp_v9c and len(exp_v9c) == 1 and len(exp_hold) >= 2))
    # W1
    got = {}
    for rule in ("hold", "v9c"):
        su.WARN_RULE = rule
        f = su.fires_of(fr)
        got[rule] = f
        warn = sorted((k, c, az) for k, az, tier, d, c in f if tier == "警告")
        ok.append((f"W1 統一採点器 {rule}: 警告音の発火が規則の関数と同一", warn == sorted(exp_hold if rule == "hold" else exp_v9c)))
    car = {r: sorted(x for x in got[r] if x[2] != "警告") for r in got}
    ok.append(("W1 車の発火は規則によらず同一", car["hold"] == car["v9c"] and len(car["hold"]) > 0))
    su.WARN_RULE = "v9c"
    # W2
    for rule in ("hold", "v9c"):
        s20.WARN_RULE = rule
        f = s20.fires_of(fr, s20.V43.Cfg43(**__import__("json").loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8"))), 100)
        warn = sorted((round(t * s20.FPS) - 1, c, az) for t, az, tier, c in f if tier == "警告")
        ok.append((f"W2 実録の採点器 {rule}: 警告音の発火が規則の関数と同一", warn == sorted(exp_hold if rule == "hold" else exp_v9c)))
    s20.WARN_RULE = "v9c"
    # W3 / W4
    if "--skip-regression" not in a:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        base_path = ROOT / "scripts/_tmp_score_unified_BASE.py"
        try:
            src = subprocess.run(["git", "-C", str(ROOT.parent), "show", f"{BASE_COMMIT}:outdoor_seld_e2e/scripts/_score_unified.py"], capture_output=True, check=True).stdout
            base_path.write_bytes(src)
            common = ["--plan", str(PLAN.relative_to(ROOT)), "--meta", str(META.relative_to(ROOT)), "--clip-max", str(CLIP_MAX), f"dev={PRED.relative_to(ROOT)}"]
            def run(script, out_md, extra):
                r = subprocess.run([PY, str(script), str(out_md)] + extra + common, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
                return r.returncode, (r.stderr or "")[-500:]
            rc0, e0 = run(base_path, OUT / "base.md", [])
            rc1, e1 = run(ROOT / "scripts/_score_unified.py", OUT / "new_hold.md", ["--warn-rule", "hold"])
            rc2, e2 = run(ROOT / "scripts/_score_unified.py", OUT / "new_v9c.md", [])
            ok.append(("W3 改訂前・hold 指定・既定の 3 回とも終了コード 0", rc0 == 0 and rc1 == 0 and rc2 == 0))
            if rc0 == 0 and rc1 == 0 and rc2 == 0:
                body = lambda p: [l for l in (OUT / p).read_text(encoding="utf-8").splitlines()[3:] if l.startswith("|")]
                b0, b1, b2 = body("base.md"), body("new_hold.md"), body("new_v9c.md")
                def main_table(rows):
                    # 2026-10-25 Opus の案（Fable の確認待ち）: 最初の表だけ（境目の感度の表 #69 r20 は後から足した）・早さの列は除く（「≥2.5s」→「≥2.0s前(全事象)」に定義ごと変えた）
                    rows = rows[:next((i for i, l in enumerate(rows) if i > 0 and l.startswith("| 境目")), len(rows))]
                    cells = [l.split("|") for l in rows]
                    drop = {i for i, c in enumerate(cells[0]) if c.strip() in ("≥2.5s", "≥2.0s前(全事象)")} if cells else set()
                    return ["|".join(c for i, c in enumerate(r) if i not in drop) for r in cells]
                m0, m1 = main_table(b0), main_table(b1)
                ok.append((f"W3 --warn-rule hold の本文が改訂前（{BASE_COMMIT}）と同一（早さの列と後から足した表を除く・{len(m0)} 行）", m0 == m1 and len(m0) > 0))
                ok.append(("W4 既定（v9c）の本文は hold と少なくとも 1 行違う", b2 != b1 and len(b2) == len(b1)))
                print("   hold:", b1[1][:160] if len(b1) > 1 else b1)
                print("   v9c :", b2[1][:160] if len(b2) > 1 else b2)
            else:
                print("   stderr:", e0 or e1 or e2)
        finally:
            base_path.unlink(missing_ok=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
