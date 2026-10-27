# -*- coding: utf-8 -*-
"""step10_calib_sets_plan.py — 距離の補正表の「作る組」calib_fit（2,000 本）と「確かめる組」calib_check（2,000 本）の plan（変更票 #72・本人 2026-10-20 03:34 (c)・Fable）。

凍結 §5 の手順（表の作り方・採用の条件）は変えない。変えるのは組だけ: fold32／fold30（v12 の頃）を、v29 の生成器で同じ分布から引いた新しい組に置き換える。
  - 引き方は eval v3（step10_eval_v3_plan.build）と同じ: 層 = (scenario, mix) の割合を最終の学習 plan（train＋val）に比例・写す条件は同じ・側・歩行・雨・高さ・徐行・D19 は別の seed で引き直す。
  - 番号 = tune_syn_clip00001〜02000（calib_fit）・02001〜04000（calib_check）。split = tune（学習の入口の門が弾く・評価セットではない）。
  - seed = 9400000000+（fit）・9450000000+（check）。学習・検証（9.10e9〜9.21e9・1.242e10・9203000000+）・eval v3（9300000000+）と重ならないことを自動検査。
  - 塩（layer の抽選・側・高さ）は eval v3 と別（md/design/calib_sets_v29.json）。
出力 = out/dataset_outdoor_siren_v2final_calib/plan/{assignment_v2final.csv, manifest_calib_strata.csv, manifest_v26_bellbike.csv, README_calib.md}・calib_sets_v29.json に sha。
使い方: python scripts/step10_calib_sets_plan.py [--dry-run] [--sample 50] / --selftest
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


e3 = _load("s10e3_calib", "step10_eval_v3_plan.py")
p27 = e3.p27
CFG = ROOT / "md/design/calib_sets_v29.json"
OUT_DIR = ROOT / "out" / "dataset_outdoor_siren_v2final_calib" / "plan"
PLAN = OUT_DIR / "assignment_v2final.csv"
MANIFEST = OUT_DIR / "manifest_calib_strata.csv"
README = OUT_DIR / "README_calib.md"
EVAL_PLAN = ROOT / "out/dataset_outdoor_siren_v2final_eval_v3/plan/assignment_v2final.csv"
RE_ID = re.compile(r"^tune_syn_clip\d{5}$")


def build_sets(src: list, P: dict, cfg: dict, n_override: int | None = None) -> tuple[list, list]:
    rows, manifest, j0 = [], [], 0
    for name in ("calib_fit", "calib_check"):
        c = cfg["sets"][name]; n = int(n_override or c["n"])
        e3.SEED0, e3.SALT_PICK, e3.SALT_SIDE, e3.SALT_Z = int(c["seed_base"]), int(c["salt_pick"]), int(c["salt_side"]), int(c["salt_z"])
        rs, man = e3.build(src, P, n)
        for k, r in enumerate(rs):
            r["clip_id"] = f"tune_syn_clip{j0 + k + 1:05d}"; r["split"] = cfg["split"]
        for m in man:
            m = dict(m); m["set"] = name
            m["clip_first"] = f"{j0 + int(m['clip_first']):05d}"; m["clip_last"] = f"{j0 + int(m['clip_last']):05d}"
            manifest.append(m)
        rows += rs; j0 += len(rs)
    return rows, manifest


def check(src: list, rows: list, manifest: list, cfg: dict, eval_rows: list | None) -> list:
    msgs = []
    ids = [r["clip_id"] for r in rows]
    n = sum(int(cfg["sets"][k]["n"]) for k in cfg["sets"])
    msgs.append((f"本数 {len(rows)} = {n}（fit {cfg['sets']['calib_fit']['n']} ＋ check {cfg['sets']['calib_check']['n']}）", len(rows) == n))
    msgs.append(("番号 tune_syn_clip00001〜 がユニークで連番", len(set(ids)) == len(ids) and all(RE_ID.match(i) for i in ids) and ids == [f"tune_syn_clip{k + 1:05d}" for k in range(len(ids))]))
    msgs.append(("split は全部 tune", all(r["split"] == "tune" for r in rows)))
    seeds = [int(r["seed"]) for r in rows]
    src_seeds = {int(r["seed"]) for r in src if r.get("seed")}
    ev_seeds = {int(r["seed"]) for r in (eval_rows or []) if r.get("seed")}
    msgs.append(("seed がユニークで、学習・検証・eval v3 の seed と重ならない", len(set(seeds)) == len(seeds) and not (set(seeds) & src_seeds) and not (set(seeds) & ev_seeds)))
    fit_seeds = {int(r["seed"]) for r in rows if int(r["clip_id"][-5:]) <= int(cfg["sets"]["calib_fit"]["n"])}
    chk_seeds = set(seeds) - fit_seeds
    msgs.append(("fit と check の seed が重ならない", not (fit_seeds & chk_seeds)))
    strata_src = Counter(e3.stratum(r) for r in src)
    for name in ("calib_fit", "calib_check"):
        got = Counter(e3.stratum(r) for r in rows if (int(r["clip_id"][-5:]) <= int(cfg["sets"]["calib_fit"]["n"])) == (name == "calib_fit"))
        n_set = int(cfg["sets"][name]["n"])
        dev = max(abs(got.get(k, 0) / n_set - strata_src[k] / len(src)) for k in strata_src)
        msgs.append((f"{name}: 層の割合が元 plan に比例（最大のずれ {dev * 100:.2f} ポイント・各層 {e3.MIN_PER_STRATUM} 本以上）", dev < 0.01 and all(got.get(k, 0) >= e3.MIN_PER_STRATUM for k in strata_src)))
    msgs.append(("mic_z が 1.4〜2.1・歩行と静止がある", all(1.4 <= float(r["mic_z"]) <= 2.1 for r in rows) and {r["motion"] for r in rows} == {"walk", "static"}))
    return msgs


def selftest() -> int:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    P = json.loads(e3.PARAMS.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    src = e3.read_rows(e3.SRC_PLAN)
    rows, man = build_sets(src, P, cfg, n_override=600)
    ok = [(f"小さい組（600 × 2）が作れる: {len(rows)} 行・層 {len(man)}", len(rows) == 1200 and len(man) > 0)]
    ok.append(("番号 00001〜01200・split tune", rows[0]["clip_id"] == "tune_syn_clip00001" and rows[-1]["clip_id"] == "tune_syn_clip01200" and all(r["split"] == "tune" for r in rows)))
    s_fit = {int(r["seed"]) for r in rows[:600]}; s_chk = {int(r["seed"]) for r in rows[600:]}
    ok.append(("fit は 9400000000〜・check は 9450000000〜・重ならない", min(s_fit) == 9400000000 and min(s_chk) == 9450000000 and not (s_fit & s_chk)))
    ok.append(("学習・検証の seed と重ならない", not (s_fit | s_chk) & {int(r["seed"]) for r in src}))
    rows2, _ = build_sets(src, P, cfg, n_override=600)
    ok.append(("決定的（同じ入力で同じ plan）", rows2 == rows))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    dry = "--dry-run" in a
    cfg = json.loads(CFG.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    P = json.loads(e3.PARAMS.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    assert P.get("status") == "fixed", "params が fixed でない"
    src = e3.read_rows(e3.SRC_PLAN)
    assert all(r["split"] in ("train", "val") for r in src), "元の plan に train/val 以外の行がある"
    eval_rows = e3.read_rows(EVAL_PLAN) if EVAL_PLAN.is_file() else None
    rows, manifest = build_sets(src, P, cfg)
    msgs = check(src, rows, manifest, cfg, eval_rows)
    for name, g in msgs:
        print(("OK " if g else "NG ") + name)
    lim = int(a[a.index("--sample") + 1]) if "--sample" in a else None
    bad = e3.sample_check(rows, lim)
    print("- 層（scenario）:", dict(sorted(Counter(r["scenario"] for r in rows).items())))
    if not all(g for _, g in msgs) or bad:
        print("検査に落ちたので書かない"); return 1
    if dry:
        print("dry-run: 書かない"); return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert not PLAN.exists(), f"{PLAN} が既にある（作り直すなら名前を変える）"
    p27.write_rows(PLAN, rows, e3.COLS)
    with open(MANIFEST, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(manifest)
    import step11_v25_render as v25
    bb = v25.write_bellbike_manifest(OUT_DIR / "manifest_v26_bellbike.csv", PLAN)
    cfg["plan_sha256"] = p27.sha256(PLAN); cfg["manifest_sha256"] = p27.sha256(MANIFEST); cfg["src_plan_sha256"] = p27.sha256(e3.SRC_PLAN)
    cfg["src_plan"] = e3.SRC_PLAN.relative_to(ROOT).as_posix(); cfg["src_rows"] = len(src); cfg["n_strata"] = len(manifest)
    cfg["bellbike_manifest"] = f"manifest_v26_bellbike.csv（{bb['n']} 本・後 {bb['after']}・前 {bb['before']}・sha256 {p27.sha256(OUT_DIR / 'manifest_v26_bellbike.csv')[:16]}）"
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    L = [f"# 補正表の組の plan（calib_fit 2,000 ＋ calib_check 2,000・変更票 #72・2026-10-20）", "",
         f"- 元の plan = {cfg['src_plan']}（{len(src):,} 行）sha256 {cfg['src_plan_sha256']}", f"- plan sha256 {cfg['plan_sha256']}・層 {len(manifest)}（fit／check で各 {len(manifest) // 2}）",
         "- 引き方は eval v3 と同じ（層の割合は元 plan に比例・写す条件は同じ・側・歩行・雨・高さ・徐行・D19 は別の seed）。番号 tune_syn_clip00001〜02000 = calib_fit・02001〜04000 = calib_check・split = tune（学習の入口の門が弾く）",
         f"- seed: fit {cfg['sets']['calib_fit']['seed_base']}〜・check {cfg['sets']['calib_check']['seed_base']}〜（学習・検証・eval v3 と重ならない・自動検査）", ""]
    L += [f"- {m['set']} {m['scenario']} {m['mix'] or '-'}: 元 {m['n_src']:,} → {m['n_eval']}（clip {m['clip_first']}〜{m['clip_last']}）" for m in manifest]
    README.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("書いた:", PLAN, "/", MANIFEST, "/", README, "/", CFG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
