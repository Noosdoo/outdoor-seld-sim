# -*- coding: utf-8 -*-
"""test_v2final_mix.py — 変更票 #64（D31 組み合わせの網羅 v19mix・D32 遠いバイク v19motofar）の実装の検査（2026-10-13・Fable）。

  M1 plan の構造: 総行数 = params の N_ROWS_TOTAL_EXPECTED_V27・先頭 31,274 行は前の plan（assignment_v2final_pre_v27.csv）と列の値が 1 つも違わない
     （新しい列 w3_class／w3_side／mix は空）・新しい行は v19mix と v19motofar だけ・clip 番号は 50001〜（mix 高さ 1）・60001〜（高さ 2）・70001〜／71001〜（motofar）・
     同じ種の 2 本は同じ split・val は 20%・manifest_v27_mix.csv と組ごとの本数が一致。
  M2 網羅: 規則に合う 2 つ組 54 種・3 つ組 209 種（電車＋バック音と電車 3 編成を除く）の全部が、最終の plan に 20 本以上ある（音源の構成がちょうど同じ本で数える）。
     足した組は 34 ＋ 190 で、足さなかった組（元から 20 本以上）は 20 ＋ 19。
  M3 サンプラ: plan の v19mix から固定 seed で 60 本・v19motofar から 20 本を引き、check_row（組み合わせ・track・マイクの共有・電車は背景・横距離）が通る。
  M4 生成ドライバ: `_run_v2final_gen.py --list` が総行数 50,574 と v19mix 17,800・v19motofar 1,500 を報告する（v27 のサンプラが登録される）。
  M5 step11_v27_render --selftest が全部 PASS（既存の場面を壊していない比較を含む）。
使い方: PYTHONPATH=scripts:src python scripts/test_v2final_mix.py [--quick]   （--quick は M5 を飛ばす）
"""
from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v27_plan as p27  # noqa: E402

PLAN, PRE, MAN, PARAMS = p27.PLAN, p27.PLAN_PRE, p27.MANIFEST, p27.PARAMS


def main() -> int:
    quick = "--quick" in sys.argv[1:]
    P = json.loads(PARAMS.read_text(encoding="utf-8"))
    d31, d32 = P["D31_MIX"], P["D32_MOTOFAR"]
    rows = p27.read_rows(PLAN); pre = p27.read_rows(PRE); man = p27.read_rows(MAN)
    ok = []
    # M1
    # 2026-10-19 #70（v29）: 総数は V29（62,102 = V28 60,502 ＋ D19 1,600）。plan には v28 の積み増し（v19mix）と D19（v11core・d19=1）も入っている
    tot = int(P.get("N_ROWS_TOTAL_EXPECTED_V29") or P.get("N_ROWS_TOTAL_EXPECTED_V28") or P["N_ROWS_TOTAL_EXPECTED_V27"])
    ok.append((f"M1 総行数 {len(rows):,} = N_ROWS_TOTAL_EXPECTED（最新） {tot:,}", len(rows) == tot))
    head = rows[:len(pre)]
    # 2026-10-19 #70 r13（v29）: 雨の dba は新しい範囲で引き直したので rain_dba だけ比べない（雨の有無・種類・rate は同じ）
    cols_cmp = [c for c in p27.COLS_V22 if c != "rain_dba"] if P.get("N_ROWS_TOTAL_EXPECTED_V29") else list(p27.COLS_V22)
    same = len(pre) == p27.N_EXISTING and all(all(h.get(c, "") == r.get(c, "") for c in cols_cmp) and all(h.get(c, "") == "" for c in p27.NEW_COLS) for h, r in zip(head, pre))
    ok.append((f"M1 先頭 {len(pre):,} 行は前の plan と同じ値（新しい列は空{'・v29 では rain_dba だけ引き直し' if P.get('N_ROWS_TOTAL_EXPECTED_V29') else ''}）", same))
    new_all = rows[len(pre):]
    d19 = [r for r in new_all if r.get("d19") == "1"]; new = [r for r in new_all if r.get("d19") != "1"]
    mix = [r for r in new if r["scenario"] == "v19mix"]; moto = [r for r in new if r["scenario"] == "v19motofar"]
    d33 = int((P.get("D33_TOPUP") or {}).get("n_clips", 0)) if (P.get("D33_TOPUP") or {}).get("adopt") else 0
    d19n = int((P.get("D19_SLOW_ADD") or {}).get("n_clips", 0)) if (P.get("D19_SLOW_ADD") or {}).get("adopt") else 0
    ok.append((f"M1 新しい行 {len(new_all):,} = v19mix {len(mix):,}（params D31 {d31['n_clips']:,} ＋ D33 {d33:,}）＋ v19motofar {len(moto):,}（params {d32['n_clips']:,}）＋ D19 {len(d19):,}（params {d19n:,}）",
               len(new_all) == len(mix) + len(moto) + len(d19) and len(mix) == int(d31["n_clips"]) + d33 and len(moto) == int(d32["n_clips"]) and len(d19) == d19n))
    ids = [int(r["clip_id"][-5:]) for r in rows]
    ok.append(("M1 clip_id がユニークで、mix は 50001〜59999／60001〜69999（v28 の積み増しは 72001〜／86001〜）・motofar は 70001〜70999／71001〜71999・D19 は 91001〜92800", len(set(r["clip_id"] for r in rows)) == len(rows)
               and all(((50001 <= int(r["clip_id"][-5:]) < 60000) if r["h_copy"] == "1" else (60001 <= int(r["clip_id"][-5:]) < 70000)) if r.get("scene_type") != "v28"
                       else ((72001 <= int(r["clip_id"][-5:]) < 86001) if r["h_copy"] == "1" else (86001 <= int(r["clip_id"][-5:]) < 91001)) for r in mix)
               and all(91001 <= int(r["clip_id"][-5:]) <= 92800 for r in d19)
               and all((70001 <= int(r["clip_id"][-5:]) < 71000) if r["h_copy"] == "1" else (71001 <= int(r["clip_id"][-5:]) < 72000) for r in moto)))
    by_seed = {}
    for r in new_all:
        by_seed.setdefault(r["seed"], []).append(r)
    ok.append(("M1 同じ種の 2 本は同じ split・同じ scenario・同じ mix・高さ 1 と 2", all(len(v) == 2 and v[0]["split"] == v[1]["split"] and v[0]["scenario"] == v[1]["scenario"] and v[0]["mix"] == v[1]["mix"] and {v[0]["h_copy"], v[1]["h_copy"]} == {"1", "2"} for v in by_seed.values())))
    c = Counter(r["split"] for r in new_all)
    ok.append((f"M1 split: train {c['train']:,} / val {c['val']:,}（val 20% ± 1%）", abs(c["val"] / len(new_all) - 0.2) < 0.01))
    mixes = Counter(r["mix"] for r in mix if r.get("scene_type") != "v28")        # v27 の manifest は v27 の行だけ（v28 の積み増しは manifest_v28_topup.csv）
    ok.append((f"M1 manifest {len(man)} 行（組 {sum(1 for m in man if m['size'] != '1')} ＋ motofar 1）と組ごとの本数が一致",
               all(mixes[m["mix"]] == int(m["n_clips"]) for m in man if m["size"] != "1") and sum(1 for m in man if m["size"] != "1") == len(mixes)
               and sum(int(m["n_clips"]) for m in man if m["size"] == "1") == len(moto)))
    # M2 網羅
    import step11_v27_render as v27
    have = Counter(p27.composition(r) for r in rows)
    floor = int(d31["rules"]["floor_existing"])
    allowed = {k: [tuple(sorted(m, key=v27.TOKENS.index)) for m in itertools.combinations_with_replacement(v27.TOKENS, k) if not v27.mix_ok(m)] for k in (2, 3)}
    missing = {k: [m for m in v for m in [m] if have.get(m, 0) < floor] for k, v in allowed.items()}
    ok.append((f"M2 網羅: 規則に合う 2 つ組 {len(allowed[2])} 種・3 つ組 {len(allowed[3])} 種の全部が最終の plan に {floor} 本以上（足りない: {len(missing[2])} / {len(missing[3])}）",
               len(allowed[2]) == 54 and len(allowed[3]) == 209 and not missing[2] and not missing[3]))
    n_pair = sum(1 for m in man if m["size"] == "2"); n_tri = sum(1 for m in man if m["size"] == "3")
    ok.append((f"M2 足した組 2 つ組 {n_pair}（params {d31['n_patterns_pair']}）・3 つ組 {n_tri}（params {d31['n_patterns_triple']}）・足さなかった組 {len(allowed[2]) - n_pair} ＋ {len(allowed[3]) - n_tri}（元から {floor} 本以上）",
               n_pair == int(d31["n_patterns_pair"]) and n_tri == int(d31["n_patterns_triple"]) and len(allowed[2]) - n_pair == 20 and len(allowed[3]) - n_tri == 19))
    ok.append(("M2 除いた組: train+backup_beep（2 つ組 1・3 つ組 10）と train×3（1）だけ",
               sum(1 for m in itertools.combinations_with_replacement(v27.TOKENS, 2) if v27.mix_ok(m)) == 1
               and sum(1 for m in itertools.combinations_with_replacement(v27.TOKENS, 3) if v27.mix_ok(m)) == 11))
    # M3 サンプラ
    import random
    rnd = random.Random(20261013)
    pick = rnd.sample(mix, 60) + rnd.sample(moto, 20)
    bad = 0
    for r in pick:
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            v27.check_row(v27.sample_scene_v19(rr), rr)
        except AssertionError as e:
            bad += 1; print("  NG", r["clip_id"], e)
    ok.append((f"M3 plan の v19mix 60 本＋v19motofar 20 本を引いて check_row: NG {bad}", bad == 0))
    # M4 ドライバ
    env = {**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONPATH": f"{ROOT / 'scripts'};{ROOT / 'src'}" if sys.platform == "win32" else f"{ROOT / 'scripts'}:{ROOT / 'src'}"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts/_run_v2final_gen.py"), "--list"], capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT))
    out = p.stdout + p.stderr
    tail = "step11_v29_render.py" if P.get("N_ROWS_TOTAL_EXPECTED_V29") else "step11_v27_render.py"
    ok.append((f"M4 _run_v2final_gen.py --list: 総行数 {tot:,}・v19mix {len(mix):,}・v19motofar {len(moto):,}・連なりの末尾 {tail}",
               p.returncode == 0 and f"total rows: {tot}" in out and f"'v19mix': {len(mix)}" in out and f"'v19motofar': {len(moto)}" in out and tail in out))
    if not (p.returncode == 0):
        print(out[-1500:])
    # M5 selftest
    if not quick:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/step11_v27_render.py"), "--selftest"], capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT))
        last = [l for l in p.stdout.splitlines() if l.startswith("selftest v27")]
        ok.append((f"M5 step11_v27_render --selftest: {last[-1] if last else '(出力なし)'}", p.returncode == 0 and last and last[-1].split()[-1].split("/")[0] == last[-1].split()[-1].split("/")[1]))
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-1000:])
    for name, good in ok:
        print(("PASS " if good else "FAIL ") + name)
    n_bad = sum(not g for _, g in ok)
    print(f"test_v2final_mix: {len(ok) - n_bad}/{len(ok)}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
