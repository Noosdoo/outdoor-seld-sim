# -*- coding: utf-8 -*-
"""test_v2final_slow_launch.py — 変更票 #36（徐行を注意帯・安全帯にも = D19、発進 = D20）の実装の検査（2026-09-21・Fable）。

監査 9/21 §9 の受け入れ条件に対応する。params の adopt の状態（本人の決定で変わる）に依らず動くように、
検査ごとに params の写しを作って D19/D20 の adopt を明示的に置く。
  T0 現行 params: --check-v16 --check が通り、総行数 = N_ROWS_TOTAL_EXPECTED。既存 out/…/plan/assignment_v2final.csv と sha256 が一致
     （plan の置き場が params と同期している証拠。一致しなければ plan を作り直すこと）。
  T1 既定 OFF の回帰: D19/D20（と 2026-09-28 の D21）を adopt=false にした写しで plan を作ると、2026-09-14 設計の plan（22,500 行・sha256 SHA_DESIGN_0914）と一致し、
     土台は assignment_v16.csv と一致する。
  T2 D19 ON: 土台の差が対象行（v11core × caution/safe × 住宅街/日常 × L/R）の 4 列だけ・critical 不変・割合 3σ 内・最接近が層の範囲内
     （step10_v2final_plan.check_d19）。critical の close_slow 行は OFF と完全一致。
  T3 D20 ON: 構造の検査（本数・fold・mix 番号 28001〜・同 seed 同 fold・雨/歩行の比率）＋ 発進条件（check_launch: v=0 待機・t_on 節点・
     加速の単調・終端速度・節点間隔 ≤ 0.5 s・x(t_cpa)=マイク x・最接近が t_cpa・無音の通過なし）が全行で違反 0。
  T4 既存 4 場面の回帰: fast/static/turn/platoon のサンプル出力（音源と マイク）が git HEAD の step11_v19_render.py と同一（乱数の消費順が不変）。
  T5 発進の統計: 300 場面で t_on・発進位置・引き直し回数・同方向の割合を出す（対向に固定 0 本を期待）。
出力は scratch（--out <dir>、既定 out/_test_v2final_slow_launch/）。既存の out/…/plan には書かない。
使い方: python scripts/test_v2final_slow_launch.py [--out <dir>] [--skip-head]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v2final_plan as plan  # noqa: E402

PARAMS = ROOT / "md/design/v2final_params.json"
EXISTING = (ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final_pre_v22.csv") if (ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final_pre_v22.csv").exists() else (ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv")   # 2026-10-08 以降は v22／v25 の行が末尾に足されているので、土台＋v19 の 25,500 行は pre_v22 の写しと比べる（2026-10-12）
V16 = ROOT / "out/dataset_outdoor_siren_v16/plan/assignment_v16.csv"
SHA_DESIGN_0914 = "2b1a43020b747ff1692a7d13abd48069881d33fb268a87adb58953d6607659fc"   # 2026-09-14 設計（22,500 行・fast/turn/platoon）の plan
COLS4 = ("close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def with_flags(P0: dict, d19: bool, d20: bool, d21: bool = False) -> dict:
    """params の写しに D19/D20/D21 の adopt を置き、採用中の新場面から総行数と mix_frac を計算し直す。
    D21（キックの遠い横距離・2026-09-28・変更票 #42）は既定 OFF にして、T1〜T3 が 9/21 までの設計（D19/D20 だけ）を見続けるようにする。
    D21 自体の検査は scripts/test_v2final_kickfar.py。"""
    P = json.loads(json.dumps(P0))
    P.setdefault("D19_SLOW_TIERS", {"tiers": ["caution", "safe"], "frac": {"caution": 0.5, "safe": 0.5}})["adopt"] = d19
    P.setdefault("D20_LAUNCH", {"n_scenes": 750, "n_clips": 1500, "train": 1200, "val": 300})["adopt"] = d20
    P.setdefault("D21_KICK_FAR", {"n_scenes": 750, "n_clips": 1500, "train": 1200, "val": 300})["adopt"] = d21
    total = int(P["N_ROWS_EXPECTED"]) + sum(int((P.get(k) or {}).get("n_clips", 0)) for k in plan.SCENE_KEYS if (P.get(k) or {}).get("adopt"))
    P["N_ROWS_TOTAL_EXPECTED"] = total
    for k in plan.SCENE_KEYS:
        d = P.get(k) or {}
        if d.get("adopt"):
            d["mix_frac"] = round(int(d["n_clips"]) / total, 4)
    return P


def run_plan(P: dict, out: Path, args: list) -> tuple[int, list, list]:
    """step10_v2final_plan.main() を出力先 out・params P で走らせる（既存の out/…/plan には書かない）。戻り = (rc, base, all)"""
    out.mkdir(parents=True, exist_ok=True)
    pp = out / "params.json"
    pp.write_text(json.dumps(P, ensure_ascii=False, indent=1), encoding="utf-8")
    plan.OUT, plan.CHAIN = out, out / "chain"
    sys.argv = ["step10_v2final_plan.py", "--params", str(pp)] + args
    rc = plan.main()
    return rc, plan.read_rows(out / "assignment_v2final_base.csv"), plan.read_rows(out / "assignment_v2final.csv")


def t0_current(P0: dict, out: Path) -> list:
    bad = []
    rc, base, allrows = run_plan(P0, out / "t0_current", ["--check-v16", "--check"])
    if rc != 0:
        bad.append(f"T0: 現行 params で --check-v16 --check が rc={rc}")
    if len(allrows) != int(P0["N_ROWS_TOTAL_EXPECTED"]):
        bad.append(f"T0: 総行数 {len(allrows)} ≠ N_ROWS_TOTAL_EXPECTED {P0['N_ROWS_TOTAL_EXPECTED']}")
    if EXISTING.exists():
        if sha(out / "t0_current/assignment_v2final.csv") != sha(EXISTING):
            bad.append("T0: 現行 params の plan が out/…/plan/assignment_v2final.csv と一致しない → step10_v2final_plan.py --check-v16 --check で作り直す")
    else:
        bad.append(f"T0: {EXISTING} が無い")
    on = [k for k in ("D19_SLOW_TIERS", "D20_LAUNCH", "D21_KICK_FAR") if (P0.get(k) or {}).get("adopt")]
    print(f"T0 現行 params（adopt ON: {on or 'なし'}）: 行数 {len(allrows):,}（土台 {len(base):,}）・置き場の plan と{'一致' if not bad else '不一致'}")
    return bad


def t1_off_regression(P0: dict, out: Path) -> list:
    bad = []
    rc, base, allrows = run_plan(with_flags(P0, False, False), out / "t1_off", ["--check-v16", "--check"])
    if rc != 0:
        bad.append(f"T1: --check-v16 --check が rc={rc}")
    # 2026-09-22 改名後: 旧名に戻して書いた sha256 で、改名前に固定した基準と比べる（名前以外が 1 バイトでも違えば不一致）
    if plan.sha256_as_old(out / "t1_off/assignment_v2final.csv") != SHA_DESIGN_0914:
        bad.append("T1: D19/D20 OFF の plan（旧名に戻したもの）が 2026-09-14 設計の plan（sha256 2b1a4302…）と一致しない")
    if V16.exists() and plan.sha256_as_old(out / "t1_off/assignment_v2final_base.csv") != sha(V16):
        bad.append("T1: 土台（旧名に戻したもの）が assignment_v16.csv と一致しない")
    print(f"T1 D19/D20 OFF: 行数 {len(allrows):,}（土台 {len(base):,}）・9/14 設計の plan と{'一致' if not bad else '不一致'}")
    return bad


def t2_d19_on(P0: dict, out: Path, base_off: list) -> list:
    bad = []
    rc, base, _ = run_plan(with_flags(P0, True, False), out / "t2_d19", ["--check"])
    if rc != 0:
        bad.append(f"T2: --check が rc={rc}（check_d19 を含む）")
    off = {r["clip_id"]: r for r in base_off}
    n_crit_cs_same = n_crit = 0
    cnt = Counter()
    for r in base:
        o = off[r["clip_id"]]
        if r["danger_tier"] == "critical":
            n_crit += 1
            n_crit_cs_same += int(all(r[k] == o[k] for k in r))
        elif r["close_slow"] == "1":
            cnt[r["danger_tier"]] += 1
    if n_crit_cs_same != n_crit:
        bad.append(f"T2: critical の行が OFF と違う（{n_crit - n_crit_cs_same}/{n_crit}）")
    spd = [float(r["cs_speed_kmh"]) for r in base if r["close_slow"] == "1" and r["danger_tier"] != "critical"]
    print(f"T2 D19 ON: 徐行に置換 caution {cnt['caution']:,} / safe {cnt['safe']:,} 本・速度 {min(spd):.1f}〜{max(spd):.1f} km/h・critical 不変 {n_crit_cs_same}/{n_crit}")
    return bad


def t3_d20_on(P0: dict, out: Path) -> list:
    bad = []
    P = with_flags(P0, False, True)
    n_new = int(P["D20_LAUNCH"]["n_clips"])
    rc, base, allrows = run_plan(P, out / "t3_d20", ["--check"])
    if rc != 0:
        bad.append(f"T3: --check が rc={rc}（構造 or 発進条件の違反）")
    new = [r for r in allrows if r["scenario"] == "v19launch"]
    mix = sorted(int(r["clip_id"][-5:]) for r in new)
    if len(new) != n_new or not mix or mix[0] != 28001 or mix[-1] != 29750:
        bad.append(f"T3: launch の本数/mix 番号が想定外: {len(new)} 本・mix {mix[0] if mix else None}〜{mix[-1] if mix else None}")
    if len(allrows) != P["N_ROWS_TOTAL_EXPECTED"]:
        bad.append(f"T3: 総行数 {len(allrows)} ≠ {P['N_ROWS_TOTAL_EXPECTED']}")
    print(f"T3 D20 ON: 総行数 {len(allrows):,}・launch {len(new):,} 本・clip {mix[0] if mix else None}〜{mix[-1] if mix else None}・区分 {dict(Counter(r['split'] for r in new))}")
    return bad


T4_BASE_COMMIT = "f55ff7ebd"   # 回帰の基準 = 発進（v19launch）を足す前の最後のコミット（2026-09-19）。監査 9/22 R10: HEAD と比べると実装のコミット後は自分自身との比較になる


def t4_head_regression(out: Path) -> list:
    """既存 4 場面のサンプル出力が、発進を足す前の固定コミット（T4_BASE_COMMIT）の step11_v19_render.py と同一か。"""
    bad = []
    rel = "outdoor_seld_e2e/scripts/step11_v19_render.py"
    try:
        src = subprocess.run(["git", "-C", str(ROOT.parent), "show", f"{T4_BASE_COMMIT}:{rel}"], capture_output=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        return [f"T4: git show に失敗（{e}）"]
    head_path = out / "step11_v19_render_HEAD.py"
    head_path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v19_head", head_path)
    v19_head = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v19_head)                                    # v13.sample_scene_v13 を HEAD 版に差し替える
    import step11_v19_render as v19r                                     # 新版（差し替えを戻す）
    v19r.v13.sample_scene_v13 = v19r.sample_scene_v19
    n_same = n_all = 0
    for scen in ("v19fast", "v19static", "v19turn", "v19platoon"):
        for i in range(60):
            row = v19r._fake_row(scen, i)
            a = v19_head.sample_scene_v19(dict(row))
            b = v19r.sample_scene_v19(dict(row))
            ja = json.dumps({"s": a["sources"], "m": np.asarray(a["_mic_arr"]).tolist(), "n": a["noise"], "v": a["v19"]}, sort_keys=True)
            jb = json.dumps({"s": b["sources"], "m": np.asarray(b["_mic_arr"]).tolist(), "n": b["noise"], "v": b["v19"]}, sort_keys=True)
            n_all += 1
            n_same += int(ja == jb)
    if n_same != n_all:
        bad.append(f"T4: 既存場面のサンプルが HEAD と違う（{n_all - n_same}/{n_all}）")
    print(f"T4 既存 4 場面の回帰（固定コミット {T4_BASE_COMMIT} と比較）: {n_same}/{n_all} 一致")
    return bad


def t5_launch_stats(n: int = 300) -> list:
    import step11_v19_render as v19r
    bad = []
    infos, tiers_cpa = [], Counter()
    for i in range(n):
        row = v19r._fake_row("v19launch", i)
        s = v19r.sample_scene_v19(row)
        src = s["sources"][0]
        try:
            v19r.check_launch(src, s["_mic_arr"])
        except AssertionError as e:
            bad.append(f"T5: {row['clip_id']}: {e}")
        infos.append(s["v19"])
        tiers_cpa[src["danger_tier"]] += 1
    tries = np.array([x["n_tries"] for x in infos])
    forced = sum(x["forced_head_on"] for x in infos)
    same = [x["same_dir_as_walk"] for x in infos if x["same_dir_as_walk"] is not None]
    start = np.array([x["start_dist_m"] for x in infos])
    vend = np.array([x["v_end_kmh"] for x in infos])
    if forced:
        bad.append(f"T5: 対向に固定した場面が {forced} 本（引き直し上限 {v19r.LAUNCH_MAX_TRIES} が足りない）")
    print(f"T5 発進 {n} 場面: t_on 5/50/95% {np.percentile([x['t_on_s'] for x in infos], [5, 50, 95]).round(2).tolist()} s・"
          f"終端速度 {vend.min():.1f}〜{vend.max():.1f} km/h・発進位置の距離 5/50/95% {np.percentile(start, [5, 50, 95]).round(1).tolist()} m")
    print(f"   引き直し 平均 {tries.mean():.2f}・最大 {tries.max()}・対向に固定 {forced} 本・歩行と同方向 {sum(same)}/{len(same)}・層 {dict(tiers_cpa)}")
    return bad


def main() -> int:
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else ROOT / "out/_test_v2final_slow_launch"
    skip_head = "--skip-head" in a
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    P0 = json.loads(PARAMS.read_text(encoding="utf-8"))
    # #70 r13（2026-10-19）: 本番の D2_RAIN_DBA は佐藤ら 2020 の範囲に変わった（新しい plan は step10_v29_plan が雨の dba だけ引き直す）。
    # この試験は「土台の鎖を壊していない」ことを凍結した設計の plan（9/14・9/21・10/8）と比べて見るので、写しの D2 は凍結当時の値（旧 45〜50／50〜58／60〜68）に戻して比べる。
    P0["D2_RAIN_DBA"] = {"light": [45.0, 50.0], "moderate": [50.0, 58.0], "heavy": [60.0, 68.0]}
    bad = []
    bad += t0_current(P0, out)
    bad += t1_off_regression(P0, out)
    base_off = plan.read_rows(out / "t1_off/assignment_v2final_base.csv")
    bad += t2_d19_on(P0, out, base_off)
    bad += t3_d20_on(P0, out)
    bad += t5_launch_stats()
    if not skip_head:
        bad += t4_head_regression(out)
    print("=" * 60)
    print("合格" if not bad else "不合格")
    for b in bad:
        print("  NG", b)
    (out / "result.txt").write_text(("合格" if not bad else "不合格\n" + "\n".join(bad)) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
