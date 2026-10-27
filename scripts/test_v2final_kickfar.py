# -*- coding: utf-8 -*-
"""test_v2final_kickfar.py — 変更票 #42（キックボードの遠い横距離 = D21・v19kickfar）の実装の検査（2026-09-28・Fable）。

params の adopt の状態に依らず動くように、検査ごとに params の写しを作って D21 の adopt を明示的に置く。
  K0 現行 params: --check-v16 --check（kickfar の横距離・最接近の検査を含む）が通り、総行数 = N_ROWS_TOTAL_EXPECTED、
     既存 out/…/plan/assignment_v2final.csv と sha256 が一致（plan の置き場が params と同期している証拠）。
  K1 既定 OFF の回帰: D21 を adopt=false（D19 OFF・D20 ON）にした写しで plan を作ると、2026-09-21 設計の plan（24,000 行・sha256 SHA_DESIGN_0921）
     と一致する。さらに D21 ON の plan の先頭 24,000 行は OFF の plan と 1 セルも違わない（末尾に足すだけ）。
  K2 D21 ON の構造: kickfar 1,500 本・train 1,200 / val 300・clip 30001〜30750（高さ 1）と 31001〜31750（高さ 2）・同 seed の 2 高さが同区分・
     scenario = v19kickfar・n_car = 0。
  K3 サンプラの統計（600 場面）: 音源はキックボード 1 台・横距離 2.5〜8.0 m（対数一様: 中央値が √(2.5·8) = 4.47 m の ±0.4 m）・
     ≥3 m と ≥5 m の割合が対数一様の期待（84% / 40%）の 3σ 内・速さ 1.2〜5.6 m/s・歩道モード 40% の 3σ 内・音量 55〜65 dB(A)@1m・
     最接近の時刻 3〜8 s で、水平距離の最小が t_cpa（±0.02 s）で値が横距離（±0.02 m）— 歩行マイクでも（check_kickfar）。
  K4 既存の場面を壊していない: step11_v21_render を読み込んでも _SAMPLERS19 の既存 5 鍵は同じ関数オブジェクトのまま、v13.sample_scene_v13 は
     v19 の sample_scene_v19 のまま、既存 5 場面のサンプル出力（60 場面ずつ）は読み込み前後で同一。
  K5 生成ドライバ: `_run_v2final_gen.py --list` が v19kickfar 1,500 本・総行数 N_ROWS_TOTAL_EXPECTED を報告し、ドライバの読み込みで
     kickfar のサンプラが登録される（未対応の scenario で止まらない）。
出力は scratch（--out <dir>、既定 out/_test_v2final_kickfar/）。既存の out/…/plan には書かない。
使い方: python scripts/test_v2final_kickfar.py [--out <dir>]
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step10_v2final_plan as plan  # noqa: E402
from test_v2final_slow_launch import run_plan, with_flags  # noqa: E402

PARAMS = ROOT / "md/design/v2final_params.json"
EXISTING = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"
PY = sys.executable
SHA_DESIGN_0921 = "93d028dce6db4207a4ec11edfe1098adc3f363d040160fbc8f93ad6b7c571543"   # 2026-09-21 設計（24,000 行・fast/turn/platoon/launch・新識別子）の plan
KEY = "D21_KICK_FAR"
SCEN = "v19kickfar"
CLIP_H1 = (30001, 30750)
CLIP_H2 = (31001, 31750)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def k0_current(P0: dict, out: Path) -> list:
    bad = []
    rc, base, allrows = run_plan(P0, out / "k0_current", ["--check-v16", "--check"])
    if rc != 0:
        bad.append(f"K0: 現行 params で --check-v16 --check が rc={rc}")
    if len(allrows) != int(P0["N_ROWS_TOTAL_EXPECTED"]):
        bad.append(f"K0: 総行数 {len(allrows)} ≠ N_ROWS_TOTAL_EXPECTED {P0['N_ROWS_TOTAL_EXPECTED']}")
    # 2026-10-08 v22（変更票 #48/#49）: 現行の plan は末尾に 3,524 行が足されているので、土台＋v19 の 25,500 行は v22 前の写し（_pre_v22）と照合する
    existing = EXISTING.with_name("assignment_v2final_pre_v22.csv") if EXISTING.with_name("assignment_v2final_pre_v22.csv").exists() else EXISTING
    if existing.exists():
        if sha(out / "k0_current/assignment_v2final.csv") != sha(existing):
            bad.append(f"K0: 現行 params の plan が {existing.name} と一致しない → step10_v2final_plan.py --check-v16 --check で作り直す")
    else:
        bad.append(f"K0: {EXISTING} が無い")
    on = (P0.get(KEY) or {}).get("adopt")
    print(f"K0 現行 params（{KEY} adopt={on}）: 行数 {len(allrows):,}（土台 {len(base):,}）・置き場の plan と{'一致' if not bad else '不一致'}")
    return bad


def k1_off_regression(P0: dict, out: Path) -> tuple[list, list]:
    bad = []
    rc, _, off = run_plan(with_flags(P0, False, True, False), out / "k1_off", ["--check-v16", "--check"])
    if rc != 0:
        bad.append(f"K1: D21 OFF で --check-v16 --check が rc={rc}")
    if sha(out / "k1_off/assignment_v2final.csv") != SHA_DESIGN_0921:
        bad.append("K1: D21 OFF の plan が 2026-09-21 設計の plan（24,000 行・sha256 93d028dc…）と一致しない")
    if any(r["scenario"] == SCEN for r in off):
        bad.append("K1: D21 OFF なのに v19kickfar の行がある")
    rc2, _, on = run_plan(with_flags(P0, False, True, True), out / "k1_on", ["--check"])
    if rc2 != 0:
        bad.append(f"K1: D21 ON で --check が rc={rc2}")
    head = on[:len(off)]
    if head != off:
        n_diff = sum(a != b for a, b in zip(head, off)) + abs(len(head) - len(off))
        bad.append(f"K1: D21 ON の plan の先頭 {len(off):,} 行が OFF の plan と違う（{n_diff} 行）— 末尾に足すだけのはず")
    print(f"K1 D21 OFF: 行数 {len(off):,}・9/21 設計の plan と{'一致' if sha(out / 'k1_off/assignment_v2final.csv') == SHA_DESIGN_0921 else '不一致'}・"
          f"ON の先頭 {len(off):,} 行は OFF と{'同一' if head == off else '相違'}")
    return bad, on


def k2_structure(P0: dict, on: list) -> list:
    bad = []
    d = with_flags(P0, False, True, True)[KEY]
    new = [r for r in on if r["scenario"] == SCEN]
    c = Counter(r["split"] for r in new)
    if (len(new), c["train"], c["val"]) != (int(d["n_clips"]), int(d["train"]), int(d["val"])):
        bad.append(f"K2: 本数 {len(new)}（train {c['train']} / val {c['val']}）≠ params {d['n_clips']}（{d['train']} / {d['val']}）")
    nums = defaultdict(list)
    for r in new:
        nums[r["h_copy"]].append(int(r["clip_id"][-5:]))
    for h, (lo, hi) in (("1", CLIP_H1), ("2", CLIP_H2)):
        ns = sorted(nums.get(h, []))
        if not ns or ns[0] != lo or ns[-1] != hi or len(ns) != hi - lo + 1:
            bad.append(f"K2: 高さ {h} の clip 番号が {lo}〜{hi} でない: {ns[:1]}〜{ns[-1:]}・{len(ns)} 本")
    by_seed = defaultdict(list)
    for r in new:
        by_seed[r["seed"]].append(r)
    if any(len(v) != 2 for v in by_seed.values()):
        bad.append("K2: seed ごとの高さの写しが 2 本でない")
    if any(len({r["split"] for r in v}) != 1 or len({r["rain"] for r in v}) != 1 for v in by_seed.values()):
        bad.append("K2: 同じ seed の 2 高さで区分か雨が違う")
    if any(r["n_car"] != "0" or r["scene_type"] != "v19" or r["grammar"] != "v19" for r in new):
        bad.append("K2: n_car / scene_type / grammar の列が想定外（n_car=0・scene_type=v19・grammar=v19 のはず）")
    seeds = sorted(int(r["seed"]) for r in new)
    if seeds and (seeds[0] != 9190500000 or seeds[-1] != 9190500749):
        bad.append(f"K2: seed が 9190500000〜9190500749 でない: {seeds[0]}〜{seeds[-1]}")
    print(f"K2 D21 ON の構造: {len(new):,} 本（train {c['train']:,} / val {c['val']:,}）・clip 高さ1 {CLIP_H1[0]}〜{CLIP_H1[1]} / 高さ2 {CLIP_H2[0]}〜{CLIP_H2[1]}・"
          f"雨 {sum(bool(r['rain']) for r in new) / max(len(new), 1):.1%}・歩行 {sum(r['motion'] == 'walk' for r in new) / max(len(new), 1):.1%}")
    return bad


def k3_sampler_stats(n: int = 600) -> list:
    import step11_v19_render as v19r
    import step11_v21_render as v21r
    bad = []
    lat, spd, walkway, n_walkmic = [], [], 0, 0
    for i in range(n):
        row = v19r._fake_row(SCEN, i)
        s = v21r.sample_scene_v19(row)
        if len(s["sources"]) != 1 or s["sources"][0]["class"] != "kickboard":
            bad.append(f"K3: {row['clip_id']} の音源がキックボード 1 台でない"); continue
        src = s["sources"][0]
        try:
            v21r.check_kickfar(src, s["_mic_arr"])
        except AssertionError as e:
            bad.append(f"K3: {row['clip_id']}: {e}")
        lat.append(float(src["lateral_m"])); spd.append(float(src["speed_mps"]))
        walkway += int(s["v19"]["walkway"]); n_walkmic += int(row["motion"] == "walk")
        if abs(abs(float(src["wp"][0][2])) - float(src["lateral_m"])) > 1e-6:
            bad.append(f"K3: {row['clip_id']} の wp の y と lateral_m が違う")
    lat, spd = np.array(lat), np.array(spd)
    lo, hi = v21r.KICKFAR_LAT
    med = float(np.median(lat))
    if abs(med - math.sqrt(lo * hi)) > 0.4:
        bad.append(f"K3: 横距離の中央値 {med:.2f} m が対数一様の期待 {math.sqrt(lo * hi):.2f} m から 0.4 m 超えて離れている")
    for th in (3.0, 5.0):
        p = (math.log(hi) - math.log(th)) / (math.log(hi) - math.log(lo))
        f = float(np.mean(lat >= th)); sd = math.sqrt(p * (1 - p) / n)
        if abs(f - p) > 3 * sd:
            bad.append(f"K3: 横距離 ≥{th} m の割合 {f:.3f} が対数一様の期待 {p:.3f} から 3σ（{3 * sd:.3f}）超えて離れている")
    p = v21r.KICKFAR_P_WALKWAY; sd = math.sqrt(p * (1 - p) / n)
    if abs(walkway / n - p) > 3 * sd:
        bad.append(f"K3: 歩道モードの割合 {walkway / n:.3f} が {p} から 3σ 超えて離れている")
    print(f"K3 サンプラ {n} 場面（歩行マイク {n_walkmic}）: 横距離 5/50/95% {np.percentile(lat, [5, 50, 95]).round(2).tolist()} m・≥3 m {np.mean(lat >= 3):.0%}・≥5 m {np.mean(lat >= 5):.0%}・"
          f"速さ {spd.min() * 3.6:.1f}〜{spd.max() * 3.6:.1f} km/h・歩道モード {walkway / n:.0%}・違反 {len(bad)}")
    return bad


def k4_existing_unchanged() -> list:
    bad = []
    import step11_v19_render as v19r
    before = {k: v19r._SAMPLERS19[k] for k in ("v19fast", "v19static", "v19turn", "v19platoon", "v19launch")}
    if set(before) - set(v19r._SAMPLERS19):
        return ["K4: v19 の既存 5 場面が _SAMPLERS19 に無い"]
    outs_before = {}
    for scen in before:
        for i in range(60):
            s = v19r.sample_scene_v19(v19r._fake_row(scen, i))
            outs_before[(scen, i)] = json.dumps({"s": s["sources"], "m": np.asarray(s["_mic_arr"]).tolist(), "n": s["noise"], "v": s["v19"]}, sort_keys=True)
    import step11_v21_render as v21r  # noqa: F401
    for k, fn in before.items():
        if v19r._SAMPLERS19.get(k) is not fn:
            bad.append(f"K4: 既存の場面 {k} の関数が差し替わっている")
    if v19r.v13.sample_scene_v13 is not v19r.sample_scene_v19:
        bad.append("K4: v13.sample_scene_v13 が v19 の sample_scene_v19 でない")
    if SCEN not in v19r._SAMPLERS19:
        bad.append("K4: v19kickfar が登録されていない")
    n_same = 0
    for (scen, i), ja in outs_before.items():
        s = v19r.sample_scene_v19(v19r._fake_row(scen, i))
        jb = json.dumps({"s": s["sources"], "m": np.asarray(s["_mic_arr"]).tolist(), "n": s["noise"], "v": s["v19"]}, sort_keys=True)
        n_same += int(ja == jb)
    if n_same != len(outs_before):
        bad.append(f"K4: 既存 5 場面のサンプルが v21 読み込み前後で違う（{len(outs_before) - n_same}/{len(outs_before)}）")
    print(f"K4 既存 5 場面: 関数 {'不変' if not bad else '変化'}・サンプル {n_same}/{len(outs_before)} 一致・登録 {sorted(v19r._SAMPLERS19)}")
    return bad


def k5_driver_list(P0: dict) -> list:
    bad = []
    r = subprocess.run([PY, str(ROOT / "scripts/_run_v2final_gen.py"), "--list"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return [f"K5: _run_v2final_gen.py --list が rc={r.returncode}: {out[-400:]}"]
    n_total = int(P0.get("N_ROWS_TOTAL_EXPECTED_V29") or P0.get("N_ROWS_TOTAL_EXPECTED_V28") or P0.get("N_ROWS_TOTAL_EXPECTED_V27") or P0.get("N_ROWS_TOTAL_EXPECTED_V25") or P0.get("N_ROWS_TOTAL_EXPECTED_V22") or P0["N_ROWS_TOTAL_EXPECTED"])   # v27（#64・2026-10-13）→ v25（大型車）→ v22 → 土台（Opus 10/13 L-1）   # 2026-10-19: 最新の総数（V29 → V28 → V27）
    want_scen = f"'{SCEN}': {int((P0.get(KEY) or {}).get('n_clips', 0))}" if (P0.get(KEY) or {}).get("adopt") else None
    if f"total rows: {n_total}" not in out:
        bad.append(f"K5: --list の総行数が {n_total} でない: {out.splitlines()[0] if out else ''}")
    if want_scen and want_scen not in out:
        bad.append(f"K5: --list に {want_scen} が無い: {out[-300:]}")
    code = ("import sys; sys.path.insert(0, r'%s'); import _run_v2final_gen as g; import step11_v19_render as v; "
            "print('kickfar' if '%s' in v._SAMPLERS19 else 'MISSING')" % (str(ROOT / "scripts"), SCEN))
    r2 = subprocess.run([PY, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    if "kickfar" not in (r2.stdout or ""):
        bad.append(f"K5: ドライバの読み込みで v19kickfar が登録されない: {(r2.stdout or '') + (r2.stderr or '')[-300:]}")
    print(f"K5 生成ドライバ --list: {'OK' if not bad else 'NG'}（{out.splitlines()[0] if out else ''}）")
    return bad


def main() -> int:
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else ROOT / "out/_test_v2final_kickfar"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    P0 = json.loads(PARAMS.read_text(encoding="utf-8"))
    # #70 r13（2026-10-19）: 本番の D2_RAIN_DBA は佐藤ら 2020 の範囲に変わった（新しい plan は step10_v29_plan が雨の dba だけ引き直す）。
    # この試験は「土台の鎖を壊していない」ことを凍結した設計の plan（9/14・9/21・10/8）と比べて見るので、写しの D2 は凍結当時の値（旧 45〜50／50〜58／60〜68）に戻して比べる。
    P0["D2_RAIN_DBA"] = {"light": [45.0, 50.0], "moderate": [50.0, 58.0], "heavy": [60.0, 68.0]}
    bad = []
    bad += k4_existing_unchanged()          # v21 を読む前の状態を先に取る
    bad += k0_current(P0, out)
    b1, on = k1_off_regression(P0, out)
    bad += b1
    bad += k2_structure(P0, on)
    bad += k3_sampler_stats()
    bad += k5_driver_list(P0)
    print("=" * 60)
    print("合格" if not bad else "不合格")
    for b in bad:
        print("  NG", b)
    (out / "result.txt").write_text(("合格" if not bad else "不合格\n" + "\n".join(bad)) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
