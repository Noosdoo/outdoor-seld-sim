# -*- coding: utf-8 -*-
"""背景の形の小さな比べ（決め直し 5・仕様 §9-7 の 6）の受け入れ試験（2026-10-26）。学習の本の試し描きと架空の正解・予測だけを使う。

  K1 幅の旗（乱数の鍵は同じ・幅だけ）: 同じ本で、傾きの幅を (ii)（−6.5〜−2.5）にすると 傾き = −6.5 + (傾き_(i) + 5) ÷ 2 × 4（同じ U）・折れ点は同じ・
     既定の幅の値は旗を足す前の draw_shape と同じ・返す辞書に幅（knee_range_hz・slope_range_db_oct）
  K2 環境変数（2026-10-27 に既定が (ii) になった）: 無し → bg_shape 入・幅 (−6.5, −2.5)／V30_BG_SHAPE=0 → 切／V30_BG_SHAPE=1・V30_BG_SLOPE_RANGE=-5,-3 → 入・(i) の幅
  K3 描画に反映: 同じ本を off／(i)／(ii) で描くと、off は scene に背景の形が無い・(i) と (ii) は shape に引いた値と幅・(ii) の傾きは K1 の写像どおり・
     帯域ゲート（v30_band）はどれも入
  K4 生成ドライバ: V2FINAL_RENDER=v30・V2FINAL_ROWS_LIST（2 本）で回すと、一覧の 2 本だけが描かれる（ほかの行は描かない）・GEN_VERSIONS に
     「render: v30 FEATURES=…」の行・ROWS_LISTS.log に一覧の名前と sha・select_rows は plan の順・一覧に plan に無い本や重複があれば止まる
  K5 部分集合: 道具の選び方をもう一度回すと、コミットした一覧（md/design/v30_bgcmp_本の一覧_2026-10-26.csv）と同じ・12,000 本・高さの双子の組は両方・
     scenario の数は train 全体に比例（各 ±1 組）
  K6 決め方: 最大を採る／1 pt 以内が複数で ii を含めば ii／ii を含まなければ最大／ちょうど 1 pt は以内／同点は ii → i → off／欠け・非有限は止まる
  K7 命令: 架空の正解と 3 通りの予測で decision.json（規則・未丸めの F・クラス別・分岐・最終）。家の 13 本の数を入れ替えても決定は同じ（参考の欄に入るだけ）
使い方: DynamicSound の venv で python scripts/test_v30_bgcmp.py（5 本ほど描くので 2〜3 分）
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RESULTS = []
CLIP = "train_syn_clip11266"            # 車だけの本（帯域ゲートの試験と同じ）
LIST = ROOT / "md/design/v30_bgcmp_本の一覧_2026-10-26.csv"


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def stops(fn, exc=(ValueError, AssertionError, SystemExit)):
    try:
        fn(); return False
    except exc:
        return True


def main() -> int:
    import _run_v2final_gen as g
    import step11_v30_render as v30
    from outdoor_seld import noise_v30 as NV
    plan = {r["clip_id"]: r for r in g.load_plan()}
    row = plan[CLIP]
    # K1
    rng_key = lambda: np.random.default_rng([int(row["seed"]), v30.v29.SALT29, v30.V30_KEYS["bg_shape"]])
    prev = (v30.BG_KNEE_RANGE, v30.BG_SLOPE_RANGE, v30.FEATURES["bg_shape"])
    try:
        v30.BG_KNEE_RANGE, v30.BG_SLOPE_RANGE = (63.0, 1000.0), (-5.0, -3.0)
        s_i = v30.bg_shape_of({}, row)
        old = NV.draw_shape(rng_key())
        v30.BG_SLOPE_RANGE = (-6.5, -2.5)
        s_ii = v30.bg_shape_of({}, row)
        mapped = -6.5 + (s_i["slope_db_oct"] + 5.0) / 2.0 * 4.0
        check("K1 幅の旗: (ii) の傾き = (i) の傾きの写像（同じ U）・折れ点は同じ・既定の幅の値は旗の前と同じ・辞書に幅",
              abs(s_ii["slope_db_oct"] - mapped) < 1e-9 and s_ii["knee_hz"] == s_i["knee_hz"] and (s_i["knee_hz"], s_i["slope_db_oct"]) == (old["knee_hz"], old["slope_db_oct"])
              and s_ii["slope_range_db_oct"] == [-6.5, -2.5] and s_i["knee_range_hz"] == [63.0, 1000.0],
              f"(i) {s_i['slope_db_oct']:.4f}・(ii) {s_ii['slope_db_oct']:.4f}（写像 {mapped:.4f}）・折れ点 {s_i['knee_hz']:.1f}")
    finally:
        v30.BG_KNEE_RANGE, v30.BG_SLOPE_RANGE, v30.FEATURES["bg_shape"] = prev
    # K2
    env0 = {k: v for k, v in os.environ.items() if not k.startswith(("V30_BG", "V30_GATE"))}
    env0["PYTHONPATH"] = os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "src")])
    code = "import step11_v30_render as v;print(v.FEATURES['bg_shape'], list(v.BG_SLOPE_RANGE), list(v.BG_KNEE_RANGE))"
    r_def = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env0, cwd=str(ROOT))
    r_off = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(env0, V30_BG_SHAPE="0"), cwd=str(ROOT))
    r_i = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(env0, V30_BG_SHAPE="1", V30_BG_SLOPE_RANGE="-5,-3"), cwd=str(ROOT))
    r_bad = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(env0, V30_BG_SHAPE="yes"), cwd=str(ROOT))
    check("K2 環境変数（10/27 から既定は (ii)）: 無し → 入・(−6.5, −2.5)／V30_BG_SHAPE=0 → 切／=1 と -5,-3 → 入・(i) の幅／0・1 以外は止める",
          r_def.stdout.strip() == "True [-6.5, -2.5] [63.0, 1000.0]" and r_off.stdout.strip() == "False [-6.5, -2.5] [63.0, 1000.0]"
          and r_i.stdout.strip() == "True [-5.0, -3.0] [63.0, 1000.0]" and r_bad.returncode != 0,
          f"{r_def.stdout.strip()}｜{r_off.stdout.strip()}｜{r_i.stdout.strip()}｜bad rc {r_bad.returncode}")
    # K3
    tmp = Path(tempfile.mkdtemp(prefix="bgcmp_"))
    try:
        sc = {}
        for name, on, sr in (("off", False, (-5.0, -3.0)), ("i", True, (-5.0, -3.0)), ("ii", True, (-6.5, -2.5))):
            v30.FEATURES["bg_shape"], v30.BG_SLOPE_RANGE = on, sr
            g.point_to(tmp / name)
            v30.generate_clip(dict(row))
            sc[name] = json.loads((tmp / name / "work" / CLIP / "scene.json").read_text(encoding="utf-8"))
        v30.BG_KNEE_RANGE, v30.BG_SLOPE_RANGE, v30.FEATURES["bg_shape"] = prev
        sh = {k: (v.get("noise") or {}).get("shape") for k, v in sc.items()}
        check("K3 描画に反映: off は形なし・(i)／(ii) は引いた値と幅・(ii) の傾きは写像どおり・帯域ゲートはどれも入",
              sh["off"] is None and sh["i"]["slope_range_db_oct"] == [-5.0, -3.0] and sh["ii"]["slope_range_db_oct"] == [-6.5, -2.5]
              and abs(sh["ii"]["slope_db_oct"] - round(-6.5 + (sh["i"]["slope_db_oct"] + 5.0) / 2.0 * 4.0, 4)) < 2e-4 and sh["i"]["knee_hz"] == sh["ii"]["knee_hz"]
              and all(v.get("gate_rule") == "v30_band" for v in sc.values()), str(sh))
    finally:
        v30.BG_KNEE_RANGE, v30.BG_SLOPE_RANGE, v30.FEATURES["bg_shape"] = prev
        shutil.rmtree(tmp, ignore_errors=True)
    # K4
    pick = ["train_syn_clip11266", "train_syn_clip00002"]
    sel = g.select_rows(list(plan.values()), str(_write_list(pick[::-1])))
    ok_order = [r["clip_id"] for r in sel] == sorted(pick, key=lambda c: list(plan).index(c))
    bad_missing = stops(lambda: g.select_rows(list(plan.values()), str(_write_list(["train_syn_clip99999"]))))
    bad_dup = stops(lambda: g.select_rows(list(plan.values()), str(_write_list([pick[0], pick[0]]))))
    dsn = "dataset_outdoor_siren_v2final_v30bgtest_k4"
    dsp = ROOT / "out" / dsn
    shutil.rmtree(dsp, ignore_errors=True)
    try:
        (dsp / "plan").mkdir(parents=True)
        shutil.copyfile(ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv", dsp / "plan/assignment_v2final.csv")
        lst = _write_list(pick)
        env = dict(env0, V2FINAL_DS_NAME=dsn, V2FINAL_RENDER="v30", V2FINAL_ROWS_LIST=str(lst), PYTHONUTF8="1")
        rk = subprocess.run([sys.executable, str(ROOT / "scripts/_run_v2final_gen.py"), "--shard", "0/1"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
        drawn = sorted(p.stem for p in (dsp / "foa").glob("*.flac")) if (dsp / "foa").is_dir() else []
        gv = (dsp / "GEN_VERSIONS.txt").read_text(encoding="utf-8") if (dsp / "GEN_VERSIONS.txt").is_file() else ""
        lg = (dsp / "ROWS_LISTS.log").read_text(encoding="utf-8") if (dsp / "ROWS_LISTS.log").is_file() else ""
        check("K4 生成ドライバ: 一覧の 2 本だけ描く・GEN_VERSIONS に render: v30 の行（一覧は入れない）・ROWS_LISTS.log に一覧と sha・plan の順・無い本と重複は止まる",
              rk.returncode == 0 and drawn == sorted(pick) and "render: v30 FEATURES=" in gv and "rows list" not in gv and str(lst) in lg and ok_order and bad_missing and bad_dup,
              f"rc {rk.returncode}・描いた {drawn}・{rk.stderr[-200:] if rk.returncode else ''}")
    finally:
        shutil.rmtree(dsp, ignore_errors=True)
    # K5
    import v30_bgcmp_subset as SUB
    rows = list(csv.DictReader(open(ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv", encoding="utf-8")))
    s1 = SUB.select(rows, 12000)
    committed = [r["clip_id"] for r in csv.DictReader(open(LIST, encoding="utf-8"))]
    seeds = Counter(r["seed"] for r in s1)
    tr = Counter(r["scenario"] for r in rows if r["split"] == "train")
    sc_ = Counter(r["scenario"] for r in s1)
    prop_ok = all(abs(sc_[k] - 12000 * tr[k] / sum(tr.values())) <= 2.0 for k in tr)
    check("K5 部分集合: もう一度選ぶとコミットした一覧と同じ・12,000 本・双子の組は両方・scenario は train 全体に比例（各 ±1 組）",
          [r["clip_id"] for r in s1] == committed and len(s1) == 12000 and all(v == 2 for v in seeds.values()) and prop_ok and all(r["split"] == "train" for r in s1),
          f"{len(s1)} 本・組 {len(seeds)}")
    # K6
    import v30_bgcmp_decide as D
    c = {"max": D.decide({"off": 0.50, "i": 0.55, "ii": 0.52})["final"],
         "near_with_ii": D.decide({"off": 0.50, "i": 0.55, "ii": 0.545})["final"],
         "near_without_ii": D.decide({"off": 0.548, "i": 0.55, "ii": 0.50})["final"],
         "exact_1pt": D.decide({"off": 0.50, "i": 0.56, "ii": 0.55})["final"],
         "tie": D.decide({"off": 0.55, "i": 0.55, "ii": 0.40})["final"],
         "missing": stops(lambda: D.decide({"off": 0.5, "i": 0.5})),
         "nonfinite": stops(lambda: D.decide({"off": 0.5, "i": float("nan"), "ii": 0.5}))}
    check("K6 決め方: 最大／1 pt 以内に ii → ii／ii なし → 最大／ちょうど 1 pt は以内／同点は ii → i → off／欠け・非有限は止まる",
          c == {"max": "i", "near_with_ii": "ii", "near_without_ii": "i", "exact_1pt": "ii", "tie": "i", "missing": True, "nonfinite": True}, str(c))
    # K7
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "gt").mkdir()
        clips = ["val_syn_clip00012", "val_syn_clip00024"]
        (td / "gt" / f"{clips[0]}.csv").write_text("".join(f"{k},4,0,30,0,2.00\n" for k in range(40, 100)), encoding="utf-8")
        (td / "gt" / f"{clips[1]}.csv").write_text("", encoding="utf-8")
        with open(td / "man.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["clip_id"]); w.writerows([[x] for x in clips])

        def pred(name, ks):
            (td / name).write_text("".join(f"{clips[0]},{k},4,0,30,0,2.0\n" for k in ks), encoding="utf-8")
        pred("off.csv", range(40, 80)); pred("i.csv", range(40, 100)); pred("ii.csv", range(40, 99))   # F = 0.8／1.0／118÷119（ii は i から 1 pt 以内）
        outs = []
        for h in (["off=13", "i=0", "ii=0"], ["off=0", "i=0", "ii=13"]):
            r = subprocess.run([sys.executable, str(ROOT / "scripts/v30_bgcmp_decide.py"), "--gt", str(td / "gt"), "--manifest", str(td / "man.csv"),
                                "--pred", f"off={td / 'off.csv'}", f"i={td / 'i.csv'}", f"ii={td / 'ii.csv'}", "--home13", *h, "--out", str(td / "d.json")],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            outs.append(json.loads((td / "d.json").read_text(encoding="utf-8")) if r.returncode == 0 else {"err": r.stderr[-300:]})
        d0, d1 = outs
        fi, fii = d0.get("variants", {}).get("i", {}).get("mean_f"), d0.get("variants", {}).get("ii", {}).get("mean_f")
        check("K7 命令: decision.json（規則・未丸めの F・クラス別・分岐・最終）・家の 13 本を入れ替えても決定は同じ（参考の欄だけ変わる）",
              "rule" in d0 and d0.get("decision") == d1.get("decision") and d0["decision"]["final"] == "ii" and abs(fi - 1.0) < 1e-12 and abs(fii - 118 / 119) < 1e-12
              and d0["reference_home13_rang"] != d1["reference_home13_rang"],
              str(d0.get("decision") or d0))
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


_TMPL = []


def _write_list(ids):
    d = Path(tempfile.mkdtemp(prefix="bglist_"))
    _TMPL.append(d)
    p = d / "list.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_id"]); w.writerows([[x] for x in ids])
    return p


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        for d in _TMPL:
            shutil.rmtree(d, ignore_errors=True)
    sys.exit(rc)
