# -*- coding: utf-8 -*-
"""本体 §3-5（検出のしきい値・§9-1 の 5・R30-05・§9-4 の 5）の受入れ試験（2026-10-26）。

  T1 クラス別のヒステリシス（hysteresis_patch）: 配列の入り／保持の列 c = 定数 enter[c]／stay[c] で掛けた列 c（ランダムな矢印 5 組）・全クラス同じ配列 = 定数の出力・
     保持 = 入り（nohyst）= norm > 入り・apply_with_start（start 40）も配列で同じ・norms の式
  T2 選び方（v30_select_thresholds）: 架空の val 48 本（選択用 4・確認用 4）・9 クラスで、同点は小さい方（0.3）・0.7 だけが F 1・0.1・
     フレーム 0〜39 の大きな norm は数えない（0.1）・確認用だけ違う norm でも入りは変わらない（0.5・確認用の F は併記だけ）→ 入り [0.3, 0.7, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1]・
     保持 = round(入り × 3/7, 4)・入力の sha（norms_manifest・ckpt）・書いた表が threshold_table.load を通る・同じ所に 2 回は止まる
  T3 止める所（選び方）: 選択用の norms が 1 本無い・選択用にクラスの正解が無い・norms_manifest が無い・norms の形が違う
  T4 表の確かめ（threshold_table.load）: 版・クラス・候補に無い入り（0.65）・保持の比・非有限・表を作った ckpt と違う → 止まる・nohyst は保持 = 入り
  T5 推論の入口（_causal_infer_v17.py の文面）: CAUSAL_THRESHOLDS_JSON／MODE・表と CAUSAL_SED_TH／STAY の同時は止める・MODE だけも止める・表は ckpt の sha で確かめる・
     判定は apply_with_start(…, ENTER, STAY, …, start=HYST_START)・run_meta に thresholds（定数の欄は None）と dump_norms（使うときだけ）・
     DUMP_NORMS は .part → 置き換え・最後に欠けを確かめて norms_manifest.json・表の sha が途中で変われば止める・code_paths に threshold_table（使うときだけ）
  T6 契約（_conf_exec_contract --thresholds-json）: 正しい表 → 通り記録に thresholds（sha・入り）／別の ckpt の表 → 3／壊れた表 → 3
使い方: python scripts/test_v30_thresholds.py（torch 不要）
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_sde"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import hysteresis_patch as HP  # noqa: E402
import threshold_table as TT  # noqa: E402
import v30_select_thresholds as VS  # noqa: E402
import _conf_exec_contract as CC  # noqa: E402

RESULTS = []
C, T = 9, 100


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def rand_doa(seed: int) -> np.ndarray:
    """方位がゆっくり動き、長さが 0〜1 を行き来する矢印（ヒステリシスが効く場面を作る）。"""
    rng = np.random.default_rng(seed)
    doa = np.zeros((3, T, 4 * C), dtype=np.float32)
    for k in range(3):
        for c in range(C):
            az = np.cumsum(rng.normal(0, 6, T)) + rng.uniform(-180, 180)
            ln = np.clip(0.5 + 0.45 * np.sin(np.linspace(0, rng.uniform(2, 9), T) + rng.uniform(0, 6)) + rng.normal(0, 0.08, T), 0, 1)
            doa[k, :, c] = ln * np.cos(np.radians(az)); doa[k, :, C + c] = ln * np.sin(np.radians(az)); doa[k, :, 2 * C + c] = rng.normal(0, 0.05, T)
            doa[k, :, 3 * C + c] = rng.uniform(0, 1, T)
    return doa


def stops(f, *a, exc=(SystemExit,)):
    try:
        f(*a)
        return False
    except exc:
        return True


def make_val(root: Path, check_variant: bool = False) -> tuple[Path, Path, Path]:
    gt, run = root / "gt", root / "run"
    gt.mkdir(parents=True); (run / "norms").mkdir(parents=True)
    clips = [f"val_syn_clip{i:05d}" for i in range(48)]
    with open(root / "man.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_id"]); w.writerows([[c] for c in clips])
    files = {}
    for cl in clips:
        i = int(cl[-5:])
        is_chk = i % 12 == 6
        (gt / f"{cl}.csv").write_text("".join(f"{k},{c},0,30,0,2.00\n" for k in range(50, 80) for c in range(C)), encoding="utf-8")
        n = np.zeros((3, T, C), dtype=np.float32)
        spec = {0: (0.65, 0.25), 1: (0.75, 0.65), 2: (0.15, 0.0), 3: (0.45, 0.0), 4: ((0.35 if (is_chk and check_variant) else 0.55), 0.45)}
        for c in range(C):
            a, b = spec.get(c, (0.8, 0.0))
            n[0, 40:, c] = b
            n[0, 50:80, c] = a
        n[1, 0:40, 3] = 0.9                       # 暖機の 0〜39 の大きな反応（数えない）
        np.save(run / "norms" / f"{cl}.npy", n)
        files[cl] = "x" * 64
    (run / "norms_manifest.json").write_text(json.dumps({"ckpt_sha256": "c" * 64, "n_clips": len(clips), "files": files}), encoding="utf-8")
    return gt, run, root / "man.csv"


def main() -> int:
    # T1
    ens = np.array([0.7, 0.5, 0.3, 0.6, 0.7, 0.4, 0.2, 0.1, 0.7]); sts = np.array([TT.stay_of(e) for e in ens])
    ok1 = True; det = ""
    for sd in range(5):
        doa = rand_doa(sd)
        pc = HP.apply(doa, C, ens, sts)
        col = all(np.array_equal(pc[:, :, c], HP.apply(doa, C, float(ens[c]), float(sts[c]))[:, :, c]) for c in range(C))
        same = np.array_equal(HP.apply(doa, C, np.full(C, 0.7), np.full(C, 0.3)), HP.apply(doa, C, 0.7, 0.3))
        nh = np.array_equal(HP.apply(doa, C, ens, ens), HP.norms(doa, C).astype(np.float64) > ens[None, None, :])
        sed_in = HP.norms(doa, C).astype(np.float64) > ens[None, None, :]
        ws = HP.apply_with_start(doa, sed_in, C, ens, sts, start=40)
        wsk = (not ws[:, :40].any()) and np.array_equal(ws[:, 40:], HP.apply(doa[:, 40:], C, ens, sts))
        diff = int((pc != (HP.norms(doa, C) > ens[None, None, :])).sum())          # ヒステリシスが実際に効いている（入りだけと違う枠がある）
        ok1 &= col and same and nh and wsk and diff > 0
        det += f"[{sd}: 列 {col}・同じ {same}・nohyst {nh}・start {wsk}・効いた枠 {diff}]"
    nr = HP.norms(rand_doa(9), C)
    check("T1 クラス別のヒステリシス: 列 c = 定数 enter[c]／stay[c] の列 c・全クラス同じ = 定数・保持 = 入りは norm > 入り・start 40 も同じ・norms は float32 (3,T,C)",
          ok1 and nr.dtype == np.float32 and nr.shape == (3, T, C) and HP.has_hysteresis(ens, sts) and not HP.has_hysteresis(ens, ens), det[:400])
    tmp = Path(tempfile.mkdtemp(prefix="thr_v30_"))
    try:
        # T2
        gt, run, man = make_val(tmp / "a")
        out = tmp / "a" / "thresholds_v30.json"
        rc = VS.main(["--gt", str(gt), "--manifest", str(man), "--run", str(run), "--label", "main", "--out", str(out)])
        J = json.loads(out.read_text(encoding="utf-8"))
        want = [0.3, 0.7, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
        gt2, run2, man2 = make_val(tmp / "b", check_variant=True)
        out2 = tmp / "b" / "thresholds_v30.json"
        VS.main(["--gt", str(gt2), "--manifest", str(man2), "--run", str(run2), "--label", "main", "--out", str(out2)])
        J2 = json.loads(out2.read_text(encoding="utf-8"))
        t = TT.load(out, 9, "hyst", ckpt_sha="c" * 64)
        check("T2 選び方: 入り [0.3, 0.7, 0.1, 0.1, 0.5, 0.1×4]（同点は小さい方・暖機は数えない）・確認用だけ違っても入りは同じ（確認用の F は下がる）・保持 = 3/7・入力の sha・load を通る・2 回目は止まる",
              rc == 0 and J["enter"] == want and J["stay"] == [TT.stay_of(e) for e in want] and J2["enter"] == want
              and J["per_class"]["4"]["f_check"] == 1.0 and J2["per_class"]["4"]["f_check"] < 1.0 and J["select"] == {"rule": "clip%12==0", "n": 4} and J["check"]["n"] == 4
              and J["inputs"]["ckpt_sha256"] == "c" * 64 and len(J["inputs"]["norms_manifest_sha256"]) == 64 and J["frames"] == [40, 99] and J["version"] == "thresholds_v30/1"
              and list(t["enter"]) == want and J["mean_f_select"] == 1.0 and J["reference_constant_0.7_in_only"]["mean_f_select"] < 1.0
              and stops(VS.main, ["--gt", str(gt), "--manifest", str(man), "--run", str(run), "--label", "main", "--out", str(out)]),
              f"{J['enter']}・確認用 F(4) {J2['per_class']['4']['f_check']}")
        # T3
        r3 = {}
        g3, run3, m3 = make_val(tmp / "c"); (run3 / "norms" / "val_syn_clip00012.npy").unlink()
        r3["norms_missing"] = stops(VS.main, ["--gt", str(g3), "--manifest", str(m3), "--run", str(run3), "--label", "main", "--out", str(tmp / "c" / "o.json")])
        g4, run4, m4 = make_val(tmp / "d")
        for i in range(0, 48, 12):
            p = g4 / f"val_syn_clip{i:05d}.csv"
            p.write_text("".join(l + "\n" for l in p.read_text(encoding="utf-8").splitlines() if l.split(",")[1] != "8"), encoding="utf-8")
        r3["class_no_gt"] = stops(VS.main, ["--gt", str(g4), "--manifest", str(m4), "--run", str(run4), "--label", "main", "--out", str(tmp / "d" / "o.json")])
        g5, run5, m5 = make_val(tmp / "e"); (run5 / "norms_manifest.json").unlink()
        r3["no_manifest"] = stops(VS.main, ["--gt", str(g5), "--manifest", str(m5), "--run", str(run5), "--label", "main", "--out", str(tmp / "e" / "o.json")])
        g6, run6, m6 = make_val(tmp / "f"); np.save(run6 / "norms" / "val_syn_clip00024.npy", np.zeros((3, 99, 9), dtype=np.float32))
        r3["bad_shape"] = stops(VS.main, ["--gt", str(g6), "--manifest", str(m6), "--run", str(run6), "--label", "main", "--out", str(tmp / "f" / "o.json")])
        g7, run7, m7 = make_val(tmp / "g")
        sl, cl_ = tmp / "g" / "sel.csv", tmp / "g" / "chk.csv"
        sl.write_text("clip_id\n" + "".join(f"val_syn_clip{i:05d}\n" for i in range(0, 48, 12)), encoding="utf-8")
        cl_.write_text("clip_id\n" + "".join(f"val_syn_clip{i:05d}\n" for i in range(6, 48, 12)), encoding="utf-8")
        rc_lists = VS.main(["--gt", str(g7), "--manifest", str(m7), "--run", str(run7), "--label", "main", "--out", str(tmp / "g" / "o.json"), "--expect-lists", str(sl), str(cl_)])
        cl_.write_text("clip_id\n" + "".join(f"val_syn_clip{i:05d}\n" for i in range(6, 36, 12)), encoding="utf-8")
        r3["lists_differ"] = stops(VS.main, ["--gt", str(g7), "--manifest", str(m7), "--run", str(run7), "--label", "main", "--out", str(tmp / "g" / "o2.json"), "--expect-lists", str(sl), str(cl_)])
        check("T3 止める（選び方）: 選択用の norms が無い・選択用にクラスの正解が無い・norms_manifest が無い・norms の形が違う・plan の名簿と違う（同じなら通る）",
              all(r3.values()) and len(r3) == 5 and rc_lists == 0, str(r3))
        # T4
        def variant(name, fn):
            d = json.loads(out.read_text(encoding="utf-8")); fn(d)
            p = tmp / f"t4_{name}.json"; p.write_text(json.dumps(d), encoding="utf-8"); return p
        E = TT.ThresholdTableError
        r4 = {"version": stops(TT.load, variant("v", lambda d: d.update(version="thresholds_v29/1")), 9, "hyst", None, exc=(E,)),
              "classes": stops(TT.load, variant("c", lambda d: d.update(classes=list(range(8)))), 9, "hyst", None, exc=(E,)),
              "not_candidate": stops(TT.load, variant("n", lambda d: (d["enter"].__setitem__(0, 0.65), d["stay"].__setitem__(0, TT.stay_of(0.65)))), 9, "hyst", None, exc=(E,)),
              "stay_ratio": stops(TT.load, variant("s", lambda d: d["stay"].__setitem__(1, 0.3001)), 9, "hyst", None, exc=(E,)),
              "nonfinite": stops(TT.load, variant("f", lambda d: d["stay"].__setitem__(2, float("nan"))), 9, "hyst", None, exc=(E,)),
              "ckpt": stops(TT.load, out, 9, "hyst", "d" * 64, exc=(E,)),
              "mode": stops(TT.load, out, 9, "both", None, exc=(E,))}
        tn = TT.load(out, 9, "nohyst", None)
        check("T4 表の確かめ: 版・クラス・候補に無い入り・保持の比・非有限・別の ckpt・mode → 止まる・nohyst は保持 = 入り",
              all(r4.values()) and np.array_equal(tn["stay"], tn["enter"]) and tn["mode"] == "nohyst", str(r4))
        # T5
        v17 = (ROOT / "server_sde/_causal_infer_v17.py").read_text(encoding="utf-8")
        need = ['THRESH_JSON = os.environ.get("CAUSAL_THRESHOLDS_JSON", "").strip()', 'os.environ.get("CAUSAL_THRESHOLDS_MODE", "hyst")',
                "CAUSAL_THRESHOLDS_JSON と CAUSAL_SED_TH／CAUSAL_SED_STAY を同時に置かない", "CAUSAL_THRESHOLDS_MODE は CAUSAL_THRESHOLDS_JSON（表）と一緒に置く",
                "threshold_table.load(THRESH_JSON, NCLS, THRESH_MODE, ckpt_sha=CK_SHA)",
                "hysteresis_patch.apply_with_start(doa_np, sed_in, NCLS, ENTER, STAY, SED_ASSOC, start=HYST_START)",
                "sed_threshold=None, sed_stay=None", "thresholds=threshold_table.record(TABLE)", 'META["settings"]["dump_norms"] = True',
                'DUMP_NORMS = os.environ.get("DUMP_NORMS", "0") == "1"', '.npy.part"', "os.replace(_np_tmp, NORMS_DIR", "norms_manifest.json",
                'raise RuntimeError("thresholds table changed during inference")', "code_paths.add(Path(threshold_table.__file__).resolve())", "norms が欠けている"]
        miss = [n for n in need if n not in v17]
        order = v17.index("if TABLE is not None:\n        # §3-5") < v17.index("elif HYST_START > 0:") if "if TABLE is not None:\n        # §3-5" in v17.replace("\r\n", "\n") else False
        check("T5 推論の入口（v17 の文面）: 環境変数・同時は止める・ckpt で確かめる・判定は apply_with_start に表・run_meta の記録・norms の保存と一覧・途中の変化で止める",
              not miss and order, str(miss))
        # T6
        ck = tmp / "e099.ckpt"; ck.write_bytes(b"ckpt")
        ck_sha = CC.sha256(ck)
        d6 = json.loads(out.read_text(encoding="utf-8")); d6["inputs"]["ckpt_sha256"] = ck_sha
        good = tmp / "th_good.json"; good.write_text(json.dumps(d6), encoding="utf-8")
        sel = tmp / "sel.json"
        sel.write_text(json.dumps({"spec_version": "v2_ckpt_selector/2.1 (x)", "not_for_registration": False, "selected": 99, "conditions": {"height": "table"},
                                   "candidates": {"99": {"verified": True, "ckpt_sha256": ck_sha, "ckpt": "runs/x_sde_w3_causal/checkpoints/epoch_099.ckpt"}}}), encoding="utf-8")
        dec = tmp / "dec.json"; dec.write_text(json.dumps({"adopt": False, "verdict": "表なし（不採用）", "inputs": {}}), encoding="utf-8")
        base = ["--select-json", str(sel), "--ckpt", str(ck), "--decide-json", str(dec)]
        rec = CC.check(sel, dec, ck, None, thresholds_json=good)
        rc_ok = CC.main(base + ["--thresholds-json", str(good), "--dry"])
        rc_ck = CC.main(base + ["--thresholds-json", str(out), "--dry"])          # 表を作った ckpt が c*64 = 主の ckpt と違う
        bad = tmp / "th_bad.json"; bad.write_text("{", encoding="utf-8")
        rc_bad = CC.main(base + ["--thresholds-json", str(bad), "--dry"])
        check("T6 契約 --thresholds-json: 正しい表 → 通り記録に sha と入り／別の ckpt の表 → 3／壊れた表 → 3",
              rc_ok == 0 and rec["thresholds"]["sha256"] == CC.sha256(good) and rec["thresholds"]["enter"] == want and rc_ck == 3 and rc_bad == 3, str((rc_ok, rc_ck, rc_bad)))
        # T7（Fable 16:5x: B・D もクラス別）: D の対象クラスの部分集合
        cmp = ROOT / "md/design/baselines_class_map_v2.json"
        dcls = sorted({int(v["ours_index"]) for v in json.loads(cmp.read_text(encoding="utf-8"))["D_pseldnets_pretrained"]["map"].values()})
        gD, runD, mD = make_val(tmp / "h")
        outD = tmp / "h" / "thresholds_v30_D.json"
        rcD = VS.main(["--gt", str(gD), "--manifest", str(mD), "--run", str(runD), "--label", "D", "--out", str(outD), "--classmap", str(cmp), "--target", "D"])
        JD = json.loads(outD.read_text(encoding="utf-8"))
        tD = TT.load(outD, 9, "hyst", "c" * 64, classes=dcls)
        others = [c for c in range(9) if c not in dcls]
        doa = rand_doa(3)
        sedD = HP.apply(doa, C, tD["enter"], tD["stay"])
        recD = TT.record(tD)
        check("T7 D の部分集合: 表の classes = 対応表の共通クラス・enter はその順・load(classes) で対象外は +inf（記録は None）・classes なしの load は止まる・対象外のクラスは決して出ない",
              rcD == 0 and JD["classes"] == dcls and len(JD["enter"]) == len(dcls) and all(np.isinf(tD["enter"][c]) for c in others)
              and all(recD["enter"][c] is None for c in others) and stops(TT.load, outD, 9, "hyst", None, exc=(E,)) and not sedD[:, :, others].any()
              and stops(VS.main, ["--gt", str(gD), "--manifest", str(mD), "--run", str(runD), "--label", "D", "--out", str(tmp / "h" / "x.json"), "--classmap", str(cmp)]),
              f"D の対象 {dcls}・入り {JD['enter']}")
        # T8 norms の保存と一覧（B・D が使う）
        nd = tmp / "nm" / "norms"; nd.mkdir(parents=True)
        for c in ("a", "b"):
            TT.save_norms(nd, c, np.ones((3, T, C)))
        M8 = TT.write_norms_manifest(tmp / "nm", ["a", "b"], "w" * 64, 9, "x", extra={"kind": "test"})
        no_part = not list(nd.glob("*.part"))
        miss8 = stops(TT.write_norms_manifest, tmp / "nm", ["a", "b", "c"], "w" * 64, 9, "x", exc=(RuntimeError,))
        check("T8 norms の保存と一覧: .part を残さない・一覧に重みの sha と本ごとの sha・欠けがあれば止まる",
              no_part and M8["ckpt_sha256"] == "w" * 64 and set(M8["files"]) == {"a", "b"} and M8["kind"] == "test" and miss8
              and np.load(nd / "a.npy").dtype == np.float32)
        # T9 B・D の推論と候補の台本（文面）
        srcB = (ROOT / "server_sde/_baseline_B_infer.py").read_text(encoding="utf-8")
        srcD = (ROOT / "server_sde/_baseline_D_pretrain170_infer.py").read_text(encoding="utf-8")
        common = ['THRESH_JSON = os.environ.get("CAUSAL_THRESHOLDS_JSON", "").strip()', "CAUSAL_THRESHOLDS_JSON と CAUSAL_SED_TH／CAUSAL_SED_STAY を同時に置かない",
                  'HYST_START = int(os.environ.get("HYST_START_FRAME", "0") or 0)', "hysteresis_patch.apply_with_start(doa9, sed_in, N_OURS, ENTER, STAY, SED_ASSOC, start=HYST_START)",
                  "thresholds=threshold_table.record(TABLE)", 'META["settings"]["hyst_start_frame"] = HYST_START', "threshold_table.save_norms(nd, fn[:-5]",
                  "threshold_table.write_norms_manifest(subs.parent", 'raise RuntimeError("thresholds table changed during inference")',
                  "code_paths.add(Path(threshold_table.__file__).resolve())", "sed9 = hysteresis_patch.apply(doa9, N_OURS, SED_TH, SED_STAY, SED_ASSOC)"]
        missB = [n for n in common + ['ckpt_sha=MODEL_SHA, classes=None'] if n not in srcB]
        missD = [n for n in common + ['ckpt_sha=CK_SHA, classes=sorted(SRC_OF)'] if n not in srcD]
        sb = (ROOT / "server_sde/v2final_candidates_infer.sbatch").read_text(encoding="utf-8").replace("\r\n", "\n")
        sb_ok = ('export SDE_INPUT_GAIN_DB DUMP_NORMS=1 HYST_START_FRAME=40' in sb and 'unset SDE_INPUT_GAIN_DB DUMP_NORMS HYST_START_FRAME' in sb
                 and 'cp $E2E/server_sde/gain_patch.py $JOBDIR/gain_patch.py' in sb and 'norms_manifest.json が無い' in sb and 'v30 は入力の倍率 SDE_INPUT_GAIN_DB を明示する' in sb)
        check("T9 B・D の推論に表の入口・HYST_START・norms の一覧（旧の定数の道も残る）／候補の推論の台本は v30 だけ norms・40・倍率の明示（v29 は 3 つとも外す）",
              not missB and not missD and sb_ok, str((missB, missD, sb_ok)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
