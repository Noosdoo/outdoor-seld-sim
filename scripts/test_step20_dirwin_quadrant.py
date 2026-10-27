# -*- coding: utf-8 -*-
"""先行分 §5（決定 ⑲・2026-10-26 Fable 仕様・Astra R30-03／09）の受け入れ試験: step20 v3 の --gt-dirwin（象限の正解を正解の軌道から）。
架空の注釈・予測・正解の軌道だけを使う（実録・本物の読みは使わない）。

  Q1 架空の gt_dirwin（左 +100°・右 −80°・後 170°・前 10°・±180 をまたぐ 175／−175）で象限が L／R／B／F／B（円周平均）・記録シートの象限は quadrant_sheet に残る
  Q2 gt_dirwin に無い本は主の方向の集計で欠測（quadrant_gt 空・理由「軌道なし」・分母に入れない）・記録シートの象限での代用は参考の別表だけ
  Q3 警告音（止めた車のクラクション E2）: gt の [t_start, t_start + 1 s] の固定位置の方位（+90）で L（step19m・§4 の規約と同じ）・同じクラスに track = event_id の行があればそれだけ
  Q4 窓の方位が打ち消し合う（+90 と −90 が同数）→ 決まらない＝欠測・窓に正解が無い → 欠測
  Q5 命令: --gt-dirwin あり → 報告に「正解 = 正解の軌道」「欠測 n 本」「食い違い」「参考（記録シートで代用）」・events に quadrant_sheet／quadrant_gt の列。
     --gt-dirwin なし → 報告と events が §5 の前（be4af06fa）の step20 と 1 バイトも同じ
使い方: python scripts/test_step20_dirwin_quadrant.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s20v3_dq", ROOT / "scripts/step20_realsmoke_score_v3.py")
S = importlib.util.module_from_spec(spec); sys.modules["s20v3_dq"] = S; spec.loader.exec_module(S)

REF_COMMIT = "be4af06fa"            # §5 の前の版（既定の出力の bit 一致の基準）
COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "区分", "状態", "横距離m", "n_car", "scored"]
CAR = S.CLS_IDX["car_drive"]; HORN = S.CLS_IDX["horn"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def car_row(clip, quad, t_cpa="8"):
    return dict(zip(COLS, [clip, "1", "T1", "", "A", "car_drive", quad, "2", t_cpa, "", "A", "静止", "2.0", "1", "1"]))


def fixture():
    """clip → (記録シートの象限, 正解の軌道の窓の方位 or None)。予測は窓（5.5〜6.5 s）に az_pred を出す。"""
    cases = {"eval_field_clip00301": ("L", [100.0] * 10, 100.0), "eval_field_clip00302": ("R", [-80.0] * 10, -80.0),
             "eval_field_clip00303": ("B", [170.0] * 10, 170.0), "eval_field_clip00304": ("F", [10.0] * 10, 10.0),
             "eval_field_clip00305": ("B", [175.0, -175.0] * 5, 178.0), "eval_field_clip00306": ("L", None, 95.0),
             "eval_field_clip00307": ("L", [90.0, -90.0] * 5, 90.0), "eval_field_clip00308": ("F", [], 0.0),
             "eval_field_clip00309": ("L", [-100.0] * 10, -100.0)}         # 記録シートは L・軌道は R（食い違い 1 本）
    rows, pred, dirwin = [], {}, {}
    for clip, (q, azs, azp) in cases.items():
        rows.append(car_row(clip, q))
        pred[clip] = {}
        for k in range(55, 66):
            pred[clip].setdefault(k, []).append((CAR, azp, 0.0, 2.0))
        if azs is not None:
            dirwin[clip] = [(55 + i, CAR, 1, a) for i, a in enumerate(azs)]
    # 警告音: E2 の 1 本（固定位置の方位 +90・記録シートは L）。gt は鳴っている 3.0〜4.0 s。track 2 の別の押しは窓の外
    rows.append(dict(zip(COLS, ["eval_field_clip00310", "1", "T2", "", "E", "horn", "L", "3.0", "3.5", "", "E", "静止", "", "1", "1"])))
    full = {"eval_field_clip00310": [(k, HORN, 1, 90.0) for k in range(30, 40)] + [(k, HORN, 2, -90.0) for k in range(70, 75)]}
    pred["eval_field_clip00310"] = {k: [(HORN, 88.0, 0.0, 3.0)] for k in range(30, 40)}
    return rows, pred, dirwin, full


def run_eval(rows, pred, dirwin, full):
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    S.GT_DIRWIN, S.GT_FULL = dirwin, full
    try:
        events, _, _ = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 0.0, True)
    finally:
        S.GT_DIRWIN = S.GT_FULL = None
    return {e["clip"]: e for e in events}


def write_inputs(td: Path, rows, pred, dirwin, full):
    with open(td / "ann.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    with open(td / "pred.csv", "w", encoding="utf-8") as f:
        for clip, fr in pred.items():
            for k, lst in sorted(fr.items()):
                for c, az, el, d in lst:
                    f.write(f"{clip},{k},{c},0,{az},{el},{d}\n")
    for name, src in (("gt_dirwin", dirwin), ("gt", full)):
        (td / name).mkdir()
        for clip, rs in src.items():
            (td / name / f"{clip}.csv").write_text("".join(f"{k},{c},{t},{int(round(a))},0,2.00\n" for k, c, t, a in rs), encoding="utf-8")


def main() -> int:
    rows, pred, dirwin, full = fixture()
    ev = run_eval(rows, pred, dirwin, full)
    want = {"eval_field_clip00301": "L", "eval_field_clip00302": "R", "eval_field_clip00303": "B", "eval_field_clip00304": "F", "eval_field_clip00305": "B"}
    got = {c: ev[c]["quadrant_gt"] for c in want}
    check("Q1 架空の gt_dirwin（+100・−80・170・10・175/−175）→ L・R・B・F・B（円周平均）・記録シートの象限は quadrant_sheet に残る",
          got == want and all(ev[c]["quadrant_sheet"] == ev[c]["quadrant"] for c in want) and ev["eval_field_clip00305"]["quad_ok"] is True, str(got))
    e6 = ev["eval_field_clip00306"]
    check("Q2 gt_dirwin に無い本 → quadrant_gt 空・欠測（理由 軌道なし）・主の一致 quad_ok は None・参考の quad_ok_sheet は記録シートで比べる",
          e6["quadrant_gt"] == "" and e6["quad_gt_missing"] == 1 and "軌道なし" in e6["quad_gt_why"] and e6["quad_ok"] is None and e6["quad_ok_sheet"] is True, str({k: e6[k] for k in ("quadrant_gt", "quad_gt_why", "quad_ok", "quad_ok_sheet")}))
    e10 = ev["eval_field_clip00310"]
    check("Q3 警告音（E2）: gt の鳴り始めの窓の固定位置の方位 +90 → L・track = event_id の行だけ（窓の外の track 2 は使わない）・発火の方位で一致",
          e10["quadrant_gt"] == "L" and e10["quad_ok"] is True, str({k: e10.get(k) for k in ("quadrant_gt", "quad_gt_why", "quad_ok", "notified")}))
    e7, e8 = ev["eval_field_clip00307"], ev["eval_field_clip00308"]
    check("Q4 打ち消し合う（+90 と −90 が同数）→ 欠測（決まらない）・窓に正解が無い → 欠測",
          e7["quadrant_gt"] == "" and "決まらない" in e7["quad_gt_why"] and e8["quadrant_gt"] == "" and "窓に正解が無い" in e8["quad_gt_why"],
          f"{e7['quad_gt_why']} / {e8['quad_gt_why']}")
    ref = ROOT / "scripts" / f"_tmp_step20_ref_{os.getpid()}.py"
    try:
        src = subprocess.run(["git", "show", f"{REF_COMMIT}:outdoor_seld_e2e/scripts/step20_realsmoke_score_v3.py"], cwd=str(ROOT), capture_output=True)
        ref.write_bytes(src.stdout)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_inputs(td, rows, pred, dirwin, full)
            env = dict(os.environ, PYTHONUTF8="1")
            base = ["--pred", str(td / "pred.csv"), "--ann", str(td / "ann.csv"), "--min-history", "0"]
            r_old = subprocess.run([sys.executable, str(ref), *base, "--out", str(td / "old.md"), "--events-out", str(td / "old.csv")], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
            r_new = subprocess.run([sys.executable, str(ROOT / "scripts/step20_realsmoke_score_v3.py"), *base, "--out", str(td / "new.md"), "--events-out", str(td / "new.csv")], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
            r_gt = subprocess.run([sys.executable, str(ROOT / "scripts/step20_realsmoke_score_v3.py"), *base, "--out", str(td / "gt.md"), "--events-out", str(td / "gt.csv"),
                                   "--gt-dirwin", str(td / "gt_dirwin")], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
            ok_run = r_old.returncode == 0 and r_new.returncode == 0 and r_gt.returncode == 0
            same = ok_run and (td / "old.md").read_bytes() == (td / "new.md").read_bytes() and (td / "old.csv").read_bytes() == (td / "new.csv").read_bytes()
            check(f"Q5a --gt-dirwin なし: 報告と events が §5 の前（{REF_COMMIT}）の step20 と 1 バイトも同じ", same, "" if ok_run else (r_old.stderr + r_new.stderr + r_gt.stderr)[-800:])
            md = (td / "gt.md").read_text(encoding="utf-8") if ok_run else ""
            hdr = (td / "gt.csv").read_text(encoding="utf-8").splitlines()[0] if ok_run else ""
            check("Q5b --gt-dirwin あり: 正解 = 正解の軌道・欠測 3 本（主の分母に入れない）・食い違い 1 本・参考の別表・events に quadrant_sheet／quadrant_gt・"
                  "#57 の内訳は軌道の象限（記録シートの内訳は参考の別表）",
                  "正解 = 正解の軌道" in md and "**3 本**" in md and "食い違い: 1/" in md and "参考（主と混ぜない" in md and "quadrant_sheet" in hdr and "quadrant_gt" in hdr
                  and "象限 = 正解の軌道・--gt-dirwin" in md and "## 参考: 来る方向の内訳（記録シートの象限" in md and "## 補助: 来る方向（象限）の内訳（変更票 #57・記述だけ" not in md,
                  "\n".join(l for l in md.splitlines() if "方向" in l or "参考（主と" in l or "食い違い" in l)[:600])
    finally:
        ref.unlink(missing_ok=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
