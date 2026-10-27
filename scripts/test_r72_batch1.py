# -*- coding: utf-8 -*-
"""test_r72_batch1.py — Astra 監査 R72（2026-10-20）の重大 3 件の直しの検査。

  A1 R72-01: 統一採点器の補正表は --dist-table-for の入力だけに掛かる（v1・オラクルは不変）。--dist-table だけ（-for 無し）は拒む
  B1 R72-02: 切り出し器が t_end・t_move_s・t_end_unknown をクリップの秒に移し、区間の終端で交差を判定する（監査の再現 3 つ）
  C1 R72-03: step19d が「長く続く相手（サイレン等）の終わりが空／nan／逆順」を不明（t_end_unknown=1）にし、短い相手の空は既定の窓のまま
  C2 R72-03: step20 が不明の区間を clip の最後まで除外し、露出と誤警告から外し、別計上の行を出す。全部除かれた本は率を未定義
使い方: python scripts/test_r72_batch1.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="r72_"))
    try:
        # ---------------- A1: 補正表の適用先 ----------------
        U = load("su_r72", "_score_unified.py")
        tbl = tmp / "t.json"; tbl.write_text(json.dumps({"centers": [0.125 + 0.25 * i for i in range(20)], "values": [1.0] * 20}), encoding="utf-8")   # 何でも 1.0 m に写す表
        pred = tmp / "p.csv"; pred.write_text("".join(f"c1,{k},4,0,30,0,2.0\n" for k in range(40, 100)), encoding="utf-8")
        U.DIST_TABLE = (__import__("numpy").array(json.loads(tbl.read_text())["centers"]), __import__("numpy").array(json.loads(tbl.read_text())["values"]))
        U.DIST_TABLE_FOR = {"v2final"}
        d_v1 = U.pred_for("v1", pred, False)["c1"][50][0][3]; d_v2 = U.pred_for("v2final", pred, False)["c1"][50][0][3]; d_or = U.pred_for("oracle", pred, False)["c1"][50][0][3]
        ok.append((f"A1 表は v2final だけ（v1 {d_v1}・v2final {d_v2}・oracle {d_or}）", d_v1 == 2.0 and d_v2 == 1.0 and d_or == 2.0))
        U.DIST_TABLE_FOR = None; U.DIST_TABLE = None
        # main が -for 無しを拒む（plan・GT は要らない: 引数の検査が先）
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts/_score_unified.py"), str(tmp / "x.md"), "--plan", "nope.csv", "--meta", "nope", "--dist-table", str(tbl), "v1=a.csv"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
        ok.append(("A1' --dist-table だけ（--dist-table-for 無し）は拒む", r.returncode != 0 and "--dist-table-for" in (r.stderr + r.stdout)))
        # ---------------- B1: 切り出しの再基準化 ----------------
        B = load("s19b_r72", "step19b_realsmoke_cut.py")
        base = dict(event_id="1", trial="x", quadrant="B", session_id="s", take_id="1", 区分="B")
        rows = [dict(base, **{"class": "car_drive"}, t_start="16.00", t_cpa="20.00", t_end="", t_move_s="16.00")]
        out = B.rebase_rows(rows, 12.0, 10.0, "o.WAV", "clip1", target_event="1", orig_duration_s=60.0)
        ok.append((f"B1 動き出し 16・最接近 20 を 12 s から切る → t_cpa 8.00・t_move_s 4.00（{out[0]['t_move_s']}）", out[0]["t_cpa"] == "8.00" and out[0]["t_move_s"] == "4.00" and out[0]["t_start"] == "4.00"))
        rows2 = [dict(base, **{"class": "car_drive"}, 区分="負例露出", t_start="300.00", t_cpa="300.00", t_end="305.00")]
        out2 = B.rebase_rows(rows2, 297.0, 10.0, "o.WAV", "clip2", own=(1.0, 10.0), orig_duration_s=600.0)
        ok.append((f"B1' 長回しの相手 [300,305] を 297 s から切る → [3.00, 8.00]（{out2[0]['t_start']}, {out2[0]['t_end']}）", out2[0]["t_start"] == "3.00" and out2[0]["t_end"] == "8.00"))
        rows3 = [dict(base, **{"class": "siren"}, 区分="負例露出", t_start="300.00", t_cpa="300.00", t_end="325.00")]
        out3 = B.rebase_rows(rows3, 306.0, 10.0, "o.WAV", "clip3", own=(1.0, 10.0), orig_duration_s=600.0)
        ok.append((f"B1'' 相手 [300,325] は 306 s からのクリップにも残る（t_start −6.00・t_end 19.00・クランプなし）", len(out3) == 1 and out3[0]["t_start"] == "-6.00" and out3[0]["t_end"] == "19.00"))
        rows4 = [dict(base, **{"class": "siren"}, 区分="負例露出", t_start="300.00", t_cpa="300.00", t_end="", t_end_unknown="1")]
        out4 = B.rebase_rows(rows4, 500.0, 10.0, "o.WAV", "clip4", own=(1.0, 10.0), orig_duration_s=600.0)
        ok.append(("B1''' 終わり不明の相手は録音末尾までのどのクリップにも残る（500 s から: t_start −200.00・t_end_unknown 1）", len(out4) == 1 and out4[0]["t_start"] == "-200.00" and out4[0]["t_end_unknown"] == "1"))
        # ---------------- C1: step19d の不明 ----------------
        D = load("s19d_r72", "step19d_field_csv_to_ann.py")
        ses = [dict(session_id="g", 用途="最終評価", 区分="負例露出", マイク高さ_cm="200")]
        ev = [dict(session_id="g", take_id="1", event_id="1", 原本="261105_001.WAV", **{"class": "none"}, 象限="", ラップ秒="", 状態="静止"),
              dict(session_id="g", take_id="1", event_id="2", 原本="261105_001.WAV", **{"class": "siren"}, 象限="L", ラップ秒="300", 終わり秒="", 状態="静止"),
              dict(session_id="g", take_id="1", event_id="3", 原本="261105_001.WAV", **{"class": "crossing"}, 象限="L", ラップ秒="400", 終わり秒="nan", 状態="静止"),
              dict(session_id="g", take_id="1", event_id="4", 原本="261105_001.WAV", **{"class": "siren"}, 象限="L", ラップ秒="450", 終わり秒="440", 状態="静止"),
              dict(session_id="g", take_id="1", event_id="5", 原本="261105_001.WAV", **{"class": "car_drive"}, 象限="L", ラップ秒="500", 終わり秒="", 状態="静止", 横距離m="2.0")]
        out_d, warns = D.convert(ses, ev, 6.0, 0.0, None, 3.0, None)
        by = {r["event_id"]: r for r in out_d}
        ok.append(("C1 サイレンの終わりが空 → 不明（t_end_unknown 1・t_start 300）＋警告", by["2"]["t_end_unknown"] == "1" and by["2"]["t_start"] == "300.00" and by["2"]["t_end"] == "" and any("不明" in w and "event 2" in w for w in warns)))
        ok.append(("C1' 踏切の終わりが nan → 不明＋警告", by["3"]["t_end_unknown"] == "1" and any("有限でない" in w or "数値でない" in w for w in warns)))
        ok.append(("C1'' サイレンの終わりが始まりより前 → 不明＋警告", by["4"]["t_end_unknown"] == "1" and any("より前" in w for w in warns)))
        ok.append(("C1''' 通り過ぎる相手（車）の終わりが空 → 既定の窓（t_start 494・t_cpa 500・不明でない）", by["5"]["t_end_unknown"] == "" and by["5"]["t_start"] == "494.00" and by["5"]["t_cpa"] == "500.00"))
        # ---------------- C2: step20 の不明の区間 ----------------
        S = load("s20_r72", "step20_realsmoke_score_v3.py")
        cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
        COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "t_end_unknown", "区分", "状態", "横距離m", "n_car", "scored"]
        rows = [dict(zip(COLS, ["eval_field_clip09002", "1", "N-01", "", "", "none", "", "0", "600", "", "", "負例露出", "歩行", "", "1", "1"])),
                dict(zip(COLS, ["eval_field_clip09002", "2", "N-01", "", "", "siren", "L", "300", "300", "", "1", "負例露出", "歩行", "", "1", "1"]))]   # 終わり不明のサイレン（300 s から）
        pred = {"eval_field_clip09002": {}}
        for k in list(range(2000, 2006)) + list(range(4000, 4006)):     # 200 s（露出の中）と 400 s（不明の区間の中）にサイレンの発火
            pred["eval_field_clip09002"].setdefault(k, []).append((0, 30.0, 0.0, float("nan")))
        events, neg, extras = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 7.5, True)
        g = neg["by"]["負例露出"]
        ok.append((f"C2 不明の区間 [299, 末尾) を除外: 露出 299 s（{g['exposure_s']:.0f}）・誤警告 1 件（{g['n_false']}）・不明 1 区間 {g['unknown_s']:.0f} s",
                   abs(g["exposure_s"] - 299.0) < 1e-6 and g["n_false"] == 1 and g["n_unknown"] == 1 and abs(g["unknown_s"] - 301.0) < 1e-6))
        rows_all = [dict(rows[0]), dict(rows[1], t_start="0")]                      # 全部が不明の区間 → 露出 0 → 率は未定義
        ev2, neg2, _ = S.evaluate(rows_all, pred, cfg43, (2.5, 1.5), 7.5, True)
        ok.append(("C2' 全部が除かれた本: 露出 0 s・誤警告 0", abs(neg2["by"]["負例露出"]["exposure_s"]) < 1e-6 and neg2["n_false"] == 0))
        ann = tmp / "ann.csv"
        with open(ann, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        pcsv = tmp / "pred.csv"
        with open(pcsv, "w", encoding="utf-8") as f:
            for clip, fr in pred.items():
                for k, dets in sorted(fr.items()):
                    for c, az, el, d in dets:
                        f.write(f"{clip},{k},{c},0,{az},{el},nan\n")
        outmd = tmp / "score.md"
        sys.argv = ["step20", "--pred", str(pcsv), "--ann", str(ann), "--out", str(outmd)]
        rc = S.main(); t = outmd.read_text(encoding="utf-8")
        ok.append(("C2'' 結果 md に「終わり不明の長い相手」の別計上の行", rc == 0 and "終わり不明の長い相手" in t and "負例露出 1 区間 301 s" in t))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
