# -*- coding: utf-8 -*-
"""scripts/demo_field_unused_score.py の試験（2026-10-27 Opus）。架空の正解と予測だけ。
  T1 自転車（8）: 正解 50〜89・予測 50〜69（向き +10°・距離 +0.5 m）→ 見つけた 20/40・向きの誤差 10°・距離の誤差 0.5 m・正解に無いクラス（車）の発火を数える
  T2 ベル（3）・除外区間 [6, 7) が履歴と窓に重なる → 部分観察（成功にも失敗にも数えない）
  T3 ベル（3）・除外なし・鳴り始め 5.0 s の 0.55 s 後に発火 → 届いた・遅れ 0.55 s
  T4 自転車・除外区間が採点のフレームを全部覆う → 欠測（recall None）・有効 0 s
  T5 表の頭: 本 4・欠測 1・警告音の事象 2（通常 1・部分観察 1・届いた 1）・除外区間の本は表で印（＊）
  T6 参考の表（2026-10-27）: 取り違えの表（自転車 20・車と取り違え 1・見逃し 19・ベル 2）・本ごとの種類 3/4（全部除外の D は未検出）
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import demo_field_unused_score as S  # noqa: E402


def wcsv(p, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="demo_score_"))
    try:
        inf = tmp / "infer"
        (inf / "gt").mkdir(parents=True)
        def gt(clip, c, frames, az=90.0, el=-10.0, d=2.0):
            (inf / "gt" / f"{clip}.csv").write_text("".join(f"{k},{c},0,{az},{el},{d}\n" for k in frames), encoding="utf-8")
        gt("A", 8, range(50, 90)); gt("B", 3, range(40, 81)); gt("C", 3, range(40, 81)); gt("D", 8, range(45, 95))
        cols = ["clip_id", "orig_file", "class", "t_start", "t_cpa", "cut_offset_s", "区分"]
        wcsv(inf / "ann_all.csv", cols, [
            {"clip_id": "A", "orig_file": "a.WAV", "class": "bicycle", "t_start": "", "t_cpa": "13.0", "cut_offset_s": "5.0", "区分": "入れる"},
            {"clip_id": "B", "orig_file": "b.WAV", "class": "bike_bell", "t_start": "10.0", "cut_offset_s": "5.0", "区分": "入れる（除外区間あり）"},
            {"clip_id": "C", "orig_file": "c.WAV", "class": "bike_bell", "t_start": "10.0", "cut_offset_s": "5.0", "区分": "入れる"},
            {"clip_id": "D", "orig_file": "d.WAV", "class": "bicycle", "t_start": "", "cut_offset_s": "5.0", "区分": "入れる（除外区間あり）"}])
        wcsv(inf / "exclusions.csv", ["clip_id", "入り_クリップの秒", "出_クリップの秒"],
             [{"clip_id": "B", "入り_クリップの秒": "6.0", "出_クリップの秒": "7.0"}, {"clip_id": "D", "入り_クリップの秒": "3.5", "出_クリップの秒": "10.0"}])
        pred = tmp / "pred.csv"
        lines = [f"A,{k},8,0,100.0,-10.0,2.5" for k in range(50, 70)] + ["A,85,4,0,0,0,10"]
        lines += [f"B,{k},3,0,90.0,-10.0,2.0" for k in (55, 66)] + ["C,55,3,0,90.0,-10.0,2.0"] + [f"D,{k},8,0,90,-10,2" for k in range(50, 60)]
        pred.write_text("\n".join(lines) + "\n", encoding="utf-8")
        buf = StringIO()
        with redirect_stdout(buf):
            rc = S.main(["--infer", str(inf), "--pred", str(pred), "--out", str(tmp / "r.json"), "--label", "試験"])
        out = buf.getvalue()
        r = json.loads((tmp / "r.json").read_text(encoding="utf-8"))
        cl = {x["clip_id"]: x for x in r["clips"]}
        a8 = cl["A"]["classes"]["8"]
        want = S.ang_err(100.0, -10.0, 90.0, -10.0)      # 仰角 −10° で方位 10° の差 = 球面の角度 9.85°
        ok.append(("T1 自転車: 20/40・向きの誤差 9.85°（方位 10°・仰角 −10°）・距離の誤差 0.5 m・正解に無い車の発火 1", rc == 0 and a8["n_gt"] == 40 and a8["n_hit"] == 20
                   and abs(a8["ang_med"] - want) < 1e-6 and abs(want - 9.85) < 0.01 and abs(a8["dist_err_med"] - 0.5) < 1e-9 and cl["A"]["other_class_frames"] == {"車": 1}))
        ok.append(("T2 ベル・除外が履歴と窓に重なる → 部分観察", cl["B"]["classes"]["3"]["notification"]["status"] == "partial"))
        n3 = cl["C"]["classes"]["3"]["notification"]
        ok.append(("T3 ベル・除外なし → 届いた・遅れ 0.55 s", n3["status"] == "success" and abs(n3["latency_s"] - 0.55) < 1e-9))
        d8 = cl["D"]["classes"]["8"]
        ok.append(("T4 除外が採点のフレームを全部覆う → 欠測・有効 0 s", d8["recall"] is None and d8["valid_s"] == 0.0))
        h = r["head"]
        ok.append(("T5 表の頭: 本 4・欠測 1・事象 2（通常 1・部分観察 1・届いた 1）・除外区間の本に印", h == {"n_takes": 4, "n_missing_takes": 1, "n_events": 2, "n_partial": 1, "n_scored_events": 1, "n_success": 1}
                   and "＊B（b.WAV" in out and "＊D（d.WAV" in out and "　A（a.WAV" in out))
        cm = r["confusion"]["matrix"]; ta = r["take_accuracy"]
        ok.append((f"T6 参考の表: 取り違え (自転車, 自転車) {cm[8][8]}・(自転車, 車) {cm[8][4]}・(自転車, なし) {cm[8][9]}・(ベル, ベル) {cm[3][3]}・本ごとの種類 {ta['correct']}/{ta['n']}",
                   cm[8][8] == 20 and cm[8][4] == 1 and cm[8][9] == 19 and cm[3][3] == 2 and ta["n"] == 4 and ta["correct"] == 3
                   and ta["per_class"]["8"]["undetected"] == 1 and "取り違えの表" in out and "本ごとの種類の正答率" in out))   # json では鍵が文字列
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
