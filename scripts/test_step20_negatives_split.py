# -*- coding: utf-8 -*-
"""test_step20_negatives_split.py — 実録の採点器 step20 v3 の負例の扱いの検査（2026-09-29・Opus 説明係の指摘 1・4 を受けて）。

合成の注釈と予測で確かめる:
  N1 負例は区分ごとに数える: 区分 G（60 秒・警告音の発火 1 件）と 負例露出（600 秒・発火 2 件）が別々の率になり、合計も合う
  N2 負例露出の中に偶然入った相手（class=car_drive・scored=1）は「計画外」として主要評価に入らず、別掲に 1 件。その窓は露出から引かれる
  N3 計画の本（区分 A）の事象は主要評価に入る
  N4 結果 md に「負例テイク G」「負例露出」「合格線は置かない」「計画外」の行がある（#70 r2・2026-10-19: 合否は出さない。旧 = 0 件なら「合格」・2 件なら「不合格」）
  N5 events csv に unplanned 列がある
使い方: python scripts/test_step20_negatives_split.py
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s20v3_test", ROOT / "scripts/step20_realsmoke_score_v3.py")
S = importlib.util.module_from_spec(spec); sys.modules["s20v3_test"] = S; spec.loader.exec_module(S)

COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "区分", "状態", "横距離m", "n_car", "scored"]


def siren_frames(frames: dict, k0: int, k1: int, az: float = 30.0):
    for k in range(k0, k1):
        frames.setdefault(k, []).append((0, az, 0.0, float("nan")))


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="s20neg_"))
    try:
        cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
        # 予測: G の本 = サイレン 3 フレーム連続が 1 回（10.0〜10.5 s）。負例露出 = 2 回（20 s と 40 s）＋ 計画外の車の窓の中で 1 回（301 s・マスクされる）
        pred = {"eval_field_clip09001": {}, "eval_field_clip09002": {}, "eval_field_clip00101": {}}
        siren_frames(pred["eval_field_clip09001"], 100, 106)
        siren_frames(pred["eval_field_clip09002"], 200, 206); siren_frames(pred["eval_field_clip09002"], 400, 406); siren_frames(pred["eval_field_clip09002"], 3010, 3016)
        rows = [
            dict(zip(COLS, ["eval_field_clip09001", "1", "G-03", "", "", "none", "", "0", "60", "", "G", "静止", "", "1", "1"])),
            dict(zip(COLS, ["eval_field_clip09002", "1", "N-01", "", "", "none", "", "0", "600", "", "負例露出", "歩行", "", "1", "1"])),
            dict(zip(COLS, ["eval_field_clip09002", "2", "N-01", "", "", "car_drive", "L", "300", "300", "305", "負例露出", "歩行", "2.0", "1", "1"])),   # 計画外の車 = 長回しの相手 [始まり 300, 終わり 305]（#69 r18・t_end）
            dict(zip(COLS, ["eval_field_clip00101", "1", "A-01", "", "", "car_drive", "L", "3", "8", "", "A", "静止", "2.0", "1", "1"])),
        ]
        events, neg, extras = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 7.5, True)
        by = neg["by"]
        ok.append(("N1 区分ごとの誤警告: G 1 件/60 s・負例露出 2 件/593 s（計画外の相手の区間 [300−1, 305+1] = 7 s を引く・t_end）・合計 3 件", by.get("G", {}).get("n_false") == 1 and abs(by["G"]["exposure_s"] - 60.0) < 1e-6
                   and by.get("負例露出", {}).get("n_false") == 2 and abs(by["負例露出"]["exposure_s"] - 593.0) < 1e-6 and neg["n_false"] == 3))
        main_ev, side, rates = S.summarize(events, True)
        ok.append(("N2 計画外の車は主要評価に入らず別掲 1 件", len(side["unplanned"]) == 1 and side["unplanned"][0]["clip"] == "eval_field_clip09002" and all(not e.get("unplanned") for e in main_ev)))
        ok.append(("N3 計画の本（A-01）は主要評価に 1 件", len(main_ev) == 1 and main_ev[0]["take_id"] == "A-01"))
        # N4: 結果 md（本番の main() を予測 csv と注釈 csv で走らせる）
        ann = tmp / "ann.csv"
        with open(ann, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        def write_pred(p: Path, pr: dict):
            with open(p, "w", encoding="utf-8") as f:
                for clip, fr in pr.items():
                    for k, dets in sorted(fr.items()):
                        for c, az, el, d in dets:
                            f.write(f"{clip},{k},{c},0,{az},{el},nan\n")
        p1 = tmp / "pred.csv"; write_pred(p1, pred)
        out1 = tmp / "score.md"
        sys.argv = ["step20", "--pred", str(p1), "--ann", str(ann), "--out", str(out1), "--include-front"]
        rc = S.main()
        t1 = out1.read_text(encoding="utf-8")
        ok.append(("N4 結果 md に G・負例露出・「合格線は置かない」（#70 r2・合否の文字なし）・計画外の行", rc == 0 and "負例テイク G" in t1 and "負例露出（連続録音）" in t1 and "合格線は置かない" in t1 and "**不合格**" not in t1 and "**合格**" not in t1 and "計画外の事象" in t1))
        pred0 = {"eval_field_clip09001": pred["eval_field_clip09001"], "eval_field_clip09002": {}, "eval_field_clip00101": {}}
        siren_frames(pred0["eval_field_clip09002"], 3010, 3016)       # 露出の中の発火は計画外の窓の中だけ → 0 件
        p0 = tmp / "pred0.csv"; write_pred(p0, pred0)
        out0 = tmp / "score0.md"
        sys.argv = ["step20", "--pred", str(p0), "--ann", str(ann), "--out", str(out0), "--include-front"]
        rc0 = S.main()
        t0 = out0.read_text(encoding="utf-8")
        ok.append(("N4' 負例露出 0 件 / 593 s → 片側上限は出すが合否は出さない（#70 r2）・G は 1 件のまま", rc0 == 0 and "合格線は置かない" in t0 and "**不合格**" not in t0 and "負例テイク G（仕込んだ 60 秒 × 本数・場面の確認用）: 誤警告 1 件" in t0))
        ev_csv = out1.with_name(out1.stem + "_events.csv")
        hdr = ev_csv.read_text(encoding="utf-8").splitlines()[0]
        ok.append(("N5 events csv に unplanned 列", "unplanned" in hdr.split(",")))
        # N6（B13・#40）: フラグ無しでも前方を含む（既定）・--exclude-front で旧の除外
        out2 = tmp / "score2.md"
        sys.argv = ["step20", "--pred", str(p1), "--ann", str(ann), "--out", str(out2)]
        rc2 = S.main(); t2 = out2.read_text(encoding="utf-8")
        out3 = tmp / "score3.md"
        sys.argv = ["step20", "--pred", str(p1), "--ann", str(ann), "--out", str(out3), "--exclude-front"]
        rc3 = S.main(); t3 = out3.read_text(encoding="utf-8")
        ok.append(("N6 既定で前方を含む（#40）・--exclude-front で除外", rc2 == 0 and "前方を含む（#40・既定）" in t2 and rc3 == 0 and "前方 F を除外（--exclude-front）" in t3))
        # N7（B11・#69 r20）: 至近の境目の感度の表がある（1.3／1.5／1.7）。A-01 は横 2.0 m → 1.3／1.5／1.7 のどれでも caution
        ok.append(("N7 至近の境目の感度の表（1.3／1.5（主）／1.7）", "至近の境目の感度" in t2 and "| 1.3 m |" in t2 and "| 1.5 m（主） |" in t2 and "| 1.7 m |" in t2))
        ev13, _, _ = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 7.5, True, t3=1.3)
        ok.append(("N7' t3 を渡しても境目は呼んだあと 1.5 に戻る", S.V2.T3 == 1.5 and any(e["gt_tier"] == "caution" for e in ev13)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
