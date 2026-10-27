# -*- coding: utf-8 -*-
"""Astra H25-01 の受け入れ（クラクション E2 の三脚の向き 90°／270° とマイクの位置・2026-10-25 Opus・Fable の確認待ち）。

架空の 10/29（車の日）の記録シートの書き出し（Version 123 の形・名前なし）だけを使う。評価の録音・実録の読みは使わない。
  1 夜の表: E2 の 8 本（向き 0/90/180/270）と A2・歩く比較 W09 が混ざった記録で、step19d --video-only が止まらない（終了コード 0）。
    A2 の目印は ±12 の 13 点（#69 r38）・W09 は ±10 の 11 点のまま・E2 は読む準備の表に出ない（止まった音源）
  2 注釈: E2 の mic_x_m・mic_y_m = (0, 2)（マイク_x_m・マイク_y_m から）・台本ID と向きがそのまま残る・A2 は (0, 0)
  3 正解: step19d → step19b の切り出し → step19k で、マイクから見た方位が 前 0°・左 +90°・後 ±180°・右 −90°、水平距離 3.00／7.00 m
    3b 本番の CLI の既定（--range-m 10・--dir-window 2.5-1.5）でも同じ（敵対レビュー 軽 4・2026-10-25）
  4 原点の二重引き: x_m・y_m がマイクから見た位置のまま（Version 122 の形）なのにマイクの座標も入っている本、マイクの座標が抜けた本を step19c の S10 が誤りにする
    （2 つは別の文で見分ける・軽 4）。4b 置けなかったときの直し（y の差 0.4〜0.5 m）は警告だけ（敵対レビュー 中 2）
  5 旧の形: 向き 0/180 で マイク_* の列が無い本の mic_x_m・mic_y_m は今までどおり（三脚 = 0・0・W09 静止 = 2・7）

使い方: python scripts/test_field_e2_90_270.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

import step19d_field_csv_to_ann as convert          # noqa: E402
import step19b_realsmoke_cut as cut                  # noqa: E402
import step19k_field_perception as gt                # noqa: E402
import step19c_ann_validate as validate              # noqa: E402

# 台本の E2（記録シート Version 123・#69 r40）: 番号 → (位置, 向き, 象限, 距離)
E2 = {1: ("前", "180", "F", 3), 2: ("左", "90", "L", 3), 3: ("後", "0", "B", 3), 4: ("右", "270", "R", 3),
      5: ("前", "180", "F", 7), 6: ("左", "90", "L", 7), 7: ("後", "0", "B", 7), 8: ("右", "270", "R", 7)}
WANT_AZ = {"前": 0, "左": 90, "後": 180, "右": -90}      # 後は ±180（ラベルは 180 を −180 にする）


def session_rows():
    return [{"session_id": "20261029_E1", "用途": "最終評価", "区分": "E", "マイク高さ_cm": "160", "装着": "三脚",
             "校正原本": "261029_001.WAV", "LAeq_dB": "50.0", "暗騒音区間_秒": "5-60", "点検原本": "261029_002.WAV", "点検方式": "4方位"},
            {"session_id": "20261029_A1", "用途": "最終評価", "区分": "A", "マイク高さ_cm": "160", "装着": "三脚",
             "校正原本": "261029_001.WAV", "LAeq_dB": "50.0", "暗騒音区間_秒": "5-60", "点検原本": "261029_020.WAV", "点検方式": "4方位"},
            {"session_id": "20261029_W1", "用途": "最終評価", "区分": "歩行対比", "マイク高さ_cm": "175", "装着": "ヘルメット",
             "校正原本": "261029_001.WAV", "LAeq_dB": "50.0", "暗騒音区間_秒": "5-60", "点検原本": "261029_030.WAV", "点検方式": "4方位"}]


def e2_event(n: int, lap: str = "", x: str | None = None, y: str = "2.0", mic=("0", "2.0")):
    pos, h, q, d = E2[n]
    return {"session_id": "20261029_E1", "take_id": str(n), "event_id": "1", "class": "horn", "原本": f"261029_{100 + n:03d}.WAV",
            "象限": q, "ラップ秒": lap, "状態": "静止", "装着": "三脚", "三脚の向き": h, "台本ID": f"E2-{n:02d}",
            "x_m": x if x is not None else str(-d), "y_m": y, "高さ_m": "0.8", "マイク_x_m": mic[0], "マイク_y_m": mic[1],
            "拍手_録音側秒": "", "動画原本": "", "direction": "", "line_m": ""}


def other_events(lap: str = ""):
    a2 = {"session_id": "20261029_A1", "take_id": "1", "event_id": "1", "class": "car_drive", "原本": "261029_021.WAV", "象限": "B",
          "ラップ秒": lap, "状態": "静止", "装着": "三脚", "三脚の向き": "0", "台本ID": "A2-01", "x_m": "", "y_m": "", "高さ_m": "",
          "マイク_x_m": "0", "マイク_y_m": "0", "拍手_録音側秒": "", "動画原本": "a2_01.mov", "direction": "後→前", "line_m": "2.0", "横距離m": "2.0"}
    w09 = {"session_id": "20261029_W1", "take_id": "1", "event_id": "1", "class": "car_drive", "原本": "261029_031.WAV", "象限": "B",
           "ラップ秒": lap, "状態": "静止", "装着": "ヘルメット", "三脚の向き": "0", "台本ID": "W09", "pair_id": "P9", "x_m": "", "y_m": "", "高さ_m": "",
           "マイク_x_m": "", "マイク_y_m": "", "歩く人の線_m": "7.0", "歩く人の静止x_m": "2", "拍手_録音側秒": "", "動画原本": "w09.mov",
           "direction": "後→前", "line_m": "3.0", "横距離m": "4.0"}
    return [a2, w09]


def write_csv(path: Path, rows: list):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


class E2Tests(unittest.TestCase):
    def test_1_night_video_only_does_not_stop(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_csv(td / "session.csv", session_rows())
            write_csv(td / "events.csv", [e2_event(n) for n in range(1, 9)] + other_events())
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "step19d_field_csv_to_ann.py"), "--session", str(td / "session.csv"),
                                "--events", str(td / "events.csv"), "--out", str(td / "ann_orig.csv"), "--video-only"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            self.assertNotIn("Traceback", r.stderr)
            with (td / "ann_orig_video_takes.csv").open(encoding="utf-8") as f:
                vt = list(csv.DictReader(f))
            self.assertFalse((td / "ann_orig.csv").exists())                       # --video-only は注釈を書かない
        by = {(v["take"], v["cls"]): v for v in vt}
        self.assertNotIn(("261029_101.WAV", "horn"), by)                           # E2 は止まった音源で目印を読まない
        self.assertEqual(by[("261029_021.WAV", "car_drive")]["markers"], ",".join(str(x) for x in range(-12, 13, 2)))   # A2 = ±12 の 13 点（後→前）
        self.assertEqual(by[("261029_031.WAV", "car_drive")]["markers"], ",".join(str(x) for x in range(-10, 11, 2)))   # W09 = ±10 の 11 点

    def test_2_annotation_carries_mic_and_script_id(self):
        rows, warns = convert.convert(session_rows(), [e2_event(n, "10") for n in range(1, 9)] + other_events("30"), 6, 0, None)
        by = {r["take_id"]: r for r in rows}
        for n in range(1, 9):
            r = by[f"20261029_E1/{n:02d}"]
            self.assertEqual((r["mic_x_m"], r["mic_y_m"]), ("0", "2"), n)
            self.assertEqual((r["x_m"], r["y_m"]), (str(-E2[n][3]), "2.0"), n)    # 音源は現場の座標のまま（ここでは引かない）
            self.assertEqual(r["三脚の向き"], E2[n][1], n)
            self.assertEqual(r["台本ID"], f"E2-{n:02d}", n)
        self.assertEqual((by["20261029_A1/01"]["mic_x_m"], by["20261029_A1/01"]["mic_y_m"]), ("0", "0"))
        self.assertEqual((by["20261029_W1/01"]["mic_x_m"], by["20261029_W1/01"]["mic_y_m"]), ("2", "7"))
        self.assertIn("台本ID", convert.OUT_COLS)

    def test_3_ground_truth_four_directions(self):
        self._ground_truth(dict())

    def test_3b_ground_truth_production_args(self):
        """敵対レビュー 軽 4（2026-10-25）: 本番の CLI の既定（--range-m 10・--dir-window 2.5-1.5＝新しい道）でも同じ正解"""
        self._ground_truth(dict(range_m=10.0, dirwin=(2.5, 1.5)))

    def _ground_truth(self, kw: dict):
        rows, _ = convert.convert(session_rows(), [e2_event(n, "10") for n in range(1, 9)], 6, 0, None)
        ann, reads = [], []
        for n, r in zip(range(1, 9), sorted(rows, key=lambda x: x["take_id"])):
            ann += cut.rebase_rows([r], 5.0, 10.0, r["orig_file"], f"eval_field_clip99{n:03d}")
            reads += [dict(take=r["orig_file"], frame="", cls="horn", track="1", x="", y="", z="", t=str(t)) for t in (10, 11)]   # 鳴っている区間（録音の秒）
        with tempfile.TemporaryDirectory() as td:
            gt.build(reads, ann, Path(td), 1.6, None, **kw)
            for n in range(1, 9):
                lines = [s for s in (Path(td) / "gt" / f"eval_field_clip99{n:03d}.csv").read_text(encoding="utf-8").splitlines() if s]
                pos, h, q, d = E2[n]
                with self.subTest(n=n, pos=pos, heading=h):
                    self.assertTrue(lines, "正解が 0 行")
                    for s in lines:
                        k, c, trk, az, el, dist = s.split(",")
                        self.assertEqual(int(c), 1)                                   # horn
                        self.assertTrue(50 <= int(k) <= 60, k)                        # 録音 10〜11 s → クリップ 5〜6 s
                        if pos == "後":
                            self.assertEqual(abs(int(az)), 180, s)
                        else:
                            self.assertEqual(int(az), WANT_AZ[pos], s)
                        self.assertAlmostEqual(float(dist), float(d), places=2)        # 水平距離（高さの差は入れない）

    def test_4_double_origin_and_missing_mic_rejected(self):
        bad = [e2_event(3, "10", x="-3", y="0"),                                      # Version 122 の形（マイクから見た位置）なのにマイクの座標も入っている
               dict(e2_event(4, "10"), **{"マイク_x_m": "", "マイク_y_m": ""})]        # マイクの座標が抜けた（音源は現場の座標 (−3, +2.0)）
        good = [e2_event(1, "10"), e2_event(7, "10")]
        rows, _ = convert.convert(session_rows(), bad + good, 6, 0, None)
        validate.errs.clear(); validate.warns.clear()
        validate.check_script_ids(rows, validate.PLAN_DEFAULT)
        errs = list(validate.errs)
        self.assertEqual(sum("20261029_E1/03" in e and "原点を二重に引く" in e for e in errs), 1, errs)
        self.assertEqual(sum("20261029_E1/04" in e and "マイクの座標" in e and "抜けた" in e for e in errs), 1, errs)
        self.assertFalse(any("20261029_E1/03" in e and "抜けた" in e for e in errs), errs)          # 軽 4: 2 つの取り違えを見分ける（文を分けた）
        self.assertFalse(any("20261029_E1/04" in e and "二重" in e for e in errs), errs)
        self.assertFalse(any(("20261029_E1/01" in e or "20261029_E1/07" in e) for e in errs), errs)

    def test_4c_coordinate_direction_vs_quadrant(self):
        """Astra MAP25-06（10/25）: E2-03（後・三脚 0°・マイク (0, 2)）の音源の x を −3 → +3 と書き間違えると、距離は 3 m のままで誤りにならないが、
        座標から求めた向き（前）と象限（後）の食い違いを注意に出す。正しい 8 本には出さない"""
        rows, _ = convert.convert(session_rows(), [e2_event(n, "10") for n in range(1, 9)], 6, 0, None)
        validate.errs.clear(); validate.warns.clear()
        validate.check_script_ids(rows, validate.PLAN_DEFAULT)
        self.assertFalse([w for w in validate.warns if "座標" in w], validate.warns)
        rows2, _ = convert.convert(session_rows(), [e2_event(3, "10", x="3")], 6, 0, None)
        validate.errs.clear(); validate.warns.clear()
        validate.check_script_ids(rows2, validate.PLAN_DEFAULT)
        self.assertFalse(validate.errs, validate.errs)
        self.assertEqual(sum("20261029_E1/03" in w and "求めた向き F" in w and "象限 B" in w for w in validate.warns), 1, validate.warns)

    def test_4b_placement_corrections_are_warnings(self):
        """敵対レビュー 中 2（2026-10-25）: 「置けなかったときだけ直す」で正しく直した本（三脚を y = 1.5 にしか置けない・車を線から 0.4 m 外に止めた）は
        誤りではなく警告。台本の距離から 0.75 m 以内なら距離の誤りも出ない"""
        fixed = [e2_event(1, "10", mic=("0", "1.5")),                                 # 差 0.5 m・距離 3.04 m
                 e2_event(5, "10", y="2.4")]                                          # 差 0.4 m・距離 7.01 m
        rows, _ = convert.convert(session_rows(), fixed, 6, 0, None)
        validate.errs.clear(); validate.warns.clear()
        validate.check_script_ids(rows, validate.PLAN_DEFAULT)
        self.assertFalse([e for e in validate.errs if "20261029_E1/01" in e or "20261029_E1/05" in e], validate.errs)
        self.assertEqual(sum(("20261029_E1/01" in w or "20261029_E1/05" in w) and "置けなかったときの直し" in w for w in validate.warns), 2, validate.warns)

    def test_5_old_form_unchanged(self):
        legacy = [{k: v for k, v in e.items() if k not in ("マイク_x_m", "マイク_y_m")} for e in other_events("30")]
        rows, _ = convert.convert(session_rows(), legacy, 6, 0, None)
        by = {r["take_id"]: r for r in rows}
        self.assertEqual((by["20261029_A1/01"]["mic_x_m"], by["20261029_A1/01"]["mic_y_m"]), ("0", "0"))
        self.assertEqual((by["20261029_W1/01"]["mic_x_m"], by["20261029_W1/01"]["mic_y_m"]), ("2", "7"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
