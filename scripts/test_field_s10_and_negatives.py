# -*- coding: utf-8 -*-
"""敵対レビュー（`md/audit/敵対的レビュー_Opus55_2026-10-25_Opusの実装.md`）の重大 1・重大 2・軽 1〜3 の直しの試験（2026-10-25 Opus・Fable の確認待ち）。

架空の行と架空の録音（24 kHz・4 ch・小さい雑音）だけを使う。評価の録音・実録の読みは使わない。
  N1 重大 1: 記録シートは三脚 180° の本の direction を現場の向き（台本の向きを裏返す・fieldDirection）で書く。
     台本どおりの C2・H2・A2 の 0°／180° の 6 本を step19d → step19c の S10 に通して、向きの誤りが 0 件。本当に向きが違う本は誤り。
  N2 重大 2: 夜の命令のとおり step19d → step19b event → step19b negative（none の行がある原本だけを 1 本ずつ・--append）→ step19c。
     ann_all.csv の車の事象は 1 回だけ scored・none の行は全部 区分と take_id つき・step19c は誤りなし。
  N3 重大 2: step19b negative に正例の原本を渡しても負例に切らない（none の行を作らない）。raw/ を丸ごと渡しても正例の本は切らない。
  N4 軽 1: 三脚の向き 90／270 はクラクション E2 だけ（A2・歩く比較・台本ID の無い本は止める）。
  N5 軽 2: E2 の本のマイクの欄が片方だけでも、夜の step19d --video-only は止まらない（知らせを出して読む準備の表を書く）。注釈を作るときは止まる。
  N6 軽 3: 同じテイクの数の欄は数で比べる（2 と 2.0・-3 と -3.0 を不一致にしない）。
  N7 歩く比較の台本ID「W01-静止」「W01-歩行」（記録シートの形）を S10 が組 W01 として数える・同じ状態の 2 回録りと状態の食い違いは誤り
     （本物の 10/22 で 16 本が「台本に無い」になっていた・2026-10-25）。
  N8 10/15 の本（台本ID が旧の番号・条件ID が対応表の新しい番号）は条件ID で照合・条件ID が空の旧の番号は注意だけ（step19d が条件ID を通す）。

使い方: python scripts/test_field_s10_and_negatives.py
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
SCRIPTS = ROOT / "scripts"

import numpy as np                                  # noqa: E402
import soundfile as sf                              # noqa: E402

import step19d_field_csv_to_ann as conv             # noqa: E402
import step19c_ann_validate as val                  # noqa: E402
from _field_orientation import tripod_heading       # noqa: E402

PLAN = "A=32,B=4,C=16,D=0,E=8,H=60,G=20,歩行=32"


def field_dir(script_dir: str, heading: str) -> str:
    """記録シートの fieldDirection と同じ（三脚 180° なら台本の向きを裏返す）"""
    return {"前→後": "後→前", "後→前": "前→後"}[script_dir] if heading == "180" else script_dir


def ses(sid: str, kubun: str, mount: str = "三脚") -> dict:
    return {"session_id": sid, "用途": "最終評価", "区分": kubun, "マイク高さ_cm": "160" if mount == "三脚" else "175", "装着": mount,
            "校正原本": "261029_001.WAV", "LAeq_dB": "50", "暗騒音区間_秒": "5-60", "点検原本": "261029_002.WAV", "点検方式": "4方位"}


def write_csv(path: Path, rows: list):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def wav(path: Path, sec: float, seed: int):
    rng = np.random.default_rng(seed)
    sf.write(str(path), rng.normal(0, 1e-3, (int(sec * 24000), 4)).astype("float32"), 24000, subtype="PCM_24")


def run(args: list, cwd: Path):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(cwd))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


class S10AndNegatives(unittest.TestCase):
    def test_N1_s10_field_direction_180(self):
        E = val.expected_takes()
        sessions = [ses("20261022_C1", "C"), ses("20261022_H1", "H"), ses("20261029_A1", "A")]
        ev = []
        for i, sid_ in enumerate(["C2-01", "C2-02", "H2-01", "H2-02", "A2-01", "A2-02"], 1):
            e = E[sid_]; h = "180" if e["side"] == 180 else "0"
            row = {"session_id": {"C": "20261022_C1", "H": "20261022_H1", "A": "20261029_A1"}[e["kind"]], "take_id": str(i), "event_id": "1",
                   "class": {"C": "bike_bell", "H": "bicycle", "A": "car_drive"}[e["kind"]], "原本": f"x_{i:03d}.WAV",
                   "象限": "B" if e["direction"] == "後→前" else "F", "ラップ秒": "15", "状態": "静止", "三脚の向き": h, "台本ID": sid_,
                   "direction": field_dir(e["direction"], h), "line_m": f"{e['line']:g}", "横距離m": f"{e['line']:g}"}
            if e["kind"] == "H":
                row["走行速さ"] = e["speed"]
            if e["kind"] == "A":
                row["速度"] = e["speed"]
            if e["kind"] == "C":
                row["ベル目印_m"] = str(e["bell"])
            ev.append(row)
        rows, _ = conv.convert(sessions, ev, 6, 0, None)
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows, val.PLAN_DEFAULT)
        self.assertFalse([e for e in val.errs if "向き" in e], val.errs)            # 旧は 180° の 3 本が「向き違い」
        bad = [dict(ev[1], take_id="7", direction=E["C2-02"]["direction"])]        # 180° なのに台本の向きのまま（現場の向きに直していない）＝本当の食い違い
        rows2, _ = conv.convert(sessions, bad, 6, 0, None)
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows2, val.PLAN_DEFAULT)
        self.assertEqual(sum("向き" in e and "マイクから見ると" in e for e in val.errs), 1, val.errs)

    def _night(self, td: Path, extra_unannotated: bool):
        raw = td / "raw"; raw.mkdir()
        wav(raw / "261029_001.WAV", 70, 1); wav(raw / "261029_002.WAV", 20, 2)            # 校正・点検
        wav(raw / "261029_010.WAV", 30, 3)                                                  # A2-01（正例）
        wav(raw / "261029_020.WAV", 60, 4)                                                  # G-01（負例・none）
        if extra_unannotated:
            wav(raw / "261029_099.WAV", 20, 5)                                              # 記録に無い録音（押し間違い・練習）
        sessions = [ses("20261029_A1", "A"), ses("20261029_G1", "G")]
        ev = [{"session_id": "20261029_A1", "take_id": "1", "event_id": "1", "class": "car_drive", "原本": "261029_010.WAV", "象限": "B", "ラップ秒": "15",
               "状態": "静止", "三脚の向き": "0", "台本ID": "A2-01", "direction": "後→前", "line_m": "2.0", "横距離m": "2.0", "速度": "15"},
              {"session_id": "20261029_G1", "take_id": "1", "event_id": "1", "class": "none", "原本": "261029_020.WAV", "象限": "", "ラップ秒": "",
               "状態": "静止", "三脚の向き": "0", "台本ID": "G-01"}]
        write_csv(td / "session.csv", sessions); write_csv(td / "events.csv", ev)
        out = td / "out"
        rc, o = run([SCRIPTS / "step19d_field_csv_to_ann.py", "--session", td / "session.csv", "--events", td / "events.csv", "--audio-dir", raw,
                     "--out", out / "ann_orig.csv"], ROOT)
        self.assertEqual(rc, 0, o[-1500:])
        common = ["--ann", out / "ann_orig.csv", "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", out / "clips", "--ann-out", out / "ann_all.csv"]
        rc, o = run([SCRIPTS / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw, *common] + (["--allow-unmatched"] if extra_unannotated else []), ROOT)
        self.assertEqual(rc, 0, o[-1500:])
        return raw, out, common

    def test_N2_night_chain_negatives_per_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw, out, common = self._night(td, extra_unannotated=True)
            rc, o = run([SCRIPTS / "step19b_realsmoke_cut.py", "--mode", "negative", "--in", raw / "261029_020.WAV", *common, "--append"], ROOT)   # 記録シート V127 の形
            self.assertEqual(rc, 0, o[-1500:])
            with (out / "ann_all.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            cars = [r for r in rows if r["class"] == "car_drive" and r.get("scored") == "1"]
            nones = [r for r in rows if r["class"] == "none"]
            self.assertEqual(len(cars), 1, [(r["clip_id"], r["class"], r.get("scored")) for r in rows])          # 同じ車を 2 回採点しない
            self.assertTrue(nones and all(r.get("take_id") and r.get("区分") == "G" for r in nones), nones)
            self.assertFalse(any(r.get("orig_file", "").lower().startswith("261029_010") for r in nones))     # 正例の本から none の行を作らない
            self.assertEqual(len({r["clip_id"] for r in rows}), len({(r["clip_id"]) for r in rows}))
            rc, o = run([SCRIPTS / "step19c_ann_validate.py", "--ann", out / "ann_all.csv", "--cut", "--plan", PLAN, "--opportunity", "X=8"], ROOT)
            self.assertNotIn("[ERROR]", o, o[-2000:])
            self.assertEqual(rc, 0, o[-2000:])

    def test_N3_negative_mode_skips_positive_takes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw, out, common = self._night(td, extra_unannotated=False)
            n_before = len(list((out / "clips").glob("*.flac")))
            rc, o = run([SCRIPTS / "step19b_realsmoke_cut.py", "--mode", "negative", "--in", raw / "261029_010.WAV", *common, "--append"], ROOT)   # 正例の原本を負例に渡した
            self.assertEqual(rc, 0, o[-1500:])
            self.assertIn("負例（class=none）の注釈が無い", o)
            self.assertEqual(len(list((out / "clips").glob("*.flac"))), n_before)                              # 新しいクリップを作らない
            rc, o = run([SCRIPTS / "step19b_realsmoke_cut.py", "--mode", "negative", "--in", raw, *common, "--append"], ROOT)   # 旧の命令の形（raw/ を丸ごと）
            self.assertEqual(rc, 0, o[-1500:])
            with (out / "ann_all.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            nones = [r for r in rows if r["class"] == "none"]
            self.assertTrue(nones and all(r.get("orig_file", "").lower().startswith("261029_020") for r in nones), [(r["clip_id"], r.get("orig_file")) for r in nones])
            self.assertEqual(sum(r["class"] == "car_drive" and r.get("scored") == "1" for r in rows), 1)

    def test_N7_walking_script_ids_with_state(self):
        """記録シートは歩く比較の台本ID を「W01-静止」「W01-歩行」で書く（HTML の台本の定義）。S10 は組の番号 W01（2 本）として数える。
        2026-10-25: 本物の 10/22 で 16 本が「台本に無い」になっていた（台本ID を注釈に通したあとに表に出た）"""
        base = {"session_id": "20261022_W1", "event_id": "1", "class": "bike_bell", "象限": "B", "ラップ秒": "15", "三脚の向き": "180",
                "pair_id": "W01", "direction": "後→前", "line_m": "2.0", "横距離m": "3.0", "歩く人の線_m": "5.0", "ベル目印_m": "-6"}
        sessions = [ses("20261022_W1", "歩行対比", "ヘルメット")]
        ok_rows = [dict(base, take_id="1", 状態="静止", 台本ID="W01-静止", 原本="w_001.WAV"), dict(base, take_id="2", 状態="歩行", 台本ID="W01-歩行", 原本="w_002.WAV")]
        rows, _ = conv.convert(sessions, ok_rows, 6, 0, None)
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows, val.PLAN_DEFAULT)
        self.assertFalse([e for e in val.errs if "台本に無い" in e or "W01" in e], val.errs)
        bad = [dict(base, take_id="3", 状態="静止", 台本ID="W01-静止", 原本="w_003.WAV"), dict(base, take_id="4", 状態="静止", 台本ID="W01-静止", 原本="w_004.WAV"),
               dict(base, take_id="5", 状態="静止", 台本ID="W03-歩行", 原本="w_005.WAV")]
        rows2, _ = conv.convert(sessions, bad, 6, 0, None)
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows2, val.PLAN_DEFAULT)
        self.assertEqual(sum("W01" in e and "同じ状態" in e for e in val.errs), 1, val.errs)        # 静止を 2 回
        self.assertEqual(sum("W03-歩行" in e and "状態 '静止'" in e for e in val.errs), 1, val.errs)  # 台本ID の状態と行の状態が違う

    def test_N8_old_script_ids_use_condition_id(self):
        """記録シートの書き出しで 10/15 の本は台本ID が旧の番号（C-05）・条件ID が対応表の新しい番号（C2-13）。S10 は条件ID を先に見る。
        条件ID が空の旧の番号（C-01 など＝対応表で数えない本）は誤りではなく注意（2026-10-25・書き出しで 36 本が「台本に無い」になっていた）"""
        base = {"session_id": "20261015_C1", "event_id": "1", "class": "bike_bell", "象限": "B", "ラップ秒": "15", "状態": "静止", "三脚の向き": "0",
                "direction": "後→前", "line_m": "3.0", "横距離m": "3.0", "ベル目印_m": "-5"}
        rows_in = [dict(base, take_id="5", 台本ID="C-05", 条件ID="C2-13", 原本="c_005.WAV"),            # C2-13 = 3.0 m・ベル 5 m・後→前・左（0°）
                   dict(base, take_id="1", 台本ID="C-01", 条件ID="", 原本="c_001.WAV", line_m="1.0", 横距離m="1.0")]
        rows, _ = conv.convert([ses("20261015_C1", "C")], rows_in, 6, 0, None)
        self.assertEqual({r["条件ID"] for r in rows}, {"C2-13", ""})                                    # step19d が条件ID を注釈に通す
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows, val.PLAN_DEFAULT)
        self.assertFalse(val.errs, val.errs)
        self.assertEqual(sum("C-01" in w and "対応表で数えない" in w for w in val.warns), 1, val.warns)
        bad = [dict(base, take_id="6", 台本ID="C-06", 条件ID="C2-13", 原本="c_006.WAV", line_m="1.0", 横距離m="1.0")]   # 条件ID は 3.0 m なのに 1.0 m の線
        rows2, _ = conv.convert([ses("20261015_C1", "C")], bad, 6, 0, None)
        val.errs.clear(); val.warns.clear()
        val.check_script_ids(rows2, val.PLAN_DEFAULT)
        self.assertEqual(sum("C2-13" in e and "線" in e for e in val.errs), 1, val.errs)

    def test_N4_quarter_turns_only_for_e2(self):
        self.assertEqual(tripod_heading({"三脚の向き": "90", "台本ID": "E2-02"}), 90)
        self.assertEqual(tripod_heading({"三脚の向き": "270", "台本ID": "e2-04"}), 270)
        for row in ({"三脚の向き": "90", "台本ID": "W01"}, {"三脚の向き": "270", "台本ID": "A2-03"}, {"三脚の向き": "90"}, {"三脚の向き": "270", "台本ID": "C2-01"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                tripod_heading(row)
        self.assertEqual(tripod_heading({"三脚の向き": "180", "台本ID": "W09"}), 180)

    def _e2_rows(self, my_blank: bool = False):
        r = {"session_id": "20261029_E1", "take_id": "2", "event_id": "1", "class": "horn", "原本": "261029_102.WAV", "象限": "L", "ラップ秒": "",
             "状態": "静止", "装着": "三脚", "三脚の向き": "90", "台本ID": "E2-02", "x_m": "-3", "y_m": "2.0", "高さ_m": "0.8",
             "マイク_x_m": "" if my_blank else "0", "マイク_y_m": "2.0"}
        a2 = {"session_id": "20261029_A1", "take_id": "1", "event_id": "1", "class": "car_drive", "原本": "261029_021.WAV", "象限": "B", "ラップ秒": "",
              "状態": "静止", "装着": "三脚", "三脚の向き": "0", "台本ID": "A2-01", "マイク_x_m": "0", "マイク_y_m": "0", "動画原本": "a2.mov",
              "direction": "後→前", "line_m": "2.0", "横距離m": "2.0"}
        return [r, a2]

    def test_N5_video_only_survives_half_mic_field(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_csv(td / "session.csv", [ses("20261029_E1", "E"), ses("20261029_A1", "A")])
            write_csv(td / "events.csv", self._e2_rows(my_blank=True))
            args = [SCRIPTS / "step19d_field_csv_to_ann.py", "--session", td / "session.csv", "--events", td / "events.csv", "--out", td / "ann_orig.csv"]
            rc, o = run(args + ["--video-only"], ROOT)
            self.assertEqual(rc, 0, o[-1500:])
            self.assertNotIn("Traceback", o)
            self.assertIn("注釈を作るときに止まる所がある", o)
            self.assertTrue((td / "ann_orig_video_takes.csv").is_file())
            rc2, o2 = run(args, ROOT)                                                     # 注釈を作るときは止まる（記録を直してから）
            self.assertNotEqual(rc2, 0, o2[-800:])

    def test_N6_numeric_take_values_compare_as_numbers(self):
        a, _ = self._e2_rows()
        b = dict(a, event_id="2", **{"マイク_y_m": "2", "x_m": "-3.0"})
        rows, _ = conv.convert([ses("20261029_E1", "E")], [dict(a, ラップ秒="10"), dict(b, ラップ秒="12")], 6, 0, None)
        self.assertEqual({(r["mic_x_m"], r["mic_y_m"]) for r in rows}, {("0", "2")})
        c = dict(a, event_id="3", **{"マイク_y_m": "2.5"})
        with self.assertRaises(ValueError):                                                # 数として違えば今までどおり止める
            conv.convert([ses("20261029_E1", "E")], [dict(a, ラップ秒="10"), dict(c, ラップ秒="12")], 6, 0, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
