# -*- coding: utf-8 -*-
"""field_calib_check.py の試験（架空の録音と騒音計の記録だけ・2026-10-25 Opus・Fable の確認待ち）。

  C1 録音の録り始め（bext）と騒音計の時刻で合わせ、相関で正しいずれ（騒音計の時計が 2 秒進み）を見つける
  C2 重なった秒の騒音計の値のエネルギー平均（ふつうの平均ではない）
  C3 判定: 画面の値との差が ±0.5 dB 以内なら画面・超えたら 1 秒の記録・画面の値が無ければ 1 秒の記録（終了コード 0／3／3）
  C4 相関が弱い（騒音計の値が録音と無関係）ときは照合できないので画面の値（知らせる）
  C5 フォルダを渡すと、中の記録のうち一番よく合うものを選ぶ
 敵対レビュー 10/25 の中 1・軽 7 の直し:
  C6 録音に bext が無い → 照合できない（画面の値・終了コード 5）。画面の値も無ければ使う値なし
  C7 選んだずれが時計どおりから 10 s 以上（時計が 15 s ずれ）→ 要確認（画面の値・終了コード 4）
  C8 騒音計の記録が窓の途中で終わる → 要確認（窓の一部しか重ならない）
  C9 騒音計の記録の途中の抜け: 時刻で引くので秒がずれない（窓の前の抜けは影響なし・窓の中の抜けは要確認）
  C10 画面の値が無いときも相関とずれを調べる（弱い → 照合できない・ずれが大きい → 要確認・使う値なし）
  C11 終了コード（0 画面・3 1 秒の記録・4 要確認・5 照合できない・2 入力の誤り。6 止める は C16）と JSON
  C12 窓は整数の秒（5.5-60 は止める・5.0-60.0 は受ける）・録音より長い窓は止める
  C13 フォルダの .TXT は大文字・小文字を問わず 1 回ずつ（Windows で 2 回数えない）
  C14 相関の下限の既定（本物の 10/15 の正しい合わせ 0.711 を通し、0.5 より高い）と、下限を上げると照合できない
  C15 値どうしの相関が高くても、1 秒ごとの変化が合わなければ要確認（ゆっくり変わる暗騒音がたまたま似る誤りを止める）
 本人 10/25 15:56 の決定 (b):
  C16 差の大きさが 1.4 dB（--stop-db）を超えたら止める（使う値・ゲインなし・終了コード 6）。0.5〜1.4 dB は替える。秒の合わせの判定が先
使い方: python scripts/test_field_calib_check.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import field_calib_check as fc        # noqa: E402

START = dt.datetime(2026, 10, 29, 10, 0, 0)


def write_wav(path: Path, seconds: int = 70, sr: int = 48000, seed: int = 0, levels=None, bext: bool = True):
    """4 ch・16 ビットの WAV（bext に録り始め 2026-10-29 10:00:00・bext=False なら bext なし）。W は 1 秒ごとに大きさの違う雑音。"""
    rng = np.random.default_rng(seed)
    lv = np.cumsum(rng.normal(0, 1.5, seconds)) if levels is None else np.asarray(levels, dtype=float)   # 1 秒ごとの大きさ [dB] の揺れ
    w = np.concatenate([rng.normal(0, 1, sr) * 0.01 * 10 ** (l / 20) for l in lv])
    x = np.stack([w, w * 0.3, w * 0.1, w * 0.3], axis=1)
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 4, sr, sr * 8, 8, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if bext:
        b = bytearray(602); b[320:338] = START.strftime("%Y-%m-%d%H:%M:%S").encode("latin1")
        body += b"bext" + struct.pack("<I", len(b)) + bytes(b)
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_slm(path: Path, values_by_time: dict):
    with open(path, "w", encoding="utf-8") as f:
        for n, (t, v) in enumerate(sorted(values_by_time.items()), 1):
            f.write(f"{n}\t{t:%Y/%m/%d}\t{t:%H:%M:%S}.000\t{v:.1f}\n")


def slm_values(rec, shift: int, seed: int, t0=dt.datetime(2026, 10, 29, 9, 59, 40), rows: int = 120, noise: float = 0.05) -> dict:
    """騒音計の記録: t0 から rows 行の暗騒音（45 dB 前後）に、録音の i 秒 = 騒音計の START + i + shift の値（録音 + 50 dB）を重ねる。"""
    rng = np.random.default_rng(seed)
    vals = {t0 + dt.timedelta(seconds=n): 45.0 + rng.normal(0, 1) for n in range(rows)}
    for i, l in enumerate(rec):
        vals[START + dt.timedelta(seconds=i + shift)] = l + 50.0 + rng.normal(0, noise)
    return vals


def quiet_main(argv) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return fc.main(argv)


class CalibTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(); cls.dir = Path(cls.td.name)
        cls.wav = cls.dir / "261029_001.WAV"; write_wav(cls.wav)
        cls.rec = fc.rec_levels(cls.wav)
        rng = np.random.default_rng(1)
        t0 = dt.datetime(2026, 10, 29, 9, 59, 40)               # 騒音計の記録は録音より前から 120 行
        vals = {t0 + dt.timedelta(seconds=n): 45.0 + rng.normal(0, 1) for n in range(120)}
        for i, l in enumerate(cls.rec):                         # 騒音計の時計は 2 秒進み: 録音の i 秒 = 騒音計の 10:00:02 + i
            vals[START + dt.timedelta(seconds=i + 2)] = l + 50.0 + rng.normal(0, 0.05)
        cls.vals = vals
        cls.slm = cls.dir / "slm_20261029_095940.TXT"; write_slm(cls.slm, vals)
        cls.slm_noise = cls.dir / "slm_20261029_other.TXT"
        write_slm(cls.slm_noise, {t0 + dt.timedelta(seconds=n): 45.0 + rng.normal(0, 3) for n in range(120)})
        cls.wav_nobext = cls.dir / "261029_002.WAV"; write_wav(cls.wav_nobext, sr=24000, bext=False)   # 同じ大きさの揺れ（seed 0）・bext なし
        cls.base = fc.check(cls.wav, [cls.slm], None, (5, 60))["laeq_log"]

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def slm_file(self, name: str, vals: dict) -> Path:
        p = self.dir / name; write_slm(p, vals); return p

    def test_c1_alignment(self):
        r = fc.check(self.wav, [self.slm], None, (5, 60))
        self.assertEqual(r["shift_from_clock_s"], 2)
        self.assertGreater(r["corr"], 0.95)
        self.assertEqual(r["n_overlap"], 55)
        self.assertEqual(r["slm_time"], "10:00:07〜10:01:01")

    def test_c2_energy_mean(self):
        r = fc.check(self.wav, [self.slm], None, (5, 60))
        want = fc.energy_mean(self.rec[5:60] + 50.0)
        self.assertAlmostEqual(r["laeq_log"], want, delta=0.05)
        self.assertGreater(fc.energy_mean([40, 50]), 45.0)       # エネルギー平均はふつうの平均より大きい

    def test_c3_decision(self):
        base = self.base
        near = fc.check(self.wav, [self.slm], round(base + 0.3, 1), (5, 60))
        far = fc.check(self.wav, [self.slm], round(base - 0.8, 1), (5, 60))
        none = fc.check(self.wav, [self.slm], None, (5, 60))
        self.assertEqual(near["use"], "画面"); self.assertEqual(near["laeq_use"], round(base + 0.3, 1))
        self.assertEqual(far["use"], "1秒の記録"); self.assertEqual(far["laeq_use"], round(base, 1))
        self.assertEqual(none["use"], "1秒の記録")
        self.assertEqual((near["judgment"], near["exit_code"]), ("画面", 0))
        self.assertEqual((far["judgment"], far["exit_code"]), ("1秒の記録", 3))
        self.assertEqual((none["judgment"], none["exit_code"]), ("1秒の記録", 3))
        for r in (near, far, none):
            self.assertTrue(np.isfinite(r["gain_db"]))

    def test_c4_weak_correlation(self):
        r = fc.check(self.wav, [self.slm_noise], 48.0, (5, 60))
        self.assertLess(r["corr"], 0.5)
        self.assertEqual(r["use"], "画面"); self.assertIn("合わせられない", r["status"])
        self.assertEqual((r["judgment"], r["exit_code"]), ("照合できない", 5))

    def test_c5_folder_picks_best(self):
        r = fc.check(self.wav, [self.slm_noise, self.slm], 48.0, (5, 60))
        self.assertEqual(r["slm"], self.slm.name)

    def test_c6_no_bext(self):
        self.assertIsNone(fc.wav_start(self.wav_nobext))
        r = fc.check(self.wav_nobext, [self.slm], 48.0, (5, 60))
        self.assertEqual((r["judgment"], r["exit_code"], r["use"], r["laeq_use"]), ("照合できない", 5, "画面", 48.0))
        self.assertIn("bext", r["status"]); self.assertNotIn("corr", r)
        self.assertTrue(np.isfinite(r["gain_db"]))
        r2 = fc.check(self.wav_nobext, [self.slm], None, (5, 60))      # 画面の値も無い → 使う値なし（1 秒の記録に逃げない）
        self.assertEqual((r2["judgment"], r2["use"], r2["laeq_use"], r2["gain_db"]), ("照合できない", "", None, None))

    def test_c7_clock_far(self):
        sp = self.slm_file("slm_clock15.TXT", slm_values(self.rec, 15, seed=7, rows=160))
        r = fc.check(self.wav, [sp], round(self.base + 0.1, 1), (5, 60))
        self.assertEqual(r["shift_from_clock_s"], 15); self.assertEqual(r["n_overlap"], 55)
        self.assertEqual((r["judgment"], r["exit_code"], r["use"]), ("要確認", 4, "画面"))
        self.assertIn("10 s 以上", r["status"])

    def test_c8_record_ends_mid_window(self):
        vals = {t: v for t, v in self.vals.items() if t < START + dt.timedelta(seconds=2 + 50)}   # 録音の 50 s 目で記録が終わる
        sp = self.slm_file("slm_short.TXT", vals)
        r = fc.check(self.wav, [sp], round(self.base + 0.1, 1), (5, 60))
        self.assertEqual(r["shift_from_clock_s"], 2); self.assertEqual((r["n_overlap"], r["n_window"]), (45, 55))
        self.assertEqual((r["judgment"], r["exit_code"], r["use"]), ("要確認", 4, "画面"))
        self.assertIn("45 秒しか", r["status"])

    def test_c9_gaps_do_not_shift_seconds(self):
        before = {t: v for t, v in self.vals.items() if not (dt.datetime(2026, 10, 29, 9, 59, 50) <= t < START)}   # 録音の前の 10 行が抜け
        r = fc.check(self.wav, [self.slm_file("slm_gap_before.TXT", before)], round(self.base + 0.1, 1), (5, 60))
        self.assertEqual((r["shift_from_clock_s"], r["n_overlap"], r["slm_time"]), (2, 55, "10:00:07〜10:01:01"))   # 行の番号で引くと −8 s にずれた
        self.assertEqual(r["judgment"], "画面")
        inside = {t: v for t, v in self.vals.items() if not (START + dt.timedelta(seconds=22) <= t < START + dt.timedelta(seconds=32))}
        r2 = fc.check(self.wav, [self.slm_file("slm_gap_inside.TXT", inside)], round(self.base + 0.1, 1), (5, 60))
        self.assertEqual((r2["shift_from_clock_s"], r2["n_overlap"], r2["slm_time"]), (2, 45, "10:00:07〜10:01:01"))
        self.assertEqual((r2["judgment"], r2["use"]), ("要確認", "画面"))
        want = fc.energy_mean([self.rec[i] + 50.0 for i in range(5, 60) if not 20 <= i < 30])     # 抜けた秒を除いた同じ秒どうし
        self.assertAlmostEqual(r2["laeq_log"], want, delta=0.05)

    def test_c10_no_display_still_checks(self):
        weak = fc.check(self.wav, [self.slm_noise], None, (5, 60))
        self.assertEqual((weak["judgment"], weak["use"], weak["laeq_use"], weak["exit_code"]), ("照合できない", "", None, 5))
        self.assertIsNone(weak["gain_db"])
        sp = self.slm_file("slm_clock15b.TXT", slm_values(self.rec, 15, seed=8, rows=160))
        far = fc.check(self.wav, [sp], None, (5, 60))
        self.assertEqual((far["judgment"], far["use"], far["laeq_use"], far["exit_code"]), ("要確認", "", None, 4))

    def test_c11_exit_codes_and_json(self):
        base = self.base
        sp15 = self.slm_file("slm_clock15c.TXT", slm_values(self.rec, 15, seed=9, rows=160))
        js = self.dir / "out.json"
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", str(round(base + 0.3, 1)), "--json", str(js)]), 0)
        self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["judgment"], "画面")
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", str(round(base - 0.8, 1)), "--json", str(js)]), 3)
        self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["exit_code"], 3)
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(sp15), "--laeq", "48.0"]), 4)
        self.assertEqual(quiet_main(["--wav", str(self.wav_nobext), "--slm", str(self.slm), "--laeq", "48.0"]), 5)
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", "48.0", "--window", "5.5-60"]), 2)
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", str(round(base + 0.3, 1)), "--window", "5.0-60.0"]), 0)
        self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", "48.0", "--window", "5-80"]), 2)
        self.assertEqual(quiet_main(["--wav", str(self.dir / "無い.WAV"), "--slm", str(self.slm), "--laeq", "48.0"]), 2)

    def test_c12_window_must_be_whole_seconds(self):
        self.assertEqual(fc.parse_window("5-60"), (5, 60)); self.assertEqual(fc.parse_window("5.0-60.0"), (5, 60))
        for bad in ("5.5-60", "5-60.5", "60-5", "5", "a-60"):
            with self.assertRaises(ValueError, msg=bad):
                fc.parse_window(bad)
        with self.assertRaises(ValueError):
            fc.check(self.wav, [self.slm], 48.0, (5.5, 60))      # 黙って 5 に切り捨てない
        with self.assertRaises(ValueError):
            fc.check(self.wav, [self.slm], 48.0, (5, 80))        # 録音（70 s）より長い窓

    def test_c13_folder_txt_once(self):
        d = self.dir / "slm_folder"; d.mkdir(exist_ok=True)
        (d / "a.TXT").write_text("", encoding="utf-8")
        self.assertEqual([p.name for p in fc.slm_files(d)], ["a.TXT"])          # Windows で *.TXT と *.txt の両方に当たっても 1 回
        (d / "b.txt").write_text("", encoding="utf-8"); (d / "c.REC").write_bytes(b"\0")
        self.assertEqual([p.name for p in fc.slm_files(d)], ["a.TXT", "b.txt"])
        self.assertEqual(fc.slm_files(self.slm), [self.slm])                      # ファイルならそれだけ
        empty = self.dir / "slm_empty"; empty.mkdir(exist_ok=True)
        r = fc.check(self.wav, fc.slm_files(empty), 48.0, (5, 60))                 # .TXT が 1 つも無い → 照合できない（「重ならない」と分ける）
        self.assertEqual((r["judgment"], r["use"]), ("照合できない", "画面")); self.assertIn("記録（.TXT）が無い", r["status"])

    def test_c14_min_corr_default(self):
        self.assertGreater(fc.MIN_CORR, 0.5)
        self.assertLessEqual(fc.MIN_CORR, 0.71)        # 本物の 10/15 の正しい合わせ（相関 0.711）を照合できないにしない
        r = fc.check(self.wav, [self.slm], round(self.base + 0.1, 1), (5, 60), min_corr=1.01)     # 下限が効く（届かない下限 → 照合できない）
        self.assertEqual((r["judgment"], r["use"]), ("照合できない", "画面"))

    def test_c15_diff_correlation(self):
        n = 70; t = np.arange(n)
        trend = 6.0 * np.sin(2 * np.pi * t / 140.0)                     # ゆっくり変わる暗騒音（両方に共通）
        rng = np.random.default_rng(15)
        wav = self.dir / "261029_015.WAV"; write_wav(wav, sr=24000, seed=15, levels=trend + rng.normal(0, 0.8, n))
        rec = fc.rec_levels(wav)
        vals = {dt.datetime(2026, 10, 29, 9, 59, 40) + dt.timedelta(seconds=m): 45.0 + rng.normal(0, 1) for m in range(120)}
        for i in range(len(rec)):                                       # 秒は正しく合っている（+2 s）が、秒ごとの上がり下がりは録音と無関係
            vals[START + dt.timedelta(seconds=i + 2)] = trend[i] + 50.0 + rng.normal(0, 0.8)
        sp = self.slm_file("slm_trend.TXT", vals)
        r = fc.check(wav, [sp], 48.0, (5, 60))
        self.assertGreaterEqual(r["corr"], fc.MIN_CORR)                 # 値どうしの相関は下限を超える
        self.assertLess(r["dcorr"], fc.MIN_DCORR)
        self.assertEqual((r["judgment"], r["use"], r["exit_code"]), ("要確認", "画面", 4))
        self.assertIn("変化どうしの相関", r["status"])
        r2 = fc.check(wav, [sp], round(r["laeq_log"], 1), (5, 60), min_dcorr=-1.0)   # 差分の相関を見なければ秒の合わせを通ってしまう
        self.assertEqual(r2["judgment"], "画面")
        self.assertGreaterEqual(fc.check(self.wav, [self.slm], None, (5, 60))["dcorr"], 0.9)    # 同じ音を測っていれば変化も合う

    def test_c17_non_finite_inputs(self):
        """Astra MAP25-09（10/25）: 画面の値・閾値が nan／inf なら入力の誤り（2）。騒音計の記録の nan の行は飛ばす（窓が欠けて要確認・値は画面）"""
        for bad in (["--laeq", "nan"], ["--laeq", "inf"], ["--laeq", "47", "--min-corr", "nan"], ["--laeq", "47", "--stop-db", "inf"]):
            with self.subTest(bad=bad):
                self.assertEqual(quiet_main(["--wav", str(self.wav), "--slm", str(self.slm), "--window", "5-60"] + bad), 2)
        vals = dict(self.vals)
        vals[START + dt.timedelta(seconds=30 + 2)] = float("nan")              # 窓の中の 1 秒が nan
        p = self.dir / "slm_nan.TXT"
        with open(p, "w", encoding="utf-8") as f:
            for n, (t, v) in enumerate(sorted(vals.items()), 1):
                f.write(f"{n}\t{t:%Y/%m/%d}\t{t:%H:%M:%S}.000\t{v}\n")
        r = fc.check(self.wav, [p], self.base, (5, 60))
        self.assertEqual(r["judgment"], "要確認")
        self.assertEqual(r["n_overlap"], 54)
        self.assertEqual(r["laeq_use"], self.base)
        self.assertEqual(fc._corr([1, 2, float("nan"), 4], [1, 2, 3, 4]), 0.0)

    def test_c16_stop_when_difference_large(self):
        base = self.base
        stop = fc.check(self.wav, [self.slm], round(base + 2.0, 1), (5, 60))      # 画面の値が 1 秒の記録より約 2.0 dB 高い
        self.assertAlmostEqual(stop["diff_db"], 2.0, delta=0.06)
        self.assertEqual((stop["judgment"], stop["exit_code"], stop["use"], stop["laeq_use"], stop["gain_db"]), ("止める", 6, "", None, None))
        self.assertIn("本人と原因を確かめて", stop["status"])
        self.assertEqual(fc.check(self.wav, [self.slm], round(base - 2.0, 1), (5, 60))["judgment"], "止める")    # 低い側も（差の大きさ）
        mid = fc.check(self.wav, [self.slm], round(base + 1.2, 1), (5, 60))        # 0.5〜1.4 dB は今までどおり替える
        self.assertEqual((mid["judgment"], mid["exit_code"], mid["laeq_use"]), ("1秒の記録", 3, round(base, 1)))
        self.assertEqual(fc.check(self.wav, [self.slm], round(base + 2.0, 1), (5, 60), stop_db=3.0)["judgment"], "1秒の記録")
        sp = self.slm_file("slm_clock15d.TXT", slm_values(self.rec, 15, seed=16, rows=160))
        far = fc.check(self.wav, [sp], round(base + 2.0, 1), (5, 60))              # 秒の合わせが要確認なら、そちらが先
        self.assertEqual((far["judgment"], far["use"], far["exit_code"]), ("要確認", "画面", 4))
        self.assertEqual(fc.check(self.wav, [self.slm_noise], round(base + 2.0, 1), (5, 60))["judgment"], "照合できない")
        js = self.dir / "stop.json"
        args = ["--wav", str(self.wav), "--slm", str(self.slm), "--laeq", str(round(base + 2.0, 1))]
        self.assertEqual(quiet_main(args + ["--json", str(js)]), 6)
        j = json.loads(js.read_text(encoding="utf-8"))
        self.assertEqual((j["judgment"], j["exit_code"], j["laeq_use"], j["gain_db"]), ("止める", 6, None, None))
        self.assertEqual(quiet_main(args + ["--stop-db", "3.0"]), 3)
        self.assertEqual(quiet_main(args + ["--stop-db", "0.3"]), 2)                 # --tol（0.5）より小さい → 入力の誤り
        with self.assertRaises(ValueError):
            fc.check(self.wav, [self.slm], 48.0, (5, 60), stop_db=0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
