# -*- coding: utf-8 -*-
"""field_ledger_draft.py の試験（架空の録音・動画・記録シートのバックアップだけ・名前なし・2026-10-25 Opus・Fable の確認待ち）。

  L1 役割の下書き: 本（台本ID）・校正（2 セッションで同じ原本なら 1 行にまとめる）・点検（向き・頭の点検）・使わない（理由あり）・記録シートに無い → 要確認
  L2 番号ずれ: H3-VR の時計のずれ（−3 s）を差し引いて、押した時刻に一番近い録音が記録シートと違う本を出す
  L3 動画: 本の動画の撮影の時間が録音と重ならない本を出す・重なる動画の列
  L4 短い録音（5 秒未満）・短い校正（60 秒未満）・記録シートにあるのに録音が無い
  L5 今ある台帳は上書きしない（--force なら history/ に写してから書く）
 敵対レビュー 10/25 の軽 6 の直し:
  L6 本が 1 本の日: 時計のずれがその本の差そのもの（番号ずれを吸い込む）→ 支えた本の数と |ずれ| > 10 s を疑いに出す。本が揃った日は出さない
  L7 bext の無い WAV は「録音の時刻が読めない」・mvhd の無い MOV は「動画の時刻が読めない」（「動画が重ならない」と分ける）・MOV を一覧から落とさない
  L8 RF64（4 GB 超）の WAV を ds64 の長さで読んで一覧に入れる・読めない WAV も一覧に残して理由を出す
  L9 出力 4 つは書き始める前に全部確かめて止める（途中まで書かない）・--force は check.txt も history/ に写す（同じ秒に何回でも写しを残す）
  L10 動画の名前は大文字・小文字を区別しない・記録シートの動画が無いときは「見つからない」
  L11 使わないの理由（自由記述）や helper を下書き・疑いの一覧に写さない（「理由あり／理由なし」だけ）
  L12 開いたファイルを閉じ忘れない（ResourceWarning が出ない）
使い方: python scripts/test_field_ledger_draft.py
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import field_ledger_draft as fl        # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))
DAY = dt.datetime(2026, 10, 29, tzinfo=JST)
CLOCK = -3.0                           # H3-VR の時計は 3 秒遅れ（録り始め = 本当の時刻 − 3 s）


def _bext(start: dt.datetime) -> bytes:
    b = bytearray(602); b[320:338] = (start + dt.timedelta(seconds=CLOCK)).strftime("%Y-%m-%d%H:%M:%S").encode("latin1")
    return b"bext" + struct.pack("<I", len(b)) + bytes(b)


def write_wav(path: Path, start: dt.datetime, seconds: float, sr: int = 1000, bext: bool = True):
    """ヘッダーと小さい音だけの WAV（4 ch・16 ビット・sr は小さくして長さだけ合わせる）。bext に録り始め（bext=False なら bext なし）。"""
    n = int(seconds * sr); data = bytes(n * 8)
    fmt = struct.pack("<HHIIHH", 1, 4, sr, sr * 8, 8, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + (_bext(start) if bext else b"") + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_wav_hist(path: Path, start: dt.datetime, samples, sr: int = 1000, hist: str = ""):
    """bext の coding history（H3-VR の形）と、音の標本（int16 の n×4）を入れた WAV（2026-10-25: 設定・割れ・リミッターの確かめ用）。"""
    b = bytearray(602); b[320:338] = (start + dt.timedelta(seconds=CLOCK)).strftime("%Y-%m-%d%H:%M:%S").encode("latin1")
    bext = bytes(b) + hist.encode("latin1") + b"\x00"
    bext += b"\x00" * (len(bext) & 1)
    data = samples.astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 4, sr, sr * 8, 8, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"bext" + struct.pack("<I", len(bext)) + bext
            + b"data" + struct.pack("<I", len(data)) + data)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_rf64(path: Path, start: dt.datetime, seconds: float, sr: int = 1000):
    """RF64 の WAV（4 GB 超の形）: RIFF と data の大きさは 0xFFFFFFFF、本当の大きさは ds64 に。"""
    n = int(seconds * sr); data = bytes(n * 8)
    fmt = struct.pack("<HHIIHH", 1, 4, sr, sr * 8, 8, 16)
    ds64 = struct.pack("<QQQI", 0, len(data), n, 0)
    body = (b"WAVE" + b"ds64" + struct.pack("<I", len(ds64)) + ds64 + b"fmt " + struct.pack("<I", len(fmt)) + fmt + _bext(start)
            + b"data" + struct.pack("<I", 0xFFFFFFFF) + data)
    path.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + body)


def write_mov(path: Path, start: dt.datetime, seconds: float):
    """moov/mvhd（版 0）だけの MOV。撮り始め（1904 年からの秒・UTC）と長さ（timescale 600）。"""
    ct = int((start.astimezone(dt.timezone.utc) - dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)).total_seconds())
    mvhd_body = bytes([0, 0, 0, 0]) + struct.pack(">IIII", ct, ct, 600, int(seconds * 600)) + bytes(80)
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"qt  " + bytes(4)
    path.write_bytes(ftyp + moov)


def write_mov_no_mvhd(path: Path):
    """ftyp と空の mdat だけの MOV（撮り始めが読めない）。"""
    path.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  " + bytes(4) + struct.pack(">I4s", 16, b"mdat") + bytes(8))


def ms(t: dt.datetime) -> int:
    return int(t.timestamp() * 1000)


def read_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


T = lambda h, m, s: DAY.replace(hour=h, minute=m, second=s)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); self.raw = Path(self.td.name) / "raw"; (self.raw / "video").mkdir(parents=True)
        # 本当の録り始め（iPad の時計）: 001 押し間違い・002 校正 70 s・003 点検・004 E2-03・005 は 1 回目のやり直し・006 は E2-02（記録シートは 005 と書いた）
        W = {"261029_001.WAV": (T(9, 0, 0), 2.0), "261029_002.WAV": (T(9, 10, 0), 70.0), "261029_003.WAV": (T(9, 15, 0), 20.0),
             "261029_004.WAV": (T(9, 20, 0), 15.0), "261029_005.WAV": (T(9, 22, 0), 12.0), "261029_006.WAV": (T(9, 24, 0), 15.0)}
        for fn, (t, s) in W.items():
            write_wav(self.raw / fn, t, s)
        write_mov(self.raw / "video" / "IMG_1001.MOV", T(9, 19, 58), 20.0)      # 004 と重なる
        write_mov(self.raw / "video" / "IMG_1002.MOV", T(9, 23, 58), 20.0)      # 006 と重なる
        write_mov(self.raw / "video" / "IMG_0999.MOV", DAY.replace(day=28, hour=9), 10.0)   # 前の日（数えない）
        sess = [{"id": "20261029_E1", "calibFile": "261029_002.WAV", "checkFile": "261029_003.WAV", "checks": {"261029_003.WAV": {"h": "0", "mode": "4"}},
                 "excluded": [{"file": "261029_006.WAV", "reason": "やり直し（E2-02）・車が通った", "t0": ms(T(9, 24, 0)), "video": "IMG_1002.MOV"}],
                 "takes": [{"take_id": 1, "file": "261029_004.WAV", "video": "IMG_1001.MOV", "scriptId": "E2-03", "t0": ms(T(9, 20, 0) + dt.timedelta(milliseconds=300))},
                           {"take_id": 2, "file": "261029_005.WAV", "video": "IMG_1002.MOV", "scriptId": "E2-02", "t0": ms(T(9, 24, 0) + dt.timedelta(milliseconds=200))},
                           {"take_id": 3, "file": "261029_009.WAV", "video": "", "scriptId": "E2-01", "t0": None}]},
                {"id": "20261029_B1", "calibFile": "261029_002.WAV", "checks": {}, "excluded": [], "takes": [], "helper": "（名前は読まない）"},
                {"id": "20261022_C1", "calibFile": "261022_002.WAV", "takes": [{"file": "261022_005.WAV", "scriptId": "C2-01"}]}]   # 別の日（数えない）
        self.bk = Path(self.td.name) / "backup.json"
        self.write_backup(sess)

    def tearDown(self):
        self.td.cleanup()

    def write_backup(self, sess, path: Path | None = None):
        (path or self.bk).write_text(json.dumps({"state": {"sessions": sess}}, ensure_ascii=False), encoding="utf-8")

    def sessions(self):
        return json.loads(self.bk.read_text(encoding="utf-8"))["state"]["sessions"]

    def run_draft(self, force=False):
        return fl.draft(self.raw, "20261029", self.bk, force)

    def rows(self):
        return {r["録音"]: r for r in read_csv(self.raw / "ledger" / "ledger_20261029_draft.csv")}

    def test_l1_roles(self):
        res = self.run_draft(); r = self.rows()
        self.assertEqual((res["n_wav"], res["n_mov"]), (6, 2))
        self.assertEqual(r["261029_004.WAV"]["役割"], "本 E1 E2-03")
        self.assertEqual(r["261029_002.WAV"]["役割"], "校正 E1・B1")
        self.assertTrue(r["261029_003.WAV"]["役割"].startswith("点検 E1: 0°・4 方向・頭の点検"))
        self.assertEqual(r["261029_006.WAV"]["役割"], "使わない E1: 理由あり")      # 理由の文は写さない（L11）
        self.assertTrue(r["261029_001.WAV"]["役割"].startswith("要確認: 記録シートに無い"))

    def test_l2_number_slip(self):
        res = self.run_draft()
        self.assertLessEqual(abs(res["clock_s"] - CLOCK), 0.5)             # 押した時刻は録り始めより 0.2〜0.3 s あと
        self.assertTrue(any("番号ずれの疑い" in f and "E2-02" in f and "261029_005.WAV" in f and "261029_006.WAV" in f for f in res["flags"]), res["flags"])
        self.assertFalse(any("番号ずれの疑い" in f and "E2-03" in f for f in res["flags"]))

    def test_l3_video_overlap(self):
        res = self.run_draft(); r = self.rows()
        self.assertEqual(r["261029_004.WAV"]["時刻が重なる動画"], "IMG_1001.MOV")
        self.assertEqual(r["261029_006.WAV"]["時刻が重なる動画"], "IMG_1002.MOV")
        self.assertTrue(any("動画が重ならない" in f and "261029_005.WAV" in f for f in res["flags"]), res["flags"])

    def test_l4_short_and_missing(self):
        res = self.run_draft()
        self.assertTrue(any("短い録音" in f and "261029_001.WAV" in f for f in res["flags"]))
        self.assertTrue(any("記録シートにあるが録音が無い" in f and "261029_009.WAV" in f for f in res["flags"]))
        self.assertFalse(any("短い校正" in f for f in res["flags"]))          # 校正は 70 s

    def test_l5_no_overwrite(self):
        self.run_draft()
        with self.assertRaises(SystemExit):
            self.run_draft()
        self.run_draft(force=True)
        self.assertTrue(list((self.raw / "ledger" / "history").glob("ledger_20261029_draft_before_*.csv")))

    def test_l6_few_takes_and_big_clock(self):
        raw = Path(self.td.name) / "raw_l6"; (raw / "video").mkdir(parents=True)
        D = lambda h, m, s: DAY.replace(day=30, hour=h, minute=m, second=s)
        write_wav(raw / "261030_001.WAV", D(10, 4, 20), 15.0)       # 押し間違いで先に録れた録音（記録シートはこれを本と書いた）
        write_wav(raw / "261030_002.WAV", D(10, 5, 0), 15.0)        # 本当の本（押した時刻どおり）
        bk = Path(self.td.name) / "backup_l6.json"
        self.write_backup([{"id": "20261030_E1", "takes": [{"file": "261030_001.WAV", "scriptId": "E2-01", "t0": ms(D(10, 5, 0))}]}], bk)
        res = fl.draft(raw, "20261030", bk)
        # 本が 1 本だけ → 時計のずれ = その本の差（−43 s）＝番号ずれを吸い込み「番号ずれの疑い」は出ない → 支えの少なさとずれの大きさを出す
        self.assertAlmostEqual(res["clock_s"], -43.0, delta=0.6)
        self.assertEqual((res["clock_n"], res["clock_support"]), (1, 1))
        self.assertFalse(any("番号ずれの疑い" in f for f in res["flags"]))
        self.assertTrue(any("時計のずれの支えが少ない" in f and "1 本のうち 1 本" in f for f in res["flags"]), res["flags"])
        self.assertTrue(any("時計のずれが大きい" in f and "-43" in f for f in res["flags"]), res["flags"])
        head = (raw / "ledger" / "ledger_20261030_check.txt").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("1 本のうち", head)
        # 本が揃った日（3 本とも時計 −3 s）は、どちらも出さない
        raw2 = Path(self.td.name) / "raw_l6b"; (raw2 / "video").mkdir(parents=True)
        takes = []
        for n in range(3):
            t = D(11, 10 * n, 0); fn = f"261030_{n + 1:03d}.WAV"
            write_wav(raw2 / fn, t, 15.0); takes.append({"file": fn, "scriptId": f"E2-0{n + 1}", "t0": ms(t + dt.timedelta(milliseconds=250))})
        bk2 = Path(self.td.name) / "backup_l6b.json"; self.write_backup([{"id": "20261030_E1", "takes": takes}], bk2)
        res2 = fl.draft(raw2, "20261030", bk2)
        self.assertEqual(res2["clock_support"], 3)
        self.assertFalse(any("時計のずれの支えが少ない" in f or "時計のずれが大きい" in f for f in res2["flags"]), res2["flags"])
        # 押した時刻のある本が 1 本も無い日 → 推定できない（0 s として扱った）
        raw3 = Path(self.td.name) / "raw_l6c"; (raw3 / "video").mkdir(parents=True)
        write_wav(raw3 / "261030_001.WAV", D(12, 0, 0), 15.0)
        bk3 = Path(self.td.name) / "backup_l6c.json"
        self.write_backup([{"id": "20261030_E1", "takes": [{"file": "261030_001.WAV", "scriptId": "E2-01", "t0": None}]}], bk3)
        res3 = fl.draft(raw3, "20261030", bk3)
        self.assertTrue(any("時計のずれを推定できない" in f for f in res3["flags"]), res3["flags"])

    def test_l7_unreadable_times(self):
        write_wav(self.raw / "261029_004.WAV", T(9, 20, 0), 15.0, bext=False)      # 004（E2-03・IMG_1001）は bext なし
        write_mov_no_mvhd(self.raw / "video" / "IMG_1002.MOV")                     # 005（E2-02）の動画は撮り始めが読めない
        res = self.run_draft(); flags = res["flags"]
        self.assertTrue(any("録音の時刻が読めない" in f and "261029_004.WAV" in f for f in flags), flags)
        self.assertTrue(any("動画の時刻が読めない" in f and "261029_005.WAV" in f and "IMG_1002.MOV" in f for f in flags), flags)
        for fn in ("261029_004.WAV", "261029_005.WAV"):
            self.assertFalse(any("動画が重ならない" in f and fn in f for f in flags), flags)
        movs = {m["file"]: m for m in read_csv(self.raw / "ledger" / "mov_20261029.csv")}
        self.assertIn("IMG_1002.MOV", movs); self.assertEqual(movs["IMG_1002.MOV"]["mvhd_utc"], "")   # 黙って落とさない（時刻なしで残す）
        self.assertEqual((res["n_mov"], res["n_mov_unreadable"]), (1, 1))
        self.assertTrue(any(f.startswith("動画の時刻が読めない: IMG_1002.MOV") for f in flags), flags)
        self.assertEqual(self.rows()["261029_004.WAV"]["開始"], "")

    def test_l8_rf64_and_unreadable_wav(self):
        write_rf64(self.raw / "261029_007.WAV", T(9, 30, 0), 3.0)
        (self.raw / "261029_008.WAV").write_bytes(b"not a wav file at all")
        res = self.run_draft()
        w = {r["file"]: r for r in read_csv(self.raw / "ledger" / "wav_20261029.csv")}
        self.assertEqual(float(w["261029_007.WAV"]["sec"]), 3.0); self.assertEqual(w["261029_007.WAV"]["bext_time"], "09:29:57")
        self.assertIn("261029_008.WAV", w); self.assertEqual(w["261029_008.WAV"]["sec"], "")       # 読めない録音も行を残す
        self.assertEqual(res["n_wav"], 8)
        self.assertTrue(any("RF64" in f and "261029_007.WAV" in f for f in res["flags"]), res["flags"])
        self.assertTrue(any("読めない録音" in f and "261029_008.WAV" in f for f in res["flags"]), res["flags"])
        r = self.rows(); self.assertIn("261029_007.WAV", r); self.assertIn("261029_008.WAV", r)

    def test_l9_outputs_checked_before_writing(self):
        led = self.raw / "ledger"; led.mkdir()
        (led / "ledger_20261029_check.txt").write_text("前の疑い\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            self.run_draft()
        self.assertIn("ledger_20261029_check.txt", str(cm.exception))
        for name in ("wav_20261029.csv", "mov_20261029.csv", "ledger_20261029_draft.csv"):
            self.assertFalse((led / name).exists(), name)                # 途中まで書いてから止まらない
        res = self.run_draft(force=True)
        hist = led / "history"
        saved = list(hist.glob("ledger_20261029_check_before_*.txt"))
        self.assertEqual(len(saved), 1); self.assertEqual(saved[0].read_text(encoding="utf-8"), "前の疑い\n")
        self.assertEqual(len(res["history"]), 1)
        self.assertNotEqual((led / "ledger_20261029_check.txt").read_text(encoding="utf-8"), "前の疑い\n")
        self.run_draft(force=True); self.run_draft(force=True)            # 同じ秒に続けても前の写しを上書きしない
        self.assertEqual(len(list(hist.glob("ledger_20261029_check_before_*.txt"))), 3)
        self.assertEqual(len(list(hist.glob("wav_20261029_before_*.csv"))), 2)

    def test_l10_video_name_case_and_missing(self):
        sess = self.sessions()
        sess[0]["takes"][0]["video"] = "img_1001.mov"           # 004 の動画を小文字で（ファイルは IMG_1001.MOV）
        sess[0]["takes"][1]["video"] = "IMG_9999.MOV"           # 005 の動画は raw/video に無い
        self.write_backup(sess)
        res = self.run_draft()
        self.assertFalse(any("261029_004.WAV" in f and ("動画が重ならない" in f or "見つからない" in f) for f in res["flags"]), res["flags"])
        self.assertTrue(any("動画が見つからない" in f and "261029_005.WAV" in f and "IMG_9999.MOV" in f for f in res["flags"]), res["flags"])

    def test_l11_free_text_not_copied(self):
        sess = self.sessions()
        sess[1]["excluded"] = [{"file": "261029_001.WAV", "reason": "", "t0": None}]       # 理由が空
        self.write_backup(sess)
        res = self.run_draft(); r = self.rows()
        self.assertEqual(r["261029_006.WAV"]["役割"], "使わない E1: 理由あり")
        self.assertEqual(r["261029_001.WAV"]["役割"], "使わない B1: 理由なし")
        outs = [p for p in (self.raw / "ledger").iterdir() if p.is_file()]
        self.assertEqual(len(outs), 4)
        for p in outs:
            text = p.read_text(encoding="utf-8-sig")
            for secret in ("車が通った", "やり直し（E2-02）", "名前は読まない"):
                self.assertNotIn(secret, text, p.name)
        self.assertFalse(any("車が通った" in f or "名前は読まない" in f for f in res["flags"]))

    def test_l13_settings_clip_and_limiter(self):
        """2026-10-25（前提の敵対的レビュー 実録 重大 3・中 4）: 録音機の設定が 10/15・10/22 と違う本・割れ（フルスケール）・リミッターの張り付き（−2 dBFS）を疑いに出す。
        設定どおりで静かな本と、coding history の無い本（ほかの録音機・今までの試験の WAV）には出さない"""
        import numpy as np
        ok = "A=PCM,F=96000,W=24,M=4ch,T=H3-VR; Rec Mode=AmbiX; Mic Position=Upright; Correction Filter=Off"
        quiet = (np.random.default_rng(0).normal(0, 300, (2000, 4))).astype(np.int16)
        clip = quiet.copy(); clip[500:600, 0] = 32767                      # 0.50〜0.60 s の続く割れ（クラクションの形）
        snap = quiet.copy(); snap[700:705, 1] = -32768                     # 5 ms の一瞬の割れ（拍手・指パッチンの形）
        lim = quiet.copy(); lim[100:1400, 0] = int(round(32767 * 10 ** (-2 / 20)))   # 1.3 s 張り付く（続く音をリミッターが潰した形・1300 個）
        write_wav_hist(self.raw / "261029_010.WAV", T(10, 0, 0), quiet, hist=ok)
        write_wav_hist(self.raw / "261029_011.WAV", T(10, 1, 0), quiet, hist=ok.replace("Correction Filter=Off", "Correction Filter=On"))
        write_wav_hist(self.raw / "261029_012.WAV", T(10, 2, 0), clip, hist=ok)
        write_wav_hist(self.raw / "261029_013.WAV", T(10, 3, 0), lim, hist=ok)
        write_wav_hist(self.raw / "261029_014.WAV", T(10, 4, 0), snap, hist=ok)
        res = self.run_draft(); fl_ = res["flags"]
        self.assertFalse([f for f in fl_ if "261029_010.WAV" in f and ("設定" in f or "割れ" in f or "リミッター" in f)], fl_)
        self.assertEqual(sum("録音機の設定が前の日と違う: 261029_011.WAV" in f and "Correction Filter=On" in f for f in fl_), 1, fl_)
        self.assertEqual(sum(f.startswith("割れの疑い（続く割れ）: 261029_012.WAV の 0.50〜0.60 s") for f in fl_), 1, fl_)
        self.assertEqual(sum(f.startswith("リミッターの疑い: 261029_013.WAV") for f in fl_), 1, fl_)
        self.assertEqual(sum(f.startswith("一瞬の割れだけの録音 1 本") and "261029_014.WAV（0.70 s）" in f for f in fl_), 1, fl_)
        self.assertFalse([f for f in fl_ if f.startswith("割れの疑い") and "261029_014.WAV" in f], fl_)
        self.assertFalse([f for f in fl_ if ("261029_004.WAV" in f or "261029_002.WAV" in f) and ("設定" in f or "割れ" in f or "リミッター" in f)], fl_)
        w = {r["file"]: r for r in read_csv(self.raw / "ledger" / "wav_20261029.csv")}
        self.assertEqual(w["261029_010.WAV"]["rec_settings"], "Rec Mode=AmbiX; Mic Position=Upright; Correction Filter=Off")
        self.assertAlmostEqual(float(w["261029_012.WAV"]["peak_dbfs"]), 0.0, delta=0.01)

    def test_l12_no_resource_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.run_draft(); gc.collect()
        self.assertFalse([str(w.message) for w in caught if issubclass(w.category, ResourceWarning)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
