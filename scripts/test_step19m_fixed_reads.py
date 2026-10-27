# -*- coding: utf-8 -*-
"""先行分 §4（決定 ⑳・2026-10-26 Fable 仕様・Astra R30-09）の受け入れ試験: 止まった音源（クラクション E2）の鳴っていた区間を知覚層の読みにする（step19m）。
架空の 10/29（車の日）の台帳の形（test_field_e2_90_270.py と同じ記録シート Version 123 の書き方）だけを使う。評価の録音・実録の読み・本物の台帳は使わない。

  M1 4 方位 × 3／7 m の 8 本（音源 (−3 か −7, +2.0)・マイク (0, +2.0)・向き 180／90／0／270）: step19m → step19d → step19b の切り出し → step19k で、
     知覚層の正解の方位が 前 0／左 +90／後 ±180／右 −90・水平距離 3.00／7.00 m・正解の行は鳴っている区間（録音 10〜11 s → クリップ 5〜6 s）だけ
  M2 押しが 2 回ある本: track が 1 と 2 に分かれ、鳴っていない間（押しと押しの間・前後）に正解の行が無い
  M3 拍手のずれ（拍手_録音側秒 − 拍手_音側秒）を鳴り始めと鳴り終わりの両方に足す（step19d と同じ約束）
  M4 止める: 鳴り終わり（終わり秒）が空・鳴り始めが空・鳴り終わり ≤ 鳴り始め・E2 が 1 本も無い・ann の座標と台帳が違う・押しが重なる → 終了コード 2・出力を書かない
  M5 field_score_members --reads-fixed: 確かめた reads の後ろに E2 の行を足した写しを step19k に渡す（E2 は印の検査で除かれない）・固定音源でない行は止める
使い方: python scripts/test_step19m_fixed_reads.py
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step19m_fixed_reads as fr                     # noqa: E402
import step19d_field_csv_to_ann as convert          # noqa: E402
import step19b_realsmoke_cut as cut                  # noqa: E402
import step19k_field_perception as gt                # noqa: E402
from test_field_e2_90_270 import E2, WANT_AZ, e2_event, session_rows, write_csv   # noqa: E402


def ev(n: int, lap: str = "10", end: str = "11", **kw) -> dict:
    return dict(e2_event(n, lap), **{"終わり秒": end}, **kw)


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "step19m_fixed_reads.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))


def gt_lines(events: list, td: Path, cut_off: float = 5.0) -> dict:
    """台帳 → step19m の読み → step19d → step19b の切り出し → step19k。clip_id → 正解の行（文字列）。"""
    reads, stops, _ = fr.fixed_reads(events)
    assert not stops, stops
    rows, _ = convert.convert(session_rows(), events, 6, 0, None)
    ann, clips = [], {}
    for r in sorted(rows, key=lambda x: (x["take_id"], x["event_id"])):
        if r["orig_file"] in clips:
            continue
        cid = f"eval_field_clip99{len(clips) + 1:03d}"
        clips[r["orig_file"]] = cid
        ann += cut.rebase_rows([r], cut_off, 10.0, r["orig_file"], cid)
    gt.build(reads, ann, td, 1.6, None)
    return {cid: [s for s in (td / "gt" / f"{cid}.csv").read_text(encoding="utf-8").splitlines() if s] for cid in clips.values()}, clips, reads


class FixedReadsTests(unittest.TestCase):
    def test_m1_four_directions_and_distance(self):
        with tempfile.TemporaryDirectory() as td:
            out, clips, reads = gt_lines([ev(n) for n in range(1, 9)], Path(td))
            self.assertEqual(len(reads), 16)                                       # 8 本 × 押し 1 回 × 2 点
        for n in range(1, 9):
            pos, h, q, d = E2[n]
            lines = out[clips[f"261029_{100 + n:03d}.WAV"]]
            with self.subTest(n=n, pos=pos, heading=h):
                self.assertTrue(lines, "正解が 0 行")
                for s in lines:
                    k, c, trk, az, el, dist = s.split(",")
                    self.assertEqual((int(c), int(trk)), (1, 1))
                    self.assertTrue(50 <= int(k) <= 60, k)                        # 録音 10〜11 s → クリップ 5〜6 s（伝わりの遅れ 0.01〜0.02 s）
                    if pos == "後":
                        self.assertEqual(abs(int(az)), 180, s)
                    else:
                        self.assertEqual(int(az), WANT_AZ[pos], s)
                    self.assertAlmostEqual(float(dist), float(d), places=2)

    def test_m2_two_presses_two_tracks(self):
        evs = [ev(1, "10", "10.5"), dict(ev(1, "12", "12.4"), event_id="2")]
        with tempfile.TemporaryDirectory() as td:
            out, clips, reads = gt_lines(evs, Path(td))
        self.assertEqual(sorted({r["track"] for r in reads}), ["1", "2"])
        lines = out[clips["261029_101.WAV"]]
        frames = {1: set(), 2: set()}
        for s in lines:
            k, c, trk, *_ = s.split(",")
            frames[int(trk)].add(int(k))
        self.assertTrue(frames[1] and frames[2], frames)
        self.assertTrue(all(50 <= k <= 55 for k in frames[1]), frames[1])           # 録音 10.0〜10.5 → クリップ 5.0〜5.5
        self.assertTrue(all(70 <= k <= 74 for k in frames[2]), frames[2])           # 録音 12.0〜12.4 → クリップ 7.0〜7.4
        allk = {int(s.split(",")[0]) for s in lines}
        self.assertFalse(allk & set(range(56, 70)), "押しと押しの間に正解がある")
        self.assertFalse({k for k in allk if k < 50 or k > 74}, "鳴っていない前後に正解がある")

    def test_m3_clap_shift_on_both_ends(self):
        reads, stops, _ = fr.fixed_reads([ev(1, "10", "11", **{"拍手_録音側秒": "3.25", "拍手_音側秒": "3.00"})])
        self.assertFalse(stops, stops)
        self.assertEqual([r["t"] for r in reads], ["10.250", "11.250"])
        reads2, _, _ = fr.fixed_reads([ev(1, "10", "11", **{"拍手_録音側秒": "3.25"})])   # 片方だけ → ずれを足さない（step19d と同じ）
        self.assertEqual([r["t"] for r in reads2], ["10.000", "11.000"])
        reads3, _, _ = fr.fixed_reads([ev(1, "10", "11")], lap_offset=0.5)
        self.assertEqual([r["t"] for r in reads3], ["10.500", "11.500"])

    def test_m4_stops(self):
        cases = {
            "鳴り終わりが空": [ev(1, "10", "")],
            "鳴り始めが空": [ev(1, "", "11")],
            "鳴り終わり ≤ 鳴り始め": [ev(1, "11", "11")],
            "E2 が無い": [dict(ev(1), **{"class": "car_drive", "台本ID": "A2-01"})],
            "押しが重なる": [ev(1, "10", "11"), dict(ev(1, "10.5", "12"), event_id="2")],
        }
        for name, evs in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as td:
                td = Path(td)
                write_csv(td / "events.csv", evs)
                r = run_cli("--events", td / "events.csv", "--out", td / "reads_fixed.csv")
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertFalse((td / "reads_fixed.csv").exists())
                self.assertIn("止める", r.stdout)
        with tempfile.TemporaryDirectory() as td:            # ann の座標と台帳が違う
            td = Path(td)
            write_csv(td / "events.csv", [ev(1)])
            write_csv(td / "ann.csv", [{"clip_id": "eval_field_clip99001", "orig_file": "261029_101.WAV", "class": "horn", "x_m": "-3.5", "y_m": "2.0", "高さ_m": "0.8"}])
            r = run_cli("--events", td / "events.csv", "--out", td / "reads_fixed.csv", "--ann", td / "ann.csv")
            self.assertEqual(r.returncode, 2, r.stdout); self.assertIn("ann の固定の座標", r.stdout)
            write_csv(td / "ann.csv", [{"clip_id": "eval_field_clip99001", "orig_file": "261029_101.WAV", "class": "horn", "x_m": "-3", "y_m": "2.0", "高さ_m": "0.8"}])
            r = run_cli("--events", td / "events.csv", "--out", td / "reads_fixed.csv", "--ann", td / "ann.csv")
            self.assertEqual(r.returncode, 0, r.stdout)
            with open(td / "reads_fixed.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(x["take"], x["cls"], x["track"], x["t"]) for x in rows], [("261029_101.WAV", "horn", "1", "10.000"), ("261029_101.WAV", "horn", "1", "11.000")])
            self.assertEqual(list(rows[0].keys()), fr.OUT_COLS)

    def test_m6_step19d_note_for_e2_end(self):
        """Fable 10/26 15:1x: step19d は E2 の 終わり秒 に「長回しだけ → 無視」ではなく「E2 では step19m が鳴り終わりとして使う」と出す（注釈の時刻は変わらない）"""
        rows, warns = convert.convert(session_rows(), [ev(1, "10", "11")], 6, 0, None)
        self.assertTrue(any("E2 では step19m が鳴り終わりとして使う" in w for w in warns), warns)
        self.assertFalse(any("相手だけに使う → 無視" in w for w in warns), warns)
        self.assertEqual((rows[0]["t_start"], rows[0]["t_end"]), ("10.00", ""))

    def test_m5_field_score_members_reads_fixed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            reads_mid = [dict(take="261029_021.WAV", frame="", cls="car_drive", track="1", x=str(x), y="2.0", t=str(20 + i * 0.5), converted="midpoint") for i, x in enumerate(range(-12, 13, 2))]
            write_csv(td / "reads_mid_A.csv", reads_mid)
            write_csv(td / "events.csv", [ev(n) for n in range(1, 3)])
            self.assertEqual(run_cli("--events", td / "events.csv", "--out", td / "reads_fixed.csv").returncode, 0)
            for f in ("ann.csv",):
                write_csv(td / f, [{"clip_id": "x", "orig_file": "261029_101.WAV", "cut_offset_s": "5", "scored": "1", "class": "horn"}])
            (td / "pred").mkdir()
            r = subprocess.run([sys.executable, str(ROOT / "scripts/field_score_members.py"), "--tag", "t", "--pred-root", str(td / "pred"), "--n", "1",
                                "--ann-score", str(td / "ann.csv"), "--ann-clips", str(td / "ann.csv"), "--reads", str(td / "reads_mid_A.csv"),
                                "--reads-fixed", str(td / "reads_fixed.csv"), "--out", str(td / "score"), "--dry", "--scoring", "old", "--no-warn-interval"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-800:])
            with open(td / "score" / "reads_with_fixed.csv", encoding="utf-8") as f:
                merged = list(csv.DictReader(f))
            self.assertEqual(len(merged), len(reads_mid) + 4)
            self.assertEqual(sum(1 for m in merged if m["cls"] == "horn"), 4)
            self.assertIn("reads_with_fixed.csv", (td / "score" / "index.md").read_text(encoding="utf-8"))
            write_csv(td / "bad_fixed.csv", [dict(take="a.WAV", frame="", cls="car_drive", track="1", x="0", y="0", z="", t="1")])
            r2 = subprocess.run([sys.executable, str(ROOT / "scripts/field_score_members.py"), "--tag", "t", "--pred-root", str(td / "pred"), "--n", "1",
                                 "--ann-score", str(td / "ann.csv"), "--ann-clips", str(td / "ann.csv"), "--reads", str(td / "reads_mid_A.csv"),
                                 "--reads-fixed", str(td / "bad_fixed.csv"), "--out", str(td / "score2"), "--dry", "--scoring", "old", "--no-warn-interval"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            self.assertEqual(r2.returncode, 3, r2.stdout); self.assertIn("固定音源でない行", r2.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
