"""#44 integration tests; generated inputs only, no evaluation recordings."""
import csv
import tempfile
import unittest
from pathlib import Path

from _field_orientation import ORIENTATION_COL, CHECK_COL, MIC_X_COL, MIC_Y_COL, tripod_heading, static_mic_x, mic_field_xy
import step19d_field_csv_to_ann as convert
import step19b_realsmoke_cut as cut
import step19k_field_perception as gt
import step20_realsmoke_score_v3 as score


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.sessions = [{"session_id": "fixture", "用途": "最終評価", "区分": "D",
                          "マイク高さ_cm": "160", "点検原本": "check0.wav", "点検方式": "前1打"}]
        self.event = {"session_id": "fixture", "take_id": "1", "event_id": "1", "class": "horn",
                      "原本": "synthetic.wav", "象限": "R", "ラップ秒": "5", "状態": "静止"}

    def rows(self, events):
        rows, warnings = convert.convert(self.sessions, events, 6, 0, None)
        self.assertEqual(warnings, [])
        return rows

    def test_legacy_and_explicit(self):
        for fields, want in [({}, 0), ({"特記": "三脚180°"}, 180),
                             ({ORIENTATION_COL: "", "特記": "三脚180°"}, 180),
                             ({ORIENTATION_COL: "0", "特記": "三脚180°"}, 0),
                             ({ORIENTATION_COL: "180°"}, 180)]:
            with self.subTest(fields=fields):
                self.assertEqual(int(self.rows([dict(self.event, **fields)])[0][ORIENTATION_COL]), want)

    def test_take_metadata_shared_across_events(self):
        rows = self.rows([self.event, dict(self.event, event_id="2", **{"特記": "三脚180°", CHECK_COL: "check180.wav"})])
        for r in rows:
            self.assertEqual(r[ORIENTATION_COL], "180")
            self.assertEqual(r["check_file"], "check180.wav")
            self.assertEqual(r["点検方式"], "4方位")
            self.assertEqual(set(r["excluded_files"].split()), {"check0.wav", "check180.wav"})

    def test_e2_headings_accepted(self):
        # #69 r40・Astra H25-01（2026-10-25）: クラクション E2 は 90／270 も録る。0／180 に書き換えずにそのまま通す
        for raw, want in [("90", 90), ("270°", 270), ("0", 0), ("180", 180)]:
            with self.subTest(raw=raw):
                self.assertEqual(tripod_heading({ORIENTATION_COL: raw, "台本ID": "E2-02"}), want)
                self.assertEqual(int(self.rows([dict(self.event, **{ORIENTATION_COL: raw, "台本ID": "E2-02"})])[0][ORIENTATION_COL]), want)
        for raw in ("90", "270"):          # 2026-10-25（敵対レビュー 軽 1）: 90／270 は E2 だけ。ほかの本・台本ID の無い本は止める
            for sid in ("A2-01", ""):
                with self.subTest(raw=raw, sid=sid), self.assertRaises(ValueError):
                    tripod_heading({ORIENTATION_COL: raw, "台本ID": sid})

    def test_walking_comparison_rejects_quarter_turns(self):
        # 歩く比較（W09〜W16）の静止の x は向き 0／180 だけで決まる。90／270 を黙って −2 にしない
        for h in ("90", "270"):
            with self.subTest(h=h), self.assertRaises(ValueError):
                static_mic_x({"台本ID": "W09", ORIENTATION_COL: h})

    def test_mic_field_xy(self):
        # 記録シート V123: 三脚の本はマイクの現場の座標（E2 = (0, +2.0)）。列が無い・空なら今までの決まり
        self.assertEqual(mic_field_xy({MIC_X_COL: "0", MIC_Y_COL: "2.0"}, "静止"), ("0", "2"))
        self.assertEqual(mic_field_xy({MIC_X_COL: "0", MIC_Y_COL: "0"}, "静止"), ("0", "0"))
        self.assertEqual(mic_field_xy({}, "静止"), ("0", "0"))
        self.assertEqual(mic_field_xy({"台本ID": "W09", "歩く人の線_m": "7.0", "歩く人の静止x_m": "2"}, "静止"), ("2", "7"))
        self.assertEqual(mic_field_xy({"台本ID": "W05"}, "歩行"), ("", "5"))
        warns = []
        self.assertEqual(mic_field_xy({"台本ID": "W05", MIC_X_COL: "0", MIC_Y_COL: "2"}, "歩行", warns), ("", "5"))
        self.assertTrue(warns and "使わない" in warns[0])
        for bad in ({MIC_X_COL: "0"}, {MIC_Y_COL: "2"}, {MIC_X_COL: "a", MIC_Y_COL: "2"}, {MIC_X_COL: "inf", MIC_Y_COL: "2"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mic_field_xy(bad, "静止")

    def test_invalid_and_conflicting_values(self):
        for bad in ("45", "360", "-90", "nan", "inf", "bad"):          # 2026-10-25: 90 は E2 で使うので不正の例から外した
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tripod_heading({ORIENTATION_COL: bad})
        for field, a, b in [(ORIENTATION_COL, "0", "180"), (CHECK_COL, "a.wav", "b.wav")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.rows([dict(self.event, **{field: a}), dict(self.event, event_id="2", **{field: b})])

    def test_cut_csv_preserves_heading_and_quadrant(self):
        rows = self.rows([dict(self.event, **{ORIENTATION_COL: "180", CHECK_COL: "check180.wav"})])
        rebased = cut.rebase_rows(rows, 0, 10, "synthetic.wav", "fixture_clip")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ann_all.csv"
            cut._write_ann(path, rebased)
            with path.open(encoding="utf-8") as f:
                reread = list(csv.DictReader(f))
        self.assertEqual(reread[0][ORIENTATION_COL], "180")
        self.assertEqual(reread[0][CHECK_COL], "check180.wav")
        self.assertEqual(reread[0]["quadrant"], "R")

    def test_real_scorer_compares_in_recorded_microphone_frame(self):
        # Both warning and distance paths; heading metadata must not cause a
        # second rotation. Wrong-side predictions must fail direction matching.
        for cls, ci in [("horn", 1), ("car_drive", 4)]:
            for angle, correct in [(-90, True), (90, False)]:
                with self.subTest(cls=cls, angle=angle):
                    r = dict(self.rows([dict(self.event, **{ORIENTATION_COL: "180"})])[0],
                             **{"class": cls, "t_start": "5", "t_cpa": "8", "横距離m": "2"})
                    frames = {k: [(ci, angle, 0, 2)] for k in range(50, 90)}
                    events, _, _ = score.evaluate([r], {r["clip_id"]: frames}, score.V43.Cfg43(),
                                                   (2.5, 1.5), 4, True)
                    self.assertEqual(bool(events[0]["quad_ok"]), correct)
                    r[ORIENTATION_COL] = "0"
                    unchanged, _, _ = score.evaluate([r], {r["clip_id"]: frames}, score.V43.Cfg43(),
                                                      (2.5, 1.5), 4, True)
                    self.assertEqual(events, unchanged)

    def test_walking_uses_heading_at_reception(self):
        # Turning +x to -x and pausing: direction follows time-ordered motion.
        path = [(0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 0, 0)]
        self.assertEqual(gt.walking_headings(path, [0.5, 1.5, 2.5]).tolist(), [0, 0, 180])


if __name__ == "__main__":
    unittest.main(verbosity=2)
