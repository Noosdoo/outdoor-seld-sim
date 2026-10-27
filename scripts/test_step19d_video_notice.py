"""CLI regression: missing optional video does not hide real annotation errors."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class VideoNoticeTests(unittest.TestCase):
    def run_case(self, present=False, invalid=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_demo"
            (raw / "video").mkdir(parents=True)
            if present:
                (raw / "video" / "fixture.mp4").write_bytes(b"path-only fixture")
            session = dict(session_id="fixture", 用途="調整用", 区分="C", マイク高さ_cm="160")
            event = dict(session_id="fixture", take_id="1", event_id="1", 原本="261005_003.WAV",
                         象限="L", ラップ秒="bad" if invalid else "8", 状態="静止",
                         動画原本="fixture.mp4", direction="back_to_front", line_m="1")
            event["class"] = "car_drive"
            for name, row in (("session", session), ("events", event)):
                with (root / (name + ".csv")).open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader(); writer.writerow(row)
            result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("step19d_field_csv_to_ann.py")),
                                     "--session", str(root / "session.csv"), "--events", str(root / "events.csv"),
                                     "--audio-dir", str(raw), "--out", str(root / "ann_orig.csv")],
                                    capture_output=True, encoding="utf-8")
            def read(name):
                with (root / name).open(encoding="utf-8-sig", newline="") as f:
                    return list(csv.DictReader(f))
            video = read("ann_orig_video_takes.csv")
            self.assertEqual(len(video), 1)
            if invalid:
                # 変更票 #59 r6・Astra FR11-01（2026-10-12）: 時刻が未確定の正例があれば注釈を書かずに止める（終了コード 2）
                self.assertFalse((root / "ann_orig.csv").exists())
            else:
                self.assertEqual(read("ann_orig.csv"), [])  # tuning never enters evaluation annotations
            self.assertEqual(Path(video[0]["video"]).is_absolute(), present)
            return result

    def test_missing_video_is_notice(self):
        result = self.run_case()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VIDEO NOTICE]", result.stdout)

    def test_existing_video_resolves(self):
        result = self.run_case(present=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("[VIDEO NOTICE]", result.stdout)

    def test_missing_video_does_not_suppress_annotation_failure(self):
        result = self.run_case(invalid=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)   # #59 r6: 未確定の正例 → 注釈を書かずに 2 で止まる（2026-10-20 に検査を今の決まりへ）
        self.assertIn("[WARN]", result.stdout)
        self.assertIn("[VIDEO NOTICE]", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
