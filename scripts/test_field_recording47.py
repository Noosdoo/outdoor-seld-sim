"""#46/#47 integration regressions, generated records only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import step19d_field_csv_to_ann as conv
import step19c_ann_validate as validate
import step19h_video_marks as video
import step19k_field_perception as gt
import step19_realsmoke_convert as audio


class FieldRecordingTests(unittest.TestCase):
    def setUp(self):
        self.session = dict(session_id="test", **{"用途":"最終評価", "区分":"D", "装着":"三脚", "マイク高さ_cm":"160"})
        self.event = dict(session_id="test", take_id="1", event_id="1", **{"class":"horn", "原本":"fixture.wav", "象限":"R", "状態":"静止", "ラップ秒":"8"})

    def test_take_overrides_and_fallback(self):
        fields = {"装着":"ヘルメット", "マイク高さ_cm":"185", "x_m":"0", "y_m":"3", "高さ_m":"1.2"}
        rows, warnings = conv.convert([self.session], [dict(self.event, **fields), dict(self.event, event_id="2")], 6, 0, None)
        self.assertFalse(warnings)
        for row in rows:
            self.assertEqual(row["mic_z"], "1.850")
            self.assertEqual(row["装着"], "ヘルメット")
            self.assertEqual(row["y_m"], "3")
        old, _ = conv.convert([self.session], [self.event], 6, 0, None)
        self.assertEqual(old[0]["mic_z"], "1.600")
        with self.assertRaises(ValueError):
            conv.convert([self.session], [dict(self.event, **fields), dict(self.event, event_id="2", **{"マイク高さ_cm":"160"})], 6, 0, None)

    def test_unplanned_exposure_passes_validator(self):
        rows, _ = conv.convert([dict(self.session, **{"区分":"負例露出"})], [dict(self.event, pair_id="old")], 6, 0, None)
        self.assertEqual(rows[0]["区分"], "負例露出")
        validate.errs.clear(); validate.warns.clear()
        validate.validate(rows, False, 6000, {})
        self.assertEqual(validate.errs, [])

    def test_video_configuration_uses_recording_clock(self):
        ev = dict(self.event, **{"class":"bike_bell", "状態":"歩行", "動画原本":"test.mp4", "拍手_音側秒":"3.2", "拍手_録音側秒":"3.45", "三脚の向き":"180", "direction":"後→前", "line_m":"3"})
        rows = conv.video_takes([ev])
        self.assertEqual([(r["cls"], r["direction"], r["line_m"]) for r in rows], [("bike_bell", "後→前", "3"), ("mic", "後→前", "0")])
        self.assertTrue(all(r["clap_audio_s"] == "3.45" for r in rows))
        self.assertEqual(conv.video_takes([dict(ev, **{"拍手_録音側秒":""})])[0]["clap_audio_s"], "")

    def test_same_take_mic_source_independent_after_resume(self):
        base = dict(take="same", video="test.mp4", cls="car", track="1", line_m="2", clap_audio_s="3", markers_list=[10, 0, -10])
        mic = dict(base, cls="mic", track="0", line_m="0")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"state.json"
            s = video.Session([base, mic], path, "A")
            for row, offset in [(base, 0), (mic, 10)]:
                s.rec(row, "clap", 30, 1)
                s.rec(row, "mark", 90, 3+offset, 10)
                s.rec(row, "mark", 180, 6+offset, 0)
            s.rec(mic, "mark", 181, 16.1, 0)
            self.assertEqual(s.state[s.key(base)]["marks"][-1]["t_video"], 6)
            s.save(); reopened = video.Session([base, mic], path, "A")
            rows, _, warnings = reopened.rows()
            self.assertFalse(warnings)
            self.assertEqual([(r["cls"], r["t"]) for r in rows], [("car", "5.000"), ("car", "8.000"), ("mic", "15.000"), ("mic", "18.100")])
            reopened.undo(mic)
            self.assertEqual(len(reopened.state[reopened.key(base)]["marks"]), 2)

    def test_legacy_video_state_unambiguous_only(self):
        car = dict(take="same", cls="car", track="1")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"state.json"
            p.write_text(json.dumps({"takes":{"same":{"clap":None, "clap2":None, "marks":[], "history":[], "done":False}}}))
            s = video.Session([car], p, "A")
            self.assertIn(s.key(car), s.state)
            with self.assertRaises(ValueError):
                video.Session([car, dict(car, cls="mic", track="0")], p, "A")

    def test_static_helmet_ignores_moving_mic_readings(self):
        src = [dict(take="fixture", cls="horn", track="1", t=str(t), x="0", y="3", z="1.8") for t in (5, 6)]
        mic = [dict(take="fixture", cls="mic", track="0", t=str(t), x=str(t), y="0") for t in (0, 10)]
        ann = dict(clip_id="static", orig_file="fixture.wav", cut_offset_s="0", scored="1", mic_z="1.8", **{"class":"horn", "状態":"静止", "装着":"ヘルメット", "三脚の向き":"180"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gt.build(src, [ann], root/"a", 1.6, None)
            gt.build(src+mic, [ann], root/"b", 1.6, None)
            self.assertEqual((root/"a/gt/static.csv").read_bytes(), (root/"b/gt/static.csv").read_bytes())

    def test_clap_peak_is_warning_not_conversion_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); p = root/"clap.wav"
            wav = np.zeros((24000, 4)); wav[2400:2420, 0] = .995
            sf.write(p, wav, 24000, subtype="FLOAT")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                out = audio.convert_with_gain(p, root/"out", 0, 0, 0, 0)
            self.assertTrue(out.exists())
            self.assertIn("CLIP注意", log.getvalue())

    def test_clap_exclusion_preserves_audio_and_other_warning(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/"261010_001.WAV"
            wav=np.zeros((24000,4)); wav[2400:2420,0]=.995
            sf.write(p,wav,24000,subtype="FLOAT")
            original=audio.convert_with_gain(p,root/"original",0,0,0,0)
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                masked=audio.convert_with_gain(p,root/"masked",0,0,0,0,((.09,.11),))
            self.assertEqual(original.read_bytes(),masked.read_bytes())
            self.assertNotIn("CLIP注意",log.getvalue())
            rec=json.loads(masked.with_suffix(".peak.json").read_text(encoding="utf-8"))
            self.assertFalse(rec["clip_warning_outside_clap"])
            self.assertGreater(rec["raw"]["peak_clap"],.99)
            wav[12000:12020,0]=.995;sf.write(p,wav,24000,subtype="FLOAT")
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                audio.convert_with_gain(p,root/"outside",-6,0,0,0,((.09,.11),))
            self.assertIn("CLIP注意",log.getvalue())  # raw clipping remains visible even after lowering gain
            for window in ((0,1),(-1,.2),(.8,.5),(0,2),(float("nan"),.5)):
                with self.assertRaises(ValueError): audio.peak_report(wav,24000,(window,))

    def test_uppercase_wav_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            for name in ("261010_001.WAV","lower.wav","mixed.WaV","ignore.txt"):
                (p/name).touch()
            self.assertEqual(len(audio.wav_files(p)),3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
