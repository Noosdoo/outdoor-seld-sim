# -*- coding: utf-8 -*-
"""test_step19_b2b4_static_hidden.py — 実録の採点の準備 B2（歩く比較の正解・静止の本人の位置）と B4（隠れた時間）の、step19d → step19k の通しの検査（2026-10-20・Fable）。

  D1 step19d: 静止の車の組（台本ID W09・歩く人の静止x_m 2・歩く人の線_m 7.0）→ 注釈に mic_x_m 2・mic_y_m 7、歩行の自転車の組（W05）→ mic_x_m 空・mic_y_m 5、
     歩く比較でない本 → 0・0。歩行の本の mic 行の line_m = 本人の線
  K1 step19k: 静止の本人が (2, 5) に立ち、車が y = 2 の線を通る → 最接近の水平距離 ≈ 3.0 m・方位 ≈ −90°（右）
  K2 step19k: 1 秒より長く隠れた区間（読み 12.0〜14.0 s）の正解の行が無く、外の行はある。0.7 s の隠れは線で埋める（行がある）
  K3 step19k: 本人（mic）が 1 秒より長く隠れた区間の正解の行も無い
使い方: python scripts/test_step19_b2b4_static_hidden.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


D = load("s19d_b2", "step19d_field_csv_to_ann.py")
K = load("s19k_b2", "step19k_field_perception.py")


def main() -> int:
    ok = []
    # --- D1 ---
    sessions = [dict(session_id="w", 用途="最終評価", 区分="歩行対比", マイク高さ_cm="200"), dict(session_id="a", 用途="最終評価", 区分="A", マイク高さ_cm="200")]
    events = [dict(session_id="w", take_id="1", event_id="1", 原本="261029_101.WAV", **{"class": "car_drive"}, 象限="B", ラップ秒="20", 状態="静止", 横距離m="3.0", pair_id="P1",
                   **{"台本ID": "W09", "歩く人の静止x_m": "2", "歩く人の線_m": "7.0", "三脚の向き": "0", "動画原本": "a.mov", "direction": "後→前", "line_m": "3.0"}),
              dict(session_id="w", take_id="2", event_id="1", 原本="261022_102.WAV", **{"class": "bicycle"}, 象限="B", ラップ秒="20", 状態="歩行", 横距離m="2.0", pair_id="P2",
                   **{"台本ID": "W05", "三脚の向き": "0", "動画原本": "b.mov", "direction": "後→前", "line_m": "2.0"}),
              dict(session_id="a", take_id="3", event_id="1", 原本="261029_001.WAV", **{"class": "car_drive"}, 象限="B", ラップ秒="20", 状態="静止", 横距離m="2.0",
                   **{"台本ID": "A2-01", "三脚の向き": "0", "動画原本": "c.mov", "direction": "後→前", "line_m": "2.0"})]
    out, warns = D.convert(sessions, events, 6.0, 0.0, None, 3.0, None)
    by = {r["take_id"]: r for r in out}
    ok.append(("D1 静止の車の組: mic_x_m 2・mic_y_m 7", by["w/01"]["mic_x_m"] == "2" and by["w/01"]["mic_y_m"] == "7"))
    ok.append(("D1 歩行の自転車の組: mic_x_m 空・mic_y_m 5", by["w/02"]["mic_x_m"] == "" and by["w/02"]["mic_y_m"] == "5"))
    ok.append(("D1 歩く比較でない本: 0・0", by["a/03"]["mic_x_m"] == "0" and by["a/03"]["mic_y_m"] == "0"))
    vt = D.video_takes(events)
    mic_rows = [r for r in vt if r["cls"] == "mic"]
    ok.append(("D1 歩行の本の mic 行の line_m = 本人の線 5", len(mic_rows) == 1 and mic_rows[0]["line_m"] == "5"))
    # --- K1 ---
    tmp = Path(tempfile.mkdtemp(prefix="s19k_b2_"))
    try:
        marks = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
        # 車が y = 2.0 を x = −10 → +10 へ 4 m/s（原録音 12.0 s に x = −10）。本人は (2, 5) に静止。x = 2 を 15.0 s に通る
        reads = [{"take": "261029_101.WAV", "frame": "", "cls": "car", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{12.0 + (x + 10) / 4.0:.3f}", "hidden": ""} for x in marks]
        ann = [{"clip_id": "eval_field_clip09101", "orig_file": "261029_101.WAV", "cut_offset_s": "7.000", "scored": "1", "class": "car_drive", "t_cpa": "8.00",
                "mic_z": "2.0", "状態": "静止", "三脚の向き": "0", "mic_x_m": "2", "mic_y_m": "5"}]
        rep = {r["clip_id"]: r for r in K.build(reads, ann, tmp / "k1", 2.0, None)}
        a = rep["eval_field_clip09101"]
        ok.append((f"K1 本人 (2, 5)・車 y=2: 最接近 ≈ 3.0 m（{a['min_dist']}）・方位 ≈ −90°（{a['az_at_cpa']}）・frame 80±2",
                   a["n_rows"] > 0 and abs(float(a["min_dist"]) - 3.0) < 0.12 and abs(int(a["az_at_cpa"]) + 90) <= 10 and abs(int(a["cpa_frame"]) - 80) <= 2))
        # --- K2: 長い隠れ（読み 12.0〜14.0 s の間 1.4 s 隠れる）→ その区間のフレーム（クリップ 5.0〜7.0 s）が無い ---
        hid = [{"take": "261029_101.WAV", "frame": "", "cls": "car", "track": "1", "x": "", "y": "2.0", "z": "", "t": "12.3", "hidden": "start"},
               {"take": "261029_101.WAV", "frame": "", "cls": "car", "track": "1", "x": "", "y": "2.0", "z": "", "t": "13.7", "hidden": "end"}]
        rep2 = {r["clip_id"]: r for r in K.build(reads + hid, ann, tmp / "k2", 2.0, None)}
        lines = [s for s in (tmp / "k2/gt/eval_field_clip09101.csv").read_text(encoding="utf-8").split("\n") if s]
        frames = {int(s.split(",")[0]) for s in lines}
        gap = {k for k in range(50, 70)}                 # 原録音 12.0〜14.0 s = クリップ 5.0〜7.0 s（切り出し 7.0 s）
        ok.append((f"K2 1.4 s 隠れた区間（クリップ 5.0〜7.0 s）の行が無い（{len(frames & gap)} 行）・外は残る（{len(frames - gap)} 行）",
                   rep2["eval_field_clip09101"]["n_rows"] > 0 and not (frames & gap) and len(frames - gap) > 20))
        hid_s = [dict(hid[0], t="12.5"), dict(hid[1], t="13.2")]
        rep3 = {r["clip_id"]: r for r in K.build(reads + hid_s, ann, tmp / "k3", 2.0, None)}
        lines3 = [s for s in (tmp / "k3/gt/eval_field_clip09101.csv").read_text(encoding="utf-8").split("\n") if s]
        ok.append(("K2' 0.7 s の隠れは線で埋める（行数が隠れ無しと同じ）", len(lines3) == len([s for s in (tmp / "k1/gt/eval_field_clip09101.csv").read_text(encoding="utf-8").split("\n") if s])))
        # --- K3: 歩行の本で本人が 1 秒より長く隠れる ---
        reads_w = [{"take": "261022_102.WAV", "frame": "", "cls": "mic", "track": "0", "x": str(x), "y": "5.0", "z": "", "t": f"{10.0 + (x + 6) / 1.2:.3f}", "hidden": ""} for x in (-6, -4, -2, 0, 2, 4, 6)]
        reads_w += [{"take": "261022_102.WAV", "frame": "", "cls": "bicycle", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{11.0 + (x + 10) / 3.0:.3f}", "hidden": ""} for x in marks]
        hid_m = [{"take": "261022_102.WAV", "frame": "", "cls": "mic", "track": "0", "x": "", "y": "5.0", "z": "", "t": "13.5", "hidden": "start"},
                 {"take": "261022_102.WAV", "frame": "", "cls": "mic", "track": "0", "x": "", "y": "5.0", "z": "", "t": "15.0", "hidden": "end"}]
        ann_w = [{"clip_id": "eval_field_clip09102", "orig_file": "261022_102.WAV", "cut_offset_s": "8.000", "scored": "1", "class": "bicycle", "t_cpa": "8.00",
                  "mic_z": "2.0", "状態": "歩行", "三脚の向き": "0", "mic_x_m": "", "mic_y_m": "5"}]
        rep4 = {r["clip_id"]: r for r in K.build(reads_w + hid_m, ann_w, tmp / "k4", 2.0, None)}
        lines4 = [s for s in (tmp / "k4/gt/eval_field_clip09102.csv").read_text(encoding="utf-8").split("\n") if s]
        f4 = {int(s.split(",")[0]) for s in lines4}
        # 本人の読み: 13.33 s（x=−2）… 15.0 s（x=0）。隠れ 13.5〜15.0 → 広げて [13.333, 15.0] = クリップ 5.33〜7.0 s
        gap4 = {k for k in range(54, 70)}
        ok.append((f"K3 本人が 1.5 s 隠れた区間（クリップ 5.33〜7.0 s）の行が無い（{len(f4 & gap4)} 行）・外は残る", rep4["eval_field_clip09102"]["n_rows"] > 0 and not (f4 & gap4) and len(f4 - gap4) > 10))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
