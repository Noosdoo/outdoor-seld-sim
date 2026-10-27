# -*- coding: utf-8 -*-
"""scripts/demo_field_unused_fix.py の試験（Astra R5-30-04・2026-10-27 Opus）。架空の置き場だけ（本物の読みや録音は開かない）。
  F1 --write で記録を作る（読み取り専用の印・本数・content の sha）        F2 --check: 何も変えなければ一致（rc 0）
  F3 1 本の読みの行（x の値）を変える → 本ごとの違いにその本が出る（rc 1）
  F4 reads_A.csv の行の並びだけ変える → 本ごとの違いは無く、ファイル全体の sha の違いだけ出る（rc 1）
  F5 違うのに理由なしで --write → rc 3・記録は変わらない              F6 理由つきの --write → 履歴に理由と旧新の sha
  F7 一覧に無い変換音が置き場にある → 知らせる（記録には入れない・本は 19 本の一覧のまま）
  F8 fixed_takes は記録の本だけを返す（フォルダの中を拾わない）
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import demo_field_unused_fix as F  # noqa: E402


def wcsv(p: Path, rows, cols=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def setup(tmp: Path):
    base = tmp / "_demo_field_unused_t"; d = base / "20261022"; led = tmp / "ledger"
    wcsv(tmp / "list.csv", [{"日付": "20261022", "原本": "261022_009.WAV", "台本ID": "C2-04", "区分": "入れる（除外区間あり）", "向きの直し_deg": "0", "線の上書き_m": "", "clip_id": "tune_field_clip82002"},
                            {"日付": "20261022", "原本": "261022_071.WAV", "台本ID": "W02-静止", "区分": "入れる", "向きの直し_deg": "180", "線の上書き_m": "", "clip_id": "tune_field_clip82009"}])
    for n in ("ledger", "events_fixed", "session_fixed"):
        wcsv(led / f"{n}_20261022.csv", [{"a": n}])
    (d / "conv").mkdir(parents=True)
    for s in ("261022_009", "261022_071"):
        (d / "conv" / f"{s}_conv.flac").write_bytes(s.encode())
    wcsv(d / "index.csv", [{"原本": "261022_009.WAV", "conv": "conv/261022_009_conv.flac", "条件の写し元": "261022_010.WAV"},
                           {"原本": "261022_071.WAV", "conv": "conv/261022_071_conv.flac", "条件の写し元": "261022_083.WAV"}])
    wcsv(d / "reads" / "reads_A.csv", [{"take": "261022_009.WAV", "frame": "70", "cls": "bike_bell", "track": "1", "x": "-12.0", "y": "1.0", "z": "1.0", "t": "7.158", "hidden": ""},
                                       {"take": "261022_009.WAV", "frame": "80", "cls": "bike_bell", "track": "1", "x": "-10.0", "y": "1.0", "z": "1.0", "t": "7.700", "hidden": ""},
                                       {"take": "261022_071.WAV", "frame": "90", "cls": "bike_bell", "track": "1", "x": "-12.0", "y": "2.0", "z": "1.0", "t": "9.000", "hidden": ""}])
    wcsv(d / "reads" / "sync_A.csv", [{"take": "261022_009.WAV", "clap": "1", "audio_s": "3.6892", "video_s": "3.468"}])
    (d / "reads" / "reads_A.state.json").write_text("{}", encoding="utf-8")
    (d / "reads" / "audio_reads_A.json").write_text(json.dumps({"261022_009.WAV": {"clap_audio_s": 3.6892, "bell_onset_s": 9.1}}), encoding="utf-8")
    wcsv(d / "除外区間メモ.csv", [{"原本": "261022_009.WAV", "自転車が最後まで通った（○／×）": "○", "除外の入り_音の秒": "13.2", "除外の出_音の秒": "", "メモ": ""}])
    for n in ("ann_demo_video_takes.csv", "events_demo.csv", "sessions_demo.csv"):
        wcsv(d / n, [{"x": n}])
    (base / "index.json").write_text("{}", encoding="utf-8")
    return base, d, led


def run(tmp, base, led, *extra):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = F.main(["--list", str(tmp / "list.csv"), "--base", str(base), "--record", str(tmp / "rec.json"), "--ledger-dir", str(led), "--root", str(tmp), *extra])
    return rc, buf.getvalue()


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="demo_fix_"))
    try:
        base, d, led = setup(tmp)
        rc, out = run(tmp, base, led)
        ok.append(("記録が無いときの --check は rc 2", rc == 2))
        rc, out = run(tmp, base, led, "--write")
        rec = json.loads((tmp / "rec.json").read_text(encoding="utf-8"))
        ro = not (os.stat(tmp / "rec.json").st_mode & stat.S_IWRITE)
        ok.append(("F1 --write で記録を作る（2 本・読み取り専用の印・content の sha・本ごとに除外区間と拍手）", rc == 0 and len(rec["takes"]) == 2 and ro and rec["content_sha256"]
                   and rec["takes"][0]["除外区間"]["入り_音の秒"] == "13.2" and rec["takes"][0]["sync"]["audio_s"] == "3.6892" and rec["takes"][0]["n_marks"] == 2))
        rc, out = run(tmp, base, led)
        ok.append(("F2 何も変えなければ一致（rc 0）", rc == 0 and "一致" in out))
        p = d / "reads" / "reads_A.csv"
        orig = p.read_bytes()
        rows = F.read_csv(p); rows[2]["x"] = "-11.5"; wcsv(p, rows)
        rc, out = run(tmp, base, led)
        ok.append(("F3 1 本の読みの行を変える → 本ごとの違いにその本（071）が出る（rc 1）", rc == 1 and "tune_field_clip82009（261022_071.WAV）の reads_rows_sha256" in out
                   and "tune_field_clip82002" not in out.split("ファイル全体")[0]))
        p.write_bytes(orig)
        rows = F.read_csv(p); wcsv(p, [rows[1], rows[0], rows[2]])
        rc, out = run(tmp, base, led)
        ok.append(("F4 行の並びだけ変える → 本ごとの違いは 0 件・ファイル全体の sha の違いだけ（rc 1）", rc == 1 and "本ごとの値の違い（意味のある変化）: 0 件" in out and "reads_A.csv: sha" in out))
        before = (tmp / "rec.json").read_bytes()
        rc, out = run(tmp, base, led, "--write")
        ok.append(("F5 違うのに理由なしで --write → rc 3・記録は変わらない", rc == 3 and (tmp / "rec.json").read_bytes() == before))
        rc, out = run(tmp, base, led, "--write", "--reason", "試験: 行の並びを変えた")
        rec2 = json.loads((tmp / "rec.json").read_text(encoding="utf-8"))
        h = rec2["history"][-1] if rec2["history"] else {}
        ok.append(("F6 理由つきの --write → 履歴に理由と旧新の sha", rc == 0 and h.get("reason") == "試験: 行の並びを変えた" and h.get("old_content_sha256") == rec["content_sha256"]
                   and h.get("new_content_sha256") == rec2["content_sha256"] != rec["content_sha256"]))
        (d / "conv" / "261022_019_conv.flac").write_bytes(b"x")
        rc, out = run(tmp, base, led)
        ok.append(("F7 一覧に無い変換音を知らせる（記録の本は 2 本のまま）", "261022_019_conv.flac" in out and len(F.build(tmp / "list.csv", base, led, tmp)["takes"]) == 2))
        ok.append(("F8 fixed_takes は記録の本だけ", [t["clip_id"] for t in F.fixed_takes(tmp / "rec.json")] == ["tune_field_clip82002", "tune_field_clip82009"]))
    finally:
        rp = tmp / "rec.json"
        if rp.exists():
            os.chmod(rp, stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
