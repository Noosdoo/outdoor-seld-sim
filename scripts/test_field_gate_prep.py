# -*- coding: utf-8 -*-
"""scripts/field_gate_prep.py の試験（2026-10-27 Opus）。架空の読みだけ（実録は読まない）。
  G1 ベルの本の bike_bell の読み → bicycle の写しを足す（元の行は残る）・mic の行は写さない
  G2 走りながらのクラクション E3（台本ID E3-）の horn の読み → car_drive の写し・止めた E2（向きも線も無い）は写さない
  G3 警告音の区間: ベルと E3 は audio_reads の b・e・E2 は reads_fixed の t の最小〜最大
  G4 読み待ち（鳴り終わりが無い）・鳴り終わり ≤ 鳴り始め → 終了コード 2・何も書かない
  G5 step19l の変換済みの読み（converted の列）を渡したら止める
  G6 出力の区間の表を step19k の load_warn_interval がそのまま読める
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import field_gate_prep as G  # noqa: E402
import step19k_field_perception as K  # noqa: E402


def wcsv(p, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def run(args):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = G.main([str(x) for x in args])
    return rc, buf.getvalue()


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="gateprep_"))
    try:
        rc_cols = ["take", "frame", "cls", "track", "x", "y", "z", "t"]
        rd = lambda take, cls, xs, trk="1": [dict(take=take, frame="", cls=cls, track=trk, x=str(x), y="2", z="", t=str(10 + i)) for i, x in enumerate(xs)]   # noqa: E731
        reads = rd("261022_071.WAV", "bike_bell", [-10, 0, 10]) + rd("261022_071.WAV", "mic", [0, 1], trk="0") + rd("261029_020.WAV", "horn", [10, 0, -10]) + rd("261029_005.WAV", "car_drive", [-10, 10])
        wcsv(tmp / "reads.csv", rc_cols, reads)
        ev_cols = ["原本", "class", "台本ID", "direction", "line_m"]
        events = [{"原本": "261022_071.WAV", "class": "bike_bell", "台本ID": "W02-静止", "direction": "後→前", "line_m": "2.0"},
                  {"原本": "261029_020.WAV", "class": "horn", "台本ID": "E3-03", "direction": "前→後", "line_m": "2.0"},
                  {"原本": "261029_004.WAV", "class": "horn", "台本ID": "E2-03", "direction": "", "line_m": ""},
                  {"原本": "261029_005.WAV", "class": "car_drive", "台本ID": "A2-01", "direction": "後→前", "line_m": "2.0"}]
        wcsv(tmp / "events.csv", ev_cols, events)
        (tmp / "aud.json").write_text(json.dumps({"261022_071.WAV": {"clap_audio_s": 3.0, "bell_onset_s": 12.5, "bell_end_s": 13.1},
                                                   "261029_020.WAV": {"clap_audio_s": 2.0, "bell_onset_s": 11.0, "bell_end_s": 11.4}}), encoding="utf-8")
        wcsv(tmp / "fixed.csv", rc_cols, [dict(take="261029_004.WAV", frame="", cls="horn", track="1", x="-3", y="2", z="0.8", t="9.000"),
                                           dict(take="261029_004.WAV", frame="", cls="horn", track="1", x="-3", y="2", z="0.8", t="9.600")])
        args = ["--reads", tmp / "reads.csv", "--audio-reads", tmp / "aud.json", "--events", tmp / "events.csv", "--reads-fixed", tmp / "fixed.csv",
                "--out-reads", tmp / "out" / "reads_gate.csv", "--out-warn", tmp / "out" / "warn.csv"]
        rc, out = run(args)
        R = list(csv.DictReader(open(tmp / "out" / "reads_gate.csv", encoding="utf-8")))
        bike = [r for r in R if r["cls"] == "bicycle"]; car = [r for r in R if r["cls"] == "car_drive"]
        ok.append(("G1 ベルの本の bike_bell → bicycle の写し 3 行（元の 3 行は残る・mic は写さない）",
                   rc == 0 and len(bike) == 3 and all(r["take"] == "261022_071.WAV" for r in bike) and sum(r["cls"] == "bike_bell" for r in R) == 3 and sum(r["cls"] == "mic" for r in R) == 2))
        ok.append(("G2 E3 の horn → car_drive の写し 3 行・A2 の車はそのまま 2 行（止めた E2 は読みが無いので何もしない）",
                   sum(r["take"] == "261029_020.WAV" for r in car) == 3 and sum(r["take"] == "261029_005.WAV" for r in car) == 2 and sum(r["cls"] == "horn" for r in R) == 3))
        W = {(r["take"], r["class"]): (r["on_s"], r["end_s"]) for r in csv.DictReader(open(tmp / "out" / "warn.csv", encoding="utf-8"))}
        ok.append(("G3 区間: ベル 12.5〜13.1・E3 11.0〜11.4（b・e）・E2 9.0〜9.6（reads_fixed）",
                   W == {("261022_071.WAV", "bike_bell"): ("12.5000", "13.1000"), ("261029_020.WAV", "horn"): ("11.0000", "11.4000"), ("261029_004.WAV", "horn"): ("9.0000", "9.6000")}))
        ok.append(("G6 step19k の load_warn_interval が読める", len(K.load_warn_interval(tmp / "out" / "warn.csv")) == 3))
        (tmp / "aud2.json").write_text(json.dumps({"261022_071.WAV": {"bell_onset_s": 12.5}, "261029_020.WAV": {"bell_onset_s": 11.0, "bell_end_s": 10.9}}), encoding="utf-8")
        shutil.rmtree(tmp / "out2", ignore_errors=True)
        args2 = [x if x != tmp / "aud.json" else tmp / "aud2.json" for x in args]
        args2 = [tmp / "out2" / Path(str(x)).name if str(x).startswith(str(tmp / "out")) else x for x in args2]
        rc2, out2 = run(args2)
        ok.append(("G4 鳴り終わりが無い・鳴り終わり ≤ 鳴り始め → 2・何も書かない", rc2 == 2 and "261022_071.WAV" in out2 and "261029_020.WAV" in out2 and not (tmp / "out2").exists()))
        # G7（Astra R142-04）: E3 の本の前輪の読み（horn）が無い → 2
        wcsv(tmp / "reads_noE3.csv", rc_cols, [r for r in reads if r["take"] != "261029_020.WAV"])
        rc7, o7 = run(["--reads", tmp / "reads_noE3.csv"] + args[2:-4] + ["--out-reads", tmp / "o7" / "r.csv", "--out-warn", tmp / "o7" / "w.csv"])
        ok.append(("G7 E3 の本に前輪の読み（horn）が無い → 2・何も書かない", rc7 == 2 and "前輪の読み（cls horn）が無い" in o7 and not (tmp / "o7").exists()))
        wcsv(tmp / "conv.csv", rc_cols + ["converted"], [dict(r, converted="") for r in reads])
        rc3, out3 = run(["--reads", tmp / "conv.csv"] + args[2:])
        ok.append(("G5 変換済みの読みを渡したら止める", rc3 == 2 and "step19l の前" in out3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
