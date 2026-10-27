# -*- coding: utf-8 -*-
"""test_field_clip_names.py — 実録クリップの命名 A（eval_field_clip<SS><NNN>・2026-09-22 本人決定）の検査。

合成の 4ch 音声 2 本（2 セッション）と step19d 形式の注釈・セッション表を仮に作り、step19b を event → negative（--append）の順に走らせて、
  1. 名前が eval_field_clip01001, 01002（セッション 1）・02001（セッション 2）になる
  2. --append の負例は同じセッションの続き番号（01003…・02002…）になり、既存と衝突しない
     （2026-10-25: 負例の段は none の行がある原本だけを切る＝敵対レビュー 重大 2。旧の試験は正例の本を負例に切らせていたので、各セッションに負例の本を 1 本ずつ足した）
  3. 注釈に session_no が入り、eval_field_index.md が出て全クリップを載せる
  4. 対応関数 parse_field で名前からセッション番号・通し番号が戻る
使い方: python scripts/test_field_clip_names.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.clip_ids import is_new, parse_field  # noqa: E402

PY = sys.executable
FS = 24000
COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa", "take_id", "session_id", "orig_file", "区分", "状態", "横距離m"]


def wav(path: Path, sec: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((int(sec * FS), 4)) * 0.01).astype(np.float32)
    sf.write(path, x, FS, subtype="PCM_24")


def run(args):
    r = subprocess.run([PY, str(ROOT / "scripts/step19b_realsmoke_cut.py")] + args, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    bad = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "conv"; src.mkdir()
        wav(src / "ZOOM0002.flac", 30.0, 1)
        wav(src / "ZOOM0003.flac", 25.0, 2)
        wav(src / "ZOOM0004.flac", 30.0, 3)          # 2026-10-25: 負例の本（none の行）をセッションごとに 1 本。step19b の負例は none の行が無い原本を切らない（敵対レビュー 重大 2）
        wav(src / "ZOOM0005.flac", 20.0, 4)
        ann = tmp / "ann_orig.csv"
        rows = [
            dict(clip_id="ZOOM0002", event_id="1", trial="A", **{"class": "car_drive"}, quadrant="R", t_start="8.0", t_cpa="12.0", take_id="1", session_id="20261005_A1", orig_file="ZOOM0002.flac", 区分="A", 状態="静止", 横距離m="2.0"),
            dict(clip_id="ZOOM0002", event_id="2", trial="A", **{"class": "car_drive"}, quadrant="L", t_start="17.0", t_cpa="21.0", take_id="1", session_id="20261005_A1", orig_file="ZOOM0002.flac", 区分="A", 状態="静止", 横距離m="2.0"),
            dict(clip_id="ZOOM0003", event_id="1", trial="B", **{"class": "kick"}, quadrant="R", t_start="10.0", t_cpa="15.0", take_id="1", session_id="20261005_B1", orig_file="ZOOM0003.flac", 区分="E", 状態="歩行", 横距離m="1.5"),
            dict(clip_id="ZOOM0004", event_id="1", trial="G", **{"class": "none"}, quadrant="", t_start="0", t_cpa="30.0", take_id="2", session_id="20261005_A1", orig_file="ZOOM0004.flac", 区分="G", 状態="静止", 横距離m=""),
            dict(clip_id="ZOOM0005", event_id="1", trial="G", **{"class": "none"}, quadrant="", t_start="0", t_cpa="20.0", take_id="2", session_id="20261005_B1", orig_file="ZOOM0005.flac", 区分="G", 状態="静止", 横距離m=""),
        ]
        with open(ann, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        with open(tmp / "ann_orig_sessions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["session_id", "session_no", "calib_file", "calibration_id", "check_file", "点検方式", "excluded_files", "用途"]); w.writeheader()
            w.writerow({"session_id": "20261005_A1", "session_no": 1, "calib_file": "", "calibration_id": "", "check_file": "", "点検方式": "", "excluded_files": "", "用途": "最終評価"})
            w.writerow({"session_id": "20261005_B1", "session_no": 2, "calib_file": "", "calibration_id": "", "check_file": "", "点検方式": "", "excluded_files": "", "用途": "最終評価"})
        out = tmp / "clips"; ann_out = tmp / "ann_all.csv"
        rc, log = run(["--in", str(src), "--ann", str(ann), "--out", str(out), "--ann-out", str(ann_out), "--mode", "event"])
        if rc != 0:
            bad.append(f"event モードが失敗: {log[-800:]}")
        else:
            names = sorted(p.stem for p in out.glob("*.flac"))
            if names != ["eval_field_clip01001", "eval_field_clip01002", "eval_field_clip02001"]:
                bad.append(f"event の名前が想定外: {names}")
            rc2, log2 = run(["--in", str(src), "--ann", str(ann), "--out", str(out), "--ann-out", str(ann_out), "--mode", "negative", "--append"])
            if rc2 != 0:
                bad.append(f"negative --append が失敗: {log2[-800:]}")
            else:
                names2 = sorted(p.stem for p in out.glob("*.flac"))
                s1 = [n for n in names2 if n.startswith("eval_field_clip01")]
                s2 = [n for n in names2 if n.startswith("eval_field_clip02")]
                if s1[:3] != ["eval_field_clip01001", "eval_field_clip01002", "eval_field_clip01003"] or len(s1) < 4:
                    bad.append(f"セッション 1 の続き番号が想定外: {s1}")
                if s2[:2] != ["eval_field_clip02001", "eval_field_clip02002"]:
                    bad.append(f"セッション 2 の続き番号が想定外: {s2}")
                if len(names2) != len(set(names2)):
                    bad.append("名前が重複")
                arows = list(csv.DictReader(open(ann_out, encoding="utf-8-sig")))
                if not all(is_new(r["clip_id"]) for r in arows):
                    bad.append("注釈に新名でない clip_id がある")
                if not all((r.get("session_no") or "") for r in arows):
                    bad.append("注釈に session_no の無い行がある")
                for r in arows:
                    s_no, k = parse_field(r["clip_id"])
                    if str(s_no) != r["session_no"]:
                        bad.append(f"名前のセッション番号と session_no が違う: {r['clip_id']} vs {r['session_no']}"); break
                idx = ann_out.with_name("eval_field_index.md")
                if not idx.exists():
                    bad.append("eval_field_index.md が無い")
                else:
                    txt = idx.read_text(encoding="utf-8")
                    miss = [n for n in names2 if n not in txt]
                    if miss:
                        bad.append(f"一覧に無いクリップ: {miss[:3]}")
                    if "20261005_A1" not in txt or "kick" not in txt:
                        bad.append("一覧にセッションやクラスが載っていない")
                print(f"event 3 本 + negative {len(names2) - 3} 本 = {len(names2)} 本。例: {names2[:4]} … 一覧 {idx.name}")
    print("=" * 60)
    print("合格" if not bad else "不合格")
    for b in bad:
        print("  NG", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
