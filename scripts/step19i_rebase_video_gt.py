# -*- coding: utf-8 -*-
"""Step 19i: 動画から読んだ正解（原録音の時刻）を、切り出した 10 秒クリップの時刻に移す（2026-09-27・敵対レビュー D の D03）。

背景: step19h の読み（`reads_<人>.csv`・t = 原録音の秒）は 20〜30 秒のテイクの時刻。知覚層の採点は step19b が切り出した
      10 秒クリップ（`eval_field_clip…`）の時刻 0〜10 s で行う。両者をつなぐ道具が無く、最接近が 10 s より後なら
      step19f は黙って 0 行を出していた。この道具が ann_all.csv の cut_offset_s でずらし、クリップ ID ごとの正解 csv を書く。

入力:
  --reads  step19h の出力（take,frame,cls,track,x,y[,z][,t]）。take は原本のファイル名（拡張子は有っても無くてもよい）
  --ann    step19b の出力（ann_all.csv。clip_id, orig_file, cut_offset_s, scored, class, t_cpa[, mic_z]）
  --out    正解 csv の置き場（<out>/<clip_id>.csv・6 列 frame,cls,track,az,el,dist。学習の正解と同じ形）
  --mic-z  マイク高さ [m]（ann に mic_z 列があればそちらを優先）
  --src-z  音源の高さ [m]（省略時はクラスごとの既定: car 0.325・kick 0.3・bike 0.6・bike_bell 1.0・その他〔自転車を含む〕0.5・#70 r16）
出力: 各クリップの正解 csv と、<out>/rebase_report.csv（clip_id, take, cut_offset_s, n_rows, cpa_frame, min_dist, note）

使い方:
  python scripts/step19i_rebase_video_gt.py --reads out/reads/reads_A.csv --ann out/realsmoke/ann_all.csv --out out/realsmoke/gt_video --mic-z 1.6
  python scripts/step19i_rebase_video_gt.py --selftest out/_test_step19i
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("step19f_video_to_gt", ROOT / "scripts" / "step19f_video_to_gt.py")
F19 = importlib.util.module_from_spec(_spec); sys.modules["step19f_video_to_gt"] = F19; _spec.loader.exec_module(F19)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_Z = {4: 0.325, 6: 0.3, 7: 0.6, 3: 1.0}   # car / kick / bike / bike_bell。#70 r16（2026-10-19）: 車 0.325・ベル 1.0（step19k と同じ・実測があればそれが優先）。旧 = 0.5／0.6（座標一覧 §1 の表）
CLIP = 10.0


def norm(name: str) -> str:
    s = name.strip().lower().replace("\\", "/").split("/")[-1]
    for ext in (".wav", ".flac", ".mov", ".mp4"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s


def rebase(reads: list[dict], ann: list[dict], out: Path, mic_z_default: float, src_z: float | None) -> list[dict]:
    by_take: dict[str, list[dict]] = {}
    for r in reads:
        by_take.setdefault(norm(r["take"]), []).append(r)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for a in ann:
        if str(a.get("scored", "1")).strip() != "1" or a.get("class", "").strip() == "none":
            continue
        take = norm(a.get("orig_file", "") or "")
        clip = a["clip_id"].strip()
        rows = by_take.get(take)
        if not rows:
            report.append({"clip_id": clip, "take": take, "cut_offset_s": a.get("cut_offset_s", ""), "n_rows": 0, "cpa_frame": "", "min_dist": "",
                           "note": "読みが無い（動画なし・未読み）"})
            continue
        off = float(a["cut_offset_s"])
        mz = float(a["mic_z"]) if a.get("mic_z") not in (None, "", "nan") else mic_z_default
        # クラスごとの音源高さ: 行の z が空なら src_z、それも無ければ既定表
        rows2 = []
        for r in rows:
            r2 = dict(r)
            if r2.get("z") in (None, "", "nan"):
                c = F19.cls_of(r2["cls"])
                r2["z"] = str(src_z if src_z is not None else SRC_Z.get(c, 0.5))
            rows2.append(r2)
        conv = F19.convert(rows2, mz, src_z if src_z is not None else 0.5, sync_offset=-off, clip_len=CLIP)
        lines = conv.get(rows[0]["take"].strip(), [])
        (out / f"{clip}.csv").write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        cpa_frame, min_d = "", ""
        if lines:
            parsed = [(int(s.split(",")[0]), float(s.split(",")[5])) for s in lines]
            k, d = min(parsed, key=lambda x: x[1]); cpa_frame, min_d = k, f"{d:.2f}"
        note = "" if lines else "0 行（最接近がクリップの外か、読み取り点がクリップの外）"
        if lines and (k < 5 or k > 95):
            note = "最接近がクリップの端（cpa_at と cut_offset を確認）"
        report.append({"clip_id": clip, "take": take, "cut_offset_s": f"{off:.3f}", "n_rows": len(lines), "cpa_frame": cpa_frame, "min_dist": min_d, "note": note})
    with open(out / "rebase_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "take", "cut_offset_s", "n_rows", "cpa_frame", "min_dist", "note"]); w.writeheader(); w.writerows(report)
    return report


def selftest(tmp: Path) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    # 原録音 25 秒のテイク。車が y = 2.0 の線を x = +10 → −10 へ 4 m/s、最接近（x = 0）は 16.5 s
    reads = [{"take": "ZOOM0007.wav", "frame": "", "cls": "car", "track": "1", "x": str(x), "y": "2.0", "z": "", "t": f"{16.5 + (0 - x) / 4.0:.3f}"}
             for x in (10.0, 5.0, 0.0, -5.0, -10.0)]
    ann = [{"clip_id": "eval_field_clip01003", "orig_file": "ZOOM0007.wav", "cut_offset_s": "8.500", "scored": "1", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6"},
           {"clip_id": "eval_field_clip01004", "orig_file": "ZOOM0007.wav", "cut_offset_s": "30.000", "scored": "1", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6"},
           {"clip_id": "eval_field_clip01005", "orig_file": "ZOOM0008.wav", "cut_offset_s": "5.000", "scored": "1", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6"},
           {"clip_id": "eval_field_clip01006", "orig_file": "ZOOM0007.wav", "cut_offset_s": "8.500", "scored": "0", "class": "car_drive", "t_cpa": "8.00", "mic_z": "1.6"}]
    rep = rebase(reads, ann, tmp / "gt", 1.6, None)
    byc = {r["clip_id"]: r for r in rep}
    r3 = byc["eval_field_clip01003"]
    ok = [("最接近がクリップの 8.0 s（frame 80 ± 1）に来る", r3["n_rows"] > 0 and abs(int(r3["cpa_frame"]) - 80) <= 1),
          ("最接近距離 ≈ 2.0 m の水平距離（±0.1）", r3["min_dist"] != "" and abs(float(r3["min_dist"]) - 2.0) < 0.1),
          ("正解の行が 0〜99 フレームの範囲", all(0 <= int(s.split(",")[0]) <= 99 for s in open(tmp / "gt" / "eval_field_clip01003.csv", encoding="utf-8").read().split("\n") if s)),
          ("最接近がクリップの外なら 0 行と注意", byc["eval_field_clip01004"]["n_rows"] == 0 and "0 行" in byc["eval_field_clip01004"]["note"]),
          ("読みの無いテイクは注意だけ", byc["eval_field_clip01005"]["n_rows"] == 0 and "読みが無い" in byc["eval_field_clip01005"]["note"]),
          ("scored=0 の行は作らない", "eval_field_clip01006" not in byc),
          ("報告 csv がある", (tmp / "gt" / "rebase_report.csv").exists())]
    # 音源高さの既定表: kick は 0.3 m → 仰角がわずかに下（mic 1.6 m）になる行があるか
    reads_k = [dict(r, cls="kick") for r in reads]
    rep_k = rebase(reads_k, ann[:1], tmp / "gt_kick", 1.6, None)
    lines_k = [s for s in open(tmp / "gt_kick" / "eval_field_clip01003.csv", encoding="utf-8").read().split("\n") if s]
    ok.append(("kick は既定の高さ 0.3 m で仰角が負", rep_k[0]["n_rows"] > 0 and all(int(s.split(",")[4]) < 0 for s in lines_k)))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    st = opt("--selftest")
    if st:
        return selftest(Path(st))
    reads_p, ann_p, out = opt("--reads"), opt("--ann"), opt("--out")
    if not (reads_p and ann_p and out):
        print(__doc__); return 2
    mic_z = float(opt("--mic-z", "1.6")); src_z = opt("--src-z"); src_z = float(src_z) if src_z else None
    with open(reads_p, newline="", encoding="utf-8-sig") as f:
        reads = list(csv.DictReader(f))
    with open(ann_p, newline="", encoding="utf-8-sig") as f:
        ann = list(csv.DictReader(f))
    rep = rebase(reads, ann, Path(out), mic_z, src_z)
    n_ok = sum(1 for r in rep if r["n_rows"] > 0 and not r["note"])
    print(f"クリップ {len(rep)} 件: 正解あり {n_ok}・注意 {len(rep) - n_ok} → {Path(out) / 'rebase_report.csv'}")
    for r in rep:
        if r["note"]:
            print(f"  {r['clip_id']}: {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
