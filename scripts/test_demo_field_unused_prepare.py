# -*- coding: utf-8 -*-
"""scripts/demo_field_unused_prepare.py の試験（2026-10-26 Opus・本人 22:02 の決定・Fable の仕様）。架空の台帳と架空の録音だけ（本物の録音は開かない）。

守り（評価の本が 1 本でもあれば終了コード 2・何も書かない）:
  G1 評価の本（台帳で「本」）が一覧に 1 本 → 2・置き場を作らない      G2 原本名にワイルドカード → 2
  G3 台本ID が台帳の役割の文に無い（取り違え） → 2                    G4 原本名の日付と一覧の日付が違う → 2
  G5 events で条件ID が入っている本 → 2                               G6 置き場の名前が _demo_field_unused で始まらない・raw/ の中 → 2
  G9 台帳で同じ録音名が 2 行（本／使わない の矛盾・両方の順）→ 2    G10 events が無い → 2（空の表として許さない）
  G7 校正の録音（役割が「使わない」でない） → 2                        G8 区分「参考（向き）」なのに向きの直し 0 → 2（本人 22:28 から「入れる」の 180 は可）
変換:
  C1 守りを通った一覧で変換（条件の行は評価の本の行の写し・原本と動画を差し替え・拍手とラップ秒と条件ID は空・index と台帳の sha）
  C2 線の上書き（line_m と 横距離m）
向きの直し（10/22 の歩く比較の 1 回目・マイクが 180° 逆向き）:
  O1 変換の yaw 180 で、逆向きのマイクで録った音（方位 θ + 180°）が装着の正面基準の θ に戻る（強度ベクトルの方位・±2°）。yaw 0 なら θ + 180° のまま
  O2 step19k: 静止は三脚の向き 180 で正解の方位が 180° 回る（＝正解側でも直せる）・歩行は三脚の向きを使わない（歩く向きが正面）＝正解側では直せない
     → 8 本は音の側（変換の yaw 180）で直し、正解は評価の本と同じ扱いにする
本物の台帳（あれば・表だけ読む）:
  R1 md/design/demo_field_unused_list_2026-10-26.csv が守りを通る（評価の本 0）   R2 その一覧に評価の本を 1 本混ぜると 2
"""
from __future__ import annotations

import csv
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import demo_field_unused_prepare as P  # noqa: E402

FS = 24000
LED_COLS = ["録音", "開始", "秒", "動画", "時刻が重なる動画", "役割"]
EV_COLS = ["session_id", "take_id", "event_id", "原本", "class", "動画原本", "ラップ秒", "拍手_録音側秒", "拍手_動画側秒", "三脚の向き", "装着",
           "line_m", "横距離m", "direction", "特記", "台本ID", "条件ID"]
SS_COLS = ["session_id", "区分", "装着", "マイク高さ_cm", "校正原本", "点検原本", "LAeq_dB", "暗騒音区間_秒", "用途", "備考"]


def wcsv(p: Path, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def plane_wave(az_deg: float, sec=2.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(int(sec * FS)) * 0.05
    a = math.radians(az_deg)
    return np.column_stack([s, s * math.sin(a), np.zeros_like(s), s * math.cos(a)])      # AmbiX W, Y, Z, X（仰角 0）


def iv_az(path: Path) -> float:
    x, _ = sf.read(str(path), dtype="float64", always_2d=True)
    w, y, xx = x[:, 0], x[:, 1], x[:, 3]
    return math.degrees(math.atan2(float(np.mean(w * y)), float(np.mean(w * xx))))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def build_raw(root: Path, extra_ev=(), led_before=(), led_after=(), no_events=False):
    raw = root / "raw"
    led = raw / "ledger"
    wcsv(led / "ledger_20261022.csv", LED_COLS, list(led_before) + [
        {"録音": "261022_002.WAV", "秒": "3", "役割": "校正 C1: 校正"},
        {"録音": "261022_009.WAV", "秒": "2", "動画": "IMG_5530.MOV", "役割": "使わない C1: やり直し（C2-04）・車が通った"},
        {"録音": "261022_010.WAV", "秒": "2", "動画": "IMG_5531.MOV", "役割": "本 C1 C2-04"},
        {"録音": "261022_071.WAV", "秒": "2", "動画": "IMG_5584.MOV", "役割": "使わない W1: 録り直し（W02-静止）"},
        {"録音": "261022_083.WAV", "秒": "2", "動画": "IMG_5594.MOV", "役割": "本 W1 W02-静止"},
        {"録音": "261022_028.WAV", "秒": "2", "動画": "IMG_5547.MOV", "役割": "使わない H1: やり直し（H2-48）・車が通った"},
        {"録音": "261022_029.WAV", "秒": "2", "動画": "IMG_5548.MOV", "役割": "本 H1 H2-48"},
    ] + list(led_after))
    ev = [
        {"session_id": "20261022_C1", "take_id": "3", "event_id": "1", "原本": "261022_010.WAV", "class": "bike_bell", "動画原本": "IMG_5531.MOV", "ラップ秒": "6.7",
         "拍手_録音側秒": "2.1", "拍手_動画側秒": "2.7", "三脚の向き": "180", "装着": "三脚", "line_m": "1.0", "direction": "後→前", "台本ID": "C2-04", "条件ID": "C2-04"},
        {"session_id": "20261022_W1", "take_id": "1", "event_id": "1", "原本": "261022_083.WAV", "class": "bike_bell", "動画原本": "IMG_5594.MOV", "ラップ秒": "11.8",
         "拍手_録音側秒": "7.0", "拍手_動画側秒": "9.0", "三脚の向き": "0", "装着": "ヘルメット", "line_m": "2.0", "direction": "後→前", "台本ID": "W02-静止", "条件ID": "W02-静止"},
        {"session_id": "20261022_H1", "take_id": "5", "event_id": "1", "原本": "261022_029.WAV", "class": "bicycle", "動画原本": "IMG_5548.MOV", "ラップ秒": "",
         "拍手_録音側秒": "2.1", "拍手_動画側秒": "26.8", "三脚の向き": "180", "装着": "三脚", "line_m": "3.0", "横距離m": "3.0", "direction": "後→前", "台本ID": "H2-48", "条件ID": "H2-48"},
    ] + list(extra_ev)
    if not no_events:
        wcsv(led / "events_fixed_20261022.csv", EV_COLS, ev)
    wcsv(led / "session_fixed_20261022.csv", SS_COLS, [
        {"session_id": s, "区分": s[-2], "装着": "三脚" if s[-2] != "W" else "ヘルメット", "マイク高さ_cm": "150", "校正原本": "261022_002.WAV",
         "点検原本": "", "LAeq_dB": "47.2", "暗騒音区間_秒": "0-2", "用途": "最終評価", "備考": "除外: 261022_009.WAV"} for s in ("20261022_C1", "20261022_H1", "20261022_W1")])
    sf.write(str(raw / "261022_002.WAV"), plane_wave(0, 3.0, 1), FS, subtype="FLOAT", format="WAV")
    sf.write(str(raw / "261022_009.WAV"), plane_wave(30, 2.0, 2), FS, subtype="FLOAT", format="WAV")
    sf.write(str(raw / "261022_028.WAV"), plane_wave(-45, 2.0, 4), FS, subtype="FLOAT", format="WAV")
    sf.write(str(raw / "261022_071.WAV"), plane_wave(60 + 180, 2.0, 3), FS, subtype="FLOAT", format="WAV")   # 逆向きのマイク: 正面基準 60° の相手が −120° に
    return raw


def wlist(p: Path, rows):
    wcsv(p, list(P.COLS), [dict(zip(P.COLS, r)) for r in rows])


OK_ROWS = [("20261022", "261022_009.WAV", "C2-04", "参考", "0", "", "tune_field_clip82002"),
           ("20261022", "261022_071.WAV", "W02-静止", "入れる", "180", "", "tune_field_clip82009")]     # 本人 22:28: 向きを確かめた本は「入れる」で 180


def run(raw: Path, lst: Path, out: Path, *extra) -> int:
    return P.main(["--list", str(lst), "--out", str(out), "--raw-dir", str(raw), "--no-readprep", *extra])


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="demo_unused_"))
    try:
        raw = build_raw(tmp)
        out = tmp / "_demo_field_unused_t"
        # G1〜G8
        cases = {
            "G1 評価の本（台帳で「本」）が 1 本 → 2": OK_ROWS + [("20261022", "261022_010.WAV", "C2-04", "参考", "0", "", "tune_field_clip82003")],
            "G2 原本名にワイルドカード → 2": [("20261022", "261022_0*.WAV", "C2-04", "参考", "0", "", "tune_field_clip82002")],
            "G3 台本ID の取り違え → 2": [("20261022", "261022_009.WAV", "C2-07", "参考", "0", "", "tune_field_clip82002")],
            "G4 原本名の日付と一覧の日付が違う → 2": [("20261015", "261022_009.WAV", "C2-04", "参考", "0", "", "tune_field_clip82002")],
            "G7 校正の録音 → 2": [("20261022", "261022_002.WAV", "C1", "参考", "0", "", "tune_field_clip82002")],
            "G8 区分「参考（向き）」なのに向きの直し 0 → 2": [("20261022", "261022_071.WAV", "W02-静止", "参考（向き）", "0", "", "tune_field_clip82009")],
        }
        for name, rows in cases.items():
            lst = tmp / "list_bad.csv"
            wlist(lst, rows)
            rc = run(raw, lst, out, "--convert")
            ok.append((name + "・置き場を作らない", rc == 2 and not out.exists()))
        raw5 = build_raw(tmp / "g5", extra_ev=[{"session_id": "20261022_C1", "take_id": "2", "event_id": "1", "原本": "261022_009.WAV", "class": "bike_bell",
                                                "台本ID": "C2-04", "条件ID": "C2-04"}])
        lst = tmp / "list_ok.csv"
        wlist(lst, OK_ROWS)
        ok.append(("G5 events で条件ID が入っている本 → 2", run(raw5, lst, tmp / "g5" / "_demo_field_unused_t", "--convert") == 2))
        # G9: 台帳の録音名の重複（同じ録音で 本／使わない が矛盾）は理由によらず止める・両方の順（Astra R5-30-03）
        evil = {"録音": "261022_009.WAV", "秒": "2", "動画": "IMG_5530.MOV", "役割": "本 C1 C2-04"}
        for tag, kw in (("本 → 使わない", {"led_before": [evil]}), ("使わない → 本", {"led_after": [evil]})):
            rawd = build_raw(tmp / f"g9_{len(ok)}", **kw)
            od = rawd.parent / "_demo_field_unused_t"
            ok.append((f"G9 台帳で同じ録音名が 2 行（{tag}）→ 2・置き場を作らない", run(rawd, lst, od, "--convert") == 2 and not od.exists()))
        # G10: events が無い → 空の表として許さず止める
        raw10 = build_raw(tmp / "g10", no_events=True)
        ok.append(("G10 events が無い → 2（空の表として許さない）", run(raw10, lst, raw10.parent / "_demo_field_unused_t", "--convert") == 2))
        ok.append(("G6 置き場の名前が違う → 2", run(raw, lst, tmp / "eval_out", "--convert") == 2 and not (tmp / "eval_out").exists()))
        ok.append(("G6 置き場が raw/ の中 → 2", run(raw, lst, raw / "_demo_field_unused_x", "--convert") == 2 and not (raw / "_demo_field_unused_x").exists()))
        ok.append(("守りだけ（--convert なし）は rc 0 で何も書かない", run(raw, lst, out) == 0 and not out.exists()))
        # C1・C2
        wlist(lst, OK_ROWS + [("20261022", "261022_028.WAV", "H2-48", "参考", "0", "4.0", "tune_field_clip82005")])
        rc = run(raw, lst, out, "--convert")
        idx = {r["原本"]: r for r in P.read_csv(out / "20261022" / "index.csv")} if rc == 0 else {}
        evd = {r["原本"]: r for r in P.read_csv(out / "20261022" / "events_demo.csv")} if rc == 0 else {}
        info = (out / "index.json").read_text(encoding="utf-8") if rc == 0 else ""
        ok.append(("C1 変換 rc 0・index 3 本・条件の写し元 = 同じ台本ID の評価の本", rc == 0 and len(idx) == 3
                   and idx["261022_009.WAV"]["条件の写し元"] == "261022_010.WAV" and idx["261022_071.WAV"]["条件の写し元"] == "261022_083.WAV"))
        e9 = evd.get("261022_009.WAV", {})
        ok.append(("C1 条件の行: 原本・動画を差し替え・拍手とラップ秒と条件ID は空・三脚の向きは写す",
                   e9.get("動画原本") == "IMG_5530.MOV" and e9.get("拍手_録音側秒") == "" and e9.get("ラップ秒") == "" and e9.get("条件ID") == ""
                   and e9.get("三脚の向き") == "180" and e9.get("take_id", "").startswith("9")))
        ok.append(("C1 index.json に台帳の sha・約束 v12_149・評価の本 0 の決まり",
                   P.sha(raw / "ledger" / "ledger_20261022.csv") in info and '"convention": "v12_149"' in info and "本番" in info))
        ok.append(("C1 評価の本の音は変換しない（conv に 010・083・029 が無い）",
                   rc == 0 and not any((out / "20261022" / "conv" / f"{s}_conv.flac").exists() for s in ("261022_010", "261022_083", "261022_029"))))
        e28 = evd.get("261022_028.WAV", {})
        ok.append(("C2 線の上書き 4.0（line_m と 横距離m）", e28.get("line_m") == "4.0" and e28.get("横距離m") == "4.0"))
        # C3: 条件の写し元の台本ID（実際の走り方が台本と違った本・本人 22:41）
        lst3 = tmp / "list_src.csv"
        wcsv(lst3, list(P.COLS) + ["条件の写し元の台本ID"], [dict(zip(P.COLS, ("20261022", "261022_028.WAV", "H2-48", "参考", "0", "", "tune_field_clip82005")), **{"条件の写し元の台本ID": "C2-04"})])
        out3 = tmp / "_demo_field_unused_c3"
        rc3 = run(raw, lst3, out3, "--convert")
        e3 = {r["原本"]: r for r in P.read_csv(out3 / "20261022" / "events_demo.csv")}.get("261022_028.WAV", {}) if rc3 == 0 else {}
        i3 = P.read_csv(out3 / "20261022" / "index.csv")[0] if rc3 == 0 else {}
        ok.append(("C3 条件の写し元の台本ID: 守りは台帳の台本ID（H2-48）で通し、条件は C2-04 の評価の本（010）から写す",
                   rc3 == 0 and e3.get("class") == "bike_bell" and e3.get("台本ID") == "H2-48" and i3.get("条件の写し元") == "261022_010.WAV"
                   and i3.get("条件の写し元の台本ID") == "C2-04"))
        ok.append(("既にある置き場は --overwrite なしで止める（rc 3）", run(raw, lst, out, "--convert") == 3))
        # M1: 除外区間メモ（案 A・本人 22:56）: 入れる（除外区間あり）と備考のある参考の本だけ・既にあれば上書きしない
        led_m = {"261022_009.WAV": {"動画": "IMG_5530.MOV"}, "261022_028.WAV": {"動画": "IMG_5547.MOV"}, "261022_011.WAV": {"動画": "IMG_5532.MOV"}}
        rows_m = [{"原本": "261022_009.WAV", "clip_id": "c1", "区分": "入れる（除外区間あり）", "読みの備考": "メモ"},
                  {"原本": "261022_028.WAV", "clip_id": "c2", "区分": "参考", "読みの備考": "最後まで通っていれば"},
                  {"原本": "261022_011.WAV", "clip_id": "c3", "区分": "参考", "読みの備考": ""}]
        mp = tmp / "memo" / "除外区間メモ.csv"
        n1 = P.write_memo(mp, rows_m, led_m)
        mr = P.read_csv(mp)
        mr[0]["除外の入り_音の秒"] = "6.2"
        wcsv(mp, list(P.MEMO_COLS), mr)
        n2 = P.write_memo(mp, rows_m, led_m)
        ok.append(("M1 除外区間メモ: 入れる（除外区間あり）と備考のある参考だけ（2 行）・既にあれば上書きしない（本人の値が残る）",
                   n1 == 2 and [r["原本"] for r in mr] == ["261022_009.WAV", "261022_028.WAV"] and n2 == -1
                   and P.read_csv(mp)[0]["除外の入り_音の秒"] == "6.2"))
        # N1: 読みの備考 → 読む表の 備考（歩く人の行も同じ原本なら同じ）
        tk = tmp / "takes_n.csv"
        wcsv(tk, ["take", "video", "cls", "line_m"], [{"take": "261022_081.WAV", "video": "a.MOV", "cls": "bike_bell", "line_m": "2.0"},
                                                     {"take": "261022_081.WAV", "video": "a.MOV", "cls": "mic", "line_m": "5"},
                                                     {"take": "261022_079.WAV", "video": "b.MOV", "cls": "bike_bell", "line_m": "2.0"}])
        nn = P.add_notes(tk, {"261022_081.WAV": "右を通った見込み", "261022_079.WAV": ""})
        rr = P.read_csv(tk)
        ok.append(("N1 読みの備考が読む表の 備考 に入る（歩く人の行も・備考の無い本は空）", nn == 2 and [r["備考"] for r in rr] == ["右を通った見込み", "右を通った見込み", ""]))
        # O1
        a180 = iv_az(out / "20261022" / "conv" / "261022_071_conv.flac") if rc == 0 else 999
        from step19_realsmoke_convert import convert_with_gain
        o0 = convert_with_gain(raw / "261022_071.WAV", tmp / "yaw0", 0.0, 0.0, 0.0, 0.0)
        a0 = iv_az(o0)
        ok.append((f"O1 yaw 180 で正面基準の 60° に戻る（{a180:.1f}°）・yaw 0 は −120° のまま（{a0:.1f}°）",
                   abs(wrap(a180 - 60.0)) <= 2.0 and abs(wrap(a0 + 120.0)) <= 2.0))
        # O2
        import step19k_field_perception as K
        src = {(8, 1): [(0.0, 10.0, 2.0, None), (10.0, -10.0, 2.0, None)]}
        s0, _ = K.gt_lines_for_clip(src, [], 0.0, 1.8, None, heading_deg=0.0, walk_heading=False)
        s180, _ = K.gt_lines_for_clip(src, [], 0.0, 1.8, None, heading_deg=180.0, walk_heading=False)
        az = lambda ls: [int(x.split(",")[3]) for x in ls]
        ok.append(("O2 静止: 三脚の向き 180 で正解の方位が 180° 回る（正解側でも直せる）", len(s0) == len(s180) > 0
                   and all(abs(wrap(b - a - 180)) <= 1 for a, b in zip(az(s0), az(s180)))))
        mic = [(0.0, -3.0, 0.0), (10.0, 3.0, 0.0)]
        w0, _ = K.gt_lines_for_clip(src, mic, 0.0, 1.8, None, heading_deg=0.0, walk_heading=True)
        w180, _ = K.gt_lines_for_clip(src, mic, 0.0, 1.8, None, heading_deg=180.0, walk_heading=True)
        ok.append(("O2 歩行: 三脚の向きを使わない（0 と 180 で正解が同じ）＝正解側では直せない → 音の側で直す", len(w0) > 0 and w0 == w180))
        # R1・R2（本物の台帳があれば・表だけ）
        real_list = E2E / "md" / "design" / "demo_field_unused_list_2026-10-26.csv"
        if (E2E / "raw" / "ledger" / "ledger_20261015.csv").is_file() and real_list.is_file():
            ok.append(("R1 本物の一覧が本物の台帳で守りを通る（評価の本 0）", P.main(["--list", str(real_list)]) == 0))
            rows = P.read_csv(real_list)
            bad = [dict(r) for r in rows]
            bad[0].update({"原本": "261015_049.WAV", "台本ID": "H-03"})              # 台帳で「本 H1 H-03」
            wcsv(tmp / "real_bad.csv", list(rows[0].keys()), bad)              # 任意の列（読みの備考）もそのまま
            ok.append(("R2 本物の一覧に評価の本を 1 本混ぜると 2", P.main(["--list", str(tmp / "real_bad.csv"), "--out", str(tmp / "_demo_field_unused_r"), "--convert"]) == 2
                       and not (tmp / "_demo_field_unused_r").exists()))
        else:
            print("（本物の台帳が無いので R1・R2 は飛ばした）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
