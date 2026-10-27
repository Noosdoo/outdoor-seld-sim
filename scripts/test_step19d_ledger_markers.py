# -*- coding: utf-8 -*-
"""test_step19d_ledger_markers.py — step19d の 2026-10-20 の追加（実録の採点の準備 B1・B5・B8）の検査。

  L1 採用の台帳（--ledger）: 役割が「本 …」の原本だけ採用・「使わない …」「校正」「点検」は理由つきで除外・台帳に無い原本は「台帳に無い」
  M1 目印の点（R97-05）: 台本ID C2-／H2-・W01〜W08 と A2（#69 r38・2026-10-25）は ±12 の 13 点、B2／E2・W09〜W16・台本ID 無し（10/15）は ±10 の 11 点
  M2 歩行の本の mic の行も同じ目印（takes CSV に markers 列）
  T1 長回し（区分 G／負例露出）の相手の 終わり秒 → t_start = 始まり・t_cpa = 始まり・t_end = 終わり（#69 r18）
  T2 長回し以外の本の 終わり秒 は無視して警告
  T4 発進 B2（#69 r23）: --when-csv の t_first_s（動き出し）があれば t_start = max(t_cpa − 6, 動き出し)・t_move_s に記録。無ければ警告
使い方: python scripts/test_step19d_ledger_markers.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s19d_t", ROOT / "scripts/step19d_field_csv_to_ann.py")
D = importlib.util.module_from_spec(spec); sys.modules["s19d_t"] = D; spec.loader.exec_module(D)


def main() -> int:
    ok = []
    # --- L1 台帳 ---
    tmp = Path(tempfile.mkdtemp(prefix="s19d_"))
    led = tmp / "ledger.csv"
    with open(led, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["録音", "開始", "秒", "動画", "時刻が重なる動画", "役割"])
        w.writerow(["261015_006.WAV", "10:20:00", "30.0", "IMG_1.MOV", "", "本 C1 C-01"])
        w.writerow(["261015_007.WAV", "10:21:00", "30.0", "IMG_2.MOV", "", "使わない C1: 録り直し（C-05）・4 m の線で録った"])
        w.writerow(["261015_001.WAV", "10:10:00", "64.6", "", "", "校正"])
    roles = D.load_ledger(led)
    ev = [dict(session_id="s", take_id="1", 原本="261015_006.WAV", **{"class": "bike_bell"}),
          dict(session_id="s", take_id="2", 原本="261015_007.WAV", **{"class": "bike_bell"}),
          dict(session_id="s", take_id="3", 原本="261015_099.WAV", **{"class": "bike_bell"}),
          dict(session_id="s", take_id="", 原本="", **{"class": ""})]
    kept, dropped = D.apply_ledger(ev, roles)
    ok.append(("L1 台帳: 本だけ採用（2 行: 本 1 ＋ 原本なしの行）・使わない 1・台帳に無い 1",
               [r["take_id"] for r in kept] == ["1", ""] and [r["台帳の役割"] for r in dropped] == ["使わない C1: 録り直し（C-05）・4 m の線で録った", "台帳に無い"]))
    # --- M1 目印 ---
    m = D.markers_for_take
    ok.append(("M1 目印: C2/H2/W01〜08/A2（#69 r38）は ±12 の 13 点・B2/E2/W09〜16/無しは ±10 の 11 点",
               m({"台本ID": "C2-03"}) == "12,10,8,6,4,2,0,-2,-4,-6,-8,-10,-12" and m({"台本ID": "H2-60"}).count(",") == 12
               and m({"台本ID": "W08"}).count(",") == 12 and m({"台本ID": "W09"}) == "10,8,6,4,2,0,-2,-4,-6,-8,-10"
               and m({"台本ID": "A2-01"}).count(",") == 12 and m({"台本ID": "A2-32"}).count(",") == 12
               and m({"台本ID": "B2-01"}).count(",") == 10 and m({"台本ID": "E2-05"}).count(",") == 10 and m({"台本ID": "W16"}).count(",") == 10 and m({}) .count(",") == 10))
    # --- M1' 並び（#69 r33・2026-10-23）: 出発側から。後→前 = −n … +n・前→後 = +n … −n・三脚の向きは見ない ---
    ok.append(("M1' 目印の並び: 後→前 は −12 から・前→後 は +12 から・三脚 180° でも裏返さない・mic に向きを渡せる",
               m({"台本ID": "C2-03", "direction": "後→前"}) == "-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12"
               and m({"台本ID": "H2-58", "direction": "前→後", "三脚の向き": "180"}) == "12,10,8,6,4,2,0,-2,-4,-6,-8,-10,-12"
               and m({"台本ID": "A2-01", "direction": "後→前", "三脚の向き": "180"}) == "-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12"          # A2 は 13 点（#69 r38）
               and m({"台本ID": "A2-02", "direction": "前→後"}) == "12,10,8,6,4,2,0,-2,-4,-6,-8,-10,-12"                               # Astra CAR24-06: A2 の両方向
               and m({"台本ID": "B2-01", "direction": "後→前", "三脚の向き": "180"}) == "-10,-8,-6,-4,-2,0,2,4,6,8,10"
               and m({"台本ID": "W05"}, "後→前") == "-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12" and m({"台本ID": "W05", "direction": "後->前"}).startswith("-12,")))
    # --- M2 takes CSV ---
    rows = [dict(session_id="s", take_id="1", event_id="1", 原本="261022_010.WAV", **{"class": "bicycle"}, 象限="B", ラップ秒="8", 状態="歩行",
                 動画原本="a.mov", direction="後→前", line_m="2.0", 台本ID="W05", **{"三脚の向き": "0"})]
    vt = D.video_takes(rows)
    ok.append(("M2 歩行の本: 相手の行と mic の行の markers が同じ ±12（後→前 = −12 から）・mic の向きは events の direction",
               len(vt) == 2 and all(r.get("markers") == "-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12" for r in vt) and "markers" in D.VIDEO_COLS
               and all(r.get("direction") == "後→前" for r in vt)))
    rows180 = [dict(rows[0], direction="前→後", **{"三脚の向き": "180"})]
    vt180 = D.video_takes(rows180)
    ok.append(("M2' 三脚 180° の歩行の本: direction（前→後）のまま +12 から・三脚で裏返さない（r33）",
               len(vt180) == 2 and all(r.get("markers") == "12,10,8,6,4,2,0,-2,-4,-6,-8,-10,-12" and r.get("direction") == "前→後" for r in vt180)))
    # --- T1 / T2 終わり秒 ---
    sessions = [dict(session_id="g", 用途="最終評価", 区分="G", マイク高さ_cm="200"), dict(session_id="a", 用途="最終評価", 区分="A", マイク高さ_cm="200")]
    events = [dict(session_id="g", take_id="1", event_id="1", 原本="261105_001.WAV", **{"class": "none"}, 象限="", ラップ秒="", 状態="静止"),
              dict(session_id="g", take_id="1", event_id="2", 原本="261105_001.WAV", **{"class": "car_drive"}, 象限="L", ラップ秒="300", 終わり秒="305", 状態="静止", 横距離m="2.0"),
              dict(session_id="a", take_id="2", event_id="1", 原本="261105_002.WAV", **{"class": "car_drive"}, 象限="L", ラップ秒="20", 終わり秒="25", 状態="静止", 横距離m="2.0")]
    out, warns = D.convert(sessions, events, 6.0, 0.0, None, 3.0, None)
    g = [r for r in out if r["session_id"] == "g" and r["class"] == "car_drive"][0]
    a = [r for r in out if r["session_id"] == "a"][0]
    ok.append(("T1 長回しの相手: t_start 300・t_cpa 300・t_end 305・規則の文", g["t_start"] == "300.00" and g["t_cpa"] == "300.00" and g["t_end"] == "305.00" and "長回しの相手" in g["t_start_rule"]))
    ok.append(("T2 長回し以外の 終わり秒 は無視（t_end 空・t_cpa 20・t_start 14）＋警告", a["t_end"] == "" and a["t_cpa"] == "20.00" and a["t_start"] == "14.00"
               and any("終わり秒 は長回し" in w for w in warns)))
    ok.append(("T3 OUT_COLS に t_end（t_cpa の次）", D.OUT_COLS[D.OUT_COLS.index("t_cpa") + 1] == "t_end"))
    # --- T4 発進 ---
    ses_b = [dict(session_id="b", 用途="最終評価", 区分="B", マイク高さ_cm="200")]
    ev_b = [dict(session_id="b", take_id="1", event_id="1", 原本="261029_050.WAV", **{"class": "car_drive"}, 象限="B", ラップ秒="", 状態="静止", 横距離m="2.0", direction="後→前", **{"台本ID": "B2-01"}),
            dict(session_id="b", take_id="2", event_id="1", 原本="261029_051.WAV", **{"class": "car_drive"}, 象限="B", ラップ秒="30", 状態="静止", 横距離m="2.0", direction="後→前", **{"台本ID": "B2-02"}),
            dict(session_id="b", take_id="3", event_id="1", 原本="261029_052.WAV", **{"class": "car_drive"}, 象限="B", ラップ秒="", 状態="静止", 横距離m="2.0", direction="後→前", **{"台本ID": "B2-03"})]
    when = {("261029_050", "1"): dict(status="ok", t_cpa_s="20.000", t_first_s="16.000", x_first_m="-10", sigma="1", L_mid_m="1.3", dt_est_s="0.05", mic="静止（x = 0）", t_in10_s="17.250", t_out10_s="22.750"),
            ("261029_052", "1"): dict(status="ok", t_cpa_s="20.000", t_first_s="16.000", x_first_m="-8", sigma="1", L_mid_m="1.3", dt_est_s="0.05", mic="静止（x = 0）")}   # 最初の読みが出発点でない（R72-04）
    out_b, warns_b = D.convert(ses_b, ev_b, 6.0, 0.0, None, 3.0, when)
    b1 = [r for r in out_b if r["take_id"] == "b/01"][0]; b2 = [r for r in out_b if r["take_id"] == "b/02"][0]
    ok.append(("T5 ±10 の時刻（#69 r31 ①）: --when-csv の t_in10_s／t_out10_s を t_in10／t_out10 に写す・無い本は空・OUT_COLS にある",
               b1["t_in10"] == "17.250" and b1["t_out10"] == "22.750" and b2["t_in10"] == "" and "t_in10" in D.OUT_COLS and "t_out10" in D.OUT_COLS))
    ok.append(("T4 発進: t_start = max(20 − 6, 16) = 16.00・t_move_s 16.00・規則の文", b1["t_start"] == "16.00" and b1["t_cpa"] == "20.00" and b1["t_move_s"] == "16.00" and "発進: 始まり" in b1["t_start_rule"]))
    ok.append(("T4' 動き出しの読みが無い発進は代替（t_start 24.00・t_move_src に理由）＋警告", b2["t_start"] == "24.00" and b2["t_move_s"] == "" and b2["t_move_src"].startswith("代替") and any("動き出しが取れない" in w for w in warns_b)))
    b3 = [r for r in out_b if r["take_id"] == "b/03"][0]
    ok.append(("T4'' 最初の読みが出発点（−10）でない発進は代替（R72-04・x_first −8）", b3["t_move_s"] == "" and "出発点" in b3["t_move_src"] and b3["t_start"] == "14.00"))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
