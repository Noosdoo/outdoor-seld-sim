# -*- coding: utf-8 -*-
"""本体 §2-6・§9-1 の 1・§9-4 の 1（2026-10-26）の受け入れ試験: 実録の採点器 step20 v3 の --warmup-v30（暖機 4.0 s・規則は frame 40 から・採点上の鳴り始め）。
架空の注釈と予測だけを使う（実録・本物の読みは使わない）。合成の統一採点器 --events の試験（test_score_unified_v30.py の E2・E4）と同じ場面で同じ結果になることを確かめる。

  W1 連続警告（サイレン 0〜10 s・予測も 0〜99 枠）: v30 は残り（鳴り終わり 10 s）・採点上の鳴り始め 4.0・最初の発火 4.3 s（規則は frame 40 から）→ 成功・遅れ 0.3
     （合成 E4 と同じ 1/1）。旧（規則を frame 0 から）は最初の発火が 0.3 s
  W2 通過の基準 3.9・4.0・4.1 s: 最接近 3.9 は除外（warmup_cpa）・4.0 と 4.1 は残る（合成 E2 と同じ）
  W3 4 秒前から続く警告（3.0〜9.0 s）は残り採点上の鳴り始め 4.0・4 秒直前に終わる警告（1.0〜3.9 s）と鳴り終わりちょうど 4.0 s は除外・4.1 s は残る（鳴り始め 4.0）（合成 E2 と同じ）
  W4 鳴り始め 5.0 のサイレン: 予測が 9.5 s から → 窓 [4.0, 9.0] の外で未通知・5.5 s から → 成功・遅れ 0.8（合成 E4 と同じ）。物理の鳴り始め t_start は 5.0 のまま
  W5 発進（B2）: 最接近 6.0・動き出し 3.0 → 残り、主の列 = 最接近 − 発火（6.0 − 4.8 = 1.2）・delay_move = 4.8 − max(3.0, 4.0) = 0.8／最接近 3.5 は除外／
     動き出しの読みが無い B2 → 止まる
  W6 4 秒前に発火済みの通知: 予測が frame 0〜39 だけの通過（最接近 6.0）→ v30 は発火なし＝未通知（旧は 3.9 s に発火して成功）
  W7 命令: --warmup-v30 → 報告の頭に暖機の除外の数・events に t_score_on／warmup_reason／delay_move の列・--min-history 7.5 と一緒は止まる
  W9 波形で読んだ鳴り終わり t_warn_end（Astra R142-05）: 暖機より前に終わる・またぐ・後の音の 3 つ・無ければ t_cpa・負例のマスクは変わらない
使い方: python scripts/test_step20_warmup_v30.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location("s20v3_w30", ROOT / "scripts/step20_realsmoke_score_v3.py")
S = importlib.util.module_from_spec(spec); sys.modules["s20v3_w30"] = S; spec.loader.exec_module(S)

COLS = ["clip_id", "event_id", "take_id", "pair_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "区分", "状態", "横距離m", "n_car", "scored", "台本ID", "t_move_s"]
CAR, HORN, SIREN = S.CLS_IDX["car_drive"], S.CLS_IDX["horn"], S.CLS_IDX["siren"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def row(clip, cls, t_start, t_cpa, t_end="", lat="1.0", script="", t_move=""):
    return dict(zip(COLS, [clip, "1", "T", "", "A", cls, "L", str(t_start), str(t_cpa), str(t_end), "A", "静止", lat, "1", "1", script, str(t_move)]))


def car_frames(t_cpa, lat=1.0, v=5.0, k0=0, k1=100):
    fr = {}
    for k in range(k0, k1):
        t = (k + 1) / 10.0
        x = v * (t - t_cpa)
        fr[k] = [(CAR, math.degrees(math.atan2(lat, -x)), 0.0, math.hypot(x, lat))]
    return fr


def warn_frames(ci, k0, k1=100):
    return {k: [(ci, 90.0, 0.0, 3.0)] for k in range(k0, k1)}


def cases():
    rows, pred = [], {}

    def add(clip, r, fr):
        rows.append(r); pred[clip] = fr
    add("w1", row("w1", "siren", 0.0, 3.0, 10.0, lat=""), warn_frames(SIREN, 0))
    for c, t in (("p39", 3.9), ("p40", 4.0), ("p41", 4.1)):
        add(c, row(c, "car_drive", 1.0, t), car_frames(t))
    add("w3a", row("w3a", "siren", 3.0, 6.0, 9.0, lat=""), warn_frames(SIREN, 30, 90))
    add("w3b", row("w3b", "siren", 1.0, 4.0, 3.9, lat=""), warn_frames(SIREN, 10, 39))
    add("w3c", row("w3c", "horn", 2.0, 5.0, 4.0, lat=""), warn_frames(HORN, 20, 40))
    add("w3d", row("w3d", "horn", 2.0, 5.0, 4.1, lat=""), warn_frames(HORN, 20, 41))
    add("w4a", row("w4a", "siren", 5.0, 8.0, 10.0, lat=""), warn_frames(SIREN, 95))
    add("w4b", row("w4b", "siren", 5.0, 8.0, 10.0, lat=""), warn_frames(SIREN, 55))
    add("b2a", row("b2a", "car_drive", 2.0, 6.0, script="B2-01", t_move=3.0), car_frames(6.0))
    add("b2b", row("b2b", "car_drive", 1.0, 3.5, script="B2-02", t_move=1.0), car_frames(3.5))
    add("pre", row("pre", "car_drive", 2.0, 6.0), car_frames(6.0, k1=40))
    return rows, pred


def run_eval(rows, pred, v30: bool, min_history: float):
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    S.WARMUP_V30 = v30
    try:
        events, _, _ = S.evaluate(rows, pred, cfg43, (2.5, 1.5), min_history, True)
    finally:
        S.WARMUP_V30 = False
    return {e["clip"]: e for e in events}


def run_eval_neg(rows, pred):
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    S.WARMUP_V30 = True
    try:
        _, neg, _ = S.evaluate(rows, pred, cfg43, (2.5, 1.5), 4.0, True)
    finally:
        S.WARMUP_V30 = False
    return (neg["n_false"], round(neg["exposure_s"], 3))


def main() -> int:
    rows, pred = cases()
    E = run_eval(rows, pred, True, 4.0)
    O = run_eval(rows, pred, False, 0.0)
    w1, o1 = E["w1"], O["w1"]
    check("W1 連続警告 0〜10 s: v30 は残り・採点上の鳴り始め 4.0・成功・遅れ 0.3（最初の発火 4.3 s）＝合成 E4 の 1/1・旧は発火 0.3 s",
          w1["history_short"] == 0 and w1["t_score_on"] == 4.0 and w1["notified"] is True and w1["delay"] == 0.3 and o1["delay"] == 0.3 and "t_score_on" not in o1,
          str({k: w1.get(k) for k in ("history_short", "t_score_on", "notified", "delay")}))
    hs = {c: (E[c]["history_short"], E[c]["warmup_reason"]) for c in ("p39", "p40", "p41")}
    check("W2 通過の基準 3.9・4.0・4.1 s: 3.9 は除外（warmup_cpa）・4.0 と 4.1 は残る（合成 E2 と同じ）",
          hs == {"p39": (1, "warmup_cpa"), "p40": (0, ""), "p41": (0, "")}, str(hs))
    w3 = {c: (E[c]["history_short"], E[c]["warmup_reason"], E[c].get("t_score_on")) for c in ("w3a", "w3b", "w3c", "w3d")}
    check("W3 4 秒前から続く警告は残り鳴り始め 4.0・3.9 と 4.0 に鳴り終わる警告は除外（warmup_end）・4.1 は残り鳴り始め 4.0（合成 E2 と同じ）",
          w3 == {"w3a": (0, "", 4.0), "w3b": (1, "warmup_end", 4.0), "w3c": (1, "warmup_end", 4.0), "w3d": (0, "", 4.0)}, str(w3))
    a, b = E["w4a"], E["w4b"]
    check("W4 鳴り始め 5.0: 予測 9.5 s から → 窓 [4.0, 9.0] の外で未通知・5.5 s から → 成功・遅れ 0.8（合成 E4 と同じ）・物理の t_start は 5.0 のまま",
          a["notified"] is False and b["notified"] is True and b["delay"] == 0.8 and a["t_score_on"] == 5.0 and next(r for r in rows if r["clip_id"] == "w4b")["t_start"] == "5.0",
          str((a["notified"], b["notified"], b["delay"])))
    b2a, b2b = E["b2a"], E["b2b"]
    bad = [dict(r) for r in rows if r["clip_id"] == "b2a"]
    bad[0]["t_move_s"] = ""
    try:
        run_eval(bad, pred, True, 4.0); stopped = False
    except SystemExit:
        stopped = True
    check("W5 発進（B2）: 残り・主の列 = 最接近 − 発火 1.2・delay_move = 発火 − max(動き出し, 4.0) = 0.8／最接近 3.5 は除外／動き出しの読みなし → 止まる",
          b2a["history_short"] == 0 and b2a["notified"] is True and b2a["lead"] == 1.2 and b2a["delay_move"] == 0.8 and b2a["t_score_on"] == 4.0
          and b2b["history_short"] == 1 and b2b["warmup_reason"] == "warmup_cpa" and stopped, str({k: b2a.get(k) for k in ("lead", "delay_move", "t_score_on")}))
    check("W6 4 秒前に発火済みの通知（予測が frame 0〜39 だけ）: v30 は未通知・旧は成功",
          E["pre"]["notified"] is False and O["pre"]["notified"] is True, str((E["pre"]["notified"], O["pre"]["notified"])))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with open(td / "ann.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        with open(td / "pred.csv", "w", encoding="utf-8") as f:
            for clip, fr in pred.items():
                for k, lst in sorted(fr.items()):
                    for c, az, el, d in lst:
                        f.write(f"{clip},{k},{c},0,{az},{el},{d}\n")
        def run(*extra):
            return subprocess.run([sys.executable, str(ROOT / "scripts/step20_realsmoke_score_v3.py"), "--pred", str(td / "pred.csv"), "--ann", str(td / "ann.csv"),
                                   "--out", str(td / "s.md"), *extra], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=dict(os.environ, PYTHONUTF8="1"))
        r1 = run("--warmup-v30")
        md = (td / "s.md").read_text(encoding="utf-8") if r1.returncode == 0 else ""
        hdr = list(csv.reader(open(td / "s_events.csv", encoding="utf-8")))[0] if r1.returncode == 0 else []
        r2 = run("--warmup-v30", "--min-history", "7.5")
        check("W7 命令: 報告の頭に暖機の除外の数（通過・発進 2 件・警告音 2 件）・events に t_score_on／warmup_reason／delay_move・--min-history 7.5 と一緒は止まる",
              r1.returncode == 0 and "暖機（v30" in md and "最接近 < 4.0 s **2 件**" in md and "鳴り終わり ≤ 4.0 s **2 件**" in md
              and all(c in hdr for c in ("t_score_on", "warmup_reason", "delay_move")) and r2.returncode != 0,
              (r1.stderr[-300:] if r1.returncode else md.split("\n")[5][:200]))
    # W8 切り出しの確かめ（L3・step19b_check_history・架空の注釈）
    spec_h = importlib.util.spec_from_file_location("s19b_hist", ROOT / "scripts/step19b_check_history.py")
    H = importlib.util.module_from_spec(spec_h); sys.modules["s19b_hist"] = H; spec_h.loader.exec_module(H)
    ann = [row("c1", "car_drive", 5.0, 8.0), row("c2", "car_drive", 3.5, 6.5), row("c3", "car_drive", 0.0, 3.0), row("c4", "siren", 2.0, 5.0, 3.5, lat=""),
           row("c5", "horn", 5.0, 8.0, lat=""), dict(row("c6", "none", 0.0, 10.0), scored="1"), dict(row("c7", "car_drive", 1.0, 3.0), scored="0")]
    hres = {x["clip_id"]: (x["short_margin"], x["clamped"], x["warmup_reason"]) for x in H.check_rows(ann, "t.csv")}
    check("W8 切り出しの確かめ: 事象の前 5.0 は足りる・3.5 は足りない・0.00 はクリップより前（最接近 3.0 で warmup_cpa）・警告音 2.0〜3.5 は足りず warmup_end・負例と scored=0 は数えない",
          hres == {"c1": (0, 0, ""), "c2": (1, 0, ""), "c3": (1, 1, "warmup_cpa"), "c4": (1, 0, "warmup_end"), "c5": (0, 0, "")}, str(hres))
    # W9（Astra R142-05・10/27）: 波形で読んだ鳴り終わり t_warn_end（field_score_members が区間の表から入れる）を暖機の判定に使う。
    #   暖機より前に終わる（3.0〜3.4）→ 除外・暖機をまたぐ（3.0〜4.6）→ 残り鳴り始め 4.0・暖機より後（5.0〜5.6）→ 残り鳴り始め 5.0。
    #   t_warn_end が無ければ従来どおり t_cpa（鳴り始め + 3 = 6.0）で 3.0〜 は残る。負例の担当区間のマスク（t_end・t_cpa）は t_warn_end で変わらない
    def wrow(clip, t0, we):
        return dict(row(clip, "horn", t0, t0 + 3.0, lat=""), t_warn_end=str(we))
    rows9 = [wrow("e1", 3.0, 3.4), wrow("e2", 3.0, 4.6), wrow("e3", 5.0, 5.6), row("e4", "horn", 3.0, 6.0, lat="")]
    pred9 = {"e1": warn_frames(HORN, 30, 34), "e2": warn_frames(HORN, 30, 46), "e3": warn_frames(HORN, 50, 56), "e4": warn_frames(HORN, 30, 34)}
    E9 = run_eval(rows9, pred9, True, 4.0)
    w9 = {c: (E9[c]["history_short"], E9[c]["warmup_reason"], E9[c].get("t_score_on")) for c in ("e1", "e2", "e3", "e4")}
    neg = lambda rs: run_eval_neg(rs, {"m": warn_frames(HORN, 30, 100)})
    m_with = [dict(row("m", "horn", 3.0, 6.0, lat=""), t_warn_end="3.4"), dict(row("m", "none", 0.0, 10.0, lat=""), event_id="2")]
    m_wo = [dict(m_with[0], t_warn_end=""), m_with[1]]
    nw, nwo = neg(m_with), neg(m_wo)
    check("W9 t_warn_end: 暖機より前に終わる 3.0〜3.4 は除外（warmup_end）・またぐ 3.0〜4.6 は残り鳴り始め 4.0・後の 5.0〜5.6 は残り鳴り始め 5.0・無ければ t_cpa 6.0 で残る・"
          "負例のマスクは変わらない（誤警告の数が同じ）",
          w9 == {"e1": (1, "warmup_end", 4.0), "e2": (0, "", 4.0), "e3": (0, "", 5.0), "e4": (0, "", 4.0)} and nw == nwo, str((w9, nw, nwo)))
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
