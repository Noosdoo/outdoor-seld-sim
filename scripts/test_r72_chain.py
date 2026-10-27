# -*- coding: utf-8 -*-
"""test_r72_chain.py — 長回しの相手の区間が step19d → step19b（切り出し）→ step19c → step20 を通して保たれるかの接続試験（Astra R72-02・R72-03 の受け入れ条件・2026-10-20）。

架空の 600 秒の長回し 1 本（区分 負例露出）に:
  - 通り過ぎる車（② 100 s・終わり空）→ 既定の窓 [最接近 − 7, 最接近 + 1] = [93, 101)
  - サイレン（② 300 s・終わり 325 s）→ [299, 326)
  - 踏切（② 500 s・終わり空 = 不明）→ [499, 録音末尾)
step19b の負例の分割（10 秒・重なり 1 秒）で 67 本に切り、各本の担当区間でタイル → step19c（--cut）が通る → step20 で:
  - 正味の露出 = 600 − 8 − 27 − 101 = 464 s（和集合・担当区間でタイル・二重に引かない）
  - 誤警告: 150 s の発火 = 1 件（区間の外）、310 s（サイレンの区間）と 550 s（不明の区間）は数えない
  - 不明として除外した秒 = 101 s・1 区間（別計上）
使い方: python scripts/test_r72_chain.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def main() -> int:
    ok = []
    D = load("s19d_chain", "step19d_field_csv_to_ann.py")
    B = load("s19b_chain", "step19b_realsmoke_cut.py")
    C = load("s19c_chain", "step19c_ann_validate.py")
    S = load("s20_chain", "step20_realsmoke_score_v3.py")
    LEN = 600.0
    # ---- step19d ----
    ses = [dict(session_id="n", 用途="最終評価", 区分="負例露出", マイク高さ_cm="200")]
    ev = [dict(session_id="n", take_id="1", event_id="1", 原本="261105_010.WAV", **{"class": "none"}, 象限="", ラップ秒="", 状態="歩行"),
          dict(session_id="n", take_id="1", event_id="2", 原本="261105_010.WAV", **{"class": "car_drive"}, 象限="L", ラップ秒="100", 終わり秒="", 状態="歩行", 横距離m="2.0"),
          dict(session_id="n", take_id="1", event_id="3", 原本="261105_010.WAV", **{"class": "siren"}, 象限="L", ラップ秒="300", 終わり秒="325", 状態="歩行"),
          dict(session_id="n", take_id="1", event_id="4", 原本="261105_010.WAV", **{"class": "crossing"}, 象限="L", ラップ秒="500", 終わり秒="", 状態="歩行")]
    rows_d, warns = D.convert(ses, ev, 6.0, 0.0, None, 3.0, None)
    none_row = [r for r in rows_d if r["class"] == "none"][0]; none_row["t_cpa"] = f"{LEN:.2f}"      # 原本が無いので長さをここで入れる
    byev = {r["event_id"]: r for r in rows_d}
    ok.append(("19d: 車は既定の窓（94〜100）・サイレンは [300, 325)・踏切は不明",
               byev["2"]["t_start"] == "94.00" and byev["2"]["t_cpa"] == "100.00" and byev["3"]["t_start"] == "300.00" and byev["3"]["t_end"] == "325.00" and byev["4"]["t_end_unknown"] == "1"))
    # ---- step19b（負例の分割）----
    plan = B.plan_negative_split(LEN, 10.0, 1.0)
    out_rows = []
    for i, (off, own0, own1) in enumerate(plan):
        clip = f"eval_field_clip{i + 1:05d}"
        nrow = B.negative_row(clip, "261105_010.WAV", off, own0, own1, none_row, orig_duration_s=LEN)
        nrow.setdefault("session_id", "n"); nrow["take_id"] = "n/01"; nrow["区分"] = "負例露出"; nrow["状態"] = "歩行"
        added = B.rebase_rows(rows_d, off, 10.0, "261105_010.WAV", clip, own=(own0, own1), orig_duration_s=LEN)
        out_rows.append(nrow); out_rows += added
    n_clips = len(plan)
    ok.append((f"19b: {n_clips} 本に分割・担当区間の合計 = 600 s", abs(sum(b - a for _, a, b in plan) - LEN) < 1e-6 and n_clips >= 60))
    with_unknown = [r for r in out_rows if r.get("t_end_unknown") == "1"]
    ok.append((f"19b: 不明の踏切は 500 s 以降の全クリップに残る（{len(with_unknown)} 本・t_start は負でもよい）",
               len(with_unknown) == sum(1 for off, _, _ in plan if off + 10.0 >= 500.0 - 1e-6) and any(float(r["t_start"]) < 0 for r in with_unknown)))
    sir = [r for r in out_rows if r["class"] == "siren"]
    ok.append((f"19b: サイレン [300,325) は交差する全クリップに残り、t_end もクリップの秒（{len(sir)} 本）", len(sir) == sum(1 for off, _, _ in plan if off <= 325.0 + 1e-6 and off + 10.0 >= 300.0 - 1e-6)
               and all(abs(float(r["t_end"]) - float(r["t_start"]) - 25.0) < 1e-6 for r in sir)))
    # ---- step19c（--cut）----
    C.errs.clear(); C.warns.clear()
    for r in out_rows:
        r.setdefault("trial", "neg"); r.setdefault("quadrant", r.get("象限", "")); r.setdefault("pair_id", "")
    C.validate(out_rows, True, 10.0, dict(C.PLAN_DEFAULT))
    s4 = [e for e in C.errs if e.startswith("S4")]; s7 = [e for e in C.errs if e.startswith("S7")]
    ok.append((f"19c --cut: S4（時刻の範囲）と S7（担当区間のタイル）に誤りなし（S4 {len(s4)}・S7 {len(s7)}）", not s4 and not s7))
    # ---- step20 ----
    cfg43 = S.V43.Cfg43(**json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
    pred = defaultdict(dict)
    for t_abs in (150.0, 310.0, 550.0):                       # サイレンの発火（警告・3 フレーム連続）を、その時刻を含む全クリップに置く
        for i, (off, _, _) in enumerate(plan):
            if off <= t_abs <= off + 10.0:
                k0 = int(round((t_abs - off) * 10))
                for k in range(k0, k0 + 6):
                    pred[f"eval_field_clip{i + 1:05d}"].setdefault(k, []).append((0, 30.0, 0.0, float("nan")))
    for r in out_rows:
        r.setdefault("scored", "1")
    events, neg, extras = S.evaluate(out_rows, dict(pred), cfg43, (2.5, 1.5), 7.5, True)
    g = neg["by"]["負例露出"]
    ok.append((f"20: 正味の露出 464 s（{g['exposure_s']:.1f}）・誤警告 1 件（{neg['n_false']}）・不明として除外 101 s（{g['unknown_s']:.1f}）",
               abs(g["exposure_s"] - 464.0) < 0.5 and neg["n_false"] == 1 and abs(g["unknown_s"] - 101.0) < 0.5))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
