# -*- coding: utf-8 -*-
"""ckpt 選択器の受け入れ試験（2026-09-14）。仕様例 22 件（scripts/audit_v2_selector_spec_20260911.py・Astra）を本番実装に移植し、
厳格 parser・来歴欠損での停止・原子的出力を足す。

使い方: python scripts/test_v2_ckpt_selector.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("v2sel", ROOT / "scripts/_v2_ckpt_selector.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)) if detail and not cond else ''}")


def R(k=40, c=4, t=0, a=0.0, d=1.0, el=0.0):
    return (k, c, t, float(a), float(el), float(d))


def M(gt, pred):
    """仕様例の measure 相当（1 clip）: E, all_E, N, D, C, coverage, capture"""
    return S.summarize([S.measure(gt, pred)])


def metrics_from(good):
    return {e: dict(good) for e in S.EPOCHS}


def sel(metrics):
    return S.rank(metrics)[1]


def main() -> int:
    # ---- 仕様例 22 件（移植） ----
    g = [R(d=1)] + [R(k=41 + i, d=10) for i in range(4)]
    a = M(g, [R(d=1.5)] + [R(k=41 + i, d=10) for i in range(4)])
    b = M(g, [R(d=1.1)] + [R(k=41 + i, d=15) for i in range(4)])
    check("1 全帯最良と至近最良の不一致（全帯は A・選定指標は B）", a["all_E"] < b["all_E"] and a["E"] > b["E"], (a, b))
    z = M([R()], [])
    check("2 未検出でも分母に残る（E は None・C は 0）", z["N"] == 1 and z["D"] == 0 and z["E"] is None)
    check("3 至近帯は GT 距離で選ぶ（予測 5 m/GT 1 m は対象・逆は非対象）", M([R(d=1)], [R(d=5)])["E"] == 4 and M([R(d=5)], [R(d=1)])["E"] is None)
    check("4 1.5 m の端は含む", M([R(d=1.5)], [R(d=1.5)])["C"] == 1)
    check("5 1.500001 m は外", M([R(d=1.500001)], [R(d=1)])["N"] == 0)
    check("6 frame 39 は外・40 は内", M([R(k=39), R(k=40)], [R(k=39, d=99), R(k=40)])["E"] == 0)
    check("7 同クラス・同 frame だけ対応", M([R()], [R(c=6), R(k=41)])["D"] == 0)
    check("8 方位差 20 度は含む・20.00001 は外", M([R()], [R(a=20)])["D"] == 1 and M([R()], [R(a=20.00001)])["D"] == 0)
    check("9 方位の折り返し（+179 と −179 は 2 度）", M([R(a=179)], [R(a=-179)])["D"] == 1)
    check("10 分母の下限 0.1", M([R(d=.05)], [R(d=.15)])["E"] < 1.0000001)
    check("11 予測 1 つで GT 2 つを満たさない", M([R(t=0), R(t=1, a=1)], [R()])["D"] == 1)
    check("12 全 GT の対応を先に作る（遠い GT が予測を占有）", M([R(t=0, a=0, d=3), R(t=1, a=5, d=1)], [R(a=0, d=1)])["D"] == 0)
    tiegt = [R(t=0, a=-5, d=1), R(t=1, a=5, d=1.4)]
    tiepred = [R(t=1, a=0, d=2), R(t=0, a=0, d=1)]
    check("13 等角度 tie は行順に依らない", M(tiegt, tiepred) == M(tiegt[::-1], tiepred[::-1]))
    check("14 距離 NaN の予測は対応不可", M([R()], [R(d=float("nan"))])["D"] == 0)
    good = M([R()], [R()])
    metrics = metrics_from(good)
    check("15 完全同率なら後の epoch（149）", sel(metrics) == 149)
    metrics[139]["C"] = 0; metrics[149]["C"] = 0
    check("16 二次は全 GT 分母の捕捉数（→ 119）", sel(metrics) == 119)
    for e in S.EPOCHS:
        metrics[e]["E"] = 0.1
    metrics[19]["E"] = 0.1 - 1e-12
    check("17 順位は丸め前（1e-12 の差で 19）", sel(metrics) == 19)
    for name, mm in (("18 候補欠損で停止", {e: m for e, m in metrics.items() if e != 149}),
                     ("19 未照合で停止", {**metrics, 149: {**metrics[149], "verified": False}})):
        try:
            S.rank(mm); ok = False
        except S.InputError:
            ok = True
        check(name, ok)
    check("20 全候補が未検出なら選定なし", S.rank({e: dict(z) for e in S.EPOCHS})[1] is None)
    sparse = M([R(k=40), R(k=41)], [R(k=40)])
    dense = M([R(k=40), R(k=41)], [R(k=40, d=1.1), R(k=41, d=1.1)])
    check("21 条件付き指標の弱点（一部だけ完全検出が勝ち得る・見える化）", sparse["E"] < dense["E"] and sparse["coverage"] < dense["coverage"])
    try:
        S.measure([R()], [R(d=-1)]); ok = False
    except Exception:
        ok = False
    # 負値の拒否は parser 側（_row）の責務: 下の 24 で確認。measure は正規化済み入力を前提にする
    check("22 偶数個の中央値は中央 2 値の平均", M([R(t=0, a=0, d=1), R(t=1, a=30, d=1)], [R(t=0, a=0, d=1.2), R(t=1, a=30, d=1.6)])["E"] == (0.2 + 0.6) / 2 or
          abs(M([R(t=0, a=0, d=1), R(t=1, a=30, d=1)], [R(t=0, a=0, d=1.2), R(t=1, a=30, d=1.6)])["E"] - 0.4) < 1e-12)

    # ---- 検出率の門（任意・仕様 §3）----
    gm = metrics_from(good)                      # 全候補 N=1・D=1・C=1・E=0
    for e in S.EPOCHS:
        gm[e] = dict(gm[e], N=100, D=100, C=80, E=0.20)
    gm[139] = dict(gm[139], D=90, C=70, E=0.10)   # 検出率 90%・誤差は最小
    check("36 門なし（report_only）なら検出を落とした候補（e139）が誤差最小で勝つ", S.rank({e: dict(m) for e, m in gm.items()})[1] == 139)
    g_order, g_sel, g_rule = S.rank({e: dict(m) for e, m in gm.items()}, coverage_gate_pt=2.0)
    check("37 門 2pt なら e139（最良 100% − 10pt）は外れ、残りの同率から後の epoch（149）", g_sel == 149 and 139 not in g_order and "coverage gate" in g_rule, (g_sel, g_order))
    gm2 = {e: dict(m) for e, m in gm.items()}; gm2[139]["D"] = 99
    check("38 門の内側（99% ≥ 100% − 2pt）なら外れない", S.rank(gm2, coverage_gate_pt=2.0)[1] == 139)
    gm3 = {e: dict(m) for e, m in gm.items()}
    for e in S.EPOCHS:
        gm3[e]["D"] = 50
    gm3[19]["D"] = 60
    check("39 最良候補は門で外れない（自分が基準）", 19 in S.rank(gm3, coverage_gate_pt=2.0)[0])

    # ---- 厳格 parser・停止・原子的出力 ----
    tmp = Path(tempfile.mkdtemp(prefix="v2sel_"))
    try:
        man = tmp / "manifest.csv"
        man.write_text("clip_id,split\nc1,fold2\nc2,fold2\n", encoding="utf-8")
        clips = S.read_manifest(man)
        check("23 manifest を読む", clips == ["c1", "c2"])
        for i, (name, text, expect) in enumerate([
            ("24 負の距離を拒否", "c1,40,4,0,0,0,-1\n", "負か無限大"),
            ("25 6 列の行を拒否", "c1,40,4,0,0,1\n", "7 列でない"),
            ("26 manifest に無い clip を拒否", "zz,40,4,0,0,0,1\n", "manifest に無い"),
            ("27 inf を拒否", "c1,40,4,0,0,0,inf\n", "負か無限大"),
        ]):
            p = tmp / f"p{i}.csv"
            p.write_text(text, encoding="utf-8")
            try:
                S.read_pred(p, clips); ok = False; msg = "受け入れた"
            except S.InputError as e:
                ok = expect in str(e); msg = str(e)
            check(name, ok, msg)
        p = tmp / "ok.csv"
        p.write_text("c1,40,4,0,0,0,1.0\nc1,40,4,0,30,0,2.0\nc1,40,4,0,0,0,1.0\nc1,40,0,0,10,0,nan\nc1,41,4,0,0,0,nan\n", encoding="utf-8")
        pred, counts = S.read_pred(p, clips)
        check("28 同じ (frame, class, track=0) の別予測は受け入れる（track は識別子でない）・値まで同じ行は残して数える",
              len(pred["c1"]) == 5 and counts["n_dup_rows"] == 1, (len(pred.get("c1", [])), counts))
        check("29 距離クラスの NaN は件数として数える（警告クラスの NaN は数えない）", counts["n_nan_dist"] == 1)
        (tmp / "empty.csv").write_text("", encoding="utf-8")
        check("30 空 csv は読める（無検出かどうかは上流の完了記録で担保）", S.read_pred(tmp / "empty.csv", clips) == ({}, dict(n_nan_dist=0, n_dup_rows=0)))
        gtd = tmp / "gt"; gtd.mkdir()
        (gtd / "c1.csv").write_text("40,4,0,0,0,1.0\n", encoding="utf-8")
        try:
            S.read_gt(gtd, clips); ok = False
        except S.InputError as e:
            ok = "GT が無い" in str(e)
        check("31 GT の欠けは入力エラー", ok)
        (gtd / "c2.csv").write_text("40,4,0,0,0,nan\n", encoding="utf-8")
        try:
            S.read_gt(gtd, clips); ok = False
        except S.InputError as e:
            ok = "NaN" in str(e)
        check("32 GT の距離 NaN は入力エラー", ok)
        # 来歴が無い候補 → 停止し、出力ファイルを残さない
        out = tmp / "sel.json"
        rc = S.main(["--pattern", str(tmp / "cand_e{epoch:03d}.csv"), "--manifest", str(man), "--gt", str(gtd), "--out", str(out), "--expected", "2"])
        check("33 来歴（csv/done/meta）が無ければ停止（rc=1）", rc == 1)
        check("34 停止時に出力（json/md/tmp）を残さない", not out.exists() and not out.with_suffix(".md").exists() and not list(tmp.glob("*.tmp")))
        rc = S.main(["--pattern", "x{epoch:03d}.csv", "--out", str(out), "--epochs", "99,119"])
        check("35 候補集合を変えるときは --label が要る（rc=2）", rc == 2)
        check("40 門を外す（--no-coverage-gate）と登録用ラベルでは拒否（rc=2）", S.main(["--pattern", "x{epoch:03d}.csv", "--out", str(out), "--no-coverage-gate"]) == 2)
        check("41 門の値を変えると登録用ラベルでは拒否（rc=2）", S.main(["--pattern", "x{epoch:03d}.csv", "--out", str(out), "--coverage-gate", "3"]) == 2)
        check("42 既定の門は 2pt", S.COVERAGE_GATE_DEFAULT == 2.0 and "2.0pt" in S.COVERAGE_POLICY)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # ---- version 2.0（2026-10-12・変更票 #55・凍結 r9 §4）: bicycle 8 を指標に足す・クラス別の併記 ----
    check("43 version 2.2（2.0 の指標のまま・--close-m 省略時は 2.1 と同じ値）・距離クラス {4,6,7,8}", S.SPEC_VERSION.startswith("v2_ckpt_selector/2.2") and S.CONDITIONS["dist_classes"] == [4, 6, 7, 8] and S.DIST_CLS == frozenset({4, 6, 7, 8}))
    # ---- version 2.2（2026-10-26・変更票 #75 ⑬）: 選ぶ指標 E の帯を --close-m で広げる・門 N/D/C は ≤1.5 のまま・E_close/E_mid/定数の行 ----
    w = S.summarize([S.measure([R(c=4, d=1.0), R(t=1, a=60, c=4, d=2.5)], [R(c=4, d=1.2), R(t=1, a=60, c=4, d=2.0)], e_close=3.2)])
    check("43b --close-m 3.2: E は ≤3.2 の 2 ペアの中央値（0.2 と 0.2）・E_close は至近だけ・E_mid は 1.5〜3.2 だけ",
          abs(w["E"] - 0.2) < 1e-9 and abs(w["E_close"] - 0.2) < 1e-9 and abs(w["E_mid"] - 0.2) < 1e-9 and w["n_pairs_e"] == 2 and w["n_pairs_mid"] == 1)
    check("43c 門の分母 N は ≤1.5 のまま（2.5 m の正解は N に入らない）", w["N"] == 1 and w["D"] == 1 and w["C"] == 1)
    w2 = S.summarize([S.measure([R(c=4, d=1.0), R(t=1, a=60, c=4, d=2.5)], [R(c=4, d=1.2), R(t=1, a=60, c=4, d=4.0)], e_close=3.2)])
    check("43d 注意の帯の誤差が E に効く（1.5 m の既定では効かない）", w2["E"] > 0.2 + 1e-9 and abs(S.summarize([S.measure([R(c=4, d=1.0), R(t=1, a=60, c=4, d=2.5)], [R(c=4, d=1.2), R(t=1, a=60, c=4, d=4.0)])])["E"] - 0.2) < 1e-9)
    rc = S.constant_reference({"c1": [R(c=4, d=1.0), R(t=1, a=60, c=4, d=2.0), R(t=2, a=90, c=4, d=3.0)]}, ["c1"], 3.2)
    check("43e 定数の行: 帯 ≤3.2 の中央値 2.0 m を全部に出した相対誤差 {1.0, 0, 1/3} の中央値 = 1/3・至近 ≤1.5 は 1.0 m で 0",
          abs(rc["e_band"]["const_m"] - 2.0) < 1e-9 and abs(rc["e_band"]["E"] - 1.0 / 3.0) < 1e-9 and rc["close"]["n"] == 1 and rc["close"]["E"] == 0.0)
    b8 = M([R(c=8, d=1)], [R(c=8, d=1.2)])
    check("44 bicycle（8）の至近 GT は分母に入り、対応と誤差に数える", b8["N"] == 1 and b8["D"] == 1 and b8["C"] == 1 and abs(b8["E"] - 0.2) < 1e-9 and b8["per_class"][8]["D"] == 1)
    check("45 警告クラス（5）は今までどおり指標に入らない", M([R(c=5, d=1)], [R(c=5, d=1)])["N"] == 0)
    mix = S.summarize([S.measure([R(c=4, d=1)], [R(c=4, d=1.1)]), S.measure([R(c=8, d=1)], [R(c=8, d=1.3)]), S.measure([R(c=6, d=1)], [])])
    check("46 クラス別の併記: N/D/C/ペアと D に占める割合（car 1・bicycle 1・kick は未検出）",
          mix["N"] == 3 and mix["D"] == 2 and mix["per_class"][4]["D"] == 1 and mix["per_class"][8]["D"] == 1 and mix["per_class"][6] == dict(N=1, D=0, C=0, n_pairs=0, share_of_D=0.0)
          and abs(mix["per_class"][8]["share_of_D"] - 0.5) < 1e-12 and mix["per_class"][7] == dict(N=0, D=0, C=0, n_pairs=0, share_of_D=0.0))
    check("47 順位はクラス別の値に依らない（指標は全クラスをまとめた E・C・epoch）", "per_class" not in S.rank.__code__.co_names)
    res_min = dict(spec_version=S.SPEC_VERSION, label="t", not_for_registration=False, epochs=[19], order=[19], selected=19, rule="r",
                   conditions=dict(S.CONDITIONS, coverage_policy="x"), manifest="m", manifest_sha256="0" * 64, n_clips=1, gt_dir="g", gt_sha256="0" * 64,
                   code_sha256=dict(selector="0" * 64, integrity="0" * 64),
                   candidates={"19": dict(mix, verified=True, n_nan_dist=0, n_dup_rows=0, invalid_reason=None, csv_sha256="0" * 64, ckpt_sha256="0" * 64, identity_sha256="0" * 64, n_rows=3)})
    md = S.to_markdown(res_min)
    check("48 md にクラス別の行（bicycle(8) と割合）が出る", "bicycle(8) N 1 / D 1 / C 1 / ペア 1 / 50.0%" in md and "距離クラス [4, 6, 7, 8]" in md)
    # ---- version 2.1（2026-10-13・変更票 #65・凍結 r11 §4）: 高さの表ありの検証セット（最新の val）を受理 ----
    import json as _json, tempfile as _tf, hashlib as _hl
    tmp2 = Path(_tf.mkdtemp(prefix="sel21_"))
    try:
        class _Integ:                                   # verify_download は本物を通さない（来歴の照合は 2.0 の試験 33 で別に見る）
            @staticmethod
            def verify_download(*a, **k):
                return None
        table = tmp2 / "plan.csv"; table.write_text("clip_id,mic_z\nc1,1.6\nc2,2.0\n", encoding="utf-8")
        tsha = _hl.sha256(table.read_bytes()).hexdigest()
        def _cand(name, height):
            c = tmp2 / f"{name}_e019_x.csv"; c.write_text("c1,40,4,0,0,0,1.0\n", encoding="utf-8")
            c.with_suffix(".done.json").write_text(_json.dumps({"n_rows": 1, "identity_sha256": "0" * 64}), encoding="utf-8")
            c.with_suffix(".run_meta.json").write_text(_json.dumps({"ckpt_epoch": 19, "ckpt": "k", "ckpt_sha256": "1" * 64, "clips": ["c1.flac", "c2.flac"], "height": height}), encoding="utf-8")
            return c
        none_h = {"on": True, "table": "", "default": 1.5, "offset": 0}
        tab_h = {"on": True, "table": str(table), "table_sha256": tsha, "default": 1.5, "offset": 0}
        bad_h = {"on": True, "table": str(table), "table_sha256": "f" * 64, "default": 1.5, "offset": 0}
        def _ok(c, ht):
            try:
                S.verify_candidate(_Integ, c, 19, ["c1", "c2"], 2, ht); return True
            except S.InputError:
                return False
        check("49 2.0 の契約（--height-table なし）: 表なし・1.5 m の候補を受理", _ok(_cand("a", none_h), None))
        check("50 2.0 の契約: 表ありの候補は拒否", not _ok(_cand("b", tab_h), None))
        check("51 2.1 の契約（--height-table あり）: 表の sha256 が一致する候補を受理", _ok(_cand("c", tab_h), table))
        check("52 2.1 の契約: 表なしの候補は拒否", not _ok(_cand("d", none_h), table))
        check("53 2.1 の契約: 表の sha256 が違う候補は拒否", not _ok(_cand("e", bad_h), table))
        check("54 2.1 の契約: offset が 0 でない候補は拒否", not _ok(_cand("f", dict(tab_h, offset=0.1)), table))
        # Astra RMIX14-02（2026-10-14・Opus）: 表の欠け・重複・数でない値・既定の高さの違いは拒否
        def _tab(name, text):
            p = tmp2 / name; p.write_text(text, encoding="utf-8"); return p, _hl.sha256(p.read_bytes()).hexdigest()
        t_miss, s_miss = _tab("miss.csv", "clip_id,mic_z\nc1,1.6\n")
        t_dup, s_dup = _tab("dup.csv", "clip_id,mic_z\nc1,1.6\nc1,1.7\nc2,2.0\n")
        t_nan, s_nan = _tab("nan.csv", "clip_id,mic_z\nc1,1.6\nc2,nan\n")
        t_empty, s_empty = _tab("empty.csv", "clip_id,mic_z\nc1,1.6\nc2,\n")
        def _th(t, s):
            return {"on": True, "table": str(t), "table_sha256": s, "default": 1.5, "offset": 0}
        check("56 2.1: 表に評価の本が欠けている（c2 が無い）候補は拒否（RMIX14-02）", not _ok(_cand("g", _th(t_miss, s_miss)), t_miss))
        check("57 2.1: 表に同じ本が 2 行ある候補は拒否", not _ok(_cand("h", _th(t_dup, s_dup)), t_dup))
        check("58 2.1: 表の高さが nan・空の候補は拒否", not _ok(_cand("i", _th(t_nan, s_nan)), t_nan) and not _ok(_cand("j", _th(t_empty, s_empty)), t_empty))
        check("59 2.1: 来歴の既定の高さが 1.5 m でない（9 m）候補は拒否", not _ok(_cand("k", dict(tab_h, default=9.0)), table))
        check("60 check_height_table: 全部の本を 1 行ずつ持つ表は通る", S.check_height_table(table, ["c1", "c2"])["n_clips_covered"] == 2)
        # Astra O14-03（2026-10-14・Opus）: 実行時（height_patch）は clip_id の文字列そのままで引く → 拡張子・空白・パス付きは拒否
        def _rej(text):
            t, _ = _tab("o14_%d.csv" % (abs(hash(text)) % 10**8), text)
            try:
                S.check_height_table(t, ["c1", "c2"]); return False
            except S.InputError:
                return True
        check("61 check_height_table: 拡張子付き（c2.flac）は拒否（実行時は既定の 1.5 m になる・O14-03）", _rej("clip_id,mic_z\nc1,1.6\nc2.flac,2.1\n"))
        check("62 check_height_table: 前後の空白（' c2 '）・パス付き（x/c2）は拒否", _rej("clip_id,mic_z\nc1,1.6\n c2 ,2.1\n") and _rej("clip_id,mic_z\nc1,1.6\nx/c2,2.1\n"))
        check("63 check_height_table: 正しい行があっても、同じ本の別の書き方の行があれば拒否", _rej("clip_id,mic_z\nc1,1.6\nc2,2.1\nc2.flac,2.0\n"))
        _hp = ROOT / "server_sde" / "height_patch.py"
        if _hp.is_file():   # 実行時の読み方そのもので、通った表の値が引けることを確かめる（height_patch を環境変数つきで読み直す）
            import importlib.util as _ilu, os as _os
            _old = {k: _os.environ.get(k) for k in ("HEIGHT_TABLE", "HEIGHT_COND")}
            _os.environ["HEIGHT_TABLE"] = str(table)
            try:
                _sp = _ilu.spec_from_file_location("_hp_o14", _hp); _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
                _same = all(abs(_m.lookup(c) - float(z)) < 1e-9 for c, z in (("c1", "1.6"), ("c2", "2.0")))
            except Exception as _e:
                _same = False; print("   height_patch:", _e)
            finally:
                for k, v in _old.items():
                    (_os.environ.__setitem__(k, v) if v is not None else _os.environ.pop(k, None))
            check("64 検査を通った表は、実行時の height_patch.lookup でも同じ値（既定の高さに落ちない）", _same)
        check("55 --height-table の CLI を受ける（候補が無いので rc=1 で停止）", S.main(["--pattern", str(tmp2 / "zz_e{epoch:03d}_x.csv"), "--out", str(tmp2 / "o.json"), "--height-table", str(table)]) == 1)
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    n_ok = sum(ok for _, ok in results)
    print(f"{n_ok}/{len(results)} {'ALL PASS' if n_ok == len(results) else 'FAILED'}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
