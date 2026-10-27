# -*- coding: utf-8 -*-
"""描き直した束（v23〜v26・凍結 r9 §2）の中身の検査と、試聴の候補の一覧（2026-10-12・Fable）。

生成の完了検査（server_sde/_v2final_gen_complete_check.py = 本数・成果物・HDF5）とは別に、**音の直しが全部の本に掛かっているか**を scene.json から確かめる:
  A. 置き場に GEN_VERSIONS.txt があり、描画の連なりに step11_v26_render.py が入っている
  B. サイレンの音源は全部 v26 の形（params.v26・v24.siren_chain = v26）で、向きの差（v24.dir_mean_db）と Δ（v24.delta_db）が付いている
  C. 車・バイクの音源は全部 v24 の大きさの規則（v24.rule = car_curve／bike_curve）。EV（car_variant = ev_v26）は通報音（avas）つき。大型車（params.heavy）は無い（D25 不採用）
  D. 警告音（ベル・クラクション・バック音・踏切の警報機）は最大値合わせ（v24.match = LAFmax・crest_db あり）。**列車の車両（class crossing・params.train_car）は RMS のまま**（AR12-08・SV26-07）。
     バック音は v23 の 3 型・クラクションは v23 の押しの列
  E. ベルつき走行（v19bellbike）は v25 のベルの時刻（scene.v25.bell_timing）で、後に置いた本の数が manifest_v26_bellbike.csv と一致
  F. 試聴の候補: クラスごとに固定 seed で数本を選び、W チャンネルの LAeq と中身を一覧にする（REGEN_CHECK.md・wav は作らない）
  G. （2026-10-13 v27・#64）v19mix の本は音源の構成が plan の mix と一致し、電車の車両は背景（no_label）、ラベルの track はクラスごとに 0..（上限 3）。v19motofar の本はバイク 1 台・横距離 4〜12 m。
     束が全部そろっているとき（scene の数 = N_ROWS_TOTAL_EXPECTED_V27）は v19mix／v19motofar の本数を params と照合する
  H. （2026-10-13・Opus R-1・Astra MIX13-04）組み合わせのゲート後: **全部の場面**（v19mix も既存の v11core なども）の構成が 2 つ以上の本ごとに metadata（クラス・track）を読み、予定した音（電車を除く）が全部ラベルに
     1 フレーム以上残る本を組ごとに数える（split 別・seed 数・同時に 2 つ以上の音が残るフレーム数の診断値つき）。電車だけの組は対象外（別記）。NG にはしない（報告）。可聴ゲートは背景（暗騒音＋雨）に対する判定で他の音源のマスキングは見ない（MIX13-05）
  A の params: 別名の版の記録が params の行だけ違うとき、束の params_snapshots/ の写しを比べ、違う鍵が帳簿・文書（D33_TOPUP・N_ROWS_TOTAL_EXPECTED_V28・EVAL_V3・status・note 類）だけなら注意、音・場面に効く鍵なら NG（MIX13-02）。報告書は全判定のあとに作る（MIX13-03）
使い方: python scripts/_v26_regen_check.py --ds out/dataset_outdoor_siren_v2final_v26 [--n-listen 3] [--no-audio]
       python scripts/_v26_regen_check.py --selftest <一時dir>   （偽の束で A〜E の判定を確かめる・SV26-07）
終了コード: 0 = 全部 OK、1 = NG あり。
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WARN = ("bike_bell", "horn", "backup_beep", "crossing")


MIX_IDX = {"car": 4, "bicycle": 8, "kick": 6, "moto": 7, "siren": 0, "horn": 1, "backup_beep": 2, "crossing": 5, "bike_bell": 3}
MIX_NAME = {v: k for k, v in MIX_IDX.items()}


def _mix_label_stats(ds: Path, clip: str, s: dict, v27):
    """H: この本の predicted した音（電車を除く）ごとの必要な track 数と、metadata に残った distinct な track 数。戻り = (mix, 全部残ったか, need, labeled, metadata が無い)"""
    from collections import Counter as _C, defaultdict as _D
    comp = v27.parse_mix(s["row"]["mix"]); need = _C(MIX_IDX[t] for t in comp.elements() if t != "train")
    meta = ds / "metadata" / f"{clip}.csv"
    if not meta.is_file():
        return s["row"]["mix"], False, need, {}, True
    have = _D(set)
    for line in meta.read_text(encoding="utf-8").splitlines():
        g = line.strip().split(",")
        if len(g) >= 3 and g[0].strip().isdigit():
            have[int(g[1])].add(int(g[2]))
    labeled = {ci: len(have.get(ci, ())) for ci in need}
    return s["row"]["mix"], all(labeled[ci] >= n for ci, n in need.items()), need, labeled, False


def _check_mix_scene(s: dict, v27) -> None:
    """描いた scene.json から読める範囲の G 検査（_mic_arr は保存されないので、マイクの共有の検査は生成前の check_scene に任せる）。"""
    from collections import Counter as _C, defaultdict as _D
    row = s["row"]; srcs = s["sources"]
    if row["scenario"] == "v19motofar":
        assert len(srcs) == 1 and srcs[0]["class"] == "motorcycle", [x["class"] for x in srcs]
        lat = abs(float(srcs[0]["wp"][0][2])); assert 4.0 - 1e-9 <= lat <= 12.0 + 1e-9, lat
        return
    comp = v27.parse_mix(row["mix"]); cnt = _C(); trains = set()
    for x in srcs:
        if (x.get("params") or {}).get("train_car") or "train_car_index" in x:
            trains.add(x.get("mix_group")); assert x.get("no_label") and x.get("train_background") == "D14", "電車の車両にラベル"
        else:
            cnt[x["mix_token"]] += 1
    cnt["train"] = len(trains)
    want = {t: comp[t] for t in v27.TOKENS if comp[t]}; got = {t: cnt[t] for t in v27.TOKENS if cnt[t]}
    assert want == got, f"構成が plan と違う: plan={want} 場面={got}"
    tr = _D(list)
    for x in srcs:
        if not x.get("no_label"):
            tr[x["class"]].append(int(x["track"]))
    for c, v in tr.items():
        assert sorted(v) == list(range(len(v))) and len(v) <= 3, f"{c} の track {v}"
    assert (s.get("v19") or {}).get("n_label_dropped", 0) == 0, "落としたラベルがある"


def selftest(tmp: Path) -> int:
    """偽の束（scene.json だけ）で A〜E の判定を確かめる。"""
    import shutil
    ok = []

    def write_bundle(root: Path, scenes: dict, gen_versions: bool = True):
        shutil.rmtree(root, ignore_errors=True)
        for clip, s in scenes.items():
            (root / "work" / clip).mkdir(parents=True, exist_ok=True)
            (root / "work" / clip / "scene.json").write_text(json.dumps({k: v for k, v in s.items() if k != "_meta"}), encoding="utf-8")
            if "_meta" in s:                                                       # H 用: metadata（frame,class,track,az,el）
                (root / "metadata").mkdir(parents=True, exist_ok=True)
                (root / "metadata" / f"{clip}.csv").write_text("".join(f"{i},{c},{t},0,0\n" for i, (c, t) in enumerate(s["_meta"])), encoding="utf-8")
        if gen_versions:
            (root / "GEN_VERSIONS.txt").write_text("render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\n", encoding="utf-8")
    siren = {"class": "siren", "params": {"v26": True, "siren_type": "peepo"}, "v24": {"rule": "siren", "siren_chain": "v26", "dir_mean_db": -0.5, "delta_db": 1.74}}
    car = {"class": "car_drive", "params": {"audio_seed": 1}, "v24": {"rule": "car_curve"}, "car_variant": "normal"}
    ev = {"class": "car_drive", "params": {"audio_seed": 2}, "v24": {"rule": "car_curve"}, "car_variant": "ev_v26", "avas": {"l1m_avas_db": 60.0}}
    alarm = {"class": "crossing", "params": {"click_seed": 3}, "v24": {"match": "LAFmax", "crest_db": 2.0}, "match_max": True}
    train_rms = {"class": "crossing", "params": {"audio_seed": 4, "train_car": True}, "train_car_index": 0, "v24": {"match": "RMS（列車の車両・AR12-08）"}}
    train_bad = {"class": "crossing", "params": {"audio_seed": 5, "train_car": True}, "train_car_index": 1, "v24": {"match": "LAFmax", "crest_db": 1.0}, "match_max": True}
    bell = {"class": "bike_bell", "params": {"bell_type": "pin"}, "v24": {"match": "LAFmax", "crest_db": 3.0}, "match_max": True}
    good = {"train_syn_clip00001": {"row": {"scenario": "v11core", "n_car": "1", "n_warnings": "2", "w1_class": "siren", "w2_class": "crossing", "split": "train", "seed": "1"}, "sources": [car, siren, alarm], "_meta": [(4, 0), (0, 0), (5, 0)]},
            "train_syn_clip00002": {"row": {"scenario": "v11core"}, "sources": [ev]},
            "train_syn_clip11001": {"row": {"scenario": "v12train"}, "sources": [train_rms, dict(train_rms, train_car_index=1)]},
            "train_syn_clip40001": {"row": {"scenario": "v19bellbike", "split": "train", "seed": "4"}, "sources": [{"class": "bicycle", "params": {}}, bell], "v25": {"bell_timing": {"side": "after"}}, "_meta": [(8, 0), (3, 0)]},
            "train_syn_clip50001": {"row": {"scenario": "v19mix", "mix": "car+bicycle", "split": "train", "seed": "2", "n_car": "1", "n_warnings": "0"}, "sources": [dict(car, mix_token="car", mix_group=0, track=0), {"class": "bicycle", "params": {}, "mix_token": "bicycle", "mix_group": 1, "track": 0}],
                                    "v19": {"n_label_dropped": 0}, "_meta": [(4, 0), (8, 0)]},
            "train_syn_clip50002": {"row": {"scenario": "v19mix", "mix": "car+bicycle", "split": "val", "seed": "3", "n_car": "1", "n_warnings": "0"}, "sources": [dict(car, mix_token="car", mix_group=0, track=0), {"class": "bicycle", "params": {}, "mix_token": "bicycle", "mix_group": 1, "track": 0}],
                                    "v19": {"n_label_dropped": 0}, "_meta": [(4, 0)]},
            "train_syn_clip70001": {"row": {"scenario": "v19motofar"}, "sources": [{"class": "motorcycle", "params": {"engine_class": "moped"}, "v24": {"rule": "bike_curve"}, "wp": [[0.0, -40.0, 6.0, 1.5], [10.0, 40.0, 6.0, 1.5]]}]},
            "train_syn_clip54051": {"row": {"scenario": "v19mix", "mix": "train+train", "split": "train", "seed": "9"}, "sources": [dict(train_rms, mix_token="train", mix_group=0, track=0, no_label=True, train_background="D14", train_index=0), dict(train_rms, mix_token="train", mix_group=1, track=0, no_label=True, train_background="D14", train_index=1)], "v19": {"n_label_dropped": 0}, "_meta": []}}
    write_bundle(tmp / "good", good)
    rc = main(["--ds", str(tmp / "good"), "--no-audio", "--n-listen", "1"])
    ok.append(("良い束（列車の車両は RMS・警報機は LAFmax・サイレン v26・EV に通報音・ベルの時刻）→ OK", rc == 0))
    bad = dict(good); bad["train_syn_clip11001"] = {"row": {"scenario": "v12train"}, "sources": [train_bad]}
    write_bundle(tmp / "bad_train", bad)
    ok.append(("列車の車両が最大値合わせ → NG（SV26-07）", main(["--ds", str(tmp / "bad_train"), "--no-audio", "--n-listen", "1"]) == 1))
    bad2 = dict(good); bad2["train_syn_clip00001"] = {"row": {"scenario": "v11core"}, "sources": [car, dict(siren, v24={"rule": "siren", "siren_chain": "old"}), alarm]}
    write_bundle(tmp / "bad_siren", bad2)
    ok.append(("サイレンが旧の連なり → NG", main(["--ds", str(tmp / "bad_siren"), "--no-audio", "--n-listen", "1"]) == 1))
    write_bundle(tmp / "no_gv", good, gen_versions=False)
    ok.append(("GEN_VERSIONS.txt が無い → NG", main(["--ds", str(tmp / "no_gv"), "--no-audio", "--n-listen", "1"]) == 1))
    bad3 = dict(good); bad3["train_syn_clip00002"] = {"row": {"scenario": "v11core"}, "sources": [dict(ev, avas=None)]}
    write_bundle(tmp / "bad_ev", bad3)
    ok.append(("EV に通報音が無い → NG", main(["--ds", str(tmp / "bad_ev"), "--no-audio", "--n-listen", "1"]) == 1))
    md = (tmp / "good" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append(("H: 良い束（car+bicycle 2 本・v11core の car+crossing+siren 1 本・bellbike 1 本）で 全部残る本 3/4・car 3/3・bicycle 2/3 と組の行が REGEN_CHECK.md に出る", "全部残る本: 3/4" in md and "bicycle 2/3（67%）" in md and "car 3/3（100%）" in md and "- car+bicycle: 1/2（train 1/1・val 0/1・seed 2" in md))
    hcsv = (tmp / "good" / "REGEN_H_counts.csv").read_text(encoding="utf-8").splitlines()
    ok.append(("H: REGEN_H_counts.csv に car+bicycle（v19mix・2 本・全部残る 1・train 1/1・val 0/1・seed 2）と car+siren+crossing（existing・1/1・札の順）", any(l.startswith("car+bicycle,v19mix,2,1,1,1,1,0,2,") for l in hcsv) and any(l.startswith("car+siren+crossing,existing,1,1,1,1,0,0,1,") for l in hcsv)))
    ok.append(("H: 電車だけの組（train+train）は対象外として別記・同時フレームの診断値が出る", "教師音源ゼロの組（電車だけ）は対象外: train+train 1" in md and "同時に 2 つ以上の予定した音" in md))
    gv_ok = dict(good); write_bundle(tmp / "gv_params", gv_ok)
    (tmp / "gv_params" / "GEN_VERSIONS_20261013_230000.txt").write_text("render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 60000\nabc  md/design/v2final_params.json\n", encoding="utf-8")
    (tmp / "gv_params" / "GEN_VERSIONS.txt").write_text("render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 50574\ndef  md/design/v2final_params.json\n", encoding="utf-8")
    ok.append(("A: 別名の版が params と行数の行だけ違うが params の写しが無い → NG（中身を比べられない）", main(["--ds", str(tmp / "gv_params"), "--no-audio", "--n-listen", "1"]) == 1))
    import hashlib as _hl
    P_real = json.loads((ROOT / "md/design/v2final_params.json").read_text(encoding="utf-8"))
    P_book = dict(P_real); P_book["D33_TOPUP"] = dict(P_real.get("D33_TOPUP") or {}, n_clips=4850); P_book["N_ROWS_TOTAL_EXPECTED_V28"] = 55424; P_book["status"] = "fixed"
    P_sem = json.loads(json.dumps(P_real)); P_sem["D1_SIREN_MIX"] = {"near": 1.0, "far": 0.0, "recede": 0.0}
    def _snap(root, P):
        b = json.dumps(P, ensure_ascii=False, indent=1).encode("utf-8"); h = _hl.sha256(b).hexdigest()
        (root / "params_snapshots").mkdir(exist_ok=True); (root / "params_snapshots" / f"params_{h[:16]}.json").write_bytes(b); return h
    for name, P2, expect in (("gv_book", P_book, 0), ("gv_sem", P_sem, 1)):
        root = tmp / name; write_bundle(root, good)
        h0 = _snap(root, P_real); h1 = _snap(root, P2)
        (root / "GEN_VERSIONS.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 50574\n{h0}  md/design/v2final_params.json\n", encoding="utf-8")
        (root / "GEN_VERSIONS_20261013_230000.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 55424\n{h1}  md/design/v2final_params.json\n", encoding="utf-8")
        rc = main(["--ds", str(root), "--no-audio", "--n-listen", "1"])
        ok.append((f"A: params の写しの差が {'帳簿の鍵（D33・V28・status）だけ → 注意' if expect == 0 else '音に効く鍵（D1_SIREN_MIX）→ NG'}（rc {rc}）", rc == expect))
    # Astra RMIX14-01（2026-10-14・Opus）: 名前は合うが中身が別の写し（D1 を変えた版の sha の名前に、元の params の中身を置く）→ NG
    root = tmp / "gv_mismatch"; write_bundle(root, good)
    h0 = _snap(root, P_real); b_sem = json.dumps(P_sem, ensure_ascii=False, indent=1).encode("utf-8"); h_sem = _hl.sha256(b_sem).hexdigest()
    (root / "params_snapshots" / f"params_{h_sem[:16]}.json").write_bytes(json.dumps(P_real, ensure_ascii=False, indent=1).encode("utf-8"))
    (root / "GEN_VERSIONS.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 50574\n{h0}  md/design/v2final_params.json\n", encoding="utf-8")
    (root / "GEN_VERSIONS_20261013_230000.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 55424\n{h_sem}  md/design/v2final_params.json\n", encoding="utf-8")
    rc = main(["--ds", str(root), "--no-audio", "--n-listen", "1"]); md_mm = (root / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"A: 写しの名前は記録の sha に合うが中身が別（RMIX14-01）→ NG（rc {rc}）", rc == 1 and "一致しない" in md_mm))
    ok.append(("params_diff_allowed: 帳簿・文書の鍵は許し、D1／SALTS は許さない", params_diff_allowed([("D33_TOPUP", "n_clips"), ("status",), ("D31_MIX", "note"), ("_note_v28",)])[1] == [] and params_diff_allowed([("D1_SIREN_MIX", "near"), ("SALTS", "v13")])[1] == ["D1_SIREN_MIX.near", "SALTS.v13"]))
    (tmp / "gv_params" / "GEN_VERSIONS_20261013_230000.txt").write_text("render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py\nplan rows expected: 60000\nabc  md/design/v2final_params.json\nxyz  scripts/step11_v27_render.py\n", encoding="utf-8")
    ok.append(("A: 別名の版が描画の連なりの行でも違う → NG", main(["--ds", str(tmp / "gv_params"), "--no-audio", "--n-listen", "1"]) == 1))
    bad5 = dict(good); bad5["train_syn_clip50001"] = {"row": {"scenario": "v19mix", "mix": "car+bicycle"}, "sources": [dict(car, mix_token="car", mix_group=0, track=0)], "v19": {"n_label_dropped": 0}, "_meta": [(4, 0)]}
    write_bundle(tmp / "bad_mix", bad5)
    ok.append(("G: v19mix の構成が plan の mix と違う（自転車が無い）→ NG", main(["--ds", str(tmp / "bad_mix"), "--no-audio", "--n-listen", "1"]) == 1))
    md_bad = (tmp / "bad_mix" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append(("G の NG が報告書の判定にも出る（Astra MIX13-03）", "判定: **NG" in md_bad and "## NG" in md_bad))
    bad6 = dict(good); bad6["train_syn_clip50001"] = {"row": {"scenario": "v19mix", "mix": "car+train"}, "sources": [dict(car, mix_token="car", mix_group=0, track=0), dict(train_rms, mix_token="train", mix_group=1, track=0)], "v19": {"n_label_dropped": 0}, "_meta": [(4, 0)]}
    write_bundle(tmp / "bad_trainlabel", bad6)
    ok.append(("G: 組み合わせの電車の車両に no_label が無い → NG", main(["--ds", str(tmp / "bad_trainlabel"), "--no-audio", "--n-listen", "1"]) == 1))
    bad7 = dict(good); bad7["train_syn_clip50001"] = {k: v for k, v in good["train_syn_clip50001"].items() if k != "_meta"}
    write_bundle(tmp / "no_meta", bad7)
    ok.append(("H: v19mix の metadata が無い → NG", main(["--ds", str(tmp / "no_meta"), "--no-audio", "--n-listen", "1"]) == 1))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    # R70-02（Astra 2026-10-19）: v29 の束で EV（ev_v26）を数える。20 台（EV 3・重い 2・ふつう 15）= 15%／10% は OK、全部ふつうは NG
    import step11_v29_render as v29
    import step11_v19_render as v19r
    tiers = ("critical", "caution", "safe")
    def ev_bundle(root: Path, variants: list, declare: str = "real", snapshot: bool = True):
        """declare = real（現物の sha）／wrong（別の sha）／none（宣言なし）。snapshot = 写しを置くか。"""
        import hashlib as _hl
        shutil.rmtree(root, ignore_errors=True)
        _pb = (ROOT / "md/design/v2final_params.json").read_bytes(); _real = _hl.sha256(_pb).hexdigest()
        _decl = _real if declare == "real" else ("f" * 64 if declare == "wrong" else "")
        for i, var in enumerate(variants):
            r = v29._fake_core_row(5000 + i, tiers[i % 3], "L" if i % 2 else "R", "static")
            r["clip_id"] = f"train_syn_clip{90001 + i:05d}"
            s = v29._conv(v19r.sample_scene_v19(dict(r))); s.pop("_mic_arr", None)
            car = next(x for x in s["sources"] if x["class"] == "car_drive" and x.get("track", 0) == 0)
            car["car_variant"] = var
            if var.startswith("ev"):
                car["avas"] = {"fixture": True}
            (root / "work" / r["clip_id"]).mkdir(parents=True, exist_ok=True)
            (root / "work" / r["clip_id"] / "scene.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        (root / "GEN_VERSIONS.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py -> step11_v27_render.py -> step11_v29_render.py\nplan rows expected: {len(variants)}\n"
                                                + (f"{_decl}  md/design/v2final_params.json\n" if _decl else ""), encoding="utf-8")
        if snapshot and _decl:
            (root / "params_snapshots").mkdir(exist_ok=True); (root / "params_snapshots" / f"params_{_decl[:16]}.json").write_bytes(_pb)
    ev_bundle(tmp / "ev_ok", ["ev_v26"] * 3 + ["heavy"] * 2 + ["normal"] * 15)
    rc_ok = main(["--ds", str(tmp / "ev_ok"), "--no-audio", "--n-listen", "1"]); md_ok = (tmp / "ev_ok" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（R70-02）: v29 の束 20 台（EV 3 = ev_v26・重い 2・ふつう 15）→ OK・EV 15.0% と数える（rc {rc_ok}）", rc_ok == 0 and "EV 15.0%" in md_ok and "割合の検査はした" in md_ok))
    ev_bundle(tmp / "ev_bad", ["normal"] * 20)
    rc_bad = main(["--ds", str(tmp / "ev_bad"), "--no-audio", "--n-listen", "1"]); md_bad = (tmp / "ev_bad" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（R70-02）: 全部ふつうの束 → NG「車の種別の割合」（rc {rc_bad}）", rc_bad == 1 and "I: 車の種別の割合" in md_bad))
    ev_bundle(tmp / "ev_part", ["normal"] * 20)
    import hashlib as _hl2
    _real2 = _hl2.sha256((ROOT / "md/design/v2final_params.json").read_bytes()).hexdigest()
    (tmp / "ev_part" / "GEN_VERSIONS.txt").write_text(f"render chain (import order): step11_v9_render.py -> ... -> step11_v26_render.py -> step11_v27_render.py -> step11_v29_render.py\nplan rows expected: 62102\n{_real2}  md/design/v2final_params.json\n", encoding="utf-8")
    rc_part = main(["--ds", str(tmp / "ev_part"), "--no-audio", "--n-listen", "1"]); md_part = (tmp / "ev_part" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（R70-02）: 部分の束（20 ≠ 宣言 62,102）は割合を見ない・契約の検査は走る（rc {rc_part}）", rc_part == 0 and "割合の検査はしていない" in md_part))
    # RR70-02: 写しが無いとき
    ev_bundle(tmp / "ev_nosnap_ok", ["ev_v26"] * 3 + ["heavy"] * 2 + ["normal"] * 15, declare="real", snapshot=False)
    rc_ns = main(["--ds", str(tmp / "ev_nosnap_ok"), "--no-audio", "--n-listen", "1"]); md_ns = (tmp / "ev_nosnap_ok" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（RR70-02）: 写しが無いが現物の sha が宣言と一致 → 現物で契約を見て OK（rc {rc_ns}）", rc_ns == 0 and "現物（sha が宣言と一致）" not in md_ns.split("## NG")[0] or rc_ns == 0))
    ev_bundle(tmp / "ev_nosnap_bad", ["ev_v26"] * 3 + ["heavy"] * 2 + ["normal"] * 15, declare="wrong", snapshot=False)
    rc_nb = main(["--ds", str(tmp / "ev_nosnap_bad"), "--no-audio", "--n-listen", "1"]); md_nb = (tmp / "ev_nosnap_bad" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（RR70-02）: 宣言 sha の写しが無く現物の sha も違う → 契約を確かめられないで NG（rc {rc_nb}）", rc_nb == 1 and "契約を確かめられない" in md_nb))
    ev_bundle(tmp / "ev_nodecl", ["ev_v26"] * 3 + ["heavy"] * 2 + ["normal"] * 15, declare="none", snapshot=False)
    rc_nd = main(["--ds", str(tmp / "ev_nodecl"), "--no-audio", "--n-listen", "1"]); md_nd = (tmp / "ev_nodecl" / "REGEN_CHECK.md").read_text(encoding="utf-8")
    ok.append((f"I（RR70-02）: params の sha の宣言が無い → NG（rc {rc_nd}）", rc_nd == 1 and "params の sha の行が無い" in md_nd))
    for name, c in ok[-6:]:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# params の差で許す鍵（帳簿・文書）。それ以外（音・場面・乱数に効く値）が違えば「生成条件の混在」= NG（Astra MIX13-02）
ALLOWED_PARAM_TOP = {"status", "declared_at", "fixed_note", "draft_note", "D33_TOPUP", "N_ROWS_TOTAL_EXPECTED_V28", "_note_v28", "EVAL_V3",
                     "N_ROWS_TOTAL_EXPECTED_V29", "_note_v29", "RAIN_KIND_P_note", "D10_note_v29"}      # 2026-10-19 #70: 帳簿の鍵（値に効く D70・D2・D19_SLOW_ADD は入れない）
ALLOWED_PARAM_LEAF = {"note", "decided", "audition", "coverage_note", "rejected", "proposed"}


def params_diff_paths(a, b, path=()) -> list:
    """2 つの params（dict）の違う葉のパスの一覧。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(path + (k,))
            else:
                out += params_diff_paths(a[k], b[k], path + (k,))
        return out
    return [] if a == b else [path]


def params_diff_allowed(paths: list) -> tuple[list, list]:
    ok, bad = [], []
    for pth in paths:
        if (pth[0] in ALLOWED_PARAM_TOP) or (pth[-1] in ALLOWED_PARAM_LEAF) or str(pth[-1]).startswith("_note"):
            ok.append(".".join(map(str, pth)))
        else:
            bad.append(".".join(map(str, pth)))
    return ok, bad


def _params_sha_of(gv_text: str) -> str | None:
    for l in gv_text.splitlines():
        if l.endswith("md/design/v2final_params.json"):
            return l.split()[0]
    return None


def _find_snapshot(ds: Path, sha: str | None):
    """版の記録の params の sha に当たる写し（params_snapshots/）を返す。無ければ None。
    Astra RMIX14-01（2026-10-14・Opus）: ファイル名の頭だけでなく、**中身のバイト列の SHA256 が記録の sha（64 桁）と完全に一致する**ことを確かめる。
    名前は合うが中身が違う・記録の sha が 64 桁でない・候補が複数 → ValueError（呼び出し側で NG）。"""
    if not sha:
        return None
    import hashlib
    cands = [q for q in sorted((ds / "params_snapshots").glob("params_*.json")) if q.stem.endswith(sha[:16]) or sha.startswith(q.stem.split("_")[-1])]
    if not cands:
        return None
    if len(cands) > 1:
        raise ValueError(f"params の写しの候補が複数ある（{[q.name for q in cands]}）")
    q = cands[0]; real = hashlib.sha256(q.read_bytes()).hexdigest()
    if len(sha) != 64 or real != sha:
        raise ValueError(f"params の写し {q.name} の中身の sha {real[:16]}… が版の記録の sha {sha[:16]}… と一致しない（名前は合うが中身が別）")
    return q


def _comp_label_stats(ds: Path, clip: str, tokens: list):
    """H: 予定した音（電車を除く）ごとの必要な track 数と、metadata に残った distinct な track 数、同時に複数の音が残るフレーム数。"""
    from collections import Counter as _C, defaultdict as _D
    need = _C(MIX_IDX[t] for t in tokens if t != "train")
    meta = ds / "metadata" / f"{clip}.csv"
    if not meta.is_file():
        return False, need, {}, 0, True
    have = _D(set); per_frame = _D(set)
    for line in meta.read_text(encoding="utf-8").splitlines():
        g = line.strip().split(",")
        if len(g) >= 3 and g[0].strip().isdigit():
            fr, ci, tr = int(g[0]), int(g[1]), int(g[2])
            if ci in need:
                have[ci].add(tr); per_frame[fr].add((ci, tr))
    labeled = {ci: len(have.get(ci, ())) for ci in need}
    all_ok = all(labeled[ci] >= n for ci, n in need.items())
    n_simul = sum(1 for fr, st in per_frame.items() if sum(min(sum(1 for (c, t) in st if c == ci), n) for ci, n in need.items()) >= 2)
    return all_ok, need, labeled, n_simul, False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    def opt(k, d=None):
        return argv[argv.index(k) + 1] if k in argv else d
    if "--selftest" in argv:
        return selftest(Path(opt("--selftest")))
    import step10_v27_plan as p27
    ds = Path(opt("--ds", str(ROOT / "out/dataset_outdoor_siren_v2final_v26")))
    n_listen = int(opt("--n-listen", "3")); no_audio = "--no-audio" in argv
    ng, notes = [], []
    # ---- A. 版の記録
    gv = ds / "GEN_VERSIONS.txt"
    if not gv.is_file() or "step11_v26_render.py" not in gv.read_text(encoding="utf-8"):
        ng.append("A: GEN_VERSIONS.txt が無いか、連なりに step11_v26_render.py が無い")
    else:
        gv_text = gv.read_text(encoding="utf-8")
        for v in sorted(ds.glob("GEN_VERSIONS_*.txt")):                    # 別名の版の記録: 本体と違えば版が混じっている（NG）・同じなら書き込みの競合の名残（注意）
            a_lines = gv_text.splitlines(); b_lines = v.read_text(encoding="utf-8").splitlines()
            diff = sorted(set(a_lines) ^ set(b_lines))
            if not diff:
                notes.append(f"A: {v.name} は GEN_VERSIONS.txt と同じ（同時に始まったシャードの書き込みの競合の名残・害なし）")
            elif all(l.endswith("md/design/v2final_params.json") or l.startswith("plan rows expected:") for l in diff):
                # params の差は中身で判定（Astra MIX13-02）: 写し（params_snapshots/）を sha で探し、違う鍵が帳簿・文書だけなら注意、音や場面に効く鍵なら NG
                try:
                    sa, sb = _find_snapshot(ds, _params_sha_of(gv_text)), _find_snapshot(ds, _params_sha_of(v.read_text(encoding="utf-8")))
                except ValueError as e:                                     # Astra RMIX14-01: 写しの中身が記録の sha と違う・候補が複数
                    ng.append(f"A: {v.name}: {e}"); continue
                if sa is None or sb is None:
                    ng.append(f"A: {v.name} は params が違うが、params の写し（params_snapshots/）が無いので中身を比べられない（元 {sa is not None}・別名 {sb is not None}）")
                else:
                    try:
                        paths = params_diff_paths(json.loads(sa.read_text(encoding="utf-8")), json.loads(sb.read_text(encoding="utf-8")))
                        okk, bad = params_diff_allowed(paths)
                        if bad:
                            ng.append(f"A: {v.name} の params は音・場面に効く鍵が違う（生成条件の混在）: {bad[:6]}")
                        else:
                            notes.append(f"A: {v.name} は params の帳簿・文書の鍵（{okk[:6]}{'…' if len(okk) > 6 else ''}）と plan の行数の行だけ違う（v28 の積み増しの名残・描画の連なりは同一）")
                    except Exception as e:  # noqa: BLE001
                        ng.append(f"A: {v.name} の params の写しが読めない: {e}")
            else:
                ng.append(f"A: {v.name} が GEN_VERSIONS.txt と違う（途中でコードが変わった束＝版が混じっている）: {diff[:4]}")
    scenes = sorted((ds / "work").glob("*/scene.json"))
    if not scenes:
        print(f"scene.json が無い: {ds}"); return 1
    cnt = Counter(); by_cls = defaultdict(list); bell_after = 0; bell_n = 0
    # H: 組ごとの集計（全部の場面・構成の大きさ 2 以上）: [本数, 全部残った本数, train 本数, train 残, val 本数, val 残, seed の集合, 同時フレーム数の一覧]
    hcnt = defaultdict(lambda: {"n": 0, "all": 0, "n_train": 0, "all_train": 0, "n_val": 0, "all_val": 0, "seeds": set(), "simul": [], "source": set()})
    htok = defaultdict(lambda: [0, 0]); h_empty = Counter()
    _pp = ROOT / "md/design/v2final_params.json"; P_all = json.loads(_pp.read_text(encoding="utf-8")) if _pp.is_file() else {}
    # I（#70・2026-10-19）: v29 で描いた束だけ（版の記録に step11_v29_render がある）、高さ・横距離・隊列・歩く速さ・D10／D19 の受け入れを全部の場面で見る
    v29c = None
    _gvp = ds / "GEN_VERSIONS.txt"
    if _gvp.is_file() and "step11_v29_render.py" in _gvp.read_text(encoding="utf-8", errors="replace"):
        import step11_v29_render as v29c  # noqa: F811
    import re as _re
    _gvtxt = _gvp.read_text(encoding="utf-8", errors="replace") if _gvp.is_file() else ""
    _m = _re.search(r"plan rows expected:\s*(\d+)", _gvtxt)
    i_total = int(_m.group(1)) if _m else -1          # R70-02: 割合の検査（全量だけ）の総数は、この束が宣言した総数（学習 62,102・eval 6,000）で見る
    i_cnt = Counter(); i_rain = defaultdict(list); i_dev = []
    for p in scenes:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            ng.append(f"scene.json が読めない: {p} ({e})"); continue
        clip = p.parent.name; row = s.get("row") or {}; scen = row.get("scenario", "")
        if v29c is not None:
            for b in v29c.check_scene_v29(s, row)[:3]:
                ng.append(f"I: {clip} {b}")
            i_cnt["scenes"] += 1
            for x in s.get("sources", []):
                if x.get("class") == "car_drive" and x.get("kind") == "vehicle":
                    var = str(x.get("car_variant", "?")); var = "ev" if var.startswith("ev") else var      # R70-02: v26 以降の EV は "ev_v26"・旧 "ev" も EV として数える
                    i_cnt["car"] += 1; i_cnt["car_" + var] += 1
            if row.get("rain") and isinstance(s.get("rain"), dict) and s["rain"].get("dba") is not None:
                i_rain[row["rain"]].append(float(s["rain"]["dba"]))
            lat_plan = row.get("cs_lat_m") if str(row.get("close_slow", "")) == "1" else (row.get("d19_lat_m") if str(row.get("d19", "")) == "1" else "")
            if lat_plan not in ("", None):
                try:
                    mic_arr = v29c.mic_arr_from_scene(s)
                    car = next(x for x in s["sources"] if x.get("class") == "car_drive" and x.get("track", 0) == 0 and (x.get("close_slow") or x.get("d19")))
                    i_dev.append(abs(v29c.horizontal_cpa({"_mic_arr": mic_arr}, car) - float(lat_plan)))
                except Exception as e:  # noqa: BLE001
                    ng.append(f"I: {clip} D10／D19 の水平の最接近を計算できない: {e}")
        if scen in ("v19mix", "v19motofar"):
            try:
                import step11_v27_render as v27
                _check_mix_scene(s, v27)
                cnt["mix_" + scen] += 1
            except AssertionError as e:
                ng.append(f"G: {clip} {scen}: {e}")
        # H（全組）
        try:
            comp = p27.composition(row) if row.get("scenario") else ()
        except Exception:  # noqa: BLE001
            comp = ()
        if len(comp) >= 2:
            tokens = list(comp); mix = "+".join(tokens)
            if all(t == "train" for t in tokens):
                h_empty[mix] += 1                                      # 教師音源ゼロの組（電車だけ）は H の対象にしない（空集合で成立するため・Astra MIX13-04）
            else:
                all_ok, need, labeled, n_simul, no_meta = _comp_label_stats(ds, clip, tokens)
                if no_meta:
                    ng.append(f"H: {clip} metadata が無い")
                else:
                    h = hcnt[mix]; sp = row.get("split", "")
                    h["n"] += 1; h["all"] += int(all_ok); h["seeds"].add(str(row.get("seed"))); h["simul"].append(n_simul); h["source"].add("v19mix" if scen == "v19mix" else "existing")
                    if sp == "train":
                        h["n_train"] += 1; h["all_train"] += int(all_ok)
                    elif sp == "val":
                        h["n_val"] += 1; h["all_val"] += int(all_ok)
                    for ci, n in need.items():
                        htok[ci][0] += n; htok[ci][1] += min(labeled[ci], n)
        if scen == "v19bellbike":
            bt = (s.get("v25") or {}).get("bell_timing")
            if not bt:
                ng.append(f"E: {clip} ベルつき走行に v25 のベルの時刻が無い")
            else:
                bell_n += 1; bell_after += int(bt.get("side") == "after")
        for x in s.get("sources", []):
            c = x.get("class"); v24 = x.get("v24") or {}; prm = x.get("params") or {}
            cnt[c] += 1; by_cls[c].append(clip)
            if c == "siren":
                if not prm.get("v26") or v24.get("siren_chain") != "v26" or v24.get("dir_mean_db") is None or v24.get("delta_db") is None:
                    ng.append(f"B: {clip} サイレンが v26／向きの差／Δ のどれかを欠く: v26={prm.get('v26')} chain={v24.get('siren_chain')} dir={v24.get('dir_mean_db')} Δ={v24.get('delta_db')}")
            elif c == "car_drive":
                if v24.get("rule") != "car_curve":
                    ng.append(f"C: {clip} 車に v24 の規則が無い: {v24.get('rule')}")
                if prm.get("heavy"):
                    ng.append(f"C: {clip} 大型車の音源がある（D25 不採用）")
                if x.get("car_variant") == "ev_v26":
                    cnt["car_ev_v26"] += 1
                    if not x.get("avas"):
                        ng.append(f"C: {clip} EV に通報音（avas）が無い")
                elif x.get("car_variant") == "ev":
                    ng.append(f"C: {clip} EV が旧の形（ev・v26 の通報音なし）")
            elif c == "motorcycle":
                if v24.get("rule") != "bike_curve":
                    ng.append(f"C: {clip} バイクに v24 の規則が無い: {v24.get('rule')}")
            elif c in WARN:
                is_tc = bool(prm.get("train_car")) or "train_car_index" in x
                if is_tc:                                              # 列車の車両（class は crossing）は RMS のまま（AR12-08・SV26-07）
                    cnt["train_car"] += 1
                    if x.get("match_max") or v24.get("match") == "LAFmax":
                        ng.append(f"D: {clip} 列車の車両が最大値合わせになっている（RMS のはず・AR12-08）")
                elif v24.get("match") != "LAFmax" or v24.get("crest_db") is None:
                    if not x.get("preroll_s"):
                        ng.append(f"D: {clip} {c} が最大値合わせでない: match={v24.get('match')} crest={v24.get('crest_db')}")
                if c == "backup_beep" and prm.get("kind") not in ("speaker", "piezo", "chime"):
                    ng.append(f"D: {clip} バック音が v23 の 3 型でない: {prm.get('kind')}")
                if c == "horn" and not (prm.get("presses") or (x.get("grammar") or {}).get("v23_presses") or x.get("v23_presses")):
                    notes.append(f"D: {clip} クラクションに押しの列の記録が無い（grammar の形を確かめる）")
    man = ds / "plan" / "manifest_v26_bellbike.csv"
    if bell_n and man.is_file():
        rows = list(csv.DictReader(open(man, newline="", encoding="utf-8")))
        exp_after = sum(r.get("side") == "after" for r in rows)
        if bell_after != exp_after or bell_n != len(rows):
            ng.append(f"E: ベルつき走行の後の本 {bell_after}/{bell_n} が manifest の {exp_after}/{len(rows)} と違う")
    elif bell_n:
        notes.append("E: manifest_v26_bellbike.csv が置き場に無い（後の本の数は照合していない）")
    # ---- G. 本数の照合（報告書を作る前に・Astra MIX13-03）
    d31 = P_all.get("D31_MIX") or {}; d32 = P_all.get("D32_MOTOFAR") or {}; d33 = P_all.get("D33_TOPUP") or {}
    n_total_exp = int(P_all.get("N_ROWS_TOTAL_EXPECTED_V29") or P_all.get("N_ROWS_TOTAL_EXPECTED_V28") or P_all.get("N_ROWS_TOTAL_EXPECTED_V27") or -1)
    exp_mix = int(d31.get("n_clips", 0)) + (int(d33.get("n_clips", 0)) if d33.get("adopt") else 0)
    if d31.get("adopt") and len(scenes) == n_total_exp:
        if cnt["mix_v19mix"] != exp_mix:
            ng.append(f"G: v19mix の本数 {cnt['mix_v19mix']} が params の {exp_mix}（D31 {d31.get('n_clips')} ＋ D33 {d33.get('n_clips', 0) if d33.get('adopt') else 0}）と違う")
        if d32.get("adopt") and cnt["mix_v19motofar"] != int(d32["n_clips"]):
            ng.append(f"G: v19motofar の本数 {cnt['mix_v19motofar']} が params の {d32['n_clips']} と違う")
    elif d31.get("adopt") and (cnt["mix_v19mix"] or cnt["mix_v19motofar"]):
        notes.append(f"G: 部分の束（scene {len(scenes)} ≠ 総数 {n_total_exp}）なので v19mix／v19motofar の本数は params と照合していない")
    # ---- I. #70 の集計（v29 の束だけ）
    if v29c is not None and i_cnt["scenes"]:
        fh = i_cnt["car_heavy"] / max(i_cnt["car"], 1); fe = i_cnt["car_ev"] / max(i_cnt["car"], 1)
        if len(scenes) == i_total and (abs(fh - v29c.HEAVY_FRAC) > 0.02 or abs(fe - 0.15) > 0.02):
            ng.append(f"I: 車の種別の割合 重い {fh:.1%}（{v29c.HEAVY_FRAC:.0%}）・EV {fe:.1%}（15%）が 2pt より外れる")
        D2 = {k: (float(v[0]), float(v[1])) for k, v in (P_all.get("D2_RAIN_DBA") or {}).items()}
        for k, vals in i_rain.items():
            if k in D2 and (min(vals) < D2[k][0] - 0.05 or max(vals) > D2[k][1] + 0.05):
                ng.append(f"I: 雨 {k} の dba {min(vals):.1f}〜{max(vals):.1f} が params の {D2[k]} の外")
        if i_dev and max(i_dev) > 0.01:
            ng.append(f"I: D10／D19 の実際の水平の最接近と plan の横距離の差が最大 {max(i_dev):.3f} m（> 0.01）")
        # R70-05: 契約 = この束が使った params（params_snapshots の写し・無ければ md/design）と描画器の定数の一致
        try:
            # RR70-02（Astra 再監査 2026-10-19）: 生成時の params を確かめる。写しが無ければ、現物の sha が宣言と一致するときだけ現物を使う。宣言が無い・どれも違う → 未検証で NG
            import hashlib as _hl
            _sha = _params_sha_of(_gvtxt); P_used, _lbl = None, ""
            if not _sha:
                ng.append("I: 契約を確かめられない: GEN_VERSIONS に params の sha の行が無い（生成時の params が分からない・RR70-02）")
            else:
                _snap = _find_snapshot(ds, _sha)
                if _snap is not None:
                    P_used, _lbl = json.loads(_snap.read_text(encoding="utf-8")), "束の params の写し"
                else:
                    _cur = _hl.sha256(_pp.read_bytes()).hexdigest() if _pp.is_file() else ""
                    if _cur == _sha:
                        P_used, _lbl = P_all, "現物（sha が宣言と一致）"
                    else:
                        ng.append(f"I: 契約を確かめられない: 宣言 sha {_sha[:16]}… の写しが無く、現物の sha {_cur[:16]}… も違う（別の params で描いた束・RR70-02）")
            if P_used is not None:
                for mmm in v29c.contract_mismatches(P_used)[:10]:
                    ng.append(f"I: 契約（{_lbl}）: {mmm}")
        except Exception as e:  # noqa: BLE001
            ng.append(f"I: 契約の検査ができない: {e}")
        notes.append(f"I: 割合の検査は{'した' if len(scenes) == i_total else 'していない（部分の束: scene ' + str(len(scenes)) + ' ≠ 宣言 ' + str(i_total) + '）'}")
        notes.append(f"I: #70 の受け入れ（v29）: 場面 {i_cnt['scenes']:,}・車 {i_cnt['car']:,}（重い {fh:.1%}・EV {fe:.1%}）・雨の dba " + "・".join(f"{k} {min(v):.1f}〜{max(v):.1f}" for k, v in sorted(i_rain.items()))
                     + (f"・D10／D19 {len(i_dev):,} 本の水平の最接近の差 最大 {max(i_dev):.4f} m" if i_dev else "・D10／D19 なし"))
    # ---- F. 試聴の候補
    rnd = random.Random(20261012); listen = []
    for c in ("siren", "car_drive", "motorcycle", "bicycle", "bike_bell", "horn", "backup_beep", "crossing", "kick"):
        pool = sorted(set(by_cls.get(c, [])))
        for clip in (rnd.sample(pool, min(n_listen, len(pool))) if pool else []):
            lv = ""
            f = ds / "foa" / f"{clip}.flac"
            if not no_audio and f.is_file():
                try:
                    import numpy as np, soundfile as sf
                    from outdoor_seld.calibration import spl_a
                    x, sr = sf.read(str(f), dtype="float64", always_2d=True)
                    lv = f"{spl_a(x[:, 0], sr):.1f} dB(A)"
                except Exception as e:
                    lv = f"(読めない: {e})"
            listen.append((c, clip, lv))
    # ---- 報告書（全判定のあと）
    L = [f"# 描き直しの検査（{ds.name}・scene {len(scenes)} 本・2026-10-13 の道具）", "",
         f"- 判定: **{'OK' if not ng else 'NG ' + str(len(ng)) + ' 件'}**",
         f"- 音源の数: " + "・".join(f"{k} {v}" for k, v in sorted(cnt.items())),
         f"- ベルつき走行: {bell_n} 本・後に置いた本 {bell_after}", ""]
    if ng:
        L += ["## NG"] + [f"- {x}" for x in ng[:50]] + ([f"- …ほか {len(ng) - 50} 件"] if len(ng) > 50 else []) + [""]
    if notes:
        L += ["## 注意"] + [f"- {x}" for x in sorted(set(notes))[:30]] + [""]
    if hcnt:
        import statistics
        tot = sum(h["n"] for h in hcnt.values()); tot_all = sum(h["all"] for h in hcnt.values())
        thin = [(m, h) for m, h in hcnt.items() if h["all"] < 20]
        simul_all = [x for h in hcnt.values() for x in h["simul"]]
        L += ["## H: 組み合わせのゲート後（予定した音が全部ラベルに 1 フレーム以上残る本・全部の組・Opus 10/13 R-1・Astra MIX13-04）", "",
              "- 可聴ゲートは **背景（暗騒音＋雨）に対する判定**で、他の音源によるマスキングは判定しない（Astra MIX13-05）。「全部残る」= 予定した各クラスの distinct な track が metadata に 1 フレーム以上ある（同時刻とは限らない）",
              f"- 対象の組 {len(hcnt)}（v19mix の組 {sum(1 for h in hcnt.values() if 'v19mix' in h['source'])}・既存の場面だけの組 {sum(1 for h in hcnt.values() if h['source'] == {'existing'})}）・教師音源ゼロの組（電車だけ）は対象外: " + ("・".join(f"{m} {n}" for m, n in sorted(h_empty.items())) or "なし"),
              f"- 全部残る本: {tot_all:,}/{tot:,}（{tot_all / max(tot, 1):.1%}）・train {sum(h['all_train'] for h in hcnt.values()):,}/{sum(h['n_train'] for h in hcnt.values()):,}・val {sum(h['all_val'] for h in hcnt.values()):,}/{sum(h['n_val'] for h in hcnt.values()):,}",
              "- 音ごとにラベルが残る割合（予定した音源の数に対して）: " + "・".join(f"{MIX_NAME[ci]} {v[1]:,}/{v[0]:,}（{v[1] / max(v[0], 1):.0%}）" for ci, v in sorted(htok.items(), key=lambda kv: MIX_NAME[kv[0]])),
              f"- 同時に 2 つ以上の予定した音のラベルがあるフレーム数（診断値・1 本 100 フレーム）: 中央値 {statistics.median(simul_all) if simul_all else 0:.0f}・10 フレーム以上の本 {sum(1 for x in simul_all if x >= 10):,}/{len(simul_all):,}",
              f"- 全部残る本が 20 本未満の組: {len(thin)}/{len(hcnt)}（積み増し v28 の候補・本人判断・20 は train＋val の合計で独立した場面の数ではない＝seed 数を併記）", ""]
        L += [f"- {m}: {h['all']}/{h['n']}（train {h['all_train']}/{h['n_train']}・val {h['all_val']}/{h['n_val']}・seed {len(h['seeds'])}・同時フレーム中央値 {statistics.median(h['simul']) if h['simul'] else 0:.0f}・{'/'.join(sorted(h['source']))}）"
              for m, h in sorted(thin, key=lambda kv: (kv[1]["all"], kv[0]))] + [""]
        with open(ds / "REGEN_H_counts.csv", "w", newline="", encoding="utf-8") as f:            # step10_v28_plan が読む
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["mix", "source", "allocated", "all_labeled", "allocated_train", "all_labeled_train", "allocated_val", "all_labeled_val", "n_seeds", "simul_frames_median"])
            for m, h in sorted(hcnt.items()):
                w.writerow([m, "/".join(sorted(h["source"])), h["n"], h["all"], h["n_train"], h["all_train"], h["n_val"], h["all_val"], len(h["seeds"]), f"{statistics.median(h['simul']) if h['simul'] else 0:.0f}"])
    L += ["## 試聴の候補（固定 seed・クラスごと・W チャンネルの LAeq）", ""] + [f"- {c}: {clip}（{lv}）" for c, clip, lv in listen]
    (ds / "REGEN_CHECK.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6])); print(f"-> {ds / 'REGEN_CHECK.md'}")
    for x in ng[:20]:
        print("NG", x)
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
