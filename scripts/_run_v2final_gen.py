# -*- coding: utf-8 -*-
"""確定評価 v2 の最終学習データの生成ドライバ（2026-09-07・2026-09-14 に新場面 D15/D17/D18 を組み込み）。

描画は step11_v19_render 経由: 土台の場面（v11core / v12kick / v12bike / v12train）は v16 と同一（generate_clip_v13）、
新場面（scenario = v19fast / v19turn / v19platoon / v19launch）は v19 のサンプラ（v19/v20a の試作と同じ）、v19kickfar（D21・2026-09-28）は
step11_v21_render のサンプラ。v19mix（組み合わせ）と v19motofar（遠いバイク）は step11_v27_render のサンプラ（2026-10-13・#64）。雨は plan の rain 列で全部に掛かる。
違いは (1) plan = out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv（土台 ＋ 新場面）、(2) D1 SIREN_MIX を md/design/v2final_params.json から、
(3) 出力先 = out/dataset_outdoor_siren_v2final/ のみ（v16/v19 以前には書かない）。

使い方:
  python scripts/_run_v2final_gen.py --list
  python scripts/_run_v2final_gen.py --rows 0-749          # シャード生成（サーバ・params が fixed のときだけ）
  python scripts/_run_v2final_gen.py --shard 3/48          # 1 つおきの区画（行 3, 51, 99, …・重い行が偏らず同時に終わる・2026-10-20）。次の描画の sbatch はこちら
  python scripts/_run_v2final_gen.py --smoke <clip_id,...> [--smoke-fresh]  # 試し描画（置き場の版が違えば止まる・--smoke-fresh で空にして描く＝RR70-01）（別フォルダ out/dataset_outdoor_siren_v2final_smoke_tmp/・git 追跡外・status が draft でも可・本番の出力先には書かない）
  V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v26 python scripts/_run_v2final_gen.py --rows 0-1042   # 新しい出力先で描き直す（2026-10-12 凍結 r9 §2・plan は先に <出力先>/plan/ に写す）
v30（2026-10-26・#75 発効のあとの描画ドライバの直し・仕様 §4 の 1・§9-7）: 環境変数 V2FINAL_RENDER=v30 で step11_v30_render.generate_clip を呼ぶ（未設定 = v29 の道のまま）。
  V2FINAL_ROWS_LIST=<csv（clip_id 列）> で plan のうち一覧の行だけを描く（plan の順・--rows／--shard は一覧の中の番号・一覧の外の行は描かない）。
  GEN_VERSIONS に描画の道・v30 の FEATURES・帯域ゲートの線・背景の形の幅を書く（シャードの間で同じ文字になる）。一覧の名前と sha は GEN_VERSIONS ではなく
  <置き場>/ROWS_LISTS.log に足す（同じ置き場にあとで別の一覧や全量を描くとき、版の違いと取り違えない）。
  例: V2FINAL_RENDER=v30 V2FINAL_ROWS_LIST=md/design/v30_bgcmp_本の一覧_2026-10-26.csv V2FINAL_DS_NAME=dataset_outdoor_siren_v2final_v30bg_ii \
      V30_BG_SHAPE=1 V30_BG_SLOPE_RANGE=-6.5,-2.5 python scripts/_run_v2final_gen.py --shard 3/48
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v19_render as v19r  # noqa: E402
import step11_v21_render  # noqa: E402,F401  v19kickfar（D21・変更票 #42・2026-09-28）を v19 のサンプラに登録する。無いと kickfar 行は「未対応の scenario」で止まる
import step11_v22_render  # noqa: E402,F401  v19bicycle／v19bellbike／v19bellpin（D22・D23・変更票 #48/#49・2026-10-08）の登録と bicycle（番号 8）・ピン型ベルのドライ音源
import step11_v25_render  # noqa: E402,F401  v23（#60 警告音のドライ音）・v24（#61 音の大きさ）・v25（#52 大型車 v25heavy（不採用・adopt=false で動かない）・#53 ベルつき走行のベルの時刻）を重ねる（2026-10-12）
import step11_v26_render  # noqa: E402,F401  v26（#62 サイレンの形・#63 EV の接近通報音・本人 2026-10-12 18:03／18:24 採用・凍結 r9）を重ねる。サイレンの生成器は v9 の名前の差し替え（v24 の向きの差を通る）
import step11_v27_render  # noqa: E402,F401  v27（#64 D31 組み合わせの網羅 v19mix・D32 遠いバイク v19motofar・本人 2026-10-13 12:46／13:09・凍結 r10）: サンプラの登録だけ（音の作り方は変えない）
import step11_v29_render  # noqa: E402,F401  v29（#70 2026-10-19: 高さはクラス別・圏は横距離・警告音の車 1.5〜15 m・重い車 10%・歩く速さ 1.0〜1.6・隊列の下限 1.0・D10 は cs_lat_m・D19 1,600 本）。土台の差し込み口を差し替える。無いと旧の値で描く

v16r = v19r.v16r
m9 = v19r.m9
v13 = v19r.v13
PARAMS = ROOT / "md/design/v2final_params.json"
P = json.loads(PARAMS.read_text(encoding="utf-8"))
import os  # noqa: E402
RENDER = os.environ.get("V2FINAL_RENDER", "v29").strip() or "v29"          # 2026-10-26: v30 の描画（#75・仕様 §4 の 1）
assert RENDER in ("v29", "v30"), f"V2FINAL_RENDER は v29 か v30: {RENDER}"
v30r = None
if RENDER == "v30":
    import step11_v30_render as v30r  # noqa: E402  import しただけでは何も変わらない（generate_clip の中だけ v30 の直しが入る）
ROWS_LIST = os.environ.get("V2FINAL_ROWS_LIST", "").strip()                   # 2026-10-26 §9-7: 描く行の一覧（clip_id 列）・未設定 = plan の全部
# 出力先: 既定は out/dataset_outdoor_siren_v2final。音の直し（v23〜v26・凍結 r9 §2）の描き直しは **新しい出力先で全部生成する**（既存の束は消さない）ので、
# 環境変数 V2FINAL_DS_NAME（例 dataset_outdoor_siren_v2final_v26）で置き場の名前を替える。名前は dataset_outdoor_siren_v2final で始まるものだけ（基準 v12 や確定評価セットに書かない）
DS_NAME = os.environ.get("V2FINAL_DS_NAME", "dataset_outdoor_siren_v2final")
assert DS_NAME.startswith("dataset_outdoor_siren_v2final") and "/" not in DS_NAME and "\\" not in DS_NAME, DS_NAME
DS = ROOT / "out" / DS_NAME
SMOKE_DS = ROOT / "out/dataset_outdoor_siren_v2final_smoke_tmp"      # 2026-10-13 AR12-15: 試し描きは git 追跡外の置き場に（旧 _smoke は 10/7 の試し描きが追跡されている・消さない）
RENDER_CHAIN = ("step11_v9_render.py", "step11_v12_render.py", "step11_v13_render.py", "step11_v14_render.py", "step11_v16_render.py", "step11_v19_render.py",
                "step11_v21_render.py", "step11_v22_render.py", "step11_v23_render.py", "step11_v24_render.py", "step11_v25_render.py", "step11_v26_render.py", "step11_v27_render.py", "step11_v29_render.py")
if RENDER == "v30":
    RENDER_CHAIN = RENDER_CHAIN + ("step11_v30_render.py",)
IS_EVAL = "_eval_" in DS_NAME                                                      # 確定評価セット（eval v3・2026-10-13 #65）: 総数は params の EVAL_V3・学習の置き場には入れない
IS_CALIB = DS_NAME.endswith("_calib")                                              # 補正表の組（calib_fit＋calib_check・変更票 #72・2026-10-20）: 総数は md/design/calib_sets_v29.json・split = tune
CALIB_CFG = ROOT / "md/design/calib_sets_v29.json"
N_TOTAL = int(P["EVAL_V3"]["n_clips"]) if IS_EVAL else int(json.loads(CALIB_CFG.read_text(encoding="utf-8"))["n_clips"]) if IS_CALIB else int(P.get("N_ROWS_TOTAL_EXPECTED_V29") or P.get("N_ROWS_TOTAL_EXPECTED_V28") or P.get("N_ROWS_TOTAL_EXPECTED_V27") or P.get("N_ROWS_TOTAL_EXPECTED_V25") or P.get("N_ROWS_TOTAL_EXPECTED_V22") or P.get("N_ROWS_TOTAL_EXPECTED", P["N_ROWS_EXPECTED"]))   # v28（積み増し）→ v27（#64）→ v25（不採用で無い）→ v22 → 土台


def point_to(ds: Path) -> None:
    """描画モジュールの出力先を ds に向ける（v19 の生成ドライバと同じ 4 か所）"""
    m9.DS = ds
    m9.WORK = ds / "work"
    v16r.PLAN_V15 = ds / "plan"
    v19r.DS_V19 = ds
    v19r.PLAN_V19 = ds / "plan"


point_to(DS)
v13.SIREN_MIX = {k: float(v) for k, v in P["D1_SIREN_MIX"].items()}      # D1（描画時に読まれる module-global）
assert abs(sum(v13.SIREN_MIX.values()) - 1.0) < 1e-6, v13.SIREN_MIX


def load_plan(plan: Path | None = None) -> list:
    rows = []
    with open(plan or (DS / "plan" / "assignment_v2final.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"]); r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def select_rows(rows: list, list_csv: str) -> list:
    """一覧（clip_id 列）の行だけを plan の順で返す。一覧に plan に無い clip があれば止める。重複も止める。"""
    with open(list_csv, newline="", encoding="utf-8-sig") as f:
        want = [r["clip_id"] for r in csv.DictReader(f)]
    assert len(want) == len(set(want)), f"一覧に重複がある: {list_csv}"
    have = {r["clip_id"] for r in rows}
    miss = [c for c in want if c not in have]
    assert not miss, f"一覧の clip が plan に無い: {miss[:3]}（{len(miss)} 本）"
    ws = set(want)
    return [r for r in rows if r["clip_id"] in ws]


def already_done(row: dict) -> bool:
    """「同じ名前のファイルがある」だけでは飛ばさない（2026-09-14）: 作ったときの plan 行（scene.json の row）が今の行と一致し、音声・ラベル・masks が
    そろっているときだけ済みとする。10 月に雨の音の大きさや歩行比が変わった行は、ここで不一致になって作り直される。"""
    ds = m9.DS
    clip = row["clip_id"]
    p = ds / "work" / clip / "scene.json"
    if not p.is_file() or not all((ds / d / f"{clip}.{ext}").is_file() for d, ext in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv"))):
        return False
    try:
        saved = json.loads(p.read_text(encoding="utf-8")).get("row", {})
    except Exception:
        return False
    return all(_same(saved.get(k), v, k) for k, v in row.items())


# 来歴だけの列（生成条件ではない）。plan に後から足した列は旧 scene.json の row に無いので、「無い」と「空」を同じとみなす
# （2026-10-08 Astra 監査 D08-02: twin_of 列を足しただけで旧 25,500 本が未完了扱いになり、ジョブ 5735 が描き直した）。
# 生成条件の列（seed・mic_z・雨・場面など）はこれまでどおり 1 文字でも違えば作り直す。
PROVENANCE_ONLY_COLS = {"twin_of"}


def _same(saved, v, k: str) -> bool:
    if k in PROVENANCE_ONLY_COLS and (saved is None or str(saved) == "") and str(v) == "":
        return True
    return str(saved) == str(v)


def render(rows: list, tag: str) -> int:
    t0 = time.time(); done = skip = 0
    for i, row in enumerate(rows):
        if already_done(row):
            skip += 1; continue
        (v30r.generate_clip if RENDER == "v30" else v19r.generate_clip)(row)
        done += 1
        if done % 50 == 0:
            print(f"[{tag}] {i+1}/{len(rows)} done={done} skip={skip} {(time.time()-t0)/max(done,1):.1f}s/clip", flush=True)
    print(f"[{tag}] FINISHED done={done} skip={skip} {time.time()-t0:.0f}s -> {m9.DS}", flush=True)
    return done


def gen_versions_text() -> str:
    """描いた版の証跡（凍結 r9 §11・AR12-07）: 描画の連なり（step11_v9〜v26）に加えて、**読み込んだ scripts／src の全モジュール**（sys.modules から）と
    params・level_match・sound_reference の sha256。"""
    import hashlib
    rels = set(f"scripts/{n}" for n in RENDER_CHAIN) | {"md/design/v2final_params.json", "md/design/level_match_v24.json", "md/design/sound_reference_v26.json", "scripts/_run_v2final_gen.py"}
    if RENDER == "v30":                                            # 描画の中で遅れて読む部品も版に入れる（帯域ゲート・背景の形）
        rels |= {"src/outdoor_seld/band_gate.py", "src/outdoor_seld/noise_v30.py"}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        if rel.parts[0] in ("scripts", "src") and rel.suffix == ".py":
            rels.add(rel.as_posix())
    lines = [f"render chain (import order): {' -> '.join(RENDER_CHAIN)}", f"plan rows expected: {N_TOTAL}", ""]
    if RENDER == "v30":                                            # 2026-10-26: v30 の旗（シャードの間で同じ文字）
        lines[1:1] = [f"render: v30 FEATURES={json.dumps(v30r.FEATURES, sort_keys=True)} GATE_BAND_DB={v30r.GATE_BAND_DB:g} "
                      f"BG_KNEE_RANGE={list(v30r.BG_KNEE_RANGE)} BG_SLOPE_RANGE={list(v30r.BG_SLOPE_RANGE)}"]
    for rel in sorted(rels):
        f = ROOT / rel
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}" if f.is_file() else f"(missing)  {rel}")
    return "\n".join(lines) + "\n"


def smoke_version_diff(old_text: str, new_text: str) -> list:
    """試し描きの置き場にある版の記録と今の版の違い（plan rows expected の行は除く）。空なら同じ版＝再利用してよい（Astra RR70-01）。"""
    norm = lambda t: {l.strip() for l in t.splitlines() if l.strip() and not l.startswith("plan rows expected")}
    a, b = norm(old_text), norm(new_text)
    return sorted(("- " + l) for l in a - b) + sorted(("+ " + l) for l in b - a)


def write_gen_versions() -> None:
    """無いときは書く。あって中身が違えば（途中でコードを直した）、日時つきの別名で残して WARNING を出す（30 シャードが同時に通る・AR12-07）"""
    import datetime, os, time, hashlib
    text = gen_versions_text(); p = DS / "GEN_VERSIONS.txt"
    DS.mkdir(parents=True, exist_ok=True)
    try:                                                            # params の写し（Astra MIX13-02）: 版の記録の params の sha と同じ名前で残す。検査器 A が中身の差を鍵ごとに見る
        pb = PARAMS.read_bytes(); psha = hashlib.sha256(pb).hexdigest()
        snap = DS / "params_snapshots" / f"params_{psha[:16]}.json"
        # 2026-10-27（job 9342 の競合・Fable 10/26 21:5x の OK）: 無いときだけ一時ファイル → 置き換えで書き、そのあと必ず中身を確かめる
        #   （前は write_bytes で直接書き、48 シャードのうち書きかけを読んだ 2 本が止まった）。Astra RMIX14-01: 名前が合って中身が違えば止める
        from _atomic_write import ContentMismatch, write_once_atomic
        try:
            write_once_atomic(snap, pb)
        except ContentMismatch:
            raise RuntimeError(f"params の写し {snap.name} の中身が今の params と違う（取り違えか書き換え）。生成を止める")
    except OSError:
        pass
    try:
        if not p.is_file():
            tmp = DS / f"GEN_VERSIONS.txt.{os.getpid()}.tmp"            # 書きかけを他のシャードに読ませない（一時ファイル → 置き換え・job 6305 の競合の直し）
            tmp.write_text(text, encoding="utf-8")
            try:
                os.replace(tmp, p) if not p.is_file() else tmp.unlink()
            except OSError:
                tmp.unlink(missing_ok=True)
        if p.is_file() and p.read_text(encoding="utf-8") != text:
            time.sleep(1.0)                                              # 同時に始まったシャードが書き終わるのを待ってもう一度
            if p.read_text(encoding="utf-8") != text:
                q = DS / f"GEN_VERSIONS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                q.write_text(text, encoding="utf-8")
                print(f"WARNING: 描いた版が GEN_VERSIONS.txt と違う（途中でコードが変わった）→ {q.name} に残した。already_done は版を見ないので、古い版の本が混じりうる。新しい出力先で作り直すこと", flush=True)
    except OSError:
        pass


def main() -> None:
    rows = load_plan()
    assert len(rows) == N_TOTAL, (len(rows), N_TOTAL)
    if ROWS_LIST:                                                  # 2026-10-26 §9-7: 一覧の行だけ（plan の順）・一覧の外の行は描かない
        rows = select_rows(rows, ROWS_LIST)
        import hashlib as _hl
        _line = f"{ROWS_LIST}\t{_hl.sha256(Path(ROWS_LIST).read_bytes()).hexdigest()}\t{len(rows)}\n"
        print(f"rows list: {_line.strip()}（plan の順）", flush=True)
        if "--list" not in sys.argv and "--smoke" not in sys.argv:
            DS.mkdir(parents=True, exist_ok=True)
            _lg = DS / "ROWS_LISTS.log"
            if not (_lg.is_file() and _line in _lg.read_text(encoding="utf-8")):
                with open(_lg, "a", encoding="utf-8") as _f:
                    _f.write(_line)
    if "--list" in sys.argv:
        from collections import Counter
        print(f"total rows: {len(rows)} -> {m9.DS} | SIREN_MIX={v13.SIREN_MIX} | params status={P.get('status')} | chain ... -> {RENDER_CHAIN[-1]}")
        print("scenario:", dict(Counter(r["scenario"] for r in rows)))
        return
    if "--smoke" in sys.argv:
        assert not IS_EVAL and not IS_CALIB, "確定評価セット・補正表の組の置き場では試し描きをしない"
        import re
        ids = re.split(r"[,;\s]+", sys.argv[sys.argv.index("--smoke") + 1])   # sbatch --export はカンマで変数を区切るので、空白区切りも受ける
        want = {c.strip() for c in ids if c.strip()}
        sel = [r for r in rows if r["clip_id"] in want]
        assert len(sel) == len(want), f"plan に無い clip: {sorted(want - {r['clip_id'] for r in sel})}"
        point_to(SMOKE_DS)
        assert m9.DS.name == "dataset_outdoor_siren_v2final_smoke_tmp", m9.DS
        import shutil
        # RR70-01（Astra 2026-10-19）: 置き場に別の版で描いた本があるときは、古い音を新版の記録で上書きしない。同じ版の再開だけ許す。
        _gvp = SMOKE_DS / "GEN_VERSIONS.txt"; _new = gen_versions_text()
        if _gvp.is_file():
            _diff = smoke_version_diff(_gvp.read_text(encoding="utf-8"), _new)
            if _diff:
                if "--smoke-fresh" in sys.argv:
                    shutil.rmtree(SMOKE_DS); print(f"smoke: 置き場の版が今の版と違う（{len(_diff)} 行）ので --smoke-fresh で空にした", flush=True)
                else:
                    print("smoke: 置き場 " + str(SMOKE_DS) + " の版の記録が今の版と違う（古い音を新版として再利用しない・RR70-01）。--smoke-fresh で空にするか、置き場を片づけてから:\n" + "\n".join(_diff[:10]), flush=True)
                    sys.exit(3)
        elif SMOKE_DS.is_dir() and any((SMOKE_DS / "work").glob("*/scene.json")):
            if "--smoke-fresh" in sys.argv:                                      # C70-01（Astra 再々監査）: 記録が無い置き場も --smoke-fresh で空にできる
                shutil.rmtree(SMOKE_DS); print("smoke: 置き場に版の記録が無いのに本があった → --smoke-fresh で空にした", flush=True)
            else:
                print("smoke: 置き場に版の記録が無いのに本がある（版が分からない・RR70-01）。--smoke-fresh で空にするか、置き場を片づけてから", flush=True); sys.exit(3)
        (SMOKE_DS / "plan").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DS / "plan" / "assignment_v2final.csv", SMOKE_DS / "plan" / "assignment_v2final.csv")
        print(f"smoke: {len(sel)} clips -> {SMOKE_DS} (params status={P.get('status')}・本番の出力先には書かない)", flush=True)
        # 2026-10-19（R70-06）: 試し描きにも版の記録と params の写しを残す（検査 A〜I を「部分の束」として掛けられる。plan rows expected は全量の N なので割合の検査は走らない）
        import hashlib as _hl
        (SMOKE_DS / "GEN_VERSIONS.txt").write_text(_new, encoding="utf-8")
        _pb = PARAMS.read_bytes(); _snap = SMOKE_DS / "params_snapshots" / f"params_{_hl.sha256(_pb).hexdigest()[:16]}.json"
        _snap.parent.mkdir(parents=True, exist_ok=True); _snap.write_bytes(_pb)
        render(sel, "smoke")
        return
    assert m9.DS == DS and m9.DS.name == DS_NAME, (m9.DS, DS_NAME)
    if P.get("status") != "fixed":
        import os
        if os.environ.get("V2FINAL_PROVISIONAL") != "1":
            print(f"❌ 設定がまだ確定していない（status={P.get('status')}）。10 月末に実測値を入れて確定してから生成する（仮の数字で作るなら V2FINAL_PROVISIONAL=1）")
            sys.exit(2)
        # 仮の数字で作る版（2026-09-14 本人「今作り始めるのはダメ？」）: 印を残す。確定した設定の束は、この束を通し直すのではなく **新しい出力先に全部描く**（SV26-05）
        import datetime
        import hashlib
        DS.mkdir(parents=True, exist_ok=True)
        (DS / "PROVISIONAL.txt").write_text(
            f"仮の数字（status={P.get('status')}）で生成した版。{datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"params sha256 {hashlib.sha256(PARAMS.read_bytes()).hexdigest()}\n"
            "本番の学習に使う前に、実測値を入れて確定し、生成器をもう一度通して（設定が変わった行だけ作り直される）この印を消す。\n", encoding="utf-8")
        print("PROVISIONAL: 仮の数字で生成する（印 PROVISIONAL.txt を置いた）", flush=True)
    else:
        if (DS / "PROVISIONAL.txt").is_file():
            # 仮の版の束は昇格しない（Astra SV26-05・2026-10-13）: 確定した設定の束は **新しい空の出力先**に全部描く。印のある束に fixed で描き足すと、
            # 旧版の本の再利用と新版の本が混じる（already_done は版を見ない）。印を消す処理は無くした
            print(f"❌ 出力先 {DS} に仮の版の印 PROVISIONAL.txt がある。確定した設定の束は新しい出力先（V2FINAL_DS_NAME）で全部描く（昇格しない）", flush=True)
            sys.exit(3)
    write_gen_versions()                          # status の検査のあと（止まった投入では証跡を残さない・AR12-07）
    lo, hi = 0, len(rows) - 1
    if "--shard" in sys.argv:
        # 2026-10-20（job 6986 の教訓・本人「今後もこの速さで」）: 行の塊で切ると重い行（混ぜ合わせ・D19）が後ろの区画に偏り、最後の 1 区画だけ 1 時間以上残る。
        # --shard i/n は行を 1 つおきに配る（i, i+n, i+2n, …）ので、どの区画も同じ重さになり同時に終わる。出来上がりは --rows と同じ（行ごとに決定的・already_done も行ごと）。
        i_, n_ = sys.argv[sys.argv.index("--shard") + 1].split("/")
        i_, n_ = int(i_), int(n_)
        assert 0 <= i_ < n_, f"--shard は 0〜n-1/n: {i_}/{n_}"
        render(rows[i_::n_], f"shard{i_}/{n_}")
        return
    if "--rows" in sys.argv:
        a, b = sys.argv[sys.argv.index("--rows") + 1].split("-")
        lo, hi = int(a), int(b)
    render(rows[lo:hi + 1], f"{lo}-{hi}")


if __name__ == "__main__":
    main()
