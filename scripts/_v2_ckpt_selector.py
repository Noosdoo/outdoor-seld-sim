# -*- coding: utf-8 -*-
"""確定評価 v2 の ckpt 選択器（2026-09-14・仕様 = md/audit/v2_ckpt選択器_実装前仕様レビュー_2026-09-11.md をそのまま実装）。

何を選ぶか: 因果 ft の候補 epoch {19,39,59,79,99,119,139,149} のうち、開発 val（固定 manifest・高さ入力 1.5 m 固定・水平距離 GT・frame 40〜99・
距離クラス {4,6,7,8}〔2.0 で bicycle を足した・変更票 #55〕）で **GT 距離 ≤ 1.5 m の対応ペアだけ**の相対距離誤差の中央値
（val の本数は --expected で渡す: v29（2026-10-20）は 12,420 本〔plan の split=val〕= 既定・旧 v12 fold2（1,800 本）を読むときは --expected 1800 を渡す〔互換用〕・凍結 r12 §4・Astra RN20-09）

    E_e = median_{(g,p) ∈ M_e} |d_p − d_g| / max(d_g, 0.1)

が最小の epoch。同率は未丸めの float64 で比べ、同値なら至近捕捉数 C_e（対応した予測も ≤1.5 m だった全 GT 分母の件数）が大きい方、
それも同値なら後の epoch（辞書順 (E_e, −C_e, −epoch)）。**検出率の門（本人決定 2026-09-14 14:47）**: 至近の検出率 D_e/N が
「8 候補の中の最良 − 2pt」を下回る候補は順位から外す（検出を落として難しい例を避けた候補が勝つ抜け道を塞ぐ。基準は候補集合の中なので
外部モデルに依らない）。`--no-coverage-gate` は試走用（report_only）で、登録の選定には使えない。

対応付け: 各 clip・各 frame で、まず**全距離帯**の GT と予測から同クラスの候補ペアを作り、円周方位差 ≤ 20 度のものを方位差の小さい順に
貪欲に 1 対 1 で採る（現行の統一採点器と同じ規則）。同率の順序は入力を (frame, class, track, az, dist) の辞書順に正規化してから
(方位差, GT index, 予測 index) で固定する。対応を作った**後**に GT 距離 ≤1.5 m のペアを抜き出す（先に近い GT だけで対応させない）。
既存の `_score_unified.aggregate` の dist_err（全距離帯の中央値）は使わない。

入力の契約: 全候補の来歴（.csv / .done.json / .run_meta.json）を `_prediction_integrity.verify_download` で照合し、epoch・高さ設定・
clip 集合（manifest と一致）も検査する。1 候補でも欠ける・照合できなければ選定を停止する（成績不良とは区別）。fold20 は受け付けない。
補正表・LAT_GATE は読まない（このモジュールは統一採点器を import しない）。

使い方:
  python scripts/_v2_ckpt_selector.py --pattern "out/v2final/C/infer_v2finalft_e{epoch:03d}_v12causal.csv" \
      --manifest out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv --gt out/dataset_outdoor_siren_v12/metadata_dist_h \
      --out out/audit_rescore/v2final_select.json
  試走（候補集合を変える・登録には使えない）: --epochs 99,119,139,149 --label 試走
出力: JSON（仕様 version・各 hash・条件・各候補の N,D,C,E・順位・選定 epoch）と、JSON から生成した md。どちらも原子的に書く。
**version 2.0（2026-10-12・変更票 #55・凍結 r9 §4）**: 指標の距離クラスを {4,6,7,8}（bicycle を足す）に。数式・候補・門・契約は 1.1 のまま。クラス別の N/D/C/対応数と D に占める割合を併記（順位には使わない）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPEC_VERSION = "v2_ckpt_selector/2.2 (2026-10-26; 変更票 #75・凍結 r16 §4 = 選ぶ指標 E の帯を --close-m で ≤3.2 m（至近＋注意）に広げられる。門 N/D/C は ≤1.5 m のまま。至近だけの E_close・注意だけの E_mid・『いつも同じ値を出す行』を併記。--close-m を渡さなければ 2.1 と同じ値; 2.1 = 2026-10-13; 変更票 #65・凍結 r11 §4 = 検証セットを最新の val に替えるため --height-table で『表あり（sha256 照合）・既定 1.5 m・offset 0』の高さ設定も受理（表なしの 2.0 の契約も残す）; 2.0 = 2026-10-12; 変更票 #55・凍結 r9 §4 = 距離クラスに bicycle 8 を足し、クラス別の N/D/C/対応数を併記; 1.1 = 2026-09-14 spec md/audit/v2_ckpt選択器_実装前仕様レビュー_2026-09-11.md; coverage gate 2pt decided 2026-09-14 14:47)"
COVERAGE_GATE_DEFAULT = 2.0   # 本人決定 2026-09-14 14:47「門 2pt で」: 至近の検出率 D/N が最良候補 − 2pt を下回る候補は選ばない
EPOCHS = (19, 39, 59, 79, 99, 119, 139, 149)
DIST_CLS = frozenset({4, 6, 7, 8})   # 選ぶ指標の距離クラス（car/kick/bike + bicycle 8。2026-10-12 変更票 #55・凍結 r9 §4・本人 10/8 21:19「常に最新のバージョンに対応」。
                                     # 1.1 までは {4,6,7}（Astra D08-06 で据え置き）。旧 version の JSON と両方残す）
CLS_NAME = {4: "car", 6: "kick", 7: "bike", 8: "bicycle"}
MAX_CLASS = 8                      # 受理するクラス番号の上限（9 クラス化・2026-10-08 変更票 #48。8 = bicycle は 2.0 で指標に入る）
AZ_PAIR = 20.0
TH_CLOSE = 1.5                     # 門（N/D/C）の帯。至近 ≤1.5 m（変えない・#75 ⑬ 本人 10/26「門は ≤1.5 のまま」）
TH_E_SPEC = 3.2                    # 変更票 #75（2026-10-26・⑬）: 選ぶ指標 E の帯の上限（至近＋注意）。選択器 2.2 では --close-m で渡す（既定は旧の 1.5）
FLOOR = 0.1
FRAME_MIN, FRAME_MAX = 40, 99
COVERAGE_POLICY = f"gate: D/N >= best - {COVERAGE_GATE_DEFAULT}pt"   # 登録に使う設定。--no-coverage-gate（report_only）は試走用
CONDITIONS = dict(frames=[FRAME_MIN, FRAME_MAX], dist_classes=sorted(DIST_CLS), az_pair_deg=AZ_PAIR, gt_close_m=TH_CLOSE,
                  denominator_floor_m=FLOOR, distance="horizontal (no cos conversion)", height="table none / default 1.5 m / offset 0",
                  correction_table="none", coverage_policy=COVERAGE_POLICY,
                  tie_order="rows normalized by (frame, class, track, az, dist); candidates sorted by (az_diff, gt_index, pred_index)")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dang(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class InputError(ValueError):
    """入力の契約違反（成績不良ではなく、選定を止める理由）"""


# ---------------------------------------------------------------- 読み込み（厳格）
def read_manifest(path: Path) -> list[str]:
    clips = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
        if header[0] != "clip_id":
            raise InputError(f"manifest の先頭列が clip_id でない: {path}")
        for line in f:
            g = line.strip().split(",")
            if g and g[0]:
                clips.append(g[0])
    if not clips or len(set(clips)) != len(clips):
        raise InputError("manifest が空か clip_id が重複している")
    return clips


def _row(g, where, allow_nan_dist_cls):
    """(frame, cls, track, az, el, dist) を検査して返す。dist は非距離クラスなら NaN でよい。"""
    try:
        k, c, t = int(g[0]), int(g[1]), int(g[2])
        az, el = float(g[3]), float(g[4])
        d = float("nan") if g[5].strip() in ("", "nan", "NaN") else float(g[5])
    except (ValueError, IndexError) as e:
        raise InputError(f"{where}: 数値が読めない: {g}") from e
    if not (0 <= k <= 99) or not (0 <= c <= MAX_CLASS) or not (0 <= t <= 2):
        raise InputError(f"{where}: frame/class/track の範囲外: {g}")
    if not (math.isfinite(az) and -180.0 <= az <= 180.0 and math.isfinite(el) and -90.0 <= el <= 90.0):
        raise InputError(f"{where}: 方位/仰角が不正: {g}")
    if math.isinf(d) or (math.isfinite(d) and d < 0):
        raise InputError(f"{where}: 距離が負か無限大: {g}")
    if c in DIST_CLS and math.isnan(d) and not allow_nan_dist_cls:
        raise InputError(f"{where}: GT の距離が NaN: {g}")
    return (k, c, t, az, el, d)


def read_gt(gt_dir: Path, clips: list[str]) -> dict[str, list[tuple]]:
    out = {}
    for clip in clips:
        f = gt_dir / f"{clip}.csv"
        if not f.is_file():
            raise InputError(f"GT が無い: {f}")
        rows, seen = [], set()
        with open(f, encoding="utf-8") as h:
            for ln, line in enumerate(h, 1):
                g = line.strip().split(",")
                if len(g) == 1 and not g[0]:
                    continue
                if len(g) != 6:
                    raise InputError(f"GT {clip}:{ln}: 6 列でない")
                r = _row(g, f"GT {clip}:{ln}", allow_nan_dist_cls=False)
                key = r[:3]
                if key in seen:
                    raise InputError(f"GT {clip}:{ln}: (frame, class, track) の重複 {key}")
                seen.add(key)
                rows.append(r)
        out[clip] = rows
    return out


def read_pred(csv_path: Path, clips: list[str]) -> tuple[dict[str, list[tuple]], dict]:
    """来歴つき 7 列 csv（clip,frame,cls,track,az,el,dist）だけを読む。6 列の行・未知 clip は拒否。
    戻り = (clip→rows, {"n_nan_dist": 距離クラスの NaN 件数, "n_dup_rows": 完全に同じ行の件数})

    仕様（§2「重複行は拒否」）からの差: 現行の推論出力は track が常に 0（`_causal_infer_v17.py`: [frame,class,track=0,az,el,dist]）で、
    同じ (frame, class) に複数の予測（ADPIT の 3 track）が並ぶ。(clip, frame, class, track) は予測の識別子にならないので重複判定に使わない。
    値まで完全に同じ行は ADPIT の 2 track が同じ出力を出したもので、統一採点器と同じくそのまま残し（拒否しない）、件数を記録する。"""
    ids = set(clips)
    out = defaultdict(list)
    seen = set()
    n_nan = n_dup = 0
    with open(csv_path, encoding="utf-8") as h:
        for ln, line in enumerate(h, 1):
            s = line.strip()
            if not s:
                continue
            g = s.split(",")
            if len(g) != 7:
                raise InputError(f"{csv_path.name}:{ln}: 7 列でない（{len(g)} 列）")
            if g[0] not in ids:
                raise InputError(f"{csv_path.name}:{ln}: manifest に無い clip {g[0]}")
            r = _row(g[1:], f"{csv_path.name}:{ln}", allow_nan_dist_cls=True)
            if s in seen:
                n_dup += 1
            seen.add(s)
            if r[1] in DIST_CLS and math.isnan(r[5]):
                n_nan += 1
            out[g[0]].append(r)
    return out, dict(n_nan_dist=n_nan, n_dup_rows=n_dup)


# ---------------------------------------------------------------- 対応と数式
def match_frame(gg: list[tuple], pp: list[tuple]) -> list[tuple[tuple, tuple]]:
    """1 frame 分。gg/pp は (frame, cls, track, az, el, dist) の正規化済みリスト。戻り = [(gt_row, pred_row)]"""
    cand = []
    for gi, g in enumerate(gg):
        for pi, p in enumerate(pp):
            if p[1] == g[1]:
                e = dang(g[3], p[3])
                if e <= AZ_PAIR:
                    cand.append((e, gi, pi))
    cand.sort()
    used_g, used_p, pairs = set(), set(), []
    for e, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        pairs.append((gg[gi], pp[pi]))
    return pairs


def _norm(rows):
    """(frame, cls, track, az, dist) の辞書順（el は並べ替えに使わない・仕様 §2）"""
    return sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3], r[5]))


def measure(gt_rows: list[tuple], pred_rows: list[tuple], e_close: float = TH_CLOSE) -> dict:
    """1 clip（または小さな模擬例）の集計。戻り: N, D, C, rels（至近 ≤TH_CLOSE のペアの相対誤差）, all_rels, n_pairs,
    rels_e（GT ≤ e_close のペア = 選ぶ指標 E の帯・2.2）, rels_mid（TH_CLOSE < GT ≤ e_close）。e_close = TH_CLOSE なら rels_e == rels。"""
    gts = _norm(r for r in gt_rows if FRAME_MIN <= r[0] <= FRAME_MAX and r[1] in DIST_CLS)
    prs = _norm(r for r in pred_rows if FRAME_MIN <= r[0] <= FRAME_MAX and r[1] in DIST_CLS and math.isfinite(r[5]))
    by_g, by_p = defaultdict(list), defaultdict(list)
    for r in gts:
        by_g[r[0]].append(r)
    for r in prs:
        by_p[r[0]].append(r)
    per = {c: dict(N=0, D=0, C=0, n_pairs=0) for c in sorted(DIST_CLS)}      # クラス別（併記だけ・順位には使わない・#55）
    N = 0
    for r in gts:
        if r[5] <= TH_CLOSE:
            N += 1; per[r[1]]["N"] += 1
    D = C = 0
    rels, all_rels, rels_e, rels_mid = [], [], [], []
    for k in sorted(by_g):
        for g, p in match_frame(by_g[k], by_p.get(k, [])):
            er = abs(p[5] - g[5]) / max(g[5], FLOOR)
            all_rels.append(er); per[g[1]]["n_pairs"] += 1
            if g[5] <= e_close:                                            # 2.2: 選ぶ指標 E の帯（既定 = 至近と同じ）
                rels_e.append(er)
                if g[5] > TH_CLOSE:
                    rels_mid.append(er)
            if g[5] <= TH_CLOSE:
                rels.append(er); D += 1; per[g[1]]["D"] += 1
                if p[5] <= TH_CLOSE:
                    C += 1; per[g[1]]["C"] += 1
    return dict(N=N, D=D, C=C, rels=rels, all_rels=all_rels, n_pairs=len(all_rels), per_class=per,
                rels_e=rels_e, rels_mid=rels_mid, e_close=e_close)


def summarize(parts: list[dict]) -> dict:
    """clip ごとの measure() を束ねて候補 1 つの指標に。E は選ぶ指標の帯（≤ e_close）の全ペアを 1 つの集合にした中央値（偶数個は中央 2 値の平均）。
    E_close = 至近 ≤TH_CLOSE だけ・E_mid = TH_CLOSE〜e_close だけ（併記）。e_close = TH_CLOSE（既定）なら E == E_close（2.1 と同じ値）。"""
    rels = [v for s in parts for v in s["rels"]]
    all_rels = [v for s in parts for v in s["all_rels"]]
    rels_e = [v for s in parts for v in s.get("rels_e", s["rels"])]
    rels_mid = [v for s in parts for v in s.get("rels_mid", [])]
    N, D, C = (sum(s[k] for s in parts) for k in ("N", "D", "C"))
    per = {c: {k: sum(s["per_class"][c][k] for s in parts) for k in ("N", "D", "C", "n_pairs")} for c in sorted(DIST_CLS)}
    for c in per:                                                          # 至近の対応 D に占めるクラスの割合（bicycle の少なさを見える化・#55）
        per[c]["share_of_D"] = (per[c]["D"] / D) if D else None
    return dict(N=N, D=D, C=C, E=(statistics.median(rels_e) if rels_e else None),
                E_close=(statistics.median(rels) if rels else None), E_mid=(statistics.median(rels_mid) if rels_mid else None),
                n_pairs_e=len(rels_e), n_pairs_mid=len(rels_mid),
                all_E=(statistics.median(all_rels) if all_rels else None), n_pairs=len(all_rels),
                coverage=(D / N if N else None), capture=(C / N if N else None), per_class=per)


def constant_reference(gt: dict, clips: list[str], e_close: float) -> dict:
    """「いつも同じ値を出す行」（Astra P25-03・#75 ⑬）: 予測を使わず、正解だけから作る参考値。
    帯（≤ e_close と ≤ TH_CLOSE）の正解の距離の中央値 c を全部の正解に出したときの相対誤差の中央値。候補の E がこれに負けたら指標が弱い印。"""
    rows = [r for c in clips for r in gt.get(c, []) if FRAME_MIN <= r[0] <= FRAME_MAX and r[1] in DIST_CLS]
    out = {}
    for name, th in (("e_band", e_close), ("close", TH_CLOSE)):
        ds = [r[5] for r in rows if r[5] <= th]
        if not ds:
            out[name] = dict(th_m=th, n=0, const_m=None, E=None); continue
        cst = statistics.median(ds)
        out[name] = dict(th_m=th, n=len(ds), const_m=cst, E=statistics.median(abs(cst - d) / max(d, FLOOR) for d in ds))
    return out


def rank(metrics: dict[int, dict], epochs=EPOCHS, coverage_gate_pt: float | None = None) -> tuple[list[int], int | None, str]:
    """metrics: epoch -> summarize() の dict（'verified' が False の候補があれば停止）。戻り = (順位, 選定 epoch or None, 理由)

    coverage_gate_pt（任意・仕様 §3 の「検出率の門」）: None なら report_only（検出率は併記だけ）。値を渡すと、有効候補のうち
    至近の検出率 D/N が「最良候補の検出率 − gate_pt」を下回る候補を順位から外す（検出を落として難しい例を避けた候補が勝つ抜け道を塞ぐ）。
    基準は候補集合の中の最良なので外部モデルに依らない。外した候補には invalid_reason を書く。最良候補は決して外れない。"""
    if set(metrics) != set(epochs):
        raise InputError(f"候補が欠けている: 期待 {sorted(epochs)} / 実際 {sorted(metrics)}")
    if not all(m.get("verified", True) for m in metrics.values()):
        raise InputError("来歴の照合を通らない候補がある")
    ns = {m["N"] for m in metrics.values()}
    if len(ns) != 1 or next(iter(ns)) == 0:
        raise InputError(f"至近 GT の分母 N が候補間で違うか 0: {sorted(ns)}")
    valid = [e for e, m in metrics.items() if m["E"] is not None]
    rule = "lexicographic (E, -C, -epoch)"
    if coverage_gate_pt is not None and valid:
        N = next(iter(ns))
        best_cov = max(metrics[e]["D"] / N for e in valid)
        floor = best_cov - coverage_gate_pt / 100.0
        kept = []
        for e in valid:
            cov = metrics[e]["D"] / N
            if cov + 1e-12 < floor:
                metrics[e]["invalid_reason"] = f"検出率の門: D/N={100 * cov:.4f}% < 最良 {100 * best_cov:.4f}% − {coverage_gate_pt}pt"
            else:
                kept.append(e)
        valid = kept
        rule = f"coverage gate (D/N ≥ best − {coverage_gate_pt}pt) then lexicographic (E, -C, -epoch)"
    order = sorted(valid, key=lambda e: (metrics[e]["E"], -metrics[e]["C"], -e))
    if not order:
        return [], None, "全候補が選択不可（至近対応ペア 0）→ 選定結果なし・fold20 に進まない"
    return order, order[0], rule


# ---------------------------------------------------------------- 来歴の照合
def check_height_table(height_table: Path, clips: list[str]) -> dict:
    """Astra RMIX14-02（2026-10-14・Opus）: 高さの表（clip_id, mic_z）が、評価 manifest の全部の本を 1 行ずつ・有限の正の値で持つかを確かめる。
    表から落ちた本は推論で既定の高さ（height_patch の DEFAULT）に回るので、登録に使う候補では欠け・重複・数でない値を許さない。
    Astra O14-03（2026-10-14・Opus）: 実行時（server_sde/height_patch.py の load_table／lookup）は clip_id の**元の文字列そのまま**を鍵にし、
    音のファイル名の stem で引く。検査も同じ規則にする（空白・拡張子・パス付きの行は、実行時には当たらず既定の高さになるので拒否）。"""
    import csv as _csv, math as _math
    rows, alias = defaultdict(list), defaultdict(list)
    with open(height_table, newline="", encoding="utf-8") as f:
        rd = _csv.DictReader(f)
        if not rd.fieldnames or "clip_id" not in rd.fieldnames or "mic_z" not in rd.fieldnames:
            raise InputError(f"高さの表 {height_table.name} に clip_id と mic_z の列が無い: {rd.fieldnames}")
        for r in rd:
            raw = r.get("clip_id") or ""
            rows[raw].append(r.get("mic_z") or "")
            near = Path(raw.strip()).stem
            if near != raw:
                alias[near].append(raw)                              # 実行時には当たらない書き方（空白・拡張子・パス付き）
    want = set(clips); missing = sorted(c for c in want if c not in rows); dup = sorted(c for c in want if len(rows.get(c, [])) > 1)
    ali = sorted(c for c in want if alias.get(c))                    # 正しい行があっても、同じ本の別の書き方の行は表の取り違えとみなして拒否
    bad = []
    for c in sorted(want - set(missing)):
        v = rows[c][0]
        try:
            z = float(v) if v else float("nan")                      # 実行時は mic_z が空の行を飛ばして既定の高さにする → 空も拒否
            if not (_math.isfinite(z) and z > 0):
                bad.append(c)
        except ValueError:
            bad.append(c)
    if missing or dup or bad or ali:
        raise InputError(f"高さの表 {height_table.name} が評価 manifest の本を覆っていない: 欠け {len(missing)}（{missing[:3]}）・重複 {len(dup)}（{dup[:3]}）・数でない／0 以下 {len(bad)}（{bad[:3]}）"
                         f"・空白・拡張子・パス付きの書き方 {len(ali)}（{[(c, alias[c][:2]) for c in ali[:3]]}・実行時は clip_id の文字列そのままで引くので当たらない）")
    return {"n_rows_table": sum(len(v) for v in rows.values()), "n_clips_covered": len(want)}


def verify_candidate(integrity, csv_path: Path, epoch: int, clips: list[str], expected: int, height_table: Path | None = None, height_none: bool = False) -> dict:
    done = csv_path.with_suffix(".done.json")
    meta = csv_path.with_suffix(".run_meta.json")
    for p in (csv_path, done, meta):
        if not p.is_file():
            raise InputError(f"候補 e{epoch:03d} の来歴が無い: {p}")
    integrity.verify_download(csv_path, expected, done, meta)
    m = json.loads(meta.read_text(encoding="utf-8"))
    if m["ckpt_epoch"] != epoch:
        raise InputError(f"候補 e{epoch:03d}: run_meta の ckpt_epoch={m['ckpt_epoch']}")
    h = m.get("height", {})
    if height_none:                                             # 変更票 #74（アブレーション noheight）: 高さの入力なし（HEIGHT_COND=0）で推論した候補だけを受ける
        if h.get("on") is not False or (h.get("table") or "") != "":
            raise InputError(f"候補 e{epoch:03d}: --height-none なのに高さの入力が付いている: {h}")
    elif height_table is None:                                  # 2.0 の契約: 表なし・既定 1.5 m・offset 0（v12 val）
        if not (h.get("on") is True and (h.get("table") or "") == "" and float(h.get("default", -1)) == 1.5 and float(h.get("offset", 1)) == 0.0):
            raise InputError(f"候補 e{epoch:03d}: 高さ設定が『表なし・既定 1.5 m・offset 0』でない: {h}")
    else:                                                       # 2.1 の契約（r11）: 表あり = 推論に渡した高さの表が --height-table と同じ中身（sha256）・offset 0
        if not (h.get("on") is True and (h.get("table") or "") != "" and h.get("table_sha256") == sha256_file(height_table) and float(h.get("offset", 1)) == 0.0
                and float(h.get("default", -1)) == 1.5):     # Astra RMIX14-02: 表示（default 1.5 m）と契約を合わせる
            raise InputError(f"候補 e{epoch:03d}: 高さ設定が『表あり（{height_table.name} と同じ sha256）・既定 1.5 m・offset 0』でない: {h}")
        check_height_table(height_table, clips)              # Astra RMIX14-02: 表が評価の本を全部・1 行ずつ・有限の正の値で持つ
    if {Path(c).stem for c in m["clips"]} != set(clips):
        raise InputError(f"候補 e{epoch:03d}: 推論 manifest が評価 manifest と一致しない")
    if any(Path(c).stem.startswith(("fold20", "eval_syn", "eval_field")) for c in m["clips"]):
        raise InputError("確定評価セット（eval_syn・旧 fold20）や実録（eval_field）は選択器の入力にできない")
    d = json.loads(done.read_text(encoding="utf-8"))
    return dict(csv=str(csv_path), csv_sha256=sha256_file(csv_path), done_sha256=sha256_file(done), meta_sha256=sha256_file(meta),
                ckpt=m.get("ckpt"), ckpt_sha256=m["ckpt_sha256"], n_rows=d.get("n_rows"), identity_sha256=d.get("identity_sha256"))


def gt_content_hash(gt_dir: Path, clips: list[str]) -> str:
    h = hashlib.sha256()
    for clip in sorted(clips):
        h.update(f"{clip}:{sha256_file(gt_dir / f'{clip}.csv')}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------- 出力
def to_markdown(res: dict) -> str:
    L = [f"# 確定評価 v2 の ckpt 選択（{res['label']}・{res['spec_version']}）", "",
         f"- 選定 epoch: **{res['selected'] if res['selected'] is not None else 'なし'}**（規則: {res['rule']}）"
         + ("　**［候補集合が仕様と違う試走・登録には使えない］**" if res["not_for_registration"] else ""),
         f"- 順位: {' > '.join(f'e{e:03d}' for e in res['order']) if res['order'] else '-'}",
         f"- 条件: frame {res['conditions']['frames'][0]}〜{res['conditions']['frames'][1]}・距離クラス {res['conditions']['dist_classes']}・方位差 ≤{res['conditions']['az_pair_deg']}°・"
         f"GT 距離 ≤{res['conditions']['gt_close_m']} m・分母下限 {res['conditions']['denominator_floor_m']} m・{res['conditions']['distance']}・高さ {res['conditions']['height']}・補正表 {res['conditions']['correction_table']}・coverage_policy = {res['conditions']['coverage_policy']}",
         f"- 入力: manifest {res['manifest']}（{res['n_clips']} 本・sha256 {res['manifest_sha256'][:16]}）・GT {res['gt_dir']}（内容 hash {res['gt_sha256'][:16]}）・code {res['code_sha256']['selector'][:16]} / integrity {res['code_sha256']['integrity'][:16]}",
         "", f"| epoch | 照合 | N（至近 GT） | D（対応） | 検出率 D/N | C（捕捉） | 捕捉率 C/N | **E（選ぶ帯 ≤{res['conditions']['gt_close_m']} m の相対誤差・%）** | E 至近 ≤{res['conditions'].get('gate_close_m', TH_CLOSE)} m（%） | E 注意 {res['conditions'].get('gate_close_m', TH_CLOSE)}〜{res['conditions']['gt_close_m']} m（%） | 全帯の相対誤差（参考・%） | 対応ペア | 距離 NaN | 同値の行 | 不可の理由 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pct = lambda v: "-" if v is None else f"{100 * v:.4f}"
    for e in sorted(res["candidates"], key=int):
        m = res["candidates"][e]
        L.append(f"| e{int(e):03d} | {'OK' if m['verified'] else 'NG'} | {m['N']} | {m['D']} | {pct(m['coverage'])} | {m['C']} | {pct(m['capture'])} | "
                 f"**{pct(m['E'])}** | {pct(m.get('E_close'))} | {pct(m.get('E_mid'))} | {pct(m['all_E'])} | {m['n_pairs']} | {m['n_nan_dist']} | {m['n_dup_rows']} | {m.get('invalid_reason') or ''} |")
    rc = res.get("reference_constant")
    if rc:
        L.append("")
        L.append("いつも同じ値を出す行（参考・予測を使わず正解だけから・#75 ⑬・Astra P25-03）: "
                 + "・".join(f"{'選ぶ帯' if k == 'e_band' else '至近'} ≤{v['th_m']} m: 定数 {v['const_m']:.3f} m → E {pct(v['E'])}（正解 {v['n']} 行）" if v.get("E") is not None
                             else f"{'選ぶ帯' if k == 'e_band' else '至近'} ≤{v['th_m']} m: 正解なし" for k, v in rc.items()))
    L += ["", "クラス別（併記だけ・順位には使わない・変更票 #55）: 至近 GT N／対応 D／捕捉 C／対応ペア（全帯）・D に占める割合"]
    for e in sorted(res["candidates"], key=int):
        m = res["candidates"][e]; pc = m.get("per_class") or {}
        parts = []
        for c in sorted(pc, key=int):
            q = pc[c]; sh = q.get("share_of_D")
            parts.append(f"{CLS_NAME.get(int(c), c)}({c}) N {q['N']} / D {q['D']} / C {q['C']} / ペア {q['n_pairs']}" + (f" / {100 * sh:.1f}%" if sh is not None else ""))
        L.append(f"- e{int(e):03d}: " + "・".join(parts))
    if all(((m.get("per_class") or {}).get("8") or (m.get("per_class") or {}).get(8) or {}).get("N", 0) == 0 for m in res["candidates"].values()):
        L.append("- 注記: この検証セットには bicycle（8）の至近の正解が無い（N = 0）。version 2.0 の {4,6,7,8} と 1.1 の {4,6,7} は同じ順位になる。bicycle を指標に効かせるには 9 クラスの検証セット（v22／v26 の val）が要る（AR12-16）")
    L += ["", "E・順位は未丸めの float64 で決めている（表示は 4 桁）。検出率・捕捉率は併記だけで門にしない（report_only）。",
          "至近ペアだけの誤差は未検出を直接罰さない: 検出を落として難しい例を避けた候補が勝ち得る（仕様 §3・受け入れ試験の『条件付き指標の弱点』）。", ""]
    L.append("## 候補の来歴")
    L.append("")
    L.append("| epoch | csv sha256 | ckpt sha256 | identity | 行数 |")
    L.append("| --- | --- | --- | --- | --- |")
    for e in sorted(res["candidates"], key=int):
        m = res["candidates"][e]
        L.append(f"| e{int(e):03d} | {m['csv_sha256'][:16]} | {m['ckpt_sha256'][:16]} | {m['identity_sha256'][:16]} | {m['n_rows']} |")
    return "\n".join(L) + "\n"


def atomic_write_all(files: dict[Path, str]) -> None:
    tmps = []
    try:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmps.append((tmp, path))
        for tmp, path in tmps:
            os.replace(tmp, path)
    except Exception:
        for tmp, _ in tmps:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def run(pattern: str, manifest: Path, gt_dir: Path, out_json: Path, epochs=EPOCHS, label="v2final", expected=12420,
        coverage_gate: float | None = None, height_table: Path | None = None, height_none: bool = False, e_close: float = TH_CLOSE) -> dict:
    integrity = _load("prediction_integrity", ROOT / "server_sde/_prediction_integrity.py")
    clips = read_manifest(manifest)
    if len(clips) != expected:
        raise InputError(f"manifest の本数 {len(clips)} ≠ 期待 {expected}")
    gt = read_gt(gt_dir, clips)
    if height_none and height_table is not None:
        raise InputError("--height-none と --height-table は同時に渡せない")
    if height_table is not None:
        check_height_table(height_table, clips)                # Astra RMIX14-02: 候補を読む前に一度（全候補で同じ表なので無駄な読み込みを避ける）
    cands, metrics = {}, {}
    for e in epochs:
        csv_path = Path(pattern.format(epoch=e))
        prov = verify_candidate(integrity, csv_path, e, clips, expected, height_table, height_none)     # 1 候補でも通らなければここで停止（出力は書かない）
        pred, counts = read_pred(csv_path, clips)
        s = summarize([measure(gt[c], pred.get(c, []), e_close) for c in clips])
        s.update(verified=True, **counts, invalid_reason=(None if s["E"] is not None else "選ぶ帯の対応ペアが 0（選択不可）"))
        cands[e] = {**s, **prov}
        metrics[e] = s
    order, selected, rule = rank(metrics, epochs, coverage_gate)
    for e in epochs:                                   # 門で外した理由を候補の表にも写す
        cands[e]["invalid_reason"] = metrics[e].get("invalid_reason")
    conditions = dict(CONDITIONS, coverage_policy=("report_only (no gate; not for registration)" if coverage_gate is None else f"gate: D/N >= best - {coverage_gate}pt"),
                      height=("no height input (ablation noheight, #74)" if height_none else ("table none / default 1.5 m / offset 0" if height_table is None else f"table {height_table.name} sha256 {sha256_file(height_table)[:16]} / default 1.5 m / offset 0")),
                      gt_close_m=e_close, gate_close_m=TH_CLOSE)        # 2.2: gt_close_m = 選ぶ指標 E の帯・gate_close_m = 門 N/D/C の帯（#75 ⑬）
    res = dict(spec_version=SPEC_VERSION, label=label, not_for_registration=(tuple(epochs) != EPOCHS or coverage_gate != COVERAGE_GATE_DEFAULT), epochs=list(epochs),
               conditions=conditions, manifest=str(manifest), manifest_sha256=sha256_file(manifest), n_clips=len(clips),
               gt_dir=str(gt_dir), gt_sha256=gt_content_hash(gt_dir, clips),
               code_sha256=dict(selector=sha256_file(Path(__file__)), integrity=sha256_file(ROOT / "server_sde/_prediction_integrity.py")),
               candidates={str(e): cands[e] for e in epochs}, order=order, selected=selected, rule=rule,
               reference_constant=constant_reference(gt, clips, e_close))
    atomic_write_all({out_json: json.dumps(res, ensure_ascii=False, indent=1), out_json.with_suffix(".md"): to_markdown(res)})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", required=True, help="候補の csv のパス。{epoch:03d} を含む（.done.json / .run_meta.json が同名で隣にあること）")
    ap.add_argument("--manifest", default=str(ROOT / "out/dataset_outdoor_siren_v12/plan/manifest_val_syn.csv"))   # 2026-09-22 改名（旧 manifest_fold2_val.csv）
    ap.add_argument("--gt", default=str(ROOT / "out/dataset_outdoor_siren_v12/metadata_dist_h"))
    ap.add_argument("--out", required=True, help="出力 JSON（同名 .md も書く）")
    ap.add_argument("--epochs", default=",".join(map(str, EPOCHS)), help="候補 epoch（仕様の 8 個以外にすると『登録には使えない』印が付く）")
    ap.add_argument("--label", default="v2final")
    ap.add_argument("--expected", type=int, default=12420, help="評価 manifest の本数（旧 v12 fold2 = 1800・v29 の val = 12420。本番では必ず渡す・凍結 r12 §4）")
    ap.add_argument("--coverage-gate", type=float, default=COVERAGE_GATE_DEFAULT,
                    help=f"検出率の門（pt・既定 {COVERAGE_GATE_DEFAULT}・本人決定 2026-09-14）。至近の検出率 D/N が最良候補 − この値 を下回る候補を外す")
    ap.add_argument("--no-coverage-gate", action="store_true", help="門を外す（report_only・併記だけ）。試走用で、登録の選定には使えない")
    ap.add_argument("--height-table", default=None, help="2.1（r11）: 推論に渡した高さの表（plan の csv）。渡すと候補の run_meta の height.table_sha256 がこの表と一致することを要求する（最新の val 用）。渡さなければ 2.0 の『表なし・1.5 m』の契約")
    ap.add_argument("--height-none", action="store_true", help="変更票 #74（アブレーション noheight）: 高さの入力なし（HEIGHT_COND=0）で推論した候補を受ける（--height-table と同時には渡せない）")
    ap.add_argument("--close-m", type=float, default=TH_CLOSE,
                    help=f"2.2（変更票 #75 ⑬）: 選ぶ指標 E の帯の上限 [m]（既定 {TH_CLOSE} = 2.1 と同じ値。v30 の登録では {TH_E_SPEC}）。門 N/D/C の帯は {TH_CLOSE} のまま")
    a = ap.parse_args(argv)
    epochs = tuple(int(x) for x in a.epochs.split(","))
    gate = None if a.no_coverage_gate else a.coverage_gate
    if (epochs != EPOCHS or gate is None or gate != COVERAGE_GATE_DEFAULT) and a.label == "v2final":
        print("候補集合・門が仕様（8 候補・門 2pt）と違う場合は --label で試走と分かる名前を付ける", file=sys.stderr)
        return 2
    if not (TH_CLOSE <= a.close_m <= 15.0):
        print(f"--close-m は {TH_CLOSE}〜15 m の範囲で（門の帯より狭くしない）: {a.close_m}", file=sys.stderr)
        return 2
    try:
        res = run(a.pattern, Path(a.manifest), Path(a.gt), Path(a.out), epochs, a.label, a.expected, gate, Path(a.height_table) if a.height_table else None, a.height_none, a.close_m)
    except (InputError, ValueError, OSError) as e:
        print(f"選定を停止: {e}", file=sys.stderr)
        return 1
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
