# -*- coding: utf-8 -*-
"""v2final の生成ジョブ（server_sde/v2final_gen.sbatch）の末尾で走る「完了の検査」。

監査 Astra 9/28 R05 → 再監査 10/1 RA01: 以前の検査は HDF5 ルート配下の *.h5 が 1 個でも新しければ通った
（中身が HDF5 でなくても・必要なラベル h5 が無くても OK になる）。この版は次を全部確かめ、1 つでも落ちたら終了値 1。

A. plan の行数が期待値と一致し、clip_id に重複が無い
B. plan の全 clip_id に foa/.flac・metadata/.csv・metadata_dist/.csv・masks/.csv がそろう（foa・masks は空ファイルを欠けと数える。metadata・metadata_dist は正解の無い本が正当に 0 バイトになるので存在だけ）
C. HDF5 ルート配下に *.h5 が 1 本以上あり、その **全部** がこの job で更新されている（1 本でも古ければ「片方だけ更新」とみなして NG）
D. 必要なラベル h5 が存在する: label/<kind>/<type>/*.h5 で kind = adpit・adpit_dist・accdoa・track（サーバの実物の 4 本・--label-kinds で変更可・<type> は --dataset-type・既定 dev）
E. 全 *.h5 が h5py で開ける（0 バイト・HDF5 でないファイルは NG）
F. ラベル h5 は plan の全 clip_id をトップのキーに持つ。抜き取り（既定 50 本）で adpit は adpit/se があり、adpit_dist は adpit/dist のフレーム数が se と同じで有限値、accdoa・track は中身が空でない
G. clip 単位でもラベルでもない h5 は空でないことだけ要求する
H. 特徴の h5（<clip>.h5）は --require-clip-features のときだけ plan の全 clip_id に 1 本ずつあり空でないことを要求する（v2final の実物は data/ が空で特徴 h5 を持たない・監査 A 10/1 F2 への対応）
I. **内容の照合（再監査 Astra 10/6 RF01）**: plan の **全 clip** について、正解 csv（metadata_dist・metadata は先頭 5 列が一致すること）から
   「クラスごとの活動フレームの集合」を作り、4 種のラベル h5 の活動（adpit・adpit_dist = adpit/se の [T,6,C]、accdoa = accdoa/se の [T,C]、
   track = sed_label の [T,3,C]）と一致することを要求する。csv に正解があるのに h5 が 0・csv が空なのに h5 に活動・クラスやフレームの食い違いは NG。
   adpit_dist は距離クラス（car・kick・bike）の活動フレームで adpit/dist > 0 も要求する。h5 のフレーム数は「csv の最後のフレーム + 1」
   （PSELDNets の規約・学習時に padding）と一致すること、csv に clip の長さ（--frames・既定 100）以上のフレームが無いことも要求する
   （Astra 10/7 V05: h5 を短く切ると後半の正解が落ちる反例。固定長 100 の要求は実物 4,138 本で誤検出したので規約どおりに）。
   track 形式だけは既知の性質を許す（1 フレーム 3 枠までで 4 本目以降が落ちる = 枠の数が min(3, 音源数) と一致すること・L02・正解の無い本に仮の 1 行 class 0 が入る。本レシピの学習は
   adpit_dist を読むので track は使わない。実物 25,500 本では adpit・adpit_dist・accdoa は 21,135 本すべて一致、track はこの 2 点だけが差）。
   **保証しない範囲**: csv と h5 が両方とも空の本（可聴ゲートで全部落ちた本など・実物 4,365 本）は「空が正当か」をここでは判定できない。
   生成側の記録（plan の場面・可聴性のログ）で見る。--no-content で I を飛ばせる（開発用の速い確認だけ）。

使い方（sbatch から）:
    python server_sde/_v2final_gen_complete_check.py --n 25500 --hdf5 $HDF5DIR --t-pre $T_PRE
    （--ds・--plan・--dataset-type は既定あり）
検査 = scripts/test_v2final_gen_complete_check.py（人工の束で OK と各 NG を再現）
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# (サブフォルダ, 拡張子, 空ファイルを欠けと数えるか)。metadata/metadata_dist は「正解が 1 行も無い本」（可聴ゲートで全部落ちた本など）が
# 正当に 0 バイトになる（サーバの実物 10/1: 25,500 本のうち 4,365 本）ので、存在だけを要求する。音声と masks は空なら欠け。
REQUIRED_SUBS = (("foa", "flac", True), ("metadata", "csv", False), ("metadata_dist", "csv", False), ("masks", "csv", True))
DIST_CLS = (4, 6, 7, 8)   # car / kick / bike / bicycle（距離クラス・adpit_dist の dist > 0 を要求する列。8 = bicycle 2026-10-08 変更票 #48）


def _has_data(g, h5py) -> bool:
    """グループの中（入れ子を含む）に要素数 1 以上のデータセットが 1 つでもあるか。空のグループだけなら False。"""
    if isinstance(g, h5py.Dataset):
        return g.size > 0
    if not isinstance(g, h5py.Group):
        return False
    return any(_has_data(g[k], h5py) for k in g.keys())


def load_plan_ids(plan: Path) -> list[str]:
    with open(plan, encoding="utf-8", newline="") as f:
        return [r["clip_id"] for r in csv.DictReader(f)]


LABEL_KINDS = ("adpit", "adpit_dist", "accdoa", "track")   # サーバの実物（10/1 確認）: label/<kind>/dev/outdoor_siren_v12.h5 の 4 本。特徴の h5 は無い（音は flac から直接読む）


def csv_rows(path: Path) -> list[tuple[int, int, str]]:
    """正解 csv の (frame, cls, 先頭 5 列を , でつないだ文字列)。空ファイル・無い・読めない行は飛ばす。"""
    out = []
    if not path.is_file() or path.stat().st_size == 0:
        return out
    for line in open(path, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) < 5:
            continue
        try:
            out.append((int(g[0]), int(g[1]), ",".join(g[:5])))
        except ValueError:
            continue
    return out


FRAMES_EXPECTED = 100     # 1 本 = 10 s × 10 フレーム/s（label_resolution 0.1 s）。csv のフレームはこれ未満でなければならない（Astra 10/7 V05）
# ラベル h5 の配列の長さは PSELDNets の extract_*_label の規約で「csv の最後のフレーム + 1」（学習時に chunk 長へ padding する）。
# 実物 25,500 本のうち 4,138 本が 100 未満（10/7 確認）。なので長さの検査は「max(csv frame)+1 と一致」で行う。


def frames_by_class(rows, t_max: int) -> dict[int, set[int]]:
    """クラスごとの活動フレームの集合。t_max 以上のフレームは捨てず、呼ぶ側で「範囲外」として数える（V05）。"""
    out: dict[int, set[int]] = defaultdict(set)
    for fr, c, _ in rows:
        if 0 <= fr:
            out[c].add(fr)
    return out


def h5_frames_by_class(g, kind: str, np):
    """clip のグループから「クラスごとの活動フレームの集合」。(frames, se, dist_or_None) を返す。配列が無ければ (None, None, None)。"""
    if kind in ("adpit", "adpit_dist"):
        if "adpit/se" not in g:
            return None, None, None
        se = g["adpit/se"][()]                           # [T, 6, C]
        act = se.reshape(se.shape[0], -1, se.shape[-1]).any(axis=1)   # [T, C]
        dist = g["adpit/dist"][()] if (kind == "adpit_dist" and "adpit/dist" in g) else None
        return {c: set(np.flatnonzero(act[:, c]).tolist()) for c in range(act.shape[1]) if act[:, c].any()}, se, dist
    if kind == "accdoa":
        if "accdoa/se" not in g:
            return None, None, None
        se = g["accdoa/se"][()]                          # [T, C]
        return {c: set(np.flatnonzero(se[:, c]).tolist()) for c in range(se.shape[1]) if se[:, c].any()}, se, None
    if kind == "track":
        if "sed_label" not in g:
            return None, None, None
        se = g["sed_label"][()]                          # [T, 3, C]
        act = se.any(axis=1)
        return {c: set(np.flatnonzero(act[:, c]).tolist()) for c in range(act.shape[1]) if act[:, c].any()}, se, None
    return None, None, None


def content_check(ids, ds: Path, label_files, hdf5: Path, log=print, frames: int = FRAMES_EXPECTED) -> list[str]:
    """I. 全 clip の正解 csv と 4 種のラベル h5 の活動を照合する。NG の理由の列（空なら OK）。"""
    import h5py  # noqa: WPS433
    import numpy as np
    bad: list[str] = []
    n_meta_mismatch = 0; ex_meta = []
    csv_fc: dict[str, dict] = {}
    n_empty_csv = 0
    for c in ids:
        rows_d = csv_rows(ds / "metadata_dist" / f"{c}.csv")
        rows_m = csv_rows(ds / "metadata" / f"{c}.csv")
        if [r[2] for r in rows_d] != [r[2] for r in rows_m]:
            n_meta_mismatch += 1
            if len(ex_meta) < 3:
                ex_meta.append(f"{c}: metadata {len(rows_m)} 行 / metadata_dist {len(rows_d)} 行")
        csv_fc[c] = rows_d
        if not rows_d:
            n_empty_csv += 1
    if n_meta_mismatch:
        bad.append(f"metadata と metadata_dist の先頭 5 列が食い違う本 {n_meta_mismatch} 例 {ex_meta}")
    log(f"内容の照合: csv に正解が無い本 {n_empty_csv:,}（h5 も空なら整合・空が正当かはここでは判定しない）")
    for p in label_files:
        kind = p.parent.parent.name
        n_mis = n_noarr = n_dist_bad = n_both_empty = n_ok = 0
        ex = []
        with h5py.File(p, "r") as hf:
            for c in ids:
                if c not in hf:
                    continue                 # F で別に数えている
                fr_h5, se, dist = h5_frames_by_class(hf[c], kind, np)
                if fr_h5 is None:
                    n_noarr += 1
                    if len(ex) < 3:
                        ex.append(f"{c}: 活動配列が無い")
                    continue
                t_max = se.shape[0]
                fr_csv = frames_by_class(csv_fc[c], t_max)
                # V05: csv の範囲外のフレーム（clip の長さ以上）は捨てずに NG。h5 の長さは「csv の最後のフレーム + 1」と一致すること
                #   （PSELDNets の規約・csv が空なら長さは見ない）。短く切られた h5 はここで止まる。
                out_of_range = sorted({fr for v in fr_csv.values() for fr in v if fr >= frames})
                t_expect = (max(fr for v in fr_csv.values() for fr in v) + 1) if fr_csv else None
                if out_of_range or (t_expect is not None and t_max != t_expect):
                    n_mis += 1
                    if len(ex) < 3:
                        ex.append(f"{c}: csv に範囲外のフレーム {out_of_range[:3]}（clip は {frames} フレーム）" if out_of_range
                                  else f"{c}: h5 のフレーム数 {t_max} ≠ csv の最後のフレーム+1 = {t_expect}")
                    continue
                if kind == "track":
                    # track 形式（PSELDNets extract_track_label）の既知の性質（実物 25,500 本で 10/6 に確認・本レシピの学習は adpit_dist を読むので track は使わない）:
                    #   ① 1 フレームの同時音源は 3 枠まで。4 本目以降は落ちる（csv で 4 本以上のフレームだけ h5 の活動が欠けてよい）
                    #   ② 正解の無い本は仮の 1 行（class 0・1 フレーム）が入る（csv が空のときだけ、class 0 の 1 フレームまで許す）
                    #   それ以外（csv に無い活動・3 本以下のフレームでの欠け）は食い違いとして NG
                    n_per_frame = Counter(fr for fr, _, _ in csv_fc[c] if 0 <= fr < t_max)
                    if not fr_csv:
                        same = sum(len(v) for v in fr_h5.values()) <= 1 and set(fr_h5) <= {0}
                    else:
                        # フレームごとに: h5 の活動枠の数 = min(3, csv の音源数)（L02: 4 本以上でも上位 3 枠は残る・全消失は NG）、
                        # かつ h5 のクラスは csv のクラスの部分集合。3 本以下のフレームはクラスの集合も一致。
                        n_h5_frame = se.reshape(se.shape[0], -1).sum(axis=1)       # [T] 活動枠の数
                        cls_csv_frame = defaultdict(set)
                        for fr, k, _ in csv_fc[c]:
                            if 0 <= fr < t_max:
                                cls_csv_frame[fr].add(k)
                        same = True
                        for fr in range(t_max):
                            n_csv = n_per_frame.get(fr, 0)
                            if int(n_h5_frame[fr]) != min(3, n_csv):
                                same = False; break
                            cls_h5 = {k for k, v in fr_h5.items() if fr in v}
                            if n_csv <= 3:
                                if cls_h5 != cls_csv_frame.get(fr, set()):
                                    same = False; break
                            elif not cls_h5 <= cls_csv_frame.get(fr, set()):
                                same = False; break
                else:
                    same = fr_csv == fr_h5
                if not same:
                    n_mis += 1
                    if len(ex) < 3:
                        cls = sorted(set(fr_csv) | set(fr_h5))
                        ex.append(f"{c}: " + " ".join(f"cls{k} csv{len(fr_csv.get(k, ()))}/h5{len(fr_h5.get(k, ()))}" for k in cls[:4]))
                    continue
                if not fr_csv:
                    n_both_empty += 1
                else:
                    n_ok += 1
                if kind == "adpit_dist" and (dist is None or dist.shape != se.shape):
                    n_dist_bad += 1
                    if len(ex) < 3:
                        ex.append(f"{c}: adpit/dist が無いか形が se と違う（{None if dist is None else dist.shape} 対 {se.shape}）")
                elif kind == "adpit_dist":
                    for k in DIST_CLS:
                        if k < se.shape[-1] and se[..., k].any():
                            dk = dist[..., k][se[..., k]]
                            if not (np.isfinite(dk).all() and (dk > 0).all()):
                                n_dist_bad += 1
                                if len(ex) < 3:
                                    ex.append(f"{c}: cls{k} の adpit/dist に 0 以下か非有限（活動フレーム）")
                                break
        log(f"内容の照合 {p.relative_to(hdf5)}: 一致 {n_ok:,}・両方空 {n_both_empty:,}・食い違い {n_mis}・配列なし {n_noarr}・距離不正 {n_dist_bad}")
        if n_mis or n_noarr or n_dist_bad:
            bad.append(f"{p.relative_to(hdf5)}: csv と h5 の活動が食い違う {n_mis}・活動配列なし {n_noarr}・距離不正 {n_dist_bad} 例 {ex}")
    return bad


def check(n_expected: int, hdf5: Path, t_pre: float, ds: Path, plan: Path, dataset_type: str,
          fresh_slack_s: float = 5.0, log=print, require_clip_features: bool = False, sample_n: int = 50,
          label_kinds=LABEL_KINDS, content: bool = True, frames: int = FRAMES_EXPECTED) -> list[str]:
    """NG の理由を列挙して返す（空なら OK）。"""
    bad: list[str] = []
    # A
    ids = load_plan_ids(plan)
    log(f"plan: {len(ids)} 行（期待 {n_expected}）")
    if len(ids) != n_expected:
        bad.append(f"plan の行数 {len(ids)} ≠ {n_expected}")
    if len(set(ids)) != len(ids):
        bad.append(f"plan の clip_id に重複 {len(ids) - len(set(ids))} 件")
    idset = set(ids)
    # B
    for sub, ext, nonempty in REQUIRED_SUBS:
        d = ds / sub
        files = list(d.glob(f"*.{ext}")) if d.is_dir() else []
        n_empty = sum(1 for p in files if p.stat().st_size == 0)
        have = {p.stem for p in files if (p.stat().st_size > 0 or not nonempty)}
        miss = [c for c in ids if c not in have]
        log(f"{sub}: {len(files)} 本（空 {n_empty}{'・空は欠け' if nonempty else '・空は正解なしの本として可'}）・plan に対する欠け {len(miss)}")
        if miss:
            bad.append(f"{sub} 欠け {len(miss)} 例 {miss[:3]}")
    # C
    h5 = sorted(p for p in hdf5.rglob("*.h5")) if hdf5.is_dir() else []
    log(f"HDF5: {len(h5)} 本 under {hdf5}")
    if not h5:
        bad.append("HDF5 が 1 本も無い")
    stale = [p for p in h5 if p.stat().st_mtime < t_pre - fresh_slack_s]
    if stale:
        bad.append(f"この job で更新されていない HDF5 {len(stale)} 本 例 {[p.name for p in stale[:3]]}")
    # D
    need_dirs = [hdf5 / "label" / k / dataset_type for k in label_kinds]
    label_files: list[Path] = []
    for d in need_dirs:
        fs = sorted(d.glob("*.h5")) if d.is_dir() else []
        if not fs:
            bad.append(f"必要なラベル h5 が無い: label/{d.parent.name}/{dataset_type}/*.h5")
        label_files += fs
    # E〜G
    try:
        import h5py  # noqa: WPS433
        import numpy as np
    except Exception as e:  # pragma: no cover - サーバには入っている
        bad.append(f"h5py / numpy を読み込めない: {e}")
        return bad
    # H. clip ごとの特徴 h5（置き方によっては <clip>.h5 にキー feature で書く・監査 A 10/1 F2）:
    #    --require-clip-features のときだけ、plan の全 clip に 1 本ずつあることを要求する。
    #    v2final のサーバの実物（10/1）は data/24000fs/wav/dev が空で特徴 h5 を持たない（音は flac から直接読む）ので既定は要求しない。
    clip_h5 = {p.stem: p for p in h5 if p.stem in idset}
    if require_clip_features:
        miss = [c for c in ids if c not in clip_h5]
        log(f"clip ごとの h5（特徴）: {len(clip_h5)} 本・plan に対する欠け {len(miss)}")
        if miss:
            bad.append(f"clip ごとの特徴 h5 の欠け {len(miss)} 例 {miss[:3]}")
    # 抜き取り（ラベルの中身・監査 A F12）: 先頭・末尾と等間隔の sample_n 本
    sample = sorted({ids[0], ids[-1], *ids[:: max(1, len(ids) // max(sample_n - 2, 1))]}) if ids else []
    openable_labels: list[Path] = []
    for p in h5:
        if p.stat().st_size == 0:
            bad.append(f"0 バイトの h5: {p.name}")
            continue
        try:
            with h5py.File(p, "r") as hf:
                keys = list(hf.keys())
                is_label = p in label_files
                if is_label:
                    openable_labels.append(p)
                    miss = [c for c in ids if c not in hf]
                    if miss:
                        bad.append(f"{p.relative_to(hdf5)} に plan の clip が無い {len(miss)} 例 {miss[:3]}")
                    kind = p.parent.parent.name          # label/<kind>/<type>/<ds>.h5
                    broken = []
                    for c in sample:
                        if c not in hf:
                            continue
                        g = hf[c]
                        if kind not in ("adpit", "adpit_dist"):
                            if not _has_data(g, h5py):
                                broken.append(f"{c}: 中身が空")
                            continue
                        if "adpit/se" not in g or g["adpit/se"].shape[0] < 1:
                            broken.append(f"{c}: adpit/se 無しか空")
                        elif kind == "adpit_dist":
                            if "adpit/dist" not in g:
                                broken.append(f"{c}: adpit/dist 無し")
                            else:
                                d = g["adpit/dist"][()]
                                if d.shape[0] < 1 or d.shape[0] != g["adpit/se"].shape[0]:
                                    broken.append(f"{c}: adpit/dist のフレーム数 {d.shape[0]} ≠ se {g['adpit/se'].shape[0]}")
                                elif not np.isfinite(d).all():
                                    broken.append(f"{c}: adpit/dist に非有限値")
                        if len(broken) >= 3:
                            break
                    if broken:
                        bad.append(f"{p.relative_to(hdf5)} の中身が壊れている（抜き取り {len(sample)} 本）例 {broken[:3]}")
                    log(f"{p.relative_to(hdf5)}: clip キー {len(keys)} 本・欠け {len(miss)}・抜き取り {len(sample)} 本 OK={not broken}")
                elif p.stem in idset:
                    # clip ごとの特徴 h5: 空でないデータセットが 1 つ以上
                    if not keys or all(getattr(hf[k], "shape", (0,)) in ((), (0,)) or (hasattr(hf[k], "shape") and 0 in hf[k].shape) for k in keys):
                        bad.append(f"{p.relative_to(hdf5)} が空（特徴なし）")
                else:
                    if not keys:
                        bad.append(f"{p.relative_to(hdf5)} が空")
                    log(f"{p.relative_to(hdf5)}: トップのキー {len(keys)} 本（clip 単位でもラベルでもない・存在と読み取りだけ確認）")
        except OSError as e:
            bad.append(f"{p.relative_to(hdf5)} を HDF5 として開けない: {e}")
    # I. 内容の照合（RF01）: 開けたラベル h5 の全 clip について csv の活動と比べる
    if content and openable_labels:
        bad += content_check(ids, ds, openable_labels, hdf5, log=log, frames=frames)
    elif not content:
        log("内容の照合: --no-content で飛ばした")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="plan の期待行数（N_ROWS_TOTAL_EXPECTED）")
    ap.add_argument("--hdf5", required=True, help="HDF5 ルート（$HDF5DIR）")
    ap.add_argument("--t-pre", type=float, required=True, help="前処理を始めた時刻（epoch 秒）")
    ap.add_argument("--ds", default="out/dataset_outdoor_siren_v2final")
    ap.add_argument("--plan", default=None, help="既定 <ds>/plan/assignment_v2final.csv")
    ap.add_argument("--dataset-type", default="dev")
    ap.add_argument("--require-clip-features", action="store_true", help="特徴を clip ごとの <clip>.h5 に置く構成のときだけ付ける（v2final の実物は持たない）")
    ap.add_argument("--label-kinds", default=",".join(LABEL_KINDS), help="必要なラベル h5 の種類（label/<kind>/<type>/*.h5）")
    ap.add_argument("--sample-n", type=int, default=50, help="ラベルの中身を抜き取りで確かめる本数")
    ap.add_argument("--no-content", action="store_true", help="I（csv と h5 の活動の照合・全 clip）を飛ばす（開発用の速い確認だけ）")
    ap.add_argument("--frames", type=int, default=FRAMES_EXPECTED, help="1 本の clip のフレーム数（既定 100 = 10 s × 10 fps）。csv のフレームはこれ未満であること（Astra 10/7 V05）")
    a = ap.parse_args(argv)
    ds = Path(a.ds)
    plan = Path(a.plan) if a.plan else ds / "plan" / "assignment_v2final.csv"
    bad = check(a.n, Path(a.hdf5), a.t_pre, ds, plan, a.dataset_type,
                require_clip_features=a.require_clip_features, sample_n=a.sample_n,
                label_kinds=tuple(k for k in a.label_kinds.split(",") if k), content=not a.no_content, frames=a.frames)
    if bad:
        print("NG:", bad)
        return 1
    print("完了の検査 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
