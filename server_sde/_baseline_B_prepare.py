# -*- coding: utf-8 -*-
"""相手 B（変更票 #73）: 公式ベースラインの入力を v29 から作る。
  1. layout: 公式の層（foa_dev/fold3_train/fold3_<clip>.wav → v29 の flac へのシンボリックリンク・metadata_dev/.../fold3_<clip>.csv = 距離を cm に）
  2. feat:   公式の特徴（log-mel 64 ＋ 強度ベクトル・20 ms）を並列に抽出（公式の extract_file_feature をそのまま呼ぶ）
  3. norm:   StandardScaler を **fold3（train）だけで fit**（Astra R73-03）し、全部を transform
  4. labels: 公式の ADPIT ラベル（cm → /100 で m）
  5. check:  単位の往復（2.00 m → csv 200 cm → ADPIT ラベル 2.0 m）と層の数の検査
使い方: python _baseline_B_prepare.py --step all [--limit-train N --limit-val N] [--jobs 32] [--check]
本体 §3-6（2026-10-26）: B_INPUT_GAIN_DB（特徴の前の倍率）・B_DIST_ENCODING（log1p ならラベルの距離を u に）を特徴の置き場の印（B_input_settings.json）に書き、
  既にある置き場とは照合する（違えば止める）。--link-feat-from <別の B_FEAT_DIR>: 距離の符号化だけ違う置き場を作るとき、特徴（標準化の前後と重み）は
  作り直さずにシンボリックリンクで借りる（元の置き場の倍率が同じことを印で確かめる）→ --step labels と check だけ回す。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _baseline_B_common as C  # noqa: E402


def read_plan(limit_train: int, limit_val: int) -> dict[str, list[str]]:
    plan = C.SRC / "plan/assignment_v2final.csv"
    out = {"train": [], "val": []}
    with open(plan, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] in out:
                out[r["split"]].append(r["clip_id"])
    for k in out:
        out[k].sort()
    if limit_train:
        out["train"] = out["train"][:limit_train]
    if limit_val:
        out["val"] = out["val"][:limit_val]
    return out


def write_meta_cm(src_csv: Path, dst_csv: Path) -> int:
    """metadata_dist（frame,cls,track,az,el,dist[m]）→ 公式の 6 列（距離は round(m×100) の cm・int）。行数を返す。"""
    n = 0
    with open(src_csv, encoding="utf-8") as f, open(dst_csv, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6 or not p[0]:
                continue
            k, c, t = int(float(p[0])), int(float(p[1])), int(float(p[2]))
            az, el = float(p[3]), float(p[4])
            d_m = float(p[5])
            if not np.isfinite(d_m):
                raise ValueError(f"{src_csv.name}: 距離が有限でない: {line.strip()}")
            g.write(f"{k},{c},{t},{az:g},{el:g},{int(round(d_m * 100.0))}\n"); n += 1
    return n


def step_layout(clips: dict[str, list[str]]) -> dict:
    rec = {}
    for split, ids in clips.items():
        fold = C.FOLD_OF[split]
        sub = f"fold{fold}_{split}"
        aud = C.DS / "foa_dev" / sub; desc = C.DS / "metadata_dev" / sub
        aud.mkdir(parents=True, exist_ok=True); desc.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        for cid in ids:
            src = C.SRC / "foa" / f"{cid}.flac"
            if not src.is_file():
                raise FileNotFoundError(src)
            link = aud / f"fold{fold}_{cid}.wav"              # 名前は .wav（公式の一覧の作り方に合わせる）・中身は flac（soundfile が中身で判定）
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            n_rows += write_meta_cm(C.SRC / "metadata_dist" / f"{cid}.csv", desc / f"fold{fold}_{cid}.csv")
        rec[split] = {"n_clips": len(ids), "n_label_rows": n_rows, "fold": fold, "sub": sub}
        print(f"[layout] {split}: {len(ids):,} 本・ラベル {n_rows:,} 行 → {aud}", flush=True)
    return rec


_FC = None      # fork の子が親から引き継ぐ特徴クラス（FeatureClassFlac は関数の中のクラスで pickle できない → 引数で渡さない。試走 job 8560・2026-10-21 02:3x）


def _feat_one(item):
    _FC.extract_file_feature(item)
    return item[2]


def step_feat(params: dict, jobs: int) -> dict:
    global _FC
    C.check_feat_marker(C.FEAT, write=True)              # 本体 §3-6: 倍率と距離の符号化の印（既にあれば照合・違えば止める）
    fc = C.feature_class(params)
    _FC = fc
    feat_dir = Path(fc.get_unnormalized_feat_dir()); feat_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for sub in sorted(os.listdir(fc._aud_dir)):
        loc = Path(fc._aud_dir) / sub
        for i, fn in enumerate(sorted(os.listdir(loc))):
            stem = fn.split(".")[0]
            out = feat_dir / f"{stem}.npy"
            if out.exists():
                continue
            items.append((i, str(loc / fn), str(out)))
    print(f"[feat] 抽出 {len(items):,} 本（既にあるものは飛ばす）→ {feat_dir}・jobs {jobs}", flush=True)
    t0 = time.time()
    if items:
        with get_context("fork").Pool(jobs) as pool:
            for k, _ in enumerate(pool.imap_unordered(_feat_one, items, chunksize=8)):
                if (k + 1) % 2000 == 0:
                    el = time.time() - t0
                    print(f"  {k+1}/{len(items)}  {el:.0f}s  残り約{el/(k+1)*(len(items)-k-1)/60:.0f}分", flush=True)
    n = len(list(feat_dir.glob("*.npy")))
    return {"feat_dir": str(feat_dir), "n_feat": n, "sec": round(time.time() - t0)}


def step_norm(params: dict) -> dict:
    """公式 preprocess_features と同じ StandardScaler だが、fit は fold3（train）だけ（Astra R73-03）。"""
    import joblib
    from sklearn import preprocessing
    fc = C.feature_class(params)
    feat_dir = Path(fc.get_unnormalized_feat_dir()); norm_dir = Path(fc.get_normalized_feat_dir()); norm_dir.mkdir(parents=True, exist_ok=True)
    wts = Path(fc.get_normalized_wts_file())
    files = sorted(feat_dir.glob("*.npy"))
    train_files = [p for p in files if p.name.startswith("fold3_")]
    if not train_files:
        raise SystemExit("fold3（train）の特徴が無い")
    sc = preprocessing.StandardScaler()
    for p in train_files:
        sc.partial_fit(np.load(p))
    wts.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc, wts)
    for p in files:
        np.save(norm_dir / p.name, sc.transform(np.load(p)))
    return {"norm_dir": str(norm_dir), "wts": str(wts), "n_fit_train": len(train_files), "n_transformed": len(files),
            "fit_rule": "fold3 (train) only (Astra R73-03)"}


def step_labels(params: dict) -> dict:
    C.check_feat_marker(C.FEAT, write=True)              # 本体 §3-6: ラベルの距離の符号化は印と同じ
    fc = C.feature_class(params)
    fc.extract_all_labels()
    lab = Path(fc.get_label_dir())
    return {"label_dir": str(lab), "n_labels": len(list(lab.glob("*.npy"))), "dist_encoding": C.DIST_ENCODING}


def link_feat_from(params: dict, src_feat: Path) -> dict:
    """本体 §3-6: 距離の符号化だけ違う置き場を作るとき、特徴（標準化の前・後）と標準化の重みを元の置き場から借りる（作り直さない）。
    元の置き場の倍率が今の環境と同じことを印で確かめる（符号化は違ってよい）。"""
    import os as _os
    src_feat = Path(src_feat)
    m = src_feat / C.FEAT_MARKER
    have = json.loads(m.read_text(encoding="utf-8")) if m.is_file() else {"input_gain_db": 0.0, "dist_encoding": "linear"}
    if float(have.get("input_gain_db", 0.0)) != C.INPUT_GAIN_DB:
        raise SystemExit(f"元の置き場 {src_feat} の倍率 {have.get('input_gain_db')} が今の B_INPUT_GAIN_DB={C.INPUT_GAIN_DB:g} と違う（特徴は借りられない）")
    fc_new = C.feature_class(params)
    P2 = dict(params); P2["feat_label_dir"] = str(src_feat) + "/"
    fc_src = C.feature_class(P2)
    out = {}
    for name in ("get_unnormalized_feat_dir", "get_normalized_feat_dir", "get_normalized_wts_file"):
        s_, d_ = Path(getattr(fc_src, name)()), Path(getattr(fc_new, name)())
        if not s_.exists():
            raise SystemExit(f"元の置き場に {name} が無い: {s_}")
        if d_.is_symlink() or d_.exists():
            if d_.is_symlink() and Path(_os.readlink(d_)) == s_:
                out[name] = str(s_); continue
            raise SystemExit(f"借りる先に既に別のものがある: {d_}")
        d_.parent.mkdir(parents=True, exist_ok=True)
        d_.symlink_to(s_, target_is_directory=s_.is_dir())
        out[name] = str(s_)
    C.check_feat_marker(C.FEAT, write=True)
    return {"linked_from": str(src_feat), "links": out}


def step_check(params: dict) -> dict:
    """単位の往復: 2.00 m → csv 200 → ADPIT ラベルの距離 2.0。層の数: foa と metadata の本数が同じ・fold の桁が合う。"""
    import tempfile
    fc = C.feature_class(params)
    tmp = Path(tempfile.mkdtemp(prefix="B_check_"))
    src = tmp / "m.csv"; src.write_text("".join(f"{k},4,0,30,0,2.00\n" for k in range(10, 20)), encoding="utf-8")
    dst = tmp / "cm.csv"; write_meta_cm(src, dst)
    first = dst.read_text(encoding="utf-8").splitlines()[0]
    desc = fc.load_output_format_file(str(dst))
    cart = fc.convert_output_format_polar_to_cartesian(desc)
    lab = fc.get_adpit_labels_for_file(cart, 100)                     # (100, 6, 4, C) [sed, x, y, z] … 距離は別の配列?
    ok_csv = first.endswith(",200")
    # ADPIT ラベルの距離ブロックの位置は公式の実装に依る → 値 2.0 がラベルの中に出ることを確かめる（sed=1 と x,y,z 以外に 2.0 が現れる）
    # 本体 §3-6: log1p のときは u(2.0) = log1p(2)/log1p(100) が出ることを確かめる（往復 = 復号で 2.0 m に戻る）
    want = 2.0 if C.DIST_ENCODING == "linear" else float(C.dist_encode_log(2.0))
    vals = np.unique(np.round(lab[lab > (1.5 if C.DIST_ENCODING == "linear" else 0.0)], 6))
    ok_lab = any(abs(v - want) < 1e-5 for v in vals) and (C.DIST_ENCODING == "linear" or abs(float(C.dist_decode_log(want)) - 2.0) < 1e-9)
    n_foa = sum(len(os.listdir(Path(fc._aud_dir) / s)) for s in os.listdir(fc._aud_dir))
    n_desc = sum(len(os.listdir(Path(fc._desc_dir) / s)) for s in os.listdir(fc._desc_dir))
    folds = sorted({fn[4] for s in os.listdir(fc._aud_dir) for fn in os.listdir(Path(fc._aud_dir) / s)})
    rec = {"csv_cm_200": ok_csv, "adpit_label_has_2.0_m": ok_lab, "dist_encoding": C.DIST_ENCODING, "label_dist_value": want, "label_shape": list(lab.shape), "n_foa": n_foa, "n_desc": n_desc, "folds": folds, "ok": ok_csv and ok_lab and n_foa == n_desc and folds == sorted({str(f) for f in C.FOLD_OF.values()})}
    print("[check]", json.dumps(rec, ensure_ascii=False), flush=True)
    if not rec["ok"]:
        raise SystemExit("単位の往復か層の数が合わない")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="all", choices=["layout", "feat", "norm", "labels", "check", "all"])
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--record", default=str(C.DS / "B_prepare_record.json"))
    ap.add_argument("--link-feat-from", default="", help="本体 §3-6: 特徴は別の B_FEAT_DIR から借りる（距離の符号化だけ違う置き場・labels と check だけ回す）")
    a = ap.parse_args()
    C.setup_path()
    params = C.get_params()
    rec = {"src": str(C.SRC), "dataset_dir": str(C.DS), "feat_dir": str(C.FEAT), "limit_train": a.limit_train, "limit_val": a.limit_val,
           "plan_sha256": hashlib.sha256((C.SRC / "plan/assignment_v2final.csv").read_bytes()).hexdigest(), "steps": {},
           "input_gain_db": C.INPUT_GAIN_DB, "dist_encoding": C.DIST_ENCODING}
    steps = ["layout", "feat", "norm", "labels", "check"] if a.step == "all" else [a.step]
    if a.link_feat_from:
        rec["steps"]["link_feat"] = link_feat_from(params, Path(a.link_feat_from))
        steps = [s for s in steps if s not in ("feat", "norm")]
    for s in steps:
        t0 = time.time()
        if s == "layout":
            rec["steps"][s] = step_layout(read_plan(a.limit_train, a.limit_val))
        elif s == "feat":
            rec["steps"][s] = step_feat(params, a.jobs)
        elif s == "norm":
            rec["steps"][s] = step_norm(params)
        elif s == "labels":
            rec["steps"][s] = step_labels(params)
        elif s == "check":
            rec["steps"][s] = step_check(params)
        rec["steps"][s]["sec"] = round(time.time() - t0)
        print(f"[{s}] 済 {rec['steps'][s]}", flush=True)
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("B_PREPARE_DONE", a.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
