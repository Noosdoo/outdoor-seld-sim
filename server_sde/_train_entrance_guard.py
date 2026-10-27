# -*- coding: utf-8 -*-
"""学習の入口の門（2026-10-12・Fable・引き継ぎ §3 の 8「学習の入口で eval_ の本と実録の置き場を弾く検査」）。

学習（v2final_train.sbatch・v2final_dev_train.sh）が読むデータの置き場（datasets ルートと HDF5 ルート）に、
  (1) 確定評価セットの本（名前が eval_syn_ で始まる・旧 fold20_room1_mix）と補正表の組（tune_syn_・変更票 #72）
  (2) 実録の本（eval_field_・旧 S<2 桁>__ZOOM）や実録の置き場（raw／raw_demo／field／realsmoke という名の dir）
  (3) 学習・検証以外の用途の印（train_／val_ 以外で始まるクリップ名）
が 1 つでもあれば、学習を始めない（終了コード 2）。「fold20 で v2 系のモデルを 1 回でも走らせたら不合格」（凍結文書）の学習側の門。
HDF5 の中身（h5 の中の名前）も見る: 置き場の名前だけでなく、特徴 h5 のキーに eval_ があれば止める。

使い方: python _train_entrance_guard.py <datasets ルート> <HDF5 ルート> [--allow-empty]
  --allow-empty = datasets が空でも・HDF5 ルートがまだ無くても止めない（生成の途中＝前処理の前の門・AR12-01）。学習の前は付けない。
  シンボリックリンクの先の束も辿る（AR12-02・os.walk followlinks・同じ実体は 1 回）。
終了コード: 0 = 通過、2 = 止める（理由を出す）、1 = 入力エラー
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# tune_syn_ = 補正表の組（#72・2026-10-20）も学習に入れない。tune_field_ = 開発用の学内の本（台帳で使わないの本・デモの組・2026-10-27）も学習に入れない
EVAL_PAT = re.compile(r"^(eval_syn_clip|eval_field_clip|tune_syn_clip|tune_field_clip|fold20_room1_mix|fold2\d_room1_mix|S\d{2}__ZOOM)", re.I)
BAD_DIR_PREFIX = ("_demo_field_unused",)   # 開発用の学内の本の置き場（out/_demo_field_unused_149 など・2026-10-27）
TRAIN_PAT = re.compile(r"^(train_syn_clip|val_syn_clip|fold[12]_room1_mix)\d+", re.I)
BAD_DIRS = {"raw", "raw_demo", "raw_bicycle_measure", "field", "realsmoke", "field_demo", "eval_field", "clips"}
CLIP_EXT = {".wav", ".flac", ".csv", ".npy", ".h5"}


def scan(root: Path, max_report: int = 10) -> tuple[list[str], int]:
    """戻り = (止める理由の一覧, 見たファイル数)。**シンボリックリンクの先も辿る**（datasets_v26/outdoor_siren_v12 → out/... の束・AR12-02）。同じ実体は 1 回だけ。"""
    import os
    reasons, n = [], 0
    if not root.exists():
        return [f"置き場が無い: {root}"], 0
    seen = set()
    for cur, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(cur)
        if real in seen:
            dirs[:] = []; continue
        seen.add(real)
        for d in list(dirs):
            if d.lower() in BAD_DIRS or d.lower().startswith(BAD_DIR_PREFIX):
                reasons.append(f"実録の置き場の名前の dir がある: {Path(cur) / d}")
        for name in files:
            n += 1
            stem = name.split(".")[0]
            if EVAL_PAT.match(stem):
                reasons.append(f"確定評価／実録の本がある: {Path(cur) / name}")
            elif Path(name).suffix.lower() in CLIP_EXT and re.match(r"^(eval|fold|S\d{2}__)", stem, re.I) and not TRAIN_PAT.match(stem):
                reasons.append(f"学習・検証でない名前の本がある: {Path(cur) / name}")
            if len(reasons) >= max_report:
                return reasons + ["…（以降は省略）"], n
    return reasons, n


def scan_h5_keys(hdf5_root: Path, max_report: int = 10) -> list[str]:
    """特徴やラベルの h5 の中のキーに eval_ があれば止める（h5py が無ければ飛ばす・理由に書く）"""
    try:
        import h5py  # type: ignore
    except Exception:
        return [] if not hdf5_root.exists() else ["注意: h5py が無いので h5 の中のキーは見ていない"]
    reasons = []
    for p in hdf5_root.rglob("*.h5"):
        try:
            with h5py.File(p, "r") as hf:
                bad = [k for k in hf.keys() if EVAL_PAT.match(str(k))]
        except Exception as e:
            reasons.append(f"h5 が開けない: {p} ({e})"); continue
        if bad:
            reasons.append(f"h5 の中に確定評価／実録のキー: {p} → {bad[:3]}")
        if len(reasons) >= max_report:
            reasons.append("…（以降は省略）"); break
    return reasons


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2 or not args[0].strip() or not args[1].strip():
        print("入力エラー: datasets ルートと HDF5 ルートを両方渡す（空文字は不可・SV26-03: 置き場を決める前に呼ぶと現在の dir を見てしまう）"); print(__doc__); return 1
    ds, h5 = Path(args[0]), Path(args[1])
    allow_empty = "--allow-empty" in argv
    reasons, n_ds = scan(ds)
    if h5.exists():
        r2, n_h5 = scan(h5)
    else:
        r2, n_h5 = ([f"注意: HDF5 ルートがまだ無い（前処理の前）: {h5}"] if allow_empty else [f"HDF5 ルートが無い: {h5}"]), 0   # AR12-01: 生成の途中（前処理の前）は --allow-empty で通す
    reasons += r2 + scan_h5_keys(h5)
    if n_ds == 0 and not allow_empty:
        reasons.append(f"datasets ルートにファイルが無い: {ds}")
    hard = [r for r in reasons if not r.startswith("注意")]
    for r in reasons:
        print(("STOP " if not r.startswith("注意") else "NOTE ") + r)
    print(f"学習の入口の門: datasets {n_ds} ファイル・HDF5 {n_h5} ファイルを見た → {'止める' if hard else '通過'}（{ds}・{h5}）")
    return 2 if hard else 0


def selftest(tmp: Path) -> int:
    import shutil
    ok = []
    shutil.rmtree(tmp, ignore_errors=True)
    ds, h5 = tmp / "datasets", tmp / "hdf5"
    (ds / "foa").mkdir(parents=True); (h5 / "label").mkdir(parents=True)
    for name in ("train_syn_clip00001.flac", "val_syn_clip25019.flac"):
        (ds / "foa" / name).write_bytes(b"x")
    try:                                                   # h5py のある環境（サーバ）では中身のある h5 を作る（空ファイルは「開けない」で止まる＝それ自体は正しい）
        import h5py  # type: ignore
        with h5py.File(h5 / "label" / "train.h5", "w") as hf:
            hf.create_dataset("train_syn_clip00001/adpit/se", data=[0.0])
    except Exception:
        (h5 / "label" / "train.h5").write_bytes(b"")
    ok.append(("学習・検証の本だけ → 通過", main([str(ds), str(h5)]) in (0,)))
    try:
        import h5py  # type: ignore
        with h5py.File(h5 / "label" / "bad.h5", "w") as hf:
            hf.create_dataset("eval_syn_clip00001/adpit/se", data=[0.0])
        ok.append(("h5 の中のキーが eval_syn_ → 止める", main([str(ds), str(h5)]) == 2))
        (h5 / "label" / "bad.h5").unlink()
    except Exception:
        pass
    (ds / "foa" / "eval_syn_clip00001.flac").write_bytes(b"x")
    ok.append(("eval_syn_ の本が 1 つ → 止める", main([str(ds), str(h5)]) == 2))
    (ds / "foa" / "eval_syn_clip00001.flac").unlink()
    (ds / "foa" / "fold20_room1_mix0001.flac").write_bytes(b"x")
    ok.append(("旧名 fold20 の本 → 止める", main([str(ds), str(h5)]) == 2))
    (ds / "foa" / "fold20_room1_mix0001.flac").unlink()
    (ds / "foa" / "tune_syn_clip00001.flac").write_bytes(b"x")
    ok.append(("補正表の組 tune_syn_ の本 → 止める（#72）", main([str(ds), str(h5)]) == 2))
    (ds / "foa" / "tune_syn_clip00001.flac").unlink()
    (ds / "raw").mkdir()
    ok.append(("実録の置き場 raw/ がある → 止める", main([str(ds), str(h5)]) == 2))
    (ds / "raw").rmdir()
    (ds / "foa" / "eval_field_clip01001.flac").write_bytes(b"x")
    ok.append(("実録の本 eval_field_ → 止める", main([str(ds), str(h5)]) == 2))
    (ds / "foa" / "eval_field_clip01001.flac").unlink()
    (ds / "foa" / "tune_field_clip81001.flac").write_bytes(b"x")
    ok.append(("開発用の学内の本 tune_field_ → 止める（2026-10-27）", main([str(ds), str(h5)]) == 2))
    (ds / "foa" / "tune_field_clip81001.flac").unlink()
    (ds / "_demo_field_unused_149").mkdir()
    ok.append(("開発用の学内の本の置き場 _demo_field_unused_* がある → 止める（2026-10-27）", main([str(ds), str(h5)]) == 2))
    (ds / "_demo_field_unused_149").rmdir()
    ok.append(("空の datasets → 止める（--allow-empty で通過）", main([str(tmp / "none"), str(h5)]) == 2))
    ok.append(("HDF5 ルートがまだ無い: 既定は止める・--allow-empty なら通す（AR12-01）", main([str(ds), str(tmp / "no_h5")]) == 2 and main([str(ds), str(tmp / "no_h5"), "--allow-empty"]) == 0))
    ok.append(("空文字の引数は入力エラー（rc 1・SV26-03）", main(["", ""]) == 1 and main([str(ds), ""]) == 1))
    # AR12-02: シンボリックリンクの先の束を見る（作れない環境＝Windows の権限なしでは NOTE にして飛ばす）
    link_ok = None
    try:
        import os
        real = tmp / "real_bundle"; (real / "foa").mkdir(parents=True); (real / "foa" / "eval_syn_clip00002.flac").write_bytes(b"x")
        link = ds / "outdoor_siren_v12"; os.symlink(str(real), str(link), target_is_directory=True)
        link_ok = main([str(ds), str(h5)]) == 2
        os.unlink(link)
    except OSError as e:
        print("NOTE symlink を作れないので AR12-02 の検査は飛ばす（Linux の実機で回す）:", e)
    if link_ok is not None:
        ok.append(("シンボリックリンクの先の束に eval_syn_ がある → 止める（AR12-02）", link_ok))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest(Path(sys.argv[sys.argv.index("--selftest") + 1])))
    sys.exit(main())
