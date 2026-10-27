# -*- coding: utf-8 -*-
"""検出のしきい値（クラスごとの入り）を val から選ぶ（本体 §3-5・§9-1 の 5〔§3-5 を置き換え〕・R30-05・§9-4 の 5・2026-10-26）。本人の裁量なし。

手順（成績を見る前に固定・ここに書いたとおり）:
  - 入力 = 選んだ 1 つの ckpt の因果推論（`_causal_infer_v17.py` を `DUMP_NORMS=1` で val 全部）が残す `<run>/norms/<clip>.npy`（3 トラック・T・9 の norm・
    ヒステリシスの前・間引かない）と `<run>/norms_manifest.json`（ckpt の sha・本ごとの sha）。B・D の norms と同じ型。
  - 選択用 = val の clip 番号を 12 で割った余りが 0 の本・**確認用 = 余りが 6 の本**（確認用は採否にも再探索にも使わない・併記だけ）。
  - 候補 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}。保持 = round(入り × 3/7, 4)。
  - 指標 = クラスごとの「フレーム単位の検出 F」: 正解 = metadata_dist にそのクラスの行がある枠・予測 = 3 トラックの最大の norm > 入り（**入りだけ・ヒステリシス無し**）・
    **フレーム 40〜99**（暖機の 0〜39 は数えない）。クラスごとに F が最大の入りを選ぶ（候補を小さい順に見て、真に大きいときだけ替える＝同点は小さい方）。
  - 選択用に正解の無いクラスがあれば止める（入りが決まらない）。平均 F（報告用）は正解の無いクラスを外したクラス平均。
  - 参考の併記（決めには使わない）: 全クラス 0.7 の定数（入りだけ）の平均 F。
出力 = `md/design/thresholds_v30.json`（版 thresholds_v30/1・手順の写し・選択用と確認用の F・クラスごとの格子・enter／stay・入力の sha）。
  既にあれば上書きしない（作り直すなら --force・理由を PROGRESS に）。sha は契約（`_conf_exec_contract.py --thresholds-json`）に入る。
  使う所: 因果推論 `CAUSAL_THRESHOLDS_JSON=<表>`（補正表の推論・開発の比べ・実録・アブレーション）。nohyst は `CAUSAL_THRESHOLDS_MODE=nohyst`（入りは主と共有・保持 = 入り）。
  scratch・noheight・因果 ft なしは同じ手順で自分の表を作る（--label で区別・出力の名前を分ける）。
対象クラス（Fable 16:5x・クラス別は主・B・D の全員・同じ手順）: 主と B は 0〜8 の全部（既定）。D は対応表の共通クラス（--classmap md/design/baselines_class_map_v2.json --target D
  か --classes 0,1,3,4,7,8）。表の classes にその一覧を書き、enter／stay はその順。D の倍率は 40 のまま（選び直さない）・D の表は v30 の val の norms で作り直す。
  B と D の norms は各推論を BASELINE_DUMP_NORMS=1 で回した run（norms_manifest.json つき・ckpt の欄 = 重みの sha）。
  主の norms は候補 8 個の推論（v2final_candidates_infer.sbatch・v30）が取る → 選んだ候補の run をそのまま --run に渡す（もう 1 回の推論はしない）。
使い方:
  python scripts/v30_select_thresholds.py --gt <metadata_dist dir> --manifest <val の名簿 csv（clip_id 列）> --run <因果推論の run dir> --label main \
         --out md/design/thresholds_v30.json [--classes 0,1,3,4,7,8 | --classmap <対応表> --target D]
試験: scripts/test_v30_thresholds.py
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "server_sde"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import _baseline_select_gain as BSG  # noqa: E402  clip 番号・部分集合・正解の読み方は B・D と同じ
import threshold_table as TT  # noqa: E402

N_CLS, NFR, F0 = 9, 100, 40
THS = TT.CANDIDATES
VERSION = "thresholds_v30/1"
PROCEDURE = ("選んだ ckpt の因果推論の norm（3 トラックの最大・ヒステリシス前・間引かない）から、val の clip%12==0 で選び %12==6 で確認（確認用は採否に使わない）。"
             "クラスごとに候補 0.1〜0.7 の中で、フレーム 40〜99 の入りだけの検出 F が最大の入り（同点は小さい方）。保持 = round(入り × 3/7, 4)。"
             "選択用に正解の無いクラスがあれば止める。平均 F は正解の無いクラスを外す（本体 §3-5・§9-1 の 5）")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def counts(gt_dir: Path, clips: list[str], run: Path, classes: list[int]) -> dict:
    """{th: {c: [tp, fp, fn]}}（フレーム 40〜99・入りだけ）。norms の無い本・形の違う本は止める。"""
    tot = {th: {c: [0, 0, 0] for c in classes} for th in THS}
    for cl in clips:
        p = run / "norms" / f"{cl}.npy"
        if not p.is_file():
            raise SystemExit(f"norms が無い: {p}（推論を DUMP_NORMS=1 で val 全部に）")
        n = np.load(p)
        if n.shape != (3, NFR, N_CLS) or not np.all(np.isfinite(n)):
            raise SystemExit(f"norms の形が (3, {NFR}, {N_CLS}) でないか非有限: {p} {n.shape}")
        g = BSG.gt_active(gt_dir, cl, classes)[F0:]
        m = n.max(axis=0)[F0:].astype(np.float64)
        for th in THS:
            pr = m > th
            for c in classes:
                tot[th][c][0] += int((g[:, c] & pr[:, c]).sum())
                tot[th][c][1] += int((~g[:, c] & pr[:, c]).sum())
                tot[th][c][2] += int((g[:, c] & ~pr[:, c]).sum())
    return tot


def f_of(tp: int, fp: int, fn: int) -> float | None:
    if tp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def choose(cnt: dict, classes: list[int]) -> dict:
    """{c: 入り}。候補を小さい順に見て、F が真に大きいときだけ替える（同点は小さい方）。正解の無いクラスは止める。"""
    out = {}
    for c in classes:
        best, bf = None, None
        for th in THS:
            f = f_of(*cnt[th][c])
            if f is None:
                raise SystemExit(f"選択用にクラス {c} の正解が無い（入りが決まらない・止める）")
            if bf is None or f > bf + 1e-12:
                best, bf = th, f
        out[c] = best
    return out


def at(cnt: dict, choice: dict, classes: list[int]) -> tuple[float | None, dict]:
    """クラスごとの入り（choice）での F とクラス平均（正解の無いクラスは外す）。"""
    fs = {c: f_of(*cnt[choice[c]][c]) for c in classes}
    v = [f for f in fs.values() if f is not None]
    return (float(np.mean(v)) if v else None), fs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--label", required=True, help="main／scratch／noheight／nocausalft／B／D（表の持ち主）")
    ap.add_argument("--classes", default="", help="対象クラス（カンマ区切り・既定 = 0〜8 の全部）")
    ap.add_argument("--classmap", type=Path, default=None, help="D の対象クラスを対応表から（--target D と一緒に）")
    ap.add_argument("--target", default="", help="--classmap と一緒に: D（対応表の D_pseldnets_pretrained の ours_index）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="既にある表を作り直す（理由を PROGRESS に）")
    ap.add_argument("--expect-lists", nargs=2, type=Path, default=None, metavar=("SELECT_CSV", "CHECK_CSV"),
                    help="選択用・確認用が plan の manifest_val_select.csv／manifest_val_check.csv と同じ本か確かめる（違えば止める）")
    a = ap.parse_args(argv)
    if a.out.exists() and not a.force:
        raise SystemExit(f"表が既にある: {a.out}（上書きしない・作り直すなら --force）")
    nm = a.run / "norms_manifest.json"
    if not nm.is_file():
        raise SystemExit(f"norms_manifest.json が無い: {nm}（_causal_infer_v17.py を DUMP_NORMS=1 で回した run か）")
    M = json.loads(nm.read_text(encoding="utf-8"))
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    sel, chk = BSG.subsets(clips)
    if not sel or not chk:
        raise SystemExit(f"選択用（{len(sel)}）か確認用（{len(chk)}）が空")
    if a.expect_lists:                                   # plan の manifest_val_select.csv／manifest_val_check.csv と同じ本か（B・D の選択と同じ部分集合）
        for want, got, nm_ in zip(a.expect_lists, (sel, chk), ("選択用", "確認用")):
            with open(want, newline="", encoding="utf-8-sig") as f:
                w = [r["clip_id"] for r in csv.DictReader(f)]
            if w != got:
                raise SystemExit(f"{nm_}の本が {want} と違う（{len(got)} 本 ≠ {len(w)} 本）")
    miss = [c for c in sel + chk if c not in (M.get("files") or {})]
    if miss:
        raise SystemExit(f"norms_manifest に無い本がある: {len(miss)} 本（例 {miss[:2]}）")
    if a.classmap and a.classes:
        raise SystemExit("--classes と --classmap は片方だけ")
    if a.classmap:
        if a.target != "D":
            raise SystemExit("--classmap は --target D と一緒に（B と主は 0〜8 の全部）")
        cm = json.loads(a.classmap.read_text(encoding="utf-8"))
        classes = sorted({int(v["ours_index"]) for v in cm["D_pseldnets_pretrained"]["map"].values()})
    elif a.classes:
        classes = sorted({int(x) for x in a.classes.split(",") if x.strip()})
    else:
        classes = list(range(N_CLS))
    if not classes or any(not (0 <= c < N_CLS) for c in classes):
        raise SystemExit(f"対象クラスが 0〜{N_CLS - 1} の中の空でない一覧でない: {classes}")
    c_sel = counts(a.gt, sel, a.run, classes)
    c_chk = counts(a.gt, chk, a.run, classes)
    choice = choose(c_sel, classes)
    mf_sel, fs_sel = at(c_sel, choice, classes)
    mf_chk, fs_chk = at(c_chk, choice, classes)
    const = {c: 0.7 for c in classes}
    ref_sel, _ = at(c_sel, const, classes)
    ref_chk, _ = at(c_chk, const, classes)
    per = {str(c): {"in": choice[c], "stay": TT.stay_of(choice[c]), "f_select": fs_sel[c], "f_check": fs_chk[c],
                    "grid_select": {f"{th:g}": f_of(*c_sel[th][c]) for th in THS}, "counts_select": c_sel[choice[c]][c], "counts_check": c_chk[choice[c]][c]}
           for c in classes}
    out = {"version": VERSION, "label": a.label, "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "procedure": PROCEDURE, "candidates": list(THS), "frames": [F0, NFR - 1], "classes": classes,
           "select": {"rule": "clip%12==0", "n": len(sel)}, "check": {"rule": "clip%12==6", "n": len(chk), "note": "採否と再探索に使わない（併記だけ）"},
           "enter": [choice[c] for c in classes], "stay": [TT.stay_of(choice[c]) for c in classes],
           "per_class": per, "mean_f_select": mf_sel, "mean_f_check": mf_chk,
           "reference_constant_0.7_in_only": {"mean_f_select": ref_sel, "mean_f_check": ref_chk, "note": "参考（決めには使わない）"},
           "inputs": {"run": a.run.as_posix(), "norms_manifest_sha256": _sha(nm), "ckpt_sha256": M.get("ckpt_sha256"),
                      "gt_dir": a.gt.as_posix(), "manifest": a.manifest.as_posix(), "manifest_sha256": _sha(a.manifest), "n_val": len(clips)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".part")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(a.out)
    TT.load(a.out, N_CLS, "hyst", ckpt_sha=M.get("ckpt_sha256"), classes=(None if classes == list(range(N_CLS)) else classes))   # 読む側の確かめを書いた直後に
    print("入り " + " ".join(f"{c}:{choice[c]:g}" for c in classes) + f"・平均 F 選択用 {mf_sel:.4f}・確認用 {mf_chk:.4f}（参考 0.7 定数 {ref_sel:.4f}／{ref_chk:.4f}）→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
