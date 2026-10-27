# -*- coding: utf-8 -*-
"""相手 D の入力ゲインと検出の閾値を、合成の val の固定の部分集合だけで選ぶ（変更票 #73・Q4 = Astra R73-05 の手順・2026-10-21・Fable）。

手順（成績を見る前に固定・ここに書いたとおり）:
  - 選ぶ標本 = val の clip 番号を 12 で割った余りが 0 の本（約 1,035 本）。確認用 = 余りが 6 の本（同数・選択には使わない・結果の併記だけ）。
  - ゲインの候補 = {0, 20, 30, 40, 50, 60, 70} dB。閾値（入り）の候補 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}。保持 = 入りの 3/7（この研究の 0.7／0.3 と同じ比）。
  - 対象 = 相手と共通のクラス（対応表の map に入っている対象クラス）。
  - 指標 = クラスごとの「フレーム単位の検出 F」（正解: metadata_dist にそのクラスの行がある frame・予測: そのクラスの norm > 入り〔ヒステリシスはここでは入りだけ〕）の
          クラス平均（見逃しと誤検出を含む）。正解の無いクラスは平均から外す。
  - 探索順 = ゲイン昇順 × 閾値昇順。同点は小さいゲイン → 小さい閾値。既定（0 dB・0.5）の成績も併記。
  - 入力 = D の推論で BASELINE_DUMP_NORMS=1 のときに残す norms/<clip>.npy（3 トラック・T・9 の norm・ゲインごとに別の run）。
使い方:
  python scripts/_baseline_select_gain.py --gt <metadata_dist dir> --manifest <val の名簿 csv> --classmap md/design/baselines_class_map_v2.json --target D
         --runs 0=<run dir of gain 0> 20=<run dir of gain 20> ... --out <json>
  python scripts/_baseline_select_gain.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GAINS = (0, 20, 30, 40, 50, 60, 70)
THS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MOD, SEL_R, CHK_R = 12, 0, 6
N_OURS, NFR = 9, 100


def clip_number(cid: str) -> int:
    import re
    m = re.search(r"clip(\d+)$", cid)
    if not m:
        raise ValueError(f"clip 番号が読めない: {cid}")
    return int(m.group(1))


def subsets(clips: list[str]) -> tuple[list[str], list[str]]:
    sel = [c for c in clips if clip_number(c) % MOD == SEL_R]
    chk = [c for c in clips if clip_number(c) % MOD == CHK_R]
    return sel, chk


def gt_active(gt_dir: Path, clip: str, classes: list[int]) -> np.ndarray:
    """(NFR, 9) bool: そのクラスの行がある frame。"""
    a = np.zeros((NFR, N_OURS), dtype=bool)
    p = gt_dir / f"{clip}.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.strip().split(",")
        if len(q) >= 2 and q[0]:
            k, c = int(float(q[0])), int(float(q[1]))
            if 0 <= k < NFR and c in classes:
                a[k, c] = True
    return a


def pred_max_norm(run_dir: Path, clip: str) -> np.ndarray:
    """norms/<clip>.npy（3, T, 9）→ (T, 9) トラックの最大。"""
    n = np.load(run_dir / "norms" / f"{clip}.npy")
    return n.max(axis=0)


def f_per_class(gt: np.ndarray, pr: np.ndarray, classes: list[int]) -> dict[int, tuple[int, int, int]]:
    out = {}
    for c in classes:
        tp = int((gt[:, c] & pr[:, c]).sum()); fp = int((~gt[:, c] & pr[:, c]).sum()); fn = int((gt[:, c] & ~pr[:, c]).sum())
        out[c] = (tp, fp, fn)
    return out


def mean_f(counts: dict[int, tuple[int, int, int]]) -> tuple[float | None, dict]:
    fs = {}
    for c, (tp, fp, fn) in counts.items():
        if tp + fn == 0:
            continue                                   # 正解の無いクラスは外す
        fs[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return (float(np.mean(list(fs.values()))) if fs else None), fs


def evaluate(gt_dir: Path, clips: list[str], runs: dict[int, Path], classes: list[int], missing_ok: bool = False) -> dict:
    """missing_ok=True（確認用の部分集合だけ）: norms の無い本は飛ばして n_missing に数える。選択用では飛ばさない（無ければ止まる）。"""
    res = {}
    for g in sorted(runs):
        have = [c for c in clips if (runs[g] / "norms" / f"{c}.npy").is_file()]
        if len(have) != len(clips) and not missing_ok:
            raise FileNotFoundError(f"ゲイン {g}: norms が無い本がある（{len(clips) - len(have)} 本・例 {[c for c in clips if c not in set(have)][:2]}）")
        gts = {c: gt_active(gt_dir, c, classes) for c in have}
        norms = {c: pred_max_norm(runs[g], c) for c in have}
        n_missing = len(clips) - len(have)
        for th in THS:
            tot = {c: [0, 0, 0] for c in classes}
            for c in have:
                pr = norms[c] > th
                for k, v in f_per_class(gts[c], pr, classes).items():
                    for i in range(3):
                        tot[k][i] += v[i]
            mf, fs = mean_f({k: tuple(v) for k, v in tot.items()})
            res[f"{g}:{th}"] = {"gain_db": g, "th": th, "stay": round(th * 3 / 7, 4), "mean_f": (mf if have else None), "f_per_class": fs, "counts": {str(k): v for k, v in tot.items()},
                                "n_clips": len(have), "n_missing": n_missing}
    return res


def choose(res: dict) -> dict | None:
    best = None
    for key in sorted(res, key=lambda k: (res[k]["gain_db"], res[k]["th"])):          # ゲイン昇順 × 閾値昇順・同点は先に来た方（小さいゲイン → 小さい閾値）
        r = res[key]
        if r["mean_f"] is None:
            continue
        if best is None or r["mean_f"] > best["mean_f"] + 1e-12:
            best = dict(r, key=key)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path); ap.add_argument("--manifest", type=Path); ap.add_argument("--classmap", type=Path); ap.add_argument("--target", default="D")
    ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--out", type=Path); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gains", default="", help="候補のゲイン（カンマ区切り・既定 = {0,20,30,40,50,60,70}）。相手 B はゲイン無し → --gains 0（閾値だけ同じ手順で選ぶ）")
    ap.add_argument("--check-runs", nargs="*", default=[], help="確認用（%12==6）の norms を別の run に置いたとき: <ゲイン>=<run dir>（無ければ --runs と同じ run の norms を探す）")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    gains = tuple(int(x) for x in a.gains.split(",") if x.strip()) if a.gains else GAINS
    if a.target == "B":                                      # B は 9 クラス全部を出す（対応表は要らない）
        classes = list(range(9))
    else:
        cm = json.loads(a.classmap.read_text(encoding="utf-8"))
        key = {"D": "D_pseldnets_pretrained", "A": "A_panns_audioset"}[a.target]
        classes = sorted(int(v["ours_index"]) for v in cm[key]["map"].values())
    with open(a.manifest, encoding="utf-8", newline="") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    sel, chk = subsets(clips)
    runs = {}
    for item in a.runs:
        g, p = item.split("=", 1); runs[int(g)] = Path(p)
    missing = [g for g in gains if g not in runs]
    extra = [g for g in runs if g not in gains]
    if missing or extra:
        raise SystemExit(f"ゲインの候補の run が足りない／余分: 足りない {missing}・余分 {extra}（候補 {list(gains)}）")
    res_sel = evaluate(a.gt, sel, runs, classes)
    best = choose(res_sel)
    check_runs = dict(runs)
    for item in a.check_runs:
        g, p = item.split("=", 1); check_runs[int(g)] = Path(p)          # 確認用の norms だけ別の run（baseline_D_select.sbatch の D_SUBSET=check）
    res_chk = evaluate(a.gt, chk, {best["gain_db"]: check_runs[best["gain_db"]]}, classes, missing_ok=True) if best else {}   # 確認用は選んだゲインの norms が揃ってから（無い分は n_missing）
    chk_rec = res_chk.get(f"{best['gain_db']}:{best['th']}") if best else None
    out = {"procedure": f"val clip%12==0 で選ぶ・%12==6 で確認・ゲイン {list(gains)}・閾値 {list(THS)}・クラス平均の frame F・同点は小さいゲイン→小さい閾値", "gains": list(gains),
           "check_runs_override": {str(g): p.as_posix() for g, p in check_runs.items() if p != runs.get(g)},
           "target": a.target, "classes": classes, "n_select": len(sel), "n_check": len(chk), "default": res_sel.get("0:0.5"), "selected": best,
           "check_at_selected": chk_rec, "check_note": (None if (chk_rec and chk_rec["n_missing"] == 0) else "確認用の部分集合は選んだゲインで未推論（n_missing）→ 推論してから再実行で埋める"),
           "all_select": res_sel}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"選んだ: ゲイン {best['gain_db']} dB・入り {best['th']}・保持 {best['stay']}・平均 F {best['mean_f']:.4f}（既定 0 dB・0.5 = {out['default']['mean_f']}）→ {a.out}" if best else "選べない（正解が無い）")
    return 0


def selftest() -> int:
    import tempfile
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="selgain_"))
    gt = tmp / "gt"; gt.mkdir()
    clips = [f"val_syn_clip{i:05d}" for i in range(1, 49)]
    sel, chk = subsets(clips)
    ok.append(("部分集合: 48 本から 余り 0 の 4 本と 余り 6 の 4 本", len(sel) == 4 and len(chk) == 4 and sel[0] == "val_syn_clip00012" and chk[0] == "val_syn_clip00006"))
    classes = [0, 4]
    rng = np.random.default_rng(0)
    runs = {}
    for g in GAINS:
        d = tmp / f"run_{g}"; (d / "norms").mkdir(parents=True)
        runs[g] = d
    for c in clips:
        act = np.zeros((NFR, N_OURS), dtype=bool); act[20:60, 4] = True            # 車が 20〜59 frame
        (gt / f"{c}.csv").write_text("".join(f"{k},4,0,30,0,2.0\n" for k in range(20, 60)), encoding="utf-8")
        for g in GAINS:
            n = np.zeros((3, NFR, N_OURS), dtype=np.float32)
            n[0, :, 4] = 0.15                                                           # 常にある雑音の反応（閾値 0.1 では誤検出になる）
            amp = {0: 0.05, 20: 0.2, 30: 0.35, 40: 0.65, 50: 0.65, 60: 0.9, 70: 0.9}[g]      # 30 dB で初めて雑音より十分大きい・60 以上は誤検出も増える
            n[0, 20:60, 4] = max(amp, 0.15)
            if g >= 60:
                n[0, 0:20, 4] = 0.8                                                      # 誤検出
            np.save(d := runs[g] / "norms" / f"{c}.npy", n)
    res = evaluate(gt, sel, runs, classes)
    best = choose(res)
    ok.append(("30 dB・入り 0.2 が最良（F 1.0 の中で最小のゲイン → 最小の閾値。20 dB は 0.1 で誤検出・0.2 で見逃し）", best["gain_db"] == 30 and best["th"] == 0.2 and abs(best["mean_f"] - 1.0) < 1e-9))
    ok.append(("既定（0 dB・0.5）は F 0（norm 0.05 は閾値未満）", res["0:0.5"]["mean_f"] == 0.0))
    ok.append(("正解の無いクラス（0）は平均から外す", "0" not in {str(k) for k in best["f_per_class"]} and 4 in best["f_per_class"]))
    ok.append(("保持 = 入りの 3/7", abs(best["stay"] - best["th"] * 3 / 7) < 1e-3))
    ok.append(("60 dB 以上は誤検出で F が落ちる", res["60:0.6"]["mean_f"] < 1.0 and res["70:0.6"]["mean_f"] < 1.0))
    # 相手 B（ゲイン無し・9 クラス全部）: --target B --gains 0 で閾値だけを同じ手順で選ぶ・候補に無いゲインの run を渡すと止まる
    man = tmp / "manifest.csv"; man.write_text("clip_id\n" + "".join(f"{c}\n" for c in clips), encoding="utf-8")
    outj = tmp / "selB.json"
    rc = main(["--gt", str(gt), "--manifest", str(man), "--target", "B", "--gains", "0", "--runs", f"0={runs[0]}", "--out", str(outj)])
    J = json.loads(outj.read_text(encoding="utf-8")) if outj.is_file() else {}
    ok.append(("B: --gains 0 で閾値だけ選ぶ（クラスは 9 つ全部・候補のゲインは [0]・選んだゲインは 0・既定 0:0.5 も併記）",
               rc == 0 and J.get("gains") == [0] and J.get("classes") == list(range(9)) and (J.get("selected") or {}).get("gain_db") == 0 and "default" in J))
    try:
        main(["--gt", str(gt), "--manifest", str(man), "--target", "B", "--gains", "0", "--runs", f"0={runs[0]}", f"20={runs[20]}", "--out", str(tmp / "selB2.json")])
        rc2 = 0
    except SystemExit as e:
        rc2 = 1 if e.code else 0
    ok.append(("B: 候補に無いゲインの run を渡すと止まる", rc2 == 1))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
