# -*- coding: utf-8 -*-
"""検出層の 2 閾値（ヒステリシス）等の探索・第 2 版（2026-09-16 21:35）。
本人「小数点第二位まで確かめて・もっといろんなパターンで」「評価は検出層の位置の誤差などで見たい（通知層の危険度ではなく）」。

第 1 版（_hysteresis_grid.py）との違い:
  - 通知層は計算しない（時間の 62% を占めていた）。指標は知覚層（_perception_eval と同じ 1 対 1・門なし・20° 固定）の全部 ＋ ちらつき
  - --workers N で並列（Windows の spawn。各ワーカーが材料を読み直す）
  - 型（pattern）を増やした:
      amp     : 入り enter／保持 stay（振幅のヒステリシス・第 1 版と同じ）
      hold    : amp ＋ 前フレームで「いる」だった相手が今フレームに候補を持たないとき、最大 K フレームまで前の位置を出し続ける（時間方向の保持）
      confirm : amp ＋ 新しい相手は K フレーム連続で入りを超えてから出す（入りの確認・出始めの誤検出を抑える）
      ema     : 度合いを前フレームの同じ相手と α で平滑化（sm = α·今 + (1−α)·前）してから入り／保持で判定
      assoc   : 前フレームとの対応幅（°）は全型に共通の引数
  - 入り == 保持 == 0.5・hold 0・confirm 1・α 1 のとき、本番の判定（sed05）と同じ

使い方:
  python scripts/_hysteresis_grid2.py <出力md> --plan <assignment.csv> [--split fold2] --meta <GTディレクトリ> --scores <scores dir>
      --csv <格子のcsv> [--workers 6] [--minframe 40]
      --enter 0.40,0.41,... --stay 0.05,0.10,...,e   （e = 入りと同じ値 = 1 閾値）
      [--hold 0,1,2] [--confirm 1,2,3] [--alpha 1,0.5,0.3] [--assoc 20,15,30]
      [--tasks tasks.json]   （直積の代わりに組を列挙: [{"enter":0.7,"stay":0.3,"assoc":20,"hold":0,"confirm":1,"alpha":1.0}, ...]）
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


D = _load("detect_sweep", "_detect_threshold_sweep.py")
PE = D.PE

FIELDS = ["pattern", "enter", "stay", "assoc", "hold", "confirm", "alpha",
          "recall_any", "recall20", "prec20", "f1", "n_pred",
          "az_med", "az_mean", "az_rmse", "az_p90", "az_le20", "az_ge150",
          "d_med", "d_mean", "d_rmse", "rel_med", "xy_med", "xy_mean", "xy_rmse", "n_dist",
          "drops_per_100", "sec"]


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def pattern_name(t) -> str:
    parts = []
    if t["hold"] > 0:
        parts.append("hold")
    if t["confirm"] > 1:
        parts.append("confirm")
    if t["alpha"] < 1.0:
        parts.append("ema")
    return "+".join(parts) if parts else "amp"


class NullAcc(PE.Acc):
    """暖機込み（acc0）の記録は要らないので捨てる（時間の節約）"""

    def add_pair(self, *a, **k):
        pass

    def add_miss(self, *a, **k):
        pass


def decode_general(frames_scores, enter: float, stay: float, assoc: float = 20.0, hold: int = 0, confirm: int = 1, alpha: float = 1.0):
    """状態つきの組み立て直し。frame -> [(cls, az, el, d, score)]（本番と同じ統合を通す）。
    状態 = 前フレームの候補（いる／いない・連続数・平滑化した度合い）と、時間保持で持ち越した相手。"""
    if not frames_scores:
        return {}
    plain = (hold == 0 and confirm <= 1 and alpha >= 1.0)
    if plain and stay >= enter:
        return D.decode(frames_scores, enter)           # 1 閾値 = 本番の判定（0.5 なら sed05）
    use_sed05 = (enter == 0.5 and alpha >= 1.0)
    kmax = max(frames_scores)
    prev = []
    out = {}
    for k in range(0, kmax + 1):
        rows = frames_scores.get(k, [])
        cur = []
        used = set()
        for r in rows:
            trk, cls, nrm, x, y, z, d, s05 = r
            az = math.degrees(math.atan2(y, x))
            best = None; bd = assoc + 1.0
            for i, p in enumerate(prev):
                if p["cls"] == cls:                      # 同じ前の相手に複数の候補が対応してよい（3 本の矢印は同じ相手に重なることがある）
                    dd = PE.dang(az, p["az"])
                    if dd <= assoc and dd < bd:
                        best, bd = i, dd
            sm = nrm if (alpha >= 1.0 or best is None) else alpha * nrm + (1.0 - alpha) * prev[best]["sm"]
            is_enter = (s05 == 1) if use_sed05 else (sm > enter)
            if best is not None:
                used.add(best)
                p = prev[best]
                if is_enter:
                    streak, active = p["streak"] + 1, True
                elif p["active"] and sm > stay:
                    streak, active = p["streak"], True
                else:
                    streak, active = 0, False
            else:
                streak, active = (1, True) if is_enter else (0, False)
            cur.append(dict(cls=cls, az=az, row=r, sm=sm, streak=streak, miss=0, active=active))
        if hold > 0:
            # 時間保持: 前フレームで「いる」だった相手に今フレームの候補が無いときだけ、前の位置を持ち越す。
            # 今フレームの候補（いる／いないを問わず）が対応幅以内にいるなら持ち越さない（同じ相手の二重出力を防ぐ）
            for i, p in enumerate(prev):
                if i in used or not p["active"] or p["streak"] < confirm or p["miss"] >= hold:
                    continue
                if any(t["cls"] == p["cls"] and PE.dang(t["az"], p["az"]) <= assoc for t in cur):
                    continue
                cur.append(dict(cls=p["cls"], az=p["az"], row=p["row"], sm=p["sm"], streak=p["streak"], miss=p["miss"] + 1, active=True))
        act_t = [t for t in cur if t["active"] and t["streak"] >= confirm]
        if hold > 0:
            # 本番の統合は種類ごとに最大 3 本しか扱わない。超えるときは持ち越した相手から捨てる
            by = defaultdict(list)
            for t in act_t:
                by[t["cls"]].append(t)
            act_t = []
            for cls, ts in by.items():
                ts.sort(key=lambda t: t["miss"])
                act_t += ts[:3]
        act = [t["row"] for t in act_t]
        if act:
            dec = D.decode({k: act}, -1.0)
            res = dec.get(k, [])
            if res:
                out[k] = res
        prev = cur
    return out


def flicker(frames, gt, minframe: int):
    """正解の (cls, trk) ごとに、対応する予測（同じ種類・20° 以内）が「あり→なし」に落ちた回数と、正解フレーム数"""
    drops = 0; n = 0
    tracks = defaultdict(list)
    for k in sorted(gt):
        for g in gt[k]:
            if g[0] in PE.DIST_CLS:
                tracks[(g[0], g[1])].append((k, g[2]))
    for (cls, trk), seq in tracks.items():
        was = None
        for k, gaz in seq:
            if k < minframe:
                continue
            n += 1
            has = any(c == cls and PE.dang(az, gaz) <= 20.0 for c, az, el, d in frames.get(k, []))
            if was is True and not has:
                drops += 1
            was = has
    return drops, n


# ---- ワーカー側 ----
G = {}


def _init(plan, split, meta, sdir, minframe):
    rows = [r for r in csv.DictReader(open(plan, newline="", encoding="utf-8")) if r.get("split", split) == split]
    clips = [r["clip_id"] for r in rows]
    G["clips"] = clips
    G["motion"] = {r["clip_id"]: r.get("motion", "") or "-" for r in rows}
    G["scores"] = {c: D.load_scores(Path(sdir) / f"{c}.csv") for c in clips}
    G["gts"] = {c: PE.load_gt_all(Path(meta), c) for c in clips}
    G["minframe"] = minframe


def evaluate(t: dict) -> dict:
    t0 = time.time()
    acc, acc0 = PE.Acc(), NullAcc()
    drops = 0; nfr = 0
    mf = G["minframe"]
    for c in G["clips"]:
        dec = decode_general(G["scores"][c], t["enter"], t["stay"], t["assoc"], t["hold"], t["confirm"], t["alpha"])
        frames = {k: [(cls, az, el, d) for cls, az, el, d, _ in v] for k, v in dec.items()}
        PE.clip_eval(acc, acc0, frames, G["gts"][c], G["motion"][c], mf)
        dd, nn = flicker(frames, G["gts"][c], mf); drops += dd; nfr += nn
    det = PE.stats(acc, "距離クラス")
    n_pred = sum(v for k, v in acc.n_pred.items() if k in PE.DIST_CLS)
    prec20 = 100.0 * det["n_tp20"] / max(n_pred, 1)
    f1 = 2 * prec20 * det["recall20"] / max(prec20 + det["recall20"], 1e-9)
    row = dict(t)
    row["pattern"] = pattern_name(t)
    row.update(recall_any=det["recall_any"], recall20=det["recall20"], prec20=prec20, f1=f1, n_pred=n_pred,
               drops_per_100=100.0 * drops / max(nfr, 1), sec=time.time() - t0)
    for key in ["az_med", "az_mean", "az_rmse", "az_p90", "az_le20", "az_ge150", "d_med", "d_mean", "d_rmse", "rel_med", "xy_med", "xy_mean", "xy_rmse", "n_dist"]:
        row[key] = det.get(key, float("nan"))
    return row


# ---- 親側 ----
def build_tasks(argv) -> list[dict]:
    tasks_json = arg(argv, "--tasks", "")
    if tasks_json:
        tasks = json.load(open(tasks_json, encoding="utf-8"))
        for t in tasks:
            t.setdefault("assoc", 20.0); t.setdefault("hold", 0); t.setdefault("confirm", 1); t.setdefault("alpha", 1.0)
    else:
        enters = [round(float(x), 4) for x in arg(argv, "--enter", "0.5,0.7,0.9").split(",")]
        stays_raw = arg(argv, "--stay", "0.3,e").split(",")
        holds = [int(x) for x in arg(argv, "--hold", "0").split(",")]
        confirms = [int(x) for x in arg(argv, "--confirm", "1").split(",")]
        alphas = [float(x) for x in arg(argv, "--alpha", "1").split(",")]
        assocs = [float(x) for x in arg(argv, "--assoc", "20").split(",")]
        tasks = []
        for e in enters:
            stays = sorted({e if s.strip() == "e" else round(float(s), 4) for s in stays_raw})
            for s in stays:
                if s > e + 1e-9:
                    continue
                for h in holds:
                    for cf in confirms:
                        for a in alphas:
                            for w in assocs:
                                tasks.append(dict(enter=e, stay=s, assoc=w, hold=h, confirm=cf, alpha=a))
    base = dict(enter=0.5, stay=0.5, assoc=20.0, hold=0, confirm=1, alpha=1.0)
    key = lambda t: (t["enter"], t["stay"], t["assoc"], t["hold"], t["confirm"], t["alpha"])
    seen = set(); uniq = []
    for t in [base] + tasks:
        if key(t) in seen:
            continue
        seen.add(key(t)); uniq.append(t)
    return uniq


def fmt_row(x, tag=""):
    return (f"| {x['pattern']} | {x['enter']:.2f} | {x['stay']:.2f} | {x['assoc']:.0f} | {x['hold']} | {x['confirm']} | {x['alpha']:.2f} | {tag} | "
            f"{x['recall_any']:.2f} | {x['recall20']:.2f} | {x['prec20']:.2f} | **{x['f1']:.2f}** | "
            f"{x['az_med']:.1f} / {x['az_mean']:.2f} / {x['az_rmse']:.2f} | {x['az_ge150']:.2f} | "
            f"{x['d_med']:.2f} / {x['d_mean']:.2f} / {x['d_rmse']:.2f} | {x['rel_med']:.2f} | "
            f"{x['xy_med']:.2f} / {x['xy_mean']:.2f} / {x['xy_rmse']:.2f} | {x['drops_per_100']:.2f} |")


HDR = ["| 型 | 入り | 保持 | 対応幅 | 保持fr | 確認fr | α | 印 | 検出率 | 再現率(20°固定) | 適合率(20°固定) | F1(20°固定) | 方位誤差 中央/平均/RMSE[°] | 前後取り違え[%] | 距離誤差 中央/平均/RMSE[m] | 相対誤差中央[%] | 位置誤差 中央/平均/RMSE[m] | ちらつき(落ち/100fr) |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]


def summarize(grid, out_md: Path, header: str):
    base = next(x for x in grid if x["enter"] == 0.5 and x["stay"] == 0.5 and x["hold"] == 0 and x["confirm"] == 1 and x["alpha"] >= 1.0 and x["assoc"] == 20.0)
    ok_r = lambda x, m: x["recall20"] >= base["recall20"] - m
    ok_p = lambda x, m: x["prec20"] >= base["prec20"] - m
    picks = [(base, "現行（1 閾値 0.5・sed05）")]
    picks.append((max(grid, key=lambda x: (round(x["f1"], 2), -x["drops_per_100"])), "**F1(20° 固定) 最大**"))
    c = [x for x in grid if ok_r(x, 0.5)]
    picks.append((min(c, key=lambda x: (x["xy_med"], x["xy_rmse"])), "位置誤差(中央) 最小（再現率 ≥ 現行 −0.5）"))
    picks.append((min(c, key=lambda x: (x["xy_rmse"], x["xy_med"])), "位置誤差(RMSE) 最小（再現率 ≥ 現行 −0.5）"))
    picks.append((min(c, key=lambda x: (x["az_rmse"], x["az_mean"])), "方位誤差(RMSE) 最小（再現率 ≥ 現行 −0.5）"))
    picks.append((min(c, key=lambda x: (x["d_rmse"], x["d_med"])), "距離誤差(RMSE) 最小（再現率 ≥ 現行 −0.5）"))
    c2 = [x for x in grid if ok_r(x, 0.0) and ok_p(x, 1.0)]
    if c2:
        picks.append((min(c2, key=lambda x: (x["drops_per_100"], -x["recall20"])), "ちらつき 最小（再現率 ≥ 現行・適合率 ≥ 現行 −1）"))
    c3 = [x for x in grid if ok_p(x, 1.0)]
    picks.append((max(c3, key=lambda x: (x["recall20"], -x["drops_per_100"])), "再現率 最大（適合率 ≥ 現行 −1）"))
    c4 = [x for x in grid if ok_r(x, 1.0)]
    picks.append((max(c4, key=lambda x: (x["prec20"], -x["drops_per_100"])), "適合率 最大（再現率 ≥ 現行 −1）"))
    lines = [f"# 検出層の 2 閾値等の探索（第 2 版・知覚層の指標だけ） — {out_md.stem}", "", header, "",
             f"組数 {len(grid):,}。指標は _perception_eval と同じ（距離クラス・1 対 1・門なし・20° 固定）。ちらつき = 正解の相手ごとの あり→なし の回数 / 正解フレーム 100 個。", "",
             "## 1. 現行と、選ばれた組", "", *HDR]
    for x, tag in picks:
        lines.append(fmt_row(x, tag))
    lines += ["", "## 2. 型ごとの最良（F1 最大）と、その型の組数", "", *HDR]
    for pat in sorted({x["pattern"] for x in grid}):
        g = [x for x in grid if x["pattern"] == pat]
        b = max(g, key=lambda x: (round(x["f1"], 2), -x["drops_per_100"]))
        lines.append(fmt_row(b, f"{pat}（{len(g)} 組）"))
    if len(grid) <= 150:
        lines += ["", "## 3. 全格子", "", *HDR, *[fmt_row(x) for x in sorted(grid, key=lambda x: (x['pattern'], x['enter'], x['stay'], x['assoc'], x['hold'], x['confirm'], x['alpha']))]]
    else:
        lines += ["", f"## 3. 全格子は csv（{len(grid):,} 組）"]
    lines += ["", "## 定義", "",
              "- 入り = この長さを超えたら「いる」。保持 = 前フレームで「いる」だった同じ種類の相手（対応幅以内）は、この長さを下回るまで「いる」を続ける。入り == 保持 なら 1 閾値（0.5 なら本番の判定 sed05）",
              "- 保持fr（hold）= 前フレームで「いる」だった相手に今フレームの候補が無いとき、最大この数のフレームまで前の位置を出し続ける",
              "- 確認fr（confirm）= 新しい相手はこの数のフレーム連続で入りを超えてから出す",
              "- α（ema）= 度合いの平滑化 sm = α·今 + (1−α)·前（同じ種類・対応幅以内の前フレームの相手）。1 = 平滑化なし",
              "- 通知層は計算しない（本人 9/16 21:33「検出層の位置の誤差などで見たい」）。開発 val での調整。選んだ値は確定評価の前に固定する"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    argv = list(sys.argv[1:])
    plan = ROOT / arg(argv, "--plan")
    split = arg(argv, "--split", "fold2")
    meta = ROOT / arg(argv, "--meta")
    sdir = ROOT / arg(argv, "--scores")
    workers = int(arg(argv, "--workers", "1"))
    minframe = int(arg(argv, "--minframe", "40"))
    out_csv = arg(argv, "--csv", "")
    tasks = build_tasks(argv)
    out_md = Path(argv[0])
    rows = [r for r in csv.DictReader(open(plan, newline="", encoding="utf-8")) if r.get("split", split) == split]
    clips = [r["clip_id"] for r in rows]
    D.check_manifest(sdir, clips)
    header = f"plan= {plan.relative_to(ROOT)} ({split}) = {len(clips):,} 本 / GT= {meta.relative_to(ROOT)} / scores= {sdir.relative_to(ROOT)} / frame≥{minframe} / workers {workers}"
    print(f"{len(tasks)} 組 / {len(clips):,} 本 / workers {workers}", flush=True)
    grid = []
    fh = open(out_csv, "w", newline="", encoding="utf-8") if out_csv else None
    wr = csv.DictWriter(fh, fieldnames=FIELDS) if fh else None
    if wr:
        wr.writeheader(); fh.flush()
    t0 = time.time()

    def take(row):
        grid.append(row)
        if wr:
            wr.writerow({k: row.get(k) for k in FIELDS}); fh.flush()
        n = len(grid)
        if n % 10 == 0 or n == len(tasks) or n <= 3:
            el = time.time() - t0
            print(f"[{n}/{len(tasks)} {el/60:.1f}分 残り≈{el/n*(len(tasks)-n)/60:.1f}分] {row['pattern']} 入り {row['enter']:.2f} 保持 {row['stay']:.2f} h{row['hold']} c{row['confirm']} α{row['alpha']:.2f} w{row['assoc']:.0f}: "
                  f"再現率20° {row['recall20']:.2f} 適合率 {row['prec20']:.2f} F1 {row['f1']:.2f} 方位RMSE {row['az_rmse']:.2f} 位置中央 {row['xy_med']:.3f} ちらつき {row['drops_per_100']:.2f}", flush=True)

    if workers <= 1:
        _init(str(plan), split, str(meta), str(sdir), minframe)
        for t in tasks:
            take(evaluate(t))
    else:
        with Pool(workers, initializer=_init, initargs=(str(plan), split, str(meta), str(sdir), minframe)) as pool:
            for row in pool.imap_unordered(evaluate, tasks, chunksize=1):
                take(row)
    if fh:
        fh.close()
    summarize(grid, out_md, header)
    print(f"-> {out_md}  ({(time.time()-t0)/60:.1f} 分)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
