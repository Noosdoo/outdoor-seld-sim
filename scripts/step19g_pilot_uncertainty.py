# -*- coding: utf-8 -*-
"""学内パイロットの「正解の確かさ」を数字にする道具（2026-09-24・Fable）。

背景: 実録の正解の不確かさは、いまの文書ではすべて「テープの幾何だけの見込み・未測定」（`実録_正解座標の一覧` §5a・再監査 9/22 R05・
Opus 5.5 監査 9/23 F07）。学内パイロットの検証用 10 本で次の 4 つを測り、その実測値で文書の見込みを差し替える。

  A. 読みの一致  … 2 人が別々に同じ動画を読んだ表（take,frame,cls,track,x,y）の差。目印ごとの通過時刻の差と、正解に直した後の方位・距離の差
  B. 格子読みとの差 … 台本方式（目印 5 点の通過時刻 → 直線補間）と、0.5 秒ごとに路面の格子で読んだ位置（独立に測った点）の差。最接近距離の差も出す
  C. 横ずれ       … 線の端から線に沿って撮った動画（または目印 0 の真横の地上カメラ）で読んだ「前輪（車は中心）が線からどれだけ離れたか」
  D. 同期         … テイクの最初と最後の 2 回の拍手で出した「音の時刻 − 動画の時刻」の差（同じなら同期は安定。違えば可変フレームレートのずれ）

入力はどれも小さな CSV（列は各コマンドの docstring）。出力は md（表）と json（数値）。数字は「実測値（n 本）」として書き、見込みの語は使わない。

使い方:
  python scripts/step19g_pilot_uncertainty.py readers   <A.csv> <B.csv> <出力dir> [--mic-z 1.6] [--src-z 0.5]
  python scripts/step19g_pilot_uncertainty.py grid      <台本読み.csv> <格子読み.csv> <出力dir> [--mic-z 1.6] [--src-z 0.5]
  python scripts/step19g_pilot_uncertainty.py lateral   <横ずれ.csv> <出力dir>
  python scripts/step19g_pilot_uncertainty.py sync      <拍手.csv> <出力dir>
  python scripts/step19g_pilot_uncertainty.py selftest  <出力dir>          # 合成データで 4 つとも通す
横ずれ.csv の列 = take,cls,line_m,x_marker,offset_m（offset_m は線からの横のずれ [m]・マイクから遠ざかる向きが正）
拍手.csv の列  = take,clap,audio_s,video_s（clap = 1 が最初・2 が最後）
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location("step19f_video_to_gt", ROOT / "scripts" / "step19f_video_to_gt.py")
F19 = importlib.util.module_from_spec(_spec); sys.modules["step19f_video_to_gt"] = F19; _spec.loader.exec_module(F19)

TIER_EDGES = (1.5, 3.2)     # 採点の区分（≤1.5 至近／≤3.2 注意／>3.2 安全・step20_realsmoke_score_v3.py と同じ）
LABEL_RES = F19.LABEL_RES


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if len(v) else float("nan")


def summ(v) -> dict:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "p95_abs": pct(np.abs(v), 95), "max_abs": float(np.max(np.abs(v)))}


def fmt(s: dict, unit: str, digits: int = 2) -> str:
    if not s.get("n"):
        return "—（n=0）"
    return (f"中央 {s['median']:+.{digits}f}{unit}・平均 {s['mean']:+.{digits}f}{unit}・SD {s['sd']:.{digits}f}{unit}・"
            f"|差| の 95% 点 {s['p95_abs']:.{digits}f}{unit}・最大 {s['max_abs']:.{digits}f}{unit}（n={s['n']}）")


def gt_rows_by_take(rows: list[dict], mic_z: float, src_z: float) -> dict:
    """読み取り表 → take -> {(cls, track, frame): (az, el, dist)}（step19f と同じ変換）"""
    out = {}
    for take, lines in F19.convert(rows, mic_z=mic_z, src_z=src_z).items():
        d = {}
        for s in lines:
            k, c, t, az, el, dist = s.split(",")
            d[(int(c), int(t), int(k))] = (float(az), float(el), float(dist))
        out[take] = d
    return out


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def compare_gt(g1: dict, g2: dict) -> dict:
    """2 つの正解（同じ take の集合）を、共通フレームで比べる。方位差・距離差・最接近距離の差"""
    daz, dd, dcpa, n_takes = [], [], [], 0
    per_take = {}
    for take in sorted(set(g1) & set(g2)):
        a, b = g1[take], g2[take]
        common = sorted(set(a) & set(b))
        if not common:
            continue
        n_takes += 1
        az_t = [dang(a[k][0], b[k][0]) for k in common]
        d_t = [a[k][2] - b[k][2] for k in common]
        daz += az_t; dd += d_t
        # 最接近距離 = トラックごとの水平距離の最小
        cpa = []
        for (c, t) in {(c, t) for c, t, _ in common}:
            ka = [a[k][2] for k in a if k[0] == c and k[1] == t]; kb = [b[k][2] for k in b if k[0] == c and k[1] == t]
            if ka and kb:
                cpa.append(min(ka) - min(kb))
        dcpa += cpa
        per_take[take] = {"n_frames": len(common), "az_abs_p95": pct(az_t, 95), "dist_p95_abs": pct(np.abs(d_t), 95), "cpa_diff": cpa}
    return {"n_takes": n_takes, "az_abs_deg": summ(daz), "dist_m": summ(dd), "cpa_m": summ(dcpa), "per_take": per_take}


def _tsec(r: dict) -> float:
    """秒の実数（t 列があればそれ・無ければ frame × 0.1）"""
    return float(r["t"]) if r.get("t") not in (None, "", "nan") else int(r["frame"]) * LABEL_RES


def marker_time_diffs(ra: list[dict], rb: list[dict]) -> dict:
    """同じ take・cls・track・(x, y) の点の時刻の差（読みの一致・目印ごと）。t 列（秒の実数・step19h の出力）があれば丸めずに比べる"""
    key = lambda r: (r["take"].strip(), F19.cls_of(r["cls"]), int(r.get("track", 0) or 0), round(float(r["x"]), 2), round(float(r["y"]), 2))
    A = {key(r): _tsec(r) for r in ra}
    B = {key(r): _tsec(r) for r in rb}
    common = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in common]
    return {"n_points": len(common), "frame_diff": summ([x / LABEL_RES for x in d]), "sec_diff": summ(d)}


# ---------------------------------------------------------------- A. 読みの一致
def cmd_readers(a_csv: Path, b_csv: Path, out: Path, mic_z: float, src_z: float) -> dict:
    ra, rb = read_csv(a_csv), read_csv(b_csv)
    md = marker_time_diffs(ra, rb)
    g = compare_gt(gt_rows_by_take(ra, mic_z, src_z), gt_rows_by_take(rb, mic_z, src_z))
    res = {"kind": "readers", "a": str(a_csv), "b": str(b_csv), "marker": md, "gt": g}
    L = ["# A. 読みの一致（2 人が別々に読んだ表の差・実測値）", "",
         f"表 A = `{a_csv.name}`・表 B = `{b_csv.name}`・共通の点 {md['n_points']}・共通のテイク {g['n_takes']}", "",
         "| 何の差 | 実測値（A − B） |", "| --- | --- |",
         f"| 目印の通過時刻（frame・0.1 s） | {fmt(md['frame_diff'], ' frame', 1)} |",
         f"| 同（秒） | {fmt(md['sec_diff'], ' s', 3)} |",
         f"| 正解に直した後の方位差（共通フレーム） | {fmt(g['az_abs_deg'], '°', 1)} |",
         f"| 正解に直した後の水平距離の差 | {fmt(g['dist_m'], ' m')} |",
         f"| 最接近距離の差（トラックごと） | {fmt(g['cpa_m'], ' m')} |", "",
         "読み方: |差| の 95% 点が「2 人の読みの違いで正解がどれだけ動くか」の実測値。文書の見込み（時刻 ±0.03 s）と比べる。"]
    return _write(out, "A_readers", res, L)


# ---------------------------------------------------------------- B. 格子読みとの差
def cmd_grid(script_csv: Path, grid_csv: Path, out: Path, mic_z: float, src_z: float) -> dict:
    rs, rg = read_csv(script_csv), read_csv(grid_csv)
    g = compare_gt(gt_rows_by_take(rs, mic_z, src_z), gt_rows_by_take(rg, mic_z, src_z))
    res = {"kind": "grid", "script": str(script_csv), "grid": str(grid_csv), "gt": g}
    L = ["# B. 台本方式と格子読み（独立に測った点）の差・実測値", "",
         f"台本方式 = `{script_csv.name}`（目印の通過時刻 → 直線補間）・格子読み = `{grid_csv.name}`（0.5 秒ごとに路面の格子で読んだ位置）・共通のテイク {g['n_takes']}", "",
         "| 何の差 | 実測値（台本 − 格子） |", "| --- | --- |",
         f"| 方位差（共通フレーム） | {fmt(g['az_abs_deg'], '°', 1)} |",
         f"| 水平距離の差 | {fmt(g['dist_m'], ' m')} |",
         f"| 最接近距離の差 | {fmt(g['cpa_m'], ' m')} |", "",
         "読み方: 格子読みの自体の粗さ（0.5 m 単位）が入るので、差は「台本方式の誤差」の上限側。学内パイロット計画 §5 の合格線 = 最接近距離 ±0.3 m 以内。"]
    for take, p in g["per_take"].items():
        L.append(f"- {take}: 共通 {p['n_frames']} フレーム・方位 |差| 95% 点 {p['az_abs_p95']:.1f}°・距離 |差| 95% 点 {p['dist_p95_abs']:.2f} m・最接近の差 {', '.join(f'{v:+.2f}' for v in p['cpa_diff'])} m")
    return _write(out, "B_grid", res, L)


# ---------------------------------------------------------------- C. 横ずれ
def cmd_lateral(csv_p: Path, out: Path) -> dict:
    rows = read_csv(csv_p)
    by_cls, by_line, all_off = defaultdict(list), defaultdict(list), []
    flips = defaultdict(lambda: [0, 0])
    for r in rows:
        off = float(r["offset_m"]); line = float(r["line_m"]); cls = r["cls"].strip()
        all_off.append(off); by_cls[cls].append(off); by_line[line].append(off)
        tier0 = _tier(line); tier1 = _tier(line + off)
        flips[line][1] += 1
        if tier0 != tier1:
            flips[line][0] += 1
    res = {"kind": "lateral", "src": str(csv_p), "all": summ(all_off), "by_cls": {k: summ(v) for k, v in by_cls.items()},
           "by_line": {str(k): summ(v) for k, v in by_line.items()},
           "tier_flip": {str(k): {"flipped": v[0], "n": v[1], "rate": (v[0] / v[1] if v[1] else float("nan"))} for k, v in flips.items()}}
    L = ["# C. 横ずれ（線からの横の外れ・実測値）", "",
         f"元 = `{csv_p.name}`・測定 {len(rows)} 件。offset_m = 前輪（車は車体の中心）が線からどれだけ離れたか（マイクから遠ざかる向きが正）", "",
         "| 層 | 実測値 |", "| --- | --- |", f"| 全体 | {fmt(res['all'], ' m')} |"]
    L += [f"| 種類 {k} | {fmt(v, ' m')} |" for k, v in res["by_cls"].items()]
    L += [f"| 線 {k} m | {fmt(v, ' m')} |" for k, v in res["by_line"].items()]
    L += ["", "採点の区分（≤1.5 至近／≤3.2 注意／>3.2 安全）が実測の横ずれで隣に移った割合:", "", "| 線 | 移った / 測定 | 割合 |", "| --- | --- | --- |"]
    L += [f"| {k} m | {v['flipped']} / {v['n']} | {100 * v['rate']:.0f}% |" for k, v in res["tier_flip"].items()]
    L += ["", "読み方: 「線の値 = 横距離」という仮定のずれの実測値。文書の見込み ±0.1〜0.3 m を、この |差| の 95% 点に置き換える。境界の上の線（1.5・3.2 m）は移った割合を見る。"]
    return _write(out, "C_lateral", res, L)


def _tier(d: float) -> str:
    return "critical" if d <= TIER_EDGES[0] else ("caution" if d <= TIER_EDGES[1] else "safe")


# ---------------------------------------------------------------- D. 同期
def cmd_sync(csv_p: Path, out: Path) -> dict:
    rows = read_csv(csv_p)
    by_take = defaultdict(dict)
    for r in rows:
        by_take[r["take"].strip()][int(r["clap"])] = (float(r["audio_s"]), float(r["video_s"]))
    offs, drifts, spans, per = [], [], [], {}
    for take, c in by_take.items():
        o = {k: a - v for k, (a, v) in c.items()}          # 音の時刻 − 動画の時刻
        offs += list(o.values())
        if 1 in c and 2 in c:
            span = c[2][1] - c[1][1]
            drift = o[2] - o[1]
            drifts.append(drift); spans.append(span)
            per[take] = {"offset_first": o[1], "offset_last": o[2], "drift_s": drift, "video_span_s": span, "drift_per_min": (drift / span * 60 if span else float("nan"))}
    res = {"kind": "sync", "src": str(csv_p), "offset": summ(offs), "drift": summ(drifts), "n_takes_two_claps": len(drifts), "per_take": per}
    L = ["# D. 同期（拍手 2 回のずれ・実測値）", "",
         f"元 = `{csv_p.name}`・テイク {len(by_take)}（2 回の拍手があるもの {len(drifts)}）。offset = 音の時刻 − 動画の時刻。drift = 最後の拍手の offset − 最初の拍手の offset", "",
         "| 何 | 実測値 |", "| --- | --- |",
         f"| offset（全拍手） | {fmt(res['offset'], ' s', 3)} |",
         f"| drift（1 テイクの中の変化） | {fmt(res['drift'], ' s', 3)} |", ""]
    L += [f"- {t}: offset {p['offset_first']:+.3f} → {p['offset_last']:+.3f} s（動画 {p['video_span_s']:.1f} s で drift {p['drift_s']:+.3f} s・{p['drift_per_min']:+.3f} s/分）" for t, p in per.items()]
    L += ["", "読み方: drift の |差| の 95% 点が「1 回の拍手だけで同期したときの時刻の不確かさ」。0.05 s を超えるなら両端の拍手で一次補正する（`step19f --sync-offset` を両端で分ける）。"]
    return _write(out, "D_sync", res, L)


# ---------------------------------------------------------------- 出力
def _write(out: Path, stem: str, res: dict, lines: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("->", out / f"{stem}.md")
    return res


# ---------------------------------------------------------------- 自己検査
def selftest(out: Path) -> int:
    """合成データ: 4 m/s で y = 2.0 m の線を通る車 3 本。読み手 B は時刻を +1 frame ずらし、格子読みは 0.5 m 単位に丸め、横ずれは既知の分布、同期は既知の drift。"""
    rng = np.random.default_rng(0)
    v = 4.0
    marks = [10, 5, 0, -5, -10]
    A, B, G = [], [], []
    for i in range(3):
        take = f"t{i + 1}"
        t0 = 1.0 + 0.3 * i
        for xm in marks:
            fr = int(round((t0 + (10 - xm) / v) / LABEL_RES))
            A.append(dict(take=take, frame=str(fr), cls="car", track="0", x=str(xm), y="2.0", z=""))
            B.append(dict(take=take, frame=str(fr + 1), cls="car", track="0", x=str(xm), y="2.0", z=""))
        for k in range(int(t0 / LABEL_RES), int((t0 + 20 / v) / LABEL_RES) + 1, 5):     # 0.5 秒ごと
            x = 10 - v * (k * LABEL_RES - t0)
            G.append(dict(take=take, frame=str(k), cls="car", track="0", x=f"{round(x * 2) / 2:.1f}", y=f"{round((2.0 + rng.normal(0, 0.1)) * 2) / 2:.1f}", z=""))
    out.mkdir(parents=True, exist_ok=True)
    def wcsv(name, rows, cols):
        p = out / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        return p
    pa = wcsv("self_A.csv", A, ["take", "frame", "cls", "track", "x", "y", "z"]); pb = wcsv("self_B.csv", B, ["take", "frame", "cls", "track", "x", "y", "z"])
    pg = wcsv("self_grid.csv", G, ["take", "frame", "cls", "track", "x", "y", "z"])
    lat = [dict(take=f"t{i}", cls="car" if i % 2 else "kick", line_m="3.2" if i % 2 else "1.5", x_marker="0", offset_m=f"{rng.normal(0, 0.15):.2f}") for i in range(40)]
    pl = wcsv("self_lateral.csv", lat, ["take", "cls", "line_m", "x_marker", "offset_m"])
    syn = []
    for i in range(5):
        off = 2.6 + 0.01 * i; drift = 0.02 * (i - 2)
        syn.append(dict(take=f"t{i}", clap="1", audio_s=f"{3.2:.3f}", video_s=f"{3.2 - off:.3f}"))
        syn.append(dict(take=f"t{i}", clap="2", audio_s=f"{25.0:.3f}", video_s=f"{25.0 - off - drift:.3f}"))
    ps = wcsv("self_sync.csv", syn, ["take", "clap", "audio_s", "video_s"])
    r1 = cmd_readers(pa, pb, out, 1.6, 0.5)
    r2 = cmd_grid(pa, pg, out, 1.6, 0.5)
    r3 = cmd_lateral(pl, out)
    r4 = cmd_sync(ps, out)
    ok = [
        ("A: frame 差の中央値 = −1", abs(r1["marker"]["frame_diff"]["median"] + 1) < 1e-9),
        ("A: 距離差の |差| 95% 点 < 0.5 m（1 frame = 0.4 m）", r1["gt"]["dist_m"]["p95_abs"] < 0.5),
        ("B: 最接近距離の差 |差| ≤ 0.35 m（格子 0.5 m 単位）", r2["gt"]["cpa_m"]["max_abs"] <= 0.35 + 1e-9),
        ("C: 横ずれ SD が 0.1〜0.2 m（生成 0.15）", 0.1 <= r3["all"]["sd"] <= 0.2),
        ("C: 境界の線で移った割合が 0 より大きい", any(v["flipped"] > 0 for v in r3["tier_flip"].values())),
        ("D: drift の中央値 ≈ 0・最大 0.04 s", abs(r4["drift"]["median"]) < 1e-6 and abs(r4["drift"]["max_abs"] - 0.04) < 1e-6),
    ]
    for name, cond in ok:
        print(("PASS " if cond else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__); return 2
    cmd = argv[0]
    def opt(k, d):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    mic_z = float(opt("--mic-z", "1.6")); src_z = float(opt("--src-z", "0.5"))
    if cmd == "selftest":
        return selftest(Path(argv[1]))
    if cmd == "readers":
        cmd_readers(Path(argv[1]), Path(argv[2]), Path(argv[3]), mic_z, src_z); return 0
    if cmd == "grid":
        cmd_grid(Path(argv[1]), Path(argv[2]), Path(argv[3]), mic_z, src_z); return 0
    if cmd == "lateral":
        cmd_lateral(Path(argv[1]), Path(argv[2])); return 0
    if cmd == "sync":
        cmd_sync(Path(argv[1]), Path(argv[2])); return 0
    print("unknown command:", cmd); return 2


if __name__ == "__main__":
    sys.exit(main())
