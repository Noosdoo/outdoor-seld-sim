# -*- coding: utf-8 -*-
"""事象台帳 v30（本体 §2-5・§9-1 の 2・§9-4 の 1／4・変更票 #75 ⑮⑰・R30-03／RR30-04・2026-10-26）。

確定評価・開発の比べの採点の「事象」を、ラベル（metadata の連続ラン）ではなく **scene.json の幾何**から作る:
  - 1 事象 = 1 音源（class・track ごとに 1 つ）。音の途切れ（可聴ゲートの穴）で割らない。ラベル無しの電車（D14・no_label）は事象にしない（背景）。
  - 物理の最接近 = 音源と（歩く）マイクの**水平距離**が最小の時刻と値（0.01 s 刻み・scene の軌道から）。圏（tier）= 水平の最接近 ≤1.5 至近・≤3.2 注意・それより遠い 安全。
  - 物理の始まり／終わり = 音源の発音区間（t_on・t_off）。発進は scene の launch を残す（動き出し）。
  - ラベルの存在範囲 = 最初と最後の正解の枠（metadata）。可聴区間 = 正解の連続ランの一覧（"a-b;c-d"・枠の番号・両端含む）。全消失なら空（物理の時刻で埋めない）。
  - **事前に定めた除外（暖機・⑰ の最終形）**: 通過・発進は**物理の最接近 < 4.0 s**（warmup_cpa）。警告音は**鳴り終わり ≤ 4.0 s**（warmup_end）だけ。
    警告音の**採点上の鳴り始め = max(物理の鳴り始め, 4.0 s)**（窓 [採点上の鳴り始め − 1, ＋4]・遅れもここから）。ラベル全消失は除外の理由にしない（分母に入る）。
  - 分母 = 除外のあとの物理の事象すべて。除外前の数・理由ごとの数・残った分母・全消失の数をクラス別に summary に保存する。
  - **軌道の方位 az_series**（Fable 10/26 16:00・Astra 3 回目）: 0〜99 枠の受音時刻の見かけの方位（学習の正解と同じ `frame_label_rows` の連続値・発音区間の中だけ・
    v30 の本は scene の mic_heading_deg で歩く向きに回す・届いていない枠は空）。統一採点器はこれを通知の帰属に使う（ラベルの枠ではない）→ ラベルのある枠では
    今と同じ方位、全消失の事象でも方向の合う発火があれば通知と数える。

出力: 台帳 CSV（1 行 = 1 事象・下の COLS）と summary JSON。統一採点器は `--events <台帳>` でこの事象を使う（既定は旧の連続ラン・`scripts/_score_unified.py`）。
使い方: python scripts/_event_ledger_v30.py --ds <描画の置き場> --plan <plan か manifest の csv> [--split val] --out <台帳.csv> [--summary <json>]
試験: scripts/test_score_unified_v30.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3, "car_drive": 4, "crossing": 5, "kickboard": 6, "motorcycle": 7, "bicycle": 8}
DIST_CLASSES = {"car_drive", "kickboard", "motorcycle", "bicycle"}
WARN_CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "crossing"}
WARMUP_S = 4.0                 # 因果の窓 KMIN = 40 枠 = システムが最初に動ける時刻（⑰）
T3, SUPP = 1.5, 3.2            # 圏の境目（水平・統一採点器・選択器と同じ）
CLIP, DT, FPS = 10.0, 0.01, 10.0
COLS = ["clip", "src_idx", "class", "cls", "track", "kind", "phys_on_s", "phys_off_s", "phys_cpa_s", "phys_cpa_h_m", "phys_cpa_3d_m", "lateral_m", "tier",
        "launch_move_s", "label_first", "label_last", "n_label", "runs", "all_lost", "score_onset_s", "excluded", "az_series"]
TEMP_C = 20.0                  # 描画の気温（step11_v9_render.TEMP_C）＝音速は geometry.sound_speed(20)


def mic_path(scene: dict) -> np.ndarray:
    """scene.json の mic からマイクの位置（静止 (3,) か折れ線 (M,4)）。描画の式と同じ（v12._mic_sample・v9 の walk_cross_y・walk_stop_x）。"""
    m = scene.get("mic") or {}
    z = float(m.get("mic_z", (scene.get("row") or {}).get("mic_z", 1.5)) or 1.5)
    mo = m.get("motion", "static")
    if mo == "static":
        return np.array([0.0, 0.0, z])
    v = float(m["walk_speed_mps"])
    if mo == "walk_cross_y":
        ts = float(m["t_stop_s"]); y0 = -0.5 - v * ts
        return np.array([[0.0, 0.0, y0, z], [ts, 0.0, -0.5, z], [CLIP, 0.0, -0.5, z]])
    if mo == "walk_stop_x":
        ts = float(m["t_stop_s"]); x0 = -v * ts
        return np.array([[0.0, x0, 0.0, z], [ts, 0.0, 0.0, z], [CLIP, 0.0, 0.0, z]])
    d = float(m.get("walk_dir_x", 1.0)); x0 = -d * v * 5.0
    return np.array([[0.0, x0, 0.0, z], [CLIP, x0 + d * v * CLIP, 0.0, z]])


def _at(path: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim == 1:
        return np.broadcast_to(p, t.shape + (3,))
    return np.stack([np.interp(t, p[:, 0], p[:, k + 1]) for k in range(3)], axis=-1)


def phys_cpa(wp, mic) -> tuple[float, float, float]:
    """(水平の最接近の秒, 水平距離, その時刻の 3 次元距離)。同じ値が並ぶときは最初の時刻（静止の音源など）。"""
    t = np.arange(0.0, CLIP + 1e-9, DT)
    ps, pm = _at(np.asarray(wp, float), t), _at(mic, t)
    h = np.hypot(ps[:, 0] - pm[:, 0], ps[:, 1] - pm[:, 1])
    i = int(np.argmin(h))
    return round(float(t[i]), 2), round(float(h[i]), 3), round(float(np.linalg.norm(ps[i] - pm[i])), 3)


def read_labels(path: Path) -> dict:
    """metadata（5 列）か metadata_dist（6 列）→ {(cls, track): sorted frames}。"""
    out = defaultdict(set)
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                v = ln.split(",")
                out[(int(v[1]), int(v[2]))].add(int(v[0]))
    return {k: sorted(v) for k, v in out.items()}


def runs_of(frames: list) -> list:
    out = []
    for k in frames:
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return out


def az_series(src: dict, mic, scene: dict) -> str:
    """受音時刻の見かけの方位（0〜99 枠・発音区間の中だけ・v30 の本は歩く向きに回す・小数 1 桁・届いていない枠は空）。描画のラベルと同じ関数で作る。"""
    sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.geometry import mic_heading_from_record, sound_speed
    from outdoor_seld.labels import frame_label_rows
    wp = np.asarray(src["wp"], dtype=float)
    pre = float(src.get("preroll_s", 0.0) or 0.0)
    act_from = float(src.get("t_on", 0.0))
    if pre > 0:                                               # v13 のプリロールと同じ（最初の区間を後ろへ直線で伸ばす）
        (t0, *p0), (t1, *p1) = wp[0], wp[1]
        p0, p1 = np.array(p0), np.array(p1)
        ext = p0 + (p1 - p0) * ((-pre - t0) / (t1 - t0))
        wp = np.vstack([[-pre, *ext], wp]); act_from = -pre
    rec = (scene.get("mic") or {}).get("mic_heading_deg")
    hd = mic_heading_from_record(rec, (np.arange(int(CLIP * FPS)) + 0.5) / FPS) if rec is not None else None
    rows, dbg = frame_label_rows(wp, mic, clip_len_sec=CLIP, class_idx=0, source_active_from=act_from,
                                 source_active_until=float(src.get("t_off", CLIP)), c=sound_speed(TEMP_C), heading_deg=hd)
    act = {r[0] for r in rows}
    return ";".join((f"{float(dbg['az'][k]):.1f}" if k in act else "") for k in range(int(CLIP * FPS)))


def tier_of(h: float) -> str:
    return "critical" if h <= T3 else "caution" if h <= SUPP else "safe"


def ledger_for_clip(ds: Path, clip: str) -> list:
    scene = json.loads((ds / "work" / clip / "scene.json").read_text(encoding="utf-8"))
    labels = read_labels(ds / "metadata" / f"{clip}.csv")
    mic = mic_path(scene)
    rows, seen = [], set()
    for i, src in enumerate(scene["sources"]):
        cls = src["class"]
        if src.get("no_label") or cls not in CLASS_IDX:
            continue
        ci, trk = CLASS_IDX[cls], int(src.get("track", 0) or 0)
        if (ci, trk) in seen:
            raise ValueError(f"{clip}: 同じ (class, track) = ({cls}, {trk}) の正解のある音源が 2 つ（事象が決まらない）")
        seen.add((ci, trk))
        t_cpa, h, d3 = phys_cpa(src["wp"], mic)
        fr = labels.get((ci, trk), [])
        rr = runs_of(fr)
        launch = src.get("launch") or {}
        kind = "warn" if cls in WARN_CLASSES else ("launch" if launch else "pass")
        on, off = float(src.get("t_on", 0.0)), float(src.get("t_off", CLIP))
        excl, onset = "", ""
        if kind == "warn":
            onset = round(max(on, WARMUP_S), 2)
            if off <= WARMUP_S:
                excl = "warmup_end"
        elif t_cpa < WARMUP_S:
            excl = "warmup_cpa"
        lat = src.get("lateral_m")
        rows.append({"clip": clip, "src_idx": i, "class": cls, "cls": ci, "track": trk, "kind": kind,
                     "phys_on_s": round(on, 3), "phys_off_s": round(off, 3), "phys_cpa_s": t_cpa, "phys_cpa_h_m": h, "phys_cpa_3d_m": d3,
                     "lateral_m": (round(float(lat), 3) if lat is not None else h), "tier": ("warn" if kind == "warn" else tier_of(h)),
                     "launch_move_s": (launch.get("t_on_s", src.get("t_on", "")) if launch else ""),
                     "label_first": (fr[0] if fr else ""), "label_last": (fr[-1] if fr else ""), "n_label": len(fr),
                     "runs": ";".join(f"{a}-{b}" for a, b in rr), "all_lost": int(not fr), "score_onset_s": onset, "excluded": excl,
                     "az_series": az_series(src, mic, scene)})
    return rows


def build(ds: Path, clips: list) -> tuple[list, dict]:
    rows = []
    for c in clips:
        rows += ledger_for_clip(ds, c)
    by = defaultdict(lambda: {"n_physical": 0, "excluded": defaultdict(int), "n_denominator": 0, "n_all_lost": 0})
    for r in rows:
        b = by[r["class"]]
        b["n_physical"] += 1
        if r["excluded"]:
            b["excluded"][r["excluded"]] += 1
        else:
            b["n_denominator"] += 1
            b["n_all_lost"] += r["all_lost"]
    summary = {"rule": "event_ledger_v30（物理の通過・class・track ごと・暖機: 通過と発進は物理の最接近 < 4.0 s・警告音は鳴り終わり ≤ 4.0 s を事前に除外・全消失は分母）",
               "warmup_s": WARMUP_S, "n_clips": len(clips), "n_events": len(rows),
               "by_class": {k: {**v, "excluded": dict(v["excluded"])} for k, v in sorted(by.items())}}
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="事象台帳 v30（scene の幾何から・本体 §2-5）")
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True, help="clip_id の列がある csv（plan か manifest）")
    ap.add_argument("--split", default="", help="plan の split 列で絞る（例 val）・空 = 全部")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    a = ap.parse_args()
    with open(a.plan, newline="", encoding="utf-8-sig") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f) if not a.split or r.get("split", "") == a.split]
    rows, summary = build(a.ds, clips)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    sp = a.summary or a.out.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in summary["by_class"].items():
        print(f"{k}: 物理 {v['n_physical']}・除外 {v['excluded'] or 0}・分母 {v['n_denominator']}（うち全消失 {v['n_all_lost']}）")
    print("->", a.out, sp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
