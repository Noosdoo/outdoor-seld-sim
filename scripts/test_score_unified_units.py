# -*- coding: utf-8 -*-
"""統一採点器（_score_unified.py）の回帰試験（2026-09-07）。合成の小さな GT/予測で、宣言どおりの定義が保たれているかを固定する。

  U1 完全一致の予測: 至近到達・強到達、検出率 100%、至近捕捉(全GT) 100%、距離誤差 0%、誤捕捉 0
  U2 予測なしのクリップ: manifest 固定なので分母に残り、到達 0・検出 0・発火 0（n_pred=0）
  U3 距離を 1.5 倍に言う予測: 検出率 100% のまま至近捕捉が落ち、距離誤差 50%
  U4 方位が 40° ずれた予測: 1 対 1 の対応（≤20°）も帰属（≤30°）も付かない
  U5 同じフレームに GT 2 台・予測 1 つ: 方位の近い 1 台だけと対応（GT を消費する）
  U6 3D 距離の予測（el=60°）は @h の水平変換で GT と一致、変換なしでは 100% 誤差
  U7 aggregate: 予測ありなしの混在で本数・分母が合う
  U8 CLI: plan（manifest）・meta・予測ファイルから md が出て、行に本数(予測あり)が入る

使い方: python scripts/test_score_unified_units.py
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location("su", ROOT / "scripts" / "_score_unified.py")
su = importlib.util.module_from_spec(spec); sys.modules["su"] = su; spec.loader.exec_module(su)

fails, n_checks = [], 0


def check(name, cond, note=""):
    global n_checks
    n_checks += 1
    print(("[PASS] " if cond else "[FAIL] ") + name + (" " + note if note else ""))
    if not cond:
        fails.append(name)


CAR = 4
FPS = int(su.FPS)


def passby(lateral=0.8, speed=8.0, t_cpa=5.0, x_dir=1.0):
    """車が横距離 lateral[m] を speed[m/s] で通り過ぎる 10 秒（100 フレーム）。frame -> (az[deg], el, d[m])。CPA は t_cpa。"""
    out = {}
    for k in range(100):
        t = (k + 1) / FPS
        x = x_dir * speed * (t - t_cpa)          # 前後方向
        y = lateral                              # 左が＋
        d = math.hypot(x, y)
        az = math.degrees(math.atan2(y, x))
        out[k] = (az, 0.0, d)
    return out


def radial(d0=10.0, v=1.5, dmin=0.8, az=90.0):
    """方位が変わらない接近（左 90° から真っすぐ寄ってきて dmin で止まる）。frame -> (az, el, d)"""
    return {k: (az, 0.0, max(dmin, d0 - v * (k + 1) / FPS)) for k in range(100)}


def write_gt(meta: Path, clip: str, tracks: dict):
    """tracks: trk -> frame -> (az, el, d)。GT 形式 = frame,cls,trk,az,el,dist"""
    lines = []
    for trk, fr in tracks.items():
        for k in sorted(fr):
            az, el, d = fr[k]
            lines.append(f"{k},{CAR},{trk},{az:.4f},{el:.4f},{d:.4f}")
    (meta / f"{clip}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pred_from(clip: str, tracks: dict, d_scale=1.0, az_off=0.0, el=0.0, d_fn=None):
    """予測 = clip -> frame -> [(cls, az, el, d)]（load_pred_rows と同じ形）"""
    out = {clip: {}}
    for trk, fr in tracks.items():
        for k in sorted(fr):
            az, _e, d = fr[k]
            dd = d_fn(d) if d_fn else d * d_scale
            out[clip].setdefault(k, []).append((CAR, az + az_off, el, dd))
    return out


with tempfile.TemporaryDirectory() as td:
    meta = Path(td) / "meta"; meta.mkdir()
    gt1 = {0: passby(0.8, 8.0, 5.0)}
    write_gt(meta, "c1", gt1)
    n_close = sum(1 for k, (az, el, d) in gt1[0].items() if k >= 40 and d <= su.TH_CLOSE)
    n_safe = sum(1 for k, (az, el, d) in gt1[0].items() if k >= 40 and d > su.TH_SAFE)

    # U1 完全一致
    s1 = su.clip_stats(pred_from("c1", gt1), meta, "c1", 40)
    a1 = su.aggregate([s1])
    check("U1 完全一致: 至近到達 1/1・強到達・検出率 100%・至近捕捉(全GT) 100%・距離誤差 0%・誤捕捉 0",
          s1["n_pred"] == 1 and s1["crit"] == [1, 1] and s1["n_strong"] == 1 and s1["n_fire"] >= 1
          and s1["n_close_gt"] == n_close and s1["n_close_det"] == n_close and s1["n_close_cap"] == n_close
          and s1["n_safe_pairs"] == n_safe and s1["n_safe_fp"] == 0 and max(s1["rel"]) < 1e-3
          and a1["crit"] == 100.0 and a1["strong"] == 100.0 and a1["det_close"] == 100.0 and a1["cap_all"] == 100.0 and a1["dist_err"] < 0.1 and a1["fp"] == 0.0
          and a1["lead"] > 0,
          f"(crit={s1['crit']}, strong={s1['n_strong']}, fires={s1['n_fire']}, close_gt={s1['n_close_gt']}, lead={a1['lead']:.2f}s)")

    # U2 予測なし
    s2 = su.clip_stats({}, meta, "c1", 40)
    a2 = su.aggregate([s2])
    check("U2 予測なし: 分母に残り（crit 0/1・至近 GT は数える）、発火 0・検出 0・捕捉 0",
          s2["n_pred"] == 0 and s2["crit"] == [0, 1] and s2["n_fire"] == 0 and s2["n_close_gt"] == n_close and s2["n_close_det"] == 0
          and a2["crit"] == 0.0 and a2["det_close"] == 0.0 and a2["cap_all"] == 0.0 and a2["n_pred_clips"] == 0 and a2["n_clips"] == 1,
          f"(crit={s2['crit']}, close_gt={s2['n_close_gt']})")

    # U3 距離 1.5 倍
    s3 = su.clip_stats(pred_from("c1", gt1, d_scale=1.5), meta, "c1", 40)
    a3 = su.aggregate([s3])
    exp_cap = sum(1 for k, (az, el, d) in gt1[0].items() if k >= 40 and d <= su.TH_CLOSE and 1.5 * d <= su.TH_CLOSE)
    check("U3 距離を 1.5 倍に言う: 検出率 100% のまま、至近捕捉(全GT) は GT≤1.0 m の分だけ、距離誤差 50%、誤捕捉 0",
          s3["n_close_det"] == n_close and s3["n_close_cap"] == exp_cap and exp_cap < n_close and abs(a3["dist_err"] - 50.0) < 1e-3 and s3["n_safe_fp"] == 0,
          f"(det={s3['n_close_det']}, cap={s3['n_close_cap']}/{n_close}, err={a3['dist_err']:.1f}%)")

    # U4 方位 40° ずれ（方位が変わらない接近で。通り過ぎる車は CPA 付近で方位が一気に回るので ±0.5 s の帰属窓に入ってしまう）
    gt4 = {0: radial(10.0, 1.5, 0.8, 90.0)}
    write_gt(meta, "c4", gt4)
    s4ok = su.clip_stats(pred_from("c4", gt4), meta, "c4", 40)
    s4 = su.clip_stats(pred_from("c4", gt4, az_off=40.0), meta, "c4", 40)
    check("U4 方位が 40° ずれた予測: 1 対 1 の対応なし（≤20°）・帰属なし（≤30°）→ 至近到達 0/1・検出 0（同じ GT でずれ無しなら 1/1）",
          s4ok["crit"] == [1, 1] and s4["n_pairs"] == 0 and s4["n_close_det"] == 0 and s4["crit"] == [0, 1] and s4["n_fire"] >= 1,
          f"(ok crit={s4ok['crit']}, off: pairs={s4['n_pairs']}, crit={s4['crit']}, fires={s4['n_fire']})")

    # U5 同じフレームに GT 2 台・予測 1 つ → 方位の近い方だけ
    fr_a = {k: (90.0, 0.0, 1.0) for k in range(40, 60)}          # 左 1.0 m（至近）
    fr_b = {k: (95.0, 0.0, 5.0) for k in range(40, 60)}          # 左 5.0 m（安全）
    write_gt(meta, "c2", {0: fr_a, 1: fr_b})
    p5 = {"c2": {k: [(CAR, 92.0, 0.0, 1.0)] for k in range(40, 60)}}
    s5 = su.clip_stats(p5, meta, "c2", 40)
    check("U5 GT 2 台・予測 1 つ: 方位の近い至近車とだけ対応（20 ペア）。安全車は対応なし（誤捕捉の分母に入らない）",
          s5["n_pairs"] == 20 and s5["n_close_det"] == 20 and s5["n_close_cap"] == 20 and s5["n_safe_pairs"] == 0 and s5["n_safe_fp"] == 0,
          f"(pairs={s5['n_pairs']}, close_det={s5['n_close_det']}, safe_pairs={s5['n_safe_pairs']})")

    # U6 3D 距離（el=60°）の @h 変換
    pred_csv = Path(td) / "pred3d.csv"
    with open(pred_csv, "w", encoding="utf-8") as f:
        for k, (az, el, d) in gt1[0].items():
            f.write(f"c1,{k},{CAR},0,{az:.3f},60.0,{d / math.cos(math.radians(60.0)):.4f}\n")
    ph = su.load_pred_rows(pred_csv, horiz=True); p3 = su.load_pred_rows(pred_csv, horiz=False)
    s6h = su.clip_stats(ph, meta, "c1", 40); s6 = su.clip_stats(p3, meta, "c1", 40)
    check("U6 3D 距離の予測は @h（d×cos el）で GT と一致（誤差 0%）、変換なしだと 100% の誤差",
          max(s6h["rel"]) < 1e-3 and abs(float(np.median(s6["rel"])) - 1.0) < 1e-3, f"(h={max(s6h['rel']):.2e}, raw={float(np.median(s6['rel'])):.3f})")

    # U7 aggregate の混在
    a7 = su.aggregate([s1, s2])
    check("U7 aggregate: 予測あり 1/2 本、至近到達 50%、至近 GT は 2 本分、至近捕捉(全GT) 50%",
          a7["n_clips"] == 2 and a7["n_pred_clips"] == 1 and a7["crit"] == 50.0 and a7["n_close_gt"] == 2 * n_close and a7["cap_all"] == 50.0,
          f"(n={a7['n_clips']}/{a7['n_pred_clips']}, crit={a7['crit']}, cap_all={a7['cap_all']})")

    # U8 CLI（manifest 固定）
    plan = Path(td) / "plan.csv"
    plan.write_text("clip_id,split,mic_z\nc1,fold2,1.5\nc2,fold2,2.0\nc9,fold2,1.5\n", encoding="utf-8")
    pred1 = Path(td) / "pred1.csv"
    with open(pred1, "w", encoding="utf-8") as f:
        for k, (az, el, d) in gt1[0].items():
            f.write(f"c1,{k},{CAR},0,{az:.3f},0.0,{d:.4f}\n")
    out_md = Path(td) / "out.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/_score_unified.py"), str(out_md), "--plan", str(plan), "--meta", str(meta), f"perfect={pred1}", f"h3d={pred_csv}@h"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
    check("U8 CLI: manifest の 3 本（予測あり 1 本）で md が出て、perfect と h3d の 2 行がある", r.returncode == 0 and "| perfect | 3(1) |" in txt and "h3d (予測を水平変換) | 3(1) |" in txt,
          f"(rc={r.returncode}, {r.stderr[-200:]!r})\n" + "\n".join(l for l in txt.splitlines() if l.startswith("| perfect") or l.startswith("| h3d")))

if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print(f"ALL PASS ({n_checks} checks)")
