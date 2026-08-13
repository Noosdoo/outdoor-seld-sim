# -*- coding: utf-8 -*-
"""実録経路の単体テスト（2026-08-13の回帰5件=T12〜T16を含む）。

T1 : step19が96kHz入力を受理し24kHz/長さ1/4で出力
T2 : 長尺負例の全区間走査（50秒地点の発火。旧実装の10秒固定では0件）
T3 : event_id窓の貪欲割当（1発火は最大1イベント）
T4 : v3.4距離トリガの方位連結±60°（40°=発火/70°=非発火）
T5 : 統計（McNemar厳密・本番のPoisson片側95%上限・クリップ単位bootstrap）
T6 : 警告音=v1定数（3フレーム連続・不応期5s）＋因果時刻(k+1)/FPS
T7 : 負例マスク（注釈イベント窓の発火は誤警告に数えず・露出の重なり控除）
T8 : caution車（横距離2.5m）は中通知で成功・リード/象限が出る
T9 : 【回帰】2m,2m,1m列で強は発火しない（T3×2連続をT2列から昇格させない）
T10: 【回帰】safe車（横距離10m）への強発火は失敗
T11: 【回帰】caution車2m予測のリード=t_cpa−(k+1)/FPS
T12: 【回帰】safe車への中通知も失敗（safe成功=強・中とも通知なし）
T13: 【回帰】同一エピソードは強・中として二重割当されない（消費フラグ共通）
T14: 【回帰】エピソード統合は正規規則（フレーム差≤1かつ方位差≤25°）
T15: 【回帰】横距離欠落の距離クラス行は未採点（分母除外）
T16: 【回帰】NaN/Infinity/負値の横距離は未採点（有限・非負値のみ有効）
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v2 = _load("s20v2", "step20_realsmoke_score_v2.py")
s19 = _load("s19rc", "step19_realsmoke_convert.py")

fails = []
n_checks = 0


def check(name, cond, note=""):
    global n_checks
    n_checks += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        fails.append(name)


# ---------------- T1: 96kHz変換 ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    td = Path(td)
    fs = 96000
    t = np.arange(2 * fs) / fs
    w = 0.05 * np.sin(2 * np.pi * 1000 * t)
    wav = np.stack([w, 0.3 * w, 0.1 * w, 0.2 * w], axis=1)
    src = td / "take96k.wav"
    sf.write(src, wav, fs)
    out = s19.convert(src, td / "conv", laeq=50.0, win=(0.0, 1.0),
                      pitch=0, roll=0, yaw=0)
    y, sr_out = sf.read(out)
    check("T1 96kHz受理", sr_out == 24000 and abs(len(y) - 2 * 24000) <= 4,
          f"(fs={sr_out}, n={len(y)})")

# ---------------- T2: 長尺負例の全区間走査 ----------------
pred = {"c1": {500: [(4, 10.0, 1.0)], 501: [(4, 12.0, 1.0)]}}
rows = [{"clip_id": "c1", "event_id": "1", "trial": "n1", "class": "none",
         "quadrant": "", "t_start": "0", "t_cpa": "120"}]
ev, neg, _ = v2.evaluate(rows, pred, link_deg=60.0, has_dist=True)
check("T2 長尺負例で50s地点の発火を検出",
      neg["n_false"] == 1 and abs(neg["exposure_s"] - 120.0) < 1e-9,
      f"(n_false={neg['n_false']}, exp={neg['exposure_s']}s)")

# ---------------- T3: event_id窓と貪欲割当 ----------------
pred3 = {"c2": {49: [(4, 0.0, 1.0)], 50: [(4, 2.0, 1.0)]}}
rows3 = [
    {"clip_id": "c2", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "3", "t_cpa": "8", "横距離m": "1.0"},
    {"clip_id": "c2", "event_id": "2", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "20", "t_cpa": "25", "横距離m": "1.0"},
]
ev3, _, _ = v2.evaluate(rows3, pred3, link_deg=60.0, has_dist=True)
got = {e["event_id"]: e["notified"] for e in ev3}
check("T3 1発火は1イベントのみ成功",
      got.get("1") is True and got.get("2") is False, f"({got})")

# ---------------- T4: ±60°連結 ----------------
h_in = v2.dist_triggers_var({10: [(10.0, 1.4)], 11: [(50.0, 1.4)]}, 1.5, 20, 60.0)
h_out = v2.dist_triggers_var({10: [(10.0, 1.4)], 11: [(80.0, 1.4)]}, 1.5, 20, 60.0)
check("T4 方位連結（40°=発火/70°=非発火）",
      len(h_in) == 1 and len(h_out) == 0, f"(in={h_in}, out={h_out})")

# ---------------- T5: 統計 ----------------
p_mc = v2.mcnemar_exact(0, 8)
up1 = v2.poisson_upper95_one_sided(0, 100 / 60.0)
bs = v2.paired_bootstrap_median_diff_by_clip(
    {"cA": [1.0, 1.0], "cB": [1.0]}, n_boot=2000, seed=0)
check("T5a McNemar厳密 p(0,8)=2^-7",
      abs(p_mc - 2 * (0.5 ** 8)) < 1e-9, f"(p={p_mc:.6f})")
check("T5b 本番の片側95%上限（0件/100分）≈1.80回/h",
      abs(up1 - 1.797) < 0.02, f"(hi={up1:.3f})")
check("T5c クリップ単位paired bootstrap（定差1s）CI=[1,1]",
      bs is not None and abs(bs[0] - 1.0) < 1e-9 and abs(bs[1] - 1.0) < 1e-9
      and abs(bs[2] - 1.0) < 1e-9, f"({bs})")

# ---------------- T6: 警告音=v1定数＋因果時刻 ----------------
f2 = v2.fires_for_clip({10: [(0, 0.0, None)], 11: [(0, 0.0, None)]}, 100, 60.0)
f3 = v2.fires_for_clip({10: [(0, 0.0, None)], 11: [(0, 0.0, None)],
                        12: [(0, 0.0, None)]}, 100, 60.0)
pred_rf = {k: [(0, 0.0, None)] for k in (10, 11, 12, 40, 41, 42, 70, 71, 72)}
f_rf = v2.fires_for_clip(pred_rf, 100, 60.0)
w3 = f3.get(0, {}).get("warn", [])
check("T6 警告=3フレーム連続＋因果時刻1.3s",
      not f2.get(0, {}).get("warn") and len(w3) == 1 and abs(w3[0][0] - 1.3) < 1e-9,
      f"(3f={w3})")
w_rf = f_rf.get(0, {}).get("warn", [])
check("T6b 不応期5s（3s後=抑制/6s後=再発火）",
      len(w_rf) == 2 and abs(w_rf[1][0] - 7.3) < 1e-9, f"({w_rf})")

# ---------------- T7: 負例マスク ----------------
pred7 = {"c3": {68: [(0, 0.0, None)], 69: [(0, 0.0, None)], 70: [(0, 0.0, None)]}}
rows7 = [
    {"clip_id": "c3", "event_id": "1", "trial": "t1", "class": "siren",
     "quadrant": "F", "t_start": "5", "t_cpa": "10"},
    {"clip_id": "c3", "event_id": "1", "trial": "n1", "class": "none",
     "quadrant": "", "t_start": "0", "t_cpa": "60"},
]
ev7, neg7, _ = v2.evaluate(rows7, pred7, link_deg=60.0, has_dist=True)
e7 = [e for e in ev7 if e["class"] == "siren"][0]
check("T7 イベント成功＋負例マスク（二重計上なし・露出控除）",
      e7["notified"] is True and neg7["n_false"] == 0
      and abs(neg7["exposure_s"] - 53.0) < 1e-6,
      f"(false={neg7['n_false']}, exp={neg7['exposure_s']})")

# ---------------- T8/T11: caution車=中通知・リード ----------------
pred8 = {"c4": {30: [(4, 0.0, 2.0)], 31: [(4, 2.0, 2.0)]}}
rows8 = [{"clip_id": "c4", "event_id": "1", "trial": "t1", "class": "car_drive",
          "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "2.5"}]
ev8, _, _ = v2.evaluate(rows8, pred8, link_deg=60.0, has_dist=True)
check("T8 caution車=中通知で成功（リード・象限あり）",
      ev8[0]["notified"] is True and ev8[0]["fired_tier"] == "中"
      and ev8[0]["quad_ok"] is True, f"(tier={ev8[0]['fired_tier']})")
check("T11 【回帰】cautionリード=5−3.2=1.8s",
      abs(ev8[0]["lead"] - 1.8) < 1e-9, f"(lead={ev8[0]['lead']})")

# ---------------- T9:【回帰】2m,2m,1mで強は発火しない ----------------
dseq9 = {10: [(0.0, 2.0)], 11: [(0.0, 2.0)], 12: [(0.0, 1.0)]}
strong9 = v2.dist_triggers_var(dseq9, v2.T3, 20, 60.0)
eps9 = v2.build_episodes(dseq9, 20, 60.0)
ctrl = v2.dist_triggers_var({10: [(0.0, 1.4)], 11: [(0.0, 1.4)]}, v2.T3, 20, 60.0)
check("T9 【回帰】2m,2m,1m→強トリガ0・エピソードtier=中（1.4,1.4→強1）",
      len(strong9) == 0 and len(eps9) == 1 and eps9[0]["tier"] == "中"
      and len(ctrl) == 1,
      f"(eps={eps9})")

# ---------------- T10:【回帰】safe車への強発火=失敗 ----------------
pred10 = {"c5": {30: [(4, 0.0, 1.0)], 31: [(4, 1.0, 1.0)]}}
rows10 = [{"clip_id": "c5", "event_id": "1", "trial": "t1", "class": "car_drive",
           "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "10"}]
ev10, _, _ = v2.evaluate(rows10, pred10, link_deg=60.0, has_dist=True)
check("T10 【回帰】safe車への強発火は失敗",
      ev10[0]["gt_tier"] == "safe" and ev10[0]["notified"] is False
      and ev10[0]["fired_tier"] == "強", f"(notified={ev10[0]['notified']})")

# ---------------- T12:【回帰】safe車への中通知も失敗 ----------------
pred12 = {"c6": {30: [(4, 0.0, 2.0)], 31: [(4, 0.0, 2.0)]}}
rows12 = [{"clip_id": "c6", "event_id": "1", "trial": "t1", "class": "car_drive",
           "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "10"}]
ev12, _, ex12 = v2.evaluate(rows12, pred12, link_deg=60.0, has_dist=True)
check("T12 【回帰】safe車への中通知も失敗（safe成功=強・中とも無し）",
      ev12[0]["notified"] is False and ev12[0]["fired_tier"] == "中"
      and ex12["n_mid_on_safe"] == 1,
      f"(notified={ev12[0]['notified']}, tier={ev12[0]['fired_tier']})")

# ---------------- T13:【回帰】同一エピソードの二重割当なし ----------------
pred13 = {"c7": {30: [(4, 0.0, 1.0)], 31: [(4, 0.0, 1.0)]}}
rows13 = [
    {"clip_id": "c7", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "2.0"},
    {"clip_id": "c7", "event_id": "2", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "2.0"},
]
ev13, _, _ = v2.evaluate(rows13, pred13, link_deg=60.0, has_dist=True)
n_ok13 = sum(1 for e in ev13 if e["notified"])
check("T13 【回帰】1エピソードは1イベントのみ（強/中の二重割当なし）",
      n_ok13 == 1, f"(成功={n_ok13}/2)")

# ---------------- T14:【回帰】エピソード統合=正規規則 ----------------
eps_a = v2.build_episodes({10: [(0.0, 1.0)], 11: [(0.0, 1.0)],
                           15: [(0.0, 1.0)], 16: [(0.0, 1.0)]}, 30, 60.0)
eps_b = v2.build_episodes({10: [(0.0, 1.0)], 11: [(0.0, 1.0), (50.0, 1.0)],
                           12: [(50.0, 1.0)]}, 30, 60.0)
check("T14 【回帰】フレーム差5→2エピソード・方位差50°→2エピソード",
      len(eps_a) == 2 and len(eps_b) == 2,
      f"(gap={len(eps_a)}, az={len(eps_b)})")

# ---------------- T15:【回帰】横距離欠落は未採点 ----------------
rows15 = [{"clip_id": "c7", "event_id": "1", "trial": "t1", "class": "car_drive",
           "quadrant": "F", "t_start": "1", "t_cpa": "5"}]
ev15, _, ex15 = v2.evaluate(rows15, pred13, link_deg=60.0, has_dist=True)
check("T15 【回帰】横距離欠落→未採点（分母除外・警告件数計上）",
      ev15[0]["notified"] is None and ex15["n_unscored"] == 1,
      f"(notified={ev15[0]['notified']}, unscored={ex15['n_unscored']})")

# ---------------- T16:【回帰】非有限・負の横距離は未採点 ----------------
invalid_lateral = ("NaN", "Infinity", "-Infinity", "-1")
rows16 = [
    {"clip_id": "c7", "event_id": str(i), "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": value}
    for i, value in enumerate(invalid_lateral, start=1)
]
rows16.append(
    {"clip_id": "c7", "event_id": "5", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "1", "t_cpa": "5", "横距離m": "0"})
ev16, _, ex16 = v2.evaluate(rows16, pred13, link_deg=60.0, has_dist=True)
check("T16 【回帰】NaN/±Infinity/負値→未採点・0m→有効",
      all(e["notified"] is None for e in ev16[:4])
      and ev16[4]["gt_tier"] == "critical" and ev16[4]["notified"] is True
      and ex16["n_unscored"] == 4,
      f"(invalid={[e['notified'] for e in ev16[:4]]}, "
      f"zero={(ev16[4]['gt_tier'], ev16[4]['notified'])}, "
      f"unscored={ex16['n_unscored']})")

print()
if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print(f"ALL PASS ({n_checks} checks)")
sys.exit(0)
