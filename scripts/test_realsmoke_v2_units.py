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

--- 2026-08-14 実録再設計レビュー反映（step19b 切り出し／境界一意化） ---
T17: 切り出し計画 — 最接近が8s地点／原録音の端でクランプしても範囲内
T18: 負例の重複分割 — 担当区間が隙間なく・重なりなくタイルし合計=録音長
T19: cut_audio — 96kHz原本から10秒/24kHzを切り出し、目印が8s地点に来る＋ゲイン適用
T20: 注釈の再基準化 — 時刻の移動・orig_file/cut_offset_s付与・最接近が外なら scored=0
T21: 【境界】重複帯の発火は1回だけ数える（半開区間）＋露出合計=録音長
T22: 【境界】3フレーム連続条件が境界で分断されない（重複なしだと取りこぼす）
T23: scored=0 行は未採点だが誤警告マスクとしては働く
T24: --gain-only の較正ゲインが全長変換と一致する（長時間録音の迂回路）
T25〜T29: FOA回転・負例行検査・歩行対・検出F率・注釈appendの回帰
T30: take_id/区分/状態の欠落で計画検査を迂回できない
T31: 計画本数は切り出しclip数ではなく物理take_id単位
T32: orig_duration_sとの照合で負例末尾欠落を検出
T33: 歩行対比は件数だけでなくpair_idごとに静止1・歩行1
T34: イベント別CSVへ対応キーと検出F率を保存
T35: 96kHz原本直切りは較正ゲイン・3軸角の明示指定を必須化
T36: 現行計画210テイク（A〜E各20＋Fバイク10＋歩行対比100）＋別枠負例100分が最終ゲートを通る
T37: C負例の時間を別枠100分へ流用できない
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import os as _os_env
_os_env.environ.setdefault("PYTHONUTF8", "1"); _os_env.environ.setdefault("PYTHONIOENCODING", "utf-8")   # 子プロセスの日本語の出力を UTF-8 に（2026-10-12・T47b/T51 が cp932 の文字化けで落ちていた）
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
cut = _load("s19bcut", "step19b_realsmoke_cut.py")
val = _load("s19cval", "step19c_ann_validate.py")

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

# ================= 2026-08-14 追加分（step19b 切り出し／境界一意化） =================

# ---------------- T17: 切り出し計画（CPA@8s・端でクランプ） ----------------
o_mid = cut.plan_event_cut(t_cpa=20.0, length_s=30.0)          # 20-8=12
o_head = cut.plan_event_cut(t_cpa=3.0, length_s=30.0)          # 負→0へ
o_tail = cut.plan_event_cut(t_cpa=29.5, length_s=30.0)         # 21.5→20へ
check("T17 CPA@8s・端はクランプ（0≤off≤length-dur）",
      abs(o_mid - 12.0) < 1e-9 and o_head == 0.0 and abs(o_tail - 20.0) < 1e-9,
      f"(mid={o_mid}, head={o_head}, tail={o_tail})")

# ---------------- T18: 負例分割の担当区間タイル ----------------
def _tiles_ok(length, dur=10.0, ovl=1.0):
    plan = cut.plan_negative_split(length, dur, ovl)
    abs_segs = sorted((s + a, s + b) for s, a, b in plan)
    if abs(abs_segs[0][0]) > 1e-9 or abs(abs_segs[-1][1] - length) > 1e-9:
        return False, plan, "端が録音全体を覆っていない"
    for (a1, b1), (a2, b2) in zip(abs_segs, abs_segs[1:]):
        if abs(a2 - b1) > 1e-9:
            return False, plan, f"隙間/重複 {b1}→{a2}"
    if any(s < -1e-9 or s + dur > length + 1e-9 for s, _, _ in plan):
        return False, plan, "音声窓が録音外"
    cov = sum(b - a for _, a, b in plan)
    return abs(cov - length) < 1e-9, plan, f"cov={cov}"

ok19, p19, note19 = _tiles_ok(19.0)
ok100, p100, note100 = _tiles_ok(100.0)
ok_odd, p_odd, note_odd = _tiles_ok(25.5)
ok_exact, p_ex, note_ex = _tiles_ok(10.0)
check("T18 担当区間が隙間なく・重なりなくタイル（19s/100s/25.5s/10s）",
      ok19 and ok100 and ok_odd and ok_exact,
      f"(19s:{note19} / 100s:{note100} / 25.5s:{note_odd} / 10s:{note_ex})")
check("T18b 重複帯は前クリップ帰属（後クリップの先頭1sは担当外）",
      abs(p19[0][1]) < 1e-9 and abs(p19[0][2] - 10.0) < 1e-9
      and abs(p19[1][1] - 1.0) < 1e-9, f"({p19})")

# ---------------- T19: cut_audio（96kHz→10s/24kHz・位置とゲイン） ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    td = Path(td)
    fs = 96000
    n = int(15 * fs)
    base = np.zeros((n, 4), dtype=np.float32)
    base[:, 0] = 0.01 * np.sin(2 * np.pi * 300 * np.arange(n) / fs)
    mark = int(11.0 * fs)                                   # 11秒地点に目印バースト
    base[mark:mark + int(0.02 * fs), :] = 0.2   # ×2してもPCM_24でクリップしない振幅
    src = td / "raw96k.wav"
    sf.write(src, base, fs, subtype="FLOAT")
    out = cut.cut_audio(src, start_s=3.0, dur_s=10.0, out_path=td / "c.flac")
    y, sr_o = sf.read(out)
    pk = int(np.argmax(np.abs(y[:, 0])))
    out_g = cut.cut_audio(src, 3.0, 10.0, td / "cg.flac", gain_db=6.0206)
    yg, _ = sf.read(out_g)
    check("T19 96kHz原本→10s/24kHz・目印が8.0s地点・ゲイン+6dBで2倍",
          sr_o == 24000 and len(y) == 240000 and abs(pk / 24000 - 8.0) < 0.01
          and abs(float(np.max(np.abs(yg))) / float(np.max(np.abs(y))) - 2.0) < 0.02,
          f"(fs={sr_o}, n={len(y)}, peak={pk/24000:.3f}s)")

# ---------------- T20: 注釈の再基準化 ----------------
rows20 = [
    {"clip_id": "takeA", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "L", "t_start": "16", "t_cpa": "20", "横距離m": "2.5"},
    {"clip_id": "takeA", "event_id": "2", "trial": "t1", "class": "car_drive",
     "quadrant": "R", "t_start": "21", "t_cpa": "25", "横距離m": "3.0"},
    {"clip_id": "takeA", "event_id": "3", "trial": "t1", "class": "bike_bell",
     "quadrant": "B", "t_start": "0.5", "t_cpa": "2.0"},
]
nb = cut.rebase_rows(rows20, off=12.0, dur=10.0, orig="takeA",
                     new_clip="takeA_e1", target_event="1")
by_ev = {r["event_id"]: r for r in nb}
check("T20 時刻の再基準化・cut_offset_s付与・範囲外イベントは除外",
      len(nb) == 2 and by_ev["1"]["t_cpa"] == "8.00"
      and by_ev["1"]["t_start"] == "4.00"
      and by_ev["1"]["cut_offset_s"] == "12.000"
      and by_ev["1"]["orig_file"] == "takeA"
      and by_ev["1"]["clip_id"] == "takeA_e1" and "3" not in by_ev,
      f"({[(r['event_id'], r['t_start'], r['t_cpa'], r['scored']) for r in nb]})")
check("T20b 最接近がクリップ外の行は scored=0（マスク専用で残す）",
      by_ev["2"]["scored"] == "0" and by_ev["1"]["scored"] == "1"
      and by_ev["2"]["t_start"] == "9.00" and by_ev["2"]["t_cpa"] == "10.00",
      f"(ev2={by_ev['2']['t_start']}-{by_ev['2']['t_cpa']}/{by_ev['2']['scored']})")

# 1つの物理イベントが2クリップで二重採点されないこと（切り出し窓は重なる）
rows20b = [
    {"clip_id": "takeC", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "L", "t_start": "10", "t_cpa": "12", "横距離m": "2.0"},
    {"clip_id": "takeC", "event_id": "2", "trial": "t1", "class": "car_drive",
     "quadrant": "R", "t_start": "14", "t_cpa": "16", "横距離m": "2.0"},
]
cutA = cut.rebase_rows(rows20b, off=4.0, dur=10.0, orig="takeC",
                       new_clip="takeC_e1", target_event="1")   # 窓 4〜14s
cutB = cut.rebase_rows(rows20b, off=8.0, dur=10.0, orig="takeC",
                       new_clip="takeC_e2", target_event="2")   # 窓 8〜18s
scored_pairs = [(r["clip_id"], r["event_id"]) for r in cutA + cutB
                if r["scored"] == "1"]
check("T20c 同じイベントが2クリップで二重採点されない（主対象だけ採点）",
      sorted(scored_pairs) == [("takeC_e1", "1"), ("takeC_e2", "2")],
      f"({sorted(scored_pairs)})")

# 負例分割では「最接近を担当区間に持つクリップ」だけが採点する
rows20d = [{"clip_id": "neg", "event_id": "1", "trial": "n", "class": "bike_bell",
            "quadrant": "F", "t_start": "9.4", "t_cpa": "9.7"}]
d0 = cut.rebase_rows(rows20d, 0.0, 10.0, "neg", "neg_s000", own=(0.0, 10.0))
d1 = cut.rebase_rows(rows20d, 9.0, 10.0, "neg", "neg_s001", own=(1.0, 10.0))
check("T20d 負例分割は担当区間が最接近を含むクリップだけ採点",
      len(d0) == 1 and d0[0]["scored"] == "1"
      and len(d1) == 1 and d1[0]["scored"] == "0",
      f"(s000={d0[0]['scored']}, s001={d1[0]['scored']})")

# ---------------- T21: 境界発火の一意化＋露出合計 ----------------
# 19秒の負例を dur=10/overlap=1 で分割した想定（clip0=担当0-10, clip1=担当1-10）
plan21 = cut.plan_negative_split(19.0, 10.0, 1.0)
rows21 = [cut.negative_row(f"neg_s{i:03d}", "neg", off, a, b)
          for i, (off, a, b) in enumerate(plan21)]
# 絶対10.0秒に警告音の発火（clip0では因果時刻10.0、clip1では1.0）
pred21 = {"neg_s000": {97: [(0, 0.0, None)], 98: [(0, 0.0, None)],
                       99: [(0, 0.0, None)]},
          "neg_s001": {7: [(0, 0.0, None)], 8: [(0, 0.0, None)],
                       9: [(0, 0.0, None)]}}
_, neg21, _ = v2.evaluate(rows21, pred21, link_deg=60.0, has_dist=True)
check("T21 重複帯の同一発火は1回だけ・露出合計=録音長19s",
      neg21["n_false"] == 1 and abs(neg21["exposure_s"] - 19.0) < 1e-9,
      f"(false={neg21['n_false']}, exp={neg21['exposure_s']})")

# ---------------- T22: 境界をまたぐ連続条件が分断されない ----------------
# 絶対 9.8/9.9/10.0 秒の3フレーム連続。clip0は9.9までしか音がない
pred22_ovl = {"neg_s000": {98: [(0, 0.0, None)], 99: [(0, 0.0, None)]},
              "neg_s001": {8: [(0, 0.0, None)], 9: [(0, 0.0, None)],
                           10: [(0, 0.0, None)]}}
_, n22a, _ = v2.evaluate(rows21, pred22_ovl, link_deg=60.0, has_dist=True)
# 重複なし(step=dur)だと後クリップは絶対10.0から始まり1フレームしか見えない
plan22b = cut.plan_negative_split(19.0, 10.0, 0.0)
rows22b = [cut.negative_row(f"nb_s{i:03d}", "nb", off, a, b)
           for i, (off, a, b) in enumerate(plan22b)]
pred22b = {"nb_s000": {98: [(0, 0.0, None)], 99: [(0, 0.0, None)]},
           "nb_s001": {0: [(0, 0.0, None)]}}
_, n22b, _ = v2.evaluate(rows22b, pred22b, link_deg=60.0, has_dist=True)
check("T22 重複1sなら境界またぎの3フレーム連続を検出（重複なしは取りこぼす）",
      n22a["n_false"] == 1 and n22b["n_false"] == 0,
      f"(重複あり={n22a['n_false']}, 重複なし={n22b['n_false']})")

# ---------------- T23: scored=0 は未採点だがマスクとして働く ----------------
rows23 = [
    {"clip_id": "c8", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "9.0", "t_cpa": "10.0", "横距離m": "2.5",
     "scored": "0"},
    {"clip_id": "c8", "event_id": "n1", "trial": "neg", "class": "none",
     "quadrant": "", "t_start": "0", "t_cpa": "10", "scored": "1"},
]
pred23 = {"c8": {90: [(4, 0.0, 2.0)], 91: [(4, 0.0, 2.0)]}}   # 因果9.2sに中通知
ev23, neg23, ex23 = v2.evaluate(rows23, pred23, link_deg=60.0, has_dist=True)
check("T23 scored=0は未採点（分母外）だが誤警告マスクとしては有効",
      ev23[0]["notified"] is None and ex23["n_maskonly"] == 1
      and neg23["n_false"] == 0 and abs(neg23["exposure_s"] - 8.0) < 1e-9,
      f"(notified={ev23[0]['notified']}, maskonly={ex23['n_maskonly']}, "
      f"false={neg23['n_false']}, exp={neg23['exposure_s']})")

# ---------------- T24: --gain-only が全長変換と同じゲインを出す ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    td = Path(td)
    fs = 96000
    t = np.arange(3 * fs) / fs
    w = 0.03 * np.sin(2 * np.pi * 1000 * t)
    sf.write(td / "g.wav", np.stack([w, 0.2 * w, 0.1 * w, 0.3 * w], axis=1), fs,
             subtype="FLOAT")
    g_stream = s19.calib_gain_db(td / "g.wav", laeq=55.0, win=(0.5, 2.5))
    wav, sr = sf.read(td / "g.wav", dtype="float64")
    from scipy.signal import resample_poly as _rp
    w24 = _rp(wav[:, 0], 1, 4)
    g_full = 55.0 - s19.spl_a(w24[int(0.5 * 24000):int(2.5 * 24000)], 24000)
    check("T24 --gain-only の較正ゲインが全長変換と一致（±0.1dB）",
          abs(g_stream - g_full) < 0.1, f"(stream={g_stream:+.3f}, full={g_full:+.3f})")

# ============ 2026-08-15 監査指摘の回帰（T25〜T29） ============

# ---------------- T25: 原本から切る経路でもFOA回転が掛かる ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    td = Path(td)
    fs = 96000
    n = int(12 * fs)
    src = np.zeros((n, 4), dtype=np.float32)
    src[:, 0] = 0.1          # W
    src[:, 3] = 0.1          # X（正面）→ yaw+90°でY（左）へ移るはず
    p = td / "raw.wav"
    sf.write(p, src, fs, subtype="FLOAT")
    o_rot = cut.cut_audio(p, 1.0, 10.0, td / "rot.flac", yaw=90.0)
    o_non = cut.cut_audio(p, 1.0, 10.0, td / "non.flac")
    yr, _ = sf.read(o_rot)
    yn, _ = sf.read(o_non)
    mid = len(yr) // 2
    # 回転あり: X→ほぼ0・Y→+0.1 / 回転なし: X=0.1・Y=0
    check("T25 【回帰】原本切り出しにも step19 と同じFOA回転が掛かる",
          abs(yr[mid, 3]) < 5e-3 and abs(yr[mid, 1] - 0.1) < 5e-3
          and abs(yn[mid, 3] - 0.1) < 5e-3 and abs(yn[mid, 1]) < 5e-3,
          f"(rot X={yr[mid,3]:.4f} Y={yr[mid,1]:.4f} / "
          f"non X={yn[mid,3]:.4f} Y={yn[mid,1]:.4f})")

# ---------------- T26: 検証器が壊れた負例行を落とす ----------------
def _val(rows, cut_mode=True, dur=10.0):
    val.errs.clear()
    val.warns.clear()
    val.validate(rows, cut_mode, dur, dict(val.PLAN_DEFAULT))
    return list(val.errs)

ok_neg = [{"clip_id": "n_s000", "event_id": "n1", "trial": "neg", "class": "none",
           "quadrant": "", "t_start": "0", "t_cpa": "10", "orig_file": "n",
           "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1",
           "take_id": "neg-n", "pair_id": "", "区分": "負例露出", "状態": "静止"}]
bad_neg = [dict(ok_neg[0], orig_file="", cut_offset_s="", scored="garbage")]
gap_neg = [dict(ok_neg[0], orig_duration_s="19"),
           {"clip_id": "n_s001", "event_id": "n1", "trial": "neg", "class": "none",
            "quadrant": "", "t_start": "2", "t_cpa": "10", "orig_file": "n",
            "orig_duration_s": "19", "cut_offset_s": "9", "scored": "1",
            "take_id": "neg-n", "pair_id": "", "区分": "負例露出", "状態": "静止"}]
head_neg = [dict(ok_neg[0], t_start="1", cut_offset_s="0")]
e_ok, e_bad, e_gap, e_head = (_val(ok_neg), _val(bad_neg), _val(gap_neg),
                              _val(head_neg))
check("T26 【回帰】負例行の orig_file/cut_offset_s/scored 不正を検出",
      not e_ok and len(e_bad) >= 3, f"(ok={len(e_ok)}, bad={len(e_bad)})")
check("T26b 【回帰】担当区間の隙間・先頭0s未達を検出",
      any("隙間" in m for m in e_gap) and any("先頭" in m for m in e_head),
      f"(gap={len(e_gap)}, head={len(e_head)})")

# ---------------- T27: 歩行対比の対不整合を検出 ----------------
def _walk(n_stat, n_walk):
    rows = []
    for i in range(n_stat + n_walk):
        rows.append({"clip_id": f"w{i}", "event_id": "1", "trial": "w",
                     "class": "car_drive", "quadrant": "L", "t_start": "1",
                     "t_cpa": "8", "横距離m": "3.0", "区分": "歩行",
                     "状態": "静止" if i < n_stat else "歩行",
                     "take_id": f"walk-take-{i}",
                     "pair_id": f"pair-{i if i < n_stat else i - n_stat}",
                     "orig_file": f"w{i}", "orig_duration_s": "10",
                     "cut_offset_s": "0", "scored": "1"})
    return _val(rows)

check("T27 歩行対比が静止/歩行の対で揃わなければエラー",
      not any("静止1・歩行1" in m for m in _walk(10, 10))
      and any("静止1・歩行1" in m for m in _walk(12, 8)),
      f"(10-10={len(_walk(10,10))}, 12-8={len(_walk(12,8))})")

# ---------------- T28: 検出フレーム率 ----------------
pred28 = {"c9": {k: [(4, 0.0, 2.0)] for k in range(10, 40)}}   # 1.0〜4.0sだけ検出
rows28 = [{"clip_id": "c9", "event_id": "1", "trial": "t1", "class": "car_drive",
           "quadrant": "F", "t_start": "0", "t_cpa": "8", "横距離m": "2.5"}]
ev28, _, _ = v2.evaluate(rows28, pred28, link_deg=60.0, has_dist=True)
check("T28 検出フレーム率＝窓内でクラスが出ているフレームの割合（30/80）",
      abs(ev28[0]["frame_recall"] - 0.375) < 1e-9,
      f"(fr={ev28[0]['frame_recall']})")

# ---------------- T29: --append で複数の負例ファイルを1本のCSVへ ----------------
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    csv_path = td / "ann_all.csv"
    cut._write_ann(csv_path, [cut.negative_row("a_s000", "a", 0.0, 0.0, 10.0)])
    cut._write_ann(csv_path, [cut.negative_row("b_s000", "b", 0.0, 0.0, 10.0)],
                   append=True)
    import csv as _csv
    got29 = list(_csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    cut._write_ann(csv_path, [cut.negative_row("b_s000", "b", 0.0, 0.0, 9.0)],
                   append=True)
    got29b = list(_csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    check("T29 --append で追記・同一clipは新しい行で置換",
          len(got29) == 2 and len(got29b) == 2
          and [r["clip_id"] for r in got29b].count("b_s000") == 1
          and next(r for r in got29b if r["clip_id"] == "b_s000")["t_cpa"] == "9.000",
          f"(n={len(got29)}→{len(got29b)})")

# ---------------- T30: 計画列は必須（--strict迂回を防ぐ） ----------------
missing_plan = [{"clip_id": "x", "event_id": "1", "trial": "x",
                 "class": "car_drive", "quadrant": "L", "t_start": "1",
                 "t_cpa": "8", "横距離m": "3", "orig_file": "x",
                 "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1"}]
e30 = _val(missing_plan)
check("T30 take_id/区分/状態の欠落を必須列エラーにする",
      any("必須列" in m for m in e30), f"(errors={e30})")

# ---------------- T31: 計画本数は切り出しclipでなくtake_id単位 ----------------
plan31 = cut.plan_negative_split(20.0)
rows31 = [cut.negative_row(f"c_s{i:03d}", "c", off, a, b,
                           {"take_id": "C-take-1", "pair_id": "", "区分": "C",
                            "状態": "静止"}, orig_duration_s=20.0)
          for i, (off, a, b) in enumerate(plan31)]
val.errs.clear()
val.warns.clear()
with_plan31 = dict.fromkeys(val.PLAN_DEFAULT, 0)
with_plan31["C"] = 1
val.validate(rows31, True, 10.0, with_plan31)
check("T31 C負例1テイクを3クリップへ分割しても計画数は1テイク",
      not val.errs and not any("テイク（計画" in w for w in val.warns),
      f"(clips={len(rows31)}, errors={val.errs}, warnings={val.warns})")

# ---------------- T32: 原録音末尾クリップ欠落を検出 ----------------
plan32 = cut.plan_negative_split(100.0)
rows32 = [cut.negative_row(f"n_s{i:03d}", "n", off, a, b,
                           {"take_id": "N-long", "pair_id": "", "区分": "負例露出",
                            "状態": "静止"}, orig_duration_s=100.0)
          for i, (off, a, b) in enumerate(plan32[:-1])]
val.errs.clear()
val.warns.clear()
val.validate(rows32, True, 10.0, dict.fromkeys(val.PLAN_DEFAULT, 0))
check("T32 100秒負例の末尾クリップ欠落をorig_duration_sで検出",
      any("担当終端" in m and "100.000" in m for m in val.errs),
      f"(errors={val.errs})")

# ---------------- T33: 同数でもpair_idが崩れた歩行比較は不合格 ----------------
rows33 = []
for i in range(20):
    rows33.append({"clip_id": f"p{i}", "event_id": "1", "trial": "walk",
                   "class": "car_drive", "quadrant": "L", "t_start": "1",
                   "t_cpa": "8", "横距離m": "3", "take_id": f"take-{i}",
                   "pair_id": f"{'S' if i < 10 else 'W'}-{i % 10}",
                   "区分": "歩行", "状態": "静止" if i < 10 else "歩行",
                   "orig_file": f"p{i}", "orig_duration_s": "10",
                   "cut_offset_s": "0", "scored": "1"})
val.errs.clear()
val.warns.clear()
val.validate(rows33, True, 10.0, {**dict.fromkeys(val.PLAN_DEFAULT, 0), "歩行": 20})
check("T33 静止10・歩行10でもpair_idが対応しなければ不合格",
      any("静止1・歩行1" in m for m in val.errs), f"(errors={len(val.errs)})")

# ---------------- T34: イベント別CSVに歩行対応キーと検出F率を出す ----------------
with tempfile.TemporaryDirectory() as td:
    p34 = Path(td) / "events.csv"
    e34 = dict(ev28[0], take_id="take-34", pair_id="pair-34", state="歩行")
    v2.write_events_csv(p34, [e34])
    import csv as _csv34
    got34 = next(_csv34.DictReader(open(p34, encoding="utf-8-sig")))
    check("T34 イベント別CSVにtake_id/pair_id/状態/検出F率を保存",
          got34["take_id"] == "take-34" and got34["pair_id"] == "pair-34"
          and got34["state"] == "歩行" and got34["frame_recall"] == "0.375",
          f"(row={got34})")

# ---------------- T35: 原本直切りの較正・回転引数を必須化 ----------------
raw_missing = False
try:
    cut.require_raw_metadata(96000, ["step19b", "--gain-db", "0"])
except ValueError:
    raw_missing = True
raw_explicit_ok = True
try:
    cut.require_raw_metadata(96000, ["step19b", "--gain-db", "0", "--pitch", "0",
                                     "--roll", "0", "--yaw", "0"])
except ValueError:
    raw_explicit_ok = False
check("T35 96kHz原本はgain/pitch/roll/yawの明示指定が必須（0値は可）",
      raw_missing and raw_explicit_ok)

# ---------------- T36: 現行計画 210 テイク＋負例100分はstrict相当で合格（2026-09-05 監査 §6: 計画は val.PLAN_DEFAULT から組む） ----------------
rows36 = []
for kind, n_kind in val.PLAN_DEFAULT.items():
    if kind == "歩行":
        continue
    for i in range(n_kind):
        common = {"clip_id": f"{kind}{i}", "event_id": "1", "trial": kind,
                  "take_id": f"{kind}-take-{i}", "pair_id": "", "区分": kind,
                  "状態": "静止", "orig_file": f"{kind}{i}",
                  "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1", "calibration_id": "S-calib"}
        if kind == "C":
            rows36.append({**common, "class": "none", "quadrant": "",
                           "t_start": "0", "t_cpa": "10"})
        elif kind == "F":
            rows36.append({**common, "class": "bike", "quadrant": "L",
                           "t_start": "1", "t_cpa": "8", "横距離m": "3"})
        else:
            rows36.append({**common, "class": "car_drive", "quadrant": "L",
                           "t_start": "1", "t_cpa": "8", "横距離m": "3"})
N_WALK = val.PLAN_DEFAULT["歩行"]
for i in range(N_WALK):
    rows36.append({"clip_id": f"walk{i}", "event_id": "1", "trial": "walk",
                   "class": "car_drive", "quadrant": "L", "t_start": "1",
                   "t_cpa": "8", "横距離m": "3", "take_id": f"walk-take-{i}",
                   "pair_id": f"walk-pair-{i % (N_WALK // 2)}", "区分": "歩行",
                   "状態": "静止" if i < N_WALK // 2 else "歩行", "orig_file": f"walk{i}",
                   "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1", "calibration_id": "S-calib"})
for i, (off, a, b) in enumerate(cut.plan_negative_split(6000.0)):
    rows36.append(cut.negative_row(
        f"exposure_s{i:04d}", "exposure", off, a, b,
        {"take_id": "exposure-take", "pair_id": "", "区分": "負例露出", "状態": "静止", "calibration_id": "S-calib"},
        orig_duration_s=6000.0))
val.errs.clear()
val.warns.clear()
import contextlib as _contextlib
import io as _io
with _contextlib.redirect_stdout(_io.StringIO()):
    val.validate(rows36, True, 10.0, dict(val.PLAN_DEFAULT))
check("T36 現行計画210テイク（F含む）＋別枠負例100分がstrict相当で合格",
      not val.errs and not val.warns,
      f"(rows={len(rows36)}, errors={val.errs}, warnings={val.warns})")

# ---------------- T37: C負例は別枠100分の代用にならない ----------------
rows37 = [cut.negative_row(
    f"c_long_s{i:04d}", "c_long", off, a, b,
    {"take_id": "C-long-take", "pair_id": "", "区分": "C", "状態": "静止"},
    orig_duration_s=6000.0)
    for i, (off, a, b) in enumerate(cut.plan_negative_split(6000.0))]
val.errs.clear()
val.warns.clear()
plan37 = dict.fromkeys(val.PLAN_DEFAULT, 0)
plan37["C"] = 1
with _contextlib.redirect_stdout(_io.StringIO()):
    val.validate(rows37, True, 10.0, plan37)
check("T37 C負例が100分あっても区分=負例露出が0分ならstrict相当で不合格",
      not val.errs and any("区分=負例露出" in w and "0.0分" in w for w in val.warns),
      f"(errors={val.errs}, warnings={val.warns})")

print()

# ---------------- T38: 校正録音（60s）→ 短い本番テイク（25s）へ補正量を引き継ぐ（監査 R01） ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    import subprocess
    td = Path(td)
    fs = 96000
    rng = np.random.default_rng(38)
    calib = (rng.standard_normal((fs * 60, 4)) * 0.01).astype(np.float32)   # 60 s の暗騒音（校正録音）
    tt = np.arange(fs * 25) / fs
    take = np.tile((0.01 * np.sin(2 * np.pi * 1000.0 * tt))[:, None], (1, 4)).astype(np.float32)    # 25 s の本番テイク（1 kHz: リサンプルで実効値が変わらない）
    sf.write(str(td / "S1_calib.wav"), calib, fs)
    sf.write(str(td / "S1_take01.wav"), take, fs)
    # 旧経路: 本番テイクに 10-70 s の窓 → 停止する（監査の再現）
    try:
        s19.calib_gain_db(td / "S1_take01.wav", 52.3, (10.0, 70.0)); old_stops = False
    except AssertionError:
        old_stops = True
    # 新経路: 校正ファイルから 1 回だけ求め、テイクには渡すだけ
    rec = s19.calibration_record(td / "S1_calib.wav", 52.3, (5.0, 55.0), "S1_calib", td / "conv")
    g_expected = s19.calib_gain_db(td / "S1_calib.wav", 52.3, (5.0, 55.0))   # 同じ関数で求めた値と一致するか（json の値の同一性）
    out = s19.convert_with_gain(td / "S1_take01.wav", td / "conv", rec["gain_db"], 0, 0, 0)
    y, fsy = sf.read(str(out))
    ratio = float(np.sqrt((y[:, 0] ** 2).mean()) / np.sqrt((take[:, 0].astype(np.float64) ** 2).mean()))
    check("T38 校正 60s→本番 25s: 旧経路は停止・新経路は補正量を 1 回求めて本番に適用（json に記録）",
          old_stops and (td / "conv" / "calibration_S1_calib.json").exists()
          and abs(rec["gain_db"] - g_expected) < 0.05 and abs(20 * np.log10(ratio) - rec["gain_db"]) < 0.3 and fsy == 24000,
          f"(old_stops={old_stops}, gain={rec['gain_db']:+.2f}dB expected={g_expected:+.2f}, applied={20*np.log10(ratio):+.2f}dB)")

# ---------------- T39: 現場 CSV（2 セッション・同じ take 番号）→ ann_orig（監査 R04） ----------------
s19d = _load("s19dfield", "step19d_field_csv_to_ann.py")
sessions = [{"session_id": "S1", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "10-70", "マイク高さ_cm": "205"},
            {"session_id": "S2", "区分": "A", "用途": "最終評価", "LAeq_dB": "50.1", "暗騒音区間_秒": "10-70", "マイク高さ_cm": "205"}]
events = [{"session_id": "S1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.4", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""},
          {"session_id": "S1", "take_id": "2", "event_id": "1", "class": "car_drive", "象限": "F", "ラップ秒": "7.0", "n_car": "1", "横距離m": "1.0", "状態": "静止", "pair_id": ""},
          {"session_id": "S1", "take_id": "3", "event_id": "1", "class": "none", "象限": "", "ラップ秒": "", "n_car": "0", "横距離m": "", "状態": "静止", "pair_id": ""},
          {"session_id": "S2", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "B", "ラップ秒": "9.0", "n_car": "2", "横距離m": "2.5", "状態": "歩行", "pair_id": "P1"},
          {"session_id": "S2", "take_id": "2", "event_id": "1", "class": "car_drive", "象限": "B", "ラップ秒": "9.5", "n_car": "1", "横距離m": "2.5", "状態": "静止", "pair_id": "P1"},
          {"session_id": "S2", "take_id": "2", "event_id": "2", "class": "horn", "象限": "R", "ラップ秒": "12.0", "n_car": "", "横距離m": "", "状態": "静止", "pair_id": "P1"}]
rows39, warns39 = s19d.convert(sessions, events, 6.0, 0.0, None)
takes39 = {r["take_id"] for r in rows39}
r_s1t1 = next(r for r in rows39 if r["session_id"] == "S1" and r["take_id"].endswith("/01"))
r_s2t1 = next(r for r in rows39 if r["session_id"] == "S2" and r["take_id"].endswith("/01"))
r_horn = next(r for r in rows39 if r["class"] == "horn")
check("T39 現場 CSV→ann_orig: session×take が一意・列名対応・t_start 規則・校正 id・歩行対比の区分・警告音はラップ＝始まり",
      len(rows39) == 6 and len(takes39) == 5 and r_s1t1["take_id"] != r_s2t1["take_id"]
      and r_horn["t_start"] == "12.00" and r_horn["t_cpa"] == "15.00" and r_s1t1["mic_z"] == "2.050"
      and r_s1t1["quadrant"] == "R" and r_s1t1["t_cpa"] == "8.40" and r_s1t1["t_start"] == "2.40"
      and r_s1t1["calibration_id"] == "S1_calib" and r_s1t1["orig_file"] == "S1_take01.wav"
      and r_s2t1["区分"] == "歩行" and r_s2t1["pair_id"] == "S2/P1" and r_s2t1["trial"] == "walk"
      and any(r["class"] == "none" and r["t_start"] == "0" for r in rows39)
      and all(w.startswith("S1/take03: 負例") for w in warns39),
      f"(rows={len(rows39)}, takes={sorted(takes39)}, warns={warns39})")
# step19c は (session_id, take_id) で数える: 2 セッションの take 1 が衝突しない
val.errs.clear(); val.warns.clear()
import contextlib as _ctx39, io as _io39
rows39c = [dict(r, **{"t_cpa": (r["t_cpa"] or "20.0")}) for r in rows39]
with _ctx39.redirect_stdout(_io39.StringIO()):
    val.validate(rows39c, False, 30.0, {"A": 3, "歩行": 2})
check("T39b step19c: 別セッションの同じ take 番号を別テイクとして数える（A=3・歩行=2 で計画一致・S8 の衝突エラー無し）",
      not [e for e in val.errs if "S8" in e] and not [w for w in val.warns if "S8 区分" in w and "負例露出" not in w],
      f"(errors={val.errs}, warnings={val.warns})")

# ---------------- T40: 機会枠（W8/D10）: D=8 は上限目標。0 本でも警告にしない ----------------
val.errs.clear(); val.warns.clear(); val.OPPORTUNITY.clear(); val.OPPORTUNITY.add("D")
rows40 = [{"clip_id": f"A{i}", "event_id": "1", "trial": "A", "class": "car_drive", "quadrant": "L", "t_start": "1", "t_cpa": "8",
           "take_id": f"A-{i}", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "2"} for i in range(2)]
with _ctx39.redirect_stdout(_io39.StringIO()):
    val.validate(rows40, False, 10.0, {"A": 2, "D": 8})
check("T40 機会枠 D=8 が 0 本でも S8 の警告にしない（A は計画どおりで警告無し）",
      not [w for w in val.warns if "S8 区分" in w and "負例露出" not in w] and not val.errs, f"(warnings={val.warns}, errors={val.errs})")
val.OPPORTUNITY.clear()

# ---------------- T41: 履歴不足フラグ（R10）と校正 id の伝播（R01） ----------------
rows41 = [{"clip_id": "S1_take01", "event_id": "1", "trial": "A", "class": "car_drive", "quadrant": "B", "t_start": "0.0", "t_cpa": "4.0",
           "take_id": "S1/01", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "1.0"}]
off41 = cut.plan_event_cut(4.0, 25.0, 10.0, 8.0)          # CPA 4 s → 先頭 0 s にクランプ
out41 = cut.rebase_rows(rows41, off41, 10.0, "S1_take01", "S1_take01_e1", target_event="1", orig_duration_s=25.0,
                        calibration_id="S1_calib", gain_db=12.34, cpa_at=8.0)
rows41b = [dict(rows41[0], t_start="10.0", t_cpa="18.0")]
off41b = cut.plan_event_cut(18.0, 25.0, 10.0, 8.0)
out41b = cut.rebase_rows(rows41b, off41b, 10.0, "S1_take01", "S1_take01_e1", target_event="1", orig_duration_s=25.0,
                         calibration_id="S1_calib", gain_db=12.34, cpa_at=8.0)
check("T41 履歴不足: CPA 4 s のテイクは history_short=1（CPA 18 s は 0）、calibration_id/gain_db が行に付く",
      off41 == 0.0 and out41[0]["history_short"] == "1" and out41[0]["calibration_id"] == "S1_calib" and out41[0]["gain_db"] == "12.34"
      and out41b[0]["history_short"] == "0" and abs(float(out41b[0]["t_cpa"]) - 8.0) < 1e-6,
      f"(off={off41}, cpa_in_clip={out41[0]['t_cpa']}, flag={out41[0]['history_short']} / off_b={off41b}, flag_b={out41b[0]['history_short']})")


# ---------------- T42: 警告音の遅れ = 発火 − 鳴り始め、方向は鳴り始め直後（再監査 N02） ----------------
v3 = _load("s20v3", "step20_realsmoke_score_v3.py")
import json as _json42
cfg43 = v3.V43.Cfg43(**_json42.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
pred42 = {"clipH": {}}
for k in range(50, 100):                    # クラクション(1): 5.0 s から鳴る。方位は最初 R(−80°)、6.0 s 以降は B(−170°)
    pred42["clipH"][k] = [(1, -80.0 if k < 60 else -170.0, -5.0, float("nan"))]
rows42 = [{"clip_id": "clipH", "event_id": "1", "trial": "D", "class": "horn", "quadrant": "R", "t_start": "5.0", "t_cpa": "8.0",
           "take_id": "D-1", "pair_id": "", "区分": "D", "状態": "静止", "横距離m": "", "n_car": "", "scored": "1"}]
ev42, _, _ = v3.evaluate(rows42, pred42, cfg43, (2.5, 1.5), 7.5, False, 1.0)
e42 = ev42[0]
check("T42 警告音: 5.0 s 鳴り始め→5.3 s 通知 = 遅れ 0.3 s（リードではない）、方向は鳴り始め直後で R と一致",
      e42["notified"] and e42["delay"] is not None and abs(e42["delay"] - 0.3) < 1e-6 and e42["lead"] is None and e42["quad_ok"] is True,
      f"(delay={e42['delay']}, lead={e42['lead']}, az_est={e42['az_est']}, quad_ok={e42['quad_ok']})")

# ---------------- T43: 歩行対比は主要評価から外し、検出フレーム率を出す（再監査 N05） ----------------
pred43 = {"clipA": {}, "clipP1s": {}, "clipP1w": {}}
for k in range(30, 90):
    for c in ("clipA", "clipP1s"):
        pred43[c][k] = [(4, 120.0, -5.0, 8.0 - (k - 30) * 0.1)]
for k in range(30, 90, 2):
    pred43["clipP1w"][k] = [(4, 120.0, -5.0, 8.0 - (k - 30) * 0.1)]     # 歩行側は半分のフレームだけ検出
base43 = {"event_id": "1", "trial": "A", "class": "car_drive", "quadrant": "L", "t_start": "3.0", "t_cpa": "8.0", "区分": "A", "横距離m": "2.5", "n_car": "1", "scored": "1"}
rows43 = [dict(base43, clip_id="clipA", take_id="A-1", pair_id="", 状態="静止"),
          dict(base43, clip_id="clipP1s", take_id="W-1", pair_id="P1", 状態="静止", 区分="歩行", trial="walk"),
          dict(base43, clip_id="clipP1w", take_id="W-2", pair_id="P1", 状態="歩行", 区分="歩行", trial="walk")]
ev43, _, _ = v3.evaluate(rows43, pred43, cfg43, (2.5, 1.5), 7.5, False)
main43, side43, _ = v3.summarize(ev43, False)
pairs43, diffs43, _amb43 = v3.walk_pairs_summary(side43["walk"])
check("T43 歩行対比: A 1 本だけが主要評価、静止側も含めた対 1 組は別集計、検出フレーム率が出る（歩行 < 静止）",
      len(main43) == 1 and len(side43["walk"]) == 2 and len(pairs43) == 1 and pairs43[0][3] is not None and pairs43[0][4] is not None
      and pairs43[0][4] < pairs43[0][3] and diffs43 and diffs43[0] < 0,
      f"(main={len(main43)}, walk={len(side43['walk'])}, fr_static={pairs43[0][3]}, fr_walk={pairs43[0][4]})")

# ---------------- T44: 区分 D の内訳検査（再監査 N03）: 0 本=不合格 / 必須 16 のみ=合格 / 満数 20=合格 ----------------
def rows_D(n_cross, n_beep, n_horn, n_siren):
    out = [{"clip_id": "A_0", "event_id": "1", "trial": "A", "class": "car_drive", "quadrant": "L", "t_start": "1", "t_cpa": "8",
            "take_id": "A-0", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "2"}]      # 空の入力にならないよう A を 1 本
    for cls, n in (("crossing", n_cross), ("backup_beep", n_beep), ("horn", n_horn), ("siren", n_siren)):
        for i in range(n):
            out.append({"clip_id": f"D_{cls}{i}", "event_id": "1", "trial": "D", "class": cls, "quadrant": "L", "t_start": "1", "t_cpa": "4",
                        "take_id": f"D-{cls}-{i}", "pair_id": "", "区分": "D", "状態": "静止", "横距離m": ""})
    return out
def s8_warns(rows, plan):
    val.errs.clear(); val.warns.clear()
    with _ctx39.redirect_stdout(_io39.StringIO()):
        val.validate(rows, False, 10.0, dict(plan))
    return [w for w in val.warns if "S8 区分D" in w]
val.SUBPLAN.clear(); val.SUBPLAN["D"] = {"crossing": ("=", 8), "backup_beep": ("=", 4), "horn": ("=", 4), "siren": ("<=", 4)}
w0 = s8_warns(rows_D(0, 0, 0, 0), {"A": 1, "D": 20}); w16 = s8_warns(rows_D(8, 4, 4, 0), {"A": 1, "D": 20}); w20 = s8_warns(rows_D(8, 4, 4, 4), {"A": 1, "D": 20}); w_bad = s8_warns(rows_D(8, 4, 0, 4), {"A": 1, "D": 20})
val.SUBPLAN.clear()
check("T44 D の内訳: 0 本→警告あり、必須 16 のみ→警告なし、満数 20→警告なし、クラクション欠落（16 本でも内訳違い）→警告あり",
      bool(w0) and not w16 and not w20 and bool(w_bad), f"(0本={len(w0)}, 16={len(w16)}, 20={len(w20)}, 内訳違い={len(w_bad)})")

# ---------------- T45: 任意の原本名・2 セッション・校正ごとの補正量で、現場 CSV→変換→校正→切り出し→検査を通す（再監査 N01/N04） ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    import subprocess, csv as _csv45
    td = Path(td); raw = td / "raw"; raw.mkdir()
    fs = 24000; tt = np.arange(fs * 12) / fs
    def wav(name, amp):
        sf.write(str(raw / name), np.tile((amp * np.sin(2 * np.pi * 1000.0 * tt))[:, None], (1, 4)).astype(np.float32), fs)
    wav("S1_calib.wav", 0.01); wav("ZOOM0001.wav", 0.01); wav("S2_calib.wav", 0.001); wav("REC_A.wav", 0.001)
    with open(td / "session.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv45.DictWriter(f, fieldnames=["session_id", "区分", "用途", "LAeq_dB", "暗騒音区間_秒", "校正原本", "マイク高さ_cm"]); w.writeheader()
        w.writerow({"session_id": "S1", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-12", "校正原本": "S1_calib.wav", "マイク高さ_cm": "205"})
        w.writerow({"session_id": "S2", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-12", "校正原本": "S2_calib.wav", "マイク高さ_cm": "205"})
    with open(td / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv45.DictWriter(f, fieldnames=["session_id", "take_id", "event_id", "class", "象限", "ラップ秒", "n_car", "横距離m", "状態", "pair_id", "原本"]); w.writeheader()
        w.writerow({"session_id": "S1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "9.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": "", "原本": "ZOOM0001.wav"})
        w.writerow({"session_id": "S2", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "L", "ラップ秒": "9.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": "", "原本": "REC_A.wav"})
    py = sys.executable; sc = ROOT / "scripts"
    r1 = subprocess.run([py, str(sc / "step19d_field_csv_to_ann.py"), "--session", str(td / "session.csv"), "--events", str(td / "events.csv"), "--out", str(td / "ann_orig.csv"), "--audio-dir", str(raw)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    conv = td / "conv"
    r2 = subprocess.run([py, str(sc / "step19_realsmoke_convert.py"), "--in", str(raw), "--calib", str(raw / "S1_calib.wav"), "--laeq", "52.3", "--gain-only", "--out", str(conv)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    r3 = subprocess.run([py, str(sc / "step19_realsmoke_convert.py"), "--in", str(raw), "--calib", str(raw / "S2_calib.wav"), "--laeq", "52.3", "--gain-only", "--out", str(conv)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    r4 = subprocess.run([py, str(sc / "step19b_realsmoke_cut.py"), "--mode", "event", "--in", str(raw), "--ann", str(td / "ann_orig.csv"), "--calib-dir", str(conv), "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", str(td / "clips"), "--ann-out", str(td / "ann_all.csv")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    r5 = subprocess.run([py, str(sc / "step19c_ann_validate.py"), "--ann", str(td / "ann_all.csv"), "--cut", "--plan", "A=2"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    ann_all = list(_csv45.DictReader(open(td / "ann_all.csv", encoding="utf-8-sig"))) if (td / "ann_all.csv").exists() else []
    g = {r["orig_file"]: (r.get("calibration_id"), float(r.get("gain_db") or 0)) for r in ann_all}
    j1 = _json42.loads((conv / "calibration_S1_calib.json").read_text(encoding="utf-8")) if (conv / "calibration_S1_calib.json").exists() else {}
    j2 = _json42.loads((conv / "calibration_S2_calib.json").read_text(encoding="utf-8")) if (conv / "calibration_S2_calib.json").exists() else {}
    ok45 = (r1.returncode == 0 and r2.returncode == 0 and r3.returncode == 0 and r4.returncode == 0 and r5.returncode == 0
            and len(ann_all) == 2 and "ZOOM0001" in g and "REC_A" in g and {r["clip_id"] for r in ann_all} == {"eval_field_clip01001", "eval_field_clip02001"}   # 2026-09-22 命名 A（旧 S1__ZOOM0001_e1 / S2__REC_A_e1）
            and g["ZOOM0001"][0] == "S1_calib" and g["REC_A"][0] == "S2_calib"
            and abs(g["ZOOM0001"][1] - j1.get("gain_db", 99)) < 0.05 and abs(g["REC_A"][1] - j2.get("gain_db", 99)) < 0.05
            and abs((g["REC_A"][1] - g["ZOOM0001"][1]) - 20.0) < 0.3
            and all(r.get("mic_z") == "2.050" for r in ann_all))
    check("T45 現場CSV→変換→校正→切り出し→検査を 2 セッション・任意の原本名で通し、各原本に自分の校正の補正量（差 20 dB）が付く",
          ok45, f"(rc={[r.returncode for r in (r1, r2, r3, r4, r5)]}, rows={len(ann_all)}, g={g}, j1={j1.get('gain_db')}, j2={j2.get('gain_db')})"
          + ("" if ok45 else f"\n  step19b: {r4.stdout[-600:]} {r4.stderr[-400:]}\n  step19c: {r5.stdout[-400:]}"))
    # 原本が無い注釈 / 注釈が無い原本 → 停止（--allow-unmatched なしで終了コード 1）
    wav("ORPHAN.wav", 0.01)
    r6 = subprocess.run([py, str(sc / "step19b_realsmoke_cut.py"), "--mode", "event", "--in", str(raw), "--ann", str(td / "ann_orig.csv"), "--calib-dir", str(conv), "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", str(td / "clips2"), "--ann-out", str(td / "ann_all2.csv")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("T45b 注釈の無い原本があると step19b は停止（終了コード 1）し、対応の不一致を表示する", r6.returncode == 1 and "対応の不一致" in r6.stdout, f"(rc={r6.returncode})")

# ---------------- T46: session.csv の 用途=調整用 は最終評価の入力から外れる（A′・再監査 N06） ----------------
sessions46 = [{"session_id": "S1", "区分": "A", "用途": "最終評価"}, {"session_id": "T1", "区分": "A", "用途": "調整用"}]
events46 = [{"session_id": "S1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""},
            {"session_id": "T1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""}]
rows46, _ = s19d.convert(sessions46, events46, 6.0, 0.0, None)
check("T46 用途列: 調整用セッションの行に 用途=調整用 が付く（変換器の出力で最終評価と分離できる）",
      len(rows46) == 2 and {r["用途"] for r in rows46} == {"最終評価", "調整用"}, f"(用途={[r['用途'] for r in rows46]})")

# ---------------- T47: step19e 校正の点検 — 指パッチン 4 方位の方位読み・取り違えの言い当て・--yaw の符号（監査 R09） ----------------
import numpy as _np47, soundfile as _sf47, math as _m47, json
_s19 = _load("s19_47", "step19_realsmoke_convert.py")
def _snaps47(azs, sr=48000, ysign=1.0):
    """SN3D で衝撃音を az（左＋）に置く。W=s, Y=s·sin, Z=0, X=s·cos。1 秒間隔、暗騒音つき。"""
    rng = _np47.random.default_rng(47); n = sr * (len(azs) + 1)
    out = rng.normal(0, 1e-4, (n, 4)).astype(_np47.float32)
    t = _np47.arange(int(sr * 0.02)) / sr
    s = (_np47.sin(2 * _np47.pi * 3000 * t) * _np47.exp(-t / 0.004)).astype(_np47.float32)
    for k, az in enumerate(azs):
        o = sr * (k + 1); r = _m47.radians(az)
        out[o:o + len(s), 0] += s; out[o:o + len(s), 1] += ysign * s * _m47.sin(r); out[o:o + len(s), 3] += s * _m47.cos(r)
    return out, sr
import tempfile as _tf47
td47 = Path(_tf47.mkdtemp())
def _run47(x, sr, expect="前,右,後,左"):
    p = td47 / "snap47.wav"; _sf47.write(str(p), x, sr, subtype="PCM_24")
    return subprocess.run([py, str(sc / "step19e_check_azimuth.py"), "--in", str(p), "--expect", expect, "--json", str(td47 / "snap47.json")], capture_output=True, text=True, encoding="utf-8", errors="replace")
x47, sr47 = _snaps47([0, -90, 180, 90]); r47 = _run47(x47, sr47)
j47 = json.loads((td47 / "snap47.json").read_text(encoding="utf-8"))
check("T47a 正しい並び（前 0 / 右 −90 / 後 180 / 左 +90）を 4 打とも ±3° で読み、合格（rc=0）",
      r47.returncode == 0 and j47["ok"] and len(j47["impulses"]) == 4 and all(abs(((r["az"] - e + 180) % 360) - 180) <= 3 for r, e in zip(j47["impulses"], [0, -90, 180, 90])),
      f"(rc={r47.returncode}, az={[r['az'] for r in j47['impulses']]})")
x47b, _ = _snaps47([0, -90, 180, 90], ysign=-1.0); r47b = _run47(x47b, sr47)
check("T47b Y の符号が逆だと「左右が反転」と言い当てて不合格（rc=1）", r47b.returncode == 1 and "左右が反転" in r47b.stdout, f"(rc={r47b.returncode})")
x47c, _ = _snaps47([12.0]); r47c = _run47(x47c, sr47, expect="前")
j47c = json.loads((td47 / "snap47.json").read_text(encoding="utf-8"))
R47 = _s19.rot_matrix(0, 0, j47c["yaw_for_step19"])
v = R47 @ _np47.array([_m47.cos(_m47.radians(12.0)), _m47.sin(_m47.radians(12.0)), 0.0])   # (X,Y,Z) の順で回す
az_after = _m47.degrees(_m47.atan2(v[1], v[0]))
check("T47c 正面が +12° にずれた 1 打: 正面ズレ_deg=+12、--yaw=−12 を提示し、step19 の rot_matrix にその値を渡すと 0° に戻る（符号が一致）",
      r47c.returncode == 1 and abs(j47c["front_offset_deg"] - 12.0) <= 1.5 and abs(j47c["yaw_for_step19"] + 12.0) <= 1.5 and abs(az_after) <= 1.5,
      f"(front={j47c['front_offset_deg']}, yaw={j47c['yaw_for_step19']}, after={az_after:.1f})")

# ---------------- T48: ノートの原本構成（校正 ZOOM0001 / 点検 ZOOM0002 / 事象 ZOOM0003 が同じ raw/）で 変換→校正→切り出し が追加指定なしで通る（ノート監査 H01） ----------------
with tempfile.TemporaryDirectory() as td48:
    import soundfile as sf
    import subprocess, csv as _csv48
    td48 = Path(td48); raw = td48 / "raw"; raw.mkdir(); conv = td48 / "conv"
    fs = 24000; tt = np.arange(fs * 12) / fs
    for name in ("ZOOM0001.wav", "ZOOM0002.wav", "ZOOM0003.wav", "ZOOM0004.wav"):
        sf.write(str(raw / name), np.tile((0.01 * np.sin(2 * np.pi * 1000.0 * tt))[:, None], (1, 4)).astype(np.float32), fs)
    with open(td48 / "session.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv48.DictWriter(f, fieldnames=["session_id", "区分", "用途", "LAeq_dB", "暗騒音区間_秒", "校正原本", "点検原本", "点検方式", "マイク高さ_cm", "備考"]); w.writeheader()
        w.writerow({"session_id": "20260920_A1", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-12", "校正原本": "ZOOM0001.wav", "点検原本": "ZOOM0002.wav", "点検方式": "4方位", "マイク高さ_cm": "205", "備考": "電池交換／除外: ZOOM0004.wav"})
    with open(td48 / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv48.DictWriter(f, fieldnames=["session_id", "take_id", "event_id", "原本", "class", "象限", "ラップ秒", "n_car", "横距離m", "状態", "pair_id"]); w.writeheader()
        w.writerow({"session_id": "20260920_A1", "take_id": "1", "event_id": "1", "原本": "ZOOM0003.wav", "class": "car_drive", "象限": "R", "ラップ秒": "9.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""})
    py = sys.executable; sc = ROOT / "scripts"
    a1 = subprocess.run([py, str(sc / "step19d_field_csv_to_ann.py"), "--session", str(td48 / "session.csv"), "--events", str(td48 / "events.csv"), "--out", str(td48 / "ann_orig.csv"), "--audio-dir", str(raw)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    a2 = subprocess.run([py, str(sc / "step19_realsmoke_convert.py"), "--in", str(raw), "--calib", str(raw / "ZOOM0001.wav"), "--laeq", "52.3", "--gain-only", "--out", str(conv)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    a3 = subprocess.run([py, str(sc / "step19b_realsmoke_cut.py"), "--mode", "event", "--in", str(raw), "--ann", str(td48 / "ann_orig.csv"), "--calib-dir", str(conv), "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", str(td48 / "clips"), "--ann-out", str(td48 / "ann_all.csv")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows48 = list(_csv48.DictReader(open(td48 / "ann_all.csv", encoding="utf-8"))) if (td48 / "ann_all.csv").exists() else []
    check("T48 校正 id = 校正原本の stem（ZOOM0001）で step19 の json と一致し、点検原本 ZOOM0002 と備考の除外 ZOOM0004 は未注釈扱いにならず、事象 ZOOM0003 だけ切り出される（rc=0）",
          a1.returncode == 0 and a2.returncode == 0 and a3.returncode == 0 and (conv / "calibration_ZOOM0001.json").exists()
          and rows48 and {r["calibration_id"] for r in rows48} == {"ZOOM0001"} and {r["orig_file"] for r in rows48} == {"ZOOM0003"} and rows48[0].get("check_file") == "ZOOM0002.wav" and rows48[0].get("点検方式") == "4方位"
          and set(rows48[0].get("excluded_files", "").split()) == {"ZOOM0004.wav", "ZOOM0002.wav"} and "除外（記録紙" in a3.stdout,
          f"(rc={[a1.returncode, a2.returncode, a3.returncode]}, ids={sorted({r.get('calibration_id') for r in rows48})}, origs={sorted({r.get('orig_file') for r in rows48})}, err={(a3.stdout + a3.stderr)[-300:]!r})")

# ---------------- T49: 区分=歩行対比 は pair_id が無くても 区分=歩行 に正規化し、pair 欠落を警告する（ノート監査 H05） ----------------
sess49 = [{"session_id": "W1", "区分": "歩行対比", "用途": "最終評価", "マイク高さ_cm": "205"}]
ev49 = [{"session_id": "W1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""},
        {"session_id": "W1", "take_id": "2", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "歩行", "pair_id": "P1"}]
rows49, warns49 = s19d.convert(sess49, ev49, 6.0, 0.0, None)
check("T49 歩行対比: 2 行とも 区分=歩行・trial=walk、pair 無しの行だけ警告", len(rows49) == 2 and all(r["区分"] == "歩行" and r["trial"] == "walk" for r in rows49)
      and sum("pair_id が無い" in w for w in warns49) == 1, f"(区分={[r['区分'] for r in rows49]}, warns={warns49})")

# ---------------- T50: 再監査2 残条件1 — 96 kHz 原本・2 セッション（校正が違う）・正例・負例 2 本・統合注釈で、追加引数なしに 変換→校正→切り出し(event/negative --append)→検査 が通る（Q01） ----------------
_td50 = tempfile.mkdtemp(); td50 = Path(_td50); raw50 = td50 / "raw"; raw50.mkdir(); conv50 = td50 / "conv"
import soundfile as _sf50, subprocess as _sp50, csv as _csv50, json as _json50
_fs50 = 96000; _tt50 = np.arange(_fs50 * 11) / _fs50
def _wav50(name, amp, fs=_fs50, tt=_tt50, d=raw50):
    _sf50.write(str(d / name), np.tile((amp * np.sin(2 * np.pi * 1000.0 * tt))[:, None], (1, 4)).astype(np.float32), fs)
for _n, _a in (("S1_calib.wav", 0.01), ("ZOOM0003.wav", 0.01), ("NEG1.wav", 0.01), ("NEG2.wav", 0.01), ("S2_calib.wav", 0.001), ("REC_A.wav", 0.001)):
    _wav50(_n, _a)
with open(td50 / "session.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=["session_id", "区分", "用途", "LAeq_dB", "暗騒音区間_秒", "校正原本", "マイク高さ_cm", "備考"]); w.writeheader()
    w.writerow({"session_id": "S1", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-11", "校正原本": "S1_calib.wav", "マイク高さ_cm": "205", "備考": ""})
    w.writerow({"session_id": "S2", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-11", "校正原本": "S2_calib.wav", "マイク高さ_cm": "205", "備考": "ゲイン変更後は新セッション"})
with open(td50 / "events.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=["session_id", "take_id", "event_id", "原本", "class", "象限", "ラップ秒", "n_car", "横距離m", "状態", "pair_id"]); w.writeheader()
    w.writerow({"session_id": "S1", "take_id": "1", "event_id": "1", "原本": "ZOOM0003.wav", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""})
    w.writerow({"session_id": "S1", "take_id": "2", "event_id": "1", "原本": "NEG1.wav", "class": "none", "象限": "", "ラップ秒": "", "n_car": "0", "横距離m": "", "状態": "静止", "pair_id": ""})
    w.writerow({"session_id": "S1", "take_id": "3", "event_id": "1", "原本": "NEG2.wav", "class": "none", "象限": "", "ラップ秒": "", "n_car": "0", "横距離m": "", "状態": "静止", "pair_id": ""})
    w.writerow({"session_id": "S2", "take_id": "1", "event_id": "1", "原本": "REC_A.wav", "class": "car_drive", "象限": "L", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""})
py = sys.executable; sc = ROOT / "scripts"
def _run50(args):
    return _sp50.run([py] + [str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
b1 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td50 / "session.csv", "--events", td50 / "events.csv", "--out", td50 / "ann_orig.csv", "--audio-dir", raw50])
b2 = _run50([sc / "step19_realsmoke_convert.py", "--in", raw50, "--calib", raw50 / "S1_calib.wav", "--laeq", "52.3", "--gain-only", "--out", conv50])
b3 = _run50([sc / "step19_realsmoke_convert.py", "--in", raw50, "--calib", raw50 / "S2_calib.wav", "--laeq", "52.3", "--gain-only", "--out", conv50])
b4 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw50, "--ann", td50 / "ann_orig.csv", "--calib-dir", conv50, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "clips", "--ann-out", td50 / "ann_all.csv"])
b5 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "negative", "--in", raw50 / "NEG1.wav", "--ann", td50 / "ann_orig.csv", "--calib-dir", conv50, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "clips", "--ann-out", td50 / "ann_all.csv", "--append"])
b6 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "negative", "--in", raw50 / "NEG2.wav", "--ann", td50 / "ann_orig.csv", "--calib-dir", conv50, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "clips", "--ann-out", td50 / "ann_all.csv", "--append"])
b7 = _run50([sc / "step19c_ann_validate.py", "--ann", td50 / "ann_all.csv", "--cut", "--plan", "A=2"])
rows50 = list(_csv50.DictReader(open(td50 / "ann_all.csv", encoding="utf-8-sig"))) if (td50 / "ann_all.csv").exists() else []
_pos50 = [r for r in rows50 if r["class"] != "none" and r.get("scored") == "1"]; _neg50 = [r for r in rows50 if r["class"] == "none"]
_gain50 = {r["orig_file"]: float(r.get("gain_db") or 0) for r in rows50}
_flac50 = sorted(p.name for p in (td50 / "clips").glob("*.flac")) if (td50 / "clips").exists() else []
_sr50 = _sf50.info(str(td50 / "clips" / _flac50[0])).samplerate if _flac50 else 0
check("T50 96 kHz・2 セッション・正例 2・負例 2 本（1 本ずつ --append）を追加引数なし（--gain-db なし・校正 json）で通し、clip_id が session__原本、校正が各セッションの補正量（差 20 dB）、負例 4 クリップ、出力 24 kHz、検査 rc=0",
      all(x.returncode == 0 for x in (b1, b2, b3, b4, b5, b6, b7)) and len(_pos50) == 2 and len(_neg50) == 4
      and {r["clip_id"] for r in _pos50} == {"eval_field_clip01001", "eval_field_clip02001"} and all(r["clip_id"].startswith("eval_field_clip01") and r["clip_id"] not in ("eval_field_clip01001",) for r in _neg50)   # 2026-09-22 命名 A（旧 S1__ZOOM0003_e1 / S2__REC_A_e1・負例 S1__NEG…）
      and {r["calibration_id"] for r in rows50 if r["orig_file"].startswith(("ZOOM", "NEG"))} == {"S1_calib"} and {r["calibration_id"] for r in rows50 if r["orig_file"] == "REC_A"} == {"S2_calib"}
      and abs((_gain50.get("REC_A", 0) - _gain50.get("ZOOM0003", 0)) - 20.0) < 0.3 and len(_flac50) == 6 and _sr50 == 24000,
      f"(rc={[x.returncode for x in (b1, b2, b3, b4, b5, b6, b7)]}, pos={[r['clip_id'] for r in _pos50]}, neg={len(_neg50)}, gains={_gain50}, flac={len(_flac50)}@{_sr50})"
      + ("" if all(x.returncode == 0 for x in (b4, b5, b6, b7)) else f"\n  19b: {b4.stdout[-300:]}{b4.stderr[-300:]}\n  neg1: {b5.stdout[-300:]}{b5.stderr[-300:]}\n  19c: {b7.stdout[-300:]}"))

# ---------------- T51: Q02 — 校正 json の欠落・CLI id の食い違い・記録紙の暗騒音区間と json の窓の不一致は停止 ----------------
_empty51 = td50 / "conv_empty"; _empty51.mkdir()
c1 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw50, "--ann", td50 / "ann_orig.csv", "--calib-dir", _empty51, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "c51a", "--ann-out", td50 / "c51a.csv"])
c2 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw50, "--ann", td50 / "ann_orig.csv", "--calib-dir", td50 / "no_such_dir", "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "c51b", "--ann-out", td50 / "c51b.csv"])
c3 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw50, "--ann", td50 / "ann_orig.csv", "--calib-dir", conv50, "--calibration-id", "S2_calib", "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "c51c", "--ann-out", td50 / "c51c.csv"])
_ann51 = list(_csv50.DictReader(open(td50 / "ann_orig.csv", encoding="utf-8-sig")))
for r in _ann51: r["暗騒音区間_秒"] = "10-70"
with open(td50 / "ann_win.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(_ann51[0].keys())); w.writeheader(); w.writerows(_ann51)
c4 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", raw50, "--ann", td50 / "ann_win.csv", "--calib-dir", conv50, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td50 / "c51d", "--ann-out", td50 / "c51d.csv"])
check("T51 Q02: 空の --calib-dir／無いフォルダ → rc=1（0 dB で続行しない）、--calibration-id が注釈と食い違う → rc=1、記録紙の暗騒音区間 10-70 と json の窓 0-11 が違う → rc=1",
      c1.returncode == 1 and "calibration_*.json がありません" in c1.stdout and c2.returncode == 1
      and c3.returncode != 0 and "違います" in (c3.stdout + c3.stderr) and c4.returncode != 0 and "暗騒音区間" in (c4.stdout + c4.stderr),
      f"(rc={[c1.returncode, c2.returncode, c3.returncode, c4.returncode]}, c3={(c3.stdout + c3.stderr)[-160:]!r}, c4={(c4.stdout + c4.stderr)[-160:]!r})")

# ---------------- T52: Q03 — 別セッションの同名原本（ZOOM0001.wav）は clip_id が session__原本 で区別され、--session なしでは停止、--append で前の行と音声が残る ----------------
td52 = Path(tempfile.mkdtemp()); rawA = td52 / "rawA"; rawB = td52 / "rawB"; rawA.mkdir(); rawB.mkdir()
_fs52 = 24000; _tt52 = np.arange(_fs52 * 12) / _fs52
_wav50("ZOOM0001.wav", 0.01, _fs52, _tt52, rawA); _wav50("ZOOM0001.wav", 0.001, _fs52, _tt52, rawB)
_wav50("S1_calib.wav", 0.01, _fs52, _tt52, rawA); _wav50("S2_calib.wav", 0.001, _fs52, _tt52, rawB)
conv52 = td52 / "conv"
with open(td52 / "session.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=["session_id", "区分", "用途", "LAeq_dB", "暗騒音区間_秒", "校正原本", "マイク高さ_cm"]); w.writeheader()
    w.writerow({"session_id": "S1", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-12", "校正原本": "S1_calib.wav", "マイク高さ_cm": "205"})
    w.writerow({"session_id": "S2", "区分": "A", "用途": "最終評価", "LAeq_dB": "52.3", "暗騒音区間_秒": "0-12", "校正原本": "S2_calib.wav", "マイク高さ_cm": "205"})
with open(td52 / "events.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=["session_id", "take_id", "event_id", "原本", "class", "象限", "ラップ秒", "n_car", "横距離m", "状態", "pair_id"]); w.writeheader()
    w.writerow({"session_id": "S1", "take_id": "1", "event_id": "1", "原本": "ZOOM0001.wav", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""})
    w.writerow({"session_id": "S2", "take_id": "1", "event_id": "1", "原本": "ZOOM0001.wav", "class": "car_drive", "象限": "L", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""})
d1 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td52 / "session.csv", "--events", td52 / "events.csv", "--out", td52 / "ann_orig.csv"])
_run50([sc / "step19_realsmoke_convert.py", "--in", rawA, "--calib", rawA / "S1_calib.wav", "--laeq", "52.3", "--gain-only", "--out", conv52])
_run50([sc / "step19_realsmoke_convert.py", "--in", rawB, "--calib", rawB / "S2_calib.wav", "--laeq", "52.3", "--gain-only", "--out", conv52])
d2 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--in", rawA, "--ann", td52 / "ann_orig.csv", "--calib-dir", conv52, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td52 / "clips", "--ann-out", td52 / "ann_all.csv"])
d3 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--session", "S1", "--in", rawA, "--ann", td52 / "ann_orig.csv", "--calib-dir", conv52, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td52 / "clips", "--ann-out", td52 / "ann_all.csv"])
d4 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--session", "S2", "--in", rawB, "--ann", td52 / "ann_orig.csv", "--calib-dir", conv52, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td52 / "clips", "--ann-out", td52 / "ann_all.csv", "--append"])
rows52 = list(_csv50.DictReader(open(td52 / "ann_all.csv", encoding="utf-8-sig"))) if (td52 / "ann_all.csv").exists() else []
_flac52 = sorted(p.name for p in (td52 / "clips").glob("*.flac")) if (td52 / "clips").exists() else []
check("T52 Q03: 同名原本が 2 セッション → --session なしは rc=1、--session S1 → S2 --append で 2 行（S1__ZOOM0001_e1 / S2__ZOOM0001_e1）と 2 本の音声が両方残り、校正もそれぞれ",
      d1.returncode == 0 and d2.returncode == 1 and "複数セッション" in d2.stdout and d3.returncode == 0 and d4.returncode == 0
      and {r["clip_id"] for r in rows52} == {"eval_field_clip01001", "eval_field_clip02001"} and {r["calibration_id"] for r in rows52} == {"S1_calib", "S2_calib"}
      and _flac52 == ["eval_field_clip01001.flac", "eval_field_clip02001.flac"],   # 2026-09-22 命名 A（旧 S1__ZOOM0001_e1 / S2__ZOOM0001_e1）
      f"(rc={[d1.returncode, d2.returncode, d3.returncode, d4.returncode]}, clips={[r['clip_id'] for r in rows52]}, flac={_flac52}, d2={d2.stdout[-160:]!r})")

# ---------------- T53: Q04 — 歩行対は (pair, class) で対応づけ、同じ状態に 2 件ある対は未集計＋警告（最後の 1 件に置き換えない） ----------------
s20q = _load("s20q", "step20_realsmoke_score_v3.py")
_we = lambda pid, st, cls, fr: {"pair_id": pid, "state": st, "class": cls, "frame_recall": fr, "notified": True}
w53 = [_we("P1", "静止", "car_drive", 1.0), _we("P1", "静止", "horn", 0.0), _we("P1", "歩行", "car_drive", 0.5), _we("P1", "歩行", "horn", 1.0),
       _we("P2", "静止", "car_drive", 0.8), _we("P2", "静止", "car_drive", 0.6), _we("P2", "歩行", "car_drive", 0.7)]
r53, dif53, amb53 = s20q.walk_pairs_summary(w53)
check("T53 Q04: P1 は 車 −0.5 と クラクション +1.0 の 2 対、P2（静止に車 2 件）は未集計として返る",
      [x[0] for x in r53] == ["P1/car_drive", "P1/horn"] and sorted(round(d, 3) for d in dif53) == [-0.5, 1.0] and len(amb53) == 1 and amb53[0][0] == "P2",
      f"(rows={[x[0] for x in r53]}, diffs={dif53}, amb={amb53})")

# ---------------- T54: Q05 — 用途が空欄・未知（「調整」）の行は評価用にも調整用にも入れず保留（rc=1・_unresolved.csv） ----------------
sess54 = [{"session_id": "U1", "区分": "A", "用途": "", "マイク高さ_cm": "205"}, {"session_id": "U2", "区分": "A", "用途": "調整", "マイク高さ_cm": "205"}, {"session_id": "U3", "区分": "A", "用途": "最終評価", "マイク高さ_cm": "205"}]
ev54 = [{"session_id": sid, "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "2.0", "状態": "静止", "pair_id": ""} for sid in ("U1", "U2", "U3")]
rows54, warns54 = s19d.convert(sess54, ev54, 6.0, 0.0, None)
td54 = Path(tempfile.mkdtemp())
with open(td54 / "session.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(sess54[0].keys())); w.writeheader(); w.writerows(sess54)
with open(td54 / "events.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(ev54[0].keys())); w.writeheader(); w.writerows(ev54)
e54 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td54 / "session.csv", "--events", td54 / "events.csv", "--out", td54 / "ann_orig.csv"])
main54 = list(_csv50.DictReader(open(td54 / "ann_orig.csv", encoding="utf-8-sig")))
unres54 = list(_csv50.DictReader(open(td54 / "ann_orig_unresolved.csv", encoding="utf-8-sig"))) if (td54 / "ann_orig_unresolved.csv").exists() else []
check("T54 Q05: 用途 空欄／「調整」は 未確定 として警告、CLI では評価用 CSV に入らず保留ファイルへ（rc=1）。最終評価だけが本体に残る",
      [r["用途"] for r in rows54] == ["未確定", "未確定", "最終評価"] and sum("用途" in w for w in warns54) == 2
      and e54.returncode == 1 and [r["session_id"] for r in main54] == ["U3"] and sorted(r["session_id"] for r in unres54) == ["U1", "U2"],
      f"(用途={[r['用途'] for r in rows54]}, rc={e54.returncode}, main={[r['session_id'] for r in main54]}, unres={[r['session_id'] for r in unres54]})")

# ---------------- T55: Q06 — 警告音の方向は「通知が出た瞬間の推定方位」を鳴り始めの向きと比べる（1 秒の平均ではない） ----------------
_cfg55 = s20q.V43.Cfg43(**_json50.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8")))
_ci55 = s20q.CLS_IDX["horn"]; _fps55 = s20q.FPS
_frames55 = {}
_k0 = int(round(3.0 * _fps55))
for k in range(_k0, _k0 + int(_fps55) + 1):          # 鳴り始め 3.0 s から 1 秒: 最初の 3 フレームは −110°（右）、その後 −170°（後）へ移る
    _frames55[k] = [(_ci55, -110.0 if k < _k0 + 3 else -170.0, 0.0, float("nan"))]
_rows55 = [{"clip_id": "w1", "event_id": "1", "trial": "t1", "class": "horn", "quadrant": "R", "t_start": "3.00", "t_cpa": "6.00", "take_id": "W/01", "pair_id": "", "区分": "D", "状態": "静止", "横距離m": "", "n_car": "1", "scored": "1"}]
_ev55, _, _ = s20q.evaluate(_rows55, {"w1": _frames55}, _cfg55, (2.5, 1.5), 7.5, False, 1.0)
_e55 = _ev55[0]
_med55 = s20q.median_az(_frames55, _ci55, 3.0, 4.0)
check("T55 Q06: 鳴り始め −110°（右）→ 1 秒で −170° に動く音: 1 秒平均は −140°（後）で不一致になるが、通知が出た瞬間の推定は右で一致（quad_ok=True）",
      _e55["notified"] is True and _e55["quad_ok"] is True and abs(_e55["az_est"] + 110.0) < 1.0 and s20q.quadrant_of(_med55) == "B",
      f"(notified={_e55['notified']}, az_est={_e55['az_est']}, quad_ok={_e55['quad_ok']}, 1秒平均={_med55:.0f}°→{s20q.quadrant_of(_med55)})")

# ---------------- T56: R11 — 横距離の幅記入（1.5-2.5）の読み取り・検査（単一値／幅は有効、abc・3-1・負は無効） ----------------
s20v2r = _load("s20v2r", "step20_realsmoke_score_v2.py")
val_r = _load("val_r11", "step19c_ann_validate.py")
_pl = s20v2r.parse_lateral
check("T56 R11: parse_lateral が 2.0→(2,2)、1.5-2.5→(1.5,2.5)、全角「１.５〜２.５」→(1.5,2.5)、abc/3-1/-1 は None。step19c の lateral_of も幅を有効とし abc を無効とする",
      _pl("2.0") == (2.0, 2.0) and _pl("1.5-2.5") == (1.5, 2.5) and _pl("１.５〜２.５") == (1.5, 2.5) and _pl("abc") is None and _pl("3-1") is None and _pl("-1") is None
      and val_r.lateral_of({"横距離m": "1.5-2.5"}) == 1.5 and val_r.lateral_of({"横距離m": "abc"}) is None and val_r.lateral_of({"横距離m": "3-1"}) is None
      and s20v2r.lateral_of({"横距離m": "1.5-2.5"}) is None and s20v2r.lateral_of({"横距離m": "2.0"}) == 2.0,
      f"(parse={[_pl(x) for x in ('2.0', '1.5-2.5', '１.５〜２.５', 'abc', '3-1', '-1')]})")
_r56, _w56 = s19d.convert([{"session_id": "R1", "区分": "A", "用途": "最終評価", "マイク高さ_cm": "205"}],
                          [{"session_id": "R1", "take_id": "1", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "1.5-2.5", "状態": "静止", "pair_id": ""},
                           {"session_id": "R1", "take_id": "2", "event_id": "1", "class": "car_drive", "象限": "R", "ラップ秒": "8.0", "n_car": "1", "横距離m": "3-1", "状態": "静止", "pair_id": ""}], 6.0, 0.0, None)
check("T56b R11: step19d は幅をそのまま通し（1.5-2.5）、3-1 のような逆順は警告する", _r56[0]["横距離m"] == "1.5-2.5" and sum("横距離m" in w for w in _w56) == 1, f"(warns={_w56})")

# ---------------- T57: R11 — 採点 v3: 幅が区分の境界（1.5 / 3.2 m）をまたぐ行は「境界例」として主要評価から外し別掲、またがない幅は通常どおり区分に入る ----------------
_ci57 = s20q.CLS_IDX["car_drive"]; _fps57 = s20q.FPS
def _frames57(d):
    return {k: [(_ci57, -90.0, 0.0, d)] for k in range(int(4 * _fps57), int(8.5 * _fps57))}
_rows57 = [{"clip_id": "b1", "event_id": "1", "trial": "t1", "class": "car_drive", "quadrant": "R", "t_start": "2.00", "t_cpa": "8.00", "take_id": "B/01", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "1.0-2.0", "n_car": "1", "scored": "1"},
           {"clip_id": "b2", "event_id": "1", "trial": "t1", "class": "car_drive", "quadrant": "R", "t_start": "2.00", "t_cpa": "8.00", "take_id": "B/02", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "2.0-3.0", "n_car": "1", "scored": "1"},
           {"clip_id": "b3", "event_id": "1", "trial": "t1", "class": "car_drive", "quadrant": "R", "t_start": "2.00", "t_cpa": "8.00", "take_id": "B/03", "pair_id": "", "区分": "A", "状態": "静止", "横距離m": "3.0-4.0", "n_car": "1", "scored": "1"}]
_ev57, _, _ex57 = s20q.evaluate(_rows57, {"b1": _frames57(1.4), "b2": _frames57(2.5), "b3": _frames57(3.5)}, _cfg55, (2.5, 1.5), 7.5, False, 1.0)
_main57, _side57, _ = s20q.summarize(_ev57, False)
_by57 = {e["clip"]: e for e in _ev57}
check("T57 R11: 1.0-2.0（critical/caution をまたぐ）と 3.0-4.0（caution/safe をまたぐ）は境界例として主要評価から外れ、2.0-3.0 は caution として主要評価に入る",
      _by57["b1"]["gt_tier"] == "boundary:critical/caution" and _by57["b3"]["gt_tier"] == "boundary:caution/safe" and _by57["b2"]["gt_tier"] == "caution"
      and [e["clip"] for e in _main57] == ["b2"] and sorted(e["clip"] for e in _side57["boundary"]) == ["b1", "b3"] and _ex57["n_boundary"] == 2
      and _by57["b1"]["notified"] in (True, False),
      f"(tiers={[(c, _by57[c]['gt_tier'], _by57[c]['notified']) for c in ('b1', 'b2', 'b3')]}, main={[e['clip'] for e in _main57]})")

# ---------------- T58: 負例の本（G・負例露出）に偶然入った相手の行は「クラスと秒だけ」= S3/S5 の対象外（本人 10/8 13:43・#47 A-3・Opus 監査 J2） ----------------
val_j2 = _load("val_j2", "step19c_ann_validate.py")
def _run19c(rows):
    val_j2.errs.clear(); val_j2.warns.clear()
    val_j2.validate(rows, False, 10.0, dict(val_j2.PLAN_DEFAULT))
    return list(val_j2.errs)
_base58 = {"clip_id": "n1", "event_id": "1", "trial": "", "class": "car_drive", "quadrant": "", "t_start": "2.00", "t_cpa": "5.00",
           "take_id": "N/01", "区分": "負例露出", "状態": "静止", "pair_id": "", "横距離m": ""}
_e58a = _run19c([_base58, dict(_base58, event_id="2", class_="none")] if False else [_base58])
_e58b = _run19c([dict(_base58, **{"区分": "A"})])
check("T58 J2: 区分=負例露出の相手の行（象限・横距離が空）は S3/S5 で止まらず、同じ行を区分 A にすると S3 と S5 で止まる",
      not any(m.startswith("S3") or m.startswith("S5") for m in _e58a) and any(m.startswith("S3") for m in _e58b) and any(m.startswith("S5") for m in _e58b),
      f"(負例露出={_e58a}, A={_e58b})")
_e58g = _run19c([dict(_base58, **{"区分": "G"})])
check("T58b J2: 区分=G の相手の行も S3/S5 の対象外", not any(m.startswith("S3") or m.startswith("S5") for m in _e58g), f"({_e58g})")

# ---------------- T59〜T62: 変更票 #59 r6・Astra FR11-01／FR11-08（2026-10-12）: 時刻が未確定の正例で止める・読む準備の表を注釈と切り離す・拍手の自動検出 ----------------
s19d_59 = _load("s19d59", "step19d_field_csv_to_ann.py")
_sess59 = [{"session_id": "S9", "区分": "H", "用途": "最終評価", "LAeq_dB": "50.0", "暗騒音区間_秒": "0-60", "マイク高さ_cm": "150"}]
_ev59 = [{"session_id": "S9", "take_id": "1", "event_id": "1", "原本": "261015_003.WAV", "class": "bicycle", "象限": "B", "ラップ秒": "", "n_car": "", "横距離m": "2.0", "状態": "静止", "pair_id": "", "動画原本": "IMG_0001.MOV", "direction": "後→前", "line_m": "2.0", "拍手_録音側秒": ""},
         {"session_id": "S9", "take_id": "2", "event_id": "1", "原本": "261015_004.WAV", "class": "bike_bell", "象限": "F", "ラップ秒": "", "n_car": "", "横距離m": "", "状態": "静止", "pair_id": "", "動画原本": "IMG_0002.MOV", "direction": "前→後", "line_m": "1.0", "拍手_録音側秒": "2.5"},
         {"session_id": "S9", "take_id": "3", "event_id": "1", "原本": "261015_005.WAV", "class": "bicycle", "象限": "B", "ラップ秒": "12.34", "n_car": "", "横距離m": "2.0", "状態": "静止", "pair_id": "", "動画原本": "IMG_0003.MOV", "direction": "後→前", "line_m": "2.0", "拍手_録音側秒": ""}]
rows59, warns59 = s19d_59.convert(_sess59, _ev59, 6.0, 0.0, None)
_r1 = next(r for r in rows59 if r["take_id"].endswith("/01")); _r2 = next(r for r in rows59 if r["take_id"].endswith("/02")); _r3 = next(r for r in rows59 if r["take_id"].endswith("/03"))
check("T59 #59 r6: ラップ秒が空の正例は t_start/t_cpa を空（0 にしない）で返し、規則に『未確定』・警告にも出る。埋まった本は今までどおり",
      _r1["t_start"] == "" and _r1["t_cpa"] == "" and "未確定" in _r1["t_start_rule"] and _r2["t_start"] == "" and _r2["t_cpa"] == ""
      and _r3["t_cpa"] == "12.34" and _r3["t_start"] == "6.34" and sum("未確定" in w for w in warns59) == 2,
      f"(r1={_r1['t_start']!r}/{_r1['t_cpa']!r}, r3={_r3['t_cpa']}, warns={warns59})")
td59 = Path(tempfile.mkdtemp())
with open(td59 / "session.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(_sess59[0].keys())); w.writeheader(); w.writerows(_sess59)
with open(td59 / "events.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(_ev59[0].keys())); w.writeheader(); w.writerows(_ev59)
(td59 / "clap_audio.csv").write_text("file,clap_audio_s,peak_dbfs,snr_db,quality,pick,n_candidates,candidates,lap_s,delta_s\n"
                                     "261015_003.WAV,3.456,-12.0,55.0,OK,first,2,3.456:55.0;9.0:40.0,,\n"
                                     "261015_004.WAV,2.600,-12.0,55.0,OK,first,1,2.600:55.0,,\n"
                                     "261015_005.WAV,4.100,-20.0,45.0,確認（近くに同じ強さの音: 4.80s）,first,2,4.100:45.0;4.800:44.0,,\n", encoding="utf-8")
e1 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td59 / "session.csv", "--events", td59 / "events.csv", "--out", td59 / "ann_orig.csv"])
e2 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td59 / "session.csv", "--events", td59 / "events.csv", "--out", td59 / "ann_orig.csv", "--video-only", "--clap-csv", td59 / "clap_audio.csv"])
vt59 = list(_csv50.DictReader(open(td59 / "ann_orig_video_takes.csv", encoding="utf-8-sig"))) if (td59 / "ann_orig_video_takes.csv").exists() else []
_vt = {r["take"]: r for r in vt59}
check("T59b #59 r6／NT12-01: 時刻が未確定の正例があると step19d は注釈を書かずに rc=2。--video-only は rc=0 で読む準備の表だけ書き、--clap-csv の値は候補の列に写すだけ（clap_audio_s は本人の値だけ・本人の値と 10 ms 超違えば注意）",
      e1.returncode == 2 and not (td59 / "ann_orig.csv").exists() and "未確定" in e1.stdout
      and e2.returncode == 0 and not (td59 / "ann_orig.csv").exists() and len(vt59) == 3
      and _vt.get("261015_003.WAV", {}).get("clap_audio_s") == "" and _vt.get("261015_003.WAV", {}).get("clap_candidate_s") == "3.456" and _vt.get("261015_003.WAV", {}).get("clap_source") == ""
      and _vt.get("261015_004.WAV", {}).get("clap_audio_s") == "2.5" and _vt.get("261015_004.WAV", {}).get("clap_source") == "events" and "261015_004" in e2.stdout and "違う" in e2.stdout
      and _vt.get("261015_005.WAV", {}).get("clap_audio_s") == "" and "確認" in _vt.get("261015_005.WAV", {}).get("clap_candidate_quality", ""),
      f"(rc={e1.returncode}/{e2.returncode}, vt={[(k, v['clap_audio_s'], v['clap_candidate_s'], v['clap_source'], v['clap_candidate_quality']) for k, v in _vt.items()]}, out1={e1.stdout[-300:]!r}, out2={e2.stdout[-400:]!r})")
# NT12-02: nan／inf／0／負のラップ秒は未確定（step19d は t を空に・CLI は rc=2）
_ev59c = [dict(_ev59[2], take_id=str(k + 1), ラップ秒=v) for k, v in enumerate(("nan", "inf", "0", "-1", "12.34"))]
rows59c, _ = s19d_59.convert(_sess59, _ev59c, 6.0, 0.0, None)
_tc59c = [r["t_cpa"] for r in sorted(rows59c, key=lambda r: r["take_id"])]
check("T59c NT12-02: ラップ秒 nan／inf／0／−1 は未確定（t_cpa 空）、12.34 だけ確定", _tc59c == ["", "", "", "", "12.34"], f"({_tc59c})")
cut59 = _load("s19bcut59", "step19b_realsmoke_cut.py")
_bad60 = cut59.unfilled_positive_rows([
    {"take_id": "S9/01", "event_id": "1", "class": "bicycle", "t_start": "", "t_cpa": ""},
    {"take_id": "S9/02", "event_id": "1", "class": "bicycle", "t_start": "0.00", "t_cpa": "0.00"},
    {"take_id": "S9/03", "event_id": "1", "class": "bike_bell", "t_start": "0.00", "t_cpa": "3.00"},
    {"take_id": "S9/04", "event_id": "1", "class": "bike_bell", "t_start": "12.10", "t_cpa": "15.10"},
    {"take_id": "S9/05", "event_id": "1", "class": "car_drive", "t_start": "2.40", "t_cpa": "8.40"},
    {"take_id": "S9/06", "event_id": "1", "class": "none", "t_start": "0", "t_cpa": ""},
    {"take_id": "S9/07", "event_id": "1", "class": "bicycle", "t_start": "nan", "t_cpa": "nan"},
    {"take_id": "S9/08", "event_id": "1", "class": "bicycle", "t_start": "inf", "t_cpa": "inf"},
    {"take_id": "S9/09", "event_id": "1", "class": "bicycle", "t_start": "-7.0", "t_cpa": "-1.0"}])
check("T60 #59 r6（FR11-01）／NT12-02: step19b は t_cpa が空・0・nan・inf・負、警告音の t_start が 0 の正例を『未確定』と判定し、埋まった正例と負例は通す",
      len(_bad60) == 6 and all(k in " ".join(_bad60) for k in ("S9/01", "S9/02", "S9/03", "S9/07", "S9/08", "S9/09")), f"({_bad60})")
_rows60 = list(_csv50.DictReader(open(td52 / "ann_orig.csv", encoding="utf-8-sig")))
_rows60[0]["t_start"] = "0.00"; _rows60[0]["t_cpa"] = "0.00"
with open(td52 / "ann_zero.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(_rows60[0].keys())); w.writeheader(); w.writerows(_rows60)
e60 = _run50([sc / "step19b_realsmoke_cut.py", "--mode", "event", "--session", "S1", "--in", rawA, "--ann", td52 / "ann_zero.csv", "--calib-dir", conv52, "--pitch", "0", "--roll", "0", "--yaw", "0", "--out", td52 / "clips_zero", "--ann-out", td52 / "ann_zero_all.csv"])
check("T60b #59 r6: step19b の実行は t_cpa=0 の正例で rc=1 で止まり、クリップを切らない（FR11-01 の再現が通らない）",
      e60.returncode == 1 and "未確定" in e60.stdout and not (td52 / "clips_zero").exists(), f"(rc={e60.returncode}, out={e60.stdout[-300:]!r})")
_base61 = {"clip_id": "S9__261015_003", "event_id": "1", "trial": "H", "class": "bicycle", "quadrant": "B", "t_start": "0.00", "t_cpa": "0.00", "take_id": "S9/01", "区分": "H", "状態": "静止", "pair_id": "", "横距離m": "2.0"}
_e61a = _run19c([_base61]); _e61b = _run19c([dict(_base61, t_start="2.34", t_cpa="8.34")]); _e61c = _run19c([dict(_base61, **{"class": "bike_bell", "t_start": "0.00", "t_cpa": "3.00", "横距離m": ""})])
check("T61 #59 r6: step19c は t_cpa=0 の正例と t_start=0 の警告音を S4 で止め、埋まった正例は通す",
      any(m.startswith("S4") and "0 です" in m for m in _e61a) and not any(m.startswith("S4") for m in _e61b) and any(m.startswith("S4") and "0 です" in m for m in _e61c),
      f"(a={_e61a}, b={_e61b}, c={_e61c})")
_j62 = _run50([sc / "step19j_clap_from_audio.py", "--selftest", td59 / "j"])
_h62 = _run50([sc / "step19h_video_marks.py", "--selftest", td59 / "h"])
check("T62 step19j（拍手の自動検出: first・REC 音を飛ばす・ベルに引かれない・二重は確認）と step19h（--export・FR11-08 の案内）の自己検査が通る",
      _j62.returncode == 0 and "ALL PASS" in _j62.stdout and _h62.returncode == 0 and "ALL PASS" in _h62.stdout,
      f"(j={_j62.stdout[-500:]!r}, h={(_h62.stdout + _h62.stderr)[-500:]!r})")

# ---------------- T63〜T64: 変更票 #59 今の決まり（2026-10-12）: 動画の読み → 「いつ」「どこ」（step19l）と step19d --when-csv ----------------
_l63 = _run50([sc / "step19l_when_where.py", "--selftest", td59 / "l"])
check("T63 step19l（中点の交わりで「いつ」・欠測 4 種・長さの欠けと範囲外で止める・2 回目の変換を止める・変換済み reads を step19f が読める）の自己検査が通る",
      _l63.returncode == 0 and "ALL PASS" in _l63.stdout, f"(l={(_l63.stdout + _l63.stderr)[-600:]!r})")
_when64 = {("261015_003", "1"): {"take": "261015_003.WAV", "track": "1", "status": "ok", "t_cpa_s": "15.123", "sigma": "1", "L_mid_m": "0.520", "dt_est_s": "0.033", "mic": "静止（x = 0）", "reason": ""},
           ("261015_005", "1"): {"take": "261015_005.WAV", "track": "1", "status": "欠測", "t_cpa_s": "", "reason": "交わらない（…）"}}
_ev64 = [dict(_ev59[0]), dict(_ev59[1]), dict(_ev59[2], ラップ秒="")]            # 1 = bicycle 空 → when で埋まる・2 = bike_bell 空（波形待ち）・3 = bicycle 空 → when は欠測
rows64, warns64 = s19d_59.convert(_sess59, _ev64, 6.0, 0.0, None, when=_when64)
_a = next(r for r in rows64 if r["take_id"].endswith("/01")); _b = next(r for r in rows64 if r["take_id"].endswith("/02")); _c = next(r for r in rows64 if r["take_id"].endswith("/03"))
check("T64 step19d --when-csv: ラップ秒が空の距離クラスは when の t_cpa_s（録音の秒・拍手のずれを足さない）で埋まり規則に『動画の目印の読みの交わり』、欠測の本と警告音は未確定のまま",
      _a["t_cpa"] == "15.12" and _a["t_start"] == "9.12" and "動画の目印の読みの交わり" in _a["t_start_rule"] and "L_mid 0.520" in _a["t_start_rule"]
      and _b["t_cpa"] == "" and _c["t_cpa"] == "" and any("欠測" in w and "take03" in w for w in warns64) and sum("ラップ秒が空" in w for w in warns64) == 2,
      f"(a={_a['t_cpa']}/{_a['t_start']}/{_a['t_start_rule'][:60]!r}, b={_b['t_cpa']!r}, c={_c['t_cpa']!r}, warns={warns64})")
_ev64b = [dict(_ev59[0]), dict(_ev59[2])]                                         # 空の bicycle 1 本（when で埋まる）＋ 埋まった bicycle → 全部確定 → rc 0 で注釈が書ける
with open(td59 / "events64.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv50.DictWriter(f, fieldnames=list(_ev64b[0].keys())); w.writeheader(); w.writerows(_ev64b)
with open(td59 / "when64.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["take", "session_id", "cls", "track", "sigma", "L_mid_m", "L_source", "n_marks", "n_mic_marks", "mic", "t_cpa_s", "status", "reason", "x_cross_m", "rel_speed_mps", "dt_est_s", "t_first_s", "t_last_s"]
    w = _csv50.DictWriter(f, fieldnames=cols); w.writeheader()
    w.writerow({"take": "261015_003.WAV", "session_id": "S9", "cls": "bicycle", "track": "1", "sigma": "1", "L_mid_m": "0.520", "L_source": "S9 備考「自転車の長さ: 軸の間 104 cm」", "n_marks": "11", "n_mic_marks": "0",
                "mic": "静止（x = 0）", "t_cpa_s": "15.123", "status": "ok", "reason": "", "x_cross_m": "0.520", "rel_speed_mps": "3.0", "dt_est_s": "0.033", "t_first_s": "10.0", "t_last_s": "16.7"})
e64 = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td59 / "session.csv", "--events", td59 / "events64.csv", "--out", td59 / "ann64.csv", "--when-csv", td59 / "when64.csv"])
_ann64 = list(_csv50.DictReader(open(td59 / "ann64.csv", encoding="utf-8-sig"))) if (td59 / "ann64.csv").exists() else []
e64m = _run50([sc / "step19d_field_csv_to_ann.py", "--session", td59 / "session.csv", "--events", td59 / "events64.csv", "--out", td59 / "ann64m.csv", "--when-csv", td59 / "no_such_when.csv"])
check("T64b step19d の実行: --when-csv で全部の正例が埋まれば rc 0 で注釈を書く（t_cpa 15.12）・無いファイルは rc 2",
      e64.returncode == 0 and len(_ann64) == 2 and {r["t_cpa"] for r in _ann64} == {"15.12", "12.34"} and e64m.returncode == 2,
      f"(rc={e64.returncode}/{e64m.returncode}, ann={[ (r['take_id'], r['t_cpa']) for r in _ann64]}, out={e64.stdout[-300:]!r})")

# ---------------- T65〜T66: 変更票 #57（来る方向の内訳の補助の表）・学習の入口の門（引き継ぎ §3-8）（2026-10-12） ----------------
_ev65 = lambda q, cls, ok, tier="critical", lead=4.0, fr=0.8: {"quadrant": q, "class": cls, "notified": ok, "gt_tier": tier, "quad_ok": ok, "lead": lead, "delay": None, "frame_recall": fr}
_main65 = [_ev65("F", "car_drive", True), _ev65("F", "car_drive", False), _ev65("B", "car_drive", True), _ev65("B", "bicycle", True, "caution"), _ev65("", "horn", True, "warn", None)]
_L65 = "\n".join(s20q.quadrant_breakdown(_main65, [_ev65("B", "car_drive", True)]))
check("T65 #57: 象限の内訳の表（F 1/2・B 2/2・記入なし 1・全部 4/5・クラス×象限・歩く比較は参考・検定しない文）が出る。主要評価の集計（summarize）は変えていない",
      "| F（前→後） | 2 | 1/2（50%） |" in _L65 and "| B（後→前） | 2 | 2/2（100%） |" in _L65 and "| 象限の記入なし | 1 |" in _L65 and "| 全部（主要評価と同じ） | 5 | 4/5（80%） |" in _L65
      and "| car_drive | 1/2 | 1/1 | — | — |" in _L65 and "歩く比較（参考" in _L65 and "検定しない" in _L65 and "critical 1/1、caution 1/1" in _L65
      and s20q.summarize.__doc__.startswith("主要評価"), f"({_L65[:400]!r})")
_g66 = _run50([ROOT / "server_sde" / "_train_entrance_guard.py", "--selftest", td59 / "guard"])
check("T66 学習の入口の門: eval_syn_／旧 fold20／eval_field_／raw の置き場があれば止め、学習・検証の本だけなら通す（自己検査 7 件・AR12-01 の HDF5 無しと AR12-02 の symlink（Windows では飛ばす）込み）",
      _g66.returncode == 0 and "ALL PASS" in _g66.stdout and "FAIL " not in _g66.stdout, f"(g={(_g66.stdout + _g66.stderr)[-500:]!r})")

if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print(f"ALL PASS ({n_checks} checks)")
sys.exit(0)
