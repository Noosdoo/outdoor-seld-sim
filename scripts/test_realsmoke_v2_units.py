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
T36: 計画120テイク＋別枠負例100分が最終ゲートを通る
T37: C負例の時間を別枠100分へ流用できない
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

# ---------------- T36: 完全な120テイク＋負例100分はstrict相当で合格 ----------------
rows36 = []
for kind in "ABCDE":
    for i in range(20):
        common = {"clip_id": f"{kind}{i}", "event_id": "1", "trial": kind,
                  "take_id": f"{kind}-take-{i}", "pair_id": "", "区分": kind,
                  "状態": "静止", "orig_file": f"{kind}{i}",
                  "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1"}
        if kind == "C":
            rows36.append({**common, "class": "none", "quadrant": "",
                           "t_start": "0", "t_cpa": "10"})
        else:
            rows36.append({**common, "class": "car_drive", "quadrant": "L",
                           "t_start": "1", "t_cpa": "8", "横距離m": "3"})
for i in range(20):
    rows36.append({"clip_id": f"walk{i}", "event_id": "1", "trial": "walk",
                   "class": "car_drive", "quadrant": "L", "t_start": "1",
                   "t_cpa": "8", "横距離m": "3", "take_id": f"walk-take-{i}",
                   "pair_id": f"walk-pair-{i % 10}", "区分": "歩行",
                   "状態": "静止" if i < 10 else "歩行", "orig_file": f"walk{i}",
                   "orig_duration_s": "10", "cut_offset_s": "0", "scored": "1"})
for i, (off, a, b) in enumerate(cut.plan_negative_split(6000.0)):
    rows36.append(cut.negative_row(
        f"exposure_s{i:04d}", "exposure", off, a, b,
        {"take_id": "exposure-take", "pair_id": "", "区分": "負例露出", "状態": "静止"},
        orig_duration_s=6000.0))
val.errs.clear()
val.warns.clear()
import contextlib as _contextlib
import io as _io
with _contextlib.redirect_stdout(_io.StringIO()):
    val.validate(rows36, True, 10.0, dict(val.PLAN_DEFAULT))
check("T36 計画120テイク＋別枠負例100分がstrict相当で合格",
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
if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print(f"ALL PASS ({n_checks} checks)")
sys.exit(0)
