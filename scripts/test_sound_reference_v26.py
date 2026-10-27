# -*- coding: utf-8 -*-
"""音の検査・第 1 版（2026-10-12・Fable・依頼文 r5 §6「再発を防ぐ決まり」・Astra AS10「『音の検査』の初版の案」）。
合成の音そのものを、出典の値（md/design/sound_reference_v26.json・事実／製品の公称値／法規／仮定の印つき）と照らす。合成を直すたびに通す。
合格の線は出典から先に決めてある（出力に合わせて後から付けない）。性能の良し悪しで物理の合格線を動かさない。

段階 1（ドライ音・発音の列・スペクトル）: ベルの基音と 1 打・クラクションの押しの列と連続音と重心・バック音の速さと割合と高さとチャイム・
          サイレンの基音と周期と形とうなり・通報音の形・大型車の低域とタイヤ・自転車の帯の形。
段階 2（標準の置き方で描いたあと・直接音＋地面の反射・24 kHz）: 車の曲線（2 m の LAFmax）・サイレンの 20 m の前の値と向きの差・警告音の 1 m の LAFmax 合わせ・
          EV＋通報音の R138 の総合と帯（条件つきの概算）。
使い方: python scripts/test_sound_reference_v26.py [--quick]（--quick は段階 2 を飛ばす）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v26_render as v26  # noqa: E402
from outdoor_seld import alert_sounds_v23 as a23, level_v24 as L24, engine_heavy_v25 as H25, siren_v26 as S26, avas_v26 as A26  # noqa: E402
from outdoor_seld.alert_sounds import make_bike_bell, make_bike_bell_ring  # noqa: E402
from outdoor_seld.bike_bell_v2 import make_bike_bell_v2  # noqa: E402
from outdoor_seld.bicycle_v2 import make_bicycle_v2  # noqa: E402
from outdoor_seld.engine import make_car_v9  # noqa: E402
from outdoor_seld.engine_ev import make_car_ev  # noqa: E402
from outdoor_seld.calibration import gain_for_spl_a, spl_a, a_weighted_rms, a_weight_amplitude, K_RMS_SPL  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.scene import decimate_to_out_rate  # noqa: E402
from outdoor_seld import ablate  # noqa: E402

v25, v24, m9 = v26.v25, v26.v24, v26.m9
FS, FS_OUT, CLIP = 48000, 24000, 10.0
REF = json.loads((ROOT / "md/design/sound_reference_v26.json").read_text(encoding="utf-8"))
G = L24.MATCH_GEOMETRY
results = []


def check(rid: str, cond: bool, note: str = "") -> None:
    ref = next((r for r in REF["references"] if r["id"] == rid), None)
    kind = ref["kind"] if ref else "?"
    results.append((rid, bool(cond)))
    print(("PASS " if cond else "FAIL ") + f"{rid}（{kind}）{note}")


def peak_hz(x, fs, lo, hi):
    X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1.0 / fs); m = (f >= lo) & (f <= hi)
    return float(f[m][int(np.argmax(X[m]))])


NOMINAL_THIRD_OCTAVE = {20: 100, 21: 125, 22: 160, 23: 200, 24: 250, 25: 315, 26: 400, 27: 500, 28: 630, 29: 800, 30: 1000, 31: 1250, 32: 1600, 33: 2000, 34: 2500,
                        35: 3150, 36: 4000, 37: 5000, 38: 6300, 39: 8000, 40: 10000}   # IEC 61260 の帯の番号 → 名目中心（R138 の表と同じ名前で引けるように・Astra SV26-02）


def third_octave_levels(x, fs, f_lo=100.0, f_hi=10000.0):
    """A 特性なし・1/3 オクターブ帯のレベル（本プロジェクトの換算）。戻り = {名目中心（IEC 61260・例 500／1600／2000）: dB}。
    以前は厳密中心を丸めた 501／1585／1995 をキーにしていて、R138 の表（名目中心）と一致せず帯域の条件が常に 0/12 のまま PASS になっていた（Astra SV26-02）。"""
    X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1.0 / fs); n = len(x)
    out = {}
    k = int(np.round(10 * np.log10(f_lo / 1000.0) + 30))
    while True:
        fc = 1000.0 * 10 ** ((k - 30) / 10.0)
        if fc > f_hi:
            break
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        p = 2.0 * np.sum(X[m]) / (n * n)
        out[NOMINAL_THIRD_OCTAVE.get(k, int(np.round(fc)))] = 10 * np.log10(max(p, 1e-30)) + K_RMS_SPL
        k += 1
    return out


def render_w(dry_cal, v_mps, lateral=G["lateral_m"], mic_z=G["mic_z_m"], src_z=G["src_z_m"], wp=None, mic=None):
    t_cpa = G["cpa_s"]
    wp = np.array([[t, (t - t_cpa) * v_mps, lateral, src_z] for t in (0.0, CLIP)]) if wp is None else np.array(wp, float)
    mic = np.array([0.0, 0.0, mic_z]) if mic is None else np.asarray(mic, float)
    total = None
    for tag in (("direct", "mirror") if ablate.ground_enabled() else ("direct",)):
        w = wp.copy()
        if tag == "mirror":
            w[:, 3] = -w[:, 3]
        mono = decimate_to_out_rate(render_mono(dry_cal, w, mic, FS, CLIP, temperature_c=20.0, pressure_atm=1.0, rel_humidity=50.0, **ablate.render_flags()), FS, FS_OUT)
        total = mono if total is None else total + mono
    return total


def stage1() -> None:
    rng = np.random.default_rng(2026)
    # ---- ベル
    bad = 0; n = 0
    for kind, maker in (("ding", lambda f0: make_bike_bell(2.0, FS, f0=f0, n_rings=1, repeat_period_sec=10.0)),
                        ("ring", lambda f0: make_bike_bell_ring(2.0, FS, f0=f0)),
                        ("pin", lambda f0: make_bike_bell_v2(2.0, FS, f0=f0, repeat_period_sec=10.0, rng=np.random.default_rng(1)))):
        lo, hi = a23.bell_f0_range(kind)
        for _ in range(15):
            f0, sid = a23.bell_f0_v23(kind, rng); x = maker(f0); pk = peak_hz(x, FS, 1500.0, 4000.0); n += 1
            if not (lo <= f0 <= hi and abs(pk / f0 - 1.0) <= 0.02):
                bad += 1
    check("BELL_F0_JBPI", bad == 0, f": 3 型 × 15 本で f0 が種類の幅の中・一番強い山が f0 ±2%（違反 {bad}/{n}）")
    x = make_bike_bell(2.0, FS, f0=2657.0, n_rings=1, repeat_period_sec=10.0)
    segs = a23.sounding_segments(x[: int(1.5 * FS)], FS, rel_db=-25.0, min_gap_sec=0.1)
    check("BELL_ONE_STRIKE", len(segs) == 1, f": ディンディン型の 1.5 s に打音の区間 {len(segs)}")
    # ---- クラクション
    cnt = {}; bad = 0
    for _ in range(20000):
        pl = a23.horn_press_plan(rng); cnt[pl["n"]] = cnt.get(pl["n"], 0) + 1
        lo, hi = a23.HORN_PRESS_SHORT if pl["kind"] == "short" else a23.HORN_PRESS_LONG
        bad += int(not all(lo - 1e-9 <= L <= hi + 1e-9 for _, L in pl["presses"]))
    fr = {k: v / 20000 for k, v in cnt.items()}
    check("HORN_PRESS_TYPES", abs(fr[1] - 0.562) <= 0.015 and abs(fr[2] - 0.363) <= 0.015 and abs(fr[3] - 0.074) <= 0.015, f": 1 回 {fr[1]:.3f}・2 回 {fr[2]:.3f}・3 回 {fr[3]:.3f}")
    bad2 = 0; cents = []; gaps_ok = True
    for _ in range(12):
        pl = a23.horn_press_plan(rng); hp = a23.horn_params_v23(rng)
        presses = [(1.0 + a, L) for a, L in pl["presses"]]
        x = a23.make_horn_v23(CLIP, FS, np.random.default_rng(3), presses, f_lo=hp["f_lo"], f_hi=hp["f_hi"] if hp["dual"] else None, alpha=hp["alpha"], onset_sec=hp["onset_sec"], release_tau_sec=hp["release_tau_sec"])
        segs = a23.sounding_segments(x, FS)
        if len(segs) != len(presses) or any(abs(s0 - p0) > 0.01 for (s0, _), (p0, _) in zip(segs, presses)):
            bad2 += 1
        for s0, s1 in segs:                                      # 押しの中に切れ目が無い（−30 dB で区間が割れない = 上の数の一致で担保）
            pass
        cents.append(a23.spectral_centroid_hz(x[int(presses[0][0] * FS): int((presses[0][0] + presses[0][1]) * FS)], FS))
    check("HORN_PRESS_LEN", bad == 0 and bad2 == 0, f": 20,000 回の長さが幅の中（違反 {bad}）・12 本の鳴る区間 = 押しの数・開始 ±10 ms（違反 {bad2}）")
    check("HORN_CONTINUOUS", bad2 == 0, ": 押しの中に −30 dB の切れ目が無い（区間の数が押しの数と同じ）")
    check("HORN_CENTROID", all(1380 <= c <= 3790 for c in cents), f": 12 本の重心 {min(cents):.0f}〜{max(cents):.0f} Hz")
    # ---- バック音（鳴る割合はクリップの端で切れた音を除いて測る・2026-10-12 の直し）
    bad_rate = 0; bad_freq = 0; n = 0
    for _ in range(30):
        p = a23.backup_params_v23(rng); x = a23.make_backup_beep_v23(CLIP, FS, 0.4, p); n += 1
        if p["kind"] == "chime":
            segs = a23.sounding_segments(x, FS, rel_db=-25.0, min_gap_sec=0.02); starts = [s for s, _ in segs]
            per = float(np.median(np.diff(starts))) if len(starts) >= 3 else 0.0
            if abs(per - 1.25 * p["time_mult"]) > 0.01:
                bad_rate += 1
        else:
            segs = [sg for sg in a23.sounding_segments(x, FS, rel_db=-20.0) if sg[1] < CLIP - 0.005]; starts = [s for s, _ in segs]
            rate = 60.0 / float(np.median(np.diff(starts))); duty = float(np.mean([e - s for s, e in segs]) / (60.0 / p["rate_per_min"]))
            f = peak_hz(x, FS, 500.0, 4000.0)
            lo, hi = a23.BACKUP_FREQ[p["kind"]]
            if abs(rate - p["rate_per_min"]) > 1.0 or abs(duty - p["duty"]) > 0.03 or not (60 <= p["rate_per_min"] <= 100) or not (0.4 <= p["duty"] <= 0.6):
                bad_rate += 1
            if abs(f / p["freq"] - 1) > 0.01 or not (lo <= p["freq"] <= hi):
                bad_freq += 1
    check("BACKUP_RATE", bad_rate == 0, f": 30 本（3 型）で速さ・割合・組の周期が params と出典の幅の中（違反 {bad_rate}）")
    check("BACKUP_FREQ", bad_freq == 0, f": 山の周波数が params ±1%・幅の中（違反 {bad_freq}）")
    p = {"kind": "chime", "pitch_mult": 1.0, "time_mult": 1.0, "tau_sec": 0.2}; x = a23.make_backup_beep_v23(4.0, FS, 0.5, p)
    ping, pong = x[int(0.50 * FS): int(0.74 * FS)], x[int(0.76 * FS): int(1.0 * FS)]
    check("BACKUP_CHIME", abs(peak_hz(ping, FS, 1000, 4000) - 2000) <= 20 and abs(peak_hz(pong, FS, 1000, 4000) - 1590) <= 20, f": ピン {peak_hz(ping, FS, 1000, 4000):.0f}・ポーン {peak_hz(pong, FS, 1000, 4000):.0f} Hz")
    # ---- サイレン
    pf = S26.fire_params_v26(rng); x = S26.make_fire_siren_v26(13.0, FS, bell_mix=0.0, **{k: v for k, v in pf.items() if k not in ("siren_type", "v26")})
    ts, f0 = S26.track_f0(x, FS)
    acf = np.correlate(f0 - f0.mean(), f0 - f0.mean(), mode="full")[len(f0) - 1:]; k = int(np.argmax(acf[int(4.0 / 0.05): int(8.0 / 0.05)])) + int(4.0 / 0.05)
    period = k * 0.05; top = np.sum(f0 >= pf["f_hi"] - 10) * 0.05 / (13.0 / pf["sweep_period_sec"])
    check("SIREN_FIRE", (S26.FIRE_F_LO[0] - 10 <= f0.min() <= S26.FIRE_F_LO[1] + 10) and (S26.FIRE_F_HI[0] - 10 <= f0.max() <= S26.FIRE_F_HI[1] + 10) and abs(period - pf["sweep_period_sec"]) <= 0.3 and top > 0.3,
          f": 基音 {f0.min():.0f}〜{f0.max():.0f} Hz・周期 {period:.2f}（params {pf['sweep_period_sec']:.2f}）・上で止まる {top:.2f} s/周期")
    pp = S26.patrol_params_v26(rng); y = S26.make_patrol_siren_v26(17.0, FS, **{k: v for k, v in pp.items() if k not in ("siren_type", "v26")})
    ts, g0 = S26.track_f0(y, FS); nper = 17.0 / pp["sweep_period_sec"]
    top = np.sum(g0 >= pp["f_hi"] - 10) * 0.05 / nper; bottom = np.sum(g0 <= pp["f_lo"] + 10) * 0.05 / nper
    check("SIREN_PATROL", (S26.PATROL_F_LO[0] - 10 <= g0.min()) and (g0.max() <= S26.PATROL_F_HI[1] + 10) and top > 0.2 and bottom <= 0.2, f": 基音 {g0.min():.0f}〜{g0.max():.0f}・上で止まる {top:.2f} s/周期・下 {bottom:.2f}")
    bad = 0
    for _ in range(6):
        q = S26.peepo_params_v26(rng)
        if q["weak_beat"]:
            continue
        z = S26.make_peepo_siren_v26(6.0, FS, **{k: v for k, v in q.items() if k not in ("siren_type", "v26", "weak_beat")})
        rate = S26.beat_rate_hz(z, FS, 2.65, 3.2); d = S26.envelope_depth_db(z, FS, 2.65, 3.2, band=(850.0, 1070.0))
        ts, h0 = S26.track_f0(z, FS, f_lo=600.0, f_hi=1100.0)
        if abs(rate - q["beat_hi_hz"]) > 1.0 or abs(d - q["depth_db"]) > 2.0 or abs(np.median(h0[(ts % 1.3) < 0.6]) - 960) > 15:
            bad += 1
    check("SIREN_PEEPO", bad == 0, f": うなりの速さ ±1 Hz・基本の音の深さ ±2 dB・基音 ±15 Hz（違反 {bad}）")
    # ---- 通報音
    bad = 0
    for _ in range(200):
        ap = A26.avas_params(rng)
        if not all(A26.AVAS_F_LOW[0] <= f <= A26.AVAS_F_LOW[1] for f in ap["freqs_hz"]) or len({A26.third_octave_band(f) for f in ap["freqs_hz"]}) < 2 or not any(A26.third_octave_band(f) <= 32 for f in ap["freqs_hz"]) or not (0.008 <= ap["pitch_per_kmh"] <= 0.02):
            bad += 1
    ap = A26.avas_params(np.random.default_rng(5)); x10 = A26.make_avas(2.0, FS, np.random.default_rng(6), ap, 10.0); x20 = A26.make_avas(2.0, FS, np.random.default_rng(6), ap, 20.0)
    f1 = ap["freqs_hz"][0]; r = peak_hz(x20, FS, f1 * 0.9, f1 * 1.35) / peak_hz(x10, FS, f1 * 0.9, f1 * 1.35)
    check("AVAS_FORM", bad == 0 and abs(r - (1 + 10 * ap["pitch_per_kmh"])) < 0.01, f": 200 台の形が条件の中（違反 {bad}）・20 km/h の山 ÷ 10 km/h の山 = {r:.3f}（params {1 + 10 * ap['pitch_per_kmh']:.3f}）")
    # ---- 大型車
    _, info = H25.make_heavy_car(4.0, FS, np.random.default_rng(1), f0=65.0, tire_frac_a=0.15)
    check("HEAVY_LOW_BAND", abs(info["low_band_gain_db"] - 10.0) < 0.3 and abs(info["tire_frac_a_measured"] - 0.15) < 0.005, f": +{info['low_band_gain_db']:.2f} dB・タイヤ {info['tire_frac_a_measured']}")
    draws = [H25.heavy_params(rng) for _ in range(1000)]
    check("HEAVY_LEVEL_OFFSET", all(4.8 <= d["offset_db"] <= 8.6 for d in draws), ": 1,000 回の差が 4.8〜8.6 の中")
    # ---- 自転車
    xb = make_bicycle_v2(CLIP, FS, np.random.default_rng(1)); lv = third_octave_levels(xb, FS, 500.0, 5000.0)
    mid = [v for fc, v in lv.items() if 630 <= fc <= 4000]
    check("BICYCLE_SPECTRUM", max(mid) - min(mid) <= 3.0, f": 630〜4,000 Hz の帯のレベルの幅 {max(mid) - min(mid):.1f} dB")


def stage2() -> None:
    # ---- 車の曲線（2 m の LAFmax）
    bad = []
    for v in (6.5, 10.0, 15.0, 20.0, 30.0):
        dry = make_car_v9(CLIP, FS, np.random.default_rng(1), f0=42.0)
        l1m = v24.base_l1m("car", v); w = render_w(dry * gain_for_spl_a(dry, FS, l1m), v / 3.6)
        got = L24.lafmax_db(w, FS_OUT); want = L24.l2m_curve(v)
        if abs(got - want) > 0.5:
            bad.append((v, round(got, 2), round(want, 2)))
    check("CAR_CURVE_2M", not bad, f": 6.5〜30 km/h で描画の LAFmax = 曲線 ±0.5 dB{'' if not bad else bad}")
    # ---- サイレン: 20 m の前の値と向きの差。plan の本の音源（params → apply_levels の l1m と Δ）を標準の置き方（前方 20 m・静止・z 2.4・マイク 1.2）に置き直し、
    #      本番の道（_make_dry の連なり = v26 の生成器 → v24 の向きの差 → v13 の RMS 合わせ → 描画）で 3 型とも測る（2026-10-12 の直し: 以前は検査が自前の src を組んでいた）
    import copy
    SG = L24.SIREN_MATCH_GEOMETRY; c = m9.sound_speed(m9.TEMP_C); mic = np.array([0.0, 0.0, SG["mic_z_m"]])
    picked = {}
    for r in v24._plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "siren", 60):
        for x in v26.sample_scene_v19(r)["sources"]:
            if x["class"] == "siren" and x["params"]["siren_type"] not in picked:
                picked[x["params"]["siren_type"]] = x
        if len(picked) >= 3:
            break
    got = {}
    for st, x0 in sorted(picked.items()):
        lv = float(x0["law_db"]); res = {}
        for name, pos in (("front", (-SG["dist_m"], 0.0)), ("side", (0.0, SG["dist_m"])), ("rear", (SG["dist_m"], 0.0))):
            src = copy.deepcopy(x0)
            src["wp"] = [[0.0, pos[0], pos[1], SG["src_z_m"]], [CLIP, pos[0], pos[1], SG["src_z_m"]]]; src["dir_x"] = 1.0; src["speed_mps"] = 0.0
            src["t_on"], src["t_off"] = 0.0, CLIP; src["_mic_arr_v24"] = mic.tolist()
            src.pop("preroll_s", None); src.pop("start_dist_m", None)                  # 静止に置き直すので遠方の preroll は無し
            dry = m9._make_dry(src); g = gain_for_spl_a(dry, FS, float(src["l1m_db"]))
            _, stem_wr = m9._render_stem(dry * g, np.array(src["wp"], float), mic, c)
            res[name] = spl_a(stem_wr[0], FS_OUT)
        got[st] = (lv, res)
    chains = sorted({x0["v24"].get("siren_chain", "?") for x0 in picked.values()})
    fronts = {st: (round(lv, 2), round(res["front"], 2)) for st, (lv, res) in got.items()}
    dirs = {st: (round(res["front"] - res["side"], 2), round(res["front"] - res["rear"], 2)) for st, (lv, res) in got.items()}
    check("SIREN_LEVEL_20M", len(got) == 3 and all(abs(res["front"] - lv) <= 0.7 for lv, res in got.values()),
          f": 3 型の前 20 m の受音 = 宣言 ±0.7 dB（宣言, 受音 {fronts}・Δ {[x0['v24'].get('delta_db') for x0 in picked.values()]}・生成器 {chains}・反射と大気吸収込み）")
    check("SIREN_DIRECTIVITY", len(got) == 3 and all(abs((res["front"] - res["side"]) - 12.0) <= 1.0 and abs((res["front"] - res["rear"]) - 14.0) <= 1.0 for _, res in got.values()),
          f": 前 − 横・前 − 後ろ = {dirs} dB")
    # ---- 警告音の 1 m の LAFmax 合わせ（plan の本で）
    rows = v24._plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] in ("horn", "bike_bell", "backup_beep"), 6) + v24._plan_rows(lambda r: r["scenario"] == "v12train", 1)
    diffs = []
    for r in rows:
        s = v26.sample_scene_v19(r)
        for x in s["sources"]:
            if x["class"] in L24.MATCH_MAX_CLASSES and not x.get("preroll_s") and not (x.get("params") or {}).get("train_car") and "train_car_index" not in x:   # 列車の車両は対象外（AR12-08）
                diffs.append(v24.check_match_max_dry(x))
    check("WARN_LAFMAX_MATCH", len(diffs) >= 4 and all(abs(d) < 0.1 for d in diffs), f": {len(diffs)} 本の差 {[round(d, 3) for d in diffs]} dB")
    rows_bk = v24._plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "backup_beep", 3)
    bk = [x for s in [v26.sample_scene_v19(r) for r in rows_bk] for x in s["sources"] if x["class"] == "backup_beep"]
    bk_lv = [float(x.get("l1m_decl_db", x["l1m_db"])) for x in bk]
    check("BACKUP_LEVEL_1M", len(bk) >= 2 and all(77.0 <= v <= 95.0 for v in bk_lv), f": バック音 {len(bk)} 本の宣言値 {[round(v, 1) for v in bk_lv]} が 77〜95 の中（以前は or True で常に合格だった・SV26-02）")
    # ---- EV＋通報音の R138（条件つきの概算・10 本の抽選・総合と 4 帯）
    mins = {10.0: {"total": 50.0, 500: 45.0, 1000: 46.0, 1600: 44.0, 2000: 42.0}, 20.0: {"total": 56.0, 500: 50.0, 1000: 51.0, 1600: 49.0, 2000: 47.0}}
    n_ok_total, n_ok_bands, n = 0, 0, 0
    for i in range(6):
        ap = A26.avas_params(np.random.default_rng(100 + i))
        for v in (10.0, 20.0):
            base = make_car_ev(CLIP, FS, np.random.default_rng(7 + i), speed_mps=v / 3.6, pwm_frac_a=ap["pwm_frac_a"], tire_frac_a=0.75 + (0.10 - ap["pwm_frac_a"]), motor_frac_a=0.15)
            A = A26.make_avas(CLIP, FS, np.random.default_rng(8 + i), ap, v)
            l1m_car = v24.base_l1m("car", v); l1m_avas = ap["l2m_dba"] + v24.delta_db("car", v)
            X = base + A * (a_weighted_rms(base, FS) / a_weighted_rms(A, FS)) * 10.0 ** ((l1m_avas - l1m_car) / 20.0)
            X = X * gain_for_spl_a(base, FS, l1m_car)                      # 下地を曲線の値に（通報音はその上）
            w = render_w(X, v / 3.6)
            seg = w[int((G["cpa_s"] - 0.5) * FS_OUT): int((G["cpa_s"] + 0.5) * FS_OUT)]
            wa = np.fft.irfft(np.fft.rfft(seg) * a_weight_amplitude(np.fft.rfftfreq(len(seg), 1.0 / FS_OUT)), n=len(seg))
            lv_a = third_octave_levels(wa, FS_OUT, 400.0, 2500.0); tot = spl_a(seg, FS_OUT)
            n += 1; n_ok_total += int(tot >= mins[v]["total"])
            bands_ok = [fc for fc in (500, 1000, 1600, 2000) if lv_a.get(fc, -99) >= mins[v][fc]]
            n_ok_bands += int(len(bands_ok) >= 2 and any(fc <= 1600 for fc in bands_ok))
    check("AVAS_R138_LEVELS", n_ok_total == n and n_ok_bands == n, f": 12 条件（6 台 × 10／20 km/h）で総合 ≥ 下限 {n_ok_total}/{n}・4 帯（500／1,000／1,600／2,000）のうち 2 帯以上（1 つは ≤1,600 Hz）が下限以上 {n_ok_bands}/{n}（帯の条件も合否に入れた・SV26-02。全帯の表（160〜5,000 Hz）・上限 75・F 時間重み・測定の全体は未検査＝次の版）")


def main() -> int:
    quick = "--quick" in sys.argv[1:]
    stage1()
    if not quick:
        stage2()
    k = sum(c for _, c in results)
    print(f"{k}/{len(results)} {'ALL PASS' if k == len(results) else 'NG'}")
    return 0 if k == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
