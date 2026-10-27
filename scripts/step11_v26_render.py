# -*- coding: utf-8 -*-
"""step11_v26_render.py — サイレンの形と、EV・HV の接近通報音（2026-10-12・依頼文 r5 §3・§10・順番の 5）。v25 の上に重ねる版。

1. サイレン（`outdoor_seld.siren_v26`）: 消防車のウー（400 → 780 Hz・上で止まる・約 6 s・鐘つき）、パトカーのウー（450 → 850・上で止まる・4／8 s）、
   ピーポー（960／770 Hz・近い 2 つの音のうなり・ピーとポーで別の幅・深さ U(5, 18) dB・30% は揺れが弱い・倍音は車ごと）。
   場面の乱数の消費は変えない（params は元の抽選のあと独立の乱数で差し替え・**変わる列: params**）。型の割合（ピーポー／ウー／消防）は今のまま。大きさ・高さ・向きの差は v24。
   生成器は v9 の _make_dry が呼ぶ名前を差し替える（params に v26 があるときだけ）。外側の hook で作ると v24 の向きの差と前の値の合わせを素通りする（17:2x に直した）。
2. 接近通報音（`outdoor_seld.avas_v26`）: EV の車（v12 の変種 ev・全車の 15%・audio_seed の決定論の抽選のまま）に、通報音の成分を**上から足す**（AS10-04）。
   下地の EV の音（タイヤ＋モーター＋PWM）は大きさを変えず（v24 の曲線）、PWM（8〜10 kHz）は A 特性パワーの 2% 以下に。通報音は横 2 m で U(46, 70) dB(A)（18:24 に上限 60 → 70）
   （1 m の値は車と同じ Δ(v) で換算・近似）、高さは速さで上がる、U(20, 30) km/h で消える。発進（v19launch）は速さの時系列で高さと利得が変わる。
   較正は下地の大きさが曲線の値のままになるように、v13 の RMS 合わせに渡す l1m_db を 20 log(RMS(下地+通報音)/RMS(下地)) だけ上げる（l1m_decl_db に宣言値）。
   **変わる列: EV の車の params（avas）・l1m_db（RMS 合わせ用）・scene の avas**。ラベルは car_drive のまま。大型車（v25）には足さない。

使い方:
  python scripts/step11_v26_render.py selftest
  python scripts/step11_v26_render.py audition <出力dir>   # サイレン 3 型の旧／新・EV の旧（PWM 10%）／新（PWM ≤2%＋通報音）のドライ音
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v25_render as v25  # noqa: E402
from outdoor_seld import siren_v26 as S26  # noqa: E402
from outdoor_seld import avas_v26 as A26  # noqa: E402
from outdoor_seld.calibration import a_weighted_rms, spl_a  # noqa: E402
from outdoor_seld.engine_ev import make_car_ev  # noqa: E402

v24, v23, v22, m9, v13, v19r = v25.v24, v25.v23, v25.v22, v25.m9, v25.v13, v25.v19r
v12 = v22.v12
SALT = 20261017
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"


def _irng(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


# ---------------------------------------------------------------- 1. サイレンの params
def _warn_params_v26(prev):
    def fn(cls: str, rng: np.random.Generator, seed: int) -> dict:
        p = prev(cls, rng, seed)
        if cls == "siren":
            r = _irng(seed, SALT, 1, round(float(p.get("f_hi", 0.0)) * 1000))
            st = p.get("siren_type")
            q = S26.peepo_params_v26(r) if st == "peepo" else (S26.fire_params_v26(r) if st == "fire" else S26.patrol_params_v26(r))
            if "audio_seed" in p:
                q["audio_seed"] = p["audio_seed"]
            q["v26_old"] = {k: v for k, v in p.items() if k != "audio_seed"}
            return q
        return p
    fn._v26 = True
    return fn


# ---------------------------------------------------------------- 2. 速さの時系列（通報音の高さと利得）
def speed_series_kmh(wp, n: int, fs: int) -> np.ndarray:
    w = np.array(wp, dtype=float)
    if len(w) < 2:
        return np.zeros(n)
    t = np.arange(n) / fs
    seg_t = w[:-1, 0]; seg_v = np.linalg.norm(w[1:, 1:4] - w[:-1, 1:4], axis=1) / np.maximum(w[1:, 0] - w[:-1, 0], 1e-6)
    idx = np.clip(np.searchsorted(w[:, 0], t, side="right") - 1, 0, len(seg_v) - 1)
    v = seg_v[idx] * 3.6
    k = int(0.25 * fs)                                              # 0.25 s の移動平均で区分等速の段差をならす
    if k > 1:
        v = np.convolve(np.pad(v, (k // 2, k - k // 2 - 1), mode="edge"), np.ones(k) / k, mode="valid")
    return v


def is_ev(src: dict) -> bool:
    p = src.get("params", {})
    return src["class"] == "car_drive" and not p.get("heavy") and (p.get("force_variant") or v12._car_variant(int(p["audio_seed"]))) == "ev"


def _make_dry_v26(prev):
    def fn(src: dict) -> np.ndarray:
        cls, p = src["class"], src.get("params", {})
        if is_ev(src):
            seed = int(p["audio_seed"]); fs = m9.FS_SIM; n = int(round(m9.CLIP * fs))
            r = _irng(seed, SALT, 2)
            ap = src.get("avas") or A26.avas_params(r)
            speed = abs(float(src.get("speed_mps", 6.0)))
            base = make_car_ev(m9.CLIP, fs, np.random.default_rng(seed), speed_mps=speed, pwm_frac_a=ap["pwm_frac_a"],
                               tire_frac_a=0.75 + (0.10 - ap["pwm_frac_a"]), motor_frac_a=0.15)   # PWM を減らした分はタイヤに（合計 1）
            src["car_variant"] = "ev_v26"
            v_t = speed_series_kmh(src["wp"], n, fs) if src.get("launch") else np.full(n, speed * 3.6)
            carrier = A26.make_avas(m9.CLIP, fs, np.random.default_rng(seed * 7 + 41), ap, v_t, apply_gain=False)   # 減衰前の搬送音（SV26-01）
            g_t = np.asarray(A26.avas_gain(v_t, ap), dtype=float)
            A = carrier * g_t                                                                                      # 速さの包絡（fade・停止）は較正のあとに掛かる
            decl = src.setdefault("l1m_decl_db", float(src["l1m_db"]))
            v_cpa = float(np.interp(float(src.get("t_cpa_rel_s", src.get("t_cpa_s", m9.CLIP / 2))), np.arange(n) / fs, v_t))
            l1m_avas = ap["l2m_dba"] + v24.delta_db("car", max(v_cpa, 1.0))
            t_on, t_off = float(src.get("t_on", 0.0)), float(src.get("t_off", m9.CLIP))
            a0, a1 = int(t_on * fs), max(int(t_off * fs), int(t_on * fs) + 1)
            # 通報音の大きさは **減衰前の搬送音**を鳴る窓 [t_on, t_off] の中で下地と比べて較正し（= 利得 1 のとき l1m_avas）、
            # そのあと速さの包絡 g_t（fade・停止）を掛ける。利得つきの音で正規化すると減衰が元に戻る（Astra SV26-01・AR12-09(a) の直しだけでは足りなかった・2026-10-13）
            ra_b = a_weighted_rms(base[a0:a1], fs)
            ra_c = a_weighted_rms(carrier[a0:a1], fs)
            if ra_c > 0:
                X = base + A * (ra_b / ra_c) * 10.0 ** ((l1m_avas - decl) / 20.0)
            else:
                X = base
            src["l1m_db"] = decl + (spl_a(X[a0:a1], fs) - spl_a(base[a0:a1], fs))      # 下地が decl のまま・通報音が l1m_avas
            src["avas"] = dict(ap, l1m_avas_db=round(l1m_avas, 2), v_cpa_kmh=round(v_cpa, 2), gain_at_cpa=round(float(A26.avas_gain(v_cpa, ap)), 3),
                               mix_db=round(spl_a(X[a0:a1], fs) - spl_a(base[a0:a1], fs), 3))
            return X
        return prev(src)
    fn._v26 = True
    return fn


# ---------------------------------------------------------------- 1b. サイレンの生成器: v9 の _make_dry が呼ぶ名前（make_fire_siren／make_peepo_siren／make_siren）を差し替える
#   params に v26 があるときだけ v26 の生成器へ。外側の hook で作ると v24 の向きの差と前の値の合わせ（_make_dry_v24）を素通りしてしまう
#   （音の検査 SIREN_DIRECTIVITY で発覚・2026-10-12 17:2x・コミット 03632da02 の v26 はこの素通りがあった）。
#   v9 は fire に rng= を、peepo／wail に params の残り全部を渡す。長さは受けたまま（preroll の本は v13 の _make_dry_len が m9.CLIP を伸ばす）。
def _siren_dispatch(orig, maker_v26, strip=()):
    def fn(duration_sec, fs, *args, v26=False, v26_old=None, **kw):
        if v26:
            return maker_v26(duration_sec, fs, *args, **{k: v for k, v in kw.items() if k not in strip})
        return orig(duration_sec, fs, *args, **kw)
    fn._v26 = True; fn._orig = orig
    return fn


# ---------------------------------------------------------------- 登録
if not getattr(m9._warn_params, "_v26", False):
    m9._warn_params = _warn_params_v26(m9._warn_params)
if not getattr(m9.make_fire_siren, "_v26", False):
    m9.make_fire_siren = _siren_dispatch(m9.make_fire_siren, S26.make_fire_siren_v26)
    m9.make_peepo_siren = _siren_dispatch(m9.make_peepo_siren, S26.make_peepo_siren_v26, strip=("audio_seed",))
    m9.make_siren = _siren_dispatch(m9.make_siren, S26.make_patrol_siren_v26, strip=("audio_seed",))
if not getattr(m9._make_dry, "_v26", False):
    m9._make_dry = _make_dry_v26(m9._make_dry)
sample_scene_v19 = v19r.sample_scene_v19
generate_clip = v19r.generate_clip


# ---------------------------------------------------------------- 検査
def _plan_rows(pred, n: int) -> list:
    out = []
    with open(PLAN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if pred(r):
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); out.append(r)
                if len(out) >= n:
                    break
    return out


def check_siren_params(src: dict) -> None:
    p = src["params"]; assert p.get("v26"), "v26 のサイレンの params が無い"
    st = p["siren_type"]
    if st == "peepo":
        assert S26.PEEPO_BEAT_HI[0] <= p["beat_hi_hz"] <= S26.PEEPO_BEAT_HI[1] and S26.PEEPO_BEAT_LO[0] <= p["beat_lo_hz"] <= S26.PEEPO_BEAT_LO[1]
        assert 0.0 <= p["depth_db"] <= S26.PEEPO_DEPTH_DB[1]
    elif st == "fire":
        assert S26.FIRE_F_LO[0] <= p["f_lo"] <= S26.FIRE_F_LO[1] and S26.FIRE_F_HI[0] <= p["f_hi"] <= S26.FIRE_F_HI[1] and S26.FIRE_PERIOD[0] <= p["sweep_period_sec"] <= S26.FIRE_PERIOD[1]
    else:
        assert S26.PATROL_F_LO[0] <= p["f_lo"] <= S26.PATROL_F_LO[1] and S26.PATROL_F_HI[0] <= p["f_hi"] <= S26.PATROL_F_HI[1]


def check_ev_front_level(src: dict) -> float:
    """EV: 通報音を足しても、v13 の RMS 合わせのあとで下地（通報音なし）の大きさが宣言値になる。差 [dB] を返す。"""
    fs = m9.FS_SIM
    X = m9._make_dry(src)
    t_on, t_off = float(src.get("t_on", 0.0)), float(src.get("t_off", m9.CLIP)); a0, a1 = int(t_on * fs), max(int(t_off * fs), int(t_on * fs) + 1)
    from outdoor_seld.calibration import gain_for_spl_a
    g = gain_for_spl_a(X[a0:a1], fs, float(src["l1m_db"]))
    base = make_car_ev(m9.CLIP, fs, np.random.default_rng(int(src["params"]["audio_seed"])), speed_mps=abs(float(src["speed_mps"])),
                       pwm_frac_a=src["avas"]["pwm_frac_a"], tire_frac_a=0.75 + (0.10 - src["avas"]["pwm_frac_a"]), motor_frac_a=0.15)
    return spl_a(base[a0:a1] * g, fs) - float(src["l1m_decl_db"])


def _V25_DUMP():
    return r"""
import json, sys
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import step11_v25_render as v25
rows = json.loads(sys.stdin.read()); out = []
for r in rows:
    s = v25.sample_scene_v19(r)
    out.append({"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"], "t_off": x["t_off"], "l1m": x.get("law_db") if x["class"] == "siren" else x.get("l1m_db")} for x in s["sources"]]})
print(json.dumps(out))
""" % (str(ROOT / "scripts"), str(ROOT / "src"))


def selftest() -> int:
    ok = []
    sr = _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "siren", 12)
    types = {}
    for r in sr:
        s = sample_scene_v19(r)
        for x in s["sources"]:
            if x["class"] == "siren":
                check_siren_params(x); types[x["params"]["siren_type"]] = types.get(x["params"]["siren_type"], 0) + 1
                dry = m9._make_dry(x)
                assert len(dry) == int(round(m9.CLIP * m9.FS_SIM)) or x.get("preroll_s")
    ok.append((f"サイレン {sum(types.values())} 本: v26 の params（型 {types}）・ドライ音が作れる", sum(types.values()) >= 10 and len(types) >= 2))
    # v24 を通る（向きの差・前の値）: 音の検査 SIREN_DIRECTIVITY で見つけた素通り（2026-10-12）の回帰
    ok.append(("v9 のサイレン生成器 3 つが v26 の振り分けに差し替わっている（外側の hook ではない）",
               all(getattr(getattr(m9, nm), "_v26", False) for nm in ("make_fire_siren", "make_peepo_siren", "make_siren"))))
    near = []
    for r in sr:
        s = sample_scene_v19(r)
        near += [x for x in s["sources"] if x["class"] == "siren" and x["params"].get("v26")]     # preroll の本も（check_siren_front_level は pre を扱う）
    near = near[:6]
    fronts = [v24.check_siren_front_level(x) for x in near]
    dirs = [x.get("v24", {}).get("dir_mean_db") for x in near]
    ok.append((f"v26 のサイレン {len(near)} 本が v24 の向きの差を通る（dir_mean {dirs} dB・前の値の差 {[round(f, 3) for f in fronts]} dB・Δ {[x['v24'].get('delta_db') for x in near]}）",
               len(near) >= 3 and all(d is not None for d in dirs) and all(abs(f) < 0.05 for f in fronts) and all("delta_db" in x["v24"] for x in near)))
    far = []
    for r in _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "siren", 60):
        s = sample_scene_v19(r)
        far += [x for x in s["sources"] if x["class"] == "siren" and x.get("preroll_s")]
        if far:
            break
    if far:
        x = far[0]; L = v13._make_dry_len(x, m9.CLIP + float(x["preroll_s"]))
        ok.append((f"preroll のサイレン（遠方・{x['params']['siren_type']}）: ドライ音の長さ = CLIP + preroll = {len(L) / m9.FS_SIM:.1f} s", abs(len(L) / m9.FS_SIM - (m9.CLIP + float(x["preroll_s"]))) < 1e-6))
    # ピーポーのドライ音: うなりの速さが params どおり
    pp = None
    for r in sr:
        s = sample_scene_v19(r)
        for x in s["sources"]:
            if x["class"] == "siren" and x["params"]["siren_type"] == "peepo" and not x["params"]["weak_beat"] and not x.get("preroll_s"):
                pp = x; break
        if pp:
            break
    if pp is not None:
        dry = m9._make_dry(pp); fs = m9.FS_SIM
        rate = S26.beat_rate_hz(dry, fs, 2.6 + 0.05, 2.6 + 0.60); d = S26.envelope_depth_db(dry, fs, 2.65, 3.2, band=(850.0, 1070.0))
        ok.append((f"ピーポー: ドライ音のうなり（速さ {rate:.1f} Hz vs params {pp['params']['beat_hi_hz']:.1f}・深さ {d:.1f} vs {pp['params']['depth_db']:.1f} dB）", abs(rate - pp["params"]["beat_hi_hz"]) < 1.0 and abs(d - pp["params"]["depth_db"]) < 2.0))
    # EV: 15% の車に通報音・下地の大きさは保つ・速さが高いと無音
    cars = _plan_rows(lambda r: r["scenario"] in ("v11core", "v19fast", "v19launch") and r["close_slow"] != "1", 400)
    ev, n_ev, diffs, stopped, sounding = [], 0, [], 0, 0
    for r in cars:
        s = sample_scene_v19(r)
        for x in s["sources"]:
            if x["class"] == "car_drive" and is_ev(x):
                n_ev += 1
                if len(ev) < 8:
                    ev.append((r, x))
        if n_ev >= 8:
            break
    for r, x in ev:
        diffs.append(check_ev_front_level(x))
        if x["avas"]["gain_at_cpa"] > 0:
            sounding += 1
        else:
            stopped += 1
    ok.append((f"EV {len(ev)} 台: 通報音を足しても下地の大きさは宣言値（差 {[round(d, 3) for d in diffs]} dB）・鳴る {sounding}／止める速さより速い {stopped}（{[(round(x['v24']['v_kmh']), x['avas']['gain_at_cpa'], round(x['avas']['l1m_avas_db'], 1)) for _, x in ev]}）",
               len(ev) >= 6 and all(abs(d) < 0.05 for d in diffs)))
    # 通報音の高さが見える（鳴っている EV の 1 台で、params の周波数に山）
    cand = [x for _, x in ev if x["avas"]["gain_at_cpa"] >= 0.99 and not x.get("launch")]
    if cand:
        x = cand[0]; fs = m9.FS_SIM; X = m9._make_dry(x)
        v = x["v24"]["v_kmh"]; mult = 1.0 + x["avas"]["pitch_per_kmh"] * (v - A26.AVAS_V_REF_KMH)
        spec = np.abs(np.fft.rfft(X)); f = np.fft.rfftfreq(len(X), 1.0 / fs)
        hits = []
        for f0 in x["avas"]["freqs_hz"]:
            fe = f0 * mult; m = (f > fe * 0.97) & (f < fe * 1.03); mb = (f > fe * 0.85) & (f < fe * 1.15)
            hits.append(float(spec[m].max() / np.median(spec[mb])))
        ok.append((f"EV の通報音: params の周波数（×速さの倍率 {mult:.3f}）に山（周辺の中央値の {[round(h) for h in hits]} 倍）", all(h > 5 for h in hits)))
    # 発進の EV: 速さの時系列で高さが上がる（1 台あれば）
    # SV26-01: 同じ EV・同じ通報音で、一定速 15 km/h（利得 1）・23.5 km/h（利得 0.5）・27 km/h（利得 0）の通報音の大きさ（X − 下地）
    ap0 = A26.avas_params(np.random.default_rng(77)); ap0["v_stop_kmh"], ap0["v_fade_kmh"] = 25.0, 22.0
    def ev_src(v_kmh):
        return {"class": "car_drive", "kind": "vehicle", "track": 0, "params": {"audio_seed": 12345, "f0": 42.0, "force_variant": "ev"}, "speed_mps": v_kmh / 3.6,
                "wp": [[0.0, -30.0, 2.0, 0.5], [m9.CLIP, 30.0, 2.0, 0.5]], "t_on": 0.0, "t_off": m9.CLIP, "l1m_db": 60.0, "law_db": 55.0, "avas": dict(ap0), "v24": {"rule": "car_curve", "v_kmh": v_kmh}}
    lv = {}
    for v in (15.0, 23.5, 27.0):
        s_ = ev_src(v); X = m9._make_dry(s_); base_ = make_car_ev(m9.CLIP, m9.FS_SIM, np.random.default_rng(12345), speed_mps=v / 3.6, pwm_frac_a=ap0["pwm_frac_a"], tire_frac_a=0.75 + (0.10 - ap0["pwm_frac_a"]), motor_frac_a=0.15)
        d = X - base_
        # X − 下地 は v13 の利得の前なので下地の生の RMS に比例する → 下地の RMS で割り、1 m への換算 Δ(v) の差も除いて「2 m の宣言値に対する比」で比べる
        lv[v] = float(a_weighted_rms(d, m9.FS_SIM) / max(a_weighted_rms(base_, m9.FS_SIM), 1e-12)) / 10.0 ** (v24.delta_db("car", v) / 20.0)
    r_half = 20 * np.log10(lv[23.5] / lv[15.0]) if lv[15.0] > 0 else 99.0
    ok.append((f"通報音の減衰が残る（SV26-01）: 23.5 km/h（利得 0.5）は 15 km/h より {r_half:.2f} dB（−6.02 ±0.1・下地の RMS と Δ(v) を除いた比）・27 km/h（止める速さの上）は無音（{lv[27.0]:.2e}）", abs(r_half + 6.02) < 0.1 and lv[27.0] < 1e-9 * max(lv[15.0], 1e-12)))
    la = _plan_rows(lambda r: r["scenario"] == "v19launch", 200)
    lev = None
    for r in la:
        s = sample_scene_v19(r)
        for x in s["sources"]:
            if x["class"] == "car_drive" and is_ev(x):
                lev = x; break
        if lev:
            break
    if lev is not None:
        fs = m9.FS_SIM; X = m9._make_dry(lev); n = len(X); v_t = speed_series_kmh(lev["wp"], n, fs)
        t_on = float(lev["t_on"]); t_end = float(lev["launch"]["t_acc_s"]) + t_on
        ok.append((f"発進の EV: 待機中（t < t_on={t_on:.2f}）は速さ 0（{v_t[int(max(t_on - 0.3, 0) * fs)]:.1f} km/h）・加速の終わり {t_end:.2f} s で {v_t[min(int(t_end * fs), n - 1)]:.1f} km/h ≈ 終端 {lev['launch']['v_end_kmh']}",
                   v_t[int(max(t_on - 0.3, 0) * fs)] < 0.5 and abs(v_t[min(int(t_end * fs) + int(0.3 * fs), n - 1)] - lev["launch"]["v_end_kmh"]) < 1.0))
    # 場面の乱数: v25 だけの別プロセスと一致（v26 は場面を変えない）
    sample = sr[:2] + cars[:2] + la[:1]
    res = subprocess.run([sys.executable, "-c", _V25_DUMP()], input=json.dumps(sample), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        ref = json.loads(res.stdout.strip().splitlines()[-1]); bad = []
        for a, r in zip(ref, sample):
            s = sample_scene_v19(r)
            mine = {"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"], "t_off": x["t_off"], "l1m": x.get("law_db") if x["class"] == "siren" else x.get("l1m_db")} for x in s["sources"]]}
            if json.dumps(a, sort_keys=True) != json.dumps(mine, sort_keys=True):
                bad.append(r["clip_id"])
        ok.append((f"場面の乱数: v25 だけの別プロセスと {len(sample)} 本が一致（v26 は場面を変えない・サイレンは 20 m の値で比べる＝1 m の値は形ごとの Δ で意図して変わる）{'' if not bad else bad}", not bad))
    else:
        ok.append(("場面の乱数: 別プロセスの v25 が動く", False)); print(res.stderr[-600:])
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# ---------------------------------------------------------------- 試聴（ドライ音）
def audition(out_dir: Path) -> int:
    import soundfile as sf
    from outdoor_seld.siren import make_peepo_siren, make_siren, make_fire_siren
    out_dir.mkdir(parents=True, exist_ok=True); fs = m9.FS_SIM
    lines = ["サイレンの形と EV の接近通報音（v26・2026-10-12）の試聴: ドライ音（伝搬なし）をピーク −3 dBFS でそろえた。鳴り方と音色だけ聞く。wav は git に入れない。", ""]

    def save(name, x, note):
        y = x / (np.max(np.abs(x)) + 1e-12) * 10 ** (-3 / 20)
        sf.write(str(out_dir / name), (y * 32767).astype(np.int16), fs, subtype="PCM_16"); lines.append(f"{name}: {note}"); print("wrote", name)
    save("siren_peepo_old.wav", make_peepo_siren(6.0, fs), "旧 ピーポー: 960／770 Hz・倍音固定・揺れなし")
    rng = np.random.default_rng(5)
    for i in range(2):
        p = S26.peepo_params_v26(rng)
        while p["weak_beat"]:
            p = S26.peepo_params_v26(rng)
        save(f"siren_peepo_new_{i + 1}.wav", S26.make_peepo_siren_v26(6.0, fs, **{k: v for k, v in p.items() if k not in ('siren_type', 'v26', 'weak_beat')}), f"新 ピーポー: うなり ピー {p['beat_hi_hz']:.1f}／ポー {p['beat_lo_hz']:.1f} Hz・深さ {p['depth_db']:.1f} dB・倍音 {p['h2_db']:.0f}／{p['h3_db']:.0f} dB")
    p = S26.peepo_params_v26(rng)
    while not p["weak_beat"]:
        p = S26.peepo_params_v26(rng)
    save("siren_peepo_new_weak.wav", S26.make_peepo_siren_v26(6.0, fs, **{k: v for k, v in p.items() if k not in ('siren_type', 'v26', 'weak_beat')}), f"新 ピーポー（揺れの弱い車・30%）: 深さ {p['depth_db']:.1f} dB")
    save("siren_fire_old.wav", make_fire_siren(13.0, fs, rng=np.random.default_rng(1)), "旧 消防車: 361 → 710 Hz・2.97 s の正弦の往復・鐘")
    p = S26.fire_params_v26(rng)
    save("siren_fire_new.wav", S26.make_fire_siren_v26(13.0, fs, rng=np.random.default_rng(1), **{k: v for k, v in p.items() if k not in ('siren_type', 'v26')}), f"新 消防車: {p['f_lo']:.0f} → {p['f_hi']:.0f} Hz・上で止まる（周期の {p['hold_frac']:.2f}）・周期 {p['sweep_period_sec']:.1f} s・鐘 {p['bell_f0']:.0f} Hz")
    save("siren_patrol_old.wav", make_siren(9.0, fs, f_lo=435.0, f_hi=870.0, sweep_period_sec=4.0), "旧 パトカー: 435 → 870 Hz・4 s の正弦の往復")
    p = S26.patrol_params_v26(rng)
    save("siren_patrol_new.wav", S26.make_patrol_siren_v26(9.0, fs, **{k: v for k, v in p.items() if k not in ('siren_type', 'v26')}), f"新 パトカー: {p['f_lo']:.0f} → {p['f_hi']:.0f} Hz・上で止まる（周期の {p['hold_frac']:.2f}）・周期 {p['sweep_period_sec']:.1f} s")
    # EV
    for v_kmh in (10.0, 20.0):
        base_old = make_car_ev(6.0, fs, np.random.default_rng(7), speed_mps=v_kmh / 3.6)
        save(f"ev_old_{int(v_kmh)}kmh.wav", base_old, f"旧 EV {v_kmh:.0f} km/h: タイヤ 75%＋PWM 10%＋モーター 15%・通報音なし")
        ap = A26.avas_params(np.random.default_rng(11)); ap["v_stop_kmh"], ap["v_fade_kmh"] = 25.0, 22.0
        base = make_car_ev(6.0, fs, np.random.default_rng(7), speed_mps=v_kmh / 3.6, pwm_frac_a=ap["pwm_frac_a"], tire_frac_a=0.75 + (0.10 - ap["pwm_frac_a"]), motor_frac_a=0.15)
        A = A26.make_avas(6.0, fs, np.random.default_rng(12), ap, v_kmh)
        l1m_car = v24.base_l1m("car", v_kmh); l1m_avas = ap["l2m_dba"] + v24.delta_db("car", v_kmh)
        X = base + A * (a_weighted_rms(base, fs) / a_weighted_rms(A, fs)) * 10.0 ** ((l1m_avas - l1m_car) / 20.0)
        save(f"ev_new_{int(v_kmh)}kmh.wav", X, f"新 EV {v_kmh:.0f} km/h: PWM {ap['pwm_frac_a'] * 100:.1f}%＋通報音（{[round(f) for f in ap['freqs_hz']]} Hz{'＋' + str(round(ap['high_hz'])) + ' Hz' if ap['high_hz'] else ''}・2 m で {ap['l2m_dba']:.0f} dB(A) vs 車 {v24.L24.l2m_curve(v_kmh):.1f}）")
    ap = A26.avas_params(np.random.default_rng(13)); ap["v_stop_kmh"], ap["v_fade_kmh"] = 25.0, 22.0
    n = int(8.0 * fs); v = np.clip(np.linspace(-2.0, 30.0, n), 0.0, 30.0)
    base = make_car_ev(8.0, fs, np.random.default_rng(7), speed_mps=15.0 / 3.6, pwm_frac_a=ap["pwm_frac_a"], tire_frac_a=0.75 + (0.10 - ap["pwm_frac_a"]), motor_frac_a=0.15)
    A = A26.make_avas(8.0, fs, np.random.default_rng(14), ap, v)
    X = base * np.clip(v / 5.0, 0.0, 1.0) + A * (a_weighted_rms(base, fs) / a_weighted_rms(A, fs)) * 10.0 ** ((ap["l2m_dba"] + 4.5 - v24.base_l1m("car", 15.0)) / 20.0)
    save("ev_new_launch_0to30kmh.wav", X, f"新 EV 発進（0 → 30 km/h・8 s）: 止まっている間は無音・走り出すと通報音（高さが上がる）・{ap['v_fade_kmh']:.0f}〜{ap['v_stop_kmh']:.0f} km/h で消える（下地は目安）")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "audition":
        sys.exit(audition(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "audition_2026-10-12_siren_ev"))
    print(__doc__); sys.exit(2)
