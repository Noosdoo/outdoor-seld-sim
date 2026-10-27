# -*- coding: utf-8 -*-
"""step11_v13_render.py — v13（合成データ修正の束）レンダラ。v12チェーンの上に積む。

設計= md/design/v13設計書_合成データ修正の束_2026-09-02.md。候補リスト=
md/design/合成データ修正候補リスト_⑫⑥_2026-09-02.md。本人試聴OK（S1/S2/R1、2026-09-02）。

⚠️ v11/v12（凍結・確定評価の土俵）は一切触らない。出力先は `out/dataset_outdoor_siren_v13/`。
v13 は探索用（開発値）。確定評価v2は11月に別途1回。

## 変更点（物理・較正・音源レシピ・車・列車・キック・バイクは v12 と同一）

S1 サイレン: 発音窓を廃止しクリップ全体で鳴動。クリップ開始時の距離 d0 を対数一様(30m, 2km)から
   引く（80%接近＝t_cpa=√(d0²−y²)/v・20%通過済み t_cpa~U(−6,2)）。**プリロール**: 放射を
   t=−PRE（PRE=ceil(d0/c)+1 s）から始めてレンダし末尾10秒を切り出す＝伝播遅延の冒頭無音を消す。
S2 クラクション: 1〜3回の短発（1回0.25〜1.2s・間隔0.15〜0.3s）、t_on = t_cpa − U(0,3)。
S3 バック音: 後退区間の全体で鳴動（t_on=0, t_off=CLIP）。
S4 ベル: 1〜3回のチリン、t_on = t_cpa − U(0,3)。
可聴ゲート: 警告音4クラス（siren/horn/backup_beep/bike_bell）のラベルも車と同じゲート
   （フレームSNR_A ≥ 0 dB・穴埋め2）を通す。
R1 雨: plan の rain 列（20%）に従い雨音（src/outdoor_seld/rain.py）を暗騒音に加算。
   可聴マスクの分母（noise）にも入る。
S7 静止:歩行=30:70 は plan 側（step10_v13_plan.py）。

乱数: 元シーンは v11core/v12ext サンプラと同一seed・同一消費順。変更量だけ別系列
rng([seed, 20260902]) で引く。

使い方:
  python scripts/step11_v13_render.py verify   # 回帰: 修正OFF+雨なしで v12 の既存クリップとビット一致
  python scripts/step11_v13_render.py smoke    # 各修正1本ずつ生成し要点を表示
  python scripts/_run_v13_gen.py --rows a-b    # 全量（サーバ・シャード）
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v12_render as v12  # noqa: E402  （v12チェーン: 都市騒音・車バリアント・8クラス）
m9, m11v = v12.m9, v12.m11v
from outdoor_seld.rain import diffuse_foa_rain  # noqa: E402

# ---- 出力先（v12/v11 本体は不変更） --------------------------------------------
DS_NAME_V13 = "outdoor_siren_v13"
m9.DS = ROOT / "out" / f"dataset_{DS_NAME_V13}"
m9.WORK = m9.DS / "work"
PLAN_V13 = m9.DS / "plan"

# ---- 警告音のラベルにも可聴ゲート ----------------------------------------------
WARN_GATED = {"siren", "horn", "backup_beep", "bike_bell"}
VEHICLE_GATED_V13 = set(v12.VEHICLE_GATED) | WARN_GATED

# ---- S1〜S4 の定数 ----------------------------------------------------------------
SIREN_START_DIST = (30.0, 2000.0)     # 遠方接近: クリップ開始時の距離（対数一様）
SIREN_TCPA_RECEDE = (-6.0, 2.0)       # 通過済み
# v13 = (近接通過 0 / 遠方接近 0.8 / 通過済み 0.2)。v13 では近接例が半減（≤50m 100→48%）したため、
# v2 用に「近接通過」（旧来どおり CPA がクリップ内・WARN_TCPA=3〜7s・鳴りっぱなし）の割合を持たせる。
# 既定は v13 と同一（回帰を壊さない）。v2 の宣言で例えば NEAR=0.5 / FAR=0.3 / RECEDE=0.2 に変える。
SIREN_MIX = {"near": 0.0, "far": 0.8, "recede": 0.2}
HORN_N_P = (0.50, 0.35, 0.15)         # 1回/2回/3回
HORN_SEC = (0.25, 1.2)
HORN_GAP = (0.15, 0.30)
BELL_N_P = (0.50, 0.35, 0.15)
NEAR_CPA_LEAD = (0.0, 3.0)            # 短発の時刻 = t_cpa − U(0,3)
REWRITE_SALT = 20260902
GRAMMAR_OFF = False                   # verify 用（Trueで S1〜S4 を無効化）
# 2026-10-26 本体 §2-1・§2-2（変更票 #75）: v30 の正解の線と背景の定義。v30 の入口（step11_v30_render.active）の中だけ
# {"line_db": −5.0, "gated": ゲートを掛けるクラスの集合} を入れる。None = 旧（線 m9.AUDIBLE_SNR_DB・背景 = 暗騒音＋雨・masks 3 列）＝v29 まで 1 バイトも変えない
GATE_V30 = None
# 2026-10-26 本体 §2-3: v30 の背景の形（本ごとの折れ点と傾き）。v30 の入口の中だけ、場面 s と行 row から {"knee_hz", "slope_db_oct"} を返す関数を入れる。
# None = 旧（noise_v12 の形）。暗騒音の白色雑音の乱数（s["noise"]["seed"]）は同じ流れをそのまま使う（形だけが変わる）
NOISE_SHAPE_V30 = None


def _lin_extrap(wp: np.ndarray, t: float) -> np.ndarray:
    """2点直線軌道 wp[(2,4)] を時刻 t に外挿した位置。"""
    (t0, *p0), (t1, *p1) = wp[0], wp[1]
    p0, p1 = np.array(p0), np.array(p1)
    return p0 + (p1 - p0) * ((t - t0) / (t1 - t0))


def _rewrite_siren(src: dict, mic, rng: np.random.Generator) -> dict:
    v, dirx, yw, z = src["speed_mps"], src["dir_x"], src["wp"][0][2], src["wp"][0][3]
    u = rng.random()
    if u < SIREN_MIX["near"]:
        t_cpa = float(src.get("t_cpa_s", rng.uniform(*m9.WARN_TCPA)))   # 近接通過（旧来の幾何・鳴りっぱなし）
        kind = "near"
    elif u < SIREN_MIX["near"] + SIREN_MIX["far"]:
        d0 = float(np.exp(rng.uniform(np.log(SIREN_START_DIST[0]), np.log(SIREN_START_DIST[1]))))
        t_cpa = float(np.sqrt(max(d0 * d0 - yw * yw, 1.0)) / v)
        kind = "far"
    else:
        t_cpa = float(rng.uniform(*SIREN_TCPA_RECEDE))
        kind = "recede"
    xr = float(m9._mic_pos_at(mic, float(np.clip(t_cpa, 0.0, m9.CLIP)))[0])
    x0 = xr - dirx * v * t_cpa
    src = dict(src)
    src["wp"] = [[0.0, x0, yw, z], [m9.CLIP, x0 + dirx * v * m9.CLIP, yw, z]]
    src["t_on"], src["t_off"] = 0.0, m9.CLIP
    src["t_cpa_s"] = round(t_cpa, 3)
    d_start = float(np.hypot(v * t_cpa, yw))
    src["start_dist_m"] = round(d_start, 1)
    src["preroll_s"] = float(np.ceil(d_start / m9.sound_speed(m9.TEMP_C)) + 1.0)
    src["grammar"] = "v13_continuous"
    src["siren_kind"] = kind
    return src


def _short_burst(src: dict, rng: np.random.Generator, dur: float, tag: str) -> dict:
    t_cpa = float(src["t_cpa_s"])
    t_on = float(np.clip(t_cpa - rng.uniform(*NEAR_CPA_LEAD), 0.3, m9.CLIP - dur - 0.3))
    src = dict(src)
    src["t_on"], src["t_off"] = round(t_on, 3), round(t_on + dur, 3)
    src["grammar"] = tag
    return src


def _rewrite_horn(src: dict, rng: np.random.Generator) -> dict:
    n = int(rng.choice([1, 2, 3], p=HORN_N_P))
    honk = float(np.exp(rng.uniform(np.log(HORN_SEC[0]), np.log(HORN_SEC[1]))))
    gap = float(rng.uniform(*HORN_GAP))
    src = dict(src)
    src["params"] = dict(src["params"], honk_sec=honk, gap_sec=gap)
    src["n_honk"] = n
    return _short_burst(src, rng, n * honk + (n - 1) * gap, "v13_short")


def _rewrite_bell(src: dict, rng: np.random.Generator) -> dict:
    n = int(rng.choice([1, 2, 3], p=BELL_N_P))
    p = src["params"]
    if p.get("bell_type") == "ring":                 # 引き打ち: burst+gap の周期
        dur = n * p["burst_sec"] + (n - 1) * p["gap_sec"]
    else:                                            # 打鐘: repeat_period ごとに1チリン
        dur = (n - 1) * p["repeat_period_sec"] + p["ring_gap_sec"] + 0.5
    src = dict(src)
    src["n_ring"] = n
    return _short_burst(src, rng, dur, "v13_short")


def _rewrite_beep(src: dict) -> dict:
    src = dict(src)
    src["t_on"], src["t_off"] = 0.0, m9.CLIP
    src["grammar"] = "v13_continuous"
    return src


def _attach_rain(s: dict, row: dict) -> dict:
    if row.get("rain"):
        s["rain"] = {"kind": row["rain"], "rate": float(row["rain_rate"]),
                     "dba": float(row["rain_dba"]), "seed": int(row["seed"]) * 7919 + 17}
    else:
        s["rain"] = None
    s["grammar"] = "v12" if GRAMMAR_OFF else "v13"
    return s


def sample_v13core(row: dict, rng: np.random.Generator) -> dict:
    s = m11v.sample_v11core(row, rng)           # 元シーン（同一seed・同一消費順）
    if GRAMMAR_OFF:
        return s
    rng2 = np.random.default_rng([int(row["seed"]), REWRITE_SALT])
    mic = s["_mic_arr"]
    out = []
    for src in s["sources"]:
        cls = src["class"]
        if src["kind"] != "vehicle" or cls not in WARN_GATED:
            out.append(src)
        elif cls == "siren":
            out.append(_rewrite_siren(src, mic, rng2))
        elif cls == "horn":
            out.append(_rewrite_horn(src, rng2))
        elif cls == "bike_bell":
            out.append(_rewrite_bell(src, rng2))
        else:
            out.append(_rewrite_beep(src))
    s["sources"] = out
    return s


def sample_scene_v13(row: dict) -> dict:
    scen = row.get("scenario")
    if scen == "v11core":
        salt = m11v._PEAK_SALT
        rng = (np.random.default_rng(row["seed"]) if salt == 0
               else np.random.default_rng([row["seed"], 20260727, salt]))
        s = sample_v13core(row, rng)
        if salt:
            s["peak_resample_salt"] = salt
    else:
        fn = v12._V12_SAMPLERS.get(scen)
        assert fn is not None, f"v13で未対応のscenario: {scen}"
        s = fn(row)
    return _attach_rain(s, row)


def load_plan_v13() -> list:
    rows = []
    with open(PLAN_V13 / "assignment_v13.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n_warnings"] = int(r["n_warnings"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


# ---- プリロール付きステムレンダ（S1） ---------------------------------------------
def _extend_back(wp: np.ndarray, pre: float) -> np.ndarray:
    """2点直線軌道を t=−pre まで直線外挿して延長（シフトなし）。"""
    p = _lin_extrap(wp, -pre)
    return np.vstack([[-pre, *p], wp])


def _shift(wp: np.ndarray, pre: float) -> np.ndarray:
    w = np.array(wp, dtype=np.float64).copy()
    w[:, 0] += pre
    return w


def _render_stem_preroll(dry_cal_ext: np.ndarray, wp_ext: np.ndarray, mic, c: float,
                         pre: float):
    """m9._render_stem と同じ物理で、時間軸を pre 秒前に伸ばしてレンダし末尾 CLIP 秒を返す。

    wp_ext: t=−pre から始まる（シフト前の）音源軌道。mic: 静止(3,) または (M,4) 軌道。
    """
    total = m9.CLIP + pre
    n_all = int(round(total * m9.FS_OUT))
    n_pre = n_all - int(m9.CLIP * m9.FS_OUT)
    tr = np.arange(n_all) / m9.FS_OUT
    mic_arr = np.asarray(mic, dtype=np.float64)
    if mic_arr.ndim == 2:
        mic_arr = _shift(_extend_back(mic_arr, pre), pre)
    wp_s = _shift(wp_ext, pre)
    foas = {}
    tags = ("direct", "mirror") if m9.ablate.ground_enabled() else ("direct",)
    for tag in tags:
        w = wp_s.copy()
        if tag == "mirror":
            w[:, 3] = 2.0 * m9.GROUND_Z - w[:, 3]
        mono48 = m9.render_mono(dry_cal_ext, w, mic_arr, m9.FS_SIM, total,
                                temperature_c=m9.TEMP_C, pressure_atm=m9.PRESS_ATM,
                                rel_humidity=m9.RH, **m9.ablate.render_flags())
        mono24 = m9.decimate_to_out_rate(mono48, m9.FS_SIM, m9.FS_OUT)
        te, ps = m9.solve_emission_times(tr, w, mic_arr, c)
        mic_at = (m9.receiver_positions_at(tr, mic_arr) if mic_arr.ndim == 2 else mic_arr)
        u, _ = m9.doa_unit_vectors(ps, mic_at)
        if m9.MIC_HEADING:                                # 先行分 §1: 伸ばした経路の ψ（伸ばした所は最初の区間の向き）
            u = m9.rotate_to_mic_frame(u, m9.mic_heading_deg(mic_arr, tr))
        foas[tag] = m9.encode_foa_timevarying(mono24, u)[:, n_pre:]
    if not m9.ablate.ground_enabled():
        return foas["direct"], foas["direct"]
    return foas["direct"], foas["direct"] + foas["mirror"]


def _make_dry_len(src: dict, seconds: float) -> np.ndarray:
    """長さ seconds のドライ音（m9.CLIP を一時的に差し替えて既存生成器を呼ぶ）。"""
    keep = m9.CLIP
    m9.CLIP = float(seconds)
    try:
        return m9._make_dry(src)
    finally:
        m9.CLIP = keep


# ---- generate_clip（v12複製＋プリロール・雨・警告音ゲート） --------------------------
def generate_clip_v13(row: dict) -> None:
    t_start = time.perf_counter()
    s = sample_scene_v13(row)
    mic = s.pop("_mic_arr")
    if m9.MIC_HEADING:                                    # 先行分 §1: マイクの進む向き（数か [[t0, ψ], …]）。walk_dir_x は残す
        s["mic"]["mic_heading_deg"] = m9.mic_heading_record(mic)
    name = row["clip_id"]
    wdir = m9.WORK / name
    wdir.mkdir(parents=True, exist_ok=True)
    for sub in ("foa", "metadata", "masks"):
        (m9.DS / sub).mkdir(parents=True, exist_ok=True)
    c = m9.sound_speed(m9.TEMP_C)
    n24 = int(m9.CLIP * m9.FS_OUT)
    tgrid = np.arange(0.0, m9.CLIP, 0.01)

    stems = []
    for src in s["sources"]:
        pre = float(src.get("preroll_s", 0.0))
        wp = np.array(src["wp"], float)
        if pre > 0:
            dry = _make_dry_len(src, m9.CLIP + pre)          # 発音窓なし（全体鳴動）
            g = m9.gain_for_spl_a(dry, m9.FS_SIM, src["l1m_db"])
            src["dry_gain"] = g
            wp_ext = _extend_back(wp, pre)
            stem_d, stem_wr = _render_stem_preroll(dry * g, wp_ext, mic, c, pre)
            wp_lab, act_from = wp_ext, -pre
        else:
            dry = m9._window(m9._make_dry(src), src["t_on"], src["t_off"])
            a0, a1 = int(src["t_on"] * m9.FS_SIM), int(src["t_off"] * m9.FS_SIM)
            g = m9.gain_for_spl_a(dry[a0:a1], m9.FS_SIM, src["l1m_db"])
            src["dry_gain"] = g
            stem_d, stem_wr = m9._render_stem(dry * g, wp, mic, c)
            wp_lab, act_from = wp, src["t_on"]
        dist = m9._dist_series(wp_lab, mic, tgrid)
        src["min_dist_m"] = round(float(np.min(dist)), 3)
        src["t_min_dist_s"] = round(float(tgrid[int(np.argmin(dist))]), 3)
        if m9.MIC_HEADING:                                # 先行分 §1: マイク座標の左右（plan の car_side などは世界座標のまま）
            src["side_mic"] = m9.side_mic_of(wp_lab, mic, src["t_min_dist_s"], c)
        if src["class"] == "car_drive" and src.get("track", 0) == 0:
            s["cpa_rel_dist_m"] = src["min_dist_m"]
            s["cpa_rel_time_s"] = src["t_min_dist_s"]
        win = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
        p_geo = float(np.mean(1.0 / np.maximum(dist[win], 1e-9) ** 2))
        src["recv_pred_db"] = round(src["l1m_db"] + 10.0 * np.log10(p_geo), 2)
        b0, b1 = int(src["t_on"] * m9.FS_OUT), max(int(src["t_off"] * m9.FS_OUT), 1)
        src["recv_meas_db"] = round(m9.spl_a(stem_d[0, b0:b1], m9.FS_OUT), 2)
        src["recv_withrefl_db"] = round(m9.spl_a(stem_wr[0, b0:b1], m9.FS_OUT), 2)
        stems.append((src, stem_d, stem_wr, wp_lab, act_from))

    rng_n = np.random.default_rng(s["noise"]["seed"])
    if NOISE_SHAPE_V30 is None:
        noise = m9.diffuse_foa_noise(n24, m9.FS_OUT, rng_n)
    else:                                                    # 本体 §2-3（v30 の入口の中だけ）: 形だけ本ごとに変える（大きさは下の dB(A) 合わせのまま）
        from outdoor_seld.noise_v30 import diffuse_foa_shaped_noise
        _shape = NOISE_SHAPE_V30(s, row)
        noise = diffuse_foa_shaped_noise(n24, m9.FS_OUT, rng_n, _shape["knee_hz"], _shape["slope_db_oct"])
        s["noise"]["shape"] = {"knee_hz": round(float(_shape["knee_hz"]), 3), "slope_db_oct": round(float(_shape["slope_db_oct"]), 4)}
        for _k in ("knee_range_hz", "slope_range_db_oct"):     # 決め直し 5（§9-7）: 引いた幅も残す
            if _k in _shape:
                s["noise"]["shape"][_k] = [float(x) for x in _shape[_k]]
    g_n = m9.gain_for_spl_a(noise[0], m9.FS_OUT, s["noise"]["dba"])
    noise = noise * g_n
    s["noise"]["gain"] = float(g_n)
    if s.get("rain"):                                        # R1: 雨は暗騒音の一部として加算
        rr = s["rain"]
        rf = diffuse_foa_rain(n24, m9.FS_OUT, np.random.default_rng(rr["seed"]), rr["rate"])
        g_r = m9.gain_for_spl_a(rf[0], m9.FS_OUT, rr["dba"])
        rr["gain"] = float(g_r)
        noise = noise + rf * g_r
        s["noise"]["dba_with_rain"] = round(m9.spl_a(noise[0], m9.FS_OUT) + v12.V12_HEADROOM_DB, 2)

    mix = noise.copy()
    for _, _, stem_wr, _, _ in stems:
        mix = mix + stem_wr
    peak = float(np.max(np.abs(mix)))
    assert peak < m9.PEAK_MAX, f"{name}: peak {peak:.3f} >= {m9.PEAK_MAX}"

    fr_noise = m9.frame_spl_a(noise[0], m9.FS_OUT)
    gate = GATE_V30                                           # 本体 §2-1（v30 の入口の中だけ・None = 旧の道）
    line_db = m9.AUDIBLE_SNR_DB if gate is None else float(gate["line_db"])
    band_db = None if gate is None else gate.get("band_db")   # 帯域ゲート（v30 の旗 FEATURES["band_gate"]・Fable 10/26 18:3x の下ごしらえ・None = 広帯域の線のまま）
    if gate is not None:
        s["gate_rule"], s["gate_line_db"], s["gate_bg"] = "v30", line_db, "noise+others"
        if band_db is not None:                               # 判定量 = 一番有利な 1/3 オクターブ帯の SN 比（outdoor_seld.band_gate）・広帯域の値は masks に併記
            from outdoor_seld import band_gate as _band_gate
            s["gate_rule"], s["gate_line_db"] = "v30_band", float(band_db)
            s["gate_broadband_line_db"], s["gate_judge"] = line_db, "best 1/3-octave band SNR (125 Hz-10 kHz, 0.1 s, no window)"
            line_db = float(band_db)
            # §9-6 の 1（Astra R4-30-01）: 来歴 = band_gate.py の内容 hash・帯の線・旗・枠長・帯と床の約束
            s["gate_band"] = {"code_sha256": _band_gate.code_sha256(), "line_db": line_db, "flag": "step11_v30_render.FEATURES['band_gate']", **_band_gate.CONTRACT}
    label_rows, mask_rows = [], []
    for _src_i, (src, _, stem_wr, wp_lab, act_from) in enumerate(stems):
        cls_idx = v12.CLASS_IDX_V12[src["class"]]
        fr_stem = m9.frame_spl_a(stem_wr[0], m9.FS_OUT)
        snr = fr_stem - fr_noise
        fr_bg = fr_noise
        if gate is not None:
            snr_bg = snr                                      # 旧の量（暗騒音＋雨だけ）→ masks の snr_a_bg_db
            fr_bg = m9.frame_spl_a((mix - stem_wr)[0], m9.FS_OUT)   # 背景 = 暗騒音＋雨＋同じ場面の自分以外の全 stem（ラベル無しの電車も）
            snr = fr_stem - fr_bg
        judge = snr                                           # ゲートの判定量（帯域ゲートのときだけ帯の SN 比に替わる・切のときは今までと同じ値）
        if band_db is not None:
            # §9-6 の 1: 非有限は band_gate が止める（BandGateError）。全帯が無効な枠（stem も背景も床以下）は valid = False ＝ゲート不通過
            band_best, band_hz, band_valid = _band_gate.band_snr_frames(stem_wr[0], (mix - stem_wr)[0], m9.FS_OUT)
            assert len(band_best) == len(snr), (len(band_best), len(snr))
            judge = np.where(band_valid, band_best, -np.inf)  # 判定は未丸めの値（無効な枠は通らない）
            _vk = np.flatnonzero(band_valid)
            _kb = int(_vk[np.argmax(band_best[_vk])]) if len(_vk) else None
            src["gate_band_max_db"] = round(float(band_best[_kb]), 2) if _kb is not None else None
            src["gate_band_hz_at_max"] = round(float(band_hz[_kb]), 1) if _kb is not None else None
            src["gate_band_n_invalid"] = int(len(band_valid) - len(_vk))
        # 2026-10-19（Astra RR70-04）: ゲートの判定量を残す（音源ごと・最大 SNR のフレームの受音量と背景〔暗騒音＋雨・v30 は他の音源も〕・音は変えない）
        _k = int(np.argmax(snr)) if len(snr) else 0
        src["gate_snr_max_db"] = round(float(snr[_k]), 2) if len(snr) else None
        src["gate_recv_max_dba"] = round(float(fr_stem[_k]), 2) if len(snr) else None
        src["gate_bg_dba_at_max"] = round(float(fr_bg[_k]), 2) if len(snr) else None
        src["gate_bg_dba_median"] = round(float(np.median(fr_bg)), 2)
        if gate is None:
            mask_rows += [(k, cls_idx, round(float(snr[k]), 2)) for k in range(len(snr))]
        else:
            _trk = int(src.get("track", 0) or 0)
            mask_rows += [(k, cls_idx, _trk, _src_i, round(float(snr[k]), 2), round(float(snr_bg[k]), 2))
                          + ((round(float(band_best[k]), 2), round(float(band_hz[k]), 1), int(band_valid[k])) if band_db is not None else ())
                          for k in range(len(snr))]   # src = scene の音源の番号（R30-10・同じ class・track の電車の車両も分かれる）・帯域ゲートのときは 3 列足す（無効な枠は −999.0・0.0・0）
        if src.get("no_label"):
            src["audible_frac"] = round(float(np.mean(judge >= line_db)), 3)
            src["n_label_frames"] = 0
            continue
        rows, _ = m9.frame_label_rows(np.array(wp_lab, float), mic,
                                      clip_len_sec=m9.CLIP, class_idx=cls_idx,
                                      track_idx=src.get("track", 0),
                                      source_active_from=act_from,
                                      source_active_until=src["t_off"], c=c,
                                      heading_deg=m9.label_heading(mic))   # 先行分 §1（切なら None＝今まで）
        gated = (VEHICLE_GATED_V13 if not GRAMMAR_OFF else v12.VEHICLE_GATED) if gate is None else gate["gated"]
        if src["class"] in gated:
            audible = m9._fill_gaps(judge >= line_db, m9.GAP_FILL)
            rows = [r for r in rows if audible[r[0]]]
            src["audible_frac"] = round(float(np.mean(judge >= line_db)), 3)
            if gate is None:
                aud_idx = np.where(snr >= line_db)[0]
                src["t_first_audible_s"] = (round(float(aud_idx[0]) * 0.1, 2)
                                            if len(aud_idx) else None)
            else:                                              # 本体 §2-2: 鳴り始め = 最初に正解の付いた枠
                src["t_first_audible_s"] = round(float(min(r[0] for r in rows)) * 0.1, 2) if rows else None
        else:
            k0, k1 = int(src["t_on"] * 10), max(int(src["t_off"] * 10), 1)
            src["audible_frac"] = round(float(np.mean(judge[k0:k1] >= line_db)), 3)
        src["n_label_frames"] = len(rows)
        label_rows += rows
    label_rows.sort(key=lambda r: (r[0], r[1]))

    for i, (src, stem_d, stem_wr, _, _) in enumerate(stems):
        tag = f"src{i}_{src['class']}"
        sf.write(wdir / f"{tag}_direct_24k.flac", stem_d.T.astype(np.float32), m9.FS_OUT,
                 subtype="PCM_24")
        sf.write(wdir / f"{tag}_withrefl_24k.flac", stem_wr.T.astype(np.float32), m9.FS_OUT,
                 subtype="PCM_24")
    sf.write(m9.DS / "foa" / f"{name}.flac", mix.T.astype(np.float32), m9.FS_OUT,
             subtype="PCM_24")
    m9.write_dcase_csv(m9.DS / "metadata" / f"{name}.csv", label_rows)
    with open(m9.DS / "masks" / f"{name}.csv", "w", newline="\n") as f:
        if gate is None:
            f.write("frame,class,snr_a_db\n")
            for k, ci, v in mask_rows:
                f.write(f"{k},{ci},{v}\n")
        elif band_db is None:                                  # 本体 §2-1: snr_a_db = 新しい定義（他の音源込み）・snr_a_bg_db = 旧の量（暗騒音＋雨）
            f.write("frame,class,track,src,snr_a_db,snr_a_bg_db\n")
            for k, ci, trk, si, v, vb in mask_rows:
                f.write(f"{k},{ci},{trk},{si},{v},{vb}\n")
        else:                                                  # 帯域ゲート: 9 列（snr_band_max_db = 一番有利な帯の SN 比 = 判定量〔小数 2 桁〕・band_hz = その帯の中心・band_valid = 有効な帯がある枠 1／全帯が無効 0）
            f.write("frame,class,track,src,snr_a_db,snr_a_bg_db,snr_band_max_db,band_hz,band_valid\n")
            for k, ci, trk, si, v, vb, bm, bh, bv in mask_rows:
                f.write(f"{k},{ci},{trk},{si},{v},{vb},{bm},{bh},{bv}\n")

    s["stats"] = {"peak": round(peak, 4),
                  "mix_spl_a": round(m9.spl_a(mix[0], m9.FS_OUT), 2),
                  "n_label_rows": len(label_rows),
                  "gen_seconds": round(time.perf_counter() - t_start, 1)}
    (wdir / "scene.json").write_text(json.dumps(s, indent=2, default=str))
    srcs = "+".join(x["class"] for x in s["sources"]) or "none"
    rain = f" rain={s['rain']['kind']}{s['rain']['dba']:.0f}" if s.get("rain") else ""
    print(f"[gen13] {name} {row['motion']} {srcs} noise={s['noise']['dba']:.1f}dBA{rain} "
          f"peak={peak:.3f} ({s['stats']['gen_seconds']:.0f}s)", flush=True)


def generate_clip(row: dict) -> None:
    """ピーク超過時のみ salt=1,2,... で引き直す決定論ラダー（v11と同方式）。"""
    if row.get("scenario") != "v11core":
        return generate_clip_v13(row)
    last = None
    for salt in range(9):
        m11v._PEAK_SALT = salt
        try:
            return generate_clip_v13(row)
        except AssertionError as e:
            if "peak" not in str(e):
                raise
            last = e
            print(f"[peak-resample] {row['clip_id']} salt={salt} NG ({e}) -> retry", flush=True)
        finally:
            m11v._PEAK_SALT = 0
    raise last


# ---- 検証・煙試験 ----------------------------------------------------------------
def verify() -> None:
    """回帰: 修正OFF＋雨なし＋元motion で v12 の既存クリップ（val_syn_clip00002・旧 fold2_room1_mix0002）とビット一致。"""
    global GRAMMAR_OFF
    import hashlib
    clip = "val_syn_clip00002"          # 2026-09-22 改名

    ref = ROOT / "out/dataset_outdoor_siren_v12"
    row = [r for r in v12.m9.load_plan("core") if r["clip_id"] == clip][0]
    row = dict(row, rain="", rain_rate="", rain_dba="", grammar="v12")
    GRAMMAR_OFF = True
    try:
        generate_clip(row)
    finally:
        GRAMMAR_OFF = False
    for sub, ext in (("foa", ".flac"), ("metadata", ".csv"), ("masks", ".csv")):
        a = hashlib.sha256((m9.DS / sub / f"{clip}{ext}").read_bytes()).hexdigest()
        b = hashlib.sha256((ref / sub / f"{clip}{ext}").read_bytes()).hexdigest()
        print(f"  {sub}: {'一致' if a == b else '不一致 ❌'}")
        assert a == b, f"回帰NG: {sub}"
    print("verify PASS: 修正OFFのv13生成は v12 とビット一致（複製にドリフトなし）")


def smoke() -> None:
    """各修正を1本ずつ: siren(遠方) / horn / bell / beep / 雨あり / v12ext雨あり。"""
    rows = load_plan_v13()
    picks = {}
    for r in rows:
        if r["split"] != "fold2":
            continue
        key = None
        if r["scenario"] == "v11core" and r["n_car"] == "1" and r["w2_class"] == "":
            if r["w1_class"] in ("siren", "horn", "bike_bell", "backup_beep") and not r["rain"]:
                key = r["w1_class"]
            elif r["w1_class"] == "" and r["rain"]:
                key = f"rain_{r['rain']}"
        elif r["scenario"] == "v12kick" and r["rain"]:
            key = "kick_rain"
        if key and key not in picks:
            picks[key] = r
    for key, r in picks.items():
        generate_clip(r)
        s = json.loads((m9.WORK / r["clip_id"] / "scene.json").read_text())
        for x in s["sources"]:
            if x["class"] in WARN_GATED:
                print(f"   [{key}] {r['clip_id']} {x['class']} t_on={x['t_on']} t_off={x['t_off']} "
                      f"start={x.get('start_dist_m')}m pre={x.get('preroll_s')} "
                      f"min={x.get('min_dist_m')}m first_aud={x.get('t_first_audible_s')} "
                      f"audible={x.get('audible_frac')} labels={x.get('n_label_frames')}")
        if s.get("rain"):
            print(f"   [{key}] {r['clip_id']} rain={s['rain']} noise={s['noise']}")


if __name__ == "__main__":
    if "verify" in sys.argv:
        verify()
    elif "smoke" in sys.argv:
        smoke()
