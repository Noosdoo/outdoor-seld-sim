# -*- coding: utf-8 -*-
"""step11_v27_render.py — 組み合わせの場面 v19mix（変更票 #64・D31・2026-10-13・本人「全部の組み合わせのものを網羅できるように」）。

9 クラス（車・自転車・キック・バイク・電車（背景）・サイレン・クラクション・バック音・踏切の警報機・ベル）の 2 つ組と 3 つ組を、
**今ある作り方をそのまま 1 つの場面に重ねる**だけで作る。音の作り方・大きさ・鳴動の規則・ラベルの規則は変えない。
  - 車と警告音（≤2）: 土台 v11core のサンプラ（v16r.sample_scene_v15 → v14 → v13 → v11）を部分場面として引く。警告音が 3 つなら
    2 つ目の部分場面（車 0 台・警告音 1 つ）をもう 1 回引く。
  - 自転車: step11_v22_render._bicycle_src（v19bicycle と同じ）。キック: v12kick（近い 0.8〜2.5 m）と v19kickfar（遠い 2.5〜8 m）を半々。
    バイク: v12bike。電車: v12train（車両は **D14 のとおり背景＝ラベルなし**・2 編成目は 4 m 外側＝複線の間隔）。
  - マイクは 1 つを全部の部分場面で共有する（部分サンプラのマイク抽選を差し替える。乱数の消費は元と同じ回数）。
  - ラベルの track はクラスごとに 0,1,2（ADPIT の上限 3）。音源は最大 3 つ（電車の車両は背景）なので上限に当たらないが、当たれば no_label にして記録する。
  - 部分場面の種は本の種から引く（ピーク再抽選の salt で全部変わる＝v19 の梯子のまま）。
plan の列: mix（例 "car+bicycle+siren"・TOKENS の順）・n_car・n_warnings（0〜3）・w1〜w3_class／_side（side は部分場面に渡す）。
登録: v19r._SAMPLERS19["v19mix"]（v21／v22 と同じ・「v19」始まりなので v19 の generate_clip の梯子を通る）。
検査: python scripts/step11_v27_render.py --selftest   /   --check-plan [plan.csv]   /   --preview N
"""
from __future__ import annotations

import contextlib
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v26_render as v26  # noqa: E402,F401  v26 までの連なり（v23 ドライ音・v24 大きさ・v25 ベルの時刻・v26 サイレンと通報音）を先に読む
import step11_v22_render as v22  # noqa: E402
import step11_v21_render as v21  # noqa: E402
import step11_v19_render as v19r  # noqa: E402

m9 = v19r.m9
v13 = v19r.v13
v12 = v19r.v12
v16r = v19r.v16r
m11v = v13.m11v

SCENARIO = "v19mix"
MOBILE = ("car", "bicycle", "kick", "moto", "train")
WARN = ("siren", "horn", "backup_beep", "crossing", "bike_bell")
TOKENS = MOBILE + WARN
CLASS_OF = {"car": "car_drive", "bicycle": "bicycle", "kick": "kickboard", "moto": "motorcycle", "train": "crossing",
            "siren": "siren", "horn": "horn", "backup_beep": "backup_beep", "crossing": "crossing", "bike_bell": "bike_bell"}
MAX_SOURCES = 3            # 2 つ組と 3 つ組
MAX_TRAIN = 2              # 電車 3 編成同時は複線でもありえない（本人 10/13 12:46 は 2 編成を残す）
MAX_WARN = 3               # 警告音 3 つ（本人 10/13 12:46「一応残そう」）。plan の列は w1〜w3
MAX_TRACKS = 3             # ADPIT のクラスあたりの track 上限
TRAIN2_Y_SHIFT_M = 4.0     # 2 編成目は 4 m 外側（複線の線路の間隔の目安）
KICK_P_FAR = 0.5           # 組み合わせの中のキック: 遠い横距離（v19kickfar）の割合 = 半々（単独の本は近い 2,250：遠い 1,500 = 60:40。遠い側を少し厚くした・Opus 10/13 D-3 で表記を直した）
MOTO_P_FAR = 0.5           # 組み合わせの中のバイク: 遠い横距離（v19motofar）の割合（半々）
EXCLUDE = (("train", "backup_beep"),)   # 一緒に置かない組（本人 10/13 12:46「電車とバック音は除いていい」）
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"

# ---- v19motofar（D32・本人 2026-10-13 13:09「バイクの遠い横距離は足してください」）: v12bike の規則のまま横距離だけ遠くする ----
SCEN_MOTOFAR = "v19motofar"
MOTOFAR_LAT = (4.0, 12.0)                    # 横距離 [m]・対数一様（v12bike の 1.5〜4.0 の続き。車線幅 3.0〜3.5 m・歩道 2 m → 2 本目の車線 ≈ 6 m・3 本目 ≈ 9.5 m・4 本目 ≈ 13 m）
MOTOFAR_P_MOPED = 0.4                        # 原付の割合（v12bike と同じ）
MOTOFAR_SPEED_MOPED = (5.6, 8.3)             # [m/s]（法定 30 km/h・v12bike と同じ）
MOTOFAR_SPEED_MOTO = (8.3, 16.7)             # [m/s]（30〜60 km/h・v12bike と同じ）
MOTOFAR_LAW_MOPED = (70.0, 79.0)             # 法規値 dB(A)@7.5 m（v12bike と同じ）
MOTOFAR_LAW_MOTO = (72.0, 82.0)
MOTOFAR_TCPA = (3.0, 8.0)                    # 最接近の時刻 [s]（v12bike と同じ）
MOTOFAR_Z = 1.5                              # 旧の音源の高さ [m]（記録用）。実際の値は m9.src_z_fixed("motorcycle") = 既定 1.5・v29 は 0.05〜0.6 m（#70・点検 K4）
MOTOFAR_REF_M = 7.5                          # 加速走行騒音規制の測定距離 → 1 m 換算（v12bike と同じ式）


def _moto_far_src(row: dict, rng: np.random.Generator, mic_arr, grammar: str) -> tuple[dict, dict]:
    """遠いバイク 1 台。v12bike と同じ速さ・音量・区分で、横距離だけ 4〜12 m（対数一様）。最接近時のマイク位置を基準に置く（v19 の車・kickfar と同じ）。"""
    sign = v19r._side(row, rng)
    lat = float(np.exp(rng.uniform(np.log(MOTOFAR_LAT[0]), np.log(MOTOFAR_LAT[1]))))
    if rng.random() < MOTOFAR_P_MOPED:
        ec, speed, law = "moped", float(rng.uniform(*MOTOFAR_SPEED_MOPED)), float(rng.uniform(*MOTOFAR_LAW_MOPED))
    else:
        ec, speed, law = "motorcycle", float(rng.uniform(*MOTOFAR_SPEED_MOTO)), float(rng.uniform(*MOTOFAR_LAW_MOTO))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    t_cpa = float(rng.uniform(*MOTOFAR_TCPA))
    y = lat * sign
    x0 = float(m9._mic_pos_at(mic_arr, t_cpa)[0]) - dirx * speed * t_cpa
    l1m = law + 20.0 * float(np.log10(MOTOFAR_REF_M))
    zm = m9.src_z_fixed("motorcycle", row)                                       # #70: 旧 MOTOFAR_Z 1.5・v29 は 0.05〜0.6 m
    src = {"class": "motorcycle", "kind": "vehicle", "track": 0,
           "wp": [[0.0, x0, y, zm], [m9.CLIP, x0 + dirx * speed * m9.CLIP, y, zm]],
           "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": speed, "dir_x": dirx,
           "t_cpa_s": t_cpa, "law_db": law, "l1m_db": l1m, "lateral_m": lat, "grammar": grammar,
           "params": {"audio_seed": int(row["seed"]) * 7 + 23, "engine_class": ec, "speed_mps": speed}}   # audio_seed は v12bike と同じ式
    info = {"lateral_m": round(lat, 2), "side": "L" if sign > 0 else "R", "engine_class": ec, "speed_kmh": round(speed * 3.6, 1),
            "dir_x": dirx, "t_cpa_s": round(t_cpa, 2), "law_db": round(law, 1)}
    return src, info


def sample_v19motofar(row: dict) -> dict:
    rng = v19r._rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)
    src, info = _moto_far_src(row, rng, mic_arr, "v27_motofar")
    return v19r._finish(row, rng, mic_arr, mic, [src], dict(kind="motofar", **info))


def check_motofar(s: dict) -> None:
    srcs = s["sources"]
    assert len(srcs) == 1 and srcs[0]["class"] == "motorcycle", [x["class"] for x in srcs]
    x = srcs[0]; wp = np.array(x["wp"], float)
    lat = abs(float(wp[0, 2]))
    assert MOTOFAR_LAT[0] - 1e-9 <= lat <= MOTOFAR_LAT[1] + 1e-9 and abs(lat - float(x["lateral_m"])) < 1e-9, lat
    zlo, zhi = m9.src_z_range("motorcycle")
    assert wp[0, 2] == wp[1, 2] and wp[0, 3] == wp[1, 3] and zlo - 1e-9 <= wp[0, 3] <= zhi + 1e-9, wp.tolist()   # #70: 高さはクラスの範囲
    ec = x["params"]["engine_class"]; v = float(x["speed_mps"])
    lo, hi = MOTOFAR_SPEED_MOPED if ec == "moped" else MOTOFAR_SPEED_MOTO
    assert ec in ("moped", "motorcycle") and lo - 1e-9 <= v <= hi + 1e-9, (ec, v)
    assert MOTOFAR_TCPA[0] <= float(x["t_cpa_s"]) <= MOTOFAR_TCPA[1], x["t_cpa_s"]
    t = float(x["t_cpa_s"]); mic = np.asarray(s["_mic_arr"], float)
    xs = float(np.interp(t, wp[:, 0], wp[:, 1])); xm = float(m9._mic_pos_at(mic, t)[0])
    assert abs(xs - xm) < 1e-6, f"最接近がマイクとずれる {xs} vs {xm}"
    assert x["params"]["audio_seed"] == int(s["row"]["seed"]) * 7 + 23

_ORIG_MAKE_MIC_11 = m11v._make_mic
_ORIG_MIC_SAMPLE_12 = v12._mic_sample


# ---------------------------------------------------------------- 組み合わせの文字列
def parse_mix(mix: str) -> Counter:
    toks = [t for t in str(mix).split("+") if t]
    bad = [t for t in toks if t not in TOKENS]
    assert not bad, f"mix に知らない札: {bad}（使えるのは {TOKENS}）"
    return Counter(toks)


def mix_string(c) -> str:
    c = Counter(c)
    return "+".join(t for t in TOKENS for _ in range(c[t]))


def mix_ok(c) -> str:
    """規則に合わない理由（空なら合格）。plan を作る側（step10_v27_plan）と同じ規則。"""
    c = Counter(c)
    n = sum(c.values())
    if not (2 <= n <= MAX_SOURCES):
        return f"音源の数 {n}（2〜{MAX_SOURCES}）"
    if c["train"] > MAX_TRAIN:
        return f"電車 {c['train']} 編成（上限 {MAX_TRAIN}）"
    if c["car"] > 3:
        return f"車 {c['car']} 台（v11core の上限 3）"
    if sum(c[w] for w in WARN) > MAX_WARN:
        return f"警告音 {sum(c[w] for w in WARN)}（上限 {MAX_WARN}）"
    for a, b in EXCLUDE:
        if c[a] and c[b]:
            return f"{a}+{b} は置かない"
    return ""


# ---------------------------------------------------------------- 部分場面の道具
def _draw_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(1, 2**31 - 1))


def _sub_row(row: dict, seed: int, tag: str, **over) -> dict:
    """部分場面に渡す plan 行（土台の列をそろえる）。scenario は v11core（車と警告音）か v12 の場面名。"""
    r = {"clip_id": f"{row['clip_id']}__{tag}", "split": row.get("split", "train"), "motion": row["motion"],
         "n_warnings": 0, "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "",
         "danger_tier": row.get("danger_tier") or "safe", "car_side": row.get("car_side") or "L",
         "scenario": "v11core", "seed": int(seed), "scene_type": "v19", "n_car": "0",
         "rain": "", "rain_rate": "", "rain_dba": "", "grammar": "v13",
         "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "",
         "mic_z": row.get("mic_z", "1.5"), "h_copy": row.get("h_copy", "1")}
    r.update(over)
    return r


@contextlib.contextmanager
def _shared_mic(mic_arr, mic_rec: dict):
    """部分サンプラのマイク抽選を差し替えて、本のマイク（同じ軌道）を返す。乱数の消費回数は元の関数と同じにする。"""
    def fake11(row, rng):                       # step11_v11_render._make_mic: uniform → random
        if row["motion"] != "static":
            rng.uniform(*m9.WALK_SPEED); rng.random()
        return np.asarray(mic_arr, float).copy(), dict(mic_rec)

    def fake12(row, rng):                       # step11_v12_render._mic_sample: uniform → choice
        if row["motion"] == "walk":
            rng.uniform(*m9.WALK_SPEED); rng.choice([-1.0, 1.0])              # #70 点検 K2: v12._mic_sample と同じ範囲を読む
        return np.asarray(mic_arr, float).copy(), dict(mic_rec)
    assert m11v._make_mic is _ORIG_MAKE_MIC_11 and v12._mic_sample is _ORIG_MIC_SAMPLE_12, "マイク抽選が差し替わったまま（入れ子）"
    m11v._make_mic = fake11
    v12._mic_sample = fake12
    try:
        yield
    finally:
        m11v._make_mic = _ORIG_MAKE_MIC_11
        v12._mic_sample = _ORIG_MIC_SAMPLE_12


def _core_sources(row: dict, rng: np.random.Generator, mic_arr, mic_rec, n_car: int, warns: list, sides: list, tag: str):
    """車 n_car 台＋警告音（≤2）= 土台 v11core の規則そのまま（v16r.sample_scene_v15 = v14 徐行なし → v13 書き換え → v11）。"""
    seed = _draw_seed(rng)
    over = {"n_car": str(n_car), "n_warnings": len(warns)}
    for i, (w, sd) in enumerate(zip(warns, sides)):
        over[f"w{i + 1}_class"] = w
        over[f"w{i + 1}_side"] = sd
    sub = _sub_row(row, seed, tag, **over)
    with _shared_mic(mic_arr, mic_rec):
        s = v16r.sample_scene_v15(sub)
    return s["sources"], {"tag": tag, "seed": seed, "sampler": "v11core", "n_car": n_car, "warns": list(warns), "sides": list(sides),
                          "danger_tier": sub["danger_tier"], "car_side": sub["car_side"]}


def _v12_sources(row: dict, rng: np.random.Generator, mic_arr, mic_rec, fn, name: str, tag: str):
    seed = _draw_seed(rng)
    sub = _sub_row(row, seed, tag, scenario=name, car_side="")
    with _shared_mic(mic_arr, mic_rec):
        s = fn(sub)
    return s["sources"], {"tag": tag, "seed": seed, "sampler": name}


def _assign_tracks(sources: list) -> int:
    """ラベルの track をクラスごとに 0,1,2 と振る（並びは組み合わせの順）。上限 3 を超えた分は no_label にして数を返す。"""
    n = defaultdict(int); dropped = 0
    for x in sources:
        if x.get("no_label"):
            x["track"] = int(x.get("track", 0) or 0)
            continue
        c = x["class"]
        if n[c] >= MAX_TRACKS:
            x["no_label"] = True; x["label_dropped"] = f"ADPIT の track 上限 {MAX_TRACKS}"; dropped += 1
            continue
        x["track"] = n[c]; n[c] += 1
    return dropped


# ---------------------------------------------------------------- 場面
def sample_v19mix(row: dict) -> dict:
    comp = parse_mix(row.get("mix", ""))
    why = mix_ok(comp)
    assert not why, f"{row.get('clip_id')}: mix={row.get('mix')} は規則に合わない: {why}"
    rng = v19r._rng(row)
    mic_arr, mic = v12._mic_sample(row, rng)                 # 本のマイク（v19／v22 の場面と同じ抽選）
    warns = [w for w in WARN for _ in range(comp[w])]
    sides = []
    for i in range(len(warns)):                               # 側は plan の w1〜w3_side を使う（空なら引く）
        sd = str(row.get(f"w{i + 1}_side") or "")
        sides.append(sd if sd in ("L", "R") else ("L" if rng.random() < 0.5 else "R"))
    sources, subs = [], []
    g = 0
    if comp["car"] or warns:
        srcs, info = _core_sources(row, rng, mic_arr, mic, comp["car"], warns[:2], sides[:2], "core")
        for x in srcs:
            x["mix_group"] = g; x["mix_token"] = "car" if x["class"] == "car_drive" else x["class"]
        sources += srcs; subs.append(info); g += 1
    if len(warns) == 3:
        srcs, info = _core_sources(row, rng, mic_arr, mic, 0, warns[2:3], sides[2:3], "warn3")
        for x in srcs:
            x["mix_group"] = g; x["mix_token"] = x["class"]
        sources += srcs; subs.append(info); g += 1
    for k in range(comp["bicycle"]):
        seed = _draw_seed(rng)
        sub = _sub_row(row, seed, f"bicycle{k}", scenario="v19bicycle", car_side="")
        src, binfo = v22._bicycle_src(sub, rng, mic_arr, "v27_mix_bicycle")
        src["mix_group"] = g; src["mix_token"] = "bicycle"
        sources.append(src); subs.append({"tag": f"bicycle{k}", "seed": seed, "sampler": "v22_bicycle_src", **binfo}); g += 1
    for k in range(comp["kick"]):
        far = bool(rng.random() < KICK_P_FAR)
        fn, name = (v21.sample_v19kickfar, "v19kickfar") if far else (v12.sample_v12kick, "v12kick")
        srcs, info = _v12_sources(row, rng, mic_arr, mic, fn, name, f"kick{k}")
        for x in srcs:
            x["mix_group"] = g; x["mix_token"] = "kick"
        info["far"] = far
        sources += srcs; subs.append(info); g += 1
    for k in range(comp["moto"]):
        far = bool(rng.random() < MOTO_P_FAR)
        if far:
            seed = _draw_seed(rng)
            sub = _sub_row(row, seed, f"moto{k}", scenario=SCEN_MOTOFAR, car_side="")
            src, minfo = _moto_far_src(sub, rng, mic_arr, "v27_mix_motofar")
            srcs, info = [src], {"tag": f"moto{k}", "seed": seed, "sampler": "v27_moto_far_src", **minfo}
        else:
            srcs, info = _v12_sources(row, rng, mic_arr, mic, v12.sample_v12bike, "v12bike", f"moto{k}")
        for x in srcs:
            x["mix_group"] = g; x["mix_token"] = "moto"
        info["far"] = far
        sources += srcs; subs.append(info); g += 1
    y_train1 = None
    for k in range(comp["train"]):
        srcs, info = _v12_sources(row, rng, mic_arr, mic, v12.sample_v12train, "v12train", f"train{k}")
        for x in srcs:
            x["mix_group"] = g; x["mix_token"] = "train"; x["no_label"] = True; x["train_background"] = "D14"   # 車両は背景（ラベルなし）
            x["train_index"] = k
            if k == 0:
                y_train1 = float(x["wp"][0][2])
            else:                                             # 2 編成目 = 1 編成目と同じ側の 4 m 外側の線路（複線）。向き・速さ・両数は自分の抽選のまま
                y2 = y_train1 + (TRAIN2_Y_SHIFT_M if y_train1 >= 0 else -TRAIN2_Y_SHIFT_M)
                for p in x["wp"]:
                    p[2] = y2
        info["y_m"] = y_train1 if k == 0 else y2
        sources += srcs; subs.append(info); g += 1
    dropped = _assign_tracks(sources)
    info = {"kind": "mix", "mix": mix_string(comp), "n_groups": g, "subs": subs, "n_sources": len(sources),
            "n_labeled": sum(not x.get("no_label") for x in sources), "n_label_dropped": dropped}
    return v19r._finish(row, rng, mic_arr, mic, sources, info)


# ---------------------------------------------------------------- 登録（v21／v22 と同じ: _SAMPLERS19 に鍵を足す）
for _scen, _fn in ((SCENARIO, sample_v19mix), (SCEN_MOTOFAR, sample_v19motofar)):
    assert _scen not in v19r._SAMPLERS19 or v19r._SAMPLERS19[_scen] is _fn, f"{_scen} が別の関数で登録済み"
    v19r._SAMPLERS19[_scen] = _fn
sample_scene_v19 = v19r.sample_scene_v19          # v24 の大きさの規則で包まれた版（v26 まで読み込み済み）
generate_clip = v19r.generate_clip


# ---------------------------------------------------------------- 検査
def check_scene(s: dict, row: dict) -> None:
    comp = parse_mix(row["mix"])
    assert not mix_ok(comp), mix_ok(comp)
    srcs = s["sources"]
    assert m11v._make_mic is _ORIG_MAKE_MIC_11 and v12._mic_sample is _ORIG_MIC_SAMPLE_12, "マイク抽選の差し替えが戻っていない"
    cnt = Counter()
    trains = set()
    for x in srcs:
        if x.get("params", {}).get("train_car") or "train_car_index" in x:
            trains.add(x["mix_group"])
            assert x.get("no_label") and x.get("train_background") == "D14", "電車の車両にラベルが付いている（D14 は背景）"
        else:
            cnt[x["mix_token"]] += 1
    cnt["train"] = len(trains)
    want = {t: comp[t] for t in TOKENS if comp[t]}
    got = {t: cnt[t] for t in TOKENS if cnt[t]}
    assert want == got, f"組み合わせが plan と違う: plan={want} 場面={got}"
    assert int(row.get("n_car") or 0) == comp["car"], f"n_car {row.get('n_car')} != {comp['car']}"
    nw = sum(comp[w] for w in WARN)
    assert int(row.get("n_warnings") or 0) == nw, f"n_warnings {row.get('n_warnings')} != {nw}"
    for i in range(3):
        c = str(row.get(f"w{i + 1}_class") or "")
        assert (c != "") == (i < nw), f"w{i + 1}_class={c!r} と警告音 {nw} 個が合わない"
    # track: ラベルつきはクラスごとに 0.. と一意・上限 3・落としたラベルは無い
    tr = defaultdict(list)
    for x in srcs:
        if not x.get("no_label"):
            tr[x["class"]].append(int(x["track"]))
    for c, v in tr.items():
        assert sorted(v) == list(range(len(v))) and len(v) <= MAX_TRACKS, f"{c} の track {v}"
    assert s["v19"]["n_label_dropped"] == 0, s["v19"]["n_label_dropped"]
    assert s["v19"]["n_labeled"] == sum(comp[t] for t in TOKENS if t != "train"), (s["v19"]["n_labeled"], comp)
    # マイクの共有: 車と自転車（最接近時のマイク位置を基準に置く音源）は、その時刻に本のマイクの x と一致する
    mic = np.asarray(s["_mic_arr"], float)
    for x in srcs:
        if x["class"] == "car_drive" and "t_cpa_rel_s" in x and not x.get("close_slow"):
            t = float(x["t_cpa_rel_s"]); wp = np.array(x["wp"], float)
            xs = float(np.interp(t, wp[:, 0], wp[:, 1])); xm = float(m9._mic_pos_at(mic, t)[0])
            assert abs(xs - xm) < 1e-6, f"車の最接近がマイクとずれる {xs} vs {xm}（マイクの共有が効いていない）"
        if x["class"] == "bicycle":
            t = float(x["t_cpa_s"]); wp = np.array(x["wp"], float)
            xs = float(np.interp(t, wp[:, 0], wp[:, 1])); xm = float(m9._mic_pos_at(mic, t)[0])
            assert abs(xs - xm) < 1e-6, f"自転車の最接近がマイクとずれる {xs} vs {xm}"
    # 電車 2 編成: 2 編成目は 4 m 外側（同じ側）
    if comp["train"] == 2:
        ys = {}
        for x in srcs:
            if x.get("train_background"):
                ys.setdefault(x["train_index"], float(x["wp"][0][2]))
        assert set(ys) == {0, 1} and abs(abs(ys[1]) - abs(ys[0]) - TRAIN2_Y_SHIFT_M) < 1e-9 and ys[0] * ys[1] > 0, ys
    # 部分場面の記録
    subs = s["v19"]["subs"]
    assert len(subs) == s["v19"]["n_groups"] and len({d["seed"] for d in subs}) == len(subs), "部分場面の種が重なる"
    assert all(d["sampler"] in ("v11core", "v22_bicycle_src", "v12kick", "v19kickfar", "v12bike", "v27_moto_far_src", "v12train") for d in subs), subs
    for x in srcs:                                            # 遠いバイクは 4〜12 m・近いバイクは 1.5〜4 m（v12bike）
        if x["class"] == "motorcycle":
            lat = abs(float(x["wp"][0][2]))
            assert (MOTOFAR_LAT[0] - 1e-9 <= lat <= MOTOFAR_LAT[1] + 1e-9) if x.get("grammar") == "v27_mix_motofar" else (1.5 - 1e-9 <= lat <= 4.0 + 1e-9), (x.get("grammar"), lat)


def fake_row(mix: str, i: int, motion: str | None = None, split: str = "train") -> dict:
    comp = parse_mix(mix)
    warns = [w for w in WARN for _ in range(comp[w])]
    tiers = ("critical", "caution", "safe")
    r = {"clip_id": f"preview_mix_{i}", "split": split, "motion": motion or ("walk" if i % 10 < 7 else "static"),
         "n_warnings": len(warns), "w1_class": "", "w1_side": "", "w2_class": "", "w2_side": "", "w3_class": "", "w3_side": "",
         "danger_tier": tiers[i % 3], "car_side": "L" if i % 2 else "R", "scenario": SCENARIO,
         "seed": 9200000000 + i, "scene_type": "v19", "n_car": str(comp["car"]), "rain": "", "rain_rate": "", "rain_dba": "",
         "grammar": "v19", "close_slow": "", "cs_cpa_m": "", "cs_speed_kmh": "", "cs_level_adj_db": "",
         "mic_z": f"{1.4 + 0.7 * ((i * 37) % 100) / 100:.3f}", "h_copy": "1", "twin_of": "", "mix": mix_string(comp)}
    for k, w in enumerate(warns):
        r[f"w{k + 1}_class"] = w; r[f"w{k + 1}_side"] = "L" if (i + k) % 2 else "R"
    return r


def _scene_key(s: dict) -> str:
    """場面の比較用（numpy は文字列化）。"""
    d = {k: v for k, v in s.items() if k not in ("row",)}
    return json.dumps(d, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def fake_row_motofar(i: int, motion: str | None = None) -> dict:
    r = fake_row("car+moto", i, motion)
    r.update({"clip_id": f"preview_motofar_{i}", "scenario": SCEN_MOTOFAR, "seed": 9201000000 + i, "n_car": "0", "mix": "", "car_side": "L" if i % 2 else "R"})
    return r


def check_row(s: dict, r: dict) -> None:
    if r.get("scenario") == SCENARIO:
        check_scene(s, r)
    elif r.get("scenario") == SCEN_MOTOFAR:
        check_motofar(s)
    else:
        raise ValueError(r.get("scenario"))


def check_plan(path: Path = PLAN, limit: int | None = None) -> int:
    bad = n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("scenario") not in (SCENARIO, SCEN_MOTOFAR):
                continue
            if limit is not None and n >= limit:
                break
            n += 1
            r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0)
            try:
                check_row(sample_scene_v19(r), r)
            except AssertionError as e:
                bad += 1
                if bad <= 10:
                    print(f"NG {r['clip_id']}: {e}")
    print(f"check_plan v27: {n} 本 OK {n - bad} / NG {bad}")
    return bad


def preview(n: int) -> None:
    import itertools
    pats = [m for m in (tuple(sorted(c, key=TOKENS.index)) for k in (2, 3) for c in itertools.combinations_with_replacement(TOKENS, k)) if not mix_ok(m)]
    rnd = np.random.default_rng(20261013)
    cnt = Counter(); ok = 0
    for i in range(n):
        mix = mix_string(pats[int(rnd.integers(len(pats)))])
        r = fake_row(mix, i)
        s = sample_scene_v19(r)
        try:
            check_scene(s, r); ok += 1
        except AssertionError as e:
            print("NG", mix, e)
        cnt[len(s["sources"])] += 1
    print(f"preview: {n} 場面・検査 OK {ok}/{n}・音源の数の分布 {dict(sorted(cnt.items()))}・組の種類 {len(pats)}")


def _ref_scenes_subprocess(rows: list, with_v27: bool) -> list:
    """別プロセスで（v27 を読む／読まない）既存の場面を引き、比較用の文字列を返す（既存の場面を壊していない証拠・v21 K4 と同じ考え）。"""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r}); sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import step11_v26_render as v26\n"
        + ("import step11_v27_render as v27\n" if with_v27 else "")
        + "import step11_v19_render as v19r\n"
        "rows = json.loads(sys.stdin.read())\n"
        "out = []\n"
        "for r in rows:\n"
        "    s = v19r.sample_scene_v19(r)\n"
        "    d = {k: v for k, v in s.items() if k != 'row'}\n"
        "    out.append(json.dumps(d, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))\n"
        "sys.stdout.write(json.dumps(out))\n")
    p = subprocess.run([sys.executable, "-c", code], input=json.dumps(rows), capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def selftest() -> int:
    ok = []
    # 1. 代表の組み合わせ（2 つ組・3 つ組・警告音 3 つ・電車 2 編成・同じクラス 3 つ・車 3 台）を歩行／静止で引いて検査
    pats = ["car+bicycle", "car+kick", "car+moto", "car+train", "bicycle+siren", "kick+horn", "moto+backup_beep", "train+crossing",
            "bicycle+bicycle", "kick+kick", "moto+moto", "train+train", "siren+horn", "bike_bell+bike_bell",
            "car+car+bicycle", "car+bicycle+siren", "bicycle+kick+moto", "siren+horn+backup_beep", "siren+siren+siren",
            "bicycle+bicycle+bicycle", "car+car+car", "train+train+siren", "car+train+crossing", "kick+kick+bike_bell", "bicycle+crossing+crossing"]
    n_ok = 0; n_src = Counter()
    for i, mix in enumerate(pats):
        for motion in ("walk", "static"):
            r = fake_row(mix, i, motion)
            s = sample_scene_v19(r)
            try:
                check_scene(s, r); n_ok += 1
            except AssertionError as e:
                print(f"NG {mix} {motion}: {e}")
            n_src[len(s["sources"])] += 1
    ok.append((f"代表 {len(pats)} 組 × 2 動き = {2 * len(pats)} 場面の検査（組み合わせ・ラベルの track・マイクの共有・電車は背景・2 編成目 4 m 外）: OK {n_ok}", n_ok == 2 * len(pats)))
    # 2. 規則: 除く組と上限
    ok.append(("規則: train+backup_beep は置かない・電車 3 編成は置かない・音源 4 つは置かない・警告音 4 つは置かない",
               bool(mix_ok(["train", "backup_beep"])) and bool(mix_ok(["train", "train", "train"])) and bool(mix_ok(["car", "car", "car", "bicycle"]))
               and bool(mix_ok(["siren", "horn", "backup_beep", "crossing"])) and not mix_ok(["train", "train"]) and not mix_ok(["siren", "horn", "backup_beep"])))
    # 3. 決定論: 同じ行を 2 回引くと同一・種が違えば違う
    r = fake_row("car+bicycle+siren", 3, "walk")
    a, b = _scene_key(sample_scene_v19(r)), _scene_key(sample_scene_v19(r))
    r2 = dict(r, seed=r["seed"] + 1)
    ok.append(("決定論: 同じ行 → 同一の場面・種が違えば違う場面", a == b and a != _scene_key(sample_scene_v19(r2))))
    # 4. 部分場面はそのままの規則: 静止の本（マイクが同一）では、車＋警告音の部分場面は同じ種・同じ側・同じ層の v11core の単独場面と一致
    r = fake_row("car+car+siren", 5, "static")
    s = sample_scene_v19(r)
    sub = next(d for d in s["v19"]["subs"] if d["sampler"] == "v11core")
    row11 = _sub_row(r, sub["seed"], "ref", n_car=str(sub["n_car"]), n_warnings=len(sub["warns"]), w1_class=sub["warns"][0], w1_side=sub["sides"][0])
    s11 = v16r.sample_scene_v15(row11)
    mine = [x for x in s["sources"] if x["mix_group"] == 0]
    same = len(mine) == len(s11["sources"]) and all(
        np.allclose(np.array(x["wp"], float)[:, :3], np.array(y["wp"], float)[:, :3]) and x["class"] == y["class"] and x["t_on"] == y["t_on"] and x["t_off"] == y["t_off"]
        and x["params"] == y["params"] and x["track"] == y["track"] and x.get("speed_mps") == y.get("speed_mps") for x, y in zip(mine, s11["sources"]))
    # v24 の大きさ（l1m・law・サイレンの高さ z）は本の外側（sample_scene_v19 の包み）で掛かるので比べない（部分場面は v16 までの素の規則）
    ok.append(("部分場面は土台の規則そのまま: 静止の本の車 2 台＋サイレンが、同じ種の v11core 単独場面と軌道 xy・時刻・params・track・速さで一致", same))
    # 5. 自転車・キック（近／遠）・バイク・電車の部分場面も単独の場面と一致（静止）
    def _one(mix, sampler, pick):
        r = fake_row(mix, 7, "static"); s = sample_scene_v19(r)
        d = next(x for x in s["v19"]["subs"] if x["sampler"] == sampler)
        sub = _sub_row(r, d["seed"], "ref", scenario=sampler, car_side="")
        ref = pick(sub)
        mine = [x for x in s["sources"] if x["mix_group"] == s["v19"]["subs"].index(d)]
        return len(mine) == len(ref) and all(np.allclose(np.array(x["wp"], float), np.array(y["wp"], float)) and x["params"] == y["params"] for x, y in zip(mine, ref))
    chk = [_one("car+kick", "v12kick", lambda sub: v12.sample_v12kick(sub)["sources"]) if any(d["sampler"] == "v12kick" for d in sample_scene_v19(fake_row("car+kick", 7, "static"))["v19"]["subs"]) else
           _one("car+kick", "v19kickfar", lambda sub: v21.sample_v19kickfar(sub)["sources"]),
           _one("car+moto", "v12bike", lambda sub: v12.sample_v12bike(sub)["sources"]),
           _one("car+train", "v12train", lambda sub: v12.sample_v12train(sub)["sources"])]
    ok.append(("キック・バイク・電車の部分場面も単独の場面（同じ種）と軌道・params で一致（静止）", all(chk)))
    # 6. キックの近／遠は半々（600 場面）
    far = 0
    for i in range(600):
        s = sample_scene_v19(fake_row("car+kick", 100 + i, "walk")); far += int(next(d for d in s["v19"]["subs"] if d["tag"] == "kick0")["far"])
    ok.append((f"組み合わせの中のキックの遠い割合 {far / 600:.1%}（期待 50% ± 3σ = 44〜56%）", 0.44 <= far / 600 <= 0.56))
    # 6b. 組み合わせの中のバイクの近／遠も半々（600 場面）・遠い側は 4〜12 m
    far = 0; lats = []
    for i in range(600):
        s = sample_scene_v19(fake_row("car+moto", 700 + i, "walk")); d = next(d for d in s["v19"]["subs"] if d["tag"] == "moto0"); far += int(d["far"])
        lats.append(abs(float(next(x for x in s["sources"] if x["class"] == "motorcycle")["wp"][0][2])))
    lats = np.array(lats)
    ok.append((f"組み合わせの中のバイクの遠い割合 {far / 600:.1%}（44〜56%）・横距離 1.5〜12 m・4 m 超 {np.mean(lats > 4.0):.1%}", 0.44 <= far / 600 <= 0.56 and lats.min() >= 1.5 - 1e-9 and lats.max() <= 12.0 + 1e-9))
    # 6c. v19motofar 単独（600 場面）: 1 台・横 4〜12 m の対数一様（中央値 √48 = 6.93 m ± 0.5）・原付 40% の 3σ 内・最接近がマイク位置
    n_ok = 0; lat = []; moped = 0
    for i in range(600):
        r = fake_row_motofar(i); s = sample_scene_v19(r)
        try:
            check_motofar(s); n_ok += 1
        except AssertionError as e:
            if n_ok == i:
                print("NG motofar", e)
        x = s["sources"][0]; lat.append(abs(float(x["wp"][0][2]))); moped += int(x["params"]["engine_class"] == "moped")
    lat = np.array(lat); med = float(np.median(lat)); sd = np.sqrt(600 * 0.4 * 0.6)
    ok.append((f"v19motofar 600 場面: 検査 OK {n_ok}・横距離 中央値 {med:.2f} m（期待 6.93 ± 0.5）・≥8 m {np.mean(lat >= 8):.0%}（対数一様の期待 37%）・原付 {moped}（240 ± 3σ {3 * sd:.0f}）",
               n_ok == 600 and abs(med - np.sqrt(48.0)) < 0.5 and abs(moped - 240) <= 3 * sd))
    # 7. マイクの差し替えは例外でも戻る
    try:
        with _shared_mic(np.array([0.0, 0.0, 1.5]), {"motion": "static"}):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    ok.append(("マイクの差し替えは例外でも元に戻る", m11v._make_mic is _ORIG_MAKE_MIC_11 and v12._mic_sample is _ORIG_MIC_SAMPLE_12))
    # 8. 既存の場面を壊していない: 既存 plan の各場面 3 本を、v27 を読まない別プロセスと読む別プロセスで引いて同一
    if PLAN.is_file():
        by = defaultdict(list)
        with open(PLAN, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["scenario"] not in (SCENARIO, SCEN_MOTOFAR) and len(by[r["scenario"]]) < 3:
                    r["seed"] = int(r["seed"]); r["n_warnings"] = int(r["n_warnings"] or 0); by[r["scenario"]].append(r)
        rows = [r for v in by.values() for r in v]
        ref, mine = _ref_scenes_subprocess(rows, False), _ref_scenes_subprocess(rows, True)
        n_diff = sum(a != b for a, b in zip(ref, mine))
        ok.append((f"既存 {len(by)} 場面 × 3 本 = {len(rows)} 本: v27 を読んでも場面が同一（別プロセス比較・違い {n_diff}）", n_diff == 0 and len(ref) == len(rows)))
    # 9. 登録
    ok.append(("登録: _SAMPLERS19['v19mix'] は sample_v19mix・['v19motofar'] は sample_v19motofar・既存の鍵は残っている",
               v19r._SAMPLERS19.get(SCENARIO) is sample_v19mix and v19r._SAMPLERS19.get(SCEN_MOTOFAR) is sample_v19motofar
               and all(k in v19r._SAMPLERS19 for k in ("v19fast", "v19turn", "v19platoon", "v19launch", "v19kickfar", "v19bicycle", "v19bellbike", "v19bellpin"))))
    for name, good in ok:
        print(("PASS " if good else "FAIL ") + name)
    n_bad = sum(not g for _, g in ok)
    print(f"selftest v27: {len(ok) - n_bad}/{len(ok)}")
    return 1 if n_bad else 0


def main() -> int:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    if "--check-plan" in a:
        i = a.index("--check-plan")
        path = Path(a[i + 1]) if i + 1 < len(a) and not a[i + 1].startswith("--") else PLAN
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        return 1 if check_plan(path, lim) else 0
    if "--preview" in a:
        preview(int(a[a.index("--preview") + 1])); return 0
    print(__doc__); return 0


if __name__ == "__main__":
    sys.exit(main())
