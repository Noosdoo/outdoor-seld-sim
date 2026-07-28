# -*- coding: utf-8 -*-
"""step11_v11_eval_render.py — v11評価拡張（実装③）レンダラ。

仕様の正= md/design/v11評価拡張_実装仕様_2026-07-28.md。
step11_v11_render.py（=v10.1bチェーン＋v11coreサンプラ＋ピーク棄却再抽選）を継承し、
出力先を独立フォルダ out/dataset_outdoor_siren_v11_eval/ に切替。**物理・音源・較正は不変**。

増量9セットは既存トークンがそのまま動く（carfree_siren / v11core / scn2系 / traffic /
probe_* / intersection_siren）。新規はN1〜N7の7サンプラのみ:
  n1_blind      車1台の突然出現（t_on=U(2,5)、出現時距離U(10,20)m→t_cpa従属、CPA=tier）
  n2_ev         静音EV（レベルU(50,56)dB@2m、速度U(1.5,5.5)m/s、他はcore車と同一規約）
  n3_parking    バック音2-3本（同一側y=U(2,8)、後退0.5-2m/s、track=0..n-1）＋徐行車1台(2-4m/s)
  n4_fast_siren 高速サイレン（15-25m/s、横3-10m、連続吹鳴、siren3型抽選は既存のまま）
  n5_downtown   v11core(車3+警告2)の幾何＋暗騒音をU(55,65)に再抽選
  n6_overtake   至近追い越しベル（横U(0.5,1.2)m、速度U(4,9)、bell_overtakeと同構造）
  n7_pullout    停車→発進（t_start=U(2,4)まで静止、a=U(1.5,2.5)m/s²、vmax=U(4,8)、
                多点waypointの2次近似。t_on=t_start=エンジン音の立ち上がり）
各サンプラの乱数消費順はコード順が正（新シード帯のため互換制約なし）。
ピーク棄却再抽選はNトークンにも適用（saltシード=[seed, 20260728, salt]）。

使い方（dynamic-sound venv）:
  python scripts/step11_v11_eval_render.py precheck | preflight | inspect | pack
  （全量生成は scripts/_run_v11_eval_gen.py）
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v11_render as m11  # noqa: E402 (v11coreサンプラ+棄却再抽選を継承)

m9 = m11.m9
m9.DS_NAME = "outdoor_siren_v11_eval"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = ROOT / "out" / "dataset_outdoor_siren_v11_eval" / "plan"
m9.WORK = m9.DS / "work"

SETS = ["halluc600", "safe600", "s1_200", "s2_100", "s3_100", "s5_200",
        "cross100", "multi200", "probe96", "n1", "n2", "n3", "n4", "n5",
        "n6", "n7"]
SALT_NS = 20260728


# ------------------------------------------------ N1〜N7 サンプラ ----

def _base_scene(row, mic, mic_rec):
    return {"row": dict(row), "mic": mic_rec, "sources": [], "_mic_arr": mic}


def _finish(s, rng, dba_range=None):
    lo, hi = dba_range if dba_range else m9.NOISE_DBA
    s["noise"] = {"dba": float(rng.uniform(lo, hi)),
                  "seed": s["row"]["seed"] * 7919 + 13}
    return s


def sample_n1_blind(row, rng):
    mic, rec = m11._make_mic(row, rng)
    s = _base_scene(row, mic, rec)
    sign = 1.0 if row["car_side"] == "L" else -1.0
    vc = float(rng.uniform(4.0, 8.0))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    z = float(rng.uniform(*m9.SRC_Z))
    dz = m9.MIC_STATIC[2] - z
    lo, hi = m9.TIER_CPA[row["danger_tier"]]
    cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
    y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * sign
    t_on = float(rng.uniform(2.0, 5.0))
    d_on = float(rng.uniform(10.0, 20.0))
    dx = float(np.sqrt(max(d_on * d_on - y * y - dz * dz, 0.25)))
    dt = min(dx / vc, 9.0 - t_on)           # t_cpaはクリップ内に収める
    t_cpa = t_on + dt
    lv, l1m = m9._draw_level("car_drive", rng)
    f0 = 42.0 * float(rng.uniform(0.8, 1.2))
    x0 = float(m9._mic_pos_at(mic, t_cpa)[0]) - dirx * vc * t_cpa
    s["sources"].append({
        "class": "car_drive", "kind": "vehicle", "track": 0,
        "wp": [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * vc * m9.CLIP, y, z]],
        "t_on": round(t_on, 3), "t_off": m9.CLIP, "speed_mps": vc,
        "dir_x": dirx, "danger_tier": row["danger_tier"],
        "cpa_rel_target_m": cpa, "t_cpa_rel_s": t_cpa,
        "appear_dist_m": d_on, "law_db": lv, "l1m_db": l1m,
        "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
    return _finish(s, rng)


def sample_n2_ev(row, rng):
    mic, rec = m11._make_mic(row, rng)
    s = _base_scene(row, mic, rec)
    sign = 1.0 if row["car_side"] == "L" else -1.0
    vc = float(rng.uniform(1.5, 5.5))       # AVAS義務域=低速
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    z = float(rng.uniform(*m9.SRC_Z))
    dz = m9.MIC_STATIC[2] - z
    lo, hi = m9.TIER_CPA[row["danger_tier"]]
    cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
    y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * sign
    t_cpa = float(rng.uniform(*m9.CAR_TCPA))
    lv = float(rng.uniform(50.0, 56.0))     # UN R138 AVAS最低要件域 @2m
    l1m = lv + 20.0 * float(np.log10(2.0))
    f0 = 42.0 * float(rng.uniform(0.8, 1.2))
    x0 = float(m9._mic_pos_at(mic, t_cpa)[0]) - dirx * vc * t_cpa
    s["sources"].append({
        "class": "car_drive", "kind": "vehicle", "track": 0,
        "wp": [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * vc * m9.CLIP, y, z]],
        "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": vc, "dir_x": dirx,
        "danger_tier": row["danger_tier"], "cpa_rel_target_m": cpa,
        "t_cpa_rel_s": t_cpa, "law_db": lv, "l1m_db": l1m,
        "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
    return _finish(s, rng)


def sample_n3_parking(row, rng):
    mic, rec = m11._make_mic(row, rng)
    s = _base_scene(row, mic, rec)
    sign = 1.0 if row["w1_side"] == "L" else -1.0
    nbeep = int(row["n_warnings"])
    for j in range(nbeep):
        vb = float(rng.uniform(0.5, 2.0))
        dirb = 1.0 if rng.random() < 0.5 else -1.0
        yb = float(rng.uniform(2.0, 8.0)) * sign
        zb = float(rng.uniform(*m9.SRC_Z))
        tcb = float(rng.uniform(3.0, 8.0))
        lvb, l1mb = m9._draw_level("backup_beep", rng)   # V10_1B混合を各ブザー独立抽選
        pb = m9._warn_params("backup_beep", rng, row["seed"])
        t_onb = float(rng.uniform(0.3, 2.0))
        t_offb = min(t_onb + float(rng.uniform(4.0, 7.0)), 9.5)
        x0b = float(m9._mic_pos_at(mic, tcb)[0]) - dirb * vb * tcb
        s["sources"].append({
            "class": "backup_beep", "kind": "vehicle", "track": j,
            "wp": [[0.0, x0b, yb, zb], [m9.CLIP, x0b + dirb * vb * m9.CLIP, yb, zb]],
            "t_on": round(t_onb, 3), "t_off": round(t_offb, 3),
            "speed_mps": vb, "dir_x": dirb, "t_cpa_s": tcb,
            "law_db": lvb, "l1m_db": l1mb, "params": pb})
    vc = float(rng.uniform(2.0, 4.0))
    dirc = 1.0 if rng.random() < 0.5 else -1.0
    zc = float(rng.uniform(*m9.SRC_Z))
    dz = m9.MIC_STATIC[2] - zc
    lo, hi = m9.TIER_CPA[row["danger_tier"]]
    cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
    yc = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * sign
    tcc = float(rng.uniform(4.0, 9.0))
    lvc, l1mc = m9._draw_level("car_drive", rng)
    f0 = 42.0 * float(rng.uniform(0.8, 1.2))
    x0c = float(m9._mic_pos_at(mic, tcc)[0]) - dirc * vc * tcc
    s["sources"].append({
        "class": "car_drive", "kind": "vehicle", "track": 0,
        "wp": [[0.0, x0c, yc, zc], [m9.CLIP, x0c + dirc * vc * m9.CLIP, yc, zc]],
        "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": vc, "dir_x": dirc,
        "danger_tier": row["danger_tier"], "cpa_rel_target_m": cpa,
        "t_cpa_rel_s": tcc, "law_db": lvc, "l1m_db": l1mc,
        "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
    return _finish(s, rng)


def sample_n4_fast_siren(row, rng):
    mic, rec = m11._make_mic(row, rng)
    s = _base_scene(row, mic, rec)
    side = 1.0 if row["w1_side"] == "L" else -1.0
    lv, l1m = m9._draw_level("siren", rng)
    params = m9._warn_params("siren", rng, row["seed"])
    vs = float(rng.uniform(15.0, 25.0))     # 54-90km/h=緊急走行
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    y = float(rng.uniform(3.0, 10.0)) * side
    z = float(rng.uniform(*m9.SRC_Z))
    t_cpa = float(rng.uniform(5.0, 8.0))
    x0 = float(m9._mic_pos_at(mic, t_cpa)[0]) - dirx * vs * t_cpa
    s["sources"].append({
        "class": "siren", "kind": "vehicle", "track": 0,
        "wp": [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * vs * m9.CLIP, y, z]],
        "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": vs, "dir_x": dirx,
        "t_cpa_rel_s": t_cpa, "law_db": lv, "l1m_db": l1m, "params": params})
    return _finish(s, rng)


def sample_n5_downtown(row, rng):
    s = m11.sample_v11core(row, rng)        # 車3台+異クラス警告2（v11coreの幾何）
    s["noise"]["dba"] = float(rng.uniform(55.0, 65.0))   # ⑥暗騒音を高域に再抽選
    return s


def sample_n6_overtake(row, rng):
    # bell_overtakeと同構造（背後=−xの定義のため歩行は+x固定）。横と速度のみ強化
    if row["motion"] == "walk":
        v = float(rng.uniform(*m9.WALK_SPEED))
        mic = np.array([[0.0, -v * 5.0, 0.0, m9.MIC_STATIC[2]],
                        [m9.CLIP, v * 5.0, 0.0, m9.MIC_STATIC[2]]])
        rec = {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": 1.0,
               "waypoints": mic.tolist()}
    else:
        mic, rec = np.array(m9.MIC_STATIC), {"motion": "static"}
    s = _base_scene(row, mic, rec)
    side = 1.0 if row["w1_side"] == "L" else -1.0
    vb = float(rng.uniform(4.0, 9.0))       # 自転車〜キックボード
    c = float(rng.uniform(0.5, 1.2))        # S2(0.8-1.5)より近い
    zb = float(rng.uniform(0.9, 1.1))
    dz = m9.MIC_STATIC[2] - zb
    t_cpa = float(rng.uniform(*m9.CAR_TCPA))
    lv, l1m = m9._draw_level("bike_bell", rng)
    params = m9._warn_params("bike_bell", rng, row["seed"])
    t_on = max(0.3, t_cpa - float(rng.uniform(2.0, 5.0)))
    t_off = min(t_on + float(rng.uniform(*m9.EVENT_DUR)), m9.CLIP - 0.3)
    x0 = float(m9._mic_pos_at(mic, t_cpa)[0]) - vb * t_cpa
    s["sources"].append({
        "class": "bike_bell", "kind": "vehicle", "track": 0,
        "wp": [[0.0, x0, c * side, zb], [m9.CLIP, x0 + vb * m9.CLIP, c * side, zb]],
        "t_on": round(t_on, 3), "t_off": round(t_off, 3),
        "speed_mps": vb, "dir_x": 1.0,
        "cpa_rel_target_m": round(float(np.hypot(c, dz)), 3),
        "t_cpa_rel_s": t_cpa, "law_db": lv, "l1m_db": l1m, "params": params})
    return _finish(s, rng)


def sample_n7_pullout(row, rng):
    mic, rec = m11._make_mic(row, rng)
    s = _base_scene(row, mic, rec)
    sign = 1.0 if row["car_side"] == "L" else -1.0
    d0 = float(rng.uniform(5.0, 15.0))      # 停車位置までの距離
    y = float(rng.uniform(1.5, 3.5)) * sign
    z = float(rng.uniform(*m9.SRC_Z))
    dz = m9.MIC_STATIC[2] - z
    t_start = float(rng.uniform(2.0, 4.0))
    a = float(rng.uniform(1.5, 2.5))
    vmax = float(rng.uniform(4.0, 8.0))
    lv, l1m = m9._draw_level("car_drive", rng)
    f0 = 42.0 * float(rng.uniform(0.8, 1.2))
    dx0 = float(np.sqrt(max(d0 * d0 - y * y - dz * dz, 0.25)))
    x_p0 = float(m9._mic_pos_at(mic, t_start)[0]) + dx0   # 前方に停車
    t_acc = vmax / a

    def travel(t):
        if t <= t_start:
            return 0.0
        u = t - t_start
        if u <= t_acc:
            return 0.5 * a * u * u
        return 0.5 * a * t_acc * t_acc + vmax * (u - t_acc)

    ts = [0.0, t_start] + [t_start + q for q in
                           (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
                           if t_start + q < m9.CLIP] + [m9.CLIP]
    wp = [[round(t, 3), x_p0 - travel(t), y, z] for t in ts]   # マイク側へ発進→通過
    s["sources"].append({
        "class": "car_drive", "kind": "vehicle", "track": 0,
        "wp": wp, "t_on": round(t_start, 3), "t_off": m9.CLIP,
        "speed_mps": vmax, "dir_x": -1.0, "t_start_s": round(t_start, 3),
        "accel_mps2": round(a, 3), "start_dist_m": round(d0, 3),
        "law_db": lv, "l1m_db": l1m,
        "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5}})
    return _finish(s, rng)


N_SAMPLERS = {"n1_blind": sample_n1_blind, "n2_ev": sample_n2_ev,
              "n3_parking": sample_n3_parking, "n4_fast_siren": sample_n4_fast_siren,
              "n5_downtown": sample_n5_downtown, "n6_overtake": sample_n6_overtake,
              "n7_pullout": sample_n7_pullout}


def sample_scene_v11eval(row: dict) -> dict:
    scen = row.get("scenario")
    if scen in N_SAMPLERS:
        salt = m11._PEAK_SALT
        rng = (np.random.default_rng(row["seed"]) if salt == 0
               else np.random.default_rng([row["seed"], SALT_NS, salt]))
        s = N_SAMPLERS[scen](row, rng)
        if salt:
            s["peak_resample_salt"] = salt
        return s
    return m11.sample_scene_v11(row)


m9.sample_scene_v9 = sample_scene_v11eval


def generate_clip_v11eval(row: dict) -> None:
    """ピーク棄却再抽選をNトークン＋v11coreに適用（それ以外は素通し）。"""
    if row.get("scenario") not in N_SAMPLERS and row.get("scenario") != "v11core":
        return m11._orig_generate_clip(row)
    last = None
    for salt in range(9):
        m11._PEAK_SALT = salt
        try:
            return m11._orig_generate_clip(row)
        except AssertionError as e:
            if "peak" not in str(e):
                raise
            last = e
            print(f"[peak-resample] {row['clip_id']} salt={salt} NG ({e}) -> retry",
                  flush=True)
        finally:
            m11._PEAK_SALT = 0
    raise last


m9.generate_clip = generate_clip_v11eval


# ------------------------------------------------------------- precheck ----

def precheck_eval() -> int:
    tgrid = np.arange(0.0, m9.CLIP, 0.05)
    lines = ["# v11評価拡張 生成前チェック（step11_v11_eval_render.py precheck）", ""]
    peak_list, n_bad, cpa_err = [], 0, 0.0
    for which in SETS:
        rows = m9.load_plan(which)
        for row in rows:
            s = sample_scene_v11eval(row)
            mic = s.pop("_mic_arr")
            scen = row["scenario"]
            ok = True
            cars = [x for x in s["sources"] if x["class"] == "car_drive"]
            ok &= len(cars) == int(row["n_car"])
            if scen == "n2_ev":
                ok &= all(50.0 <= c["law_db"] <= 56.0 for c in cars)
            if scen == "n4_fast_siren":
                ok &= 15.0 <= s["sources"][0]["speed_mps"] <= 25.0
            if scen == "n3_parking":
                beeps = [x for x in s["sources"] if x["class"] == "backup_beep"]
                ok &= (len(beeps) == row["n_warnings"]
                       and [b["track"] for b in beeps] == list(range(len(beeps))))
            if scen == "n1_blind":
                ok &= 2.0 <= s["sources"][0]["t_on"] <= 5.0
            if scen == "n7_pullout":
                w = s["sources"][0]["wp"]
                ok &= w[0][1] == w[1][1]        # t_startまで静止
            if scen == "n5_downtown":
                ok &= 55.0 <= s["noise"]["dba"] <= 65.0
            if not ok:
                n_bad += 1
                print(f"  [BAD] {row['clip_id']}")
            noise_dba = s["noise"]["dba"]
            pb = 4.0 * 10.0 ** ((noise_dba - m9.K_RMS_SPL) / 20.0)
            for src in s["sources"]:
                dist = m9._dist_series(src["wp"], mic, tgrid)
                active = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
                dmin = float(np.min(dist[active]))
                wpm = np.array(src["wp"], float)
                wpm[:, 3] = 2.0 * m9.GROUND_Z - wpm[:, 3]
                dmin_m = float(np.min(m9._dist_series(wpm, mic, tgrid)))
                rms = 10.0 ** ((src["l1m_db"] - m9.K_RMS_SPL) / 20.0) / dmin
                pb += m9.CREST[src["class"]] * rms * (1.0 + dmin / dmin_m)
                if src.get("cpa_rel_target_m") is not None and src["t_on"] == 0.0:
                    cpa_err = max(cpa_err, abs(float(np.min(dist))
                                               - src["cpa_rel_target_m"]))
            peak_list.append((float(pb), row["clip_id"]))
    peak_list.sort(reverse=True)
    lines.append(f"- 行数: {sum(len(m9.load_plan(w)) for w in SETS)} / "
                 f"実現値不一致 {n_bad} 行 / CPA検算最大差 {cpa_err:.3f} m"
                 "（t_on=0の等速源のみ。参考値）")
    lines.append("- ピーク上限バウンド上位5（参考値。正=生成時assert＋棄却再抽選）:")
    for pb, cid in peak_list[:5]:
        lines.append(f"    {cid}: {pb:.3f}")
    report = "\n".join(lines) + "\n"
    (m9.PLAN / "precheck_report.md").write_text(report, encoding="utf-8")
    print(report)
    assert n_bad == 0
    return 0


def preflight() -> int:
    # 決定論（各セット先頭3行×2回）＋各セット先頭1本のスモーク生成→検品
    for which in SETS:
        for row in m9.load_plan(which)[:3]:
            a = json.dumps(sample_scene_v11eval(dict(row)), sort_keys=True, default=str)
            b = json.dumps(sample_scene_v11eval(dict(row)), sort_keys=True, default=str)
            assert a == b, f"determinism NG: {row['clip_id']}"
    print("preflight 1) 決定論（16セット×3行）: OK")
    for which in SETS:
        row = m9.load_plan(which)[0]
        print(f"preflight 2) smoke: {row['clip_id']} ({row['scenario']})")
        m9.generate_clip(row)
    fail = m9.inspect_all()
    print(f"preflight 2) スモーク{len(SETS)}本 検品 fail={fail}")
    return 1 if fail else 0


def pack_eval() -> None:
    """評価拡張zip。zip内パスは **datasets/outdoor_siren_v11e/**（独立データセット名）。

    本体v11に混ぜない理由: 混ぜると既存の前処理インデックス(7,199本)が古くなり
    1万本超の再前処理が必要になる。別名なら追加分3,246本だけの前処理で済み、
    学習済みデータセットに一切触れない（プロビナンスも保たれる）。
    """
    zip_path = ROOT / "out" / "dataset_outdoor_siren_v11_eval.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for sub in ("foa", "metadata", "masks"):
            for p in sorted((m9.DS / sub).glob("*")):
                zf.write(p, f"datasets/outdoor_siren_v11e/{sub}/{p.name}")
                n += 1
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    if mode == "precheck":
        sys.exit(precheck_eval())
    elif mode == "preflight":
        sys.exit(preflight())
    elif mode == "inspect":
        sys.exit(1 if m9.inspect_all() else 0)
    elif mode == "pack":
        pack_eval()
    else:
        print("usage: precheck | preflight | inspect | pack")
        sys.exit(1)
