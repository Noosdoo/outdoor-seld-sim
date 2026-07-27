# -*- coding: utf-8 -*-
"""step11_v11_render.py — v11 core レンダラ（設計= md/design/v11データセット拡張_設計書_2026-07-27.md §1）。

step11_v10_1b_render.py のチェーン（v10のJP物理定数・車速11-36km/h・V10_1消防車・
V10_1B backup混合）をそのまま継承し、出力先を独立フォルダ
out/dataset_outdoor_siren_v11/ に切り替える。**物理・音源・較正は一切変更しない**
（凍結ポリシー: 変更したのはデータの構成だけ。評価専用セットがv10とビット同一に
留まる根拠でもある=設計書§1.5）。

v11の新規はサンプラ1本のみ: scenario='v11core' → sample_v11core(row, rng)
  乱数消費順（変更禁止。plan_check_report.md の申し送りと一致させる）:
    ①マイク（coreと同一: 歩行は速度と±x方向を抽選）
    ②車 n_car台（1台目=planのcar_side/danger_tier、2台目以降=seedから。track=j。
      t_cpa: n_car==1 は CAR_TCPA(6-9s)=coreの「接近の頭を収める」規約 /
      n_car>=2 は U(4,9)=v10.2 A枠(traffic)の実証済みregime）
    ③w1 ④w2（core通常経路と同一規約。crossingは静止・同一クラスw2はtrack=1）
    ⑤暗騒音
  n_car==0 は車なし。純静穏(warn0)は音源ゼロ=空ラベルCSVになる——Colab側の
  空ラベル許容preproc（設計書§1.5、実証済み）で学習に載せる。ローカルの検品・
  採点は空CSVをそのまま正として扱う。

使い方（dynamic-sound venv の python で）:
  python scripts/step11_v11_render.py precheck        # 全7,200行の解析事前チェック
  python scripts/step11_v11_render.py preflight       # 決定論・実現値・スモーク生成
  python scripts/step11_v11_render.py gen A-B         # core行のindex範囲を生成
  python scripts/step11_v11_render.py inspect         # 全生成済みクリップの検品
  python scripts/step11_v11_render.py pack            # Colab用zip（datasets/outdoor_siren_v11/）
（全量生成は scripts/_run_v11_gen.py を使う）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1b_render as m10_1b  # noqa: E402 (V10_1+V10_1B+v10設定を継承)

m9 = m10_1b.m9
m9.DS_NAME = "outdoor_siren_v11"
m9.DS = ROOT / "out" / f"dataset_{m9.DS_NAME}"
m9.PLAN = ROOT / "out" / "dataset_outdoor_siren_v11" / "plan"
m9.WORK = m9.DS / "work"


# ------------------------------------------------- v11core 専用サンプラ ----

def _make_mic(row: dict, rng: np.random.Generator):
    """coreの通常経路と同一のマイク規約（sample_scene_v9 ①と同じ乱数消費）。"""
    if row["motion"] == "static":
        return np.array(m9.MIC_STATIC), {"motion": "static"}
    v = float(rng.uniform(*m9.WALK_SPEED))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    x0 = -dirx * v * 5.0
    mic = np.array([[0.0, x0, 0.0, m9.MIC_STATIC[2]],
                    [m9.CLIP, x0 + dirx * v * m9.CLIP, 0.0, m9.MIC_STATIC[2]]])
    return mic, {"motion": "walk", "walk_speed_mps": v, "walk_dir_x": dirx}


def sample_v11core(row: dict, rng: np.random.Generator) -> dict:
    n_car = int(row["n_car"])
    mic, mic_rec = _make_mic(row, rng)
    s = {"row": dict(row), "mic": mic_rec, "sources": [], "_mic_arr": mic}

    # ② 車 n_car台（v10a経路と同じ構成式。t_cpaレンジのみn_carで切替）
    tcpa_rng = m9.CAR_TCPA if n_car == 1 else (4.0, 9.0)
    for j in range(n_car):
        if j == 0:
            sign = 1.0 if row["car_side"] == "L" else -1.0
            tier = row["danger_tier"]
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            tier = m9.TIERS[int(rng.integers(3))]
        vc = float(rng.uniform(*m9.SPEED["car_drive"]))
        dirx = 1.0 if rng.random() < 0.5 else -1.0
        z = float(rng.uniform(*m9.SRC_Z))
        dz = m9.MIC_STATIC[2] - z
        lo, hi = m9.TIER_CPA[tier]
        cpa = float(rng.uniform(max(lo, dz + 0.05), hi))
        y = float(np.sqrt(max(cpa * cpa - dz * dz, 1e-6))) * sign
        t_cpa = float(rng.uniform(*tcpa_rng))
        lv, l1m = m9._draw_level("car_drive", rng)
        f0 = 42.0 * float(rng.uniform(0.8, 1.2))
        x0 = float(m9._mic_pos_at(mic, t_cpa)[0]) - dirx * vc * t_cpa
        s["sources"].append({
            "class": "car_drive", "kind": "vehicle", "track": j,
            "wp": [[0.0, x0, y, z], [m9.CLIP, x0 + dirx * vc * m9.CLIP, y, z]],
            "t_on": 0.0, "t_off": m9.CLIP, "speed_mps": vc, "dir_x": dirx,
            "danger_tier": tier, "cpa_rel_target_m": cpa, "t_cpa_rel_s": t_cpa,
            "law_db": lv, "l1m_db": l1m,
            "params": {"f0": f0, "audio_seed": row["seed"] * 7 + 5 + j}})

    # ③④ 警告音（coreの通常経路と同一。同一クラスw2はtrack=1）
    for key in ("w1", "w2"):
        cls = row[f"{key}_class"]
        if not cls:
            continue
        track = 1 if (key == "w2" and cls == row["w1_class"]) else 0
        side = 1.0 if row[f"{key}_side"] == "L" else -1.0
        lv, l1m = m9._draw_level(cls, rng)
        params = m9._warn_params(cls, rng, row["seed"])
        if cls == "crossing":
            d = float(rng.uniform(*m9.CROSS_DIST))
            az = float(rng.uniform(5.0, 175.0)) * side
            p5 = m9._mic_pos_at(mic, 5.0)
            pos = [float(p5[0] + d * np.cos(np.radians(az))),
                   float(p5[1] + d * np.sin(np.radians(az))), m9.CROSS_Z]
            src = {"class": cls, "kind": "static",
                   "wp": [[0.0, *pos], [m9.CLIP, *pos]],
                   "t_on": 0.0, "t_off": m9.CLIP, "dist_draw_m": d,
                   "law_db": lv, "l1m_db": l1m, "params": params}
        else:
            v = float(rng.uniform(*m9.SPEED[cls]))
            dirx = 1.0 if rng.random() < 0.5 else -1.0
            yw = float(rng.uniform(*m9.WARN_OFFSET)) * side
            t_cpa = float(rng.uniform(*m9.WARN_TCPA))
            z = float(rng.uniform(*m9.SRC_Z))
            dur = float(rng.uniform(*m9.EVENT_DUR))
            t_on = float(rng.uniform(0.3, m9.CLIP - dur - 0.3))
            x0 = -dirx * v * t_cpa
            src = {"class": cls, "kind": "vehicle",
                   "wp": [[0.0, x0, yw, z], [m9.CLIP, x0 + dirx * v * m9.CLIP, yw, z]],
                   "t_on": round(t_on, 3), "t_off": round(t_on + dur, 3),
                   "speed_mps": v, "dir_x": dirx, "t_cpa_s": t_cpa,
                   "law_db": lv, "l1m_db": l1m, "params": params}
        src["track"] = track
        s["sources"].append(src)

    # ⑤ 暗騒音
    s["noise"] = {"dba": float(rng.uniform(*m9.NOISE_DBA)),
                  "seed": row["seed"] * 7919 + 13}
    return s


_orig_sample = m9.sample_scene_v9

# ピーク棄却再抽選（2026-07-27、mix3642 peak1.248の対応）:
# v11の新密度（同一クラスsiren×2=消防車2本の近接同時など）では、絶対較正
# 0dBFS≡143dB SPLのヘッドルームを合法な条件の裾が稀に超える。物理・較正・音源
# レシピ（生成式）は凍結のため、**ピーク超過クリップのみシーン内の全乱数抽選
# （音色型・法規音量・幾何）をsalt付きシードで引き直す**決定論的な棄却再抽選で
# 対応する（plan行の離散条件=クラス/側/tier/n_carは不変だが、siren型など
# クリップ内の離散抽選は変わり得る。mix3642実測: salt0=fire→salt1=wail。
# saltはscene.jsonに peak_resample_salt として記録、検品・再開・再現に影響なし）。
# ※precheckのバウンドがこの裾を見逃す理由も特定済み: CREST表のsiren=1.8はwail想定
#   （fire型は bell_mix=0.3 でも約2.3）＋fire基本361Hzは A特性−3dB ぶん線形RMSが
#   大きい。バウンドは参考値とし、正はこの生成時assert＋再抽選とする。
_PEAK_SALT = 0


def sample_scene_v11(row: dict) -> dict:
    if row.get("scenario") == "v11core":
        if _PEAK_SALT == 0:
            rng = np.random.default_rng(row["seed"])
        else:
            rng = np.random.default_rng([row["seed"], 20260727, _PEAK_SALT])
        s = sample_v11core(row, rng)
        if _PEAK_SALT:
            s["peak_resample_salt"] = _PEAK_SALT
        return s
    return _orig_sample(row)


m9.sample_scene_v9 = sample_scene_v11   # generate_clip/precheckの参照先を差し替え

_orig_generate_clip = m9.generate_clip


def generate_clip_v11(row: dict) -> None:
    """ピークassert超過時のみ salt=1,2,... で連続値を引き直す（決定論ラダー）。"""
    global _PEAK_SALT
    if row.get("scenario") != "v11core":
        return _orig_generate_clip(row)
    last = None
    for salt in range(9):
        _PEAK_SALT = salt
        try:
            return _orig_generate_clip(row)
        except AssertionError as e:
            if "peak" not in str(e):
                raise
            last = e
            print(f"[peak-resample] {row['clip_id']} salt={salt} NG ({e}) -> retry",
                  flush=True)
        finally:
            _PEAK_SALT = 0
    raise last


m9.generate_clip = generate_clip_v11


# ------------------------------------------------------------- precheck ----

def precheck_v11() -> int:
    """全7,200行の解析事前チェック（レンダなし）。m9.precheckのv11適応版:
    setはcoreのみ、n_car実現値・警告実現値・track規約の全数検算を追加。"""
    rows = m9.load_plan("core")
    tgrid = np.arange(0.0, m9.CLIP, 0.05)
    peak_list, aud, car_leads, recv_by_class = [], {}, [], {}
    cpa_err_max, n_zero_src, n_bad = 0.0, 0, 0
    for row in rows:
        s = sample_scene_v11(row)
        mic = s.pop("_mic_arr")
        # v11追加検算: 実現値がplan行と一致するか
        cars = [x for x in s["sources"] if x["class"] == "car_drive"]
        warns = [x for x in s["sources"] if x["class"] != "car_drive"]
        ok = (len(cars) == int(row["n_car"])
              and len(warns) == row["n_warnings"]
              and [c["track"] for c in cars] == list(range(len(cars))))
        if row["n_warnings"] == 2 and row["w1_class"] == row["w2_class"]:
            ok = ok and [w["track"] for w in warns] == [0, 1]
        if int(row["n_car"]) >= 1:
            ok = ok and cars[0]["danger_tier"] == row["danger_tier"]
        if not ok:
            n_bad += 1
            print(f"  [BAD] {row['clip_id']}")
        if not s["sources"]:
            n_zero_src += 1
        noise_dba = s["noise"]["dba"]
        peak_bound = 4.0 * 10.0 ** ((noise_dba - m9.K_RMS_SPL) / 20.0)
        for src in s["sources"]:
            dist = m9._dist_series(src["wp"], mic, tgrid)
            active = (tgrid >= src["t_on"]) & (tgrid < src["t_off"])
            recv = src["l1m_db"] - 20.0 * np.log10(np.maximum(dist, 0.3))
            dmin = float(np.min(dist[active]))
            wpm = np.array(src["wp"], float)
            wpm[:, 3] = 2.0 * m9.GROUND_Z - wpm[:, 3]
            dmin_m = float(np.min(m9._dist_series(wpm, mic, tgrid)))
            rms = 10.0 ** ((src["l1m_db"] - m9.K_RMS_SPL) / 20.0) / dmin
            peak_bound += m9.CREST[src["class"]] * rms * (1.0 + dmin / dmin_m)
            frac = float(np.mean(recv[active] >= noise_dba))
            aud.setdefault(src["class"], []).append(frac)
            p_geo = float(np.mean(1.0 / np.maximum(dist[active], 0.3) ** 2))
            recv_by_class.setdefault(src["class"], []).append(
                src["l1m_db"] + 10.0 * np.log10(p_geo))
            if src["class"] == "car_drive":
                cpa_err_max = max(cpa_err_max, abs(float(np.min(dist))
                                                   - src["cpa_rel_target_m"]))
                audible_t = tgrid[recv >= noise_dba]
                car_leads.append(src["t_cpa_rel_s"] - float(audible_t[0])
                                 if len(audible_t) else -1.0)
        peak_list.append((float(peak_bound), row["clip_id"]))

    peak_list.sort(reverse=True)
    leads = np.array(car_leads)
    lines = ["# v11 生成前チェック（step11_v11_render.py precheck、レンダなしの解析計算）",
             "",
             f"- 行数: {len(rows)}（coreのみ。評価専用セットはv10とビット同一のため対象外）",
             f"- plan行との実現値一致（n_car/警告数/track/1台目tier）: 不一致 {n_bad} 行",
             f"- 音源ゼロ（純静穏=空ラベル）: {n_zero_src} 行"
             "（設計値582。Colabは空ラベル許容preprocで学習に載せる）",
             f"- 相対CPAの検算: 全車の設計値との最大差 {cpa_err_max:.3f} m"]
    lines.append("- ピーク上限（クレスト率×反射同相の保守的バウンド）上位5:")
    for pb, cid in peak_list[:5]:
        lines.append(f"    {cid}: {pb:.3f}")
    lines.append(f"  -> 最大 {peak_list[0][0]:.3f}（1.0未満なら本番assertも確実。"
                 "超える行はスモークで実測ピークを確認する）")
    lines.append("- クラス別の可聴フレーム率の予測（発音窓内、吸収・反射無視の点近似）:")
    for k in sorted(aud):
        v = np.array(aud[k])
        lines.append(f"    {k}: mean={v.mean():.2f} p10={np.percentile(v, 10):.2f} "
                     f"完全不可聴率={(v < 0.05).mean():.2%} (n={len(v)})")
    if len(leads):
        lines.append(f"- 車のオラクル・リードタイム予測（可聴開始→相対CPA): "
                     f"median={np.median(leads):.1f}s  >=2.5s: "
                     f"{(leads >= 2.5).mean():.1%}  >=2.0s: "
                     f"{(leads >= 2.0).mean():.1%}  "
                     f"クリップ内不可聴: {(leads < 0).mean():.1%}")
    lines.append("- 音量AUCの事前予測（受音レベルのみでのクラス対判別）:")
    cl = sorted(recv_by_class)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a = np.asarray(recv_by_class[cl[i]])[:, None]
            b = np.asarray(recv_by_class[cl[j]])[None, :]
            auc = float(((a > b).mean() + 0.5 * (a == b).mean()))
            auc = max(auc, 1.0 - auc)
            lines.append(f"    {cl[i]} vs {cl[j]}: AUC={auc:.3f}")
    report = "\n".join(lines) + "\n"
    outdir = m9.DS / "plan"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "precheck_report.md").write_text(report, encoding="utf-8")
    print(report)
    assert n_bad == 0, "plan行と実現値の不一致あり"
    return 0


# ------------------------------------------------------------- preflight ----

# スモーク対象: 代表条件を網羅する行を条件で自動選抜する
_SMOKE_CONDS = [
    ("純静穏(car0,warn0,static)", lambda r: int(r["n_car"]) == 0
     and r["n_warnings"] == 0 and r["motion"] == "static"),
    ("警告のみ(car0,warn2同一クラス,walk)", lambda r: int(r["n_car"]) == 0
     and r["n_warnings"] == 2 and r["w1_class"] == r["w2_class"]
     and r["motion"] == "walk"),
    ("車1台+踏切(walk)", lambda r: int(r["n_car"]) == 1
     and r["w1_class"] == "crossing" and r["n_warnings"] == 1
     and r["motion"] == "walk"),
    ("車2台+warn1(static)", lambda r: int(r["n_car"]) == 2
     and r["n_warnings"] == 1 and r["motion"] == "static"),
    ("車3台+warn2異クラス(walk)", lambda r: int(r["n_car"]) == 3
     and r["n_warnings"] == 2 and r["w1_class"] != r["w2_class"]
     and r["motion"] == "walk"),
    ("車1台+warn0(static,critical)", lambda r: int(r["n_car"]) == 1
     and r["n_warnings"] == 0 and r["motion"] == "static"
     and r["danger_tier"] == "critical"),
    # precheckのピーク上限バウンドが唯一1.0を超えた行（1.046、保守的バウンド）。
    # 実測ピーク<0.99をスモークで確認する（precheck_report.md 2026-07-27）
    ("ピーク上限最大行", lambda r: r["clip_id"] == "fold1_room1_mix4568"),
]


def preflight() -> int:
    rows = m9.load_plan("core")
    # 1) 決定論: 同一行を2回サンプル → 完全一致（代表100行）
    for row in rows[::72][:100]:
        a = json.dumps(sample_scene_v11(dict(row)), sort_keys=True, default=str)
        b = json.dumps(sample_scene_v11(dict(row)), sort_keys=True, default=str)
        assert a == b, f"determinism NG: {row['clip_id']}"
    print("preflight 1) 決定論（代表100行の2回サンプル一致）: OK")
    # 2) スモーク生成＋検品（代表6条件）
    picks = []
    for name, cond in _SMOKE_CONDS:
        row = next((r for r in rows if cond(r)), None)
        assert row is not None, f"smoke条件に合う行がない: {name}"
        picks.append((name, row))
        print(f"preflight 2) smoke: {row['clip_id']} <- {name}")
    for name, row in picks:
        m9.generate_clip(row)
    fail = m9.inspect_all()
    print(f"preflight 2) スモーク{len(picks)}本 検品 fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    if mode == "precheck":
        sys.exit(precheck_v11())
    elif mode == "preflight":
        sys.exit(preflight())
    elif mode == "gen":
        rows = m9.load_plan("core")
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        for i in range(int(a), int(b) + 1):
            m9.generate_clip(rows[i])
    elif mode == "inspect":
        sys.exit(1 if m9.inspect_all() else 0)
    elif mode == "pack":
        m9.pack()
    else:
        print("usage: precheck | preflight | gen A-B | inspect | pack")
        sys.exit(1)
