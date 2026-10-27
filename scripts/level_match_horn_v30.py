# -*- coding: utf-8 -*-
"""クラクションの地面反射の二重がけ（約 2.3 dB）の直し（Fable 仕様（先行分）§2・決定 ⑪・2026-10-26）: 基準の置き方で描いた
反射込みの 7 m の受音と宣言値の差 Δ を**本番の描画の道で**出し、v30 の定数 HORN_REFL_DELTA_DB（小数 2 桁）にする。

なぜ: クラクションの大きさは法規の「前方 7 m の値」lv（87〜112 dB・step11_v10_render の LAW["horn"]）を 20 log10(7) で 1 m に戻し、
**直接音だけ**に当てている。そのあと地面の鏡像（係数 1.0）が足されるので、7 m の受音は lv より大きい。
定義（Fable 10/26 14:39 の答え (b)・サイレンの #61 の siren_delta_db〔step11_v24_render.py:65・level_match_v24.siren_tables〕と同じ）:
  Δ = 反射込みの 7 m の受音（LAFmax）− 宣言値 lv。7 m までの伝わり（1/r・空気の吸収・経路の長さ）と間引きの小さな差も Δ に入る。
  v30 は l1m = lv + 20 log10(7) − Δ にする → 代表型の反射込みの 7 m が lv に合う。
Δ は確定値ではなく**基準点の代表値**（Astra R30-11）: 代表型 = 単音・長押し（1 押し 1.4 s）・音源の高さ = 描画のクラクションの高さの真ん中
（step11_v29_render.SRC_Z_CLASS["horn"] を読む）・マイク (0, 0, 1.0 m)。±0.1 dB は代表型だけに求め、二音・短押し・音源の高さの両端・
マイクの高さ 1.4／2.1 m は同じ Δ で描いた残差（受音 − lv）を md に記録する。
道（本番と同じ部品）: m9._warn_params("horn")（v23 の音色の抽選）→ 押しの列（params["presses"]）→ m9._make_dry（v23 のドライ音 → v24 の
LAFmax 合わせで l1m_db を直す）→ v13 と同じ RMS 合わせ（m9.gain_for_spl_a・v12 の −6 dB の包み込み・測った値に 6 dB を足し戻して宣言値と同じ約束で比べる）
→ m9._render_stem（direct と direct＋mirror）→ W の鳴っている窓の LAFmax（大きさの宣言 LAW と同じ指標）。LAeq は併記。

使い方: python scripts/level_match_horn_v30.py [--n 12] [--out md/design/level_match_horn_v30.md]
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v29_render as v29  # noqa: E402  鎖 v9 → … → v29（v23 の音色・v24 の最大値合わせ）
from outdoor_seld import level_v24 as L24  # noqa: E402
from outdoor_seld.calibration import spl_a  # noqa: E402

m9, v12 = v29.m9, v29.v12
FRONT_M = 7.0                                  # 法規の測り方: 前方 7 m
MIC_Z_REF = 1.0                                # 代表型: マイクの高さ 1.0 m
SRC_Z_REF = float(np.mean(v29.SRC_Z_CLASS["horn"]))
T_ON = 1.0
LONG = [[0.0, 1.4]]                            # 代表型: 長押し 1 回（alert_sounds_v23.HORN_PRESS_LONG 0.8〜2.0 s の真ん中）
SHORT = [[0.0, 0.3], [0.6, 0.3], [1.2, 0.3]]   # 短押し 3 回（HORN_PRESS_SHORT 0.15〜0.70・間 HORN_GAP 0.20〜0.50 の中ほど）


def horn_srcs(n: int, dual: bool, presses: list, start: int = 0, lv: float = 100.0) -> list:
    """本番の抽選（m9._warn_params の v23 の音色）から、単音／二音が指定どおりの音源を n 本（種 start から順に探す）。"""
    out, seed = [], start
    while len(out) < n:
        rng = np.random.default_rng([20261026, seed])
        p = m9._warn_params("horn", rng, 30000 + seed)
        seed += 1
        if bool(p.get("dual")) != dual:
            continue
        total = max(a + L for a, L in presses)
        out.append({"class": "horn", "kind": "vehicle", "track": 0, "params": dict(p, presses=[list(x) for x in presses]),
                    "t_on": T_ON, "t_off": round(T_ON + total, 4), "law_db": lv, "l1m_db": lv + 20.0 * np.log10(FRONT_M),
                    "match_max": True, "v24": {"match": "LAFmax"}, "_seed": seed - 1})
    return out


def levels(src: dict, src_z: float, mic_z: float) -> dict:
    """本番の道で描いて、直接音だけ・反射込みの W の LAFmax・LAeq（鳴っている窓・v12 の −6 dB を足し戻した dB）。"""
    src = dict(src, params=dict(src["params"]))
    dry = m9._window(m9._make_dry(src), src["t_on"], src["t_off"])           # v23 → v24（l1m_db が LAFmax 合わせに直る）
    a0, a1 = int(src["t_on"] * m9.FS_SIM), int(src["t_off"] * m9.FS_SIM)
    g = m9.gain_for_spl_a(dry[a0:a1], m9.FS_SIM, src["l1m_db"])               # v13 と同じ（v12 の −6 dB 込み）
    wp = np.array([[0.0, FRONT_M, 0.0, src_z], [m9.CLIP, FRONT_M, 0.0, src_z]])
    mic = np.array([0.0, 0.0, mic_z])
    stem_d, stem_wr = m9._render_stem(dry * g, wp, mic, m9.sound_speed(m9.TEMP_C))
    b0, b1 = int(src["t_on"] * m9.FS_OUT), int(src["t_off"] * m9.FS_OUT) + int(0.3 * m9.FS_OUT)   # LAFmax: 伝わりの遅れと余韻の分だけ窓の後ろを伸ばす
    lag = int(round(np.hypot(FRONT_M, src_z - mic_z) / m9.sound_speed(m9.TEMP_C) * m9.FS_OUT))
    e0, e1 = int(src["t_on"] * m9.FS_OUT) + lag, int(src["t_off"] * m9.FS_OUT) + lag              # LAeq: 鳴っている窓を伝わりの遅れだけずらす（押しの間の無音は含む）
    hr = float(v12.V12_HEADROOM_DB)
    return {"d_max": L24.lafmax_db(stem_d[0, b0:b1], m9.FS_OUT) + hr, "r_max": L24.lafmax_db(stem_wr[0, b0:b1], m9.FS_OUT) + hr,
            "d_eq": spl_a(stem_d[0, e0:e1], m9.FS_OUT) + hr, "r_eq": spl_a(stem_wr[0, e0:e1], m9.FS_OUT) + hr,
            "f_lo": float(src["params"].get("f_lo", np.nan)), "dual": bool(src["params"].get("dual", False))}


def with_delta(src: dict, delta: float) -> dict:
    s = dict(src, params=dict(src["params"]))
    s["l1m_db"] = float(src["l1m_db"]) - delta
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="代表型の音色の本数（Δ の平均に使う）")
    ap.add_argument("--n-resid", type=int, default=4, help="残差を測るほかの型の音色の本数")
    ap.add_argument("--out", default="md/design/level_match_horn_v30.md")
    a = ap.parse_args()
    assert m9.ablate.ground_enabled(), "地面の反射が切られている（ABLATE）。本番と同じ道で測る"
    lv = 100.0
    rep = horn_srcs(a.n, False, LONG)
    rows = []
    for s in rep:
        r = levels(s, SRC_Z_REF, MIC_Z_REF)
        rows.append(r)
        print(f"代表型 種 {s['_seed']:3d} f_lo {r['f_lo']:6.1f} Hz: 直接 {r['d_max']:6.2f}・反射込み {r['r_max']:6.2f} dB(A) LAFmax → 受音 − lv {r['r_max'] - lv:5.2f}"
              f"（反射の上乗せ {r['r_max'] - r['d_max']:5.2f}・LAeq の受音 − lv {r['r_eq'] - lv:5.2f}）", flush=True)
    excess = np.array([r["r_max"] - lv for r in rows])
    refl = np.array([r["r_max"] - r["d_max"] for r in rows])
    direct = np.array([r["d_max"] - lv for r in rows])
    delta = round(float(np.mean(excess)), 2)
    print(f"Δ（代表型 {a.n} 本の受音 − lv の平均）= {delta:.2f} dB", flush=True)
    # 残差: 同じ Δ を引いて描いた受音 − lv（代表型以外は記録だけ・±0.1 は求めない）
    lo, hi = v29.SRC_Z_CLASS["horn"]
    cases = [("代表型（単音・長押し・音源 %.2f・マイク %.2f）・別の音色" % (SRC_Z_REF, MIC_Z_REF), False, LONG, SRC_Z_REF, MIC_Z_REF, 500),
             ("二音・長押し", True, LONG, SRC_Z_REF, MIC_Z_REF, 600),
             ("単音・短押し 3 回", False, SHORT, SRC_Z_REF, MIC_Z_REF, 700),
             ("二音・短押し 3 回", True, SHORT, SRC_Z_REF, MIC_Z_REF, 800),
             (f"音源の高さ {lo:g} m", False, LONG, lo, MIC_Z_REF, 900),
             (f"音源の高さ {hi:g} m", False, LONG, hi, MIC_Z_REF, 1000),
             ("マイクの高さ 1.4 m", False, LONG, SRC_Z_REF, 1.4, 1100),
             ("マイクの高さ 2.1 m", False, LONG, SRC_Z_REF, 2.1, 1200)]
    resid = []
    for name, dual, presses, sz, mz, start in cases:
        rs = [levels(with_delta(s, delta), sz, mz) for s in horn_srcs(a.n_resid, dual, presses, start)]
        e = np.array([r["r_max"] - lv for r in rs]); q = np.array([r["r_eq"] - lv for r in rs])
        resid.append((name, float(np.mean(e)), float(np.min(e)), float(np.max(e)), float(np.mean(q))))
        print(f"残差 {name}: 受音 − lv 平均 {resid[-1][1]:+.3f}（{resid[-1][2]:+.3f}〜{resid[-1][3]:+.3f}）・LAeq {resid[-1][4]:+.3f}", flush=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# クラクションの地面反射の Δ（v30・先行分 §2・決定 ⑪・{now}・`scripts/level_match_horn_v30.py --n {a.n} --n-resid {a.n_resid}` の出力）", "",
        f"- 定数: **HORN_REFL_DELTA_DB = {delta:.2f} dB**（`scripts/step11_v30_render.py`）。v30 は l1m = lv + 20 log10(7) − Δ。",
        "- 定義（Fable 10/26 14:39・(b)）: Δ = 代表型の反射込みの 7 m の受音（LAFmax）− 宣言値 lv（サイレンの #61 と同じ）。7 m までの伝わり（1/r・空気の吸収・経路の長さ）と間引きの小さな差も Δ に入る。",
        f"- 代表型（Astra R30-11）: 単音・長押し 1 回 {LONG[0][1]:g} s・音源の高さ {SRC_Z_REF:.2f} m（`step11_v29_render.SRC_Z_CLASS[\"horn\"]` = {lo:g}〜{hi:g} の真ん中）・"
        f"マイク {MIC_Z_REF:g} m・前方 {FRONT_M:g} m・静止・地面は本番の鏡像（係数 1.0）。Δ は**基準点の代表値**で確定値ではない。",
        "- 道: `m9._warn_params(\"horn\")`（v23 の音色）→ 押しの列 → `m9._make_dry`（v23 → v24 の LAFmax 合わせ）→ v13 と同じ RMS 合わせ（v12 の −6 dB 込み・測った値に 6 dB を足し戻して宣言値と同じ約束で比べる）"
        "→ `m9._render_stem` → W の窓の LAFmax（大きさの宣言と同じ指標）。",
        f"- 代表型 {a.n} 本: 受音 − lv 平均 {np.mean(excess):.3f}・最小 {np.min(excess):.3f}・最大 {np.max(excess):.3f} dB。"
        f"内訳の目安: 反射の上乗せ（反射込み − 直接音だけ）平均 {np.mean(refl):.3f}・直接音だけの 7 m − lv 平均 {np.mean(direct):+.3f}。",
        "- 残差（同じ Δ を引いて描いた反射込みの 7 m の受音 − lv・LAFmax。±0.1 dB は代表型だけに求める・ほかは記録。"
        "LAeq は鳴っている窓〔伝わりの遅れだけずらす・押しの間の無音を含む〕の平均なので LAFmax より小さく、参考）:",
    ]
    lines += [f"  - {name}: 平均 {m:+.3f}・最小 {mn:+.3f}・最大 {mx:+.3f} dB（LAeq の平均 {q:+.3f}）" for name, m, mn, mx, q in resid]
    lines += [
        "- 既知の小さな偏り（0.2 dB 未満・今回は直さない・Fable 10/26 14:39）: 最大値で合わせるほかの音（ベル・バック音・踏切）にも、直接音だけの受音が宣言値より"
        f" 約 {abs(float(np.mean(direct))):.2f} dB 小さくなる同じ形の差（7 m までの伝わり・間引き）がある。直すと試聴の対象が増えるため。",
        "- 代表型の音色ごと:",
    ]
    lines += [f"  - 種 {s['_seed']}: f_lo {r['f_lo']:.1f} Hz・受音 − lv {r['r_max'] - lv:.3f}・反射の上乗せ {r['r_max'] - r['d_max']:.3f}（LAeq の受音 − lv {r['r_eq'] - lv:.3f}）"
              for s, r in zip(rep, rows)]
    out = ROOT / a.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nHORN_REFL_DELTA_DB = {delta:.2f}  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
