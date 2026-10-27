# -*- coding: utf-8 -*-
"""先行分 §2（決定 ⑪・2026-10-26 Fable 仕様）の受け入れ試験: クラクションの地面反射の二重がけの直し（v30 の HORN_REFL_DELTA_DB）。

H1 代表型（単音・長押し・音源はクラクションの高さの真ん中・マイク 1.0 m・前方 7 m）で、Δ を出した音色とは別の音色 12 本を v30 の入口の中の
   apply_horn_delta で直して描くと、反射込みの LAFmax（鳴っている窓）− 宣言値 lv の平均が ±0.1 dB（Δ の定義 = #61 と同じ・Fable 10/26 (b)）。
   ほかの型の残差は md/design/level_match_horn_v30.md に記録（±0.1 は求めない・Astra R30-11）。
H2 plan のクラクションの本（静止のマイク）を、v30 の入口の中で Δ だけを切／入にして描くと、クラクションの stem は −Δ ± 0.1 dB・ほかの音源の stem と
   乱数の欄（scene.json）は同じ・scene.json に horn_refl_delta_db と l1m_before_horn_delta_db がある。
H3 ほかのクラス（サイレン・ベル・バック音・車・キック・バイク・自転車・踏切）の本は、Δ の切／入で FOA・ラベル・masks が bit 一致。
H4 v30 の入口の外（v29 の道）では apply_horn_delta は何もしない。
出力は一時フォルダ（終わったら消す・--keep で残す）。

使い方: PYTHONPATH=scripts:src python scripts/test_v30_horn_delta.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = []
HORN_ROW = "train_syn_clip00208"
OTHER_ROWS = {"train_syn_clip00097": "siren", "train_syn_clip00067": "bike_bell", "train_syn_clip00170": "backup_beep",
              "train_syn_clip00001": "car_drive", "train_syn_clip04806": "kickboard", "train_syn_clip05703": "motorcycle",
              "train_syn_clip34008": "bicycle", "train_syn_clip00003": "crossing"}


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'NG'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def main() -> int:
    import soundfile as sf
    import _run_v2final_gen as g
    import step11_v30_render as v30
    import level_match_horn_v30 as LM
    m9 = g.m9
    D = v30.HORN_REFL_DELTA_DB
    md = (ROOT / "md/design/level_match_horn_v30.md")
    check("H0 v30 の定数 HORN_REFL_DELTA_DB は小数 2 桁で、出し方の md に同じ値がある", round(D, 2) == D and md.is_file()
          and f"HORN_REFL_DELTA_DB = {D:.2f} dB" in md.read_text(encoding="utf-8"), f"Δ = {D}")

    # H1: 代表型・別の音色 12 本（Δ を出したのは種 0 から・ここは種 300 から）
    lv = 100.0
    lit, lit_eq = [], []
    for src in LM.horn_srcs(12, False, LM.LONG, start=300, lv=lv):
        s = {"sources": [src]}
        with v30.active():
            v30.apply_horn_delta(s)
        r = LM.levels(s["sources"][0], LM.SRC_Z_REF, LM.MIC_Z_REF)
        lit.append(r["r_max"] - lv); lit_eq.append(r["r_eq"] - lv)
    em, ee = np.array(lit), np.array(lit_eq)
    check("H1 代表型: 直したあとの反射込みの 7 m の LAFmax − lv の平均が ±0.1 dB（別の音色 12 本）", abs(float(np.mean(em))) <= 0.1,
          f"平均 {np.mean(em):+.3f}・最小 {np.min(em):+.3f}・最大 {np.max(em):+.3f} dB（LAeq − lv 平均 {np.mean(ee):+.3f}・参考）")
    s4 = {"sources": [dict(LM.horn_srcs(1, False, LM.LONG, start=300, lv=lv)[0])]}
    l0 = s4["sources"][0]["l1m_db"]
    v30.apply_horn_delta(s4)
    check("H4 v30 の入口の外では apply_horn_delta は何もしない（v29 の道は変わらない）", s4["sources"][0]["l1m_db"] == l0 and "horn_refl_delta_db" not in s4["sources"][0])

    rows = {r["clip_id"]: r for r in g.load_plan() if r["clip_id"] in set(OTHER_ROWS) | {HORN_ROW}}
    tmp = Path(tempfile.mkdtemp(prefix="test_v30_horn_delta_"))
    print("一時フォルダ:", tmp, flush=True)
    try:
        for on, sub in ((False, "off"), (True, "on")):
            v30.FEATURES["horn_delta"] = on                   # v30 の入口の中で Δ だけを切／入（ほかの v30 の直しは両方で同じ）
            g.point_to(tmp / sub)
            for cid in [HORN_ROW, *OTHER_ROWS]:
                v30.generate_clip(dict(rows[cid]))
        v30.FEATURES["horn_delta"] = True
        A, B = tmp / "off", tmp / "on"
        # H2: クラクションの本
        sa = json.loads((A / "work" / HORN_ROW / "scene.json").read_text())
        sb = json.loads((B / "work" / HORN_ROW / "scene.json").read_text())
        hb = [x for x in sb["sources"] if x["class"] == "horn"]
        check("H2a scene.json のクラクションに horn_refl_delta_db = Δ・l1m_before_horn_delta_db − Δ = l1m_decl_db",
              len(hb) >= 1 and all(x.get("horn_refl_delta_db") == D and abs(x["l1m_before_horn_delta_db"] - D - x["l1m_decl_db"]) < 1e-9 for x in hb)
              and not any("horn_refl_delta_db" in x for x in sa["sources"]), str([(x.get("horn_refl_delta_db"), x.get("l1m_decl_db")) for x in hb]))
        diffs, others_same = [], True
        for i, (xa, xb) in enumerate(zip(sa["sources"], sb["sources"])):
            fa, _ = sf.read(A / "work" / HORN_ROW / f"src{i}_{xa['class']}_withrefl_24k.flac", dtype="float64", always_2d=True)
            fb, _ = sf.read(B / "work" / HORN_ROW / f"src{i}_{xb['class']}_withrefl_24k.flac", dtype="float64", always_2d=True)
            if xa["class"] == "horn":
                a0, a1 = int(xa["t_on"] * 24000), int(xa["t_off"] * 24000)
                diffs.append(10 * np.log10(np.sum(fb[a0:a1, 0] ** 2) / np.sum(fa[a0:a1, 0] ** 2)))
            else:
                others_same &= bool(np.array_equal(fa, fb))
        check("H2b クラクションの stem の差 = −Δ ± 0.1 dB（同じ seed・直す前の道との比）", len(diffs) >= 1 and all(abs(d + D) <= 0.1 for d in diffs),
              f"差 {[round(float(d), 3) for d in diffs]} dB")
        check("H2c 同じ本のほかの音源の stem は bit 一致", others_same)
        drop = {"gen_seconds", "horn_refl_delta_db", "l1m_before_horn_delta_db", "l1m_db", "l1m_decl_db", "dry_gain",
                "recv_pred_db", "recv_meas_db", "recv_withrefl_db", "gate_snr_max_db", "gate_recv_max_dba", "audible_frac",
                "t_first_audible_s", "n_label_frames", "peak", "mix_spl_a", "n_label_rows",
                "gate_band_max_db", "gate_band_hz_at_max", "gate_band_n_invalid"}      # 決め直し 4（帯域ゲートが既定）: 帯の判定量も大きさで変わる欄

        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in drop}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o
        check("H2d scene.json は大きさの欄のほか同じ（乱数・時刻・軌道・音色）", strip(sa) == strip(sb))
        # H3: ほかのクラス
        for cid, cls in OTHER_ROWS.items():
            same = all((A / d / f"{cid}.{e}").read_bytes() == (B / d / f"{cid}.{e}").read_bytes()
                       for d, e in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")))
            sc = json.loads((B / "work" / cid / "scene.json").read_text())
            has = [x["class"] for x in sc["sources"]]
            check(f"H3 {cid}（{cls}）: 直しの切／入で FOA・ラベル・masks が bit 一致", same and cls in has and "horn" not in has, str(has))
    finally:
        if "--keep" in sys.argv:
            print("残した:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(ng)}/{len(RESULTS)} OK" + (f"  NG: {ng}" if ng else ""))
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
