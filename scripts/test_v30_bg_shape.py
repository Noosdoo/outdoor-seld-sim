# -*- coding: utf-8 -*-
"""本体 §2-3（背景の形の乱数・R1 (b)・2026-10-26）の受け入れ試験。

  S1 折れ点と傾きを固定して作った暗騒音（30 秒）のオクターブ帯のレベル差が指定の形に ±0.5 dB（125 Hz を基準・63〜8 kHz）。v12 の形と同じ節点（折れ点 63・傾き −4）は v12 の帯のレベルと ±0.5 dB
  S2 形の抽選: 折れ点は 63〜1,000 Hz（対数一様・中央値 ≈ 251 Hz）・傾きは v30 の既定 (ii) −6.5〜−2.5（2026-10-27・部品 noise_v30 の既定は −5〜−3 のまま）・
     別の乱数の流れ（rng([seed, SALT29, 21])）で、同じ seed（高さの双子）は同じ形・V30_KEYS は一意・v30 の既定は bg_shape 入
  S3 描画: v30 の入口で形を入にして描いた本は、形を切った本と場面の抽選（音源の軌道・時刻・大きさ・暗騒音の seed と dB(A)）が同じ・scene.json の noise に shape と幅（(ii)）・暗騒音の dB(A) は目標どおり
  S4 v30 の直しを全部切る（FEATURES 全部 False）と、v29 の道と FOA・ラベル・masks が bit 一致（歩く向き −1 の本・クラクションの本でも）＝差し込み口は切なら何もしない
出力は一時フォルダ（終わったら消す）。
使い方: PYTHONPATH=scripts:src python scripts/test_v30_bg_shape.py
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

from outdoor_seld.noise_v12 import urban_colored_noise  # noqa: E402
from outdoor_seld.noise_v30 import draw_shape, octave_levels, shape_anchors, shaped_colored_noise  # noqa: E402

RESULTS = []
CENTERS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'NG'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def want_rel(knee, slope):
    """指定の形のオクターブ帯のレベル（帯の中で PSD ∝ 10^(L/10)/f を積分・125 Hz を基準）。帯の中で傾きが変わる帯（63 Hz の帯など）も正しく比べる。"""
    af, adb = shape_anchors(knee, slope)
    out = []
    for fc in CENTERS:
        f = np.geomspace(max(fc / np.sqrt(2), 20.0), fc * np.sqrt(2), 4000)
        lv = np.interp(np.log2(f), np.log2(af), adb)
        out.append(10.0 * np.log10(np.trapezoid(10.0 ** (lv / 10.0) / f, f)))
    out = np.array(out)
    return out - out[1]


def main() -> int:
    fs, n = 24000, 24000 * 30
    worst = 0.0
    for knee, slope in ((125.0, -3.0), (500.0, -5.0), (1000.0, -4.0), (63.0, -3.5)):
        x = shaped_colored_noise(n, fs, np.random.default_rng(1), knee, slope)
        ob = octave_levels(x, fs); ob = ob - ob[1]
        worst = max(worst, float(np.max(np.abs(ob - want_rel(knee, slope)))))
    xv = shaped_colored_noise(n, fs, np.random.default_rng(2), 63.0, -4.0)
    x12 = urban_colored_noise(n, fs, np.random.default_rng(2))
    a, b = octave_levels(xv, fs), octave_levels(x12, fs)
    d12 = float(np.max(np.abs((a - a[1]) - (b - b[1]))))
    check("S1 固定の形 4 通りのオクターブ帯のレベル差が指定の形に ±0.5 dB・v12 と同じ節点（63・−4）は v12 の帯と ±0.5 dB", worst <= 0.5 and d12 <= 0.5,
          f"最大のずれ {worst:.2f} dB・v12 との差 {d12:.2f} dB")

    import _run_v2final_gen as g
    import step11_v30_render as v30
    shapes = [v30.bg_shape_of({}, {"seed": 9300000000 + i}) for i in range(2000)]
    kn = np.array([x["knee_hz"] for x in shapes]); sl = np.array([x["slope_db_oct"] for x in shapes])
    twin = v30.bg_shape_of({}, {"seed": 9300000007}) == shapes[7]
    med = float(np.median(kn))
    lib = [draw_shape(np.random.default_rng([9300000000 + i, 7, 21])) for i in range(500)]
    lsl = np.array([x["slope_db_oct"] for x in lib])
    check("S2 折れ点 63〜1,000 Hz の対数一様（中央値 ≈ 251）・傾きは v30 の既定 (ii) −6.5〜−2.5（部品の既定は −5〜−3 のまま）・同じ seed は同じ形・鍵は一意（v29 の KEYS ＋ bg_shape 21）・既定は入",
          kn.min() >= 63.0 and kn.max() <= 1000.0 and 200 < med < 315 and sl.min() >= -6.5 and sl.max() <= -2.5 and sl.min() < -5.0 and sl.max() > -3.0 and twin
          and lsl.min() >= -5.0 and lsl.max() <= -3.0 and v30.BG_SLOPE_RANGE == (-6.5, -2.5) and v30.FEATURES["bg_shape"] is True
          and len(set(v30.V30_KEYS.values())) == len(v30.V30_KEYS) and v30.V30_KEYS["bg_shape"] == 21,
          f"折れ点 {kn.min():.0f}〜{kn.max():.0f}（中央値 {med:.0f}）・傾き {sl.min():.2f}〜{sl.max():.2f}・部品の既定 {lsl.min():.2f}〜{lsl.max():.2f}")

    rows = {r["clip_id"]: r for r in g.load_plan() if r["clip_id"] in ("train_syn_clip00001", "train_syn_clip00009", "train_syn_clip00208")}
    tmp = Path(tempfile.mkdtemp(prefix="test_v30_bg_shape_"))
    try:
        A, B = tmp / "off", tmp / "on"
        bg0 = v30.FEATURES["bg_shape"]
        v30.FEATURES["bg_shape"] = False                  # 10/27 から既定は入 → 切の本は切を明示して描く
        try:
            g.point_to(A)
            v30.generate_clip(dict(rows["train_syn_clip00001"]))
            v30.FEATURES["bg_shape"] = True
            g.point_to(B)
            v30.generate_clip(dict(rows["train_syn_clip00001"]))
        finally:
            v30.FEATURES["bg_shape"] = bg0
        sa = json.loads((A / "work" / "train_syn_clip00001" / "scene.json").read_text())
        sb = json.loads((B / "work" / "train_syn_clip00001" / "scene.json").read_text())
        keep = ("class", "wp", "t_on", "t_off", "l1m_db", "law_db", "speed_mps", "params", "track")
        same_src = [{k: x.get(k) for k in keep} for x in sa["sources"]] == [{k: x.get(k) for k in keep} for x in sb["sources"]]
        shp = sb["noise"].get("shape")
        want = v30.bg_shape_of(sb, rows["train_syn_clip00001"])
        from outdoor_seld.calibration import spl_a
        import soundfile as sf
        check("S3 形を入にした本: 場面の抽選（軌道・時刻・大きさ・音色）と暗騒音の seed・dB(A) は形を切った本と同じ・scene.json の noise に shape（別の流れの値）",
              same_src and sa["noise"]["seed"] == sb["noise"]["seed"] and sa["noise"]["dba"] == sb["noise"]["dba"] and "shape" not in sa["noise"]
              and shp is not None and abs(shp["knee_hz"] - want["knee_hz"]) < 1e-3 and abs(shp["slope_db_oct"] - want["slope_db_oct"]) < 1e-4
              and shp.get("slope_range_db_oct") == [-6.5, -2.5] and shp.get("knee_range_hz") == [63.0, 1000.0],
              f"shape {shp}")
        C = tmp / "alloff"
        g.point_to(C)
        old = dict(v30.FEATURES)
        for k in v30.FEATURES:
            v30.FEATURES[k] = False
        try:
            for c in rows:
                v30.generate_clip(dict(rows[c]))
        finally:
            v30.FEATURES.update(old)
        D = tmp / "v29"
        g.point_to(D)
        for c in rows:
            g.v19r.generate_clip(dict(rows[c]))
        same = all((C / d / f"{c}.{e}").read_bytes() == (D / d / f"{c}.{e}").read_bytes()
                   for c in rows for d, e in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")))
        check("S4 v30 の直しを全部切ると v29 の道と FOA・ラベル・masks が bit 一致（静止・歩く向き −1・クラクションの 3 本）", same)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(ng)}/{len(RESULTS)} OK" + (f"  NG: {ng}" if ng else ""))
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
