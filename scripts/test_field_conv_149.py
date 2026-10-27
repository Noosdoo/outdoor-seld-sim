# -*- coding: utf-8 -*-
"""先行分 §3（決定 ③・2026-10-26 Fable 仕様・Astra R30-08）の受け入れ試験: 実録の換算を合成の約束（実効 149 dB SPL）にそろえる**新しい入口**。
既定（convention 省略＝v9_143）は旧のまま（既存の試験の期待値は書き換えない）。149 は明示の引数 convention="v12_149"／--convention v12_149 で −6 dB。

  F1 calib_gain_db: 既定 = 計器 − 録音の A 特性レベル（旧の式と同じ値）・v12_149 = 既定 − 6.00 dB（合成の正弦波の録音で）
  F2 定数: step11_v12_render.V12_HEADROOM_DB == calibration.SYNTH_HEADROOM_DB == 6.0・FIELD_FULLSCALE_SINE_SPL == 149.0・約束は v9_143／v12_149
  F3 calibration_record の json: 既定は convention v9_143・fullscale_sine_spl 143・headroom_db 0・gain は旧と同じ／v12_149 は 149・6・gain −6
  F4 変換の出力: 校正済みの変換（convert_with_gain）と全長の変換（convert）で、v12_149 の出力が既定のちょうど −6 dB（標本の比）
  F5 知らない約束は止める（黙って既定にしない）
  F6 命令の入口: step19_realsmoke_convert --calib … --gain-only --convention v12_149 の gain-db が既定 − 6・json に約束
  F7 field_calib_check: --convention v12_149 で gain_db が既定 − 6・json に convention（照らし合わせの判定と終了コードは同じ）
  F8 家の 10/7 の較正録音（raw_demo/261007_001.WAV・学内の実録ではない・あれば）: 記録の値 −42.335 と既定のゲインが同じ・v12_149 はちょうど −6.00
使い方: python scripts/test_field_conv_149.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step19_realsmoke_convert as cv  # noqa: E402
from outdoor_seld import calibration as C  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'NG'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def write_sine(p: Path, seconds: float = 4.0, fs: int = 96000, amp: float = 0.02) -> None:
    t = np.arange(int(seconds * fs)) / fs
    w = amp * np.sin(2 * np.pi * 1000.0 * t)
    sf.write(str(p), np.stack([w, 0.3 * w, 0.1 * w, 0.2 * w], axis=1), fs, subtype="FLOAT")


def main() -> int:
    from scipy.signal import resample_poly
    py = sys.executable
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_sine(td / "S1_calib.wav")
        write_sine(td / "S1_take01.wav", amp=0.05)
        g0 = cv.calib_gain_db(td / "S1_calib.wav", 55.0, (0.5, 3.5))
        w, _ = sf.read(str(td / "S1_calib.wav"), dtype="float64")
        seg = resample_poly(w[int(0.5 * 96000):int(3.5 * 96000), 0], 1, 4)      # 旧の calib_gain_db と同じ切り方（窓だけ読んで 24 kHz に）
        g_old = 55.0 - cv.spl_a(seg, 24000)
        g1 = cv.calib_gain_db(td / "S1_calib.wav", 55.0, (0.5, 3.5), convention="v12_149")
        g9 = cv.calib_gain_db(td / "S1_calib.wav", 55.0, (0.5, 3.5), convention="v9_143")
        check("F1 calib_gain_db: 既定 = 旧の式（計器 − 測った値）・v9_143 = 既定・v12_149 = 既定 − 6.00", abs(g0 - g_old) < 1e-9 and g9 == g0 and abs((g0 - g1) - 6.0) < 1e-12,
              f"既定 {g0:+.4f}・v12_149 {g1:+.4f}")

        import step11_v12_render as v12
        check("F2 V12_HEADROOM_DB == SYNTH_HEADROOM_DB == 6.0・FIELD_FULLSCALE_SINE_SPL == 149.0・FULLSCALE_SINE_SPL は 143 のまま・約束は v9_143／v12_149（既定 v9_143）",
              v12.V12_HEADROOM_DB == C.SYNTH_HEADROOM_DB == 6.0 and C.FIELD_FULLSCALE_SINE_SPL == 149.0 and C.FULLSCALE_SINE_SPL == 143.0
              and set(C.FIELD_CONVENTIONS) == {"v9_143", "v12_149"} and C.FIELD_CONVENTION_DEFAULT == "v9_143")

        r0 = cv.calibration_record(td / "S1_calib.wav", 55.0, (0.5, 3.5), "S1_calib", td / "conv0")
        r1 = cv.calibration_record(td / "S1_calib.wav", 55.0, (0.5, 3.5), "S1_calib", td / "conv1", convention="v12_149")
        j1 = json.loads((td / "conv1" / "calibration_S1_calib.json").read_text(encoding="utf-8"))
        check("F3 較正の json: 既定は v9_143・143・0・gain は旧と同じ／v12_149 は 149・6・gain −6（json の中身も同じ）",
              (r0["convention"], r0["fullscale_sine_spl"], r0["headroom_db"]) == ("v9_143", 143.0, 0.0) and r0["gain_db"] == round(g0, 3)
              and (j1["convention"], j1["fullscale_sine_spl"], j1["headroom_db"]) == ("v12_149", 149.0, 6.0) and abs(j1["gain_db"] - round(g1, 3)) < 1e-9,
              f"{r0['gain_db']} / {j1['gain_db']}")

        # 変換の出力は 24 bit なので、丸めが効かない大きさ（計器 110 dB・出力が 0.1 前後）で比べる
        q0 = cv.calib_gain_db(td / "S1_calib.wav", 110.0, (0.5, 3.5))
        q1 = cv.calib_gain_db(td / "S1_calib.wav", 110.0, (0.5, 3.5), convention="v12_149")
        o0 = cv.convert_with_gain(td / "S1_take01.wav", td / "cw0", q0, 0, 0, 0)
        o1 = cv.convert_with_gain(td / "S1_take01.wav", td / "cw1", q1, 0, 0, 0)
        y0, _ = sf.read(str(o0), dtype="float64"); y1, _ = sf.read(str(o1), dtype="float64")
        rw = 20 * np.log10(np.sqrt(np.mean(y1[:, 0] ** 2)) / np.sqrt(np.mean(y0[:, 0] ** 2)))
        p0 = cv.convert(td / "S1_take01.wav", td / "cf0", 110.0, (0.5, 3.5), 0, 0, 0)
        p1 = cv.convert(td / "S1_take01.wav", td / "cf1", 110.0, (0.5, 3.5), 0, 0, 0, convention="v12_149")
        z0, _ = sf.read(str(p0), dtype="float64"); z1, _ = sf.read(str(p1), dtype="float64")
        rz = 20 * np.log10(np.sqrt(np.mean(z1[:, 0] ** 2)) / np.sqrt(np.mean(z0[:, 0] ** 2)))
        check("F4 変換の出力: 校正済みの変換・全長の変換とも v12_149 は既定のちょうど −6 dB（24 bit の丸めの範囲）", abs(rw + 6.0) < 0.001 and abs(rz + 6.0) < 0.001,
              f"校正済み {rw:+.4f}・全長 {rz:+.4f} dB")

        try:
            cv.calib_gain_db(td / "S1_calib.wav", 55.0, (0.5, 3.5), convention="v12_155"); stopped = False
        except ValueError:
            stopped = True
        check("F5 知らない約束は止める", stopped)

        outs = {}
        for conv in (None, "v12_149"):
            args = [py, str(ROOT / "scripts/step19_realsmoke_convert.py"), "--in", str(td), "--calib", str(td / "S1_calib.wav"), "--laeq", "55.0",
                    "--laeq-window", "0.5-3.5", "--gain-only", "--out", str(td / f"cli_{conv}")] + (["--convention", conv] if conv else [])
            r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            outs[conv] = json.loads((td / f"cli_{conv}" / "calibration_S1_calib.json").read_text(encoding="utf-8")) if r.returncode == 0 else None
        ok6 = outs[None] is not None and outs["v12_149"] is not None and abs(outs[None]["gain_db"] - outs["v12_149"]["gain_db"] - 6.0) < 1e-6 \
            and outs[None]["convention"] == "v9_143" and outs["v12_149"]["convention"] == "v12_149"
        check("F6 命令の入口 --convention v12_149: gain-db が既定 − 6・json に約束", ok6, str({k: (v or {}).get("gain_db") for k, v in outs.items()}))

        import test_field_calib_check as tfc
        import field_calib_check as fc
        import datetime as dt
        wav = td / "261029_001.WAV"; tfc.write_wav(wav)
        rec = fc.rec_levels(wav)
        rng = np.random.default_rng(1)
        t0 = dt.datetime(2026, 10, 29, 9, 59, 40)
        vals = {t0 + dt.timedelta(seconds=n): 45.0 + rng.normal(0, 1) for n in range(120)}
        for i, l in enumerate(rec):
            vals[tfc.START + dt.timedelta(seconds=i + 2)] = l + 50.0 + rng.normal(0, 0.05)
        slm = td / "slm_20261029_095940.TXT"; tfc.write_slm(slm, vals)
        a = fc.check(wav, [slm], None, (5, 60))
        b = fc.check(wav, [slm], None, (5, 60), convention="v12_149")
        rc = subprocess.run([py, str(ROOT / "scripts/field_calib_check.py"), "--wav", str(wav), "--slm", str(slm), "--window", "5-60",
                             "--json", str(td / "fc.json"), "--convention", "v12_149"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
        jj = json.loads((td / "fc.json").read_text(encoding="utf-8")) if (td / "fc.json").is_file() else {}
        check("F7 field_calib_check: v12_149 の gain_db = 既定 − 6・判定と終了コードは同じ・json に convention（命令の --convention も）",
              a["gain_db"] is not None and abs(a["gain_db"] - b["gain_db"] - 6.0) < 0.011 and a["judgment"] == b["judgment"] and a["exit_code"] == b["exit_code"]
              and a["convention"] == "v9_143" and b["convention"] == "v12_149" and jj.get("convention") == "v12_149" and rc.returncode == a["exit_code"],
              f"既定 {a['gain_db']}・v12_149 {b['gain_db']}・{a['judgment']}")

    import demo_bike_check_prepare as dp
    with tempfile.TemporaryDirectory() as td2:
        old = Path(td2) / "old.json"; old.write_text(json.dumps({"gain_db": -42.335}), encoding="utf-8")                       # 鍵の無い古い json（143）
        new = Path(td2) / "new.json"; new.write_text(json.dumps({"gain_db": -48.335, "convention": "v12_149"}), encoding="utf-8")
        g = [dp.gain_from_calib_json(old, None), dp.gain_from_calib_json(old, "v12_149"), dp.gain_from_calib_json(new, "v12_149"), dp.gain_from_calib_json(new, None)]
    check("F9 家のデモの json の読み（Fable の守り）: json の約束を見て二重に引かない（古い json → 143 は −42.335・149 は −48.335／149 の json → 149 は −48.335・143 は −42.335）",
          [round(x, 3) for x in g] == [-42.335, -48.335, -48.335, -42.335] and dp.GAIN_DB == -42.335, str(g))

    home = ROOT / "raw_demo" / "261007_001.WAV"
    if home.is_file():
        rec0 = json.loads((ROOT / "out/field_demo/conv/calibration_261007_001.json").read_text(encoding="utf-8"))
        h0 = cv.calib_gain_db(home, rec0["laeq_dba"], tuple(rec0["laeq_window_s"]))
        h1 = cv.calib_gain_db(home, rec0["laeq_dba"], tuple(rec0["laeq_window_s"]), convention="v12_149")
        check("F8 家の 10/7 の較正録音（学内の実録ではない）: 既定のゲインが記録の −42.335 と同じ・v12_149 はちょうど −6.00", abs(round(h0, 3) - rec0["gain_db"]) < 1e-9 and abs(h0 - h1 - 6.0) < 1e-12,
              f"既定 {h0:+.3f}・v12_149 {h1:+.3f} dB")
    else:
        print("（F8 は raw_demo/261007_001.WAV が無いので飛ばす）")
    ng = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(ng)}/{len(RESULTS)} OK" + (f"  NG: {ng}" if ng else ""))
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
