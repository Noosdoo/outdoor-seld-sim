"""家で録った 10/7 の自転車で、今の主のモデルの手応えを確かめる準備（2026-10-25 Opus・本人 20:25「測定の 10 本も足しましょう」）。

使うのは 10/7 に家で録った調整用のデータだけ（評価には使わない＝10/7 の決まり ⑥）。学内の実録（raw/）には触らない。
  - 走行音の測定: raw_bicycle_measure/261007_007.WAV（96 kHz・ベルなしの通過 10 回）と reads/reads_A.csv（前輪が目印を通った録音の秒）
  - デモ 3 本: out/field_demo/clips/eval_field_clip01001〜01003.flac（10/7 に step19b で切り出し済み）

やること:
  1. 測定の 10 回を、前輪が x = 0 を通った秒を 8 秒目に置いた 10 秒に切る（step19b の plan_event_cut・cut_audio と同じ処理・
     校正 261007_001 のゲイン −42.335 dB・傾きの補正 0（デモと同じ）・24 kHz・PCM_24）。名前は eval_field_clip90001〜90010（90 = この確かめ用の番号）。
  2. 読みの座標（x = 目印の番号・y = 線の +1.0）とマイクの座標（x 前・y 左）の対応 sx・sy を、録音の音の向きで確かめる（モデルは使わない）:
     2〜8 kHz で、通過の瞬間の W·Y の符号（左 = +）と、通過の前後の W·X の符号（前 = +）。x_mic = sx·x_読み・y_mic = sy·y_読み。
     デモ 3 本は記録の三脚の向き（0 → sx = sy = +1・180 → −1・−1）で、音の向きとも合う（10/25 に確かめた）。
     測定 10 本は三脚の向きの記録が無いので、音の向きで決める（10/25: 10 本とも sx = +1・sy = −1 ＝ 前後はデモの向き 0 と同じで、自転車はマイクの右側）。
  3. 推論の置き場: <out>/clips/（測定 10 本＋デモ 3 本の写し）・<out>/height_table.csv（mic_z 1.5）・<out>/takes.csv（採点が使う本ごとの情報）。

使い方: python scripts/demo_bike_check_prepare.py [--out out/_demo_bike_check] [--convention v12_149]
  --convention v12_149（先行分 §3・2026-10-26）: 合成の約束（実効 149 dB）にそろえる＝測定 10 本のゲインとデモ 3 本の写しを −6 dB。既定 v9_143 は旧のまま。
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
from step19b_realsmoke_cut import cut_audio, plan_event_cut, DUR  # noqa: E402

MEAS_WAV = E2E / "raw_bicycle_measure" / "261007_007.WAV"
MEAS_READS = E2E / "raw_bicycle_measure" / "reads" / "reads_A.csv"
MEAS_TAKES = E2E / "raw_bicycle_measure" / "takes_bicycle_2026-10-07.csv"
DEMO_DIR = E2E / "out" / "field_demo"
GAIN_DB = -42.335          # out/field_demo/conv/calibration_261007_001.json（騒音計 42.6 dB・窓 0〜60 s）
# ↑ **v9_143（143 の約束）で記録した値**。--convention v12_149 のときだけ headroom（6 dB）を引く（先行分 §3）。
#   将来 json から読む形に変えるときは gain_from_calib_json を使う: json の convention が v12_149（もう −6 dB 済み）なら headroom を引かない（二重に引かない）


def gain_from_calib_json(path, convention: str | None) -> float:
    """較正の json（step19_realsmoke_convert の calibration_record）から、求める約束のゲインを返す。json の約束を見て二重に引かない（Fable 10/26 の守り）。"""
    import json as _json
    from step19_realsmoke_convert import convention_of
    rec = _json.loads(Path(path).read_text(encoding="utf-8"))
    have = convention_of(rec.get("convention") or "v9_143")      # 鍵が無い古い json は 143 の約束で書かれている
    want = convention_of(convention)
    assert have[0] in ("v9_143", "v12_149") and want[0] in ("v9_143", "v12_149"), (have, want)
    return float(rec["gain_db"]) + have[2] - want[2]              # json の約束を 143 に戻してから求める約束へ
MIC_Z = 1.5


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def side_check(path: Path, t_cpa: float, x_start_sign: int):
    """通過の瞬間の左右と、通過の前後の前後から、読みとマイクの座標の対応を決める。返り値 (sx, sy, Iy, Ix前, Ix後, 前後の一致)。
    t_cpa = 前輪が x = 0 を通った秒（その音声ファイルの秒）・x_start_sign = 読みの x で出発した側（前→後 = +1）。"""
    with sf.SoundFile(str(path)) as f:
        fs = f.samplerate
        a = int((t_cpa - 1.5) * fs)
        assert a >= 0, (path.name, t_cpa)
        f.seek(a)
        y = f.read(int(3.0 * fs), dtype="float64", always_2d=True)
    sos = butter(4, [2000, 8000], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, y, axis=0)
    w, yy, xx = y[:, 0], y[:, 1], y[:, 3]                  # AmbiX: W, Y, Z, X

    def seg(s0, s1):
        i0, i1 = int((s0 + 1.5) * fs), int((s1 + 1.5) * fs)
        e = np.mean(w[i0:i1] ** 2) + 1e-30
        return float(np.mean(w[i0:i1] * yy[i0:i1]) / e), float(np.mean(w[i0:i1] * xx[i0:i1]) / e)

    iy, _ = seg(-0.1, 0.5)          # 前輪が x = 0 を通ったあと（車体の真ん中が通るころ）
    _, ixb = seg(-1.2, -0.5)
    _, ixa = seg(0.8, 1.4)
    sy = 1 if iy > 0 else -1                       # 線（読みの y = +1.0）がマイクの左なら +1
    sx = int(np.sign(ixb)) * x_start_sign          # 出発した側（読みの x の符号）がマイクの前なら +1
    ok = int(np.sign(ixa)) == -sx * x_start_sign   # 通過のあとは反対側にいるはず
    return sx, sy, iy, ixb, ixa, bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(E2E / "out" / "_demo_bike_check"))
    ap.add_argument("--convention", default=None, help="v9_143（既定・旧）／v12_149（合成の約束・−6 dB）")
    a = ap.parse_args()
    from step19_realsmoke_convert import convention_of
    conv, _, hr = convention_of(a.convention)
    gain_db = GAIN_DB - hr                                   # 校正 261007_001 のゲインは 143 の約束（v9_143）で記録されている
    print(f"約束 {conv}: 測定のゲイン {gain_db:+.3f} dB・デモ 3 本の写しに {-hr:+.1f} dB")
    out = Path(a.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    reads = read_csv(MEAS_READS)
    takes = {r["take"]: r for r in read_csv(MEAS_TAKES)}
    length = sf.info(str(MEAS_WAV)).duration
    rows = []
    for i, tk in enumerate(sorted(takes), start=1):
        rr = [r for r in reads if r["take"] == tk]
        t0 = [float(r["t"]) for r in rr if abs(float(r["x"])) < 1e-9]
        assert len(t0) == 1, (tk, t0)
        t_x0 = t0[0]
        off = plan_event_cut(t_x0, length)
        cid = f"eval_field_clip{90000 + i}"
        cut_audio(MEAS_WAV, off, DUR, clips / f"{cid}.flac", gain_db=gain_db)
        xs = [float(r["x"]) for r in sorted(rr, key=lambda r: float(r["t"]))]
        x_start_sign = 1 if xs[0] > xs[-1] else -1          # 前→後 = x が減る
        sx, sy, iy, ixb, ixa, ok = side_check(MEAS_WAV, t_x0, x_start_sign)
        rows.append({"clip_id": cid, "source": "measure", "take": tk, "orig_file": MEAS_WAV.name, "cut_offset_s": f"{off:.3f}",
                     "direction": takes[tk]["direction"], "line_m": takes[tk]["line_m"], "sx": sx, "sy": sy, "sxsy_from": "音の向き",
                     "side_check": f"Iy={iy:+.3f} Ix前={ixb:+.3f} Ix後={ixa:+.3f} 通過のあとの一致={'○' if ok else '×'}"})
        print(f"{cid} {tk} off={off:.3f} sx={sx:+d} sy={sy:+d} {rows[-1]['side_check']}")
    # デモ 3 本（10/7 の切り出しと三脚の向きをそのまま使う・音の向きとの一致も見る）
    demo_reads = read_csv(DEMO_DIR / "reads" / "reads_A.csv")
    for r in read_csv(DEMO_DIR / "ann_all.csv"):
        cid = r["clip_id"]
        if hr == 0.0:
            shutil.copy2(DEMO_DIR / "clips" / f"{cid}.flac", clips / f"{cid}.flac")
        else:                                                # 10/7 の切り出しは 143 の約束 → 合成の約束に −hr dB
            x, fsr = sf.read(str(DEMO_DIR / "clips" / f"{cid}.flac"), dtype="float64", always_2d=True)
            sf.write(str(clips / f"{cid}.flac"), x * 10.0 ** (-hr / 20.0), fsr, subtype="PCM_24")
        tk = "20261007_C1_t" + cid[-2:]
        rr = sorted([x for x in demo_reads if x["take"] == tk], key=lambda x: float(x["t"]))
        assert rr, tk
        h = int(float(r["三脚の向き"]))
        s = 1 if h == 0 else -1
        t_x0 = [float(x["t"]) for x in rr if abs(float(x["x"])) < 1e-9][0] - float(r["cut_offset_s"])
        xs = [float(x["x"]) for x in rr]
        asx, asy, iy, ixb, ixa, ok = side_check(clips / f"{cid}.flac", t_x0, 1 if xs[0] > xs[-1] else -1)
        rows.append({"clip_id": cid, "source": "demo", "take": tk, "orig_file": r["orig_file"], "cut_offset_s": r["cut_offset_s"],
                     "direction": r["direction"], "line_m": r["line_m"], "sx": s, "sy": s, "sxsy_from": f"三脚の向き {h}",
                     "side_check": f"Iy={iy:+.3f} Ix前={ixb:+.3f} Ix後={ixa:+.3f} 音の向きとの一致={'○' if (asx, asy) == (s, s) and ok else '×'}"})
        print(f"{cid} {tk} 向き={h} sx=sy={s:+d} {rows[-1]['side_check']}")
    with open(out / "takes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "height_table.csv", "w", encoding="utf-8", newline="") as f:
        f.write("clip_id,mic_z\n")
        for r in rows:
            f.write(f"{r['clip_id']},{MIC_Z}\n")
    m = sorted({(r["sx"], r["sy"]) for r in rows if r["source"] == "measure"})
    print("測定の (sx, sy)（音で決めたもの）:", m, "・本数", len(rows), "→", out)


if __name__ == "__main__":
    main()
