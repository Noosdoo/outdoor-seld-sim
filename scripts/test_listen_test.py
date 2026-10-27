# -*- coding: utf-8 -*-
"""人と比べる測定の道具の試験（2026-10-27 Opus）。架空の音と架空の結果だけ（実録は読まない）。
  T1 両耳の音: 左 +90° から来る雑音 → 左が 3 dB 以上大きく、低い帯（100〜800 Hz）で右が 0.55〜0.9 ms 遅れる（球の頭の理論 0.66〜0.77 ms）
  T2 正面 0° → 左右の差 0.5 dB 以内・右 −90° → T1 の左右反対
  T3 準備: 架空のデモの置き場（2 本）→ 聞く wav 2・反応の速さ 6・volume・trials（8 行）・_key（最接近 = 8 − 切り詰め・切り詰め 0〜2 s）・一番大きい標本 ≤ −1 dBFS
  T4 守り: 名前が eval_field_ なら止まる・置き場の名前に _demo_field_unused が無ければ止まる
  T6 数をそろえる・T7 拍手が切り出しの中なら拍手の 0.5 s 後から聞かせる
  T5 採点: 反応の速さ 0.3 s・自転車は人が 0.9 s 早い（人の時刻の距離は「10 m より遠い」）・ベルの本はモデルの自転車と比べる（最接近は自転車の正解から 7.05 s）・
     参考のベルの鳴り始めから 人 +0.2 s・モデル +0.5 s・押しがベルの後・通った側は人もモデルも正解
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import listen_test_prepare as P  # noqa: E402
import listen_test_score as S  # noqa: E402


def plane(az_deg: float, n: int, rng) -> np.ndarray:
    s = rng.standard_normal(n)
    a = np.deg2rad(az_deg)
    return np.stack([s, s * np.sin(a), np.zeros(n), s * np.cos(a)], axis=1)     # W, Y, Z, X（仰角 0）


def lr(b: np.ndarray, sr: int) -> tuple:
    """広い帯の左右の大きさの差 [dB] と、低い帯（100〜800 Hz・人が時間差を使う帯）の右の遅れ [s]。"""
    from scipy.signal import butter, sosfiltfilt
    db = 10 * np.log10(np.mean(b[:, 0] ** 2) / np.mean(b[:, 1] ** 2))
    sos = butter(4, [100, 800], btype="band", fs=sr, output="sos")
    L, R = sosfiltfilt(sos, b[:, 0]), sosfiltfilt(sos, b[:, 1])
    x = np.correlate(R[4000:12000], L[4000:12000], mode="full")                # 右を左にずらしてそろう遅れ
    lag = (np.argmax(x) - (8000 - 1)) / sr
    return db, lag


def wcsv(p, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main() -> int:
    ok = []
    rng = np.random.default_rng(0)
    sr = 48000
    dl, gl = lr(P.foa_to_binaural(plane(90, sr, rng), sr), sr)
    ok.append((f"T1 左 +90°: 左が {dl:.1f} dB 大きい・低い帯の右の遅れ {gl * 1e3:.2f} ms", dl > 3 and 0.55e-3 <= gl <= 0.9e-3))
    d0, _ = lr(P.foa_to_binaural(plane(0, sr, rng), sr), sr)
    dr, gr = lr(P.foa_to_binaural(plane(-90, sr, rng), sr), sr)
    ok.append((f"T2 正面 {d0:.2f} dB・右 −90° は {dr:.1f} dB／{gr * 1e3:.2f} ms", abs(d0) < 0.5 and dr < -3 and -0.9e-3 <= gr <= -0.55e-3))
    tmp = Path(tempfile.mkdtemp(prefix="listen_"))
    try:
        inf = tmp / "_demo_field_unused_test" / "infer"
        (inf / "clips").mkdir(parents=True)
        for cid in ("tune_field_clip99001", "tune_field_clip99002"):
            sf.write(str(inf / "clips" / f"{cid}.flac"), plane(60, 240000, rng) * 0.05, 24000, subtype="PCM_24")
        cols = ["clip_id", "class", "t_start", "t_cpa", "cut_offset_s"]
        wcsv(inf / "ann_all.csv", cols, [{"clip_id": "tune_field_clip99001", "class": "bicycle", "t_start": "6.0", "t_cpa": "12.0", "cut_offset_s": "4.0"},
                                          {"clip_id": "tune_field_clip99002", "class": "bike_bell", "t_start": "9.5", "t_cpa": "12.0", "cut_offset_s": "4.0"}])
        out = tmp / "_demo_field_unused_test" / "listen"
        m = P.prepare(out, seed=1, infer=inf)
        key = list(csv.DictReader(open(out / "_key.csv", encoding="utf-8")))
        tr = list(csv.DictReader(open(out / "trials.csv", encoding="utf-8")))
        mk = [r for r in key if r["block"] == "main"]
        files_ok = all((out / r["file"]).is_file() for r in tr) and (out / "volume.wav").is_file()
        cpa_ok = all(abs(float(r["cpa_file_s"]) - (8.0 - float(r["trim_s"]))) < 1e-6 and 0 <= float(r["trim_s"]) <= 2.0 for r in mk)
        bell = [r for r in mk if r["class"] == "bike_bell"][0]
        bell_ok = abs(float(bell["bell_onset_file_s"]) - (5.5 - float(bell["trim_s"]))) < 1e-6
        peak = max(float(np.max(np.abs(sf.read(str(out / r["file"]))[0]))) for r in mk)
        ok.append(("T3 準備: wav・trials 8 行・最接近とベルの秒・−1 dBFS", files_ok and len(tr) == 8 and len(mk) == 2 and cpa_ok and bell_ok
                   and peak <= 10 ** (-1 / 20) + 1e-3 and m["n_rt"] == 6))
        bad = tmp / "_demo_field_unused_bad" / "infer"
        shutil.copytree(inf, bad)
        wcsv(bad / "ann_all.csv", cols, [{"clip_id": "eval_field_clip01001", "class": "bicycle", "t_start": "6", "t_cpa": "12", "cut_offset_s": "4"}])
        g1 = g2 = False
        try:
            P.load_targets(bad)
        except P.GuardError:
            g1 = True
        other = tmp / "elsewhere" / "infer"
        shutil.copytree(inf, other)
        try:
            P.load_targets(other)
        except P.GuardError:
            g2 = True
        ok.append(("T4 守り: eval_field_ の名前・デモでない置き場で止まる", g1 and g2))
        # T6 数をそろえる: 自転車 2・ベル 1 → 自転車のうち採点に使える正解が少ない方（除外区間で削られた方）を外す
        bal = tmp / "_demo_field_unused_bal" / "infer"
        shutil.copytree(inf, bal)
        sf.write(str(bal / "clips" / "tune_field_clip99003.flac"), plane(60, 240000, rng) * 0.05, 24000, subtype="PCM_24")
        wcsv(bal / "ann_all.csv", cols, [{"clip_id": "tune_field_clip99001", "class": "bicycle", "t_start": "6", "t_cpa": "12", "cut_offset_s": "4"},
                                          {"clip_id": "tune_field_clip99002", "class": "bike_bell", "t_start": "9.5", "t_cpa": "12", "cut_offset_s": "4"},
                                          {"clip_id": "tune_field_clip99003", "class": "bicycle", "t_start": "6", "t_cpa": "12", "cut_offset_s": "4"}])
        (bal / "gt").mkdir()
        for cid in ("tune_field_clip99001", "tune_field_clip99003"):
            (bal / "gt" / f"{cid}.csv").write_text("".join(f"{k},8,0,90,-10,3\n" for k in range(50, 91)), encoding="utf-8")
        wcsv(bal / "exclusions.csv", ["clip_id", "入り_クリップの秒", "出_クリップの秒"], [{"clip_id": "tune_field_clip99001", "入り_クリップの秒": "5.0", "出_クリップの秒": "8.0"}])
        keep, drop = P.balance(P.load_targets(bal), bal)
        # T7 拍手が切り出しの中（1.9 s）→ 切り詰めは拍手の 0.5 s 後（2.4 s・無作為の上限 2.0 s より後）・理由は「拍手のあと」
        cl = tmp / "_demo_field_unused_clap"
        shutil.copytree(inf, cl / "infer")
        wcsv(cl / "infer" / "ann_all.csv", cols + ["orig_file"], [
            {"clip_id": "tune_field_clip99001", "class": "bicycle", "t_start": "6", "t_cpa": "12", "cut_offset_s": "4", "orig_file": "raw/x1.WAV"},
            {"clip_id": "tune_field_clip99002", "class": "bike_bell", "t_start": "9.5", "t_cpa": "12", "cut_offset_s": "4", "orig_file": "raw/x2.WAV"}])
        (cl / "20261022" / "reads").mkdir(parents=True)
        (cl / "20261022" / "reads" / "audio_reads_A.json").write_text(json.dumps({"x1.WAV": {"clap_audio_s": 5.9}, "x2.WAV": {"clap_audio_s": 2.0}}), encoding="utf-8")
        P.prepare(cl / "listen", seed=3, infer=cl / "infer")
        kc7 = {r["clip_id"]: r for r in csv.DictReader(open(cl / "listen" / "_key.csv", encoding="utf-8")) if r["block"] == "main"}
        k1, k2 = kc7["tune_field_clip99001"], kc7["tune_field_clip99002"]
        ok.append((f"T7 拍手が中（1.9 s）→ 切り詰め {k1['trim_s']}（{k1['trim_why']}）・拍手が外の本は無作為（{k2['trim_why']}）",
                   abs(float(k1["trim_s"]) - 2.4) < 1e-9 and k1["trim_why"] == "拍手のあと" and abs(float(k1["clap_clip_s"]) - 1.9) < 1e-9
                   and k2["clap_clip_s"] == "" and k2["trim_why"] == "無作為"))
        ok.append(("T6 数をそろえる: 自転車 2・ベル 1 → 除外で削られた 99001 を外して 1 本ずつ",
                   sorted(t["clip_id"] for t in keep) == ["tune_field_clip99002", "tune_field_clip99003"] and [t["clip_id"] for t in drop] == ["tune_field_clip99001"]))
        # T5 採点（架空の key・結果・正解・予測）
        L = tmp / "_demo_field_unused_s" / "listen"; I = tmp / "_demo_field_unused_s" / "infer"
        kc = ["block", "order", "file", "clip_id", "class", "trim_s", "cpa_file_s", "bell_onset_file_s", "flac_sha256"]
        wcsv(L / "_key.csv", kc, [
            {"block": "rt", "order": 1, "file": "rt_01.wav", "class": "beep", "bell_onset_file_s": "2.0"},
            {"block": "rt", "order": 2, "file": "rt_02.wav", "class": "beep", "bell_onset_file_s": "3.0"},
            {"block": "rt", "order": 3, "file": "rt_03.wav", "class": "beep", "bell_onset_file_s": "2.5"},
            {"block": "main", "order": 1, "file": "listen_01.wav", "clip_id": "c1", "class": "bicycle", "trim_s": "1.0", "cpa_file_s": "7.0", "bell_onset_file_s": ""},
            {"block": "main", "order": 2, "file": "listen_02.wav", "clip_id": "c2", "class": "bike_bell", "trim_s": "0.5", "cpa_file_s": "7.5", "bell_onset_file_s": "4.5"}])
        rc = ["block", "order", "file", "presses_s", "first_press_s", "side", "played_at"]
        wcsv(L / "results_T.csv", rc, [
            {"block": "rt", "file": "rt_01.wav", "first_press_s": "2.3"}, {"block": "rt", "file": "rt_02.wav", "first_press_s": "3.3"},
            {"block": "rt", "file": "rt_03.wav", "first_press_s": "2.8"},
            {"block": "main", "file": "listen_01.wav", "first_press_s": "4.0", "side": "L"},
            {"block": "main", "file": "listen_02.wav", "first_press_s": "5.0", "side": "R"}])
        (I / "gt").mkdir(parents=True)
        (I / "gt" / "c1.csv").write_text("".join(f"{k},8,0,90.0,-10.0,{max(1.0, (80 - k) * 0.3):.2f}\n" for k in range(50, 91)), encoding="utf-8")
        # ベルの本: ベル（3）は 50〜90・走っている自転車（8）は 40〜79（距離は 70 以降 1.0 m で最小 → 最接近 7.05 s）
        (I / "gt" / "c2.csv").write_text("".join(f"{k},3,0,-90.0,-10.0,3.0\n" for k in range(50, 91))
                                         + "".join(f"{k},8,0,-90.0,-10.0,{max(1.0, (75 - k) * 0.2):.2f}\n" for k in range(40, 80)), encoding="utf-8")
        pred = tmp / "pred.csv"
        pred.write_text("\n".join([f"c1,{k},8,0,80.0,-10.0,2.0" for k in range(55, 85)] + [f"c2,{k},3,0,-85.0,-10.0,3.0" for k in range(54, 70)]
                                  + [f"c2,{k},8,0,-85.0,-10.0,2.0" for k in range(58, 71)]) + "\n", encoding="utf-8")
        r = S.score(L, "T", I, pred)
        t1, t2 = r["trials"]
        sb, sl = r["summary"]["bicycle"], r["summary"]["bike_bell"]
        ok.append((f"T5 採点: 反応 {r['rt_median_s']:.2f} s・自転車 人 − モデル {t1.get('human_minus_model_s'):+.2f} s・距離 {t1['human_dist_m']}・"
                   f"ベルの本の自転車 人 {t2['human_rel_s']:+.2f}／モデル {t2['model_rel_s']:+.2f}・参考 ベルの鳴り始めから 人 {t2['bell_human_rel_s']:+.2f}／モデル {t2['bell_model_rel_s']:+.2f}",
                   abs(r["rt_median_s"] - 0.3) < 1e-9 and abs(t1["human_rel_s"] - (4.7 - 7.75)) < 1e-9 and abs(t1["model_rel_s"] - (5.6 - 7.75)) < 1e-9   # 最接近 = 正解の距離が最小の最初の枠 77 → 7.75 s
                   and abs(t1["human_minus_model_s"] - (-0.9)) < 1e-9 and t1["human_dist_m"] == ">10" and abs(t1["model_dist_m"] - 7.5) < 1e-9
                   and abs(t2["ref_s"] - 7.05) < 1e-9 and abs(t2["human_rel_s"] - (5.2 - 7.05)) < 1e-9 and abs(t2["model_rel_s"] - (5.9 - 7.05)) < 1e-9
                   and abs(t2["bell_human_rel_s"] - 0.2) < 1e-9 and abs(t2["bell_model_rel_s"] - 0.5) < 1e-9 and t2["human_after_bell"]
                   and t1["gt_side"] == "L" and t1["model_side"] == "L" and t2["gt_side"] == "R" and t2["model_side"] == "R"
                   and sb["human_earlier"] == 1 and sb["human_side_correct"] == 1 and sl["model_side_correct"] == 1 and sl["human_after_bell"] == 1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
