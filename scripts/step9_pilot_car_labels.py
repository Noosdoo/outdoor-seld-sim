# -*- coding: utf-8 -*-
"""パイロット実験用: v8の音声はそのまま、車（走行音）を第5クラスとしてラベル追加。

目的（v9設計の先行検証、out/v9_design_2026-07-16.md）:
  Q1 車を音色で判別できるか（substitution≈0が5クラスでも保たれるか）
  Q2 警告音と車が重なった時間帯に両方を方向付きで拾えるか

ラベル規約（v9と同じ「聞こえる瞬間」ルール）:
  車のフレームを active とする条件 =
    (a) 車の直接音方向が定義できる（放射時刻が解ける）
    (b) そのフレームの車の実エネルギー >= 背景雑音の実エネルギー（フレームSNR>=0dB）
        車エネルギー: work/<clip>/foa_interf_24k.flac のW（混合に入っている形＝反射込み）
        雑音エネルギー: foa(混合) - foa_clean - foa_interf の残差W（構成上正確に雑音のみ）
  フリッカ防止: activeの穴が2フレーム(0.2s)以下なら埋める
  方向: 直接音の放射時刻DOA（警告音と同じ規約）

出力:
  out/dataset_outdoor_siren_v9pilot/metadata/*.csv （警告音+車のマージ済み）
  out/dataset_outdoor_siren_v9pilot/cls_indices_train.tsv （5クラス）
  out/pilot_v9_labels.zip （Colab持ち込み用: metadata+tsvのみ。音声はv8 zipを流用）
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.geometry import apparent_azel_deg, sound_speed  # noqa: E402

V8 = ROOT / "out" / "dataset_outdoor_siren_v8"
DST = ROOT / "out" / "dataset_outdoor_siren_v9pilot"
CLS_SRC = ROOT / "configs" / "cls_indices_v9pilot.tsv"
CAR_CLASS = 4
LABEL_RES = 0.1
GAP_FILL = 2       # activeの穴埋め（フレーム）
SNR_GATE_DB = 0.0  # 可聴判定: 車エネルギー >= 雑音エネルギー + この値


def frame_energies(x_w: np.ndarray, fs: int, n_frames: int) -> np.ndarray:
    spf = int(LABEL_RES * fs)
    return np.array([float(np.mean(x_w[k * spf:(k + 1) * spf] ** 2))
                     for k in range(n_frames)])


def main():
    (DST / "metadata").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(CLS_SRC, DST / "cls_indices_train.tsv")

    act_fracs = []
    poly2 = 0
    total_alert = 0
    n_clips = 0
    for meta in sorted((V8 / "metadata").glob("*.csv")):
        name = meta.stem
        sj = json.loads((V8 / "work" / name / "scene.json").read_text())
        itf = sj["interferer"]
        sc = sj["scene_config"]
        clip_len = sc["clip_len_sec"]
        n_frames = int(round(clip_len / LABEL_RES))
        c = sound_speed(sc["temperature_c"])
        mic = np.array(sc["mic_pos"])

        # --- 可聴性: 車の実エネルギー vs 雑音の実エネルギー（フレーム毎、W） ---
        foa_ds = np.asarray(sf.read(V8 / "foa" / f"{name}.flac")[0], np.float64).T
        fs = 24000
        foa_cl = np.asarray(
            sf.read(V8 / "work" / name / "foa_clean_24k.flac")[0], np.float64).T
        foa_it = np.asarray(
            sf.read(V8 / "work" / name / "foa_interf_24k.flac")[0], np.float64).T
        e_car = frame_energies(foa_it[0], fs, n_frames)
        e_noise = frame_energies((foa_ds - foa_cl - foa_it)[0], fs, n_frames)
        audible = 10.0 * np.log10(e_car / np.maximum(e_noise, 1e-30)) >= SNR_GATE_DB

        # 穴埋め（フリッカ防止）
        idx = np.where(audible)[0]
        if len(idx) > 1:
            for a, b in zip(idx[:-1], idx[1:]):
                if 1 < b - a <= GAP_FILL + 1:
                    audible[a:b] = True

        # --- 車の方向（直接音・放射時刻。警告音と同じ規約） ---
        wp_i = np.array(
            [[0.0, itf["x_start"], itf["offset_m"], itf["src_z_m"]],
             [clip_len, itf["x_end"], itf["offset_m"], itf["src_z_m"]]])
        t_frames = (np.arange(n_frames) + 0.5) * LABEL_RES
        az, el, te, _ = apparent_azel_deg(t_frames, wp_i, mic, c)
        ok_dir = np.isfinite(az)

        # --- マージ ---
        rows = [l.strip() for l in meta.read_text().strip().split("\n") if l.strip()]
        total_alert += len(rows)
        car_rows = []
        for k in range(n_frames):
            if audible[k] and ok_dir[k]:
                car_rows.append(
                    f"{k},{CAR_CLASS},0,{int(np.rint(az[k]))},{int(np.rint(el[k]))}")
        all_rows = rows + car_rows
        all_rows.sort(key=lambda r: (int(r.split(",")[0]), int(r.split(",")[1])))
        (DST / "metadata" / f"{name}.csv").write_text("\n".join(all_rows) + "\n")

        # 統計
        act_fracs.append(float(np.mean(audible)))
        alert_frames = {int(r.split(",")[0]) for r in rows}
        car_frames = {int(r.split(",")[0]) for r in car_rows}
        poly2 += len(alert_frames & car_frames)
        n_clips += 1

    af = np.array(act_fracs)
    print(f"clips: {n_clips}")
    print(f"car audible fraction per clip: median={np.median(af):.2f} "
          f"p10={np.percentile(af,10):.2f} p90={np.percentile(af,90):.2f} "
          f"min={af.min():.2f} max={af.max():.2f}")
    print(f"alert rows total: {total_alert}, polyphony-2 frames (alert+car): {poly2}")

    # --- Colab持ち込み用の小型zip（音声はv8のzipを流用するため含めない） ---
    zip_path = ROOT / "out" / "pilot_v9_labels.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DST / "cls_indices_train.tsv", "datasets/cls_indices_train.tsv")
        for p in sorted((DST / "metadata").glob("*.csv")):
            zf.write(p, f"datasets/outdoor_siren_v9pilot/metadata/{p.name}")
    print(f"wrote {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
