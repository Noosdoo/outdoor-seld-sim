"""Step 6: シーン量産（train 30 / val 10）＋自動検品＋Colab用パッケージング。

各クリップは「サイレンが直線道路を通過する」シーンで、以下を乱数で振る
（シード固定＝再現可能。クリップごとの条件は work/<name>_scene.json と
inspect 時の scenes.csv に全記録）:
  - 通過する側: 左(y>0) / 右(y<0)
  - 横距離 |y|: 3〜15 m
  - 速度: 5〜15 m/s (18〜54 km/h)
  - 進行方向: +x / −x
  - 最接近時刻 t_cpa: 3〜7 s
  - 音源高さ z: 0.8〜1.3 m
  - サイレン型: peepo / wail（周波数±3〜5%・周期±10〜15%のジッタ）
  - ゲイン: −6〜0 dB

命名と分割（PSELDNets の rooms フィルタに整合）:
  train: fold1_room1_mix001..030  /  val: fold2_room1_mix001..010

使い方（dynamic-sound venv の python で）:
  python scripts/step6_batch_scenes.py plan          # 条件表の確認のみ（シミュなし）
  python scripts/step6_batch_scenes.py gen 0-2       # index範囲を生成（両端含む）
  python scripts/step6_batch_scenes.py inspect       # 全生成済みクリップを自動検品
  python scripts/step6_batch_scenes.py pack          # Colab用 zip を作成
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tqdm as _tqdm_mod  # noqa: E402

_tqdm_mod.tqdm = lambda iterable=None, **kw: iterable  # type: ignore

import soundfile as sf  # noqa: E402

from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.foa import encode_foa_timevarying, intensity_vector_doa  # noqa: E402
from outdoor_seld.geometry import (doa_unit_vectors, solve_emission_times,  # noqa: E402
                                   sound_speed)
from outdoor_seld.labels import (frame_label_rows, read_dcase_csv,  # noqa: E402
                                 write_dcase_csv)
from outdoor_seld.scene import (SceneConfig, decimate_to_out_rate,  # noqa: E402
                                run_mono_sim)
from outdoor_seld.siren import make_peepo_siren, make_siren  # noqa: E402

# 既定は高速レンダラ（fastsim: DynamicSoundと波形一致 rel_rms~3e-5 を検証済み、約330倍）
# `--dynamicsound` 指定で従来のDynamicSound直接実行に切替可能
USE_FAST = "--dynamicsound" not in sys.argv

GLOBAL_SEED = 20260711
N_TRAIN, N_VAL = 30, 10
DS_NAME = "outdoor_siren_v1"
DS = ROOT / "out" / f"dataset_{DS_NAME}"
WORK = DS / "work"
CLS_SRC = (ROOT.parent / "SELD-Data-Generator" / "database"
           / "seld_FSD50K_5_ov1_train" / "cls_indices.tsv")

# 検品閾値
INSPECT_AZ_MEDIAN_DEG = 2.0
INSPECT_EL_MEDIAN_DEG = 2.0
INSPECT_PEAK_MAX = 0.99
INSPECT_RMS_MIN = 1e-4


def clip_name(idx: int) -> str:
    if idx < N_TRAIN:
        return f"fold1_room1_mix{idx + 1:03d}"
    return f"fold2_room1_mix{idx - N_TRAIN + 1:03d}"


def sample_scene(idx: int) -> dict:
    """クリップ idx の条件を決定論的にサンプルする。"""
    rng = np.random.default_rng(GLOBAL_SEED * 1000 + idx)
    # 側とサイレン型は交互割当で train/val とも完全均衡にする（他は乱数）
    side = 1.0 if idx % 2 == 0 else -1.0
    siren_type = "peepo" if (idx // 2) % 2 == 0 else "wail"
    offset = float(rng.uniform(3.0, 15.0)) * side
    speed = float(rng.uniform(5.0, 15.0))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    t_cpa = float(rng.uniform(3.0, 7.0))
    z = float(rng.uniform(0.8, 1.3))
    x0 = -dirx * speed * t_cpa
    x1 = x0 + dirx * speed * 10.0
    if siren_type == "peepo":
        siren_params = {
            "f_hi": 960.0 * float(rng.uniform(0.97, 1.03)),
            "f_lo": 770.0 * float(rng.uniform(0.97, 1.03)),
            "tone_sec": 0.65 * float(rng.uniform(0.9, 1.1)),
        }
    else:
        siren_params = {
            "f_lo": 650.0 * float(rng.uniform(0.95, 1.05)),
            "f_hi": 1450.0 * float(rng.uniform(0.95, 1.05)),
            "sweep_period_sec": 4.8 * float(rng.uniform(0.85, 1.15)),
        }
    return {
        "idx": idx, "name": clip_name(idx),
        "split": "train" if idx < N_TRAIN else "val",
        "side": "L" if side > 0 else "R", "offset_m": offset,
        "speed_mps": speed, "dir_x": dirx, "t_cpa_s": t_cpa,
        "src_z_m": z, "x_start": x0, "x_end": x1,
        "siren_type": siren_type, "siren_params": siren_params,
        "gain_db": float(rng.uniform(-6.0, 0.0)),
    }


def build_scene_config(s: dict) -> SceneConfig:
    return SceneConfig(
        clip_name=s["name"],
        src_start=(s["x_start"], s["offset_m"], s["src_z_m"]),
        src_end=(s["x_end"], s["offset_m"], s["src_z_m"]),
        siren_type=s["siren_type"],
        source_gain_db=s["gain_db"],
    )


def generate_clip(idx: int) -> None:
    s = sample_scene(idx)
    scene = build_scene_config(s)
    name = s["name"]
    wdir = WORK / name
    wdir.mkdir(parents=True, exist_ok=True)
    (DS / "foa").mkdir(parents=True, exist_ok=True)
    (DS / "metadata").mkdir(parents=True, exist_ok=True)
    c = sound_speed(scene.temperature_c)

    # 1) ドライ音源
    gen = make_peepo_siren if s["siren_type"] == "peepo" else make_siren
    dry = gen(scene.clip_len_sec, scene.fs_sim, **s["siren_params"])
    sf.write(wdir / "dry_48k.wav", dry.astype(np.float32), scene.fs_sim)

    # 2) シミュレーション（直接・鏡像）
    t0 = time.perf_counter()
    monos = {}
    for tag, mirror in [("direct", False), ("mirror", True)]:
        out_wav = wdir / f"mono_{tag}_48k.wav"
        if USE_FAST:
            wp = scene.waypoints_mirror() if mirror else scene.waypoints_direct()
            mono48 = render_mono(dry, wp, scene.mic_pos, scene.fs_sim,
                                 scene.clip_len_sec,
                                 temperature_c=scene.temperature_c,
                                 pressure_atm=scene.pressure_atm,
                                 rel_humidity=scene.rel_humidity,
                                 gain_db=scene.source_gain_db)
            sf.write(out_wav, mono48.astype(np.float32), scene.fs_sim)
        else:
            run_mono_sim(scene, str(wdir / "dry_48k.wav"), str(out_wav),
                         mirror=mirror)
            mono48 = np.asarray(sf.read(out_wav)[0], np.float64)
        monos[tag] = decimate_to_out_rate(mono48, scene.fs_sim, scene.fs_out)
    sim_sec = time.perf_counter() - t0

    # 3) FOA（直接のみ＝データセット用、反射込み＝将来のablation用に保管）
    mic = np.array(scene.mic_pos)
    foas = {}
    for tag, wp in [("direct", scene.waypoints_direct()),
                    ("mirror", scene.waypoints_mirror())]:
        tr = np.arange(len(monos[tag])) / scene.fs_out
        te, ps_te = solve_emission_times(tr, wp, mic, c)
        u, _ = doa_unit_vectors(ps_te, mic)
        foas[tag] = encode_foa_timevarying(monos[tag], u)
    foa_direct = foas["direct"]
    foa_withrefl = foas["direct"] + foas["mirror"]
    assert np.max(np.abs(foa_withrefl)) < INSPECT_PEAK_MAX, "clipping risk"
    sf.write(DS / "foa" / f"{name}.flac", foa_direct.T.astype(np.float32),
             scene.fs_out, subtype="PCM_24")
    sf.write(wdir / "foa_withrefl_24k.flac", foa_withrefl.T.astype(np.float32),
             scene.fs_out, subtype="PCM_24")

    # 4) ラベル
    rows, _ = frame_label_rows(scene.waypoints_direct(), mic,
                               clip_len_sec=scene.clip_len_sec,
                               class_idx=scene.class_idx, track_idx=0,
                               source_active_until=scene.clip_len_sec, c=c)
    write_dcase_csv(DS / "metadata" / f"{name}.csv", rows)

    # 5) 条件と統計の記録
    s["stats"] = {
        "sim_seconds": round(sim_sec, 1),
        "foa_direct_peak": float(np.max(np.abs(foa_direct))),
        "foa_direct_rms_W": float(np.sqrt(np.mean(foa_direct[0] ** 2))),
        "n_label_frames": len(rows),
        "first_label_frame": rows[0][0], "last_label_frame": rows[-1][0],
    }
    s["scene_config"] = dataclasses.asdict(scene)
    (wdir / "scene.json").write_text(json.dumps(s, indent=2))
    print(f"[gen {idx:02d}] {name} {s['split']} {s['side']} "
          f"|y|={abs(s['offset_m']):.1f}m v={s['speed_mps']:.1f}m/s "
          f"{s['siren_type']} peak={s['stats']['foa_direct_peak']:.3f} "
          f"({sim_sec:.0f}s)")


def inspect_all() -> int:
    """全生成済みクリップの自動検品。scenes.csv と inspection.csv を出力。"""
    rows_out = []
    n_fail = 0
    clips = sorted((DS / "foa").glob("*.flac"))
    print(f"inspecting {len(clips)} clips ...")
    for flac in clips:
        name = flac.stem
        scene_json = WORK / name / "scene.json"
        s = json.loads(scene_json.read_text()) if scene_json.exists() else {}
        foa, fs = sf.read(flac)
        foa = np.asarray(foa, np.float64).T
        peak = float(np.max(np.abs(foa)))
        rms_w = float(np.sqrt(np.mean(foa[0] ** 2)))

        _, az_iv, el_iv, energy = intensity_vector_doa(
            foa, fs, frame_sec=0.1, fmin=200, fmax=4000)
        az_iv[energy < np.max(energy) * 1e-6] = np.nan
        labels = read_dcase_csv(DS / "metadata" / f"{name}.csv")
        az_err, el_err = [], []
        for k, evs in labels.items():
            if k < len(az_iv) and np.isfinite(az_iv[k]):
                da = (az_iv[k] - evs[0][1] + 180) % 360 - 180
                az_err.append(abs(da))
                el_err.append(abs(el_iv[k] - evs[0][2]))
        az_med = float(np.median(az_err))
        az_max = float(np.max(az_err))
        el_med = float(np.median(el_err))

        ok = (az_med < INSPECT_AZ_MEDIAN_DEG and el_med < INSPECT_EL_MEDIAN_DEG
              and peak < INSPECT_PEAK_MAX and rms_w > INSPECT_RMS_MIN)
        n_fail += 0 if ok else 1
        rows_out.append({
            "name": name, "split": s.get("split", "?"),
            "side": s.get("side", "?"),
            "offset_m": round(s.get("offset_m", float("nan")), 2),
            "speed_mps": round(s.get("speed_mps", float("nan")), 2),
            "t_cpa_s": round(s.get("t_cpa_s", float("nan")), 2),
            "siren_type": s.get("siren_type", "?"),
            "gain_db": round(s.get("gain_db", float("nan")), 2),
            "peak": round(peak, 4), "rms_W": round(rms_w, 5),
            "az_med_err": round(az_med, 3), "az_max_err": round(az_max, 2),
            "el_med_err": round(el_med, 3),
            "n_frames": len(labels), "result": "PASS" if ok else "FAIL",
        })
        print(f"  {name} {rows_out[-1]['result']} az_med={az_med:.2f} "
              f"az_max={az_max:.1f} el_med={el_med:.2f} peak={peak:.3f}")

    import csv
    keys = list(rows_out[0].keys())
    with open(DS / "inspection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n{len(rows_out)} clips, {n_fail} FAIL -> {DS/'inspection.csv'}")
    return n_fail


def pack() -> None:
    """Colab 持ち込み用 zip（PSELDNets 直下で unzip する構成）。"""
    import shutil
    shutil.copyfile(CLS_SRC, DS / "cls_indices_train.tsv")
    zip_path = ROOT / "out" / f"dataset_{DS_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        zf.write(DS / "cls_indices_train.tsv", "datasets/cls_indices_train.tsv")
        for sub in ["foa", "metadata"]:
            for p in sorted((DS / sub).glob("*")):
                zf.write(p, f"datasets/{DS_NAME}/{sub}/{p.name}")
    mb = zip_path.stat().st_size / 1e6
    print(f"wrote {zip_path} ({mb:.1f} MB)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "plan":
        for i in range(N_TRAIN + N_VAL):
            s = sample_scene(i)
            print(f"{i:02d} {s['name']} {s['split']:5s} {s['side']} "
                  f"|y|={abs(s['offset_m']):5.1f}m v={s['speed_mps']:4.1f}m/s "
                  f"dir={'+x' if s['dir_x']>0 else '-x'} tcpa={s['t_cpa_s']:.1f}s "
                  f"z={s['src_z_m']:.2f} {s['siren_type']:5s} "
                  f"gain={s['gain_db']:+.1f}dB")
    elif mode == "gen":
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        for i in range(int(a), int(b) + 1):
            generate_clip(i)
    elif mode == "inspect":
        sys.exit(1 if inspect_all() else 0)
    elif mode == "pack":
        pack()
    else:
        print("usage: plan | gen A-B | inspect | pack")
