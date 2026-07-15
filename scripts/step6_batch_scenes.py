"""Step 6: シーン量産（train 30 / val 10）＋自動検品＋Colab用パッケージング。

各クリップは「サイレンが直線道路を通過する」シーンで、以下を乱数で振る
（シード固定＝再現可能。クリップごとの条件は work/<name>_scene.json と
inspect 時の scenes.csv に全記録）:
  - 通過する側: 左(y>0) / 右(y<0)
  - 横距離 |y|: 3〜15 m（v4は5〜20m。OFFSET_RANGE_M参照）
  - 速度: 5〜15 m/s (18〜54 km/h)（v4は15〜30m/s=54〜108km/h。SPEED_RANGE_MPS参照）
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
  （既定はv3。--v1/--v2/--v4/--v5/--v6/--v7 で切替。v4=幹線道路版、v3と同じ妨害音/雑音レンジのまま
   速度・距離レンジだけ変更。v5=マルチクラス拡張版、対象をSiren/Horn/BackupBeep/BikeBell
   の4クラスに増やし、妨害音（車）もクリーン合成に置き換え（実録音不使用）。
   v6=v5と同一構成でクラス毎データを4倍（train240/val80）にした増量版。
   v7=v6と同一シーンで地面反射をON（対象・妨害車とも）にしたフル物理版=ablation基準。
   いずれも新規フォルダに出力し、既存バージョンは一切上書きしない）
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

from outdoor_seld.alert_sounds import (make_backup_beep, make_bike_bell,  # noqa: E402
                                       make_horn)
from outdoor_seld.engine import make_car_driveby  # noqa: E402
from outdoor_seld.fastsim import render_mono  # noqa: E402
from outdoor_seld.foa import encode_foa_timevarying, intensity_vector_doa  # noqa: E402
from outdoor_seld.geometry import (doa_unit_vectors, solve_emission_times,  # noqa: E402
                                   sound_speed)
from outdoor_seld.labels import (frame_label_rows, read_dcase_csv,  # noqa: E402
                                 write_dcase_csv)
from outdoor_seld.noise import (diffuse_foa_noise, measure_snr_db,  # noqa: E402
                                mix_at_snr)
from outdoor_seld.scene import (SceneConfig, decimate_to_out_rate,  # noqa: E402
                                run_mono_sim)
from outdoor_seld.siren import make_peepo_siren, make_siren  # noqa: E402

# 既定は高速レンダラ（fastsim: DynamicSoundと波形一致 rel_rms~3e-5 を検証済み、約330倍）
# `--dynamicsound` 指定で従来のDynamicSound直接実行に切替可能
USE_FAST = "--dynamicsound" not in sys.argv

GLOBAL_SEED = 20260711
# v1 = クリーン（雑音なし、常時発音）
# v2 = v1と同一40シーン＋拡散性背景雑音（SNR 0-20dB）
# v3 = スパース発音（2-6秒だけ鳴る）＋車実録のラベルなし指向性妨害（SIR 0-15dB）
#      ＋背景雑音。train60/val20。速度5-15m/s・距離3-15m（住宅街想定）。
# v4 = v3と同一構成（妨害音/雑音/発音窓のレンジは完全に同じ）で、速度・距離レンジのみ
#      幹線道路想定に変更。「土俵（速度・距離レンジ）が変わるとablationの結論が変わるか」
#      を検証するための第二の土俵（2026-07-12 追加、out/ablation_plan_2026-07.md 4.5節）。
# v5 = v3と同一の土俵（速度・距離・雑音・妨害レンジ）だが、検出対象をSiren単独から
#      Siren/Horn/BackupBeep/BikeBellの4クラスに拡張（マルチクラスSELD化）。
#      妨害音（車）も実録音(suv_dirt_road.wav)からクリーン合成(engine.py)に置換
#      （対象と妨害音の音質を揃え、質感の違いがショートカット手がかりになるのを防ぐ。
#      2026-07-12 追加、本人指示）。
# v6 = v5と完全同一の構成・レンジで、クラス毎データだけ4倍（各クラスtrain60/val20、
#      計320本）。v5誤り解剖の発見（疎発音クラスのdir_err集中）が「クラス毎データ不足
#      （v5はv3の1/4）」で消えるか「本質的限界」かを切り分ける増量版（2026-07-13 追加）。
# v7 = v6と完全同一のシーン・件数で、**地面反射（two-ray/像音源、剛体地面R=+1）をON**
#      にしたフル物理版（対象音・妨害車の両方に適用）。ablationの基準土俵:
#      「ここから物理要素を1つずつ外す」一要因設計の参照条件（2026-07-14 追加）。
#      ラベルは従来どおり直接音の放射時刻DOA（=真の音源方向）。反射はIV仰角を
#      中央値1.4-3.3°押し下げるが方位は不変（v6クリップで実測）→検品の仰角閾値のみ緩和
# v8 = 敵対的レビュー（2026-07-14、out/adversarial_review_2026-07-14.md）の指摘を
#      全て織り込んだ仕切り直し版（v7と数値の直接比較はしない、新しい土俵）:
#      ①側×クラス独立化（side=(idx//4)%2, siren_type=(idx//8)%2。v5-v7の
#        「siren/backup_beep全数左」交絡の修正） ②test fold新設（train240/val80/
#        test80=idx320-399/fold3。最終報告はtestで1回だけ） ③SNR/SIR正規化基準を
#        直接音Wに固定（反射条件でも共通=ablationを真の1要素差に） ④検品に
#        実効SNR・帯域内SIR・全長SIRの記録列 ⑤全クラスに周波数±5%・時間構造±10%・
#        車回転±20%のジッタ（「クラス識別完全は同一波形の産物」批判への対処）
#      ⑥反射込みIV仰角バイアスのレンジゲート[-8,0]° ⑦基準物理=反射ON（理想剛面two-ray）
V1 = "--v1" in sys.argv
V2 = "--v2" in sys.argv
V4 = "--v4" in sys.argv
V5 = "--v5" in sys.argv
V6 = "--v6" in sys.argv
V7 = "--v7" in sys.argv
V8 = "--v8" in sys.argv
VARIANT = ("v1" if V1 else ("v2" if V2 else ("v4" if V4 else
           ("v5" if V5 else ("v6" if V6 else ("v7" if V7 else
           ("v8" if V8 else "v3")))))))
MULTICLASS = VARIANT in ("v5", "v6", "v7", "v8")  # 4クラス構成
WITH_REFL = VARIANT in ("v7", "v8")         # 地面反射をデータセット音声に含める
INDEP_SIDE = VARIANT == "v8"                # ①側×クラス独立化＋⑤全クラスジッタ
SNR_BASIS_DIRECT = VARIANT == "v8"          # ③正規化基準を直接音Wに固定
DS_NAME = f"outdoor_siren_{VARIANT}"
N_TRAIN, N_VAL, N_TEST = ((240, 80, 80) if VARIANT == "v8" else
                          (240, 80, 0) if VARIANT in ("v6", "v7") else
                          (60, 20, 0) if VARIANT in ("v3", "v4", "v5") else
                          (30, 10, 0))
WITH_NOISE = VARIANT != "v1"
SPARSE_INTERF = VARIANT in ("v3", "v4", "v5", "v6", "v7", "v8")
# 速度・距離レンジ: v4だけ幹線道路想定に変更。他のレンジ(SNR/SIR/発音区間)はv3と揃えたまま
# にして、「速度・距離だけを変数にする」比較にする（他の要因が同時に変わると土俵比較が汚れる）
SPEED_RANGE_MPS = (15.0, 30.0) if VARIANT == "v4" else (5.0, 15.0)   # 54-108 / 18-54 km/h
OFFSET_RANGE_M = (5.0, 20.0) if VARIANT == "v4" else (3.0, 15.0)
SNR_RANGE_DB = (0.0, 20.0)   # 背景雑音（W基準、v3ではサイレン発音区間基準）
NOISE_SLOPE = 1.0            # ピンク雑音（パワー 1/f）
SIR_RANGE_DB = (0.0, 15.0)   # サイレン対 妨害音（W基準、発音区間で定義）
EVENT_DUR_RANGE = (2.0, 6.0)
CAR_WAV = (ROOT.parent / "dynamic-sound" / "examples" / "resources"
           / "sounds" / "suv_dirt_road.wav")  # 60s/48kHz/mono 実録（v3/v4のみ使用）
DS = ROOT / "out" / f"dataset_{DS_NAME}"
WORK = DS / "work"
# v5/v6は独自の4クラス辞書（プロジェクト内、実録音源に依存しない）。他は既存の屋外10クラス辞書
CLS_SRC = ((ROOT / "configs" / "cls_indices_v5.tsv") if MULTICLASS else
           (ROOT.parent / "SELD-Data-Generator" / "database"
            / "seld_FSD50K_5_ov1_train" / "cls_indices.tsv"))
# マルチクラス設定（行順=class_idx、cls_indices_v5.tsvの並びと一致させること）
HAZARD_CLASSES = ["siren", "horn", "backup_beep", "bike_bell"]
HAZARD_CLASS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3}
# 帯域内SIR計測用のクラス主要帯域[Hz]（検品の記録列。レビューP4: 名目SIRは
# 広帯域W基準のため、低域寄りの合成車に対して帯域内の実効SIRを11-25dB過小表示する）
BAND_BY_CLASS = {"siren": (500.0, 1700.0), "horn": (300.0, 2600.0),
                 "backup_beep": (850.0, 1200.0), "bike_bell": (2400.0, 11500.0)}

# 検品閾値。v7（反射ON）では反射がIV仰角を物理的に押し下げる（中央値1.4-3.3°、
# 近距離・低音源高の裾では約8°に達する=320本検品で実測）ため、ラベル整合ゲートは
# 「直接音成分のみのFOA（foa_direct_24k.flac）」に対して行い、閾値は全版共通のまま
# にする（緩和でゲートを骨抜きにしない）。反射込み音声のIV仰角バイアスは既知物理
INSPECT_AZ_MEDIAN_DEG = 2.0
INSPECT_EL_MEDIAN_DEG = 2.0
INSPECT_PEAK_MAX = 0.99
INSPECT_RMS_MIN = 1e-4


def clip_name(idx: int) -> str:
    if idx < N_TRAIN:
        return f"fold1_room1_mix{idx + 1:03d}"
    if idx < N_TRAIN + N_VAL:
        return f"fold2_room1_mix{idx - N_TRAIN + 1:03d}"
    # v8以降: 最終報告用のheld-out test（学習にもckpt選択にも使わない）
    return f"fold3_room1_mix{idx - N_TRAIN - N_VAL + 1:03d}"


def split_name(idx: int) -> str:
    if idx < N_TRAIN:
        return "train"
    return "val" if idx < N_TRAIN + N_VAL else "test"


def sample_event(idx: int) -> dict:
    """サイレンの発音区間（v3）。2〜6秒、開始位置もランダム。"""
    rng = np.random.default_rng(GLOBAL_SEED * 31337 + idx)
    dur = float(rng.uniform(*EVENT_DUR_RANGE))
    t_on = float(rng.uniform(0.3, 10.0 - dur - 0.3))
    return {"t_on": round(t_on, 3), "t_off": round(t_on + dur, 3),
            "dur_s": round(dur, 3)}


def sample_interferer(idx: int) -> dict:
    """妨害音（車）の軌道・SIR・切り出し位置（v3/v4/v5）。"""
    rng = np.random.default_rng(GLOBAL_SEED * 104729 + idx)
    side = 1.0 if rng.random() < 0.5 else -1.0
    offset = float(rng.uniform(*OFFSET_RANGE_M)) * side
    speed = float(rng.uniform(*SPEED_RANGE_MPS))
    dirx = 1.0 if rng.random() < 0.5 else -1.0
    t_cpa = float(rng.uniform(2.0, 8.0))
    z = float(rng.uniform(0.8, 1.3))
    x0 = -dirx * speed * t_cpa
    # v5/v6は妨害音もクリーン合成(engine.make_car_driveby)に置換、実録音は使わない
    source = "synth_car_driveby" if MULTICLASS else "suv_dirt_road.wav"
    out = {"offset_m": offset, "speed_mps": speed, "t_cpa_s": t_cpa,
           "src_z_m": z, "x_start": x0, "x_end": x0 + dirx * speed * 10.0,
           "sir_db": float(rng.uniform(*SIR_RANGE_DB)),
           "excerpt_start_s": float(rng.uniform(0.0, 49.0)),
           "source": source}
    if INDEP_SIDE:
        # v8: 車のエンジン回転(f0)も±20%ジッタ（乱数は既存draw列の後に追加）
        out["f0_engine"] = 42.0 * float(rng.uniform(0.8, 1.2))
    return out


def sample_noise(idx: int) -> dict:
    """雑音条件のサンプル。ジオメトリ用の乱数系列と独立（v1と同一シーンを保証）。"""
    rng = np.random.default_rng(GLOBAL_SEED * 7919 + idx)
    return {"snr_db": float(rng.uniform(*SNR_RANGE_DB)),
            "slope": NOISE_SLOPE,
            "noise_seed": GLOBAL_SEED * 7919 + idx}


def sample_scene(idx: int) -> dict:
    """クリップ idx の条件を決定論的にサンプルする。"""
    rng = np.random.default_rng(GLOBAL_SEED * 1000 + idx)
    # v5/v6/v7のみ4クラス均等割当（idx%4）。他は常にsiren（既存v1-v4の割当式は変更しない＝
    # 既に検証済みのv3/v4のシーンをbit単位で再現できるようにするため）
    if MULTICLASS:
        hazard_class = HAZARD_CLASSES[idx % 4]
        if INDEP_SIDE:
            # v8: 側とサイレン型をクラス(idx%4)と互いに独立化（レビュー#1対応。
            # 全組合せの均衡はドライラン検算済み: クラス×側30/30・10/10・10/10、
            # サイレン型×側15/15・5/5・5/5）
            side = 1.0 if (idx // 4) % 2 == 0 else -1.0
            siren_type = (("peepo" if (idx // 8) % 2 == 0 else "wail")
                          if hazard_class == "siren" else "-")
        else:
            # v5-v7の式（凍結。side=idx%2との組でクラス×側が100%交絡していた欠陥あり、
            # 再現性のためコードは残す。詳細=out/adversarial_review_2026-07-14.md #1）
            side = 1.0 if idx % 2 == 0 else -1.0
            siren_type = (("peepo" if (idx // 4) % 2 == 0 else "wail")
                          if hazard_class == "siren" else "-")
    else:
        side = 1.0 if idx % 2 == 0 else -1.0
        hazard_class = "siren"
        siren_type = "peepo" if (idx // 2) % 2 == 0 else "wail"
    offset = float(rng.uniform(*OFFSET_RANGE_M)) * side
    speed = float(rng.uniform(*SPEED_RANGE_MPS))
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
    elif siren_type == "wail":
        siren_params = {
            "f_lo": 650.0 * float(rng.uniform(0.95, 1.05)),
            "f_hi": 1450.0 * float(rng.uniform(0.95, 1.05)),
            "sweep_period_sec": 4.8 * float(rng.uniform(0.85, 1.15)),
        }
    else:
        siren_params = {}
    gain_db = float(rng.uniform(-6.0, 0.0))
    # v8のみ: siren以外の3クラスにも周波数±5%・時間構造±10%のジッタ（レビューP5対応。
    # v5-v7ではbeep/bellが全クリップ同一波形＝周波数照合のショートカットを許していた）。
    # 乱数はgain_dbの後に引く＝既存版の乱数系列に影響しない
    dry_params = {}
    if INDEP_SIDE:
        if hazard_class == "horn":
            dry_params = {"f_lo": 410.0 * float(rng.uniform(0.95, 1.05)),
                          "f_hi": 500.0 * float(rng.uniform(0.95, 1.05)),
                          "honk_sec": 0.35 * float(rng.uniform(0.9, 1.1)),
                          "gap_sec": 0.15 * float(rng.uniform(0.9, 1.1))}
        elif hazard_class == "backup_beep":
            dry_params = {"freq": 1000.0 * float(rng.uniform(0.95, 1.05)),
                          "on_sec": 0.5 * float(rng.uniform(0.9, 1.1)),
                          "off_sec": 0.5 * float(rng.uniform(0.9, 1.1))}
        elif hazard_class == "bike_bell":
            dry_params = {"f0": 3000.0 * float(rng.uniform(0.95, 1.05)),
                          "ring_gap_sec": 0.45 * float(rng.uniform(0.9, 1.1)),
                          "repeat_period_sec": 2.0 * float(rng.uniform(0.9, 1.1))}
    out = {
        "idx": idx, "name": clip_name(idx),
        "split": split_name(idx),
        "side": "L" if side > 0 else "R", "offset_m": offset,
        "speed_mps": speed, "dir_x": dirx, "t_cpa_s": t_cpa,
        "src_z_m": z, "x_start": x0, "x_end": x1,
        "hazard_class": hazard_class,
        "class_idx": HAZARD_CLASS_IDX[hazard_class] if MULTICLASS else 4,
        "class_name": hazard_class if MULTICLASS else "Siren",
        "siren_type": siren_type, "siren_params": siren_params,
        "gain_db": gain_db,
    }
    if INDEP_SIDE:
        # キー自体をv8限定にする（v5-v7のscene.jsonのバイト再現性を保つため）
        out["dry_params"] = dry_params
    return out


def build_scene_config(s: dict) -> SceneConfig:
    return SceneConfig(
        clip_name=s["name"],
        src_start=(s["x_start"], s["offset_m"], s["src_z_m"]),
        src_end=(s["x_end"], s["offset_m"], s["src_z_m"]),
        siren_type=s["siren_type"],
        source_gain_db=s["gain_db"],
        class_idx=s["class_idx"],
        class_name=s["class_name"],
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

    # 1) ドライ音源（v3以降: 発音区間 2-6 秒に窓掛け、20ms のフェード付き）
    hazard_class = s.get("hazard_class", "siren")
    if hazard_class == "siren":
        gen = make_peepo_siren if s["siren_type"] == "peepo" else make_siren
        dry = gen(scene.clip_len_sec, scene.fs_sim, **s["siren_params"])
    elif hazard_class == "horn":
        rng_dry = np.random.default_rng(GLOBAL_SEED * 610271 + idx)
        dry = make_horn(scene.clip_len_sec, scene.fs_sim, rng_dry,
                        **s.get("dry_params", {}))
    elif hazard_class == "backup_beep":
        dry = make_backup_beep(scene.clip_len_sec, scene.fs_sim,
                               **s.get("dry_params", {}))
    elif hazard_class == "bike_bell":
        dry = make_bike_bell(scene.clip_len_sec, scene.fs_sim,
                             **s.get("dry_params", {}))
    else:
        raise ValueError(f"unknown hazard_class: {hazard_class}")
    if SPARSE_INTERF:
        ev = sample_event(idx)
        s["event"] = ev
        fs_sim = scene.fs_sim
        env = np.zeros(len(dry))
        i0, i1 = int(ev["t_on"] * fs_sim), int(ev["t_off"] * fs_sim)
        env[i0:i1] = 1.0
        fade = int(0.02 * fs_sim)
        env[i0:i0 + fade] = np.linspace(0.0, 1.0, fade)
        env[i1 - fade:i1] = np.linspace(1.0, 0.0, fade)
        dry = dry * env
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
    sf.write(wdir / "foa_withrefl_24k.flac", foa_withrefl.T.astype(np.float32),
             scene.fs_out, subtype="PCM_24")
    # v7: データセット音声の対象音は反射込み。ラベルは直接音DOAのまま
    # （反射を「見抜いて」真の方向を答えるタスクにする）
    foa_target = foa_withrefl if WITH_REFL else foa_direct

    foa_interf = None
    if WITH_NOISE:
        # SNR/SIR の基準パワー: v3 はサイレン発音区間の W、v2 は全長の W
        if SPARSE_INTERF:
            a0 = int(s["event"]["t_on"] * scene.fs_out)
            a1 = int(s["event"]["t_off"] * scene.fs_out)
        else:
            a0, a1 = 0, foa_target.shape[1]
        # SNR/SIRの正規化基準: v8以降は全条件共通で「直接音W」に固定
        # （反射や物理スイッチの状態に依らず同じ基準＝ablationを真の1要素差にする。
        # レビュー#5/P3対応。v7以前は foa_target 基準のまま凍結）
        foa_basis = foa_direct if SNR_BASIS_DIRECT else foa_target
        p_sig = float(np.mean(foa_basis[0, a0:a1] ** 2))

        mix = foa_target.copy()
        if SPARSE_INTERF:
            # 妨害音: 車を別軌道で通過させ、独自の時変DOAでFOA化
            itf = sample_interferer(idx)
            if MULTICLASS:
                # クリーン合成の走行音（実録音は使わない、engine.py参照）
                rng_itf_audio = np.random.default_rng(GLOBAL_SEED * 727999 + idx)
                car10 = make_car_driveby(scene.clip_len_sec, scene.fs_sim,
                                         rng_itf_audio,
                                         f0=itf.get("f0_engine", 42.0))
            else:
                car, car_sr = sf.read(CAR_WAV)
                assert car_sr == scene.fs_sim, "car source must be 48 kHz"
                e0 = int(itf["excerpt_start_s"] * car_sr)
                car10 = np.asarray(
                    car[e0:e0 + int(scene.clip_len_sec * car_sr)], np.float64)
            wp_i = np.array(
                [[0.0, itf["x_start"], itf["offset_m"], itf["src_z_m"]],
                 [scene.clip_len_sec, itf["x_end"], itf["offset_m"],
                  itf["src_z_m"]]])
            mono_i = decimate_to_out_rate(
                render_mono(car10, wp_i, scene.mic_pos, scene.fs_sim,
                            scene.clip_len_sec,
                            temperature_c=scene.temperature_c,
                            pressure_atm=scene.pressure_atm,
                            rel_humidity=scene.rel_humidity),
                scene.fs_sim, scene.fs_out)
            tr_i = np.arange(len(mono_i)) / scene.fs_out
            te_i, ps_i = solve_emission_times(tr_i, wp_i, mic, c)
            u_i, _ = doa_unit_vectors(ps_i, mic)
            foa_interf = encode_foa_timevarying(mono_i, u_i)
            if WITH_REFL:
                # v7: 妨害車にも地面反射を適用（対象音と同じ像音源モデル。
                # 世界の物理は音源の種別によらず一貫させる）
                wp_im = wp_i.copy()
                wp_im[:, 3] = 2 * scene.ground_z - wp_im[:, 3]
                mono_im = decimate_to_out_rate(
                    render_mono(car10, wp_im, scene.mic_pos, scene.fs_sim,
                                scene.clip_len_sec,
                                temperature_c=scene.temperature_c,
                                pressure_atm=scene.pressure_atm,
                                rel_humidity=scene.rel_humidity),
                    scene.fs_sim, scene.fs_out)
                te_im, ps_im = solve_emission_times(tr_i, wp_im, mic, c)
                u_im, _ = doa_unit_vectors(ps_im, mic)
                foa_interf = foa_interf + encode_foa_timevarying(mono_im, u_im)
            p_int = float(np.mean(foa_interf[0, a0:a1] ** 2))
            g_i = float(np.sqrt(p_sig / (p_int * 10.0 ** (itf["sir_db"] / 10.0))))
            foa_interf = foa_interf * g_i
            itf["gain"] = g_i
            s["interferer"] = itf
            mix = mix + foa_interf

        noise_cond = sample_noise(idx)
        rng_n = np.random.default_rng(noise_cond["noise_seed"])
        noise = diffuse_foa_noise(mix.shape[1], scene.fs_out, rng_n,
                                  slope=noise_cond["slope"])
        g_n = float(np.sqrt(p_sig / (float(np.mean(noise[0] ** 2))
                                     * 10.0 ** (noise_cond["snr_db"] / 10.0))))
        noise_cond["noise_gain"] = g_n
        noise_cond["snr_ref"] = ("siren active window (W)" if SPARSE_INTERF
                                 else "full clip (W)")
        foa_out = mix + g_n * noise
        s["noise"] = noise_cond
    else:
        foa_out = foa_target

    # ヘッドルーム保護: 超えそうなら全成分を等倍で縮小（SNR/SIR比は不変）
    peak_out = float(np.max(np.abs(foa_out)))
    post_scale = 1.0
    if peak_out > 0.95:
        post_scale = 0.95 / peak_out
        foa_out = foa_out * post_scale
        s["post_scale"] = post_scale
    if WITH_NOISE:
        # 「混合に入っている形の」クリーン対象音（v7は反射込み）。検品のSNR/SIR実測と
        # step8のフレームエネルギー分析がこれを使う
        sf.write(wdir / "foa_clean_24k.flac",
                 (foa_target * post_scale).T.astype(np.float32),
                 scene.fs_out, subtype="PCM_24")
    if WITH_REFL:
        # 反射ON版では、ラベル整合ゲート（音とラベルのDOA照合）用に直接音のみの
        # FOAも保存する（反射込みではIV仰角が物理的にラベルからズレるため）
        sf.write(wdir / "foa_direct_24k.flac",
                 (foa_direct * post_scale).T.astype(np.float32),
                 scene.fs_out, subtype="PCM_24")
    if foa_interf is not None:
        sf.write(wdir / "foa_interf_24k.flac",
                 (foa_interf * post_scale).T.astype(np.float32),
                 scene.fs_out, subtype="PCM_24")
    sf.write(DS / "foa" / f"{name}.flac", foa_out.T.astype(np.float32),
             scene.fs_out, subtype="PCM_24")

    # 4) ラベル（v3: 発音区間のみ。音側と同じ放射時刻補正）
    rows, _ = frame_label_rows(
        scene.waypoints_direct(), mic,
        clip_len_sec=scene.clip_len_sec,
        class_idx=scene.class_idx, track_idx=0,
        source_active_from=(s["event"]["t_on"] if SPARSE_INTERF else 0.0),
        source_active_until=(s["event"]["t_off"] if SPARSE_INTERF
                             else scene.clip_len_sec), c=c)
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
    snr_note = f" snr={s['noise']['snr_db']:.1f}dB" if WITH_NOISE else ""
    label = s["siren_type"] if hazard_class == "siren" else hazard_class
    print(f"[gen {idx:02d}] {name} {s['split']} {s['side']} "
          f"|y|={abs(s['offset_m']):.1f}m v={s['speed_mps']:.1f}m/s "
          f"{label} peak={s['stats']['foa_direct_peak']:.3f}"
          f"{snr_note} ({sim_sec:.0f}s)")


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
        foa_ds, fs = sf.read(flac)
        foa_ds = np.asarray(foa_ds, np.float64).T  # データセット版（v2では雑音入り）
        labels = read_dcase_csv(DS / "metadata" / f"{name}.csv")

        def iv_errors(foa_x):
            _, az_iv, el_iv, energy = intensity_vector_doa(
                foa_x, fs, frame_sec=0.1, fmin=200, fmax=4000)
            az_iv[energy < np.max(energy) * 1e-6] = np.nan
            az_err, el_err = [], []
            for k, evs in labels.items():
                if k < len(az_iv) and np.isfinite(az_iv[k]):
                    da = (az_iv[k] - evs[0][1] + 180) % 360 - 180
                    az_err.append(abs(da))
                    el_err.append(abs(el_iv[k] - evs[0][2]))
            return (float(np.median(az_err)), float(np.max(az_err)),
                    float(np.median(el_err)))

        # 検品ゲートはクリーン版（音とラベルの整合は物理の性質。雑音は既知の付加）。
        # 反射ON版では foa_direct_24k（直接音のみ）でラベル整合を照合する
        # —— 反射込みだとIV仰角が物理的にラベル（=直接音方向）からズレるため
        clean_path = WORK / name / "foa_clean_24k.flac"
        foa_clean = (np.asarray(sf.read(clean_path)[0], np.float64).T
                     if clean_path.exists() else None)
        direct_path = WORK / name / "foa_direct_24k.flac"
        if direct_path.exists():
            foa_gate = np.asarray(sf.read(direct_path)[0], np.float64).T
        elif foa_clean is not None:
            foa_gate = foa_clean
        else:
            foa_gate = foa_ds
        peak = float(np.max(np.abs(foa_ds)))
        rms_w = float(np.sqrt(np.mean(foa_gate[0] ** 2)))
        az_med, az_max, el_med = iv_errors(foa_gate)

        # 実SNR/SIRの独立測定（目標±0.5dB）と、混合後のIV劣化の記録
        snr_target, snr_meas, az_med_noisy = "", "", ""
        sir_target, sir_meas = "", ""
        snr_active, sir_inband, sir_fullclip = "", "", ""
        snr_ok = True
        if clean_path.exists():
            snr_target = round(s.get("noise", {}).get("snr_db", float("nan")), 2)
            az_med_noisy = round(iv_errors(foa_ds)[0], 2)
            itf_path = WORK / name / "foa_interf_24k.flac"
            if itf_path.exists():
                foa_i = np.asarray(sf.read(itf_path)[0], np.float64).T
                residual = foa_ds - foa_clean - foa_i   # ≒ 背景雑音成分のみ
                ev = s.get("event", {})
                a0 = int(ev.get("t_on", 0.0) * fs)
                a1 = int(ev.get("t_off", 10.0) * fs)
                # SNR/SIRの照合基準は生成時と同じ（v8=直接音W、v7以前=混合対象音W）
                foa_basis = (foa_gate if (SNR_BASIS_DIRECT
                                          and direct_path.exists())
                             else foa_clean)
                p_sig = float(np.mean(foa_basis[0, a0:a1] ** 2))
                p_int = float(np.mean(foa_i[0, a0:a1] ** 2))
                sir_target = round(
                    s.get("interferer", {}).get("sir_db", float("nan")), 2)
                sir_meas = round(10.0 * np.log10(p_sig / p_int), 2)
                p_noise = float(np.mean(residual[0] ** 2))
                snr_meas = round(10.0 * np.log10(p_sig / p_noise), 2)
                snr_ok = (abs(snr_meas - snr_target) < 0.5
                          and abs(sir_meas - sir_target) < 0.5)
                # --- 記録列（レビュー#10/P4/P11対応、ゲートには使わない） ---
                # 発音瞬間基準の実効SNR（窓内0.1sフレームのうちエネルギー中央値以上）
                spf = int(0.1 * fs)
                fr = [float(np.mean(foa_clean[0, k:k + spf] ** 2))
                      for k in range(a0, max(a0 + spf, a1 - spf + 1), spf)]
                fr = np.array(fr)
                hi_frames = fr[fr >= np.median(fr)]
                snr_active = round(10.0 * np.log10(
                    float(np.mean(hi_frames)) / p_noise), 2)
                # ターゲット帯域内SIR（クラス帯域のFFTパワー比）
                blo, bhi = BAND_BY_CLASS.get(s.get("hazard_class", "siren"),
                                             (200.0, 4000.0))
                def band_p(x):
                    X = np.abs(np.fft.rfft(x[a0:a1])) ** 2
                    f = np.fft.rfftfreq(a1 - a0, 1.0 / fs)
                    m = (f >= blo) & (f < bhi)
                    return float(np.mean(X[m]) + 1e-30)
                sir_inband = round(10.0 * np.log10(
                    band_p(foa_basis[0]) / band_p(foa_i[0])), 2)
                # クリップ全長の妨害レベル基準SIR（窓外の車支配の監視用）
                sir_fullclip = round(10.0 * np.log10(
                    p_sig / float(np.mean(foa_i[0] ** 2))), 2)
            else:
                snr_meas = round(measure_snr_db(foa_clean, foa_ds), 2)
                snr_ok = abs(snr_meas - snr_target) < 0.5

        # 反射込み音声のIV仰角バイアスのレンジゲート（レビューP8対応）:
        # 物理予測は「全数負・−8〜0°」。範囲外なら鏡像エンコードのバグを疑う
        el_bias_med = ""
        refl_ok = True
        if WITH_REFL and foa_clean is not None and direct_path.exists():
            _, _, el_iv_c, en_c = intensity_vector_doa(
                foa_clean, fs, frame_sec=0.1, fmin=200, fmax=4000)
            el_iv_c[en_c < np.max(en_c) * 1e-6] = np.nan
            diffs = [el_iv_c[k] - evs[0][2] for k, evs in labels.items()
                     if k < len(el_iv_c) and np.isfinite(el_iv_c[k])]
            if diffs:
                el_bias_med = round(float(np.median(diffs)), 2)
                # 下限-15°: v7実測(-7.8〜-0.6)＋v8新規シーンの裾(-9.7)に物理的余裕を
                # 持たせた値。バグ（鏡映のz符号ミス等）なら正側や桁違いになるので
                # 検知力は保たれる（当初の-8はv7の経験レンジそのままで狭すぎた）
                refl_ok = -15.0 <= el_bias_med <= 0.0

        ok = (az_med < INSPECT_AZ_MEDIAN_DEG and el_med < INSPECT_EL_MEDIAN_DEG
              and peak < INSPECT_PEAK_MAX and rms_w > INSPECT_RMS_MIN
              and snr_ok and refl_ok)
        n_fail += 0 if ok else 1
        rows_out.append({
            "name": name, "split": s.get("split", "?"),
            "side": s.get("side", "?"),
            "offset_m": round(s.get("offset_m", float("nan")), 2),
            "speed_mps": round(s.get("speed_mps", float("nan")), 2),
            "t_cpa_s": round(s.get("t_cpa_s", float("nan")), 2),
            "hazard_class": s.get("hazard_class", "siren"),
            "siren_type": s.get("siren_type", "?"),
            "gain_db": round(s.get("gain_db", float("nan")), 2),
            "peak": round(peak, 4), "rms_W": round(rms_w, 5),
            "az_med_err": round(az_med, 3), "az_max_err": round(az_max, 2),
            "el_med_err": round(el_med, 3),
            "snr_db_target": snr_target, "snr_db_measured": snr_meas,
            "sir_db_target": sir_target, "sir_db_measured": sir_meas,
            "t_on": s.get("event", {}).get("t_on", ""),
            "t_off": s.get("event", {}).get("t_off", ""),
            "az_med_err_noisy": az_med_noisy,
            "snr_active_med_db": snr_active, "sir_inband_db": sir_inband,
            "sir_fullclip_db": sir_fullclip, "el_bias_med": el_bias_med,
            "n_frames": len(labels), "result": "PASS" if ok else "FAIL",
        })
        snr_note = f" snr={snr_meas}dB az_noisy={az_med_noisy}" if snr_meas != "" else ""
        print(f"  {name} {rows_out[-1]['result']} az_med={az_med:.2f} "
              f"az_max={az_max:.1f} el_med={el_med:.2f} peak={peak:.3f}{snr_note}")

    import csv
    keys = list(rows_out[0].keys())
    with open(DS / "inspection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n{len(rows_out)} clips, {n_fail} FAIL -> {DS/'inspection.csv'}")
    return n_fail


def preview(indices) -> None:
    """試作クリップの確認素材: スペクトログラム＋ラベル/妨害方位の図、試聴用WAV。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    from outdoor_seld.geometry import apparent_azel_deg

    pv = DS / "preview"
    pv.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        name = clip_name(idx)
        sj = json.loads((WORK / name / "scene.json").read_text())
        foa, fs = sf.read(DS / "foa" / f"{name}.flac")
        foa = np.asarray(foa, np.float64)
        sf.write(pv / f"{name}_listen.wav", foa[:, 0].astype(np.float32), fs)

        labels = read_dcase_csv(DS / "metadata" / f"{name}.csv")
        frames = sorted(labels)
        t_lab = [(k + 0.5) * 0.1 for k in frames]
        az_lab = [labels[k][0][1] for k in frames]

        c = sound_speed(sj["scene_config"]["temperature_c"])
        mic = np.array(sj["scene_config"]["mic_pos"])
        ev = sj.get("event", {})
        itf = sj.get("interferer")
        az_i, tt = None, np.arange(0.05, 10.0, 0.1)
        if itf:
            wp_i = np.array(
                [[0.0, itf["x_start"], itf["offset_m"], itf["src_z_m"]],
                 [10.0, itf["x_end"], itf["offset_m"], itf["src_z_m"]]])
            az_i, _, _, _ = apparent_azel_deg(tt, wp_i, mic, c)

        f_, t_, Z = stft(foa[:, 0], fs=fs, nperseg=1024, noverlap=1024 - 128,
                         padded=False)
        S = 20 * np.log10(np.abs(Z) + 1e-12)
        S -= S.max()
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].pcolormesh(t_, f_, S, shading="auto", cmap="viridis",
                           vmin=-70, vmax=0)
        axes[0].set_ylim(0, 4000)
        axes[0].set_ylabel("Frequency (Hz)")
        for tx in (ev.get("t_on"), ev.get("t_off")):
            if tx is not None:
                axes[0].axvline(tx, color="w", ls=":", lw=1.5)
        title = name
        if itf:
            title += (f"  siren {ev['t_on']:.1f}-{ev['t_off']:.1f}s  "
                      f"SNR={sj['noise']['snr_db']:.1f}dB  "
                      f"SIR={itf['sir_db']:.1f}dB")
        axes[0].set_title(title)
        axes[1].plot(t_lab, az_lab, "s", ms=4,
                     label="siren label az (active frames only)")
        if az_i is not None:
            axes[1].plot(tt, az_i, "--", color="0.5", lw=1.2,
                         label="interferer (car) true az - UNLABELED")
        axes[1].set_ylim(-185, 185)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Azimuth (deg)")
        axes[1].grid(alpha=0.4)
        axes[1].legend(loc="best")
        plt.tight_layout()
        plt.savefig(pv / f"{name}_preview.png", dpi=150)
        plt.close()
        print(f"preview: {pv / (name + '_preview.png')}")


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
        for i in range(N_TRAIN + N_VAL + N_TEST):
            s = sample_scene(i)
            label = s["hazard_class"] if s["hazard_class"] != "siren" else s["siren_type"]
            print(f"{i:02d} {s['name']} {s['split']:5s} {s['side']} "
                  f"|y|={abs(s['offset_m']):5.1f}m v={s['speed_mps']:4.1f}m/s "
                  f"dir={'+x' if s['dir_x']>0 else '-x'} tcpa={s['t_cpa_s']:.1f}s "
                  f"z={s['src_z_m']:.2f} {label:12s} "
                  f"gain={s['gain_db']:+.1f}dB")
    elif mode == "gen":
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        for i in range(int(a), int(b) + 1):
            generate_clip(i)
    elif mode == "inspect":
        sys.exit(1 if inspect_all() else 0)
    elif mode == "preview":
        a, b = (sys.argv[2].split("-") + [sys.argv[2]])[:2]
        preview(range(int(a), int(b) + 1))
    elif mode == "pack":
        pack()
    else:
        print("usage: plan | gen A-B | preview A-B | inspect | pack"
              "  [--v1|--v2|--v4|--v5|--v6|--v7|--v8] (default: v3)")
