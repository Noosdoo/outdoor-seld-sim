"""シーン定義と DynamicSound 実行のラッパ。

方針（NOTES.md の解析エンコードFOA方式）:
- DynamicSound には原点の無指向 Microphone 1本を置き、音源ごとに**別々の**
  シミュレーションを実行して物理適用済みモノラルを得る
  （FOA の方向ゲインは音源ごとに異なるため、混ざる前に取り出す必要がある）
- 地面反射は z→−z 鏡映軌道の第2音源として別シミュレーション
  （DynamicSound 内蔵の鏡像法と同じ物理・同じ完全反射係数 1.0）
- 基準距離 r0 = 1 m は DynamicSound `attenuations.geometric = 1/distance` に固定
  （1 m でゲイン1）。設定として記録するが変更不可であることに注意。
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np


@dataclasses.dataclass
class SceneConfig:
    clip_name: str = "fold0_room0_mix001"
    clip_len_sec: float = 10.0
    fs_sim: int = 48000          # シミュレーション実行レート
    fs_out: int = 24000          # PSELDNets 入力レート（デシメート後）
    # マイク（静止・原点付近・高さ1.5m）
    mic_pos: tuple = (0.0, 0.0, 1.5)
    # 音源軌道（等速直線, 10 m/s）
    src_start: tuple = (-50.0, 5.0, 1.0)
    src_end: tuple = (50.0, 5.0, 1.0)
    # 大気条件（DynamicSound 既定＝論文実験と同一）
    temperature_c: float = 20.0
    pressure_atm: float = 1.0
    rel_humidity: float = 50.0
    # 減衰の基準距離（DynamicSound 実装で 1/distance に固定。記録用）
    r0_m: float = 1.0
    # 音源
    source_gain_db: float = 0.0
    siren_type: str = "wail"     # "wail"（うねり型）| "peepo"（ピーポー2音交互）
    class_name: str = "Siren"
    class_idx: int = 4           # cls_indices_train.tsv の行順（0始まり）で Siren=4
    ground_z: float = 0.0        # 反射面 z=0

    def waypoints_direct(self):
        return np.array([[0.0, *self.src_start], [self.clip_len_sec, *self.src_end]])

    def waypoints_mirror(self):
        s = np.array(self.src_start, dtype=float)
        e = np.array(self.src_end, dtype=float)
        s[2] = 2 * self.ground_z - s[2]
        e[2] = 2 * self.ground_z - e[2]
        return np.array([[0.0, *s], [self.clip_len_sec, *e]])

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


def run_mono_sim(scene: SceneConfig, source_wav_path: str, out_wav_path: str,
                 mirror: bool = False) -> None:
    """1音源×無指向1chマイクで DynamicSound を実行し、モノラルWAVを出力する。"""
    import dynamic_sound as ds

    wp = scene.waypoints_mirror() if mirror else scene.waypoints_direct()
    quat = [1.0, 0.0, 0.0, 0.0]
    source_path = ds.Path([[wp[0, 0], *wp[0, 1:4], *quat],
                           [wp[1, 0], *wp[1, 1:4], *quat]])
    mic_path = ds.Path([[0.0, *scene.mic_pos, *quat],
                        [scene.clip_len_sec, *scene.mic_pos, *quat]])

    sim = ds.Simulation(temperature=scene.temperature_c,
                        pressure=scene.pressure_atm,
                        relative_humidity=scene.rel_humidity)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    mic = ds.microphones.Microphone(out_wav_path, sample_rate=scene.fs_sim)
    sim.add_microphone(path=mic_path, microphone=mic)
    # loop=False 必須（既定 True は短い音源をループ再生してしまう）
    src = ds.sources.AudioFile(filename=source_wav_path,
                               gain_db=scene.source_gain_db, loop=False)
    sim.add_source(path=source_path, source=src)
    sim.run()


def decimate_to_out_rate(x: np.ndarray, fs_sim: int, fs_out: int) -> np.ndarray:
    """アンチエイリアス込みの整数比デシメーション（48k→24k なら 1/2）。"""
    from scipy.signal import resample_poly

    assert fs_sim % fs_out == 0, "fs_sim must be an integer multiple of fs_out"
    q = fs_sim // fs_out
    if q == 1:
        return x.astype(np.float64)
    return resample_poly(x.astype(np.float64), up=1, down=q)
