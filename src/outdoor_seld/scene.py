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

import dataclasses  # SceneConfigを「フィールドの塊」として簡潔に書くため
import json          # scene_config を人間が読める形でファイルに残すため
import os             # 出力ディレクトリの作成に使う

import numpy as np


@dataclasses.dataclass
class SceneConfig:
    # このクリップの識別名（ファイル名の元になる）
    clip_name: str = "fold0_room0_mix001"
    clip_len_sec: float = 10.0   # クリップ全体の長さ [秒]
    fs_sim: int = 48000          # シミュレーション実行レート（高いほど正確・重い）
    fs_out: int = 24000          # PSELDNets 入力レート（デシメート後、これに合わせて渡す）
    # マイク（静止・原点付近・高さ1.5m）＝人が装着するイメージの高さ
    mic_pos: tuple = (0.0, 0.0, 1.5)
    # 音源軌道（等速直線, 10 m/s）＝ (x,y,z) の開始点と終了点。速度は自動的に
    # (終了−開始)/clip_len_sec で決まる（別途速度を指定する変数はない）
    src_start: tuple = (-50.0, 5.0, 1.0)
    src_end: tuple = (50.0, 5.0, 1.0)
    # 大気条件（DynamicSound 既定＝論文実験と同一）。音速や大気吸収の計算に使われる
    temperature_c: float = 20.0
    pressure_atm: float = 1.0
    rel_humidity: float = 50.0
    # 減衰の基準距離（DynamicSound 実装で 1/distance に固定。ここでは変更できず記録のみ）
    r0_m: float = 1.0
    # 音源の音量調整（dB）。0なら siren.py が作った波形そのまま
    source_gain_db: float = 0.0
    siren_type: str = "wail"     # "wail"（うねり型）| "peepo"（ピーポー2音交互）
    class_name: str = "Siren"    # ラベルCSVに書くクラス名（表示用）
    class_idx: int = 4           # cls_indices_train.tsv の行順（0始まり）で Siren=4 という約束
    ground_z: float = 0.0        # 反射面（地面）の高さ。z=0が地面という設定

    def waypoints_direct(self):
        # 直接音の軌道: [開始時刻,x,y,z] と [終了時刻,x,y,z] の2点だけ
        # （between は DynamicSound / geometry.py 側が等速直線で線形補間する）
        return np.array([[0.0, *self.src_start], [self.clip_len_sec, *self.src_end]])

    def waypoints_mirror(self):
        # 地面反射音の軌道: 直接音の軌道を z 方向にだけ地面(ground_z)に対して鏡映させる
        s = np.array(self.src_start, dtype=float)
        e = np.array(self.src_end, dtype=float)
        s[2] = 2 * self.ground_z - s[2]   # 地面を挟んで反対側にある「鏡像音源」のz座標
        e[2] = 2 * self.ground_z - e[2]
        return np.array([[0.0, *s], [self.clip_len_sec, *e]])

    def save_json(self, path):
        # このシーンの全パラメータをそのままJSON化して保存（あとで再現・検算するため）
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


def run_mono_sim(scene: SceneConfig, source_wav_path: str, out_wav_path: str,
                 mirror: bool = False) -> None:
    """1音源×無指向1chマイクで DynamicSound を実行し、モノラルWAVを出力する。"""
    import dynamic_sound as ds  # ここで初めて重いライブラリを読み込む（呼ばれた時だけ）

    # 直接音か鏡像音か、どちらの軌道を使うかを選ぶ
    wp = scene.waypoints_mirror() if mirror else scene.waypoints_direct()
    quat = [1.0, 0.0, 0.0, 0.0]  # 向き（回転）は今回使わないので単位クォータニオン固定
    # DynamicSound用の「時刻付き位置」の経路オブジェクトを2点で作る（音源用）
    source_path = ds.Path([[wp[0, 0], *wp[0, 1:4], *quat],
                           [wp[1, 0], *wp[1, 1:4], *quat]])
    # マイク用の経路（静止なので同じ位置を開始・終了に置くだけ）
    mic_path = ds.Path([[0.0, *scene.mic_pos, *quat],
                        [scene.clip_len_sec, *scene.mic_pos, *quat]])

    # 気温・気圧・湿度を渡してシミュレータ本体を初期化（大気吸収の計算に使われる）
    sim = ds.Simulation(temperature=scene.temperature_c,
                        pressure=scene.pressure_atm,
                        relative_humidity=scene.rel_humidity)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)  # 出力先フォルダがなければ作る
    # 無指向性1chマイクをシミュレーションに登録。出力はこのwavファイルに書かれる
    mic = ds.microphones.Microphone(out_wav_path, sample_rate=scene.fs_sim)
    sim.add_microphone(path=mic_path, microphone=mic)
    # loop=False 必須（既定 True は短い音源をループ再生してしまう）
    src = ds.sources.AudioFile(filename=source_wav_path,
                               gain_db=scene.source_gain_db, loop=False)
    sim.add_source(path=source_path, source=src)
    sim.run()  # ここで実際に1サンプルずつの物理計算が走る（重い処理）


def decimate_to_out_rate(x: np.ndarray, fs_sim: int, fs_out: int) -> np.ndarray:
    """アンチエイリアス込みの整数比デシメーション（48k→24k なら 1/2）。"""
    from scipy.signal import resample_poly

    assert fs_sim % fs_out == 0, "fs_sim must be an integer multiple of fs_out"
    q = fs_sim // fs_out          # 何分の1に間引くか（48000/24000=2）
    if q == 1:
        return x.astype(np.float64)  # 変換不要ならそのまま返す
    # up=1,down=q でローパスフィルタ込みのリサンプリング（折り返し雑音を防ぐ）
    return resample_poly(x.astype(np.float64), up=1, down=q)
