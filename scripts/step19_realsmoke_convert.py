# -*- coding: utf-8 -*-
"""Step 19: 実録スモーク用の変換 — H3-VR録音をパイプライン形式に変換する。

入力: H3-VRのAmbiXモード録音（4ch WAV, 96kHz=現行規約/48kHz=旧規約, W,Y,Z,X = ACN/SN3D）
     ※96kHz原本は超音波解析用にそのまま保持し、本スクリプトは本体モデル用の
       24kHz版を別ファイルとして出力する（スモーク計画書2026-08-13改訂節1-2）
処理: ①24kHzへリサンプル（polyphase・アンチエイリアス込み） ②傾き補正のFOA回転（校正音の解析結果を
     --pitch/--roll/--yaw で与える。1次アンビソニックスの回転は厳密）
     ③絶対較正 — 騒音計LAeq読み値と「録音内の暗騒音区間」のA特性レベルを照合し、
     録音全体を学習の音量規約（フルスケール=143dB SPL）に合わせるゲインを適用
出力: <name>_conv.flac（W,Y,Z,X / 24kHz / PCM_24。そのままColabのdatasets/へ置ける）

使い方:
  python scripts/step19_realsmoke_convert.py --in <wav|dir> --out <dir> \
      --laeq 52.5 --laeq-window 10-70 [--pitch 8] [--roll 2] [--yaw 0]
  --laeq: 騒音計の読み値dB(A)（暗騒音1分収録時にメモした値）
  --laeq-window: その暗騒音区間の録音内秒数（例 10-70）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.calibration import spl_a  # noqa: E402

FS_OUT = 24000


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def rot_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    p, r, y = (math.radians(pitch_deg), math.radians(roll_deg),
               math.radians(yaw_deg))
    Rp = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]])
    Rr = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Ry @ Rr @ Rp


def calib_gain_db(path: Path, laeq: float, win: tuple) -> float:
    """較正ゲイン[dB]だけを、LAeq窓の区間**だけ**読んで求める。

    長時間の負例録音（例: 30分×96kHz×4ch）は全長をメモリに載せられないため、
    ゲインだけ先に求めて step19b_realsmoke_cut.py --gain-db に渡す経路を用意する。"""
    with sf.SoundFile(str(path)) as f:
        sr, n_total = f.samplerate, len(f)
        assert f.channels == 4, f"{path.name}: 4ch(AmbiX)ではありません"
        assert sr in (96000, 48000, 24000), f"{path.name}: 想定外のfs={sr}"
        a, b = int(win[0] * sr), int(win[1] * sr)
        assert b <= n_total, "laeq-windowが録音長を超えています"
        f.seek(a)
        seg = f.read(b - a, dtype="float64", always_2d=True)[:, 0]
    down = sr // FS_OUT
    if down > 1:
        seg = resample_poly(seg, 1, down)
    measured = spl_a(seg, FS_OUT)
    return float(laeq - measured)


def convert(path: Path, out_dir: Path, laeq: float, win: tuple,
            pitch: float, roll: float, yaw: float) -> Path:
    wav, sr = sf.read(path, dtype="float64")
    assert wav.ndim == 2 and wav.shape[1] == 4, \
        f"{path.name}: 4ch(AmbiX)ではありません shape={wav.shape}"
    assert sr in (96000, 48000, 24000), f"{path.name}: 想定外のfs={sr}"
    if sr == 96000:
        wav = resample_poly(wav, 1, 4, axis=0)
    elif sr == 48000:
        wav = resample_poly(wav, 1, 2, axis=0)
    # 傾き補正（W不変、X,Y,ZベクトルにR適用。ch順=W,Y,Z,X）
    if pitch or roll or yaw:
        R = rot_matrix(pitch, roll, yaw)
        xyz = wav[:, [3, 1, 2]] @ R.T
        wav[:, 3], wav[:, 1], wav[:, 2] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # 絶対較正: 暗騒音区間のWのA特性レベルを騒音計読み値に一致させるゲイン
    a, b = int(win[0] * FS_OUT), int(win[1] * FS_OUT)
    assert b <= len(wav), "laeq-windowが録音長を超えています"
    measured = spl_a(wav[a:b, 0], FS_OUT)
    gain = 10.0 ** ((laeq - measured) / 20.0)
    wav = wav * gain
    peak = float(np.max(np.abs(wav)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + "_conv.flac")
    sf.write(out, wav.astype(np.float64), FS_OUT, subtype="PCM_24")
    print(f"{path.name}: 実測{measured:.1f}dBA→計器{laeq:.1f}dBA gain={20*np.log10(gain):+.1f}dB "
          f"peak={peak:.3f}{'  ⚠️CLIP注意' if peak > 0.99 else ''} -> {out.name}")
    return out


def main() -> int:
    src = Path(_arg("--in"))
    out_dir = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "converted")))
    laeq = float(_arg("--laeq"))
    a, b = _arg("--laeq-window", "0-60").split("-")
    win = (float(a), float(b))
    pitch = float(_arg("--pitch", "0"))
    roll = float(_arg("--roll", "0"))
    yaw = float(_arg("--yaw", "0"))
    files = sorted(src.glob("*.wav")) if src.is_dir() else [src]
    assert files, f"wavが見つかりません: {src}"
    if "--gain-only" in sys.argv:
        # 長時間録音用: 全長を読まずにゲインだけ出す（step19b --gain-db へ渡す）
        for f in files:
            print(f"{f.name}: gain-db {calib_gain_db(f, laeq, win):+.2f}")
        return 0
    for f in files:
        with sf.SoundFile(str(f)) as sfh:
            secs = len(sfh) / sfh.samplerate
        assert secs <= 600.0, (
            f"{f.name}: {secs/60:.1f}分は全長変換にはメモリ的に長すぎます。"
            "`--gain-only` でゲインを求め、step19b_realsmoke_cut.py --gain-db で"
            "切り出しながら変換してください")
        convert(f, out_dir, laeq, win, pitch, roll, yaw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
