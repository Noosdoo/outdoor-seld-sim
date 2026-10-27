# -*- coding: utf-8 -*-
"""Step 19: 実録スモーク用の変換 — H3-VR録音をパイプライン形式に変換する。

入力: H3-VRのAmbiXモード録音（4ch WAV, 96kHz=現行規約/48kHz=旧規約, W,Y,Z,X = ACN/SN3D）
     ※96kHz原本は超音波解析用にそのまま保持し、本スクリプトは本体モデル用の
       24kHz版を別ファイルとして出力する（スモーク計画書2026-08-13改訂節1-2）
処理: ①24kHzへリサンプル（polyphase・アンチエイリアス込み） ②傾き補正のFOA回転（校正音の解析結果を
     --pitch/--roll/--yaw で与える。1次アンビソニックスの回転は厳密）
     ③絶対較正 — 騒音計LAeq読み値と「録音内の暗騒音区間」のA特性レベルを照合し、
     録音全体を学習の音量規約（フルスケール=143dB SPL）に合わせるゲインを適用
     2026-10-26（Fable 仕様（先行分）§3・決定 ③・Astra R30-08）: --convention v12_149 で**合成と同じ実効の約束**（0 dBFS の正弦波 ≡ 149 dB SPL・
     ゲイン = 計器 − 録音の A 特性レベル − 6 dB）。既定は v9_143（旧のまま）。較正の json に convention・fullscale_sine_spl・headroom_db を書く
出力: <name>_conv.flac（W,Y,Z,X / 24kHz / PCM_24。そのままColabのdatasets/へ置ける）

使い方:
  python scripts/step19_realsmoke_convert.py --in <wav|dir> --out <dir> \
      --laeq 52.5 --laeq-window 10-70 [--pitch 8] [--roll 2] [--yaw 0]
  --clap-window: 波形で確認した拍手区間（原本秒 START:END、複数はカンマ）。
      原本1本ずつ指定。拍手のピークを品質検査から除外するだけで、音声・時刻は変更しない。
      *.peak.json に原本／変換後の全体・拍手・拍手以外のピークを保存する。
  --laeq: 騒音計の読み値dB(A)（暗騒音1分収録時にメモした値）
  --laeq-window: その暗騒音区間の録音内秒数（例 10-70）
  --convention: v9_143（既定・旧）／v12_149（合成の約束にそろえる・学習し直し v30 からの評価用・calibration.FIELD_CONVENTIONS）
  --calib <wav>: **校正録音（セッション冒頭の 60 秒）を別ファイルで指定**（2026-09-06・監査 R01）。
      補正量はこのファイルの --laeq-window から 1 回だけ求め、--in の全テイク（20〜30 秒の本番）に同じ量を適用する。
      calibration_id（既定 = 校正ファイル名）と補正量を <out>/calibration_<id>.json に書く。step19b には
      --gain-db と --calibration-id を渡す（`--gain-only --calib ...` でその 2 つを表示）。
      録音機のゲイン設定を変えたら校正を取り直し、別の calibration_id にする。
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

from outdoor_seld.calibration import FIELD_CONVENTION_DEFAULT, FIELD_CONVENTIONS, spl_a  # noqa: E402

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


def convention_of(name: str | None) -> tuple:
    """約束の名前 → (名前, 0 dBFS の正弦波の dB SPL, ゲインから引く dB)。知らない名前は止める。"""
    name = name or FIELD_CONVENTION_DEFAULT
    if name not in FIELD_CONVENTIONS:
        raise ValueError(f"知らない約束 {name!r}（{sorted(FIELD_CONVENTIONS)} のどれか）")
    spl, hr = FIELD_CONVENTIONS[name]
    return name, float(spl), float(hr)


def calib_gain_db(path: Path, laeq: float, win: tuple, convention: str | None = None) -> float:
    """較正ゲイン[dB]だけを、LAeq窓の区間**だけ**読んで求める。

    長時間の負例録音（例: 30分×96kHz×4ch）は全長をメモリに載せられないため、
    ゲインだけ先に求めて step19b_realsmoke_cut.py --gain-db に渡す経路を用意する。
    convention: None／"v9_143" = 旧（計器 − 測った値）・"v12_149" = 合成の約束（さらに −6 dB）。"""
    _, _, hr = convention_of(convention)
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
    return float(laeq - measured - hr)


def wav_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def peak_report(wav, sr, clap_windows=()):
    """明示した拍手区間だけを品質検査から除く。波形・時刻は変更しない。"""
    mask = np.ones(len(wav), dtype=bool)
    duration = len(wav) / sr
    for a, b in clap_windows:
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration):
            raise ValueError("拍手区間は録音内の有限な開始:終了秒で指定してください")
        mask[int(a * sr):int(math.ceil(b * sr))] = False
    if not mask.any():
        raise ValueError("録音全体を拍手区間として除外できません")
    return {"peak_all": float(np.max(np.abs(wav))),
            "peak_outside_clap": float(np.max(np.abs(wav[mask]))),
            "peak_clap": float(np.max(np.abs(wav[~mask]))) if (~mask).any() else None}


def save_peak_report(path, out, raw_report, wav, clap_windows):
    import json
    converted = peak_report(wav, FS_OUT, clap_windows)
    warning = max(raw_report["peak_outside_clap"], converted["peak_outside_clap"]) > .99
    rec = {"source": path.name, "clap_windows_s": list(clap_windows),
           "raw": raw_report, "converted": converted, "clip_warning_outside_clap": warning,
           "policy": "拍手だけの音割れはテイク失敗に数えない。波形と時刻は保持。"}
    out.with_suffix(".peak.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    if warning:
        print("  ⚠️CLIP注意（拍手以外。原本と変換後のピークを別途確認）")
    elif clap_windows and max(raw_report["peak_all"], converted["peak_all"]) > .99:
        print("  拍手区間だけのピーク超過：テイク失敗に数えない")


def convert(path: Path, out_dir: Path, laeq: float, win: tuple,
            pitch: float, roll: float, yaw: float, clap_windows=(), convention: str | None = None) -> Path:
    wav, sr = sf.read(path, dtype="float64")
    assert wav.ndim == 2 and wav.shape[1] == 4, \
        f"{path.name}: 4ch(AmbiX)ではありません shape={wav.shape}"
    assert sr in (96000, 48000, 24000), f"{path.name}: 想定外のfs={sr}"
    raw_report = peak_report(wav, sr, clap_windows)
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
    gain = 10.0 ** ((laeq - measured - convention_of(convention)[2]) / 20.0)
    wav = wav * gain
    peak = float(np.max(np.abs(wav)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + "_conv.flac")
    sf.write(out, wav.astype(np.float64), FS_OUT, subtype="PCM_24")
    print(f"{path.name}: 実測{measured:.1f}dBA→計器{laeq:.1f}dBA gain={20*np.log10(gain):+.1f}dB "
          f"peak={peak:.3f} -> {out.name}")
    save_peak_report(path, out, raw_report, wav, clap_windows)
    return out


def calibration_record(calib: Path, laeq: float, win: tuple, calib_id: str, out_dir: Path, convention: str | None = None) -> dict:
    """校正録音から補正量を 1 回だけ求め、calibration_<id>.json に記録する（R01）。約束（convention）と換算の値も書く（先行分 §3）。"""
    import json
    name, spl, hr = convention_of(convention)
    gain_db = calib_gain_db(calib, laeq, win, convention=name)
    rec = {"calibration_id": calib_id, "calib_file": calib.name, "laeq_dba": laeq,
           "laeq_window_s": [win[0], win[1]], "gain_db": round(gain_db, 3),
           "convention": name, "fullscale_sine_spl": spl, "headroom_db": hr}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"calibration_{calib_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def convert_with_gain(path: Path, out_dir: Path, gain_db: float, pitch: float, roll: float, yaw: float, clap_windows=()) -> Path:
    """校正済みの補正量（dB）を渡して変換する（本番テイク用。テイク自身から LAeq は測らない）。"""
    wav, fs = sf.read(str(path), dtype="float64", always_2d=True)
    assert wav.shape[1] == 4, f"{path.name}: 4ch(FOA)ではありません"
    raw_report = peak_report(wav, fs, clap_windows)
    if fs != FS_OUT:
        wav = resample_poly(wav, FS_OUT, fs, axis=0)
    R = rot_matrix(pitch, roll, yaw)
    xyz = wav[:, [3, 1, 2]] @ R.T
    wav = np.column_stack([wav[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 0]])
    wav = wav * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(wav)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + "_conv.flac")
    sf.write(out, wav, FS_OUT, subtype="PCM_24")
    print(f"{path.name}: 校正済み gain={gain_db:+.1f}dB peak={peak:.3f} -> {out.name}")
    save_peak_report(path, out, raw_report, wav, clap_windows)
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
    conv = convention_of(_arg("--convention"))[0]
    files = wav_files(src) if src.is_dir() else [src]
    assert files, f"wavが見つかりません: {src}"
    clap_arg = _arg("--clap-window")
    if clap_arg and src.is_dir():
        raise ValueError("--clap-window は原本ごとに指定してください（ディレクトリ一括指定不可）")
    clap_windows = tuple(tuple(map(float, w.split(":"))) for w in clap_arg.split(",")) if clap_arg else ()
    if any(len(w) != 2 for w in clap_windows):
        raise ValueError("--clap-window START:END[,START:END] を指定してください")
    calib = _arg("--calib")
    if calib:
        calib = Path(calib)
        calib_id = _arg("--calib-id", calib.stem)
        if not _arg("--laeq-window"):
            # 再監査 N04: 騒音計で測った区間と同じ区間を使う。校正録音が「測った 60 秒そのもの」なら全長（既定）
            with sf.SoundFile(str(calib)) as fh:
                win = (0.0, len(fh) / fh.samplerate)
            print(f"校正窓は未指定 → 校正録音の全長 0-{win[1]:.1f}s を使う（騒音計の測定区間と同じ区間であること）")
        rec = calibration_record(calib, laeq, win, calib_id, out_dir, convention=conv)
        print(f"校正 {calib.name}: 窓 {win[0]:g}-{win[1]:g}s 計器 {laeq:.1f} dBA → gain-db {rec['gain_db']:+.2f}  calibration_id={calib_id}  約束 {conv}")
        if "--gain-only" in sys.argv:
            print(f"step19b へ: --gain-db {rec['gain_db']:+.2f} --calibration-id {calib_id}")
            return 0
        for f in files:
            if f.resolve() == calib.resolve():
                continue
            convert_with_gain(f, out_dir, rec["gain_db"], pitch, roll, yaw, clap_windows)
        return 0
    if "--gain-only" in sys.argv:
        # 長時間録音用: 全長を読まずにゲインだけ出す（step19b --gain-db へ渡す）
        for f in files:
            print(f"{f.name}: gain-db {calib_gain_db(f, laeq, win, convention=conv):+.2f}  約束 {conv}")
        return 0
    for f in files:
        with sf.SoundFile(str(f)) as sfh:
            secs = len(sfh) / sfh.samplerate
        assert secs <= 600.0, (
            f"{f.name}: {secs/60:.1f}分は全長変換にはメモリ的に長すぎます。"
            "`--gain-only` でゲインを求め、step19b_realsmoke_cut.py --gain-db で"
            "切り出しながら変換してください")
        convert(f, out_dir, laeq, win, pitch, roll, yaw, clap_windows, convention=conv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
