# -*- coding: utf-8 -*-
"""E0: val(fold2) の FOA flac にゼロ位相HPFをかけた派生データセットを作る。

サーバーの datasets/outdoor_siren_v11 を入力に、
datasets/outdoor_siren_v11_hpf{50,100,200}/foa/ を新規作成する（元データ不変更）。
foa 以外（metadata 等）はシンボリックリンクで共有する。

使い方（PSELDNetsルートで）:
    .venv/bin/python _e0_hpf_render.py [--src datasets/outdoor_siren_v11] [--cutoffs 50 100 200]

検品として、最初の1本でカットオフの±1オクターブ点の実測減衰[dB]を表示する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def hpf_zero_phase(x: np.ndarray, fs: int, fc: float) -> np.ndarray:
    """ゼロ位相HPF（バタワース4次×前後向き=実効8次相当）。x: (n, ch)"""
    sos = butter(4, fc, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def band_level_db(x: np.ndarray, fs: int, f0: float) -> float:
    """f0±1/6oct帯のWchパワー[dB]（検品用）。"""
    spec = np.abs(np.fft.rfft(x[:, 0])) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f >= f0 * 2 ** (-1 / 6)) & (f < f0 * 2 ** (1 / 6))
    p = spec[band].sum()
    return 10.0 * np.log10(max(p, 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/outdoor_siren_v11")
    ap.add_argument("--cutoffs", nargs="+", type=float, default=[50.0, 100.0, 200.0])
    ap.add_argument("--fold", default="fold2")  # val のみ
    args = ap.parse_args()

    src = Path(args.src).resolve()
    flacs = sorted((src / "foa").glob(f"{args.fold}_*.flac"))
    assert flacs, f"no {args.fold} flacs under {src / 'foa'}"
    print(f"src={src} {args.fold}: {len(flacs)}本")

    for fc in args.cutoffs:
        dst = src.parent / f"{src.name}_hpf{int(fc)}"
        (dst / "foa").mkdir(parents=True, exist_ok=True)
        # foa 以外はシンボリックリンクで共有（既存リンクはそのまま）
        for item in src.iterdir():
            if item.name == "foa":
                continue
            link = dst / item.name
            if not link.exists():
                link.symlink_to(item, target_is_directory=item.is_dir())
        done = 0
        for f in flacs:
            out = dst / "foa" / f.name
            if out.exists():
                continue
            x, fs = sf.read(f)                       # (n, 4)
            y = hpf_zero_phase(np.asarray(x, np.float64), fs, fc)
            sf.write(out, y.astype(np.float32), fs, subtype="PCM_24")
            if done == 0:   # 検品: 最初の1本で減衰量を実測
                for probe in [fc / 2, fc, fc * 2, 1000.0]:
                    d = band_level_db(y, fs, probe) - band_level_db(
                        np.asarray(x, np.float64), fs, probe)
                    print(f"  [検品 fc={fc:.0f}Hz] {probe:7.1f}Hz: {d:+6.1f} dB")
            done += 1
        print(f"hpf{int(fc)}: {done}本 新規作成 -> {dst / 'foa'}")


if __name__ == "__main__":
    main()
