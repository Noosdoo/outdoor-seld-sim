# -*- coding: utf-8 -*-
"""本体 §2-3 の試聴用（2026-10-26）: 同じ本の暗騒音を「今の形（v12）」「折れ点 125 Hz・傾き −3 dB/oct」「折れ点 1,000 Hz・傾き −5 dB/oct」の 3 本で描き、
家の背景の抜粋（10/7 の測定の録音の通過の間の 10 秒・家の調整用の録音・学内の実録ではない）と並べる。本人が聴いて「街の暗騒音として変ではない」と
言ったら背景の形の乱数（step11_v30_render.FEATURES["bg_shape"]）を入にする（言わなければ入れない）。

出力 = out/_audition_v30_bg/（git の外）: 1_今の形.wav・2_折れ点125Hz_傾き-3.wav・3_折れ点1000Hz_傾き-5.wav・4_家の背景.wav・README.txt。
どれも W（全方向）の 1 チャンネル・24 kHz・10 秒。**聴き比べのため A 特性の大きさを 4 本とも同じ値にそろえた**（形＝音色の違いだけを聴く。
大きさの違いは描画の暗騒音の dB(A) の抽選〔40〜65〕で別に決まる）。

使い方: python scripts/v30_bg_audition.py [--clip train_syn_clip00001] [--out out/_audition_v30_bg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.calibration import spl_a  # noqa: E402
from outdoor_seld.noise_v12 import diffuse_foa_urban_noise  # noqa: E402
from outdoor_seld.noise_v30 import diffuse_foa_shaped_noise, octave_levels  # noqa: E402

FS, N = 24000, 240000
TARGET_DBFS_A = -30.0          # 聴き比べの大きさ（A 特性の RMS を dBFS で・4 本とも同じ）
HOME_BG = ROOT / "out/_demo_bike_check_valmix_v2/bg/bg01.flac"     # 家の 10/7 の測定の録音の通過の間（149 の約束にそろえた写し・HB25-02 の直したあと）


def a_dbfs(x: np.ndarray) -> float:
    from outdoor_seld.calibration import K_RMS_SPL
    return spl_a(x, FS) - K_RMS_SPL


def norm(x: np.ndarray) -> np.ndarray:
    return x * 10.0 ** ((TARGET_DBFS_A - a_dbfs(x)) / 20.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="train_syn_clip00001")
    ap.add_argument("--out", default=str(ROOT / "out/_audition_v30_bg"))
    a = ap.parse_args()
    import _run_v2final_gen as g
    import step11_v30_render as v30
    row = next(r for r in g.load_plan() if r["clip_id"] == a.clip)
    s = v30.sample_scene(dict(row))
    seed = int(s["noise"]["seed"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    variants = [("1_今の形.wav", None), ("2_折れ点125Hz_傾き-3.wav", (125.0, -3.0)), ("3_折れ点1000Hz_傾き-5.wav", (1000.0, -5.0))]
    lines = [f"背景の形の試聴（本体 §2-3・{a.clip} の暗騒音の乱数 seed {seed} で同じ白色雑音から形だけ変えた・W の 1 ch・24 kHz・10 秒）", "",
             f"大きさは 4 本とも A 特性 {TARGET_DBFS_A:g} dBFS にそろえた（音色＝形の違いだけを聴く）。", ""]
    for name, shp in variants:
        rng = np.random.default_rng(seed)
        foa = diffuse_foa_urban_noise(N, FS, rng) if shp is None else diffuse_foa_shaped_noise(N, FS, rng, shp[0], shp[1])
        w = norm(foa[0])
        sf.write(str(out / name), w.astype(np.float32), FS, subtype="PCM_16")
        ob = octave_levels(w, FS); ob = ob - ob[1]
        lines.append(f"- {name}: " + ("v12 の形（63 Hz が峰・上へ −4 dB/oct）" if shp is None else f"63 Hz〜{shp[0]:g} Hz は平ら・その上 {shp[1]:g} dB/oct") +
                     "・オクターブ帯（125 Hz を 0）: " + "・".join(f"{fc:g} {v:+.1f}" for fc, v in zip((63, 125, 250, 500, 1000, 2000, 4000, 8000), ob)))
    if HOME_BG.is_file():
        y, fs = sf.read(str(HOME_BG), dtype="float64", always_2d=True)
        assert fs == FS, fs
        w = norm(y[:N, 0])
        sf.write(str(out / "4_家の背景.wav"), w.astype(np.float32), FS, subtype="PCM_16")
        ob = octave_levels(w, FS); ob = ob - ob[1]
        lines.append("- 4_家の背景.wav: 10/7 の家の測定の録音（通過の間の 10 秒・家の調整用の録音）・オクターブ帯（125 Hz を 0）: "
                     + "・".join(f"{fc:g} {v:+.1f}" for fc, v in zip((63, 125, 250, 500, 1000, 2000, 4000, 8000), ob)))
    else:
        lines.append(f"- 4_家の背景.wav: 元の {HOME_BG} が無いので作らなかった")
    lines += ["", "聴いてほしいこと: 1〜3 が「街の暗騒音（遠くの交通・空調など）」として変ではないか。2・3 は家の背景（4）に近い形の端と端（折れ点の範囲 63〜1,000 Hz・傾き −5〜−3 の両端）。",
              "変でなければ、描画 v30 で本ごとに折れ点と傾きを乱数で振る（FEATURES[\"bg_shape\"] を入）。変なら入れない（ほかの直しは進める）。"]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
