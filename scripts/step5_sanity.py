"""Step 5: サニティチェックと可視化（ゼミ発表用素材）。

1. サニティチェック1（FOA規約・E2E版）: 真横 az=+90 deg の静止音源を
   DynamicSound→本パイプラインの FOA 化に通し、Y=W・X=Z=0 を確認
   （単体テスト tests/test_foa.py の実シミュレーション版）
2. サニティチェック2（音とラベルの整合）: 生成 FOA から音響インテンシティ
   ベクトル法で DOA を推定し、Step 3 ラベルと重ね描き → 一致度を数値化
3. スペクトログラム（ドップラー）と 直接音のみ vs 地面反射込み の比較
   （反射のコムフィルタ縞、論文 IV-D と同傾向）

出力: out/figures/*.png, out/figures/step5_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PEEPO = "--peepo" in sys.argv
CLIP = ROOT / "out" / ("clip_peepo" if PEEPO else "clip")
FIG = ROOT / "out" / ("figures_peepo" if PEEPO else "figures")
FIG.mkdir(parents=True, exist_ok=True)

import tqdm as _tqdm_mod  # noqa: E402

_tqdm_mod.tqdm = lambda iterable=None, **kw: iterable  # type: ignore

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf  # noqa: E402

from outdoor_seld.foa import encode_foa_timevarying, intensity_vector_doa  # noqa: E402
from outdoor_seld.geometry import (doa_unit_vectors, solve_emission_times,  # noqa: E402
                                   sound_speed)
from outdoor_seld.labels import read_dcase_csv  # noqa: E402
from outdoor_seld.scene import SceneConfig, decimate_to_out_rate, run_mono_sim  # noqa: E402
from outdoor_seld.siren import make_siren  # noqa: E402

RESULTS = {}


# ------------------------------------------------ sanity 1 (E2E static)
def sanity1_static_side():
    print("[sanity1] static source at az=+90 (left), el=0 through full pipeline")
    scene = SceneConfig(clip_len_sec=1.0, fs_sim=48000, fs_out=24000,
                        mic_pos=(0.0, 0.0, 1.5),
                        src_start=(0.0, 5.0, 1.5), src_end=(0.0, 5.0, 1.5))
    dry = make_siren(1.0, scene.fs_sim)
    sf.write(CLIP / "_sanity_dry.wav", dry.astype(np.float32), scene.fs_sim)
    run_mono_sim(scene, str(CLIP / "_sanity_dry.wav"),
                 str(CLIP / "_sanity_mono.wav"))
    x, sr = sf.read(CLIP / "_sanity_mono.wav")
    mono = decimate_to_out_rate(np.asarray(x, np.float64), 48000, 24000)
    tr = np.arange(len(mono)) / 24000
    c = sound_speed(scene.temperature_c)
    te, ps = solve_emission_times(tr, scene.waypoints_direct(),
                                  np.array(scene.mic_pos), c)
    u, _ = doa_unit_vectors(ps, np.array(scene.mic_pos))
    foa = encode_foa_timevarying(mono, u)
    w, y, z, xch = foa
    act = np.abs(w) > 0.05 * np.max(np.abs(w))
    ry = float(np.sqrt(np.mean(y[act] ** 2) / np.mean(w[act] ** 2)))
    rx = float(np.sqrt(np.mean(xch[act] ** 2) / np.mean(w[act] ** 2)))
    rz = float(np.sqrt(np.mean(z[act] ** 2) / np.mean(w[act] ** 2)))
    ok = abs(ry - 1.0) < 1e-6 and rx < 1e-6 and rz < 1e-6
    print(f"  Y/W={ry:.8f} X/W={rx:.2e} Z/W={rz:.2e} -> {'PASS' if ok else 'FAIL'}")
    RESULTS["sanity1_static_az90"] = {"Y_over_W": ry, "X_over_W": rx,
                                      "Z_over_W": rz, "pass": bool(ok)}


# ------------------------------------------------ sanity 2 (IV-DOA vs labels)
def sanity2_iv_vs_labels():
    print("[sanity2] intensity-vector DOA vs Step3 labels")
    scene = SceneConfig(**json.loads((CLIP / "scene_config.json").read_text()))
    foa, fs = sf.read(CLIP / "foa_direct_24k.flac")
    foa = np.asarray(foa, np.float64).T  # (4, N)
    t_iv, az_iv, el_iv, energy = intensity_vector_doa(
        foa, fs, frame_sec=0.1, fmin=200, fmax=4000)
    # 無音フレーム除外
    e_thr = np.max(energy) * 1e-6
    az_iv[energy < e_thr] = np.nan
    el_iv[energy < e_thr] = np.nan

    labels = read_dcase_csv(CLIP / f"{scene.clip_name}.csv")
    frames = sorted(labels.keys())
    t_lab = np.array([(k + 0.5) * 0.1 for k in frames])
    az_lab = np.array([labels[k][0][1] for k in frames])
    el_lab = np.array([labels[k][0][2] for k in frames])

    # 一致度（両方有効なフレーム）
    az_err, el_err = [], []
    for k in frames:
        if k < len(az_iv) and np.isfinite(az_iv[k]):
            da = (az_iv[k] - labels[k][0][1] + 180) % 360 - 180
            az_err.append(abs(da))
            el_err.append(abs(el_iv[k] - labels[k][0][2]))
    az_err, el_err = np.array(az_err), np.array(el_err)
    res = {
        "n_label_frames": len(frames),
        "n_compared": int(len(az_err)),
        "azimuth_median_abs_err_deg": float(np.median(az_err)),
        "azimuth_max_abs_err_deg": float(np.max(az_err)),
        "elevation_median_abs_err_deg": float(np.median(el_err)),
        "elevation_max_abs_err_deg": float(np.max(el_err)),
    }
    RESULTS["sanity2_iv_vs_labels"] = res
    print(f"  az median|err|={res['azimuth_median_abs_err_deg']:.2f} deg "
          f"(max {res['azimuth_max_abs_err_deg']:.2f}), "
          f"el median|err|={res['elevation_median_abs_err_deg']:.2f} deg")

    dbg = np.load(CLIP / "label_debug.npz")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(dbg["t_frames"], dbg["az"], "-", lw=1, color="0.6",
                 label="apparent DOA (continuous, geometry.py)")
    axes[0].plot(t_lab, az_lab, "s", ms=3.5, label="DCASE label (0.1 s, int deg)")
    axes[0].plot(t_iv, az_iv, ".", ms=4, alpha=0.7,
                 label="intensity-vector estimate from FOA")
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.4)
    axes[0].set_title("sanity check 2: audio vs label consistency (direct-only FOA)")
    axes[1].plot(dbg["t_frames"], dbg["el"], "-", lw=1, color="0.6")
    axes[1].plot(t_lab, el_lab, "s", ms=3.5)
    axes[1].plot(t_iv, el_iv, ".", ms=4, alpha=0.7)
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG / "doa_labels_vs_iv.png", dpi=150)
    plt.close()
    print("  wrote figures/doa_labels_vs_iv.png")


# ------------------------------------------------ spectrograms
def spectrograms():
    print("[spec] spectrograms: Doppler + ground-reflection comb filter")
    from scipy.signal import stft

    def spec_db(path):
        x, fs = sf.read(path)
        w = np.asarray(x, np.float64).T[0]
        f, t, Z = stft(w, fs=fs, nperseg=1024, noverlap=1024 - 128, padded=False)
        S = 20 * np.log10(np.abs(Z) + 1e-12)
        return f, t, S - S.max()

    f1, t1, S1 = spec_db(CLIP / "foa_direct_24k.flac")
    f2, t2, S2 = spec_db(CLIP / "foa_withrefl_24k.flac")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (f, t, S, title) in zip(
            axes, [(f1, t1, S1, "direct only"),
                   (f2, t2, S2, "direct + ground reflection (comb filter)")]):
        im = ax.pcolormesh(t, f, S, shading="auto", cmap="viridis",
                           vmin=-70, vmax=0)
        ax.set_ylim(0, 6000)
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, label="dB")
    plt.savefig(FIG / "spectrogram_direct_vs_refl.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  wrote figures/spectrogram_direct_vs_refl.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = args[0] if args else "all"
    if mode in ("all", "1"):
        sanity1_static_side()
    if mode in ("all", "2"):
        sanity2_iv_vs_labels()
    if mode in ("all", "spec"):
        spectrograms()
    out = FIG / "step5_results.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(RESULTS)
    out.write_text(json.dumps(prev, indent=2))
    print("wrote figures/step5_results.json")
