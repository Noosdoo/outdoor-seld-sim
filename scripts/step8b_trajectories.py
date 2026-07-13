# -*- coding: utf-8 -*-
"""step8の結果(summary_mc.json)から「クラス別に dir_err 最多のクリップ」を選び、
GT方位 vs 予測方位の時系列を並べた図を作る。方向誤差が
「発音の合間（低エネルギー区間）でのDOAドリフト」であることを見せるのが目的。

入力:
  --pred    : 予測CSVフォルダ（step8と同じ）
  --ds      : データセットフォルダ（step8と同じ）
  --summary : step8が書いた summary_mc.json（per_clipからクリップを選ぶ）
出力:
  --out     : trajectories_mc.png

実行例:
  python scripts/step8b_trajectories.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import step8_error_anatomy_mc as s8  # noqa: E402
from outdoor_seld.labels import read_dcase_csv  # noqa: E402

SOUND_DB = -30.0  # クリーンWがクリップ最大比これ以上のフレームを「発音中」とみなす

# dataviz標準パレット（検証済み: blue/aqua, aquaは直接ラベルで補強）
C_GT = "#2a78d6"
C_PRED = "#1baf7a"
C_BAND = "#e1e0d9"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"


def break_wrap(t, az):
    """方位の±180°またぎで線が横断しないよう NaN を挿入。"""
    t, az = list(t), list(az)
    i = 1
    while i < len(az):
        if abs(az[i] - az[i - 1]) > 180:
            t.insert(i, (t[i] + t[i - 1]) / 2)
            az.insert(i, np.nan)
            i += 1
        i += 1
    return np.array(t), np.array(az)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "out" / "predictions_v5_run2"))
    ap.add_argument("--ds", default=str(ROOT / "out" / "dataset_outdoor_siren_v5"))
    ap.add_argument("--summary",
                    default=str(ROOT / "out" / "figures_v5_analysis" / "summary_mc.json"))
    ap.add_argument("--out",
                    default=str(ROOT / "out" / "figures_v5_analysis" / "trajectories_mc.png"))
    args = ap.parse_args()
    pred_dir, ds_dir = Path(args.pred), Path(args.ds)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    classes = summary["classes"]

    # クラスごとに dir_err 最多のクリップを選ぶ
    picks = {}
    for pc in summary["per_clip"]:
        c = pc["gt_class"]
        if c not in picks or pc["dir_err"] > picks[c]["dir_err"]:
            picks[c] = pc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.patch.set_facecolor(C_SURFACE)

    for pi, cname in enumerate(classes):
        ax = axes.flat[pi]
        ax.set_facecolor(C_SURFACE)
        pc = picks[cname]
        name = pc["name"]

        gts = read_dcase_csv(ds_dir / "metadata" / f"{name}.csv")
        preds = s8.load_pred_csv(pred_dir / f"{name}.csv")
        sj = json.loads((ds_dir / "work" / name / "scene.json")
                        .read_text(encoding="utf-8"))
        n_frames = int(round(sj["scene_config"]["clip_len_sec"] / s8.LABEL_RES))
        e_db = s8.frame_energy_db(
            ds_dir / "work" / name / "foa_clean_24k.flac", n_frames)
        ci = classes.index(cname)

        # 発音中（クリーン音が鳴っている）フレームを帯で示す
        on = e_db > SOUND_DB
        k = 0
        first_band = True
        while k < n_frames:
            if on[k]:
                k2 = k
                while k2 + 1 < n_frames and on[k2 + 1]:
                    k2 += 1
                ax.axvspan(k * s8.LABEL_RES, (k2 + 1) * s8.LABEL_RES,
                           color=C_BAND, alpha=0.6, lw=0, zorder=0,
                           label=("source audible (clean W > -30 dB)"
                                  if first_band and pi == 0 else None))
                first_band = False
                k = k2 + 1
            else:
                k += 1

        # GT軌跡（青の実線）
        tg, azg = zip(*[((k + 0.5) * s8.LABEL_RES, g[1])
                        for k in sorted(gts) for g in gts[k] if int(g[0]) == ci])
        tg, azg = break_wrap(tg, azg)
        ax.plot(tg, azg, color=C_GT, lw=2, zorder=2,
                label="ground truth" if pi == 0 else None)

        # 予測（aquaの点。>20°ずれは×印で区別 = 色以外の符号化）
        ok_t, ok_az, ng_t, ng_az = [], [], [], []
        gt_map = {k: [g for g in gts[k] if int(g[0]) == ci] for k in gts}
        for k in sorted(preds):
            for p in preds[k]:
                if int(p[0]) != ci:
                    continue
                t = (k + 0.5) * s8.LABEL_RES
                gl = gt_map.get(k) or []
                if gl:
                    d = min(s8.ang_dist_deg(p[1], p[2], g[1], g[2]) for g in gl)
                else:
                    d = np.inf  # GTなし区間の予測はfa（×で示す）
                (ok_t if d <= s8.MATCH_DEG else ng_t).append(t)
                (ok_az if d <= s8.MATCH_DEG else ng_az).append(p[1])
        ax.scatter(ok_t, ok_az, s=22, color=C_PRED, zorder=3,
                   label="pred (<=20 deg)" if pi == 0 else None)
        ax.scatter(ng_t, ng_az, s=34, color=C_PRED, marker="x", lw=1.8,
                   zorder=3, label="pred (>20 deg / fa)" if pi == 0 else None)

        # 発音区間ウィンドウ
        for x, lab in ((pc["t_on"], "event window" if pi == 0 else None),
                       (pc["t_off"], None)):
            ax.axvline(x, color=C_MUTED, lw=1.2, ls="--", zorder=1, label=lab)

        ax.set_title(f"{cname}  ({name}, dir_err={pc['dir_err']})",
                     color=C_INK, fontsize=11)
        ax.grid(color=C_GRID, lw=0.7, zorder=0)
        for sp in ax.spines.values():
            sp.set_color(C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        if pi >= 2:
            ax.set_xlabel("time (s)", color=C_INK)
        if pi % 2 == 0:
            ax.set_ylabel("azimuth (deg)", color=C_INK)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("predicted vs GT azimuth - worst dir_err clip per class (v5 run2)",
                 color=C_INK)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=C_SURFACE)
    plt.close()
    print(f"wrote {args.out}")
    for cname in classes:
        pc = picks[cname]
        print(f"  {cname:<11} -> {pc['name']} (dir_err={pc['dir_err']}, "
              f"snr={pc['snr_db']}dB, sir={pc['sir_db']}dB)")


if __name__ == "__main__":
    main()
