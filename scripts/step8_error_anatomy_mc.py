# -*- coding: utf-8 -*-
"""マルチクラス対応の誤り解剖（v5以降用）。step7（v3のシングルクラス版・予測埋め込み）は
検証記録として無傷のまま残し、本スクリプトは予測CSVをフォルダから読む汎用設計。

入力:
  --pred : Colab infer(mode=test) が出した予測CSVのフォルダ
           （Drive submissions/ から取得してローカル保存したもの。4列 frame,class,az,el）
  --ds   : データセットフォルダ（metadata/ と work/*/scene.json と work/*/foa_clean_24k.flac）
出力:
  --out  : summary_mc.json / confusion_mc.png / per_class_mc.png / miss_energy_mc.png

フレーム単位（0.1s）の分類（マルチクラス版）:
  ok           = 同クラスの予測が20°以内
  dir_err      = 同クラスの予測はあるが20°超（場所違い）
  substitution = 同クラス予測なし・別クラスの予測がGT方向20°以内（場所は合ってるがクラス違い＝取り違え）
  miss         = どの予測にも拾われていない（見逃し）
  fa           = どのGTにも対応しない予測（誤通知）。予測クラス別に集計
検証したい仮説:
  仮説3: 周波数帯の重複による取り違え
         BackupBeep(1000Hz)↔Siren peepo(960Hz)、Horn(410/500Hz+奇数次倍音)↔Siren wail(650-1450Hz)
         → substitution の組と、Siren は siren_type(wail/peepo) 別に内訳を出す
  仮説4: BikeBell はチリン(2s周期・減衰)の合間が無音に近く、発音区間内でもエネルギーが疎
         → クラス別 miss 率＋「見逃したフレーム vs 検出できたフレーム」のクリーン音エネルギー比較

実行例:
  python scripts/step8_error_anatomy_mc.py \
      --pred out/predictions_v5_run2 --ds out/dataset_outdoor_siren_v5 --out out/figures_v5_analysis
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.geometry import apparent_azel_deg, sound_speed  # noqa: E402
from outdoor_seld.labels import read_dcase_csv  # noqa: E402

EDGE_SEC = 0.3       # 発音区間の端とみなす幅（step7と同一）
MATCH_DEG = 20.0     # DCASEのTP判定と同じ角度しきい値
LABEL_RES = 0.1


def ang_dist_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(np.deg2rad, (az1, el1, az2, el2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def load_pred_csv(path: Path):
    """予測CSV（frame,class,az,el）→ {frame: [(cls, az, el), ...]}"""
    out = {}
    for line in path.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        v = [x.strip() for x in line.split(",")]
        out.setdefault(int(v[0]), []).append(
            (int(v[1]), float(v[2]), float(v[3])))
    return out


def frame_energy_db(clean_flac: Path, n_frames: int, fs_out: int = 24000):
    """クリーンFOAのWチャンネルから 0.1s フレームごとの RMS[dB]（クリップ内最大フレーム基準）。"""
    x, fs = sf.read(clean_flac)
    w = np.asarray(x, np.float64)[:, 0]
    spf = int(LABEL_RES * fs)
    rms = np.array([np.sqrt(np.mean(w[k * spf:(k + 1) * spf] ** 2) + 1e-30)
                    for k in range(n_frames)])
    return 20.0 * np.log10(rms / max(rms.max(), 1e-30) + 1e-30)


def analyze(pred_dir: Path, ds_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # クラス辞書（3列目=クラス名、行順=class_idx）→ 将来のクラス構成変更にも追従
    cls_lines = (ds_dir / "cls_indices_train.tsv").read_text().strip().split("\n")
    classes = [l.split("\t")[2] for l in cls_lines]
    n_cls = len(classes)

    pred_files = sorted(pred_dir.glob("*.csv"))
    if not pred_files:
        print(f"ERROR: no prediction csv in {pred_dir}")
        sys.exit(1)

    # 集計器
    conf = np.zeros((n_cls, n_cls), dtype=int)     # [gt, pred] 空間一致した組（対角=ok+dir_err）
    dir_err = np.zeros(n_cls, dtype=int)           # 同クラス一致だが>20°
    miss = np.zeros(n_cls, dtype=int)
    miss_edge = np.zeros(n_cls, dtype=int)
    n_gt = np.zeros(n_cls, dtype=int)
    fa = np.zeros(n_cls, dtype=int)                # 予測クラス別の誤通知
    fa_near_car = np.zeros(n_cls, dtype=int)       # うち車の見かけ方向±20°以内
    le_lists = [[] for _ in range(n_cls)]          # 同クラス一致フレームの角度誤差
    # 仮説3: Siren の取り違えを siren_type 別に
    siren_sub_by_type = {}                         # {(siren_type, pred_cls_name): count}
    # 仮説4: 見逃し/検出フレームのクリーン音エネルギー[dB]
    energy_miss = [[] for _ in range(n_cls)]
    energy_det = [[] for _ in range(n_cls)]
    per_clip = []

    for pf in pred_files:
        name = pf.stem
        meta_path = ds_dir / "metadata" / f"{name}.csv"
        scene_path = ds_dir / "work" / name / "scene.json"
        if not meta_path.exists() or not scene_path.exists():
            print(f"  skip {name}: metadata/scene.json not found")
            continue
        preds = load_pred_csv(pf)
        gts = read_dcase_csv(meta_path)  # {frame: [(cls, az, el), ...]}
        sj = json.loads(scene_path.read_text(encoding="utf-8"))
        ev = sj.get("event", {})
        itf = sj.get("interferer")
        gt_cls_clip = int(sj.get("class_idx", 0))
        siren_type = sj.get("siren_type", "") if gt_cls_clip == 0 else ""

        c = sound_speed(sj["scene_config"]["temperature_c"])
        mic = np.array(sj["scene_config"]["mic_pos"])
        wp_i = None
        if itf:
            wp_i = np.array(
                [[0.0, itf["x_start"], itf["offset_m"], itf["src_z_m"]],
                 [sj["scene_config"]["clip_len_sec"], itf["x_end"],
                  itf["offset_m"], itf["src_z_m"]]])

        n_frames = int(round(sj["scene_config"]["clip_len_sec"] / LABEL_RES))
        e_db = frame_energy_db(ds_dir / "work" / name / "foa_clean_24k.flac",
                               n_frames)

        clip_stat = {"name": name, "gt_class": classes[gt_cls_clip],
                     "siren_type": siren_type,
                     "snr_db": round(sj["noise"]["snr_db"], 1),
                     "sir_db": round(itf["sir_db"], 1) if itf else "",
                     "t_on": ev.get("t_on"), "t_off": ev.get("t_off"),
                     "ok": 0, "dir_err": 0, "sub": 0, "miss": 0, "fa": 0}

        for k in range(n_frames):
            g_list = list(gts.get(k, []))
            p_list = list(preds.get(k, []))
            t = (k + 0.5) * LABEL_RES

            # ① 同クラス優先マッチ
            for g in list(g_list):
                gc = int(g[0])
                same = [p for p in p_list if p[0] == gc]
                if same:
                    p_best = min(same, key=lambda p: ang_dist_deg(
                        p[1], p[2], g[1], g[2]))
                    d = ang_dist_deg(p_best[1], p_best[2], g[1], g[2])
                    n_gt[gc] += 1
                    conf[gc, gc] += 1
                    le_lists[gc].append(d)
                    if d > MATCH_DEG:
                        dir_err[gc] += 1
                        clip_stat["dir_err"] += 1
                    else:
                        clip_stat["ok"] += 1
                    energy_det[gc].append(e_db[k])
                    g_list.remove(g)
                    p_list.remove(p_best)

            # ② 取り違え（別クラスの予測がGT方向20°以内）
            for g in list(g_list):
                gc = int(g[0])
                near = [(p, ang_dist_deg(p[1], p[2], g[1], g[2]))
                        for p in p_list]
                near = [pd for pd in near if pd[1] <= MATCH_DEG]
                if near:
                    p_best = min(near, key=lambda pd: pd[1])[0]
                    pc = int(p_best[0])
                    n_gt[gc] += 1
                    conf[gc, pc] += 1
                    clip_stat["sub"] += 1
                    energy_det[gc].append(e_db[k])  # 音としては拾えている
                    if gc == 0 and siren_type:
                        key = (siren_type, classes[pc])
                        siren_sub_by_type[key] = siren_sub_by_type.get(key, 0) + 1
                    g_list.remove(g)
                    p_list.remove(p_best)

            # ③ 見逃し
            for g in g_list:
                gc = int(g[0])
                n_gt[gc] += 1
                miss[gc] += 1
                clip_stat["miss"] += 1
                energy_miss[gc].append(e_db[k])
                if (ev and (t < ev["t_on"] + EDGE_SEC
                            or t > ev["t_off"] - EDGE_SEC)):
                    miss_edge[gc] += 1

            # ④ 誤通知（残った予測）
            for p in p_list:
                pc = int(p[0])
                fa[pc] += 1
                clip_stat["fa"] += 1
                if wp_i is not None:
                    az_car, el_car, _, _ = apparent_azel_deg(
                        np.array([t]), wp_i, mic, c)
                    if np.isfinite(az_car[0]) and ang_dist_deg(
                            p[1], p[2], az_car[0], el_car[0]) <= MATCH_DEG:
                        fa_near_car[pc] += 1

        per_clip.append(clip_stat)

    # ---- 集計・仮説検証 ----
    per_class = []
    for ci, cname in enumerate(classes):
        subs_out = {classes[cj]: int(conf[ci, cj])
                    for cj in range(n_cls) if cj != ci and conf[ci, cj] > 0}
        le = np.array(le_lists[ci]) if le_lists[ci] else np.array([np.nan])
        em = np.array(energy_miss[ci])
        ed = np.array(energy_det[ci])
        per_class.append({
            "class": cname,
            "n_gt_frames": int(n_gt[ci]),
            "ok": int(conf[ci, ci] - dir_err[ci]),
            "dir_err_gt20": int(dir_err[ci]),
            "substituted_as": subs_out,
            "miss": int(miss[ci]),
            "miss_at_edge": int(miss_edge[ci]),
            "miss_rate_pct": round(100 * miss[ci] / max(n_gt[ci], 1), 1),
            "fa_as_this_class": int(fa[ci]),
            "fa_near_car": int(fa_near_car[ci]),
            "le_same_class_median_deg": round(float(np.nanmedian(le)), 2),
            "missed_frame_energy_median_db": (round(float(np.median(em)), 1)
                                              if len(em) else None),
            "detected_frame_energy_median_db": (round(float(np.median(ed)), 1)
                                                if len(ed) else None),
        })

    summary = {
        "classes": classes,
        "confusion_matrix_gt_x_pred_spatial_match": conf.tolist(),
        "per_class": per_class,
        "siren_substitution_by_type": {f"{k[0]} -> {k[1]}": v
                                       for k, v in sorted(
                                           siren_sub_by_type.items())},
        "totals": {
            "n_gt": int(n_gt.sum()),
            "ok": int(conf.trace() - dir_err.sum()),
            "dir_err": int(dir_err.sum()),
            "substitution": int(conf.sum() - conf.trace()),
            "miss": int(miss.sum()),
            "fa": int(fa.sum()),
        },
        "per_clip": per_clip,
    }
    (out_dir / "summary_mc.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    make_figures(classes, conf, dir_err, miss, fa, fa_near_car,
                 energy_miss, energy_det, out_dir)
    return summary


def make_figures(classes, conf, dir_err, miss, fa, fa_near_car,
                 energy_miss, energy_det, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cls = len(classes)

    # 図1: 混同行列（GT×pred の空間一致）＋miss列
    mat = np.concatenate([conf, miss[:, None]], axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n_cls + 1))
    ax.set_xticklabels(classes + ["MISS"], rotation=30, ha="right")
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted class (spatially matched <=20 deg)")
    ax.set_ylabel("ground-truth class")
    ax.set_title("confusion matrix (frames) - off-diagonal = substitution")
    for i in range(n_cls):
        for j in range(n_cls + 1):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > mat.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_mc.png", dpi=150)
    plt.close()

    # 図2: クラス別の誤り内訳（GTフレームに対する割合）
    n_gt = conf.sum(axis=1) + miss
    ok = np.array([conf[i, i] for i in range(n_cls)]) - dir_err
    sub = conf.sum(axis=1) - np.array([conf[i, i] for i in range(n_cls)])
    with np.errstate(divide="ignore", invalid="ignore"):
        def rate(x):
            return 100 * x / np.maximum(n_gt, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(n_cls)
    width = 0.2
    # 結果の深刻度に沿ったステータス色（ok=良 / dir_err=注意 / substitution=重 / miss=致命）
    axes[0].bar(xs - 1.5 * width, rate(miss), width, label="miss",
                color="#d03b3b")
    axes[0].bar(xs - 0.5 * width, rate(sub), width, label="substitution",
                color="#ec835a")
    axes[0].bar(xs + 0.5 * width, rate(dir_err), width, label="dir err >20deg",
                color="#fab219")
    axes[0].bar(xs + 1.5 * width, rate(ok), width, label="ok",
                color="#0ca30c", alpha=0.45)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylabel("% of GT frames")
    axes[0].set_title("per-class outcome breakdown")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(xs - 0.2, fa, 0.4, label="false alarms (as this class)",
                color="#2a78d6")
    axes[1].bar(xs + 0.2, fa_near_car, 0.4, label="of which near car direction",
                color="#1baf7a")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylabel("frames")
    axes[1].set_title("false alarms by predicted class")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_mc.png", dpi=150)
    plt.close()

    # 図3: 仮説4 — 見逃しフレーム vs 検出フレーム のクリーン音エネルギー分布
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 4), sharey=True)
    for ci, ax in enumerate(np.atleast_1d(axes)):
        data, labels = [], []
        if len(energy_det[ci]):
            data.append(energy_det[ci])
            labels.append(f"detected (n={len(energy_det[ci])})")
        if len(energy_miss[ci]):
            data.append(energy_miss[ci])
            labels.append(f"missed (n={len(energy_miss[ci])})")
        if data:
            ax.boxplot(data, tick_labels=labels)
        ax.set_title(classes[ci])
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("clean W energy per frame (dB rel clip max)")
    plt.suptitle("hypothesis 4: are misses concentrated in low-energy frames?")
    plt.tight_layout()
    plt.savefig(out_dir / "miss_energy_mc.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "out" / "predictions_v5_run2"))
    ap.add_argument("--ds", default=str(ROOT / "out" / "dataset_outdoor_siren_v5"))
    ap.add_argument("--out", default=str(ROOT / "out" / "figures_v5_analysis"))
    args = ap.parse_args()

    summary = analyze(Path(args.pred), Path(args.ds), Path(args.out))

    t = summary["totals"]
    print("\n=== totals (frames) ===")
    print(f"  n_gt={t['n_gt']}  ok={t['ok']}  dir_err={t['dir_err']}  "
          f"substitution={t['substitution']}  miss={t['miss']}  fa={t['fa']}")
    print("\n=== per class ===")
    for pc in summary["per_class"]:
        print(f"  {pc['class']:<11} gt={pc['n_gt_frames']:4d}  ok={pc['ok']:4d}  "
              f"dir>{20}={pc['dir_err_gt20']:3d}  miss={pc['miss']:3d} "
              f"({pc['miss_rate_pct']}%)  sub_as={pc['substituted_as']}  "
              f"fa={pc['fa_as_this_class']}  LE={pc['le_same_class_median_deg']}deg")
    if summary["siren_substitution_by_type"]:
        print("\n=== siren substitutions by siren_type (hypothesis 3) ===")
        for k, v in summary["siren_substitution_by_type"].items():
            print(f"  {k}: {v}")
    print(f"\nwrote {args.out}/summary_mc.json, confusion_mc.png, "
          f"per_class_mc.png, miss_energy_mc.png")


if __name__ == "__main__":
    main()
