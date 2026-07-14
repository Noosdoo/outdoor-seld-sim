# -*- coding: utf-8 -*-
"""イベント単位の当事者指標（軸3: 見逃し率・誤通知率）v1。

フレーム単位のmiss/fa（step8）は当事者体験と乖離する（missの大半が区間端規約の産物、
faの「回数」感覚が無い）ため、発音イベント単位で再集計する。敵対的レビュー#6/#15対応。

定義（v1、ゼミで要合意）:
  - 検出        : GTクラスの予測フレームが発音区間[t_on-0.15, t_off+0.15]内に1つ以上ある
  - 見逃し(イベント): 上が無い（=その危険音は一度も通知されない）
  - 初検出遅延  : 最初の検出フレーム時刻 - t_on（負は0に切り上げ）
  - 方向正解    : 初検出フレームの予測方向がGT方向から20°以内
  - 誤通知イベント: GTと対応しない予測フレーム（区間外のGTクラス or 全区間の他クラス）を
                   同クラス・時間ギャップ0.3s以内で連結した塊。件数と時間率で報告

実行例:
  python scripts/step8d_event_metrics.py --pred out/predictions_v6_run1 \
      --ds out/dataset_outdoor_siren_v6 --tag v6_run1
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

TOL_SEC = 0.15       # 発音区間端の許容（ラベル分解能0.1sの半分+α）
FA_GAP_FRAMES = 3    # 誤通知イベントの連結ギャップ（0.3s）


def analyze(pred_dir: Path, ds_dir: Path, tag: str, out_dir: Path):
    cls_lines = (ds_dir / "cls_indices_train.tsv").read_text().strip().split("\n")
    classes = [l.split("\t")[2] for l in cls_lines]
    n_cls = len(classes)

    ev_total = np.zeros(n_cls, dtype=int)
    ev_miss = np.zeros(n_cls, dtype=int)
    ev_dir_ok = np.zeros(n_cls, dtype=int)   # 初検出時に方向20°以内
    latencies = [[] for _ in range(n_cls)]
    fa_events = []                            # (clip, cls, t_start, dur_s)
    total_audio_sec = 0.0

    for pf in sorted(pred_dir.glob("*.csv")):
        name = pf.stem
        preds = s8.load_pred_csv(pf)
        gts = read_dcase_csv(ds_dir / "metadata" / f"{name}.csv")
        sj = json.loads((ds_dir / "work" / name / "scene.json")
                        .read_text(encoding="utf-8"))
        ev = sj["event"]
        gc = int(sj["class_idx"])
        clip_len = sj["scene_config"]["clip_len_sec"]
        total_audio_sec += clip_len
        n_frames = int(round(clip_len / s8.LABEL_RES))

        # --- イベント検出・遅延・方向 ---
        det_frames = []
        for k in sorted(preds):
            t = (k + 0.5) * s8.LABEL_RES
            if ev["t_on"] - TOL_SEC <= t <= ev["t_off"] + TOL_SEC:
                if any(int(p[0]) == gc for p in preds[k]):
                    det_frames.append(k)
        ev_total[gc] += 1
        if not det_frames:
            ev_miss[gc] += 1
        else:
            k0 = det_frames[0]
            t0 = (k0 + 0.5) * s8.LABEL_RES
            latencies[gc].append(max(0.0, t0 - ev["t_on"]))
            # 初検出フレームの方向正否（GTフレームがあればそれと比較）
            g = (gts.get(k0) or [None])[0]
            p = next(p for p in preds[k0] if int(p[0]) == gc)
            if g is not None:
                d = s8.ang_dist_deg(p[1], p[2], g[1], g[2])
                ev_dir_ok[gc] += d <= s8.MATCH_DEG

        # --- 誤通知イベント（クラス別にフレームを集めて連結） ---
        fa_frames = {c: [] for c in range(n_cls)}
        for k, plist in preds.items():
            t = (k + 0.5) * s8.LABEL_RES
            in_win = ev["t_on"] - TOL_SEC <= t <= ev["t_off"] + TOL_SEC
            for p in plist:
                pc = int(p[0])
                if pc == gc and in_win:
                    continue                  # 正しい通知
                fa_frames[pc].append(k)
        for pc, ks in fa_frames.items():
            if not ks:
                continue
            ks = sorted(ks)
            start = prev = ks[0]
            for k in ks[1:] + [None]:
                if k is not None and k - prev <= FA_GAP_FRAMES:
                    prev = k
                    continue
                fa_events.append((name, classes[pc],
                                  round(start * s8.LABEL_RES, 1),
                                  round((prev - start + 1) * s8.LABEL_RES, 1)))
                if k is not None:
                    start = prev = k

    # ---- 集計・出力 ----
    minutes = total_audio_sec / 60.0
    summary = {"tag": tag, "n_clips": int(ev_total.sum()),
               "total_audio_min": round(minutes, 1), "per_class": [],
               "fa_events_total": len(fa_events),
               "fa_events_per_min": round(len(fa_events) / minutes, 2),
               "fa_event_median_dur_s": (float(np.median([d for *_, d in fa_events]))
                                          if fa_events else 0.0),
               "fa_events": [
                   {"clip": c, "pred_class": pc, "t": t, "dur_s": d}
                   for c, pc, t, d in fa_events]}
    print(f"=== event-level metrics [{tag}] "
          f"({int(ev_total.sum())} events, {minutes:.0f} min audio) ===")
    for ci, cname in enumerate(classes):
        lat = np.array(latencies[ci]) if latencies[ci] else np.array([np.nan])
        n_det = ev_total[ci] - ev_miss[ci]
        row = {
            "class": cname, "n_events": int(ev_total[ci]),
            "event_miss": int(ev_miss[ci]),
            "event_miss_rate_pct": round(100 * ev_miss[ci] / max(ev_total[ci], 1), 1),
            "latency_median_s": round(float(np.nanmedian(lat)), 2),
            "latency_p90_s": round(float(np.nanpercentile(lat, 90)), 2),
            "first_detection_dir_ok_pct": round(100 * ev_dir_ok[ci] / max(n_det, 1), 1),
        }
        summary["per_class"].append(row)
        print(f"  {cname:<11} events={row['n_events']:3d}  "
              f"miss={row['event_miss']} ({row['event_miss_rate_pct']}%)  "
              f"latency med={row['latency_median_s']}s p90={row['latency_p90_s']}s  "
              f"dir_ok@first={row['first_detection_dir_ok_pct']}%")
    print(f"  FA events: {len(fa_events)} total = "
          f"{summary['fa_events_per_min']}/min "
          f"(median dur {summary['fa_event_median_dur_s']}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"event_metrics_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out" / "event_metrics"))
    args = ap.parse_args()
    analyze(Path(args.pred), Path(args.ds), args.tag, Path(args.out))


if __name__ == "__main__":
    main()
