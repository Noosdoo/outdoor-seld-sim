"""切り分け 3: 合成の自転車（val の走行音だけの本）に、家の 10/7 の背景の音（室外機を含む）を足すと、今のモデルが取れなくなるかを見る準備（2026-10-25 Opus・本人 20:52）。

使うもの（評価データ・学内の実録には触らない）:
  - val の自転車の走行音だけの本（plan の scenario = v19bicycle・motion = static）から、近い（critical）20 本・中くらい（caution）10 本を clip_id の順に取ったもの
    （結果を見て選ばない・一覧は --pick の csv）。音はサーバの dataset_outdoor_siren_v2final_v29/foa/ から手元へ写したもの（--val-dir）。
  - 背景: raw_bicycle_measure/261007_007.WAV の、通過と通過のちょうど真ん中の 10 s（自転車は 10 m 以上離れている）を、デモと同じ校正（−42.335 dB）で 24 kHz に。
作るもの（<out>/clips/）: eval_field_clip91NNN = 合成のまま・eval_field_clip92NNN = 合成＋家の背景（同じ番号どうしが同じ val の本）。
  <out>/map.csv（番号 → val の本・背景の区間）・<out>/height_table.csv（plan の mic_z）。レベルの表も出す（自転車がいる間と背景の W の大きさ）。

使い方: python scripts/demo_bike_check_valmix.py --pick <val_pick.csv> --val-dir <手元に写した val の flac の置き場> [--out out/_demo_bike_check_valmix]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
from step19b_realsmoke_cut import cut_audio  # noqa: E402

MEAS_WAV = E2E / "raw_bicycle_measure" / "261007_007.WAV"
PASSES = E2E / "raw_bicycle_measure" / "analysis" / "passes.csv"
VAL_GT = E2E / "out" / "_v29_val_gt" / "metadata_dist"
GAIN_DB = -42.335
SYN_HEADROOM_DB = -6.0   # 合成は全部の音を −6 dB（scripts/step11_v12_render.py:45-53）＝実効 149 dB の約束。家の背景（143 dB の約束）を同じ物理の大きさで足すにはこれを足す（Astra HB25-02）


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x) ** 2) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", default=str(E2E / "out" / "_demo_bike_check_valmix"))
    ap.add_argument("--bg-extra-db", type=float, default=SYN_HEADROOM_DB, help="家の背景に足す dB（既定 −6＝合成の約束にそろえる・10/25 の初回は 0 で 6 dB 大きかった）")
    ap.add_argument("--drop-bg", default="9", help="使わない背景の区間の番号（カンマ区切り・既定 9＝動画の読みと 0.313 s 重なる・HB25-06）")
    ap.add_argument("--mix-base", type=int, default=92000, help="混ぜた本の番号の頭（初回は 92000・直した版は別の番号にする）")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "bg").mkdir(parents=True, exist_ok=True)
    t = sorted(float(r["t_cpa_s"]) for r in read_csv(PASSES))
    mids = [(t[i] + t[i + 1]) / 2 for i in range(len(t) - 1)]
    bgs = []
    drop = {int(x) for x in a.drop_bg.split(",") if x.strip()}
    for j, m in enumerate(mids):
        if j + 1 in drop:
            continue
        p = out / "bg" / f"bg{j + 1:02d}.flac"
        cut_audio(MEAS_WAV, m - 5.0, 10.0, p, gain_db=GAIN_DB + a.bg_extra_db)
        y, fs = sf.read(str(p), always_2d=True)
        bgs.append((p.name, f"{m - 5.0:.2f}-{m + 5.0:.2f}", y))
    print("背景の区間:", [(n, s, f"{db(y[:, 0]):.1f} dBFS") for n, s, y in bgs])
    pick = read_csv(a.pick)
    maprows = []
    hrows = []
    for i, r in enumerate(pick, start=1):
        vid = r["val_id"]
        y, fs = sf.read(str(Path(a.val_dir) / f"{vid}.flac"), always_2d=True)
        assert fs == 24000 and y.shape == (240000, 4), (vid, fs, y.shape)
        bname, bseg, b = bgs[(i - 1) % len(bgs)]
        n91, n92 = f"eval_field_clip{91000 + i}", f"eval_field_clip{a.mix_base + i}"
        sf.write(str(out / "clips" / f"{n91}.flac"), y, fs, subtype="PCM_24")
        sf.write(str(out / "clips" / f"{n92}.flac"), y + b, fs, subtype="PCM_24")
        gt = [list(map(float, l.split(","))) for l in open(VAL_GT / f"{vid}.csv", encoding="utf-8") if l.strip()]
        fr = sorted({int(g[0]) for g in gt if int(g[1]) == 8})
        seg = np.concatenate([y[k * 2400:(k + 1) * 2400, 0] for k in fr]) if fr else None   # 混合音の W（自転車だけの大きさではない・HB25-02）
        dmin = min((g[5] for g in gt if int(g[1]) == 8), default=float("nan"))
        maprows.append({"clip_orig": n91, "clip_mix": n92, "val_id": vid, "tier": r["tier"], "mic_z": r["mic_z"], "bg": bname, "bg_seg_s": bseg,
                        "mixW_labeled_dBFS": (f"{db(seg):.1f}" if seg is not None else ""), "bg_extra_db": a.bg_extra_db, "bg_W_dBFS": f"{db(b[:, 0]):.1f}", "gt_frames_bike": len(fr), "gt_min_dist_m": f"{dmin:.2f}"})
        hrows += [(n91, r["mic_z"]), (n92, r["mic_z"])]
    with open(out / "map.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(maprows[0].keys()))
        w.writeheader()
        w.writerows(maprows)
    with open(out / "height_table.csv", "w", encoding="utf-8", newline="") as f:
        f.write("clip_id,mic_z\n")
        for c, z in hrows:
            f.write(f"{c},{z}\n")
    bw = [float(m["mixW_labeled_dBFS"]) for m in maprows if m["mixW_labeled_dBFS"]]
    print(f"{len(maprows)} 本 × 2・ラベルのある時間の混合音の W（自転車だけの大きさではない）: 中央値 {np.median(bw):.1f} dBFS（{len(bw)} 本）・背景に足した dB {a.bg_extra_db:+g}・外した背景 {sorted(drop)} → {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
