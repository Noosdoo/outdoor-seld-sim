# -*- coding: utf-8 -*-
"""動画（または走行の台本）から読んだ相手の位置（x, y）を、学習データと同じ正解の形（frame,cls,track,az,el,dist）に変換する。
2026-09-16・学内パイロット計画 §3。Astra A9（同日）で学習データと同じ時刻の規約に合わせた:
  - フレーム k の代表時刻は中心 (k+0.5)×0.1 s（outdoor_seld.labels.frame_label_rows と同じ）
  - 方位・仰角・距離は**音がマイクに届いた時刻**基準（放射時刻を解く = 伝搬遅延を含む）。同じ関数 apparent_azel_deg を呼ぶ
  - 距離 = 放射位置までの距離 × cos(仰角) = 水平距離（step19_dist_labels_v16 と同じ）。角度は np.rint で整数化・180 は −180

入力 CSV（動画を止めて読んだ表、または台本の点）: 列 = take,frame,cls,track,x,y[,z][,t]
  take   テイクの名前（出力ファイル名になる）
  frame  0.1 秒刻みの番号。**その時刻に相手がその位置にいた**（拍手の同期で音の 0 秒に合わせた後の値）
  t      任意。秒の実数（例 3.417）。あれば frame より優先して**丸めずに**使う（2026-09-26・step19h の出力。0.1 s に丸めると 4 m/s で最大 ±0.2 m ずれるため）。
         t だけで frame が無い表も受け付ける
  cls    種類の名前（car / kick / bike / siren / horn / backup_beep / bike_bell / crossing）か番号
  track  同じ相手なら同じ番号
  x, y   マイクを原点にした位置 [m]。x = 前、y = 左（az = atan2(y, x)・反時計回り）
  z      音源の高さ [m]（省略時は --src-z）
飛び飛びの点は同じ take・cls・track の中で直線補間して軌道（waypoints）にし、その軌道から学習データと同じ関数で各フレームの正解を作る。

使い方:
  python scripts/step19f_video_to_gt.py <入力csv> <出力dir> --mic-z 1.60 [--src-z 0.5] [--sync-offset 0.0] [--check]
  --sync-offset 秒 = 動画と音のずれの見積もり（拍手の読みの誤差）。正なら動画の時刻を遅らせる。±0.05 s で作って差を見ると、同期誤差の影響が分かる。
  --check = 自己検査（学習側の frame_label_rows と一致するか・重複と不正値の検出）。
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from outdoor_seld.geometry import apparent_azel_deg, SOUND_SPEED_20C  # noqa: E402
from outdoor_seld.labels import frame_label_rows  # noqa: E402

CLASS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3, "car": 4, "car_drive": 4, "crossing": 5, "kick": 6, "kickboard": 6, "bike": 7, "motorcycle": 7, "bicycle": 8}   # bicycle = 2026-10-08 変更票 #48
LABEL_RES = 0.1
CLIP = 10.0
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cls_of(s: str) -> int:
    s = s.strip()
    if s.lstrip("-").isdigit():
        return int(s)
    if s not in CLASS_IDX:
        raise ValueError(f"unknown class name: {s!r}（使える名前: {sorted(CLASS_IDX)}）")
    return CLASS_IDX[s]


def rows_from_waypoints(wp: np.ndarray, mic_z: float, cls: int, trk: int, clip_len: float, c: float = SOUND_SPEED_20C):
    """学習データと同じ関数で 6 列の行を作る。wp = (M,4) [t, x, y, z]"""
    mic = np.array([0.0, 0.0, mic_z], dtype=float)
    rows5, dbg = frame_label_rows(wp, mic, clip_len_sec=clip_len, class_idx=cls, track_idx=trk,
                                  source_active_from=float(wp[0, 0]), source_active_until=float(wp[-1, 0]), c=c)
    out = []
    for k, c_, t_, az_i, el_i in rows5:
        d3 = float(dbg["dist"][k]); el = float(dbg["el"][k])
        dh = d3 * math.cos(math.radians(el))
        out.append((k, f"{k},{c_},{t_},{az_i},{el_i},{dh:.2f}"))
    return out


def convert(rows, mic_z: float, src_z: float, sync_offset: float = 0.0, clip_len: float = CLIP):
    """rows: dict 行のリスト → {take: [出力行]}。重複（同じ take・cls・track・frame）と不正値は止める"""
    by = defaultdict(lambda: defaultdict(dict))   # take -> (cls, trk) -> frame -> (x, y, z)
    tsec = defaultdict(lambda: defaultdict(dict))  # take -> (cls, trk) -> frame -> 秒の実数（t 列があるとき。丸めない）
    for i, r in enumerate(rows, 2):
        if (r.get("hidden") or "").strip():
            continue                                                   # step19h の隠れた印（x 空）は位置の読みではない（B4・2026-10-20）
        take = r["take"].strip()
        try:
            c = cls_of(r["cls"]); t = int(r.get("track", 0) or 0)
            has_t = r.get("t") not in (None, "", "nan")
            if has_t:
                ts = float(r["t"]); k = int(round(ts / LABEL_RES))
            else:
                k = int(r["frame"]); ts = None
            x = float(r["x"]); y = float(r["y"])
            z = float(r["z"]) if r.get("z") not in (None, "", "nan") else src_z
        except (KeyError, ValueError) as e:
            raise ValueError(f"{i} 行目: 値が読めない {dict(r)} ({e})")
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)) or k < 0 or k >= int(clip_len / LABEL_RES) * 10 or (ts is not None and not math.isfinite(ts)):
            raise ValueError(f"{i} 行目: 不正な値 frame={k} t={ts} x={x} y={y} z={z}")
        if k in by[take][(c, t)]:
            raise ValueError(f"{i} 行目: 重複（take={take} cls={c} track={t} frame={k}）")
        by[take][(c, t)][k] = (x, y, z)
        if ts is not None:
            tsec[take][(c, t)][k] = ts
    out = {}
    for take, tracks in by.items():
        lines = []
        for (c, t), fr in tracks.items():
            ks = sorted(fr)
            if len(ks) < 2:
                raise ValueError(f"take={take} cls={c} track={t}: 点が 1 つしかない（軌道を作れない。出発と終点の 2 点以上）")
            tk = tsec[take][(c, t)]
            wp = np.array([[tk.get(k, k * LABEL_RES) + sync_offset, *fr[k]] for k in ks], dtype=float)
            lines += rows_from_waypoints(wp, mic_z, c, t, clip_len)
        out[take] = [s for _, s in sorted(lines)]
    return out


def self_check() -> int:
    """(1) 直線を横切る車の台本（2 点）から作った正解が、同じ軌道を学習側の frame_label_rows に渡した結果と**全行一致**する
    (2) 最接近付近の方位が幾何（フレーム中心時刻・伝搬遅延込み）と合う (3) 重複・不正値を止める"""
    v = 20 / 3.6
    rows = [dict(take="check", frame="0", cls="car", track="0", x=f"{20.0:.3f}", y="3.0", z=""),
            dict(take="check", frame="90", cls="car", track="0", x=f"{20.0 - v * 9.0:.3f}", y="3.0", z="")]
    out = convert(rows, mic_z=1.6, src_z=0.5)["check"]
    wp = np.array([[0.0, 20.0, 3.0, 0.5], [9.0, 20.0 - v * 9.0, 3.0, 0.5]])
    ref = rows_from_waypoints(wp, 1.6, 4, 0, CLIP)
    ok = [s for _, s in sorted(ref)] == out
    print("学習側の frame_label_rows との全行一致:", "OK" if ok else "NG", f"（{len(out)} 行）")
    # 幾何: 受信時刻 t=(k+0.5)*0.1 に対して放射時刻 te を解いた位置の方位
    k = 36; t_r = (k + 0.5) * LABEL_RES
    az, el, te, dist = apparent_azel_deg(np.array([t_r]), wp, np.array([0.0, 0.0, 1.6]))
    line = [l for l in out if l.startswith(f"{k},")][0]
    print(f"frame {k}: 出力 {line} / 幾何 az={az[0]:.2f}° el={el[0]:.2f}° 水平 {dist[0] * math.cos(math.radians(el[0])):.2f} m（放射時刻 {te[0]:.3f} s・受信 {t_r:.2f} s）")
    ok2 = abs(int(np.rint(az[0])) - int(line.split(",")[3])) == 0
    # 重複・不正値
    try:
        convert(rows + [dict(take="check", frame="0", cls="car", track="0", x="1", y="1", z="")], 1.6, 0.5); print("重複検出: NG（止まらなかった）"); ok3 = False
    except ValueError:
        print("重複検出: OK"); ok3 = True
    try:
        convert([dict(take="c", frame="0", cls="car", track="0", x="nan", y="1", z=""), dict(take="c", frame="9", cls="car", track="0", x="1", y="1", z="")], 1.6, 0.5); print("不正値検出: NG"); ok4 = False
    except ValueError:
        print("不正値検出: OK"); ok4 = True
    # 同期ずれの感度: ±0.05 s で最接近付近の方位がどれだけ動くか
    d = {}
    for off in (-0.05, 0.0, 0.05):
        o = convert(rows, 1.6, 0.5, sync_offset=off)["check"]
        d[off] = int([l for l in o if l.startswith(f"{k},")][0].split(",")[3])
    print(f"同期ずれの感度（frame {k} の方位）: −0.05 s → {d[-0.05]}°, 0 → {d[0.0]}°, +0.05 s → {d[0.05]}°")
    good = ok and ok2 and ok3 and ok4
    print("自己検査:", "OK" if good else "NG")
    return 0 if good else 1


def main() -> int:
    argv = list(sys.argv[1:])
    if "--check" in argv:
        return self_check()

    def opt(key, default):
        if key in argv:
            i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
        return default
    mic_z = opt("--mic-z", None)
    src_z = float(opt("--src-z", "0.5"))
    sync = float(opt("--sync-offset", "0"))
    if mic_z is None or len(argv) < 2:
        print(__doc__); return 2
    mic_z = float(mic_z)
    src, out_dir = Path(argv[0]), Path(argv[1])
    with open(src, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    need = {"take", "cls", "x", "y"}
    cols = set(rows[0].keys()) if rows else set()
    if rows and (not need <= cols or not ({"frame", "t"} & cols)):
        raise ValueError(f"列が足りない: {need - cols} と frame か t")
    out = convert(rows, mic_z, src_z, sync)
    out_dir.mkdir(parents=True, exist_ok=True)
    for take, lines in out.items():
        (out_dir / f"{take}.csv").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        print(f"{take}: {len(lines)} 行 → {out_dir / (take + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
