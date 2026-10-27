# -*- coding: utf-8 -*-
"""Step 19e: 校正の点検 — 指パッチン（手拍子）の録音から方位を読み、前・右・後・左の並びと正面ズレを確かめる（2026-09-06・監査 R09）。

入力: H3-VR の AmbiX 録音（4ch WAV, W,Y,Z,X）か step19 の変換後 flac。どちらも ch 順は W,Y,Z,X。
やること:
  1. W の短時間エネルギーから「短い衝撃音」を時間順に拾う（指パッチン・手拍子）
  2. 各衝撃音の方位角を FOA の強度（W·X, W·Y）から出す。方位角は左が＋、右が−（採点器と同じ規約）
  3. --expect の並び（既定 前,右,後,左 = 0,−90,180,+90）と比べ、順序・符号の取り違えを言い当てる
  4. 正面の衝撃音の方位 = 正面ズレ_deg。step19 に渡す --yaw の値（= −正面ズレ）も出す

使い方:
  python scripts/step19e_check_azimuth.py --in raw/ZOOM0002.wav                    # 初日: 4 方位
  python scripts/step19e_check_azimuth.py --in raw/ZOOM0002.wav --expect 前          # 2 日目以降: 前 1 打
  --tol 25（許容の度）, --min-gap 0.5（衝撃音の最小間隔 秒）, --json <出力先>, --pick strongest|first（多いときの採り方・既定 strongest）
終了コード: 0 = 合格、1 = 不合格（順序・符号の取り違え、正面ズレ >10°、衝撃音の数が合わない）
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

EXPECT_DEG = {"前": 0.0, "右": -90.0, "後": 180.0, "左": 90.0}


def wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def load_foa(path: Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] != 4:
        raise SystemExit(f"{path.name}: 4ch(AmbiX W,Y,Z,X) ではありません shape={x.shape}")
    return x, sr


def find_impulses(w: np.ndarray, sr: int, min_gap: float, frame_ms: float = 2.0, ratio: float = 20.0):
    """W の 2 ms 枠のエネルギーが中央値の ratio 倍を超える立ち上がりを、時間順に返す（サンプル位置）。"""
    n = max(1, int(sr * frame_ms / 1000))
    m = len(w) // n
    e = (w[: m * n].reshape(m, n) ** 2).mean(axis=1)
    thr = max(np.median(e) * ratio, 1e-9)
    above = e > thr
    onsets, last = [], -1e9
    for i, a in enumerate(above):
        if a and (i * n - last) > min_gap * sr:
            onsets.append(i * n); last = i * n
    return onsets


def azimuth_of(x: np.ndarray, sr: int, onset: int, win_ms: float = 20.0):
    """衝撃音の直後 win_ms だけの強度で方位角・仰角（度）を出す。ch 順 W,Y,Z,X。左が＋。"""
    seg = x[onset: onset + int(sr * win_ms / 1000)]
    w, y, z, xx = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    ix, iy, iz = float((w * xx).sum()), float((w * y).sum()), float((w * z).sum())
    az = math.degrees(math.atan2(iy, ix))
    el = math.degrees(math.atan2(iz, math.hypot(ix, iy)))
    return az, el


def diagnose(expect: list[str], got: list[float], tol: float) -> tuple[bool, str]:
    """並びの取り違えを言い当てる。"""
    exp = [EXPECT_DEG[e] for e in expect]
    def ok(fn):
        return all(abs(wrap(fn(g) - e)) <= tol for g, e in zip(got, exp))
    if ok(lambda g: g):
        return True, "並びと符号は正しい"
    if ok(lambda g: -g):
        return False, "左右が反転している（Y の符号が逆。ch 順や AmbiX/FuMa の設定を疑う）"
    if ok(lambda g: wrap(180 - g)):
        return False, "前後が反転している（X の符号が逆）"
    if ok(lambda g: wrap(90 - g)):
        return False, "X と Y が入れ替わっている（ch 順を疑う）"
    if ok(lambda g: wrap(g - 90)) or ok(lambda g: wrap(g + 90)):
        return False, "全体が 90° 回っている（マイクの向き、または ch 順を疑う）"
    return False, "並びが期待と合わない（衝撃音の拾い違いか、環境が騒がしい）。録り直しを勧める"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--expect", default="前,右,後,左")
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--min-gap", type=float, default=0.5)
    ap.add_argument("--yaw-limit", type=float, default=10.0, help="正面ズレがこれを超えたら --yaw 補正が必要（不合格）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pick", choices=("strongest", "first"), default="strongest", help="衝撃音が期待より多いときの採り方（既定 strongest = ピークの大きい順）")
    a = ap.parse_args()
    expect = [e.strip() for e in a.expect.split(",") if e.strip()]
    for e in expect:
        if e not in EXPECT_DEG:
            raise SystemExit(f"--expect は 前/右/後/左 で: {e}")
    x, sr = load_foa(Path(a.inp))
    onsets = find_impulses(x[:, 0], sr, a.min_gap)
    # 衝撃音が期待より多いとき（REC の操作音・足音・雑音が混じる）は、ピークの大きい順に期待の数だけ採って時間順に並べる
    # （2026-10-07 家のデモ: 0.22 s の REC 操作音と 10 s 以降の小さな雑音で、先頭 4 つを採ると並びが崩れた）。--pick first で旧動作。
    if a.pick == "strongest" and len(onsets) > len(expect):
        w = x[:, 0]
        peaks = {o: float(np.abs(w[o:o + int(0.03 * sr)]).max()) for o in onsets}
        chosen = sorted(sorted(onsets, key=lambda o: -peaks[o])[: len(expect)])
    else:
        chosen = list(onsets[: len(expect)])
    rows = []
    for o in chosen:
        az, el = azimuth_of(x, sr, o)
        rows.append({"t": round(o / sr, 3), "az": round(az, 1), "el": round(el, 1)})
    print(f"衝撃音: {len(onsets)} 個（期待 {len(expect)} 個）" + ("・ピークの大きい順に期待の数だけ採用" if len(onsets) > len(expect) and a.pick == "strongest" else ""))
    if len(onsets) > len(expect):
        w = x[:, 0]
        print("  見つかった衝撃音（全部）: " + "  ".join(f"{o / sr:.2f}s({20 * np.log10(float(np.abs(w[o:o + int(0.03 * sr)]).max()) + 1e-9):.0f}dBFS)" for o in onsets))
    for e, r in zip(expect, rows):
        print(f"  {e}: t={r['t']:7.3f}s  方位 {r['az']:+7.1f}°（期待 {EXPECT_DEG[e]:+.0f}°） 仰角 {r['el']:+.1f}°")
    fail = []
    if len(onsets) < len(expect):
        fail.append(f"衝撃音が足りない（{len(onsets)} < {len(expect)}）。静かな場所で 1 打ずつ、間を 1 秒あけて録り直す")
    elif len(onsets) > len(expect) and a.pick != "strongest":
        fail.append(f"衝撃音の数が合わない（{len(onsets)} ≠ {len(expect)}）。--pick strongest でピークの大きい順に採るか、録り直す")
    ok, msg = (False, "衝撃音が足りない") if len(rows) < len(expect) else diagnose(expect, [r["az"] for r in rows], a.tol)
    print(("合格: " if ok else "不合格: ") + msg)
    if not ok:
        fail.append(msg)
    front = next((r["az"] for e, r in zip(expect, rows) if e == "前"), None)
    yaw = None
    if front is not None:
        yaw = round(-front, 1)
        print(f"正面ズレ_deg = {front:+.1f}（session.csv にそのまま書く）")
        if abs(front) > a.yaw_limit:
            print(f"  → 10° を超えたので step19 に --yaw {yaw:+.1f} を渡す（補正後は 0° になる）")
            fail.append(f"正面ズレ {front:+.1f}° > {a.yaw_limit:.0f}°（--yaw {yaw:+.1f} で補正）")
        else:
            print("  → 10° 以内。--yaw は不要")
    out = {"file": Path(a.inp).name, "expect": expect, "impulses": rows, "n_impulses": len(onsets),
           "ok": not fail, "message": msg, "front_offset_deg": front, "yaw_for_step19": yaw, "fail": fail}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
