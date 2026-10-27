# -*- coding: utf-8 -*-
"""v9f = 案 F-lite「名札の照合に距離を足す」（2026-09-28 段 2d・監査 Astra R03 で警告音の距離出力が使える精度（相対誤差の中央値 14〜24%）と分かったので試す）。

v9c（案 C2 = 名札を 1 台ずつ追跡・門 45°・同一方位 30° の二重出力を 1 つに）に、距離の条件を足す:
  照合: 音を名札に結ぶ条件を「方位差 ≤ 45°」かつ「距離の比が R 以内（|ln(d_det / d_tag)| ≤ ln R）」にする。名札の距離は半分ずつ寄せる（方位と同じ EMA）。
  まとめ: 同じフレームの二重出力を 1 つにまとめる条件も「30° 以内」かつ「距離の比が R 以内」にする（同じ方向でも距離が違えば 2 台として残す）。
  距離が無い検出（nan）は距離の条件を素通り（方位だけで v9c と同じ）。
狙い = 限界 1（同じ方向の 2 台）。同じ方向 30° 以内でも距離が R 倍以上違えば別の名札になり、2 台目に震える。
危険 = 距離の揺れ（誤差 14〜24%）で同じ車が 2 枚の名札に割れて重複が増える。R を 1.5／2／3 で振って綱引きを見る。
使い方:
  python scripts/step12_notify_v9f_distance.py --selftest
  python scripts/step12_notify_v9f_distance.py --pred <csv> --meta <metadata_dir> --out <dir> --manifest <csv> [--tag <名前>]
    → <out>/distance_compare_<tag>.md / .json（hold・案 C2・案 F の変種を同じ枡で採点。分母は名簿に固定）
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step12_notify_v9c_multitrack as C  # noqa: E402
from outdoor_seld.clip_ids import normalize  # noqa: E402

H = C.H
v9 = C.v9


def load_pred_dist(path: Path):
    """{clip: {frame: [(cls, az, el, dist), ...]}} 距離つき（7 列 [clip,frame,cls,track,az,el,dist]・6 列 [clip,frame,cls,az,el,dist]）"""
    out = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6 or not p[1].isdigit():
                continue
            if len(p) >= 7:
                c, az, el, d = int(p[2]), float(p[4]), float(p[5]), (float(p[6]) if p[6] not in ("", "nan") else float("nan"))
            else:
                c, az, el, d = int(p[2]), float(p[3]), float(p[4]), (float(p[5]) if p[5] not in ("", "nan") else float("nan"))
            out[normalize(p[0])][int(p[1])].append((c, az, el, d))
    return dict(out)


def _dist_ok(d1: float, d2: float, ratio: float) -> bool:
    """距離の比が ratio 以内か。どちらかが無ければ素通り（方位だけで判断）"""
    if ratio <= 0 or not (math.isfinite(d1) and math.isfinite(d2)) or d1 <= 0 or d2 <= 0:
        return True
    return abs(math.log(d1 / d2)) <= math.log(ratio)


def dedup_dist(dets, deg: float, ratio: float):
    """同じフレームの二重出力を 1 つに: 方位 deg 以内かつ距離の比 ratio 以内なら同一。dets = [(az, d)]"""
    keep = []
    for az, d in dets:
        if all(C.cdiff(az, k_az) > deg or not _dist_ok(d, k_d, ratio) for k_az, k_d in keep):
            keep.append((az, d))
    return keep


def warn_fires_v9f(pred_clip, gate: float = 45.0, nframes: int = 100, confirm: int = v9.WARN_CONFIRM, expire: int = v9.REFRACTORY,
                   ema: float = C.EMA_ALPHA, dedup_deg: float = 30.0, ratio: float = 2.0, ratio_dedup: float | None = None):
    """[(frame, cls, az)]。pred_clip = {frame: [(cls, az, el, dist)]}。ratio ≤ 0 なら距離を使わない（= v9c 案 C2 と同じ動き）。"""
    if ratio_dedup is None:
        ratio_dedup = ratio
    fires = []
    tracks = defaultdict(list)     # cls -> [dict(az, d, last, run, fired)]
    for k in range(nframes):
        dets = defaultdict(list)
        for c, az, el, d in pred_clip.get(k, []):
            if c in C.WARN:
                dets[c].append((az, d))
        for c in list(dets):
            dets[c] = dedup_dist(dets[c], dedup_deg, ratio_dedup)
        for c in C.WARN:
            alive = [t for t in tracks[c] if k - t["last"] < expire]
            tracks[c] = alive
            ds = dets.get(c, [])
            pairs = sorted(((C.cdiff(az, t["az"]), i, j) for i, (az, d) in enumerate(ds) for j, t in enumerate(alive)), key=lambda x: x[0])
            used_d, used_t = set(), set()
            for dang, i, j in pairs:
                t = alive[j]
                if dang > gate or i in used_d or j in used_t or not _dist_ok(ds[i][1], t["d"], ratio):
                    continue
                used_d.add(i); used_t.add(j)
                t["az"] = C.circ_mean(t["az"], ds[i][0], ema)
                if math.isfinite(ds[i][1]) and ds[i][1] > 0:
                    t["d"] = ds[i][1] if not (math.isfinite(t["d"]) and t["d"] > 0) else math.exp((1 - ema) * math.log(t["d"]) + ema * math.log(ds[i][1]))
                t["run"] = t["run"] + 1 if t["last"] == k - 1 else 1
                t["last"] = k
            for i, (az, d) in enumerate(ds):
                if i not in used_d:
                    alive.append({"az": az, "d": d, "last": k, "run": 1, "fired": False})
            for t in alive:
                if not t["fired"] and t["run"] >= confirm and t["last"] == k:
                    fires.append((k, c, t["az"]))
                    t["fired"] = True
    return fires


VARIANTS = {
    "案 C2（基準・距離を使わない）": dict(ratio=0.0),
    "案 F 距離の比 3 以内で同一（照合とまとめ）": dict(ratio=3.0),
    "案 F 距離の比 2 以内で同一（照合とまとめ）": dict(ratio=2.0),
    "案 F 距離の比 1.5 以内で同一（照合とまとめ）": dict(ratio=1.5),
    "案 F' まとめだけ距離の比 2（照合は方位だけ）": dict(ratio=0.0, ratio_dedup=2.0),
    "案 F'' 照合だけ距離の比 2（まとめは方位だけ）": dict(ratio=2.0, ratio_dedup=0.0),
}


def selftest() -> int:
    ok = []
    # 1) 同じ方向 0°・距離 5 m と 20 m の 2 台（同時・毎フレーム）: C2 は 1 回、案 F（比 2）は 2 回
    two = {k: [(0, 0.0, 0.0, 5.0), (0, 10.0, 0.0, 20.0)] for k in range(100)}
    ok.append(("同じ方向・距離 5 m と 20 m: C2 は 1 回", len(warn_fires_v9f(two, ratio=0.0)) == 1))
    ok.append(("同じ方向・距離 5 m と 20 m: 案 F（比 2）は 2 回", len(warn_fires_v9f(two, ratio=2.0)) == 2))
    # 2) 1 台が近づく（距離 30 → 3 m を 10 秒で・方向 0°）: 距離が連続して変わるので比 2 でも 1 枚の名札
    approach = {k: [(0, 0.0, 0.0, 30.0 - 0.27 * k)] for k in range(100)}
    ok.append(("近づく 1 台（30 → 3 m）: 案 F（比 2）でも 1 回", len(warn_fires_v9f(approach, ratio=2.0)) == 1))
    # 3) 距離の揺れ（±25%）のある 1 台: 比 2 なら 1 回、比 1.2 なら割れて 2 回以上
    import numpy as np
    rng = np.random.default_rng(1)
    jit = {k: [(0, 0.0, 0.0, float(10.0 * rng.uniform(0.75, 1.25)))] for k in range(100)}
    ok.append(("距離が ±25% 揺れる 1 台: 比 2 は 1 回", len(warn_fires_v9f(jit, ratio=2.0)) == 1))
    ok.append(("距離が ±25% 揺れる 1 台: 比 1.2 は割れて 2 回以上", len(warn_fires_v9f(jit, ratio=1.2)) >= 2))
    # 4) 距離が無い検出（nan）: ratio を付けても v9c と同じ
    nan = {k: [(0, 1.0 * k, 0.0, float("nan"))] for k in range(100)}
    pc = {k: [(c, az, el) for c, az, el, d in v] for k, v in nan.items()}
    ok.append(("距離なし: 案 F = v9c", warn_fires_v9f(nan, ratio=2.0) == C.warn_fires_v9c(pc, 45.0, dedup_deg=30.0)))
    # 5) ratio=0 は v9c と同一（乱数の場面 20 本・距離つき）
    same = 0
    for s in range(20):
        r = np.random.default_rng(s)
        clip = {k: [(0, float(r.uniform(-180, 180)), 0.0, float(r.uniform(3, 40)))] for k in range(100) if r.random() < 0.8}
        pc = {k: [(c, az, el) for c, az, el, d in v] for k, v in clip.items()}
        same += int(warn_fires_v9f(clip, ratio=0.0) == C.warn_fires_v9c(pc, 45.0, dedup_deg=30.0))
    ok.append(("比 0（距離を使わない）= v9c と同一（20 場面）", same == 20))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest()

    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    pred_p, meta, out = Path(opt("--pred")), Path(opt("--meta")), Path(opt("--out"))
    manifest = opt("--manifest")
    tag = opt("--tag", "main")
    out.mkdir(parents=True, exist_ok=True)
    pred_d = load_pred_dist(pred_p)
    pred, events = C.load_fixed(pred_p, meta, Path(manifest) if manifest else None)     # 分母と方位だけの予測（hold 用）
    pred_d = {c: pred_d.get(c, {}) for c in events}
    collapsed = {clip: {k: [(c, az, el) for c, az, el in dets] for k, dets in pred[clip].items()} for clip in events}
    results = {"hold（現行・前回検出から 5 s・45°）": C.score({c: H.warn_fires(collapsed[c], True) for c in events}, events)}
    for name, kw in VARIANTS.items():
        results[name] = C.score({c: warn_fires_v9f(pred_d[c], **kw) for c in events}, events)
    L = [f"# 警告音の規則の比較（段 2d・名札に距離を足す = 案 F・2026-09-28・{tag}）", "",
         f"- 予測 = `{pred_p.as_posix()}`。正解 = `{meta.as_posix()}`。クリップ {len(events):,} 本・出来事 {sum(len(v) for v in events.values()):,} 件"
         + (f"（分母 = 名簿 `{Path(manifest).as_posix()}` に固定）" if manifest else "") + "。数字は**開発用の数値**（確定評価のデータは見ていない）。",
         "- 案 F = 音を名札に結ぶ条件と、同じフレームの二重出力をまとめる条件に「距離の比が R 以内」を足す。R が小さいほど同じ方向の 2 台を分けやすいが、距離の揺れで同じ車が割れやすい。",
         "- 見る指標: 到達・重複した出来事と余分な回数・出来事の外の発火・同クラス 2 台目の到達・方向つき到達（45° 以内）。", ""]
    summary = {}
    for name, S in results.items():
        head, lines, tot = C.fmt(S)
        L += [f"## {name}", "", f"- {head}"] + lines + [""]
        summary[name] = dict(tot)
    (out / f"distance_compare_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"distance_compare_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
