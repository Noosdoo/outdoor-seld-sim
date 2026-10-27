# -*- coding: utf-8 -*-
"""v9d = 案 D「予測つきの名札」（2026-09-28 15:11 本人「お願いします」・警告音の規則 段 2b・開発データだけの比較）。

v9c（案 C2 = 名札を 1 台ずつ追跡・門 45°・同一方位 30° の二重出力を 1 つに）の**照合先**を、名札の「現在地」から「予想位置」に替える。
  予想位置 = 現在地 ＋ 角速度 × 抜けたフレーム数（角速度 = 直近の方位変化の平滑値 [deg/frame]・上限 ±MAX_RATE）
  門の広げ方（任意）= 門 ＋ GROW × (抜けたフレーム数 − 1)・上限 GATE_MAX（検出が抜けた時間に応じて門を広げる）
他は v9c と同じ: 3 フレーム連続の検出で確定・名札ごとに 1 回だけ鳴る・5 s 検出が無ければ名札が消える・近い順の 1 対 1 割り当て。
なぜ: 門 45° は固定値なので、速い車が近くを通るとき（54 km/h・横 3 m で約 29°/フレーム）に検出が 1 フレーム抜けただけで名札が切れ、
同じ車で 2 回目が鳴る（段 2 の残り 151 件の主因）。予想位置と照合すれば、門の幅を速さで変えなくても追従する。

使い方:
  python scripts/step12_notify_v9d_predict.py --selftest
  python scripts/step12_notify_v9d_predict.py --pred <csv|dir> --meta <metadata_dir> --out <dir> [--tag <名前>]
    → <out>/predict_compare_<tag>.md / .json（hold・案 C2 と案 D の変種を同じ枡で採点）
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step12_notify_v9c_multitrack as C  # noqa: E402

H = C.H
v9 = C.v9
MAX_RATE = 60.0          # 角速度の上限 [deg/frame]（600°/s。横 3 m を 54 km/h で通る車の最大 約 29°/frame を十分に覆う）
VEL_BETA = 0.5           # 角速度の平滑（新しい観測の重み）
GROW = 15.0              # 抜け 1 フレームごとに門を広げる幅 [deg]
GATE_MAX = 120.0         # 広げた門の上限 [deg]


def sdiff(a: float, b: float) -> float:
    """b − a の符号つき円周差（−180, 180]"""
    return (b - a + 180.0) % 360.0 - 180.0


def warn_fires_v9d(pred_clip, gate: float, nframes: int = 100, confirm: int = v9.WARN_CONFIRM, expire: int = v9.REFRACTORY,
                   ema: float = C.EMA_ALPHA, dedup_deg: float = 30.0, predict: bool = True, vel_beta: float = VEL_BETA,
                   max_rate: float = MAX_RATE, grow: float = 0.0, gate_max: float = GATE_MAX):
    """[(frame, cls, az)]。predict=False・grow=0 なら v9c（案 C2）と同じ動き。"""
    fires = []
    tracks = defaultdict(list)     # cls -> [dict(az, vel, last, run, fired, born)]
    for k in range(nframes):
        dets = defaultdict(list)
        for c, az, el in pred_clip.get(k, []):
            if c in C.WARN:
                dets[c].append(az)
        for c in list(dets):
            dets[c] = C.dedup(dets[c], dedup_deg)
        for c in C.WARN:
            alive = [t for t in tracks[c] if k - t["last"] < expire]
            tracks[c] = alive
            ds = dets.get(c, [])
            for t in alive:                                    # 予想位置と、抜けに応じた門
                gap = k - t["last"]
                t["pred"] = (t["az"] + (t["vel"] * gap if predict else 0.0)) % 360.0
                t["g"] = min(gate + grow * (gap - 1), gate_max)
            pairs = sorted(((C.cdiff(az, t["pred"]), i, j) for i, az in enumerate(ds) for j, t in enumerate(alive)), key=lambda x: x[0])
            used_d, used_t = set(), set()
            for d, i, j in pairs:
                t = alive[j]
                if d > t["g"] or i in used_d or j in used_t:
                    continue
                used_d.add(i); used_t.add(j)
                gap = k - t["last"]
                v_obs = sdiff(t["az"], ds[i]) / gap                       # 観測した角速度 [deg/frame]
                t["vel"] = float(np.clip((1.0 - vel_beta) * t["vel"] + vel_beta * v_obs, -max_rate, max_rate))
                t["az"] = C.circ_mean(t["pred"] if predict else t["az"], ds[i], ema)
                t["run"] = t["run"] + 1 if t["last"] == k - 1 else 1
                t["last"] = k
            for i, az in enumerate(ds):
                if i not in used_d:
                    alive.append({"az": az, "vel": 0.0, "last": k, "run": 1, "fired": False, "born": k})
            for t in alive:
                if not t["fired"] and t["run"] >= confirm and t["last"] == k:
                    fires.append((k, c, t["az"]))
                    t["fired"] = True
    return fires


VARIANTS = {
    # 名前: (gate, predict, grow)
    "案 C2（v9c と同じ・門 45°・予測なし）": (45.0, False, 0.0),
    "案 D1 予測のみ・門 45°": (45.0, True, 0.0),
    "案 D1 予測のみ・門 30°": (30.0, True, 0.0),
    "案 D2 予測＋抜けで門を広げる（45° ＋ 15°/frame・上限 120°）": (45.0, True, GROW),
    "案 D2 予測＋抜けで門を広げる（30° ＋ 15°/frame・上限 120°）": (30.0, True, GROW),
    "案 D3 予測なし・抜けで門を広げるだけ（45° ＋ 15°/frame）": (45.0, False, GROW),
}


def selftest() -> int:
    ok = []
    # 1) 速い音源（20°/frame）が 2 フレーム抜ける: v9c は名札が切れて 2 回、案 D は予測で 1 回
    fast = {k: [(0, 20.0 * k, 0.0)] for k in range(9) if k not in (4, 5)}
    ok.append(("速い音源が 2 フレーム抜け: v9c は 2 回", len(C.warn_fires_v9c(fast, 45.0, dedup_deg=30.0)) == 2))
    ok.append(("速い音源が 2 フレーム抜け: 案 D1 は 1 回", len(warn_fires_v9d(fast, 45.0)) == 1))
    # 案 D3（予測なし・抜けで門を広げるだけ）: 20°/frame の 2 フレーム抜けは、抜けの直後に名札が 30° 遅れて次のフレームで切れる（2 回）。
    # 13°/frame の 2 フレーム抜けなら v9c は切れる（差 48.75° > 45°）が D3 はつながる（門 75°）
    ok.append(("20°/frame・2 フレーム抜け: 案 D3 は次のフレームで切れて 2 回（EMA の遅れ・予測が要る例）", len(warn_fires_v9d(fast, 45.0, predict=False, grow=GROW)) == 2))
    mid = {k: [(0, 13.0 * k, 0.0)] for k in range(9) if k not in (4, 5)}
    ok.append(("13°/frame・2 フレーム抜け: v9c は 2 回", len(C.warn_fires_v9c(mid, 45.0, dedup_deg=30.0)) == 2))
    ok.append(("13°/frame・2 フレーム抜け: 案 D3（広げるだけ）は 1 回", len(warn_fires_v9d(mid, 45.0, predict=False, grow=GROW)) == 1))
    ok.append(("13°/frame・2 フレーム抜け: 案 D1（予測）も 1 回", len(warn_fires_v9d(mid, 45.0)) == 1))
    # 2) 2 音源が 0° と 120° に同時: 2 回
    two = {k: [(0, 0.0, 0.0), (0, 120.0, 0.0)] for k in range(100)}
    f2 = warn_fires_v9d(two, 45.0)
    ok.append(("2 音源（0°・120°）: 2 回", len(f2) == 2 and {round(a) % 360 for _, _, a in f2} == {0, 120}))
    # 3) 5 s 以上途切れたら 2 回
    gap = {k: [(0, 10.0, 0.0)] for k in list(range(0, 21)) + list(range(80, 100))}
    ok.append(("5 s 以上途切れたら 2 回", len(warn_fires_v9d(gap, 45.0)) == 2))
    # 4) 静止音源の方位が ±10° で揺れる: 予測が暴れずに 1 回
    rng = np.random.default_rng(0)
    jit = {k: [(0, float(rng.uniform(-10, 10)), 0.0)] for k in range(100)}
    ok.append(("方位が揺れる静止音源: 1 回", len(warn_fires_v9d(jit, 45.0)) == 1))
    # 5) 2 フレームでは鳴らない
    ok.append(("2 フレームでは鳴らない", len(warn_fires_v9d({0: [(0, 10.0, 0.0)], 1: [(0, 10.0, 0.0)]}, 45.0)) == 0))
    # 6) predict=False・grow=0 は v9c（案 C2）と同じ発火（乱数の場面 20 本）
    same = 0
    for s in range(20):
        r = np.random.default_rng(s)
        clip = {k: [(0, float(r.uniform(-180, 180)), 0.0)] for k in range(100) if r.random() < 0.8}
        same += int(warn_fires_v9d(clip, 45.0, predict=False, grow=0.0) == C.warn_fires_v9c(clip, 45.0, dedup_deg=30.0))
    ok.append(("予測なし・広げなし = v9c と同一（20 場面）", same == 20))
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
    tag = opt("--tag", "main")
    manifest = opt("--manifest")                     # 監査 R01（9/28）: 名簿を分母に固定する
    out.mkdir(parents=True, exist_ok=True)
    pred, events = C.load_fixed(pred_p, meta, Path(manifest) if manifest else None)
    collapsed = {clip: {k: [(c, az, el) for c, az, el in dets] for k, dets in pred[clip].items()} for clip in events}
    results = {"hold（現行・前回検出から 5 s・45°）": C.score({c: H.warn_fires(collapsed[c], True) for c in events}, events)}
    for name, (g, pr, gr) in VARIANTS.items():
        results[name] = C.score({c: warn_fires_v9d(pred[c], g, predict=pr, grow=gr) for c in events}, events)
    L = [f"# 警告音の規則の比較（段 2b・予測つきの名札 = 案 D・2026-09-28・{tag}）", "",
         f"- 予測 = `{pred_p.as_posix()}`。正解 = `{meta.as_posix()}`。クリップ {len(events):,} 本・出来事 {sum(len(v) for v in events.values()):,} 件"
         + (f"（分母 = 名簿 `{Path(manifest).as_posix()}` に固定・監査 R01）" if manifest else "（分母 = 予測があり警告音の正解がある clip だけ）") + "。数字は**開発用の数値**（確定評価のデータは見ていない）。",
         "- 案 D = 案 C2 の照合先を「名札の現在地」から「予想位置（現在地 ＋ 角速度 × 抜けたフレーム）」に替える。D2 は抜けた時間に応じて門も広げる。D3 は予測なしで広げるだけ。",
         "- 見る指標: 到達（出来事の窓 ±1 s に 1 回以上）、重複した出来事と余分な回数（減らしたい）、出来事の外の発火（増やさない）、同クラス 2 台目の到達（落とさない）。", ""]
    summary = {}
    for name, S in results.items():
        head, lines, tot = C.fmt(S)
        L += [f"## {name}", "", f"- {head}"] + lines + [""]
        summary[name] = dict(tot)
    (out / f"predict_compare_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"predict_compare_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
