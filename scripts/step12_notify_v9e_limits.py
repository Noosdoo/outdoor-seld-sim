# -*- coding: utf-8 -*-
"""v9e = 名札の規則（案 C2・v9c）の 3 つの限界に対する改善案を同じ枡で採点する（2026-09-28 16:06 本人「そうしましょう」・開発データだけ）。

  限界 1 同じ方向の 2 台 → 案 E「近い 2 枚目は長めに確かめる」: 既にある名札の near_deg 以内に出た音は、まとめずに名札を作るが、
         震える条件を confirm（3 フレーム = 0.3 s）ではなく confirm_near（10 フレーム = 1 s）にする。二重出力の揺れは 1 s 続かない。
         （距離で分ける案は不可: 警告音の距離出力は相対誤差の中央値が 100%（実質ゼロ）で、モデルが警告音の距離を学習していない）
  限界 2 方位の飛び（前後の取り違え） → 案 G「鏡の照合」: 直前 mirror_win フレーム以内に音が付いていた名札が今フレームに結ばれず、
         新しい音がその名札の反対側（180° ± mirror_deg）に出たら、同じ音源として名札を引き継ぐ（新しい名札を作らない）。
  限界 3 名札の寿命 5 s → expire を 3／5／8／10 s で振る。
他（門 45°・近い順の 1 対 1・半分ずつ寄せる・名札ごとに 1 回）は v9c と同じ。dedup_deg=30・near_deg=0・mirror=False・expire=50 なら v9c（案 C2）と同一。

使い方:
  python scripts/step12_notify_v9e_limits.py --selftest
  python scripts/step12_notify_v9e_limits.py --pred <csv|dir> --meta <metadata_dir> --out <dir> [--tag <名前>]
    → <out>/limits_compare_<tag>.md / .json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step12_notify_v9c_multitrack as C  # noqa: E402

H = C.H
v9 = C.v9
CONFIRM_NEAR = 10        # 案 E: 近い 2 枚目の確定に要する連続フレーム（1.0 s）
NEAR_DEG = 30.0          # 案 E: 「近い」の幅（既存名札からの方位差）
MIRROR_WIN = 2           # 案 G: 名札が消えてから何フレーム以内なら鏡の照合をするか
MIRROR_DEG = 45.0        # 案 G: 反対側（180°）からの許容幅


def warn_fires_v9e(pred_clip, gate: float = 45.0, nframes: int = 100, confirm: int = v9.WARN_CONFIRM, expire: int = v9.REFRACTORY,
                   ema: float = C.EMA_ALPHA, dedup_deg: float = 30.0, near_deg: float = 0.0, confirm_near: int = CONFIRM_NEAR,
                   mirror: bool = False, mirror_win: int = MIRROR_WIN, mirror_deg: float = MIRROR_DEG):
    fires = []
    tracks = defaultdict(list)     # cls -> [dict(az, last, run, fired, need, born)]
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
            pairs = sorted(((C.cdiff(az, t["az"]), i, j) for i, az in enumerate(ds) for j, t in enumerate(alive)), key=lambda x: x[0])
            used_d, used_t = set(), set()
            for d, i, j in pairs:
                if d > gate or i in used_d or j in used_t:
                    continue
                used_d.add(i); used_t.add(j)
                t = alive[j]
                t["az"] = C.circ_mean(t["az"], ds[i], ema)
                t["run"] = t["run"] + 1 if t["last"] == k - 1 else 1
                t["last"] = k
            for i, az in enumerate(ds):
                if i in used_d:
                    continue
                if mirror:                                   # 案 G: 消えた直後の名札の反対側なら引き継ぐ
                    cand = [(C.cdiff(az, t["az"] + 180.0), j) for j, t in enumerate(alive)
                            if j not in used_t and 1 <= k - t["last"] <= mirror_win and C.cdiff(az, t["az"] + 180.0) <= mirror_deg]
                    if cand:
                        _, j = min(cand)
                        used_t.add(j)
                        t = alive[j]
                        t["az"] = az                          # 飛んだ先へ移す（平均すると中間の無い方向になる）
                        t["run"] = t["run"] + 1 if t["last"] == k - 1 else 1
                        t["last"] = k
                        continue
                need = confirm
                if near_deg > 0 and any(C.cdiff(az, t["az"]) <= near_deg for t in alive):
                    need = confirm_near                       # 案 E: 近い 2 枚目は長めに確かめる
                alive.append({"az": az, "last": k, "run": 1, "fired": False, "need": need, "born": k})
            for t in alive:
                if not t["fired"] and t["run"] >= t.get("need", confirm) and t["last"] == k:
                    fires.append((k, c, t["az"]))
                    t["fired"] = True
    return fires


VARIANTS = {
    # 名前: kwargs
    "案 C2（基準・門 45°・同一方位 30° を 1 つに・寿命 5 s）": dict(),
    "案 E1 近い 2 枚目は 1 s 確認（まとめ無し・近い = 30°）": dict(dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=10),
    "案 E2 近い 2 枚目は 1 s 確認（15° はまとめる・近い = 30°）": dict(dedup_deg=15.0, near_deg=NEAR_DEG, confirm_near=10),
    "案 E3 近い 2 枚目は 0.5 s 確認（まとめ無し・近い = 30°）": dict(dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=5),
    "案 G1 鏡の照合（消えて 2 フレーム以内・180° ± 45°）": dict(mirror=True),
    "寿命 3 s": dict(expire=30),
    "寿命 8 s": dict(expire=80),
    "寿命 10 s": dict(expire=100),
    "案 E1 ＋ G1": dict(dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=10, mirror=True),
    "案 E1 ＋ G1 ＋ 寿命 8 s": dict(dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=10, mirror=True, expire=80),
}


def selftest() -> int:
    ok = []
    # 基準 = v9c と同一
    import numpy as np
    same = 0
    for s in range(20):
        r = np.random.default_rng(s)
        clip = {k: [(0, float(r.uniform(-180, 180)), 0.0)] for k in range(100) if r.random() < 0.8}
        same += int(warn_fires_v9e(clip) == C.warn_fires_v9c(clip, 45.0, dedup_deg=30.0))
    ok.append(("既定 = v9c（案 C2）と同一（20 場面）", same == 20))
    # 案 E: 0° と 20° に 2 台（ずっと両方）: C2 は 1 回、E1 は 2 回（2 回目は 10 フレーム目以降）
    two = {k: [(0, 0.0, 0.0), (0, 20.0, 0.0)] for k in range(100)}
    ok.append(("同じ方向 20° 差の 2 台: C2 は 1 回", len(warn_fires_v9e(two)) == 1))
    fe = warn_fires_v9e(two, dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=10)
    ok.append(("同じ方向 20° 差の 2 台: E1 は 2 回・2 回目は 1 s 以降", len(fe) == 2 and fe[1][0] >= 9))
    # 案 E: 二重出力の揺れ（20° の 2 本目が 0.5 s だけ）: E1 は 1 回
    flick = {k: [(0, 0.0, 0.0)] + ([(0, 20.0, 0.0)] if k < 5 else []) for k in range(100)}
    ok.append(("0.5 s だけの二重出力: E1 は 1 回", len(warn_fires_v9e(flick, dedup_deg=0.0, near_deg=NEAR_DEG, confirm_near=10)) == 1))
    # 案 G: 0° で 20 フレーム → 180° に飛んで 20 フレーム: C2 は 2 回、G1 は 1 回
    flip = {k: [(0, 0.0 if k < 20 else 180.0, 0.0)] for k in range(40)}
    ok.append(("前後に飛ぶ 1 台: C2 は 2 回", len(warn_fires_v9e(flip)) == 2))
    ok.append(("前後に飛ぶ 1 台: G1 は 1 回", len(warn_fires_v9e(flip, mirror=True)) == 1))
    # 案 G: 反対側に本物の 2 台目（1 台目は鳴り続ける）: G1 でも 2 回
    real2 = {k: [(0, 0.0, 0.0)] + ([(0, 180.0, 0.0)] if k >= 20 else []) for k in range(40)}
    ok.append(("反対側に本物の 2 台目: G1 も 2 回", len(warn_fires_v9e(real2, mirror=True)) == 2))
    # 寿命: 6 s 途切れ → 5 s なら 2 回・8 s なら 1 回
    gap = {k: [(0, 10.0, 0.0)] for k in list(range(0, 20)) + list(range(80, 100))}
    ok.append(("6 s 途切れ: 寿命 5 s は 2 回", len(warn_fires_v9e(gap, expire=50)) == 2))
    ok.append(("6 s 途切れ: 寿命 8 s は 1 回", len(warn_fires_v9e(gap, expire=80)) == 1))
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
    for name, kw in VARIANTS.items():
        results[name] = C.score({c: warn_fires_v9e(pred[c], **kw) for c in events}, events)
    L = [f"# 警告音の規則の比較（段 2c・3 つの限界の改善案・2026-09-28・{tag}）", "",
         f"- 予測 = `{pred_p.as_posix()}`。正解 = `{meta.as_posix()}`。クリップ {len(events):,} 本・出来事 {sum(len(v) for v in events.values()):,} 件"
         + (f"（分母 = 名簿 `{Path(manifest).as_posix()}` に固定・監査 R01）" if manifest else "（分母 = 予測があり警告音の正解がある clip だけ）") + "。数字は**開発用の数値**（確定評価のデータは見ていない）。",
         "- 案 E =既存の名札の 30° 以内に出た音はまとめずに名札を作り、震える条件を 1 s（E3 は 0.5 s）の連続にする。案 G = 消えた直後の名札の反対側（180° ± 45°）に出た音は同じ音源として引き継ぐ。寿命 = 名札が消えるまでの無検出時間。",
         "- 見る指標: 到達（出来事の窓 ±1 s に 1 回以上）、重複した出来事と余分な回数（減らしたい）、出来事の外の発火（増やさない）、同クラス 2 台目の到達（落とさない）。", ""]
    summary = {}
    for name, S in results.items():
        head, lines, tot = C.fmt(S)
        L += [f"## {name}", "", f"- {head}"] + lines + [""]
        summary[name] = dict(tot)
    (out / f"limits_compare_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"limits_compare_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
