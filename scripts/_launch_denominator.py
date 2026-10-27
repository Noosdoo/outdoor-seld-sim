# -*- coding: utf-8 -*-
"""発進（v19launch）の val 300 本の通知層の数字を、場面（seed・150）単位と、分母の内訳つきで出す（2026-09-24・Fable）。

背景: Opus 5.5 監査 B（`md/audit/監査_Opus55_dev2結果_2026-09-24.md` B09・P01）。統一採点器の安全抑制の分母は「そのクリップにいる安全帯の車のトラック全部」なので、
発進のクリップに背景の車がいれば混ざる、300 本 = 150 場面 × 2 高さなので本数単位の 100% は場面単位で見直すべき、という指摘。
→ (1) 各クリップの距離クラスのトラック数を正解から数えて「発進の車だけか」を確かめる（生成側 `sample_v19launch` は音源を 1 台しか置かない）
   (2) 統一採点器の clip_stats をそのまま使い、seed（= 場面）ごとに 2 本をまとめた表を出す（到達・抑制は場面の 2 本がどちらも当たった／どちらも黙った、で数える）

入力: plan（launch val の行）・正解フォルダ（metadata_dist・新名）・予測 csv（自己 val・selfcausal）。出力: md と json。
使い方:
  python scripts/_launch_denominator.py <出力dir> --plan out/launch_denominator_2026-09-24/launch_val_plan.csv \
      --meta out/launch_denominator_2026-09-24/gt --pred out/launch_denominator_2026-09-24/dev2_e149_selfcausal.csv [--minframe 40]
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("_score_unified", ROOT / "scripts" / "_score_unified.py")
S = importlib.util.module_from_spec(_spec); sys.modules["_score_unified"] = S; _spec.loader.exec_module(S)


def arg(argv, k, d=None):
    if k in argv:
        i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
    return d


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[0])
    plan = ROOT / arg(argv, "--plan"); meta = ROOT / arg(argv, "--meta"); pred_p = ROOT / arg(argv, "--pred")
    minframe = int(arg(argv, "--minframe", "40"))
    rows = list(csv.DictReader(open(plan, encoding="utf-8")))
    rows = [r for r in rows if r.get("scenario") == "v19launch" and r.get("split") == "val"]
    pred = S.load_pred_rows(pred_p, horiz=False, strict=False)
    per_clip, tracks_per_clip, tiers = {}, {}, {}
    ev_rows = []      # (clip, seed, track, tier, hit) — 発進の車は track 0（`_car_src(0, …)`）、それ以外は背景（v13 の場面が足す車・キック・自転車）
    for r in rows:
        c = r["clip_id"]
        trk = list(S.DG.gt_tracks(meta, c))
        evs = [S.DG.mk_event(cl, t, fr) for cl, t, fr in trk]
        tracks_per_clip[c] = len(evs)
        tiers[c] = [e["tier"] for e in evs]
        per_clip[c] = S.clip_stats(pred, meta, c, minframe)
        # イベント単位の当たり（clip_stats と同じ帰属・同じ「最接近の 1 秒後まで」）
        frames = pred.get(c, {})
        fires = S.fires_of(frames) if frames else []
        att = S.A.attribute(fires, evs) if fires else []
        got = {}
        for f_, i in zip(fires, att):
            if i is not None and f_[0] <= evs[i]["cpa"] + S.FPS:
                got.setdefault(i, set()).add(f_[2])
        for i, ((cl, t, fr), ev) in enumerate(zip(trk, evs)):
            # 発進の車 = クラス car(4) の track 0。track 番号はクラスごとに 0 から振られるので、キック・自転車の track 0 と区別する
            ev_rows.append((c, r["seed"], (int(cl) == 4 and int(t) == 0), ev["tier"], bool(got.get(i)), "強" in got.get(i, set()), int(cl), int(t)))
    n_multi = sum(1 for c, n in tracks_per_clip.items() if n != 1)
    # 発進の車の正解が「途切れた区間」に分かれている本（遠くて小さい音の始まりでラベルが飛び飛びになる）は、統一採点器では区間ごとに別イベントになる。
    # ここでは 1 本 = 発進の車 1 台として、区間をまとめる（層 = 最接近が最も近い区間の層・当たり = どれかの区間で当たり）。
    merged = {}
    for e in ev_rows:
        if not e[2]:
            continue
        m = merged.setdefault(e[0], {"seed": e[1], "tiers": [], "hit": False, "strong": False, "n_seg": 0})
        m["tiers"].append(e[3]); m["hit"] |= e[4]; m["strong"] |= e[5]; m["n_seg"] += 1
    rank = {"critical": 0, "caution": 1, "safe": 2}
    for m in merged.values():
        m["tier"] = min(m["tiers"], key=lambda t: rank[t])
    n_frag = sum(1 for m in merged.values() if m["n_seg"] > 1)
    def _rate(sel, ok):
        v = [e for e in ev_rows if sel(e)]
        return (sum(1 for e in v if ok(e)), len(v))
    def _mrate(tier, ok):
        v = [m for m in merged.values() if m["tier"] == tier]
        return (sum(1 for m in v if ok(m)), len(v))
    launch_only = {"crit": _mrate("critical", lambda m: m["hit"]), "strong": _mrate("critical", lambda m: m["strong"]),
                   "caut": _mrate("caution", lambda m: m["hit"]), "safe": _mrate("safe", lambda m: not m["hit"]),
                   "n_clips_fragmented": n_frag, "n_segments_total": sum(m["n_seg"] for m in merged.values())}
    background = {"n_events": sum(1 for e in ev_rows if not e[2]),
                  "tiers": {t: sum(1 for e in ev_rows if not e[2] and e[3] == t) for t in ("critical", "caution", "safe")},
                  "classes": {str(k): sum(1 for e in ev_rows if not e[2] and e[6] == k) for k in sorted({e[6] for e in ev_rows if not e[2]})},
                  "safe": _rate(lambda e: not e[2] and e[3] == "safe", lambda e: not e[4])}
    # 場面単位（発進の車だけ・区間をまとめた後・2 高さがどちらも成功）
    scene0 = {"crit": [0, 0], "strong": [0, 0], "caut": [0, 0], "safe": [0, 0]}
    by_seed_m = defaultdict(list)
    for m in merged.values():
        by_seed_m[m["seed"]].append(m)
    for seed, ms in by_seed_m.items():
        tier = ms[0]["tier"]
        key = {"critical": "crit", "caution": "caut", "safe": "safe"}[tier]
        scene0[key][1] += 1
        if (all(m["hit"] for m in ms) if tier != "safe" else all(not m["hit"] for m in ms)):
            scene0[key][0] += 1
        if tier == "critical":
            scene0["strong"][1] += 1
            if all(m["strong"] for m in ms):
                scene0["strong"][0] += 1
    # 場面（seed）単位
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["clip_id"])
    scene = {"n_scenes": len(by_seed), "crit": [0, 0], "caut": [0, 0], "safe": [0, 0], "strong": [0, 0], "mixed": 0}
    for seed, clips in by_seed.items():
        st = [per_clip[c] for c in clips]
        for key in ("crit", "caut", "safe"):
            n = sum(x[key][1] for x in st)
            if n == 0:
                continue
            scene[key][1] += 1
            hits = [x[key][0] for x in st]
            if all(h == x[key][1] for h, x in zip(hits, st)):     # 2 本とも当たった（safe は 2 本とも黙った）
                scene[key][0] += 1
            elif any(h > 0 for h in hits):
                scene["mixed"] += 1
        n_crit = sum(x["crit"][1] for x in st)
        if n_crit:
            scene["strong"][1] += 1
            if all(x["n_strong"] == x["crit"][1] for x in st):
                scene["strong"][0] += 1
    agg = S.aggregate(list(per_clip.values()))
    res = {"n_clips": len(rows), "n_tracks_per_clip": {"all_one": n_multi == 0, "n_not_one": n_multi,
                                                        "hist": dict(sorted(defaultdict(int, {n: list(tracks_per_clip.values()).count(n) for n in set(tracks_per_clip.values())}).items()))},
           "tier_per_clip": {t: sum(1 for v in tiers.values() for x in v if x == t) for t in ("critical", "caution", "safe")},
           "clip_unit": {k: agg[k] for k in ("crit", "strong", "caut", "safe", "n_crit", "n_caut", "n_safe", "n_fire", "lead", "lead25")},
           "launch_car_only": launch_only, "background": background, "scene_unit": scene, "scene_unit_launch_car_only": scene0}
    pc = lambda t: f"{100 * t[0] / max(t[1], 1):.1f}% ({t[0]}/{t[1]})"
    L = ["# 発進（v19launch・val 300 本）— 分母の内訳と場面単位の表", "",
         f"plan = `{plan.relative_to(ROOT)}`・正解 = `{meta.relative_to(ROOT)}`・予測 = `{pred_p.relative_to(ROOT)}`・frame ≥ {minframe}", "",
         "## 1. 分母の内訳（正解のトラック数）", "",
         f"- 距離クラスのトラック数がちょうど 1 のクリップ: {len(rows) - n_multi} / {len(rows)}（1 でないクリップ {n_multi}・内訳 {res['n_tracks_per_clip']['hist']}）",
         f"- 発進の車 = クラス car の track 0（生成側 `sample_v19launch` → `_car_src(0, …)`）。それ以外の距離クラスのトラックは場面の仕組みが足す背景。背景のイベント {background['n_events']} 件（至近 {background['tiers']['critical']}・注意 {background['tiers']['caution']}・安全 {background['tiers']['safe']}・クラス別 {background['classes']}。4 = car・6 = kick・7 = bike）。",
         f"- 層の内訳（全イベント・本数単位）: 至近 {res['tier_per_clip']['critical']}・注意 {res['tier_per_clip']['caution']}・安全 {res['tier_per_clip']['safe']}", "",
         f"- 発進の車の正解が途切れて 2 つ以上の区間になっている本: **{launch_only['n_clips_fragmented']} / {len(rows)}**（区間の合計 {launch_only['n_segments_total']}）。遠く（13〜22 m）で小さく鳴り始めた最初の数フレームだけラベルが飛び飛びになるため。統一採点器はイベント = 連続区間なので、この本では 1 台の車が 2〜4 イベントに数えられる（安全帯の分母が {res['tier_per_clip']['safe']} になっているのはこれ）。下の「発進の車だけ」は区間を 1 台にまとめて数えた値。", "",
         "### 発進の車（car・track 0・区間をまとめて 1 本 1 台）だけの値と、背景だけの値", "",
         "| 分母 | 至近到達 | 強到達 | 注意到達 | 安全抑制 |", "| --- | --- | --- | --- | --- |",
         f"| 発進の車だけ | {pc(launch_only['crit'])} | {pc(launch_only['strong'])} | {pc(launch_only['caut'])} | {pc(launch_only['safe'])} |",
         f"| 背景の車だけ | — | — | — | {pc(background['safe'])} |",
         f"| 場面単位（発進の車だけ・2 高さとも） | {pc(scene0['crit'])} | {pc(scene0['strong'])} | {pc(scene0['caut'])} | {pc(scene0['safe'])} |", "",
         "## 2. 本数単位（統一採点器と同じ）と場面単位（seed = 150 場面・2 高さをまとめる）", "",
         "| 単位 | 至近到達 | 強到達 | 注意到達 | 安全抑制 | 備考 |", "| --- | --- | --- | --- | --- | --- |",
         f"| 本数（{len(rows)}） | {agg['crit']:.1f}% ({agg['n_crit_hit']}/{agg['n_crit']}) | {agg['strong']:.1f}% ({agg['n_strong']}/{agg['n_crit']}) | {agg['caut']:.1f}% | {agg['safe']:.1f}% (黙った {round(agg['safe'] * agg['n_safe'] / 100)}/{agg['n_safe']}) | 発火 {agg['n_fire']:,}・リード中央 {agg['lead']:.2f} s |",
         f"| 場面（{scene['n_scenes']}） | {100 * scene['crit'][0] / max(scene['crit'][1], 1):.1f}% ({scene['crit'][0]}/{scene['crit'][1]}) | {100 * scene['strong'][0] / max(scene['strong'][1], 1):.1f}% ({scene['strong'][0]}/{scene['strong'][1]}) | {100 * scene['caut'][0] / max(scene['caut'][1], 1):.1f}% ({scene['caut'][0]}/{scene['caut'][1]}) | {100 * scene['safe'][0] / max(scene['safe'][1], 1):.1f}% ({scene['safe'][0]}/{scene['safe'][1]}) | 場面の 2 本（高さ違い）が**どちらも**当たった／黙った割合。2 本で食い違った場面 {scene['mixed']} |", "",
         "読み方: 場面単位の値は本数単位より厳しい（2 本とも成功が条件）。50 場面で 100% の下限の目安は約 94%（3 の法則）。"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch_denominator.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "launch_denominator.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
