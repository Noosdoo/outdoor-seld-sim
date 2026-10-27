# -*- coding: utf-8 -*-
"""警告音の通知規則 v9c「音源を 1 台ずつ追いかける」（案 C）と、角度を変える案 A の比較（2026-09-28・依頼書 段 2）。

依頼: `md/handoff/Fable依頼_警告音の重複振動_2026-09-27.md`。段 1 の結果 = 2 回目以降の振動は全部「アンカーから 45° 超え」で、
うち 64% は音源が本当に動いていた。だから角度を広げるだけ（案 A）では動く音源に追いつかず、2 台目を飲み込む危険と表裏。
案 C = モデルが出す 1 クラス最大 3 音源を捨てず、フレームをまたいで同じ音源をつなぎ（追跡）、同じ音源と判定できたものは 1 回だけ振動させる。

既存の規則（v9 の v1・v9b の hold）は触らない。この版は**新しいファイル**で、採点は旧規則と並べて出す。採用は変更票で本人が決める。
数字は開発用の検証データ（v12 の val・因果 ft）のもの。

規則 v9c（1 クラスごとに独立）:
  - 各フレームの検出（class, az）を、生きている追跡（track）に方位の近い順に割り当てる（門 GATE_DEG 以内・1 追跡に 1 検出）。
    割り当てられなかった検出は新しい追跡になる。追跡の方位は検出に追従する（指数平滑 EMA_ALPHA）。
  - 追跡は WARN_CONFIRM（3 フレーム）連続で検出が付いたら「確定」し、そのとき 1 回だけ振動する。以後その追跡が生きている間は振動しない。
  - 検出が REFRACTORY（5 s）付かなければ追跡は消える。同じ場所に音が戻れば新しい追跡 → もう 1 回振動（旧規則と同じ考え）。
  - 2 台目: 門の外の検出は別の追跡になるので、飲み込まない。門の中に 2 台目が入ったときは区別できない（限界）。

使い方:
  python scripts/step12_notify_v9c_multitrack.py --pred out/causal_ft_2026-08-19/val_all_causalft.csv --meta out/dataset_outdoor_siren_v12/metadata_dist --out out/warn_dup_2026-09-28
  python scripts/step12_notify_v9c_multitrack.py --selftest
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.clip_ids import normalize  # noqa: E402

_spec = importlib.util.spec_from_file_location("nv9b", ROOT / "scripts" / "step12_notify_v9b_hold.py")
H = importlib.util.module_from_spec(_spec); sys.modules["nv9b"] = H; _spec.loader.exec_module(H)
v9 = H.v9
FPS = 10
WARN = H.WARN_CLASSES
CLS_JP = H.CLS_JP
GATES = (30.0, 45.0, 60.0, 90.0)          # 案 C の門（追跡に検出を付ける方位の幅）
ANGLES_A = (30.0, 45.0, 60.0, 90.0, 120.0)  # 案 A（旧 hold 規則の DIR_REFRACT_DEG を変える）
EMA_ALPHA = 0.5


def load_pred_all(path: Path):
    """{clip: {frame: [(cls, az, el), ...]}} 同じクラスの複数検出を潰さずに残す"""
    out = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                out[normalize(p[0])][int(p[1])].append((int(p[2]), float(p[4]), float(p[5])))
            elif len(p) == 6:
                out[normalize(p[0])][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return dict(out)


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circ_mean(a, b, w):
    """a に b を重み w で混ぜた方位（円周上）"""
    import math
    x = (1 - w) * math.cos(math.radians(a)) + w * math.cos(math.radians(b))
    y = (1 - w) * math.sin(math.radians(a)) + w * math.sin(math.radians(b))
    return math.degrees(math.atan2(y, x))


def dedup(azs, deg: float):
    """同じ音源の二重出力（モデルの 2 本目・3 本目が同じ方位に出る）を 1 つにまとめる。deg 以内は同一とみなし最初の 1 つを残す"""
    if deg <= 0:
        return list(azs)
    keep = []
    for az in azs:
        if all(cdiff(az, k) > deg for k in keep):
            keep.append(az)
    return keep


def warn_fires_v9c(pred_clip, gate: float, nframes: int = 100, confirm: int = v9.WARN_CONFIRM, expire: int = v9.REFRACTORY, ema: float = EMA_ALPHA, dedup_deg: float = 0.0):
    """[(frame, cls, az)] 案 C。dedup_deg > 0 なら、同じフレームの同じクラスで方位が dedup_deg 以内の検出は 1 つにまとめる（案 C2）。"""
    fires = []
    tracks = defaultdict(list)     # cls -> [dict(az, last, run, fired, born)]
    for k in range(nframes):
        dets = defaultdict(list)
        for c, az, el in pred_clip.get(k, []):
            if c in WARN:
                dets[c].append(az)
        for c in list(dets):
            dets[c] = dedup(dets[c], dedup_deg)
        for c in WARN:
            alive = [t for t in tracks[c] if k - t["last"] < expire]
            tracks[c] = alive
            ds = dets.get(c, [])
            # 割り当て: 距離（方位差）の小さい順・1 追跡 1 検出
            pairs = sorted(((cdiff(az, t["az"]), i, j) for i, az in enumerate(ds) for j, t in enumerate(alive)), key=lambda x: x[0])
            used_d, used_t = set(), set()
            for d, i, j in pairs:
                if d > gate or i in used_d or j in used_t:
                    continue
                used_d.add(i); used_t.add(j)
                t = alive[j]
                t["az"] = circ_mean(t["az"], ds[i], ema)
                t["run"] = t["run"] + 1 if t["last"] == k - 1 else 1
                t["last"] = k
            for i, az in enumerate(ds):
                if i not in used_d:
                    alive.append({"az": az, "last": k, "run": 1, "fired": False, "born": k})
            for t in alive:
                if not t["fired"] and t["run"] >= confirm and t["last"] == k:
                    fires.append((k, c, t["az"]))
                    t["fired"] = True
    return fires


def warn_fires_hold_angle(pred_clip_collapsed, angle: float):
    """案 A: 旧 hold 規則の DIR_REFRACT_DEG だけ変えて走らせる（規則のコードは v9b をそのまま呼ぶ）"""
    old = v9.DIR_REFRACT_DEG
    try:
        v9.DIR_REFRACT_DEG = angle
        return H.warn_fires(pred_clip_collapsed, True)
    finally:
        v9.DIR_REFRACT_DEG = old


def gt_events(meta_dir: Path, clip: str):
    """(cls, track, j0, j1, {frame: az}) の連続区間。同じクラスの出来事が同時に 2 つ以上あるクリップ（踏切の複数の警報機など）で、
    発火をどの出来事のものか方位で振り分けるために、正解の方位を frame ごとに持つ"""
    f = meta_dir / f"{clip}.csv"
    per = defaultdict(dict)
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            g = line.strip().split(",")
            if len(g) >= 5 and int(g[1]) in WARN:
                per[(int(g[1]), int(g[2]))][int(g[0])] = float(g[3])
    out = []
    for (c, t), fr in per.items():
        js = sorted(fr); run = [js[0]]
        for j in js[1:]:
            if j == run[-1] + 1:
                run.append(j)
            else:
                out.append((c, t, run[0], run[-1], {k: fr[k] for k in run})); run = [j]
        out.append((c, t, run[0], run[-1], {k: fr[k] for k in run}))
    return out


def gt_az_near(azmap, k):
    if k in azmap:
        return azmap[k]
    ks = sorted(azmap)
    j = min(ks, key=lambda x: abs(x - k))
    return azmap[j]


DIR_TOL_DEG = 45.0     # 補助指標「方向つきの到達」: 振り分けた出来事の正解方位から 45° 以内の発火が 1 回以上（監査 Astra 9/28 R04・新旧両方に同じ条件）


def load_fixed(pred_p: Path, meta: Path, manifest: Path | None):
    """(pred, events)。manifest があれば名簿の全 clip を分母にする（予測の無い clip は空の予測・警告音の正解が無い clip は空の出来事）。
    無ければ従来どおり「予測があり警告音の正解がある clip」だけ（段 2 の最初の表と同じ・監査 R01 の指摘のとおり分母が予測に依存する）。"""
    pred = load_pred_all(pred_p)
    if manifest:
        with open(manifest, newline="", encoding="utf-8") as f:
            ids = [normalize(r["clip_id"]) for r in csv.DictReader(f)]
        ids = list(dict.fromkeys(ids))
        events = {c: gt_events(meta, c) for c in ids}
        pred = {c: pred.get(c, {}) for c in ids}
        return pred, events
    events = {clip: ev for clip in pred if (ev := gt_events(meta, clip))}
    return pred, events


def score(fires_by_clip, events_by_clip):
    """クラス別: 出来事数・到達・重複した出来事・2 回目以降の回数・出来事の外の発火・同クラス 2 台目の到達・方向つきの到達（45° 以内）"""
    S = {c: defaultdict(int) for c in WARN}
    for clip, evs in events_by_clip.items():
        fires = fires_by_clip.get(clip, [])
        # 同クラスで時間が重なる出来事（2 台目の判定用）
        overlap = set()
        for i, (c, t, f0, f1, _) in enumerate(evs):
            for j, (c2, t2, g0, g1, _) in enumerate(evs):
                if i != j and c == c2 and not (f1 < g0 or g1 < f0):
                    overlap.add(i)
        # 発火を出来事に振り分ける: 同じクラスで窓 ±1 s に入る出来事のうち、正解の方位が最も近いもの
        assigned = defaultdict(list)
        assigned_dir = defaultdict(bool)
        for k, c, az in fires:
            S[c]["fires"] += 1
            cands = [(cdiff(az, gt_az_near(evs[i][4], k)), i) for i, (c2, t, f0, f1, _) in enumerate(evs) if c2 == c and f0 - FPS <= k <= f1 + FPS]
            if not cands:
                S[c]["outside"] += 1
                continue
            d_az, i = min(cands)
            assigned[i].append(k)
            if d_az <= DIR_TOL_DEG:
                assigned_dir[i] = True
        for i, (c, t, f0, f1, _) in enumerate(evs):
            st = S[c]; st["n_ev"] += 1
            n = len(assigned.get(i, []))
            if n:
                st["reach"] += 1
                if n > 1:
                    st["multi_ev"] += 1; st["extra"] += n - 1
            if assigned_dir.get(i):
                st["reach_dir"] += 1
            if i in overlap:
                st["n_ev_2src"] += 1
                if n:
                    st["reach_2src"] += 1
    return S


def fmt(S):
    tot = defaultdict(int)
    lines = []
    for c in WARN:
        s = S[c]
        for k, v in s.items():
            tot[k] += v
        lines.append(f"  - {CLS_JP[c]}: 到達 {s['reach']}/{s['n_ev']}・重複した出来事 {s['multi_ev']}（{s['extra']} 回余分）・発火 {s['fires']}・出来事の外 {s['outside']}・同クラス 2 台目の到達 {s['reach_2src']}/{s['n_ev_2src']}・方向つき到達（{DIR_TOL_DEG:.0f}° 以内）{s['reach_dir']}/{s['n_ev']}")
    head = (f"到達 {tot['reach']}/{tot['n_ev']}・重複した出来事 {tot['multi_ev']}（余分 {tot['extra']} 回）・発火 {tot['fires']}・出来事の外 {tot['outside']}・"
            f"同クラス 2 台目の到達 {tot['reach_2src']}/{tot['n_ev_2src']}・方向つき到達（{DIR_TOL_DEG:.0f}° 以内）{tot['reach_dir']}/{tot['n_ev']}")
    return head, lines, tot


def selftest() -> int:
    # 1 音源が 0° → 100° へ動く（10 秒で 100°・毎フレーム検出）: 旧 hold は 45° を超えた所で 2 回目、案 C は 1 回
    clip = {k: [(0, k * 1.0, 0.0)] for k in range(100)}
    f_c = warn_fires_v9c(clip, 45.0)
    f_h = H.warn_fires({k: v for k, v in clip.items()}, True)
    ok = [("動く 1 音源: 案 C は 1 回", len(f_c) == 1), ("動く 1 音源: 旧 hold は 2 回以上", len(f_h) >= 2)]
    # 2 音源が 0° と 120° に同時（両方毎フレーム）: 案 C は 2 回（別の追跡）、旧 hold も 2 回
    clip2 = {k: [(0, 0.0, 0.0), (0, 120.0, 0.0)] for k in range(100)}
    f_c2 = warn_fires_v9c(clip2, 45.0)
    ok.append(("2 音源（0°・120°）: 案 C は 2 回", len(f_c2) == 2 and {round(a) for _, _, a in f_c2} == {0, 120}))
    # 2 音源が 0° と 30°（門 45° の中）: 案 C は割り当てが 1 追跡 1 検出なので 2 追跡 → 2 回
    clip3 = {k: [(0, 0.0, 0.0), (0, 30.0, 0.0)] for k in range(100)}
    ok.append(("2 音源（0°・30°）: 1 追跡 1 検出なので 2 回", len(warn_fires_v9c(clip3, 45.0)) == 2))
    # 途切れ: 0〜20 フレーム検出、80〜99 で再検出 → 5 s 以上空くので 2 回
    clip4 = {k: [(0, 10.0, 0.0)] for k in list(range(0, 21)) + list(range(80, 100))}
    ok.append(("5 s 以上途切れたら 2 回", len(warn_fires_v9c(clip4, 45.0)) == 2))
    # 2 フレームだけの検出は確定しない（WARN_CONFIRM 3）
    clip5 = {0: [(0, 10.0, 0.0)], 1: [(0, 10.0, 0.0)]}
    ok.append(("2 フレームでは鳴らない", len(warn_fires_v9c(clip5, 45.0)) == 0))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    pred_p, meta, out = Path(opt("--pred")), Path(opt("--meta")), Path(opt("--out"))
    manifest = opt("--manifest")                     # 監査 R01（9/28）: 名簿を分母に固定する（予測の無い clip も数える）
    tag = opt("--tag", "")
    out.mkdir(parents=True, exist_ok=True)
    pred, events = load_fixed(pred_p, meta, Path(manifest) if manifest else None)
    n_nopred = sum(1 for c in events if not pred.get(c))
    n_events = sum(len(v) for v in events.values())
    # 同クラスの複数検出がどれだけあるか（モデルが 3 音源を出しているか）
    n_frames = n_multi = 0
    for clip in events:
        for k, dets in pred[clip].items():
            cnt = defaultdict(int)
            for c, az, el in dets:
                if c in WARN:
                    cnt[c] += 1
            for c, n in cnt.items():
                n_frames += 1
                if n >= 2:
                    n_multi += 1
    collapsed = {clip: {k: [(c, az, el) for c, az, el in dets] for k, dets in pred[clip].items()} for clip in events}
    results = {}
    results["v1（前回発火から 5 s・45°）"] = score({c: H.warn_fires(collapsed[c], False) for c in events}, events)
    results["hold（現行・前回検出から 5 s・45°）"] = score({c: H.warn_fires(collapsed[c], True) for c in events}, events)
    for a in ANGLES_A:
        if a == 45.0:
            continue
        results[f"案 A hold・{a:.0f}°"] = score({c: warn_fires_hold_angle(collapsed[c], a) for c in events}, events)
    for g in GATES:
        results[f"案 C 追跡・門 {g:.0f}°"] = score({c: warn_fires_v9c(pred[c], g) for c in events}, events)
    for g in GATES:
        for dd in (15.0, 30.0):
            results[f"案 C2 追跡・門 {g:.0f}°・同一方位 {dd:.0f}° 以内の二重出力を 1 つに"] = score({c: warn_fires_v9c(pred[c], g, dedup_deg=dd) for c in events}, events)
    L = [f"# 警告音の規則の比較（段 2・案 A と案 C・2026-09-28{'・' + tag if tag else ''}）", "",
         f"- 予測 = `{pred_p.as_posix()}`（開発用の検証データ・v12 の val・因果 ft）。正解 = `{meta.as_posix()}`。クリップ {len(events):,} 本・警告音の出来事 {n_events:,} 件・予測の無い clip {n_nopred:,} 本"
         + (f"（分母 = 名簿 `{Path(manifest).as_posix()}` に固定・監査 R01）" if manifest else "（分母 = 予測があり警告音の正解がある clip だけ）") + "。数字は**開発用の数値**。",
         f"- 同じクラスの検出が 1 フレームに 2 つ以上あるフレーム: {n_multi:,} / {n_frames:,}（{100 * n_multi / max(n_frames, 1):.1f}%）。旧規則はこれを 1 つに潰していた。",
         "- 見る指標: 到達（出来事の窓 ±1 s に 1 回以上鳴る・落とさない）、重複した出来事と余分な回数（減らしたい）、出来事の外の発火（増やさない）、同クラス 2 台目の到達（同じクラスの出来事が時間で重なる本での到達・落とさない）。",
         "- 発火の振り分け: 同じクラスの出来事が同時に 2 つ以上あるとき（踏切の複数の警報機など）、各発火は正解の方位が最も近い出来事のものと数える（hold_compare.md の数え方は窓に入る発火を全部その出来事に数えていたので、2 台目への発火が 1 台目の「重複」に見えていた）。",
         "- 案 A = 現行 hold の 45° を変える。案 C = 音源を 1 台ずつ追跡し、追跡ごとに 1 回だけ鳴らす（門 = 追跡に検出を付ける方位の幅・検出に追従・5 s 検出が無ければ追跡が消える）。", ""]
    summary = {}
    for name, S in results.items():
        head, lines, tot = fmt(S)
        L += [f"## {name}", "", f"- {head}"] + lines + [""]
        summary[name] = dict(tot)
    sfx = f"_{tag}" if tag else ""
    (out / f"plan_compare{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"plan_compare{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
