# -*- coding: utf-8 -*-
"""警告音の重複振動の内訳（2026-09-28・本人依頼 `md/handoff/Fable依頼_警告音の重複振動_2026-09-27.md` 段 1）。

同じ出来事（GT の警告音イベント）に 2 回以上振動したとき、2 回目以降がなぜ許されたかを、
規則 hold（`step12_notify_v9b_hold.py` の warn_fires・前回検出から 5 秒・方位 ±45° のアンカー固定）を
そのまま再生して数える。規則も既存ファイルも変えない（読むだけ）。

原因の区分（2 回目以降の発火 1 つごと）:
  a  方位の基準（アンカー）から 45° より離れて、別の出来事として鳴った
       a1 = 正解の音源そのものが 45° 以上動いていた（動く音源）
       a2 = 正解は 45° 未満しか動いていない（予測の方位のばらつき・別の音源との混同）
  b  45° 以内のアンカーはあったが、検出が 5 秒以上途切れて時計が切れていた
  c  それ以外（同じクラスの先の出来事のアンカーしか無い等）

使い方:
  python scripts/_warn_dup_breakdown.py --pred out/causal_ft_2026-08-19/val_all_causalft.csv --meta out/dataset_outdoor_siren_v12/metadata_dist --out out/warn_dup_2026-09-28
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
CLS_JP = H.CLS_JP


def load_pred(path: Path):
    out = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                out[normalize(p[0])][int(p[1])].append((int(p[2]), float(p[4]), float(p[5])))
            elif len(p) == 6:
                out[normalize(p[0])][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return dict(out)


def gt_tracks(meta_dir: Path, clip: str):
    """{(cls, track): {frame: az}} 警告音クラスだけ"""
    f = meta_dir / f"{clip}.csv"
    per = defaultdict(dict)
    if not f.exists():
        return per
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            g = line.strip().split(",")
            if len(g) >= 5 and int(g[1]) in H.WARN_CLASSES:
                per[(int(g[1]), int(g[2]))][int(g[0])] = float(g[3])
    return per


def gt_events(tracks):
    """(cls, track, j0, j1) の連続区間"""
    out = []
    for (c, t), fr in tracks.items():
        js = sorted(fr)
        run = [js[0]]
        for j in js[1:]:
            if j == run[-1] + 1:
                run.append(j)
            else:
                out.append((c, t, run[0], run[-1])); run = [j]
        out.append((c, t, run[0], run[-1]))
    return out


def gt_az_at(tracks, c, k):
    """frame k に最も近い正解の方位（同クラスのどれかのトラック・|Δframe| ≤ 5）"""
    best = None
    for (cc, t), fr in tracks.items():
        if cc != c:
            continue
        for kk in (k, k - 1, k + 1, k - 2, k + 2, k - 3, k + 3, k - 4, k + 4, k - 5, k + 5):
            if kk in fr:
                d = abs(kk - k)
                if best is None or d < best[0]:
                    best = (d, fr[kk], t)
                break
    return None if best is None else (best[1], best[2])


def warn_fires_traced(pred_clip, tracks, nframes=100):
    """warn_fires(hold=True) と同じ発火列を作り、各発火に「なぜ許されたか」を付ける。規則のコードは v9b と同じ式。"""
    fires = []
    last = defaultdict(list)            # class -> [[基準frame, az, anchor_frame]]
    byframe = {k: {c: (a, e) for c, a, e in evs} for k, evs in pred_clip.items()}
    for k in range(nframes):
        for c in H.WARN_CLASSES:
            det = byframe.get(k, {}).get(c)
            if det is not None:
                for e in last[c]:
                    if H.cdiff(det[0], e[1]) <= v9.DIR_REFRACT_DEG:
                        e[0] = k
            ks = range(k - v9.WARN_CONFIRM + 1, k + 1)
            if k - v9.WARN_CONFIRM + 1 < 0:
                continue
            if all(c in byframe.get(kk, {}) for kk in ks):
                az = byframe[k][c][0]
                blocked = any(k - kp < v9.REFRACTORY and H.cdiff(az, ap) <= v9.DIR_REFRACT_DEG for kp, ap, _ in last[c])
                if blocked:
                    continue
                # 原因づけ
                if not last[c]:
                    cause = "first"; detail = {}
                else:
                    near = [(k - kp, kp, ap, ka) for kp, ap, ka in last[c] if H.cdiff(az, ap) <= v9.DIR_REFRACT_DEG]
                    if near:
                        gap = min(n[0] for n in near)
                        cause = "b"; detail = {"gap_s": gap / FPS}
                    else:
                        # 最も近い角度のアンカー（時刻が最新のもの）と、正解の動き
                        kp, ap, ka = max(last[c], key=lambda e: e[0])
                        dang = H.cdiff(az, ap)
                        g_now = gt_az_at(tracks, c, k); g_anc = gt_az_at(tracks, c, ka)
                        if g_now is not None and g_anc is not None:
                            gmove = H.cdiff(g_now[0], g_anc[0])
                            sub = "a1" if gmove >= v9.DIR_REFRACT_DEG else "a2"
                            same_track = g_now[1] == g_anc[1]
                        else:
                            gmove = None; sub = "a?"; same_track = None
                        cause = sub; detail = {"pred_dang": dang, "gt_move": gmove, "same_track": same_track}
                fires.append((k, c, az, cause, detail))
                last[c].append([k, az, k])
    return fires


def main() -> int:
    argv = sys.argv[1:]
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    pred_p, meta, out = Path(opt("--pred")), Path(opt("--meta")), Path(opt("--out"))
    out.mkdir(parents=True, exist_ok=True)
    pred = load_pred(pred_p)
    per_cls = defaultdict(lambda: defaultdict(int))   # cls -> cause -> n（2 回目以降の発火）
    ev_multi = defaultdict(int); ev_n = defaultdict(int); ev_reach = defaultdict(int)
    gaps = defaultdict(list); dangs = defaultdict(list)
    multi_src_clips = defaultdict(set); multi_src_events = defaultdict(int)   # 同クラス 2 音源以上
    n_clips = 0
    rows = []
    for clip in sorted(pred):
        tracks = gt_tracks(meta, clip)
        if not tracks:
            continue
        n_clips += 1
        evs = gt_events(tracks)
        # 同じクラスの音源が 2 つ以上（トラック数）
        by_c = defaultdict(set)
        for (c, t) in tracks:
            by_c[c].add(t)
        for c, ts in by_c.items():
            if len(ts) >= 2:
                multi_src_clips[c].add(clip)
                # 時間が重なる組があるか
                spans = [(min(tracks[(c, t)]), max(tracks[(c, t)])) for t in ts]
                spans.sort()
                if any(spans[i + 1][0] <= spans[i][1] for i in range(len(spans) - 1)):
                    multi_src_events[c] += 1
        fires = warn_fires_traced(pred[clip], tracks)
        by_cls = defaultdict(list)
        for f in fires:
            by_cls[f[1]].append(f)
        for (c, t, f0, f1) in evs:
            ev_n[c] += 1
            inwin = [f for f in by_cls.get(c, []) if f0 - FPS <= f[0] <= f1 + FPS]
            if inwin:
                ev_reach[c] += 1
            if len(inwin) > 1:
                ev_multi[c] += 1
                for f in inwin[1:]:
                    k, _, az, cause, det = f
                    per_cls[c][cause] += 1
                    if cause == "b":
                        gaps[c].append(det["gap_s"])
                    if cause.startswith("a"):
                        dangs[c].append(det.get("pred_dang"))
                    rows.append({"clip": clip, "cls": CLS_JP[c], "track": t, "event": f"{f0}-{f1}", "frame": k, "az": az, "cause": cause,
                                 **{k2: v for k2, v in det.items()}})
    with open(out / "dup_fires.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "cls", "track", "event", "frame", "az", "cause", "gap_s", "pred_dang", "gt_move", "same_track"]); w.writeheader(); w.writerows(rows)
    # まとめ
    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "—"
    L = ["# 警告音の重複振動の内訳（段 1・2026-09-28）", "",
         f"- 予測 = `{pred_p.as_posix()}`（開発用の検証データ・v12 の val・因果 ft・10 秒クリップ）。正解 = `{meta.as_posix()}`。クリップ {n_clips:,} 本。規則 = hold（前回検出から {v9.REFRACTORY / 10:.0f} s・方位 ±{v9.DIR_REFRACT_DEG:.0f}°・アンカー固定）。数字は**開発用の数値**。",
         "- 2 回目以降の振動 1 つごとに、なぜ許されたかを規則の式で判定した（a = アンカーから 45° 超え、b = 検出が 5 秒途切れた、c = その他）。a はさらに、正解の音源が本当に 45° 以上動いていたか（a1）、動いていないのに予測の方位が散った・別の音源と取り違えた（a2）に分けた。", ""]
    tot = defaultdict(int)
    for c in H.WARN_CLASSES:
        n2 = sum(per_cls[c].values())
        L += [f"## {CLS_JP[c]}", "",
              f"- 出来事 {ev_n[c]} 件・到達 {ev_reach[c]} 件・2 回以上振動した出来事 {ev_multi[c]} 件（{pct(ev_multi[c], ev_n[c])}）・2 回目以降の振動 {n2} 回"]
        for cause, label in (("a1", "a1 音源が 45° 以上動いた"), ("a2", "a2 音源は動いていない（予測の散り・取り違え）"), ("a?", "a? 正解が引けない"), ("b", "b 検出が 5 秒途切れた"), ("c", "c その他")):
            if per_cls[c].get(cause):
                L.append(f"  - {label}: {per_cls[c][cause]} 回（{pct(per_cls[c][cause], n2)}）")
                tot[cause] += per_cls[c][cause]
        if gaps[c]:
            g = sorted(gaps[c]); L.append(f"  - b の途切れの長さ: 中央 {g[len(g) // 2]:.1f} s・最小 {g[0]:.1f} s・最大 {g[-1]:.1f} s")
        if dangs[c]:
            d = sorted(x for x in dangs[c] if x is not None); L.append(f"  - a の予測方位のずれ（アンカーから）: 中央 {d[len(d) // 2]:.0f}°・最小 {d[0]:.0f}°")
        L.append(f"- 同じクラスの音源が 2 つ以上あるクリップ: {len(multi_src_clips[c])} 本（時間が重なる組のあるクリップ {multi_src_events[c]} 本）")
        L.append("")
    n2all = sum(tot.values())
    L += ["## 全体", "", f"- 2 回目以降の振動 {n2all} 回の内訳: " + "・".join(f"{k} {v} 回（{pct(v, n2all)}）" for k, v in sorted(tot.items())),
          f"- 2 回以上振動した出来事: {sum(ev_multi.values())} 件 / {sum(ev_n.values())} 件（到達 {sum(ev_reach.values())} 件）",
          f"- 同じクラスの音源が 2 つ以上あるクリップ: " + "・".join(f"{CLS_JP[c]} {len(multi_src_clips[c])}" for c in H.WARN_CLASSES),
          "- 発火ごとの一覧 = `dup_fires.csv`（clip・クラス・出来事・frame・原因・途切れ秒・方位のずれ・正解の動き）"]
    (out / "breakdown.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"n_clips": n_clips, "per_cls": {CLS_JP[c]: dict(per_cls[c]) for c in H.WARN_CLASSES}, "ev_n": {CLS_JP[c]: ev_n[c] for c in H.WARN_CLASSES},
                                                   "ev_multi": {CLS_JP[c]: ev_multi[c] for c in H.WARN_CLASSES}, "multi_src_clips": {CLS_JP[c]: len(multi_src_clips[c]) for c in H.WARN_CLASSES},
                                                   "multi_src_overlap": {CLS_JP[c]: multi_src_events[c] for c in H.WARN_CLASSES}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
