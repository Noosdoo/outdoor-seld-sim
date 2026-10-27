# -*- coding: utf-8 -*-
"""警告音クラスの持続抑制（ホールド方式）— 設計 2026-08-24 の実装（2026-08-30）。

設計= md/design/警告音の持続抑制_設計_2026-08-24.md。v9のルールv1は
「前回**発火**から5秒」で不応期を判定するため、**鳴り続けている音でも5秒おきに
再通知される**（踏切の前に30秒立つと最大6回）。本実装は
「前回**検出**から5秒」= 検出が続く限り不応期の時計を更新し続ける方式に変える。

- 対象は**警告音5クラスのみ**（siren/horn/backup_beep/bike_bell/crossing）。
  車は v4.1/v4.2（最接近予測）の管轄なので触らない
- 設計の予測: 到達率**不変**（最初の発火は変わらない）・重複発火**減**
- 折衷案（持続30秒超で1回だけ再通知）は**入れない**（30秒に根拠が無い。設計§4どおり
  「最初の1回を見落としたら次が来ない」を限界として記録する）

step12_notify_v9.py は変更しない（新バージョンは新ファイル、の規約）。
定数（WARN_CONFIRM/REFRACTORY/DIR_REFRACT_DEG）はv9から読み込んで共有する。

使い方（新旧比較の採点）:
  python scripts/step12_notify_v9b_hold.py <pred_csv> <metadata_distディレクトリ> <出力dir>
到達（GTイベント窓内に発火）・イベントあたり発火数・イベント外発火を新旧で並べる。
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location(
    "nv9", ROOT / "scripts" / "step12_notify_v9.py")
v9 = importlib.util.module_from_spec(_spec)
sys.modules["nv9"] = v9
_spec.loader.exec_module(v9)

FPS = 10
WARN_CLASSES = (0, 1, 2, 3, 5)          # siren/horn/backup_beep/bike_bell/crossing
CLS_JP = {0: "siren", 1: "horn", 2: "backup_beep", 3: "bike_bell", 5: "crossing"}


def cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def warn_fires(pred_clip, hold: bool, nframes: int = 100):
    """警告音5クラスの発火列 [(frame, class, az)]。

    hold=False: v1と同一（前回**発火**から REFRACTORY 未満は抑止）。
    hold=True : 発火済みエピソードの時計を**検出のたびに更新**（前回**検出**から5秒）。
                検出が5秒以上途切れてから再検出されたときだけ新イベントとして鳴る。
    """
    fires = []
    last = defaultdict(list)            # class -> [[基準frame, az]]
    byframe = {k: {c: (a, e) for c, a, e in evs} for k, evs in pred_clip.items()}

    def blocked(c, k, az):
        return any(k - kp < v9.REFRACTORY and cdiff(az, ap) <= v9.DIR_REFRACT_DEG
                   for kp, ap in last[c])

    for k in range(nframes):
        for c in WARN_CLASSES:
            det = byframe.get(k, {}).get(c)
            if hold and det is not None:
                # 検出があれば（発火の有無に関わらず）方位の合うエピソードの時計だけ更新。
                # ⚠️ 方位は発火時のアンカーに固定する。検出に追従させる案は、移動音源の
                #    エピソードが「動く網」になって同一クラス2音源目を飲み込み、
                #    到達が6件変化した（2026-08-30の検証で棄却）。アンカー固定なら
                #    静止持続音（踏切前）は抑制され、移動音源の方向別再通知はv1と同じ
                for e in last[c]:
                    if cdiff(det[0], e[1]) <= v9.DIR_REFRACT_DEG:
                        e[0] = k
            ks = range(k - v9.WARN_CONFIRM + 1, k + 1)
            if k - v9.WARN_CONFIRM + 1 < 0:
                continue
            if all(c in byframe.get(kk, {}) for kk in ks):
                az = byframe[k][c][0]
                if blocked(c, k, az):
                    continue
                fires.append((k, c, az))
                last[c].append([k, az])
    return fires


# --------------------------------------------------------- 新旧比較の採点
def load_pred7(path: Path):
    """7列(clip,frame,class,track,az,el,dist)・旧6列の両対応でaz位置を正しく読む。"""
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[4]), float(p[5])))
        elif len(p) == 6:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return dict(out)


def warn_events(meta_dir: Path, clip: str):
    """metadata_dist から警告音クラスのGTイベント（(cls,track)の連続ラン）を返す。"""
    f = meta_dir / f"{clip}.csv"
    if not f.exists():
        return []
    per = defaultdict(list)
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) >= 5 and int(g[1]) in WARN_CLASSES:
            per[(int(g[1]), int(g[2]))].append(int(g[0]))
    out = []
    for (c, t), js in per.items():
        js.sort()
        run = [js[0]]
        for j in js[1:]:
            if j == run[-1] + 1:
                run.append(j)
            else:
                out.append((c, run[0], run[-1]))
                run = [j]
        out.append((c, run[0], run[-1]))
    return out


def main() -> int:
    pred_path, meta_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    pred = load_pred7(pred_path)

    stats = {False: defaultdict(lambda: [0, 0, 0, 0]),   # cls -> [到達, イベント数, 発火計, イベント外発火]
             True: defaultdict(lambda: [0, 0, 0, 0])}
    n_multi = {False: 0, True: 0}                        # 1イベントに2発火以上
    reach_diff = []
    for clip in sorted(pred):
        evs = warn_events(meta_dir, clip)
        reach = {}                       # (hold, イベント番号) -> 到達したか
        for hold in (False, True):
            fires = warn_fires(pred[clip], hold)
            by_cls = defaultdict(list)
            for k, c, az in fires:
                by_cls[c].append(k)
            used = {c: [False] * len(v) for c, v in by_cls.items()}
            for ei, (c, f0, f1) in enumerate(evs):
                st = stats[hold][c]
                st[1] += 1
                inwin = [i for i, k in enumerate(by_cls.get(c, []))
                         if f0 - FPS <= k <= f1 + FPS]
                reach[(hold, ei)] = bool(inwin)
                if inwin:
                    st[0] += 1
                    if len(inwin) > 1:
                        n_multi[hold] += 1
                    for i in inwin:
                        used[c][i] = True
            for c, ks in by_cls.items():
                stats[hold][c][2] += len(ks)
                stats[hold][c][3] += sum(1 for u in used[c] if not u)
        for ei, (c, f0, f1) in enumerate(evs):
            if reach[(False, ei)] != reach[(True, ei)]:
                reach_diff.append((clip, c, reach[(False, ei)], reach[(True, ei)]))

    R = [f"# 警告音の持続抑制（ホールド方式）新旧比較 pred={pred_path.name}", "",
         f"- 規則: v1=前回**発火**から{v9.REFRACTORY/10:.0f}s / "
         f"hold=前回**検出**から{v9.REFRACTORY/10:.0f}s（方向±{v9.DIR_REFRACT_DEG:.0f}°共通）",
         "", "| クラス | 到達(v1) | 到達(hold) | 発火数(v1) | 発火数(hold) | "
         "イベント外発火(v1) | (hold) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    tot = {h: [0, 0, 0, 0] for h in (False, True)}
    for c in WARN_CLASSES:
        a, b = stats[False][c], stats[True][c]
        for i in range(4):
            tot[False][i] += a[i]
            tot[True][i] += b[i]
        if a[1] == 0 and a[2] == 0 and b[2] == 0:
            continue
        R.append(f"| {CLS_JP[c]} | {a[0]}/{a[1]} | {b[0]}/{b[1]} "
                 f"| {a[2]:,} | {b[2]:,} | {a[3]:,} | {b[3]:,} |")
    a, b = tot[False], tot[True]
    R += [f"| **計** | **{a[0]}/{a[1]}** | **{b[0]}/{b[1]}** | **{a[2]:,}** "
          f"| **{b[2]:,}** | **{a[3]:,}** | **{b[3]:,}** |", "",
          f"- 1イベントに2発火以上（重複通知）: v1 {n_multi[False]:,}件 → "
          f"hold {n_multi[True]:,}件",
          "- ⚠️ 上は**10秒クリップでの実測**。設計の動機（踏切前30秒静止で6回→1回）は"
          "規則からの**外挿**であり実測ではない（第10回監査の条件・卒論でもこの表記を保つ）",
          f"- **到達の変化したイベント: {len(reach_diff)}件**"
          + ("（設計どおり0＝到達率不変）" if not reach_diff else " ⚠️ 設計の予測と異なる。要調査"),
          ""]
    if reach_diff:
        R += [f"  - {c_}: {cl}" for cl, c_, *_ in reach_diff[:10]]
    out_md = outdir / "hold_compare.md"
    out_md.write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
