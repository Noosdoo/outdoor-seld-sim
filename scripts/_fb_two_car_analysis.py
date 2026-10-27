# -*- coding: utf-8 -*-
"""前後取り違えの中身を見る解析（2026-10-05・結果 md dev3 r6 §3g の数字の出どころ。10/6 Astra 再監査 RF04・RF05 で手直し）。

知覚層の評価（_perception_eval.py）と同じ対応付け（フレーム・クラスごとに方位差の合計が最小の 1 対 1・門なし・frame ≥ 40・車クラス）で
「車が同時に 2 台以上鳴っているフレーム」の組を条件で分けて、取り違え（方位差 ≥150°）の率と、予測がどこを指しているかを出す。
対応付けは scipy の linear_sum_assignment（知覚層と同じ・同点の規約も同じ）。scipy が無い環境では総当たりに落ちる（同点のときの割り当てが
変わりうるので、結果の見出しにどちらを使ったかを出す。正本の数字は scipy で出す）。

使い方:
  python scripts/_fb_two_car_analysis.py gtstats --plan <manifest.csv> [--split val] --meta <GTディレクトリ>
      GT 側の 2 台配置の分布（車が同時に 2 台鳴るフレームの割合・近い車と遠い車の方向の組）
  python scripts/_fb_two_car_analysis.py cond --plan <manifest.csv> --meta <GTディレクトリ> <ラベル>=<pred.csv> ...
      (自分の方向, 一番近い別の車の方向, 自分の遠近) ごとの 組数・誤差>20°・取り違え率
  python scripts/_fb_two_car_analysis.py mech --plan <manifest.csv> --meta <GTディレクトリ> <ラベル>=<pred.csv> ...
      取り違えた組の予測が「自分の前後鏡像」「自分の点対称」「一番近い別の車」のどれに近いか・左右の側・予測距離のずれ
対象の本は --plan（clip_id 列・split 列があれば --split の本だけ・既定 val）で決める。GT フォルダの全件を数えることはしない（RF05）。
旧名の clip_id は clip_ids.normalize で新名に直し、重複は止める。GT が無い本も止める。
GT = metadata_dist(_h) の形（frame,cls,trk,az,el,dist）。pred = 推論の 7 列 csv（clip,frame,cls,track,az,el,dist）。
方位の規約 = _perception_eval.py と同じ（0=前・+90=左・±180=後・−90=右）。
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("clip_ids", ROOT / "src/outdoor_seld/clip_ids.py")
CI = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CI)
try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    MATCHER = "scipy.linear_sum_assignment（知覚層と同じ）"
except Exception:  # pragma: no cover - 手元の素の python
    np = None; linear_sum_assignment = None
    MATCHER = "総当たり（scipy なし・同点の割り当てが知覚層と違いうる）"

DIST_CLS = {4, 6, 7}
CAR = 4
MINFRAME = 40
FLIP_DEG = 150.0


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def sector(az):
    a = (az + 180.0) % 360.0 - 180.0
    if abs(a) <= 45: return "前"
    if abs(a) >= 135: return "後"
    return "左" if a > 0 else "右"


def xflip(az):          # x -> -x（前後の鏡像・左右は保つ）: az -> 180 - az
    return (180.0 - az + 180.0) % 360.0 - 180.0


def arg(argv, key, default=None):
    if key in argv:
        i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def load_pred(p):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(p, encoding="utf-8"):
        q = line.strip().split(",")
        if len(q) < 7: continue
        d = float(q[6]) if q[6] not in ("", "nan") else float("nan")
        out[CI.normalize(q[0])][int(q[1])].append((int(q[2]), float(q[4]), float(q[5]), d))
    return out


def load_gt(f):
    out = defaultdict(list)
    for line in open(f, encoding="utf-8"):
        g = line.strip().split(",")
        if len(g) < 5: continue
        c = int(g[1])
        d = float(g[5]) if (len(g) >= 6 and c in DIST_CLS and g[5] not in ("", "nan")) else float("nan")
        out[int(g[0])].append((c, int(g[2]), float(g[3]), float(g[4]), d))
    return out


def match(gts, preds):
    """同じクラスの正解と予測を方位差の合計が最小になるよう 1 対 1 に。戻り [(gi, pi, err)]。_perception_eval.match_frame と同じ手順。"""
    if not gts or not preds: return []
    if len(gts) == 1 and len(preds) == 1: return [(0, 0, dang(preds[0][1], gts[0][2]))]
    if linear_sum_assignment is not None:
        cost = np.array([[dang(p[1], g[2]) for p in preds] for g in gts])
        gi, pi = linear_sum_assignment(cost)
        return [(int(a), int(b), float(cost[a, b])) for a, b in zip(gi, pi)]
    n, m = len(gts), len(preds); best = None
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            cost = sum(dang(preds[pi][1], gts[gi][2]) for gi, pi in enumerate(perm))
            if best is None or cost < best[0]: best = (cost, list(enumerate(perm)))
    else:
        for perm in itertools.permutations(range(n), m):
            cost = sum(dang(preds[pi][1], gts[gi][2]) for pi, gi in enumerate(perm))
            if best is None or cost < best[0]: best = (cost, [(gi, pi) for pi, gi in enumerate(perm)])
    return [(gi, pi, dang(preds[pi][1], gts[gi][2])) for gi, pi in best[1]]


def clips_from_plan(plan: Path, split: str, gtdir: Path) -> list[str]:
    """plan（clip_id 列・split 列は任意）から対象の本。旧名は新名に直す。重複・GT 無しは止める。"""
    clips = []
    with open(plan, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if "split" in r and r["split"] and CI.split_to_new(r["split"]) != split:
                continue
            clips.append(CI.normalize(r["clip_id"]))
    if not clips:
        raise ValueError(f"plan {plan} に split={split} の本が無い")
    dup = sorted(c for c, k in Counter(clips).items() if k > 1)
    if dup:
        raise ValueError(f"plan に同じ本が重複（旧名と新名の混在か重複・{len(dup)} 本・例 {dup[:3]}）")
    missing = [c for c in clips if not (gtdir / f"{c}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"GT が無い本 {len(missing)} 例 {missing[:3]}")
    return clips


def gtstats(gtdir: Path, clips):
    n_frames = n_two = n_clip_multi = 0
    conf = Counter(); far_sector = Counter(); same = opp = 0; dist_far = []
    for c in clips:
        gt = load_gt(gtdir / f"{c}.csv")
        if len({(g[0], g[1]) for k in gt for g in gt[k] if g[0] == CAR}) >= 2: n_clip_multi += 1
        for k, gs in gt.items():
            if k < MINFRAME: continue
            n_frames += 1
            cars = sorted((g for g in gs if g[0] == CAR and math.isfinite(g[4])), key=lambda g: g[4])
            if len(cars) < 2: continue
            n_two += 1
            near, far = cars[0], cars[-1]
            conf[(sector(near[2]), sector(far[2]))] += 1; far_sector[sector(far[2])] += 1
            if dang(near[2], far[2]) <= 90: same += 1
            else: opp += 1
            dist_far.append(far[4])
    print(f"\n=== GT {gtdir}: {len(clips)} 本（plan）・frame>={MINFRAME} のフレーム {n_frames:,}")
    print(f"  車が 2 台以上いる本: {n_clip_multi} ({100 * n_clip_multi / max(len(clips), 1):.0f}%)")
    print(f"  車が同時に 2 台以上鳴っているフレーム: {n_two:,} ({100 * n_two / max(n_frames, 1):.1f}%)")
    if n_two:
        print(f"  （近い車の方向, 遠い車の方向）上位: {conf.most_common(8)}")
        print(f"  遠い車の方向: {dict(far_sector)}  後/前 = {far_sector['後'] / max(far_sector['前'], 1):.2f}")
        print(f"  同じ側(方位差<=90°) {same:,} / 反対側 {opp:,}")
        dist_far.sort()
        print(f"  遠い車の距離 中央値 {dist_far[len(dist_far) // 2]:.1f} m・90%点 {dist_far[int(0.9 * len(dist_far))]:.1f} m")


def two_car_pairs(gts, pred, clips):
    """同時に 2 台以上の車がいるフレームの、車クラスの対応した組を順に返す: (g, q, err, cars)"""
    for c in clips:
        fr = pred.get(c, {})
        for k in sorted(set(gts[c]) | set(fr)):
            if k < MINFRAME: continue
            cars = [g for g in gts[c].get(k, []) if g[0] == CAR]
            if len(cars) < 2: continue
            preds = [q for q in fr.get(k, []) if q[0] == CAR]
            for gi, pi, err in match(cars, preds):
                yield cars[gi], preds[pi], err, cars


def cond(gtdir: Path, clips, specs):
    gts = {c: load_gt(gtdir / f"{c}.csv") for c in clips}
    for spec in specs:
        label, p = spec.split("=", 1)
        pred = load_pred(p)
        n = Counter(); fl = Counter(); big = Counter()
        for g, q, err, cars in two_car_pairs(gts, pred, clips):
            others = [o for o in cars if o is not g]
            o = min(others, key=lambda o: o[4] if math.isfinite(o[4]) else 1e9)   # 一番近い（音が大きいはずの）別の車
            rel = "自分が遠い" if (math.isfinite(g[4]) and math.isfinite(o[4]) and g[4] > o[4]) else "自分が近い"
            key = (sector(g[2]), sector(o[2]), rel)
            n[key] += 1
            if err >= FLIP_DEG: fl[key] += 1
            if err > 20: big[key] += 1
        print(f"\n===== {label}（plan {len(clips)} 本・予測が合った本 {sum(1 for c in clips if c in pred)}・対応付け = {MATCHER}）: "
              f"同時 2 台以上の車のフレームの組・(自分の方向, 一番近い別の車の方向, 自分の遠近) → 組数・誤差>20°・取り違え(>={FLIP_DEG:.0f}°)")
        for key in sorted(n, key=lambda k: -n[k]):
            if n[key] < 200: continue
            print(f"  {key[0]} / 他={key[1]} / {key[2]}: 組 {n[key]:,}  誤差>20° {100 * big[key] / n[key]:5.1f}%  取り違え {100 * fl[key] / n[key]:5.2f}%")


def mech(gtdir: Path, clips, specs):
    gts = {c: load_gt(gtdir / f"{c}.csv") for c in clips}
    for spec in specs:
        label, p = spec.split("=", 1)
        pred = load_pred(p)
        d_x, d_o, d_m = [], [], []
        side_gt = side_other = side_both = n = 0
        rel_flip, rel_ok = [], []
        for g, q, err, cars in two_car_pairs(gts, pred, clips):
            if math.isfinite(g[4]) and math.isfinite(q[3]):
                (rel_flip if err >= FLIP_DEG else rel_ok).append((q[3] - g[4]) / max(g[4], 0.1))
            if err < FLIP_DEG: continue
            others = [o for o in cars if o is not g]
            o = min(others, key=lambda o: dang(q[1], o[2]))
            n += 1
            d_x.append(dang(q[1], xflip(g[2]))); d_o.append(dang(q[1], o[2])); d_m.append(dang(q[1], g[2] + 180))
            sg = math.copysign(1, math.sin(math.radians(g[2]))); so = math.copysign(1, math.sin(math.radians(o[2])))
            sp = math.copysign(1, math.sin(math.radians(q[1])))
            if sg != so:
                if sp == sg: side_gt += 1
                else: side_other += 1
            else:
                side_both += 1

        def med(v):
            v = sorted(v); return v[len(v) // 2] if v else float("nan")
        print(f"\n===== {label}（plan {len(clips)} 本・対応付け = {MATCHER}）: 2 台以上の車のフレームで取り違えた組 {n:,}")
        if n:
            print(f"  予測と「自分の前後鏡像(x だけ反転)」の方位差 中央値 {med(d_x):.0f}°・≤20° の割合 {100 * sum(1 for v in d_x if v <= 20) / n:.0f}%")
            print(f"  予測と「自分の点対称(+180°)」の方位差   中央値 {med(d_m):.0f}°・≤20° の割合 {100 * sum(1 for v in d_m if v <= 20) / n:.0f}%")
            print(f"  予測と「一番近い別の車」の方位差       中央値 {med(d_o):.0f}°・≤20° の割合 {100 * sum(1 for v in d_o if v <= 20) / n:.0f}%")
            tot = side_gt + side_other
            print(f"  自分と別の車が左右で逆側のとき: 予測は自分の側 {side_gt:,} / 別の車の側 {side_other:,}"
                  f"（自分の側 {100 * side_gt / max(tot, 1):.0f}%）・同じ側で判定不能 {side_both:,}")
        print(f"  予測距離の相対ずれの中央値: 取り違えた組 {100 * med(rel_flip):+.0f}% / それ以外の組 {100 * med(rel_ok):+.0f}%")


def main() -> int:
    argv = list(sys.argv[1:])
    if len(argv) < 1 or argv[0] not in ("gtstats", "cond", "mech"):
        print(__doc__); return 2
    mode = argv.pop(0)
    plan = arg(argv, "--plan"); meta = arg(argv, "--meta"); split = CI.split_to_new(arg(argv, "--split", "val"))
    if not plan or not meta:
        print("--plan と --meta が要る"); print(__doc__); return 2
    gtdir = Path(meta)
    clips = clips_from_plan(Path(plan), split, gtdir)
    if mode == "gtstats":
        gtstats(gtdir, clips)
    elif mode == "cond":
        cond(gtdir, clips, argv)
    else:
        mech(gtdir, clips, argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
