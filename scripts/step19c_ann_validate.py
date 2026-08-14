# -*- coding: utf-8 -*-
"""Step 19c: 実録注釈CSVのスキーマ検証（2026-08-14 実録再設計レビュー反映）。

採点(step20)に回す前に、注釈が採点可能な形になっているかを機械チェックする。
現場で作った注釈の取りこぼしを、帰宅後その日のうちに潰すための道具。

検査項目:
  S1 必須列の存在（--cut 指定時は orig_file / cut_offset_s / scored も必須）
  S2 (clip_id, event_id) の一意性
  S3 class が既知クラス∪none／quadrant が F/B/L/R（正例のみ）
  S4 t_start ≤ t_cpa、いずれも [0, --dur] の範囲内
  S5 距離クラス(car_drive/kick/bike)の採点対象行に有限・非負の 横距離m がある
  S6 cut_offset_s が非負の数、orig_file が非空（--cut時）
  S6 cut_offset_s が非負の数、orig_file が非空、scored が 0/1（**負例行も含め全行**）
  S7 **負例の担当区間が原録音上で先頭0sから末尾まで隙間なく・重なりなくタイルすること**
     （step19b --mode negative の重複分割が正しく効いているかの一発検査。
       ここが崩れていると誤警告の二重計上/取りこぼしが起きる）
  S8 事前登録との突き合わせ（区分A〜E・歩行の本数、歩行対比の静止/歩行が同数か）

使い方:
  python scripts/step19c_ann_validate.py --ann out/realsmoke/ann_all.csv --cut
  python scripts/step19c_ann_validate.py --ann ann_orig.csv          # 原録音注釈
  python scripts/step19c_ann_validate.py --ann ann_all.csv --cut --strict  # 収録完了後
  （--plan で区分ごとの計画本数を上書き。既定は 2026-08-14 確定案の A〜E各20＋歩行20。
    --strict は本数不足などの警告も不合格にする＝全収録完了後の最終ゲート用）
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa"]
CUT_COLS = ["orig_file", "cut_offset_s", "scored"]
CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "car_drive", "crossing",
           "kick", "bike"}
DIST_CLASSES = {"car_drive", "kick", "bike"}
QUADS = {"F", "B", "L", "R"}
LATERAL_KEYS = ("横距離m", "横距離", "lateral_m")
PLAN_DEFAULT = {"A": 20, "B": 20, "C": 20, "D": 20, "E": 20, "歩行": 20}
EPS = 1e-6

errs, warns = [], []


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def err(msg):
    errs.append(msg)


def warn(msg):
    warns.append(msg)


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def is_scored(row) -> bool:
    v = row.get("scored")
    if v is None or str(v).strip() == "":
        return True
    return str(v).strip().lower() not in ("0", "false", "no")


def lateral_of(row):
    for k in LATERAL_KEYS:
        if row.get(k) not in (None, ""):
            x = fnum(row[k])
            return x if x is not None and x >= 0 else None
    return None


def validate(rows, cut: bool, dur: float, plan: dict):
    cols = set(rows[0].keys()) if rows else set()
    need = BASE_COLS + (CUT_COLS if cut else [])
    for c in need:
        if c not in cols:
            err(f"S1 必須列がありません: {c}")
    if errs:
        return

    seen = set()
    for i, r in enumerate(rows, start=2):          # 2 = ヘッダの次の行番号
        cid, ev = r["clip_id"].strip(), str(r.get("event_id", "")).strip()
        cls = r["class"].strip()
        tag = f"L{i} {cid}/{ev}"
        if (cid, ev) in seen:
            err(f"S2 {tag}: (clip_id,event_id) が重複しています")
        seen.add((cid, ev))
        if cls != "none" and cls not in CLASSES:
            err(f"S3 {tag}: 未知のクラス '{cls}'")
        t0, t1 = fnum(r["t_start"]), fnum(r["t_cpa"])
        if t0 is None or t1 is None:
            err(f"S4 {tag}: t_start/t_cpa が数値ではありません")
            continue
        if t0 > t1 + EPS:
            err(f"S4 {tag}: t_start({t0}) > t_cpa({t1})")
        if t0 < -EPS or t1 > dur + EPS:
            err(f"S4 {tag}: 時刻がクリップ長 [0,{dur}] の外です ({t0},{t1})")
        if cut:
            # 負例行も含めて全行を検査する（class=noneで抜けると、担当区間の
            # 根拠になる cut_offset_s が壊れていても素通りしてしまう）
            if not r.get("orig_file", "").strip():
                err(f"S6 {tag}: orig_file が空です")
            if fnum(r.get("cut_offset_s")) is None or fnum(r["cut_offset_s"]) < -EPS:
                err(f"S6 {tag}: cut_offset_s が非負の数値ではありません "
                    f"（現在 '{r.get('cut_offset_s','')}'）")
            sv = str(r.get("scored", "")).strip().lower()
            if sv not in ("", "0", "1", "true", "false", "yes", "no"):
                err(f"S6 {tag}: scored の値が不正です（'{r.get('scored')}'）。"
                    "0/1 以外は採点側が採点対象と誤認します")
        if cls == "none":
            if not is_scored(r):
                err(f"S6 {tag}: 負例行に scored=0 は使えません"
                    "（露出時間と誤警告の担当区間が消えます）")
            continue
        if is_scored(r):
            if r.get("quadrant", "").strip() not in QUADS:
                err(f"S3 {tag}: quadrant は F/B/L/R "
                    f"（現在 '{r.get('quadrant','')}'）")
            if cls in DIST_CLASSES and lateral_of(r) is None:
                err(f"S5 {tag}: 距離クラス {cls} に有効な 横距離m がありません"
                    "（欠けると採点の分母から落ちます）")

    # ---- S7 負例担当区間のタイル検査（--cut時のみ・原録音単位） ----
    if cut:
        segs = defaultdict(list)
        for r in rows:
            if r["class"].strip() != "none":
                continue
            off = fnum(r.get("cut_offset_s"))
            t0, t1 = fnum(r["t_start"]), fnum(r["t_cpa"])
            if None in (off, t0, t1):
                continue
            segs[r["orig_file"].strip()].append((off + t0, off + t1,
                                                 r["clip_id"].strip()))
        for orig, ss in sorted(segs.items()):
            ss.sort()
            total = sum(b - a for a, b, _ in ss)
            if abs(ss[0][0]) > EPS:
                err(f"S7 {orig}: 担当が原録音の先頭0.000sから始まっていません"
                    f"（{ss[0][0]:.3f}sから）→ 冒頭の誤警告を取りこぼします")
            last_local = fnum(next(r["t_cpa"] for r in rows
                                   if r["clip_id"].strip() == ss[-1][2]))
            if last_local is not None and abs(last_local - dur) > EPS:
                err(f"S7 {orig}: 末尾クリップ {ss[-1][2]} の担当が"
                    f"クリップ終端({dur}s)まで届いていません（{last_local:.3f}s）"
                    "→ 録音末尾の誤警告を取りこぼします")
            for (a1, b1, c1), (a2, b2, c2) in zip(ss, ss[1:]):
                if a2 < b1 - EPS:
                    err(f"S7 {orig}: 担当区間が重複 {c1}[{a1:.3f},{b1:.3f}) と "
                        f"{c2}[{a2:.3f},{b2:.3f}) → 誤警告を二重計上します")
                elif a2 > b1 + EPS:
                    err(f"S7 {orig}: 担当区間に隙間 [{b1:.3f},{a2:.3f}) "
                        f"（{c1}と{c2}の間）→ その区間の誤警告を取りこぼします")
            print(f"  S7 {orig}: {len(ss)}クリップ 担当合計 {total:.2f}s "
                  f"（{ss[0][0]:.2f}〜{ss[-1][1]:.2f}s 連続）")

    # ---- S8 事前登録との突き合わせ ----
    clips = {}
    for r in rows:
        clips.setdefault(r["clip_id"].strip(), r)
    if any("区分" in r for r in rows):
        def kind(r):
            v = r.get("区分", "").strip()
            return v if v in plan else v[:1]
        got = Counter(kind(r) for r in clips.values())
        for k in sorted(plan):
            if got.get(k, 0) != plan[k]:
                warn(f"S8 区分{k}: {got.get(k,0)}本（計画 {plan[k]}本）")
        print("  S8 区分別クリップ数: "
              + " ".join(f"{k}={got.get(k,0)}" for k in sorted(plan))
              + f"  計{sum(got.get(k,0) for k in plan)}本（計画 {sum(plan.values())}本）")
        # 歩行対比は静止版・歩行版が対で揃っていなければ比較が成立しない
        walk = [r for r in clips.values() if kind(r) == "歩行"]
        if walk:
            st = Counter(r.get("状態", "").strip() for r in walk)
            if st.get("静止", 0) != st.get("歩行", 0):
                err(f"S8 歩行対比が対になっていません（静止{st.get('静止',0)}・"
                    f"歩行{st.get('歩行',0)}）。対応比較には同数が必要です")
            if len(walk) != plan.get("歩行", 20):
                warn(f"S8 歩行対比: {len(walk)}本（計画 {plan.get('歩行',20)}本）")
    if any("状態" in r for r in rows):
        st = Counter(r.get("状態", "").strip() for r in clips.values() if r.get("状態"))
        print("  S8 状態別クリップ数: " + " ".join(f"{k}={v}" for k, v in sorted(st.items())))

    # ---- 集計表示 ----
    pos = [r for r in rows if r["class"].strip() != "none"]
    scored = [r for r in pos if is_scored(r)]
    tier = Counter()
    for r in scored:
        if r["class"].strip() in DIST_CLASSES:
            m = lateral_of(r)
            tier["critical" if m is not None and m <= 1.5 else
                 ("caution" if m is not None and m <= 3.2 else "safe")] += 1
        else:
            tier["warn"] += 1
    neg_s = sum((fnum(r["t_cpa"]) or 0) - (fnum(r["t_start"]) or 0)
                for r in rows if r["class"].strip() == "none")
    print(f"\n  クリップ {len(clips)} / 行 {len(rows)}"
          f"（正例 {len(pos)}・うち採点対象 {len(scored)}）")
    print("  GT区分: " + " ".join(f"{k}={tier[k]}" for k in
                                  ("critical", "caution", "safe", "warn")))
    print(f"  クラス: " + " ".join(f"{k}={v}" for k, v in
                                   sorted(Counter(r['class'].strip() for r in pos).items())))
    print(f"  負例露出 {neg_s/60.0:.1f}分（事前登録 ≥100分）")
    if neg_s < 100 * 60 - EPS:
        warn(f"S8 負例露出が事前登録の100分に届いていません（{neg_s/60.0:.1f}分）")


def main() -> int:
    ann = Path(_arg("--ann"))
    cut = "--cut" in sys.argv
    dur = float(_arg("--dur", "10.0")) if cut else float(_arg("--dur", "1e9"))
    plan = dict(PLAN_DEFAULT)
    if _arg("--plan"):
        for kv in _arg("--plan").split(","):
            k, v = kv.split("=")
            plan[k.strip()] = int(v)
    rows = list(csv.DictReader(open(ann, encoding="utf-8-sig")))
    print(f"# 注釈検証: {ann.name}（{'切り出し後' if cut else '原録音'}・"
          f"クリップ長{dur if cut else '制限なし'}）\n")
    if not rows:
        print("行がありません")
        return 1
    validate(rows, cut, dur, plan)
    strict = "--strict" in sys.argv
    print()
    for w in warns:
        print(f"[{'ERROR(strict)' if strict else 'WARN'}] {w}")
    for e in errs:
        print(f"[ERROR] {e}")
    if errs or (strict and warns):
        print(f"\nNG: エラー{len(errs)}件・警告{len(warns)}件"
              + ("（--strict: 警告も不合格）" if strict else ""))
        return 1
    print(f"\nOK: エラーなし（警告{len(warns)}件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
