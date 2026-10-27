# -*- coding: utf-8 -*-
"""field_score_members.py — 実録の採点を「顔ぶれ」全員に同じ道具で掛ける（変更票 #73 比べる相手・#74 アブレーション・2026-10-21 Fable）。

入力 = サーバの field_eval_infer.sbatch の出力を fetch_field_preds.sh で手元に写したもの（<pred-root>/infer_field_<TAG>[_<member>]/val_all_causal.csv ＋ 来歴）。
顔ぶれ: v2final（提案法）・abl_nocausalft・abl_nohyst・abl_scratch・abl_noheight・baselineD・baselineB。無いものは飛ばして index.md に「無し」と書く。
各メンバーに:
  1. 来歴の照合: _prediction_integrity.py <csv> <N> --download --done --meta（本数 N・sha・識別子）
  2. 通知の採点 step20_realsmoke_score_v3.py（v4.3 ＋ 警告音 v9c）。v2final だけ --warn-rule hold（併記）と --notify-rule detect（規則だけの比較・#73 Q2）も。
     **baselineD は距離を出さないので通知の採点には入れない（#73・Q3）**
  3. 知覚層 step19k_field_perception.py（目印の通過時刻から軌道の正解 → _perception_eval.py）
  4. baselineD があれば、D と共通の 6 クラス {0,1,3,4,7,8} に正解と予測を絞った写し（_filter_classes.py）で _perception_eval.py（R73-04・提案法も同じクラスで）
出力: <out>/step20_<member>.md（＋ _events.csv）・perception_<member>/・commonD/・index.md（入出力の sha と命令の記録）

使い方:
  python scripts/field_score_members.py --tag f20261029 --pred-root out/_field_preds/f20261029 --n 172 \
      --ann-score out/realsmoke/ann_all.csv --ann-clips out/realsmoke/ann_all.csv --reads out/realsmoke/reads/reads_mid_A.csv \
      --scoring v30 --warn-interval out/realsmoke/reads/warn_interval_A.csv [--reads-fixed out/realsmoke/reads/reads_fixed_<日付>.csv] \
      [--sync out/realsmoke/reads/sync_A.csv] [--extend-takes <C の延長の本の原本名,…>] --out out/field_score_f20261029 [--mic-z 1.6] [--members ...] [--dry]
  --ann-score・--ann-clips: step19b の ann_all.csv（クリップの秒・eval_field_clip の名前）。負例は step19b --mode negative --append で同じファイルに足したもの
    （2026-10-25 地図 §8-4: 旧の説明「step19d の採点用 csv」は誤り＝step19d の ann_orig は録音の秒・名前も違う）
  --reads: step19l の reads_mid_<読む人>.csv（車軸の中点にずらした読み・converted = midpoint の列つき）。生の step19h の reads を渡すと
    正解が前輪の位置になるので止める（距離クラスの行に印が 1 つも無いとき・2026-10-25）。感度用の reads_front（車は front・自転車などは midpoint）は
    --reads-front を付けて渡す。印は本・クラス・track ごとに確かめ、足りない・合わない本は止めずに除いた写し（<out>/reads_checked.csv）を step19k に渡す
    （その本の正解は作らない＝欠測・理由は index に。前輪のままの座標を主の正解に混ぜない・Astra MAP25-03）
  --reads-fixed: step19m の reads_fixed_<日付>.csv（止まった音源 E2 の鳴り始め・鳴り終わり・先行分 §4）。--reads（確かめた写し）の後ろに足した写し
    <out>/reads_with_fixed.csv を step19k に渡す。固定音源（horn・backup_beep・crossing）の行だけを受ける（距離クラスの行があれば止める）。
    E2 は距離クラスではないので、上の印の検査（除く本）には掛からない（固定音源は midpoint の印が要らない）
  --sync・--extend-takes: step19k にそのまま渡す（拍手がクリップに入った本の数え・C の延長の本・#69 r31）。旧はこの道具から渡していなかった
  --warn-interval（2026-10-27・#69 r49）: field_gate_prep.py の警告音の区間の表を step19k に渡す（ベル・走りながらのクラクション E3・E2 は鳴り始め〜鳴り終わりだけを正解に）。
    読み（--reads）も field_gate_prep で足した読み（ベルの本の自転車・E3 の車）を step19l に通したものを渡す
  --scoring v30|old（必須・Astra R142-05・2026-10-27）: v30 = step20 に --warmup-v30 --min-history 4.0 を全員・全部の通知の規則（v9c・hold・detect）に付け、
    区間の表の鳴り終わりを cut_offset_s でクリップの秒へ移した写し <out>/ann_score_v30.csv（列 t_warn_end・暖機の判定だけ・t_end と通知の窓は変えない）を渡す。
    v30 では警告音の本に区間の表が要る（--no-warn-interval は使えない）。old = 旧の再現（暖機なし）
  Astra R142-04（10/27）: 注釈に警告音があるのに --warn-interval が無ければ止める（旧の再現だけ --no-warn-interval）。--reads の E3 の horn は horn_offset・
    ベルは bell_offset の印（step19l がずらした）が無い本を除いて知らせる
  --dry: 命令を印字するだけ（検査用）。--reads のファイルがあれば、上の印の検査は --dry でも行う
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MEMBERS = ("v2final", "abl_nocausalft", "abl_nohyst", "abl_scratch", "abl_noheight", "baselineD", "baselineB")
DIST_READ_CLS = {"car", "car_drive", "bicycle", "kick", "kickboard", "bike", "motorcycle"}   # step19l が車軸の中点にずらす距離クラス（reads の cls）


FIXED_READ_CLS = {"horn", "backup_beep", "crossing"}   # step19k の build が ann の固定の座標を当てる音源（--reads-fixed で受ける cls）
CAR_READ_CLS = {"car", "car_drive"}   # step19l の reads_front で front のままになるのは車だけ（自転車・キック・バイクは midpoint のまま）
WARN_MARK = {"horn": "horn_offset", "bike_bell": "bell_offset"}   # Astra R142-04（10/27）: --reads に来る警告音の読み（E3 の horn・ベル）に要る step19l の印（E2 の horn は --reads-fixed で別に足す）
WARN_ANN_CLS = {"bike_bell", "horn", "siren", "backup_beep"}      # 注釈にこれがあれば、正解を鳴り始め〜鳴り終わりに絞る区間の表（--warn-interval）が要る


def check_reads(path: Path, front_ok: bool = False):
    """reads の印（converted）を本・クラス・track ごとに確かめる（Astra MAP25-03・2026-10-25）。返り値 = (止める理由・無ければ空, 知らせの一覧, 残す行・除く本が無ければ None, 列)。
    - 主の採点: 距離クラスは全部 midpoint（step19l の reads_mid）。感度（--reads-front）: 車は front・ほかの距離クラスは midpoint（step19l の reads_front の正式な形）。
    - 止める: 距離クラスの行に印が 1 つも無い（生の step19h の reads）・主なのに車に midpoint が無く front だけ・感度なのに車に front が無い（取り違え）。
    - それ以外で、隠れでない行の印が足りない・合わない本（take・クラス・track）は、止めずにその本の行を除く（その本の正解は作らない＝欠測・理由を知らせる）。
      前輪のままの座標を主の正解に混ぜない（step19k は印を見ずに座標を使うため）。本人 10/24「厳格にしない」。"""
    import csv as _csv
    from collections import defaultdict as _dd
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd_ = _csv.DictReader(f)
        rows, fields = list(rd_), list(rd_.fieldnames or [])
    cls_of = lambda r: (r.get("cls") or "").strip().lower()
    mark = lambda r: (r.get("converted") or "").strip()
    dist = [r for r in rows if cls_of(r) in DIST_READ_CLS]
    # Astra R142-04（10/27）: 走りながらのクラクション E3 の horn は horn_offset・ベルは bell_offset の印が要る（step19l がずらした印）。足りない本は除いて知らせる
    wbad = {}
    wgroups = _dd(list)
    for r in rows:
        if cls_of(r) in WARN_MARK:
            wgroups[((r.get("take") or "").strip(), cls_of(r), (r.get("track") or "1").strip() or "1")].append(r)
    for key, rs in wgroups.items():
        vis = {mark(r) for r in rs if not (r.get("hidden") or "").strip()}
        if vis and vis != {WARN_MARK[key[1]]}:
            wbad[key] = ("印なし（step19l がずらしていない＝前輪の位置のまま）" if vis == {""} else f"印 {sorted(vis)}（期待 {WARN_MARK[key[1]]}）")
    if not dist:
        if not wbad:
            return "", [], None, fields
        kept = [r for r in rows if ((r.get("take") or "").strip(), cls_of(r), (r.get("track") or "1").strip() or "1") not in wbad]
        return "", [f"--reads {path.name}: 警告音の印の足りない本 {len(wbad)} を除いた（その本のその音の正解は作らない＝欠測）: "
                    + "; ".join(f"{k[0]} {k[1]} track{k[2]} {why}" for k, why in sorted(wbad.items())[:10])], kept, fields
    if not ({mark(r) for r in dist} - {""}):
        return (f"--reads {path.name} は距離クラスの行 {len(dist)} 行に印が 1 つも無い＝生の step19h の reads に見える → "
                "step19l の reads_mid_<読む人>.csv を渡す（生の読みだと正解が前輪の位置になる）"), [], None, fields
    car_marks = {mark(r) for r in dist if cls_of(r) in CAR_READ_CLS} - {""}
    if front_ok and car_marks and "front" not in car_marks:
        return f"--reads-front なのに {path.name} の車の行に front の印が無い（印 {sorted(car_marks)}）→ 感度は step19l の reads_front_<読む人>.csv を渡す", [], None, fields
    if not front_ok and car_marks and "midpoint" not in car_marks:
        return f"{path.name} の車の行の印が {sorted(car_marks)}（主の採点は midpoint）→ 主は reads_mid・感度の reads_front は --reads-front を付けて渡す", [], None, fields
    groups = _dd(list)
    for r in dist:
        groups[((r.get("take") or "").strip(), cls_of(r), (r.get("track") or "1").strip() or "1")].append(r)
    bad = {}
    for key, rs in groups.items():
        want = "front" if (front_ok and key[1] in CAR_READ_CLS) else "midpoint"
        vis = {mark(r) for r in rs if not (r.get("hidden") or "").strip()}
        if vis and vis != {want}:
            bad[key] = ("印なし（step19l が変換していない＝前輪の位置のまま）" if vis == {""} else f"印 {sorted(vis)}（期待 {want}）")
    if not bad and not wbad:
        return "", [], None, fields
    allbad = {**bad, **wbad}
    kept = [r for r in rows if ((r.get("take") or "").strip(), cls_of(r), (r.get("track") or "1").strip() or "1") not in allbad]
    notes = [f"--reads {path.name}: 印の足りない・合わない本 {len(allbad)} を除いた（その本の正解は作らない＝欠測）: "
             + "; ".join(f"{k[0]} {k[1]} track{k[2]} {why}" for k, why in sorted(allbad.items())[:10]) + (" …" if len(allbad) > 10 else "")]
    return "", notes, kept, fields


def warn_end_ann(ann: Path, warn: Path, dst: Path):
    """v30 の暖機（Astra R142-05・2026-10-27）: 区間の表（field_gate_prep・録音の秒）の鳴り終わりを、注釈の cut_offset_s でクリップの秒へ移して
    t_warn_end の列に入れた写しを作る（step20 の warn_end_of が暖機の判定だけに使う・t_end と通知の窓は変えない）。
    計画外（区分 G・負例露出の本に偶然入った相手・主に入らない）と scored = 0 は入れない。それ以外の警告音で区間が無い・cut_offset_s が無い行があれば止める。
    返り値 = (止める理由・無ければ空, 知らせの一覧)。"""
    import csv as _csv
    key = lambda s: Path(str(s or "").strip()).stem.lower()
    with open(warn, newline="", encoding="utf-8-sig") as f:
        W = {(key(r.get("take")), (r.get("class") or "").strip()): r for r in _csv.DictReader(f)}
    with open(ann, newline="", encoding="utf-8-sig") as f:
        rd_ = _csv.DictReader(f); rows, fields = list(rd_), list(rd_.fieldnames or [])
    miss, notes, n = [], [], 0
    for r in rows:
        cls = (r.get("class") or "").strip()
        r["t_warn_end"] = ""
        if cls not in WARN_ANN_CLS or (r.get("scored") or "1").strip() == "0" or (r.get("区分") or "").strip() in ("G", "負例露出"):
            continue
        w, off = W.get((key(r.get("orig_file")), cls)), (r.get("cut_offset_s") or "").strip()
        if w is None or not off:
            miss.append(f"{r.get('clip_id', '?')} {cls}（原本 {r.get('orig_file') or '?'}・{'区間の表に無い' if w is None else 'cut_offset_s が無い'}）"); continue
        on_c, end_c = float(w["on_s"]) - float(off), float(w["end_s"]) - float(off)
        if (r.get("t_start") or "").strip() and abs(on_c - float(r["t_start"])) > 0.5:
            notes.append(f"{r.get('clip_id')} {cls}: 区間の表の鳴り始め {on_c:.2f} s と注釈の t_start {r['t_start']} s が 0.5 s より離れている（読みの取り違えを確かめる）")
        r["t_warn_end"] = f"{end_c:.2f}"; n += 1
    if miss:
        return (f"--scoring v30: 警告音の鳴り終わり（区間の表）を注釈に移せない行 {len(miss)}: " + "; ".join(miss[:10]) + (" …" if len(miss) > 10 else "")
                + " → field_gate_prep の区間の表と step19b の注釈（orig_file・cut_offset_s）を確かめる"), []
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w_ = _csv.DictWriter(f, fieldnames=fields + ([] if "t_warn_end" in fields else ["t_warn_end"])); w_.writeheader(); w_.writerows(rows)
    return "", [f"v30 の暖機: 警告音 {n} 行の鳴り終わり（区間の表）をクリップの秒に移した写し {dst} を step20 に渡す（暖機の判定だけ・t_end と通知の窓は変えない・Astra R142-05）"] + notes


NO_NOTIFY = ("baselineD",)                    # 距離を出さない相手は通知の表に入れない（#73 Q3）
COMMON_D = "0,1,3,4,7,8"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def member_dir(root: Path, tag: str, m: str) -> Path:
    return root / (f"infer_field_{tag}" if m == "v2final" else f"infer_field_{tag}_{m}")


class Runner:
    def __init__(self, dry: bool, log: list):
        self.dry, self.log = dry, log

    def run(self, what: str, cmd: list[str]) -> int:
        self.log.append({"what": what, "cmd": cmd})
        print(("[dry] " if self.dry else "[run] ") + what + ": " + " ".join(str(c) for c in cmd), flush=True)
        if self.dry:
            return 0
        p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print(p.stdout[-800:], p.stderr[-800:], flush=True)
        self.log[-1]["rc"] = p.returncode
        return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True); ap.add_argument("--pred-root", type=Path, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ann-score", type=Path, required=True); ap.add_argument("--ann-clips", type=Path, required=True); ap.add_argument("--reads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--mic-z", default="1.6"); ap.add_argument("--members", default=",".join(MEMBERS))
    ap.add_argument("--minframe", default="40"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sync", type=Path, default=None, help="step19h の sync_<読む人>.csv（step19k に渡す）")
    ap.add_argument("--extend-takes", default="", help="C の延長の本（原本名をカンマ区切り・step19k に渡す）")
    ap.add_argument("--reads-fixed", type=Path, default=None, help="step19m の reads_fixed_<日付>.csv（E2 の鳴っていた区間・知覚層の読みに足す・先行分 §4）")
    ap.add_argument("--reads-front", action="store_true", help="感度用: step19l の reads_front（前輪のまま・印 front）を渡すとき（主の採点は付けない）")
    ap.add_argument("--warn-interval", type=Path, default=None, help="field_gate_prep.py の警告音の区間の表（step19k の --warn-interval・ベル・E3・E2 は鳴り始め〜鳴り終わりだけを正解に・#69 r49）")
    ap.add_argument("--no-warn-interval", action="store_true", help="旧の形の再現だけ: 警告音の本があっても区間の表なしで回す（ふつうは付けない・--scoring old だけ・Astra R142-04）")
    ap.add_argument("--scoring", choices=("v30", "old"), required=True,
                    help="v30 = 暖機（step20 --warmup-v30 --min-history 4.0）を全員・全部の通知の規則に付け、波形で読んだ鳴り終わりを暖機の判定に渡す（--warn-interval が要る）／"
                         "old = 旧の再現（暖機なし・最接近 7.5 s の履歴不足）。取り違えないよう必ず指定する（Astra R142-05）")
    a = ap.parse_args(argv)
    if a.scoring == "v30" and a.no_warn_interval:
        print("NG --scoring v30 に --no-warn-interval は使えない（旧の形の再現は --scoring old）"); return 3
    if a.ann_clips.is_file() and not a.warn_interval and not a.no_warn_interval:   # Astra R142-04（10/27）: 区間の表を忘れて r49 と違う正解で採点しない
        import csv as _csv
        with open(a.ann_clips, newline="", encoding="utf-8-sig") as f:
            wcls = sorted({(r.get("class") or "").strip() for r in _csv.DictReader(f)} & WARN_ANN_CLS)
        if wcls:
            print(f"NG 注釈に警告音（{wcls}）があるのに --warn-interval（field_gate_prep の区間の表）が無い → 渡す（旧の形を再現するときだけ --no-warn-interval）"); return 3
    members = [m for m in a.members.split(",") if m]
    bad = [m for m in members if m not in MEMBERS]
    if bad:
        print(f"NG 知らないメンバー: {bad}（{MEMBERS}）"); return 3
    for p in (a.ann_score, a.ann_clips, a.reads) + ((a.sync,) if a.sync else ()) + ((a.reads_fixed,) if a.reads_fixed else ()) + ((a.warn_interval,) if a.warn_interval else ()):
        if not a.dry and not p.is_file():
            print(f"NG 入力が無い: {p}"); return 3
    read_notes: list[str] = []
    reads_use = a.reads
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    if a.reads.is_file():
        why, read_notes, kept, fields = check_reads(a.reads, a.reads_front)
        if why:
            print("NG " + why); return 3
        if kept is not None:                                # 印の足りない本を除いた写しを step19k に渡す（元の reads は変えない・MAP25-03）
            import csv as _csv
            reads_use = out / "reads_checked.csv"
            with open(reads_use, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(kept)
            read_notes.append(f"step19k には除いた写し {reads_use} を渡す（元 {a.reads}）")
        for n_ in read_notes:
            print("[注意] " + n_)
    if a.reads_fixed is not None and a.reads_fixed.is_file() and a.reads.is_file():
        # 先行分 §4（2026-10-26）: 止まった音源の読みを足した写しを step19k に渡す（元の 2 つは変えない）
        import csv as _csv
        with open(reads_use, newline="", encoding="utf-8-sig") as f:
            r_main = _csv.DictReader(f); main_rows, main_cols = list(r_main), list(r_main.fieldnames or [])
        with open(a.reads_fixed, newline="", encoding="utf-8-sig") as f:
            r_fix = _csv.DictReader(f); fix_rows, fix_cols = list(r_fix), list(r_fix.fieldnames or [])
        wrong = sorted({(r.get("cls") or "").strip() for r in fix_rows} - FIXED_READ_CLS)
        if wrong:
            print(f"NG --reads-fixed {a.reads_fixed.name} に固定音源でない行がある（cls {wrong}）→ step19m の出力だけを渡す"); return 3
        cols = list(dict.fromkeys(main_cols + fix_cols))
        merged = out / "reads_with_fixed.csv"
        with open(merged, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(main_rows + fix_rows)
        n_fix_takes = len({r['take'] for r in fix_rows})
        read_notes.append(f"止まった音源の読み {a.reads_fixed}（{n_fix_takes} 本・{len(fix_rows)} 行）を足した写し {merged} を step19k に渡す")
        print("[注意] " + read_notes[-1])
        reads_use = merged
    ann_score_use = a.ann_score
    if a.scoring == "v30" and a.warn_interval and a.ann_score.is_file() and a.warn_interval.is_file():
        why, notes_w = warn_end_ann(a.ann_score, a.warn_interval, out / "ann_score_v30.csv")
        if why:
            print("NG " + why); return 3
        ann_score_use = out / "ann_score_v30.csv"
        for n_ in notes_w:
            print("[注意] " + n_)
        read_notes += notes_w
    s20_extra = ["--warmup-v30", "--min-history", "4.0"] if a.scoring == "v30" else []   # Astra R142-05: v30 は全員・全部の通知の規則に同じ暖機
    log: list = []; R = Runner(a.dry, log)
    PY = sys.executable
    present: dict[str, Path] = {}; absent: list[str] = []
    for m in members:
        d = member_dir(a.pred_root, a.tag, m); csv = d / "val_all_causal.csv"
        if csv.is_file() or (a.dry and d.is_dir()):
            present[m] = csv
        else:
            absent.append(m)
    index = [f"# 実録の採点（顔ぶれ）・{a.tag}", "", f"- 予測の置き場: `{a.pred_root}`・本数 N = {a.n}", f"- 無し（推論の出力が無い）: {absent if absent else 'なし'}",
             "- 採点の形: " + ("v30（step20 --warmup-v30 --min-history 4.0 を全員・全部の通知の規則に・警告音の暖機は波形で読んだ鳴り終わり）" if a.scoring == "v30" else "旧の再現（暖機なし）")]
    index += (["- reads: 感度用の reads_front（前輪のまま・--reads-front）"] if a.reads_front else []) + [f"- 注意: {n_}" for n_ in read_notes] + [""]
    rc_all = 0
    for m, csv in present.items():
        d = csv.parent
        # 1. 来歴
        rc = R.run(f"{m}: 来歴の照合", [PY, ROOT / "server_sde/_prediction_integrity.py", csv, a.n, "--download", "--done", d / "val_all_causal.done.json", "--meta", d / "run_meta.json"])
        rc_all |= rc
        if rc:
            index.append(f"- **{m}: 来歴の照合に失敗 → この相手の採点は無効**"); continue
        # 2. 通知の採点（D は入れない）
        if m not in NO_NOTIFY:
            rc_all |= R.run(f"{m}: step20 v9c", [PY, ROOT / "scripts/step20_realsmoke_score_v3.py", "--pred", csv, "--ann", ann_score_use, "--out", out / f"step20_{m}.md", "--events-out", out / f"step20_{m}_events.csv"] + s20_extra)
            if m == "v2final":
                rc_all |= R.run(f"{m}: step20 hold（併記）", [PY, ROOT / "scripts/step20_realsmoke_score_v3.py", "--pred", csv, "--ann", ann_score_use, "--warn-rule", "hold", "--out", out / f"step20_{m}_hold.md", "--events-out", out / f"step20_{m}_hold_events.csv"] + s20_extra)
                rc_all |= R.run(f"{m}: step20 detect（規則だけの比較・#73 Q2）", [PY, ROOT / "scripts/step20_realsmoke_score_v3.py", "--pred", csv, "--ann", ann_score_use, "--notify-rule", "detect", "--out", out / f"step20_{m}_detect.md", "--events-out", out / f"step20_{m}_detect_events.csv"] + s20_extra)
        # 3. 知覚層
        k_extra = (["--sync", a.sync] if a.sync else []) + (["--extend-takes", a.extend_takes] if a.extend_takes else [])   # 2026-10-25: 旧は渡していなかった
        k_extra += (["--warn-interval", a.warn_interval] if a.warn_interval else [])   # 2026-10-27（#69 r49）: 警告音は鳴り始め〜鳴り終わりだけを正解に
        rc_all |= R.run(f"{m}: step19k 知覚層", [PY, ROOT / "scripts/step19k_field_perception.py", "--reads", reads_use, "--ann", a.ann_clips, "--out", out / f"perception_{m}", "--mic-z", a.mic_z, "--minframe", a.minframe, "--pred", csv] + k_extra)
        index.append(f"- {m}: `{csv}`（sha {('dry' if a.dry else sha(csv)[:12])}）→ " + ("step20 ＋ " if m not in NO_NOTIFY else "（通知の表には入れない・距離なし）") + f"perception_{m}/")
    # 4. D と共通の 6 クラス
    if "baselineD" in present and "v2final" in present:
        cd = out / "commonD"; cd.mkdir(exist_ok=True)
        gt = out / "perception_v2final/gt"; man = out / "perception_v2final/manifest.csv"
        rc_all |= R.run("commonD: 正解と v2final の写し", [PY, ROOT / "scripts/_filter_classes.py", "--classes", COMMON_D, "--meta-in", gt, "--meta-out", cd / "gt", "--pred-in", present["v2final"], "--pred-out", cd / "pred_v2final.csv"])
        rc_all |= R.run("commonD: D の写し", [PY, ROOT / "scripts/_filter_classes.py", "--classes", COMMON_D, "--meta-in", gt, "--meta-out", cd / "gt", "--pred-in", present["baselineD"], "--pred-out", cd / "pred_baselineD.csv"])
        rc_all |= R.run("commonD: 知覚層（6 クラス）", [PY, ROOT / "scripts/_perception_eval.py", cd / "perception_commonD.md", "--plan", man, "--split", "eval", "--meta", cd / "gt", "--minframe", a.minframe, "--json", cd / "perception_commonD.json", f"v2final={cd / 'pred_v2final.csv'}", f"baselineD={cd / 'pred_baselineD.csv'}"])
        index.append("- commonD/: D と共通の 6 クラス {0,1,3,4,7,8} に絞った知覚層（v2final と baselineD・R73-04）")
    index += ["", "## 入力の sha256", ""]
    for p in (a.ann_score, a.ann_clips, a.reads) + ((a.sync,) if a.sync else ()) + ((a.reads_fixed,) if a.reads_fixed else ()) + ((reads_use,) if reads_use != a.reads else ()) + ((ann_score_use,) if ann_score_use != a.ann_score else ()):
        index.append(f"- `{p}`: {('dry' if a.dry or not p.is_file() else sha(p))}")
    if a.extend_takes:
        index.append(f"- C の延長の本（--extend-takes）: {a.extend_takes}")
    if a.warn_interval:
        index.append(f"- 警告音の正解の区間（--warn-interval・field_gate_prep・#69 r49）: {a.warn_interval}")
    index += ["", "## 命令の記録", ""] + [f"- {e['what']}（rc {e.get('rc', '-')}）: `{' '.join(str(c) for c in e['cmd'])}`" for e in log]
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (out / "run_log.json").write_text(json.dumps({"tag": a.tag, "n": a.n, "members_present": list(present), "members_absent": absent, "dry": a.dry,
                                                  "log": [{**e, "cmd": [str(c) for c in e["cmd"]]} for e in log]}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'DRY ' if a.dry else ''}done: 採点 {list(present)}・無し {absent} → {out / 'index.md'}" + ("" if rc_all == 0 else "（失敗あり）"))
    return 0 if rc_all == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
