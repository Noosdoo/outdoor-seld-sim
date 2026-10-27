# -*- coding: utf-8 -*-
"""Step 19d: 現場の記入用 CSV（session.csv ＋ events.csv）→ 解析入力 ann_orig.csv（2026-09-06・監査 R04）。

現場では「session.csv（セッション 1 行）」と「events.csv（テイク×イベント 1 行）」だけを書く。
このスクリプトが step19b/19c/20 が必要とする列（BASE_COLS ＋ PLAN_COLS ＋ 校正・原本の対応）に**規則で**変換する。
規則はここに 1 か所だけ書く（記入用 CSV の README と食い違ったらこちらが正）。

変換規則:
  - 一意キー = session_id × take_id。take_id は "<session_id>/<take>"。clip_id（原録音キー）は `原本` 列があればその stem（例 ZOOM0001）、無ければ "<session_id>_take<NN>"
  - 原本ファイル名 = `原本` 列（任意名可・再監査 N01）、無ければ "<session_id>_take<NN>.wav"。step19b は orig_file で音声と照合する
  - session.csv の `用途`（最終評価／調整用・既定 最終評価）: 調整用のセッションの行は ann_orig に入れず <out>_tuning.csv に出す（A′・再監査 N06）
  - 象限 → quadrant、状態 → 状態（そのまま）、区分は session.csv の区分（歩行対比の行は "歩行"）
  - 距離クラス（car_drive/kick/bike）: ラップ = 前端が肩の線を通った瞬間（最接近）。t_cpa = ラップ秒（＋ --lap-offset）、t_start = t_cpa − --pre（既定 6 s。負なら 0）
  - **拍手のずれ（2026-09-28・敵対レビュー D02）**: ラップはストップウォッチの秒で、録音ファイルの秒とは手で同時に押したぶん（0.1〜0.5 s）ずれる。
    events.csv に `拍手_録音側秒`（波形から読んだ拍手の秒・本人が確認）と `拍手_音側秒`（叩いた瞬間のラップ）が両方あるテイクは、
    その差（録音側 − ラップ）をそのテイクの全ラップに足して録音ファイルの秒に直す（テイクごとに 1 つ・一定である必要はない）。片方でも無ければ足さない
  - 警告音クラス（siren/horn/backup_beep/bike_bell/crossing）: ラップ = **音の始まり（気づいた瞬間）**。t_start = ラップ、t_cpa = ラップ + --warn-span（既定 3 s）
    → 成功の窓は [ラップ−1 s, ラップ＋4 s]、方向はこの区間の推定方位で比べる（採点器 v3）
  - class=none の行（負例）は t_start=0、t_cpa=原本の長さ（--audio-dir から読む。無ければ空欄で step19b が埋める）
  - 校正: session.csv の LAeq_dB・暗騒音区間_秒・校正ファイル名（`校正原本` 列。無ければ "<session_id>_calib.wav"）から
    calibration_id = 校正原本の stem（例 ZOOM0001）を付ける。step19 --calib raw/ZOOM0001.wav の既定 id と一致する（監査 H01）
  - 点検原本（`点検原本` 列）と `点検方式` を check_file / 点検方式 として引き継ぐ。step19b は check_file を切り出し対象から外す
  - #44: テイクの `三脚の向き`（0/180・クラクション E2 は 90/270 も・#69 r40・H25-01）を同名列で引き継ぐ。旧 CSV は特記「三脚180°」なら 180、無ければ 0。
  - 記録シート Version 123（H25-01・2026-10-25 Opus・Fable の確認待ち）: 三脚の本の `マイク_x_m`・`マイク_y_m`（マイクの現場の座標・E2 は (0, +2.0)）を mic_x_m・mic_y_m に運ぶ。
    x_m・y_m（音源）は現場の座標のまま（ここでは引かない。差は step19k で 1 回だけ）。列が無い・空の本は今までの決まり（_field_orientation.mic_field_xy）。
    `台本ID` も注釈に通す（step19c の S10・step20 の B2 の代替の数えが読む）。
    `回転後点検原本` があるテイクは check_file に優先し方式は 4方位。セッションの点検原本も除外に残す。
    音声の --yaw に三脚の 180° は加えない（設置の正面ずれだけ）。方位変換は step19k で行う。
  - pair_id は "<session_id>/<pair_id>"、trial は 区分（歩行対比は "walk"）
  - session.csv の `マイク高さ_cm`（実測・必須）→ `mic_z`[m]。推論（_causal_infer_v17.py）は HEIGHT_TABLE にこの csv（clip_id, mic_z）を渡す（v17b・2026-09-06）

使い方: python scripts/step19d_field_csv_to_ann.py --session session.csv --events events.csv --out ann_orig.csv
        [--pre 6] [--lap-offset 0] [--warn-span 3] [--audio-dir raw/]
        [--video-only] [--clap-csv out/realsmoke/clap_audio.csv] [--when-csv out/reads/when_where_A.csv]

変更票 #59 r6・Astra FR11-01／FR11-08（2026-10-12）:
  - 「いつ」（ラップ秒）は現場では押さず、帰ってから動画の目印と録音の波形から埋める。**ラップ秒が空の正例（class ≠ none）があると、
    注釈（ann_orig.csv）は書かずに終了コード 2 で止まる**（0 秒にして進めない。step19b・step19c も同じ条件で止まる）。
  - **--video-only**: 読む準備の表（<out>_video_takes.csv・step19h の入力）だけを書く（注釈は書かない・ラップ秒が空でも止まらない）。録った日の夜はこれ。
  - **--clap-csv**: step19j_clap_from_audio.py が録音から拾った拍手の**候補**（file, clap_audio_s, quality）。読む準備の表の clap_candidate_s・
    clap_candidate_quality の列に書くだけで、**clap_audio_s（同期に使う値）には入れない**（Astra NT12-01: 弱い拍手を飛ばして後のベルを採る穴）。
    clap_audio_s は events.csv の 拍手_録音側秒（本人が合成を固めたあとに波形で読んだ値・#59 r7）だけ。本人の値と候補が 0.010 s より違えば注意を出す。
  - ラップ秒は「数値で、有限で、0 より大きい」ものだけ確定とする（nan／inf／0／負は未確定・Astra NT12-02）。
  - **--when-csv**（2026-10-12・変更票 #59 今の決まり）: step19l_when_where.py の when_where_<reader>.csv（動画の目印の読みから計算した「いつ」= 車軸の中点が肩の線を
    越えた録音の秒）。ラップ秒が空の**距離クラス**の正例だけ、status = ok の t_cpa_s で埋める（拍手のずれは足さない = もう録音の秒）。欠測（status ≠ ok）は空のまま
    「未確定」に残し、step19d は止まる（理由を出す）。警告音（ベル・クラクション）は波形の鳴り始め（events.csv のラップ秒）のまま。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict
from _field_orientation import ORIENTATION_COL, CHECK_COL, tripod_heading, walker_line_m, static_mic_x, mic_field_xy, MIC_X_COL, MIC_Y_COL, SCRIPT_ID_COL

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa", "t_end", "t_end_unknown", "t_move_s", "t_move_src",
            "take_id", "pair_id", "区分", "状態", "横距離m", "n_car", "車種", "速度", "特記",
            "session_id", "orig_file", "calibration_id", "calib_file", "check_file", "点検方式", "excluded_files", "LAeq_dB", "暗騒音区間_秒", "t_start_rule", "用途", "mic_z", "mic_x_m", "mic_y_m",
            "overlap_group_id", "車両区分", "確認方法", CHECK_COL, ORIENTATION_COL, "装着", "x_m", "y_m", "高さ_m", "direction", "line_m", "走行速さ", "ベル目印_m", "機材メモ",
            "t_in10", "t_out10",          # 変更票 #69 r31 ①（2026-10-23）: 相手の中点が道に沿って ±10 に入っている時刻（step19l・検出フレーム率の分母の窓・step20 v3 が読む）
            SCRIPT_ID_COL,                # 2026-10-25（地図 §8-4）: 台本ID を通す（無いと step19c の S10 が飛び、step20 の B2 の代替の本数が 0 になっていた）
            "条件ID"]                     # 2026-10-25: 10/15 の本は台本ID が旧の番号（C-05 など）で、条件ID が対応表の新しい番号（C2-13 など）。S10 は条件ID を先に見る
CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "car_drive", "crossing", "kick", "bike", "bicycle", "none"}   # bicycle = 2026-10-08 変更票 #48（Opus の指摘）
WARN_CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "crossing"}
LONG_KINDS = {"siren", "crossing", "backup_beep"}   # 長回しで「長く続く相手」（記録シート V109: 終わりが空のまま止めると指摘する種類・#69 r21）。終わりが空なら「不明」= 始まりから録音末尾まで除外（R72-03）
QUADS = {"F", "B", "L", "R", ""}


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def read_csv(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]


def take_num(v: str) -> str:
    try:
        return f"{int(float(v)):02d}"
    except ValueError:
        return v


NUMERIC_TAKE_FIELDS = {"マイク高さ_cm", "x_m", "y_m", "高さ_m", MIC_X_COL, MIC_Y_COL}


def _canon_take_value(field: str, v: str) -> str:
    """同じテイクの一致の検査で使う値（数の欄は :g にそろえる＝ 2 と 2.0・-3 と -3.0 を同じに・敵対レビュー 軽 3・2026-10-25）"""
    if v and field in NUMERIC_TAKE_FIELDS:
        try:
            return f"{float(v):g}"
        except ValueError:
            return v
    return v


def load_when(path: Path) -> dict:
    """step19l の when_where_*.csv → {(原本の stem 小文字, track): 行}。track = events の event_id（step19d の video_takes が track に event_id を書く）"""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(Path(r.get("take", "")).stem.lower(), str(r.get("track") or "1"))] = r
    return out


def convert(sessions, events, pre: float, lap_offset: float, audio_dir: Path | None, warn_span: float = 3.0, when: dict | None = None):
    """返り値: (rows, warnings)。when = load_when() の辞書（任意・距離クラスの空のラップ秒を動画の読みの交わりで埋める）"""
    warns = []
    ses = {}
    for r in sessions:
        sid = r.get("session_id", "")
        if not sid or sid.startswith("←"):
            continue
        ses[sid] = r
    # #44: heading/check recording are take-level data. Legacy note chips may
    # appear on only one event; apply their value to every event in that take.
    take_rows = defaultdict(list)
    for r in events:
        take_rows[(r.get("session_id", ""), take_num(r.get("take_id", "")))].append(r)
    take_meta = {}
    take_values = {}
    for key, group in take_rows.items():
        vals = {}
        for field in ("装着", "マイク高さ_cm", "x_m", "y_m", "高さ_m", MIC_X_COL, MIC_Y_COL):
            raw_vals = [str(r.get(field, "") or "").strip() for r in group]
            supplied = {_canon_take_value(field, v) for v in raw_vals} - {""}   # 2026-10-25（敵対レビュー 軽 3）: 数の欄は数で比べる（2 と 2.0 を不一致にしない）
            if len(supplied) > 1:
                raise ValueError(f"{key}: 同じテイクの {field} が不一致")
            vals[field] = next((v for v in raw_vals if v), "")                  # 出力は元の書き方のまま（比べるときだけそろえる）
        take_values[key] = vals
        explicit = {tripod_heading(r) for r in group if str(r.get(ORIENTATION_COL, "") or "").strip()}
        checks = {r[CHECK_COL].strip() for r in group if r.get(CHECK_COL, "").strip()}
        if len(explicit) > 1 or len(checks) > 1:
            raise ValueError(f"{key}: 同じテイクの三脚の向き／回転後点検原本が不一致")
        heading = next(iter(explicit)) if explicit else max(tripod_heading(r) for r in group)
        take_meta[key] = (heading, next(iter(checks), ""))
    out = []
    seen = set()
    for r in events:
        sid = r.get("session_id", "")
        if not sid or sid.startswith("←") or r.get("take_id", "") == "":
            continue
        if sid not in ses:
            warns.append(f"session.csv に無い session_id: {sid}（take {r.get('take_id')}）")
            srow = {}
        else:
            srow = ses[sid]
        tk = take_num(r["take_id"])
        heading, take_check = take_meta[(sid, tk)]
        meta = take_values[(sid, tk)]
        cls = r.get("class", "")
        if cls not in CLASSES:
            warns.append(f"{sid}/take{tk}: 未知の class '{cls}'")
        quad = r.get("象限", r.get("quadrant", ""))
        if quad not in QUADS:
            warns.append(f"{sid}/take{tk}: 象限 '{quad}' は F/B/L/R")
        state = r.get("状態", "") or "静止"
        mic_cm = meta["マイク高さ_cm"] or srow.get("マイク高さ_cm", "")
        try:
            mic_z = f"{float(mic_cm) / 100.0:.3f}"
        except ValueError:
            mic_z = ""
            if sid in ses and (sid, "mic") not in seen:
                warns.append(f"{sid}: session.csv に マイク高さ_cm が無い（v17b は装着高さをモデルに入力する。実測して cm で書く）"); seen.add((sid, "mic"))
        kubun = srow.get("区分", "")
        yoto = (srow.get("用途", "") or "").strip()
        if yoto not in ("最終評価", "調整用"):
            if (sid, "yoto") not in seen:
                warns.append(f"{sid}: 用途 '{yoto}' は 最終評価／調整用 のどちらでもない → 保留（<out>_unresolved.csv）。記録紙を直す（再監査2 Q05）"); seen.add((sid, "yoto"))
            yoto = "未確定"
        kind = kubun if kubun == "負例露出" else "歩行" if (r.get("pair_id", "") or kubun == "歩行対比") else kubun
        if kubun == "歩行対比" and cls != "none" and not r.get("pair_id", ""):
            warns.append(f"{sid}/take{tk}: 区分=歩行対比 なのに pair_id が無い（静止/歩行の 2 本に同じ P 番号を付ける）")
        if kubun == "歩行対比" and cls != "none" and not (r.get("状態", "") or "").strip():
            warns.append(f"{sid}/take{tk}: 区分=歩行対比 なのに 状態（静止/歩行）が無い")
        ev = r.get("event_id", "1") or "1"
        lat_txt = (r.get("横距離m", "") or "").strip()
        if lat_txt:
            t_ = lat_txt.replace("〜", "-").replace("～", "-").replace("－", "-")
            t_ = "".join(chr(ord(c) - 0xFEE0) if "０" <= c <= "９" or c == "．" else c for c in t_)
            parts_ = t_.split("-")
            try:
                vals_ = [float(x) for x in parts_]
                ok_ = len(vals_) in (1, 2) and all(v >= 0 for v in vals_) and (len(vals_) == 1 or vals_[0] < vals_[1])
            except ValueError:
                ok_ = False
            if not ok_:
                warns.append(f"{sid}/take{tk}: 横距離m '{lat_txt}' は 2.0 か 幅 1.5-2.5 の形で（R11）")
        key = (sid, tk, ev)
        if key in seen:
            warns.append(f"{sid}/take{tk}: event_id {ev} が重複")
        seen.add(key)
        orig = r.get("原本", "") or f"{sid}_take{tk}.wav"
        clip_key = f"{sid}__{Path(orig).stem}" if r.get("原本", "") else f"{sid}_take{tk}"   # session__原本: 別セッションの同名原本と衝突しない（再監査2 Q03）
        calib_file = srow.get("校正原本", "") or f"{sid}_calib.wav"
        # 校正 id は校正原本の stem（step19 --calib の既定 calib_id と同じ）。校正原本が無い旧様式だけ <session>_calib
        calib_id = Path(calib_file).stem
        check_file = take_check or srow.get("点検原本", "")
        # 備考の「除外: a.wav b.wav」（ノートの取り消し・事象 0 件の除外）→ step19b が切り出し対象から外す（ノート監査 H02/H04）
        import re as _re
        m_ex = _re.search(r"除外[:：]\s*([^／\n]+)", srow.get("備考", "") or "")
        excluded_files = m_ex.group(1).split() if m_ex else []
        # Retain the session check too when every event uses a rotated check.
        excluded_files += [srow.get("点検原本", "")]
        excluded_files += [v[1] for k, v in take_meta.items() if k[0] == sid]
        excluded = " ".join(dict.fromkeys(f for f in excluded_files if f))
        t_end = ""                                       # #69 r18（B8・2026-10-20）: 長回しの相手の「② の終わり」（終わり秒）。無ければ空
        t_end_unknown = ""                               # R72-03: 長く続く相手の終わりが空・読めない → 不明（始まりから録音末尾まで除外・別計上）
        t_in10 = t_out10 = ""                            # #69 r31 ①（2026-10-23）: 相手の中点が ±10 に入っている時刻（--when-csv の status = ok の本だけ）
        t_move = ""; t_move_src = ""                     # #69 r23（B6・2026-10-20）: 発進 B2 の動き出した時刻（動画の最初の目印の読み・step19l の t_first_s）。R72-04: 最初の読みが出発点（∓10）のときだけ
        if cls == "none":
            t_start = "0"
            t_cpa = ""
            if audio_dir is not None:
                p = audio_dir / orig
                if p.exists():
                    import soundfile as sf
                    with sf.SoundFile(str(p)) as f:
                        t_cpa = f"{len(f) / f.samplerate:.2f}"
            if t_cpa == "":
                warns.append(f"{sid}/take{tk}: 負例の長さが分からない（--audio-dir に原本が無い）→ t_cpa 空欄。step19b --mode negative は原本から長さを取る")
            rule = "none: [0, 録音長)"
        else:
            lap = r.get("ラップ秒", r.get("t_cpa", ""))
            when_note = ""
            if when is not None and cls not in WARN_CLASSES and (lap or "").strip() == "":
                # 変更票 #59（今の決まり）: 距離クラスの「いつ」は動画の目印の読みの交わり（車軸の中点・step19l）。欠測は空のまま
                wrow = when.get((Path(r.get("原本", "")).stem.lower(), str(ev)))
                if wrow is None:
                    warns.append(f"{sid}/take{tk} event {ev} ({cls}): --when-csv に無い（step19l を通していない本か、track = event_id の対応違い）")
                elif (wrow.get("status") or "") != "ok":
                    warns.append(f"{sid}/take{tk} event {ev} ({cls}): 動画の読みの交わりが欠測: {wrow.get('reason', '')} → 時刻は未確定のまま")
                else:
                    lap = wrow.get("t_cpa_s", ""); when_note = (f"動画の目印の読みの交わり（車軸の中点・step19l・σ {wrow.get('sigma')}・L_mid {wrow.get('L_mid_m')} m・"
                                                                 f"見込み ±{wrow.get('dt_est_s')} s・{wrow.get('mic')}）")
                    t_in10, t_out10 = (wrow.get("t_in10_s") or "").strip(), (wrow.get("t_out10_s") or "").strip()      # r31 ①: ±10 に入っている時刻（録音の秒）
                    if _is_launch(r):
                        t_move, t_move_src = _launch_move(r, wrow, warns, sid, tk, ev)
            if when is not None and _is_launch(r) and (lap or "").strip() and not when_note:
                # R72-04: ラップ秒が埋まっている発進の本でも、動き出しは動画の読み（--when-csv）から取る
                wrow2 = when.get((Path(r.get("原本", "")).stem.lower(), str(ev)))
                if wrow2 is not None and (wrow2.get("status") or "") == "ok":
                    t_move, t_move_src = _launch_move(r, wrow2, warns, sid, tk, ev)
            # 拍手のずれ（テイクごと）: 拍手_録音側秒（波形・本人確認）− 拍手_音側秒（ラップ）。両方あるときだけ足す（--when-csv で埋めた秒は録音の秒なので足さない）
            clap_fix = 0.0; clap_note = ""
            cf, ca = r.get("拍手_録音側秒", ""), r.get("拍手_音側秒", "")
            if cf and ca and not when_note:
                try:
                    clap_fix = float(cf) - float(ca)
                    clap_note = f" + 拍手のずれ {clap_fix:+.2f} s（録音側 {float(cf):.2f} − ラップ {float(ca):.2f}）"
                    if abs(clap_fix) > 2.0:
                        warns.append(f"{sid}/take{tk}: 拍手のずれが {clap_fix:+.2f} s と大きい（拍手_録音側秒 と 拍手_音側秒 の写し間違い？）")
                except ValueError:
                    warns.append(f"{sid}/take{tk}: 拍手_録音側秒 '{cf}' か 拍手_音側秒 '{ca}' が数値でない → ずれを足さない")
            try:
                import math as _math
                _v = float(lap)
                if not _math.isfinite(_v) or _v <= 0.0:
                    raise ValueError(lap)                    # nan／inf／0／負は未確定（Astra NT12-02）
                tc = _v + (0.0 if when_note else lap_offset) + clap_fix          # 動画から計算した秒（when）は録音の秒なので、ラップ向けの補正は足さない（AR12-13）
                if not _math.isfinite(tc) or tc <= 0.0:
                    raise ValueError(f"{lap} + {lap_offset} + {clap_fix}")
                unfilled = False
            except ValueError:
                # 変更票 #59 r6・Astra FR11-01（2026-10-12）: ラップ秒が空の正例を 0 秒にして進めない。時刻は空のまま「未確定」にし、main は注釈を書かずに止める
                warns.append(f"{sid}/take{tk} event {ev} ({cls}): ラップ秒が空か、数値でないか、0 以下か、有限でない '{lap}' → 時刻は未確定（動画の目印と録音の波形から埋めてから通し直す・変更票 #59 r6・NT12-02）")
                tc = 0.0; unfilled = True
            if unfilled:
                t_start = t_cpa = ""
                rule = "未確定: ラップ秒が空（変更票 #59 r6: 距離クラスは動画の目印の読みから、警告音は録音の波形の鳴り始めから埋める）"
            elif cls in WARN_CLASSES:
                t_start = f"{max(0.0, tc):.2f}"; t_cpa = f"{tc + warn_span:.2f}"
                rule = f"警告音: t_start = ラップ（音の始まり）{clap_note}, t_cpa = ラップ + {warn_span:g} s"
            else:
                t_cpa = f"{tc:.2f}"
                t_start = f"{max(0.0, tc - pre):.2f}"
                rule = f"t_start = t_cpa − {pre:g} s（規則）; t_cpa = {when_note or 'ラップ秒'} {'+ ' + str(lap_offset) + ' s' if lap_offset else ''}{clap_note}"
                if _is_launch(r):
                    # 変更票 #69 r23（本人 10/20 03:34・B6）: 発進 B2 は始まり = max(最接近 − pre, 動き出し)。かける音で窓より前に出た知らせは数えない（合成の発進と同じ条件）
                    if t_move:
                        t_start = f"{max(float(t_start), float(t_move)):.2f}"
                        rule += f"; 発進: 始まり = max(最接近 − {pre:g}, 動き出し {t_move} s)（#69 r23・{t_move_src}）"
                    else:
                        t_move_src = t_move_src or "代替（動き出しの読みなし・最接近 − 6 s）"
                        rule += f"; 発進: 動き出しの読みが無い → 始まりは最接近 − {pre:g} s のまま（代替・#69 r23・R72-04）"
                        warns.append(f"{sid}/take{tk} event {ev}: 発進（B2）の動き出しが取れない（{t_move_src}）→ 始まりは最接近 − {pre:g} s のまま。代替した本数は step20 が別に数える")
        end_txt = (r.get("終わり秒") or "").strip()
        if cls != "none" and kubun in ("G", "負例露出") and not unfilled:
            # 変更票 #69 r18（本人 10/18 21:28・B8）: 長回し（区分 G／負例露出）で「本当に知らせるべき相手がいた間」= [② の秒, ② の終わりの秒]（半開区間）。
            # この区間に出た通知はまちがいに数えない（step20 がマスクの窓に t_end を使う）。相手は計画外なので主要評価にも入れない。
            # R72-03（Astra 2026-10-20）: 長く続く相手（LONG_KINDS）の終わりが空・読めない・逆順 → 「不明」= 始まりから録音末尾まで除外して別計上（短い既定の窓に落とさない）
            import math as _m
            te = None; bad = ""
            if end_txt:
                try:
                    te = float(end_txt) + (0.0 if when_note else lap_offset) + clap_fix
                    if not _m.isfinite(te):
                        bad = f"終わり秒 '{end_txt}' が有限でない"; te = None
                    elif te < tc:
                        bad = f"終わり秒 {end_txt} が始まり {tc:.2f} より前"; te = None
                except ValueError:
                    bad = f"終わり秒 '{end_txt}' が数値でない"; te = None
            if te is not None:
                t_start = f"{max(0.0, tc):.2f}"; t_cpa = t_start; t_end = f"{te:.2f}"
                rule = f"長回しの相手: [始まり {t_start}, 終わり {t_end}）（#69 r18・区間の通知はまちがいに数えない・主要評価に入れない）"
            elif cls in LONG_KINDS:
                t_start = f"{max(0.0, tc):.2f}"; t_cpa = t_start; t_end = ""; t_end_unknown = "1"
                rule = f"長回しの長い相手・終わり不明: [始まり {t_start}, 録音末尾）を除外して不明として別計上（R72-03・#69 r22「押し忘れは空のまま（不明）」）"
                warns.append(f"{sid}/take{tk} event {ev} ({cls}): 長く続く相手の終わりが{bad or '空'} → 不明として始まりから録音末尾まで除外（短い窓に置き換えない）")
            elif bad:
                warns.append(f"{sid}/take{tk} event {ev} ({cls}): {bad} → 通り過ぎる相手なので既定の窓（最接近 ± 6 s）のまま")
        elif end_txt and cls != "none":
            _e2 = cls == "horn" and (str(r.get("台本ID") or "").strip().upper().startswith("E2-") or kubun == "E")   # 先行分 §4（Fable 10/26 15:1x）: E2 の鳴り終わりは step19m が使う
            warns.append(f"{sid}/take{tk} event {ev}: 終わり秒: 長回しの相手では無視・E2 では step19m が鳴り終わりとして使う（注釈の時刻には使わない）" if (kubun not in ("G", "負例露出") and _e2)
                         else f"{sid}/take{tk} event {ev}: 終わり秒 は長回し（区分 G／負例露出）の相手だけに使う → 無視（区分 {kubun}）" if kubun not in ("G", "負例露出")
                         else f"{sid}/take{tk} event {ev}: 始まり（ラップ秒）が未確定なので 終わり秒 も使えない")
        out.append({
            "clip_id": clip_key, "event_id": ev, "trial": ("walk" if kind == "歩行" else (srow.get("区分", "") or "?")),
            "class": cls, "quadrant": quad, "t_start": t_start, "t_cpa": t_cpa, "t_end": t_end, "t_end_unknown": t_end_unknown, "t_move_s": t_move, "t_move_src": t_move_src,
            "t_in10": t_in10, "t_out10": t_out10,
            "take_id": f"{sid}/{tk}", "pair_id": (f"{sid}/{r['pair_id']}" if r.get("pair_id") else ""),
            "区分": kind, "状態": state, "横距離m": r.get("横距離m", ""), "n_car": r.get("n_car", ""),
            "車種": r.get("車種", ""), "速度": r.get("速度", ""), "特記": r.get("特記", ""),
            "overlap_group_id": r.get("overlap_group_id", ""), "車両区分": r.get("車両区分", ""), "確認方法": r.get("確認方法", ""),
            "session_id": sid, "orig_file": orig, "calibration_id": calib_id, "calib_file": calib_file,
            "check_file": check_file, "点検方式": ("4方位" if take_check else srow.get("点検方式", "")), "excluded_files": excluded,
            ORIENTATION_COL: str(heading), CHECK_COL: take_check,
            "LAeq_dB": srow.get("LAeq_dB", ""), "暗騒音区間_秒": srow.get("暗騒音区間_秒", ""), "t_start_rule": rule,
            "用途": yoto,
            "mic_z": mic_z, "装着": meta["装着"] or srow.get("装着", ""),
            # B2（#69 §00 の 4・V88-02・R97-01・2026-10-20）: 本人（マイク）の現場座標。歩行は x を動画の mic 読みから（空）・y = 本人の線。静止は x = 歩く人の静止x_m（空なら台本と向きから）・y = 本人の線（歩く比較でない本は 0,0 = 原点）
            # H25-01（2026-10-25）: 三脚の本に マイク_x_m・マイク_y_m があればそれ（E2 = (0, +2.0)）。テイクで同じ値（take_values）を使う
            **dict(zip(("mic_x_m", "mic_y_m"), mic_field_xy(dict(r, **{MIC_X_COL: meta[MIC_X_COL], MIC_Y_COL: meta[MIC_Y_COL]}), state, warns))),
            SCRIPT_ID_COL: r.get(SCRIPT_ID_COL, ""), "条件ID": r.get("条件ID", ""),
            **{k: meta[k] for k in ("x_m", "y_m", "高さ_m")},
            **{k: r.get(k, "") for k in ("direction", "line_m", "走行速さ", "ベル目印_m", "機材メモ")}})
    return out, warns


VIDEO_COLS = ["take", "video", "cls", "track", "line_m", "direction", "clap_audio_s", "clap2_audio_s", "src_z", "markers",
              "clap_candidate_s", "clap_candidate_quality", "clap_source"]   # 候補の 2 列は step19j の値（同期には使わない・NT12-01）


def load_claps(path: Path) -> dict:
    """step19j_clap_from_audio.py の csv（file, clap_audio_s, quality, candidates …）→ {原本の stem（小文字）: 行}（2026-10-12・#59 r6）"""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = (r.get("file") or "").strip()
            if name:
                out[Path(name).stem.lower()] = {k: (v or "").strip() for k, v in r.items()}
    return out


def _launch_move(r: dict, wrow: dict, warns: list, sid: str, tk: str, ev: str) -> tuple[str, str]:
    """発進（B2）の動き出し = 動画の最初の目印の読み（step19l の t_first_s）。R72-04: 最初の読みが出発点の十字（後→前 = −10・前→後 = +10）でなければ使わない（代替・理由を残す）。"""
    tf = (wrow.get("t_first_s") or "").strip(); xf = (wrow.get("x_first_m") or "").strip()
    if not tf:
        return "", "代替（動画の読みに t_first_s が無い）"
    d_ = (r.get("direction") or "").strip().replace("->", "→")
    expect = -10.0 if d_.startswith("後") else (10.0 if d_.startswith("前") else None)
    if expect is None:
        return "", "代替（向き direction が無く出発点を決められない）"
    if not xf:
        return "", "代替（動画の読みに x_first_m が無い・step19l を通し直す）"
    if abs(float(xf) - expect) > 0.05:
        return "", f"代替（最初の読み x = {xf} m が出発点 {expect:+g} m でない・動き出しを読めていない）"
    return f"{float(tf):.2f}", f"動画の最初の目印（出発点 {expect:+g} m を離れた時刻）"


def _is_launch(r: dict) -> bool:
    """発進（B2）の本か: 台本ID が B2- で始まるか、区分 B（旧様式）。"""
    sid = (r.get("台本ID") or "").strip().upper()
    return sid.startswith("B2") or (not sid and (r.get("区分") or "").strip() == "B")


def markers_for_take(r: dict, direction: str | None = None) -> str:
    """Astra R97-05（B5・2026-10-20）: 目印の点を本ごとに決める（step19h の takes CSV の markers 列）。
    10/22 の自転車の本（台本ID C2-／H2-・歩く比較 W01〜W08）と車の日の A2（#69 r38・Astra CAR24-06・2026-10-25 Opus・Fable の確認待ち）= ±12 の 13 点。
    10/15 の本（台本ID 無し・旧の C-／H-）と車の日の B2／E2・W09〜W16（7.0 m の線も）= ±10 の 11 点（B2 は ±10 に止まって出る・E2 は止まって鳴らす・W09〜W16 は静止を 2 m ずらして ±10 に入れてある）。
    全部の既定を ±12 にはしない。知覚の正解の範囲（検出フレーム率の分母）は ±10 のまま（r31）。
    並び（変更票 #69 r33・2026-10-23）: **出発側から**。events の direction は記録シートが現場の向きに直してある（三脚 180° の本は台本の向きを裏返して書く）ので、
    「後→前」= 北（−）から南（+）= −n … +n、「前→後」= 南（+）から北（−）= +n … −n。三脚の向きでもう一度裏返さない。歩く人の行（mic）も同じ決まり（direction を渡す）。
    step19h は markers が与えられた本をその並びのまま使う（direction での並べ替えは既定の目印のときだけ）。"""
    sid = (r.get("台本ID") or "").strip().upper()
    w = sid[1:3] if sid.startswith("W") and sid[1:3].isdigit() else ""
    thirteen = sid.startswith(("C2", "H2", "A2")) or (w != "" and 1 <= int(w) <= 8)
    n = 12 if thirteen else 10
    d = (direction if direction is not None else (r.get("direction") or "")).strip().replace("->", "→")
    if "後" in d and "前" in d and d.index("後") < d.index("前"):          # 後→前 = x が増える向き（北 − → 南 +）
        return ",".join(str(x) for x in range(-n, n + 1, 2))
    return ",".join(str(x) for x in range(n, -n - 1, -2))                    # 前→後（既定）= x が減る向き（南 + → 北 −）


def load_ledger(path: Path) -> dict:
    """採用の台帳（raw/ledger/ledger_YYYYMMDD.csv・列 録音,開始,秒,動画,時刻が重なる動画,役割）→ {録音ファイル名(小文字): 役割}（V88-03・B1・2026-10-20）。
    役割が「本 …」の録音だけを評価用に変換する。「使わない …」「校正」「点検 …」は理由つきで別に出す。"""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = (r.get("録音") or "").strip()
            if k:
                out[k.lower()] = (r.get("役割") or "").strip()
    return out


def apply_ledger(events: list, roles: dict):
    """events を台帳で採用（役割が「本」で始まる）と使わないに分ける。台帳に無い原本は使わない側（役割 = 台帳に無い）。"""
    kept, dropped = [], []
    for r in events:
        orig = (r.get("原本") or "").strip()
        if not orig:
            kept.append(r); continue                      # 原本が無い行（見出しの行など）は今までどおり convert に任せる
        role = roles.get(Path(orig).name.lower())
        if role is None:
            dropped.append({**r, "台帳の役割": "台帳に無い"})
        elif role.startswith("本"):
            kept.append(r)
        else:
            dropped.append({**r, "台帳の役割": role})
    return kept, dropped


def video_takes(events, claps=None, notices=None):
    """Prepare step19h configuration in FIELD coordinates (exported by the note).

    clap_audio_s is the measured recording time, never the stopwatch lap.
    Mic and counterpart have separate rows; stationary sounds still need two
    measured sounding endpoint times in reads, not a moving-marker template.
    claps（任意）= load_claps() の辞書。step19j の候補を clap_candidate_s／clap_candidate_quality に写す（clap_audio_s には入れない・NT12-01）。
    notices（任意）= 候補と本人の値の食い違い・候補の無い本の説明を足すリスト。
    """
    result, mic_rows = [], {}
    for r in events:
        if r.get("class") not in ("car_drive", "kick", "bike", "bicycle", "bike_bell", "horn"):   # bicycle = 2026-10-08・horn = 走りながらのクラクション E3（10/27・下の行で向きと線のある動く本だけ通る＝止めた車の E2 は向きも線も空なので入らない）
            continue
        if not r.get("動画原本") or not r.get("direction") or not r.get("line_m"):
            continue
        take = r.get("原本", "")
        cs = (r.get("拍手_録音側秒") or "").strip()          # 同期に使う値 = 本人が波形で読んだ値だけ（#59 r7・NT12-01）
        cand, quality, source = "", "", ("events" if cs else "")
        if claps is not None:
            c = claps.get(Path(take).stem.lower()) if take else None
            if c is None:
                if notices is not None:
                    notices.append(f"{take}: 拍手の自動検出（--clap-csv）に無い（step19j を通していない本）")
            else:
                cand, quality = c.get("clap_audio_s", ""), (c.get("quality", "") or "不明")
                if cs and cand:
                    try:
                        if abs(float(cs) - float(cand)) > 0.010 and notices is not None:
                            notices.append(f"{take}: 本人の読み 拍手_録音側秒 {cs} と自動検出の候補 {cand}（{quality}）が 0.010 s より違う → 本人の値を使う。確かめる")
                    except ValueError:
                        pass
                elif not cs and notices is not None:
                    notices.append(f"{take}: clap_audio_s は空（本人が合成を固めたあとに波形で読む）。自動検出の候補 {cand or '—'}（{quality}）は突き合わせ用で同期には使わない")
        row = dict(take=take, video=r["動画原本"], cls=r["class"],
                   track=r.get("event_id") or "1", line_m=r["line_m"], direction=r["direction"],
                   clap_audio_s=cs, clap2_audio_s="", src_z=r.get("高さ_m", ""),
                   markers=markers_for_take(r),                                   # R97-05（B5・2026-10-20）: 本ごとの目印の点
                   clap_candidate_s=cand, clap_candidate_quality=quality, clap_source=source)
        result.append(row)
        if r.get("状態") == "歩行":
            # 変更票 #69 r33（2026-10-23）: 歩く人の行の向きも events の direction（現場の向き・歩く人は相手と同じ向きに進む = 追い越される）。
            # 旧は三脚の向きから作っていて、三脚 180° の本で逆になっていた（Opus の指摘・10/15・10/22 の表は Opus が並べ直した）
            mic = dict(row, cls="mic", track="0", line_m=walker_line_m(r), src_z="",          # B2: 本人の線（歩く人の線_m・既定 5.0・車 5 m は 7.0）。旧は 0 に決め打ち
                       direction=r["direction"], markers=markers_for_take(r, r["direction"]))
            key = (r.get("session_id"), row["take"])
            if key in mic_rows and mic_rows[key] != mic:
                raise ValueError("同じテイクの mic 用動画・同期が不一致")
            mic_rows[key] = mic
    return result + list(mic_rows.values())


def main() -> int:
    sessions = read_csv(Path(_arg("--session")))
    events = read_csv(Path(_arg("--events")))
    out_path = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "ann_orig.csv")))
    if _arg("--ledger"):                                       # V88-03（B1・2026-10-20）: 採用の台帳で採用した行だけを評価用に変換する
        lp = Path(_arg("--ledger"))
        if not lp.is_file():
            print(f"❌ --ledger が無い: {lp}（raw/ledger/ledger_YYYYMMDD.csv）"); return 2
        events, dropped = apply_ledger(events, load_ledger(lp))
        dpath = out_path.with_name(out_path.stem + "_ledger_excluded.csv")
        dpath.parent.mkdir(parents=True, exist_ok=True)
        cols = list(dict.fromkeys([k for r in dropped for k in r.keys()])) or ["台帳の役割"]
        with open(dpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(dropped)
        print(f"採用の台帳 {lp.name}: 採用 {len(events)} 行・使わない {len(dropped)} 行（理由つき）-> {dpath}")
    pre = float(_arg("--pre", "6"))
    lap_offset = float(_arg("--lap-offset", "0"))
    audio_dir = Path(_arg("--audio-dir")) if _arg("--audio-dir") else None
    warn_span = float(_arg("--warn-span", "3"))
    video_only = "--video-only" in sys.argv                      # 2026-10-12 #59 r6: 読む準備の表だけ（録った日の夜）
    clap_csv = Path(_arg("--clap-csv")) if _arg("--clap-csv") else None
    when = None
    if _arg("--when-csv"):
        wp = Path(_arg("--when-csv"))
        if not wp.is_file():
            print(f"❌ --when-csv が無い: {wp}（step19l_when_where.py で作る）"); return 2
        when = load_when(wp)
    convert_error = ""
    try:
        rows, warns = convert(sessions, events, pre, lap_offset, audio_dir, warn_span, when)
    except ValueError as ex:
        if not video_only:
            raise
        # 2026-10-25（敵対レビュー 軽 2）: 夜の --video-only は読む準備の表だけ。注釈用の検査（マイクの欄の片方だけ・三脚の向きなど）で止めず、知らせにして表は書く
        rows, warns, convert_error = [], [], str(ex)
    video_path = out_path.with_name(out_path.stem + "_video_takes.csv")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_notices = []  # Paths may be filled in after audio annotation; not conversion failures.
    claps = None
    if clap_csv is not None:
        if not clap_csv.is_file():
            print(f"❌ --clap-csv が無い: {clap_csv}（step19j_clap_from_audio.py --dir raw/ --out <csv> で作る）"); return 2
        claps = load_claps(clap_csv)
    video_rows = video_takes(events, claps, video_notices)
    for r in video_rows:
        video = Path(r["video"])
        if audio_dir is not None and not video.is_absolute():
            candidate = audio_dir / "video" / video
            if candidate.is_file():
                r["video"] = str(candidate.resolve())
            else:
                video_notices.append(f"動画が見つかりません: {candidate} → step19h の前に {video_path.name} の video 列を実際のパスに直してください")
    with video_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_COLS); w.writeheader(); w.writerows(video_rows)
    print(f"動画読みの設定 -> {video_path}（動画パスを配置先に合わせる。clap_audio_s = 拍手_録音側秒 = 本人が波形で読んだ値だけ。--clap-csv の候補は clap_candidate_s に写すだけで同期には使わない・NT12-01。空の本は step19h が出力しない）")
    if video_only:
        if convert_error:
            print(f"[VIDEO NOTICE] 注釈を作るときに止まる所がある（夜の読む準備の表には関係しない）: {convert_error} → 本読みのあとに注釈を作る前に記録を直す")
        for notice in video_notices:
            print("[VIDEO NOTICE]", notice)
        n_clap = sum(1 for r in video_rows if r.get("clap_audio_s")); n_cand = sum(1 for r in video_rows if r.get("clap_candidate_s"))
        print(f"--video-only: 読む準備の表だけ書いた（注釈 {out_path.name} は書かない・変更票 #59 r6）: {len(video_rows)} 行（同期の拍手の秒あり {n_clap}・自動検出の候補あり {n_cand}） -> {video_path}")
        print("次: python scripts/step19h_video_marks.py --takes <上の csv> --out out/realsmoke/reads --reader A（夜は 1〜2 本だけ読めるか確かめる・本読みは合成を固めたあと・#59 r7）")
        return 0
    # 変更票 #59 r6・Astra FR11-01（2026-10-12）: ラップ秒が空の正例があれば注釈を書かずに止める（0 秒で切り出すと相手の音がクリップから消え、本数の検査も通ってしまう）
    unfilled = [r for r in rows if r.get("class") != "none" and (not r.get("t_cpa") or not r.get("t_start"))]
    if unfilled:
        for wmsg in warns:
            print("[WARN]", wmsg)
        for notice in video_notices:                       # 動画の不在の知らせは、注釈で止まるときも出す（test_step19d_video_notice・2026-10-20）
            print("[VIDEO NOTICE]", notice)
        print(f"❌ 時刻が未確定の正例が {len(unfilled)} 行あります → 注釈（{out_path.name}）は書きません（終了コード 2・変更票 #59 r6・Astra FR11-01）")
        for r in unfilled[:30]:
            print(f"   {r['take_id']} event {r['event_id']} {r['class']} 原本 {r['orig_file']}")
        print("   距離クラス（車・自転車）: 動画の目印の読み → 「いつ」の計算で ラップ秒 を埋める。警告音（ベル・クラクション）: 合成を固めたあとに録音の波形の鳴り始めを読んで埋める。")
        print("   読む準備の表だけ要るときは --video-only（録った日の夜）。")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tuning = [r for r in rows if r.get("用途") == "調整用"]
    unresolved = [r for r in rows if r.get("用途") == "未確定"]
    rows = [r for r in rows if r.get("用途") == "最終評価"]
    if unresolved:
        upath = out_path.with_name(out_path.stem + "_unresolved.csv")
        with open(upath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(unresolved)
        print(f"用途が未確定の {len(unresolved)} 行は評価用にも調整用にも入れず保留 -> {upath}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(rows)
    # セッション表（<out>_sessions.csv）: 事象が 0 行のセッションでも校正・点検・除外の原本名を step19b に渡す（ノート監査 H01/H02）
    spath = out_path.with_name(out_path.stem + "_sessions.csv")
    # session_no（2026-09-22・実録クリップの命名 A）: session.csv の登録順に 1, 2, … を振り、既にある表の番号は動かさない（後から足したセッションは末尾の番号）
    prev_no = {}
    if spath.exists():
        with open(spath, encoding="utf-8-sig", newline="") as fh:
            for r0 in csv.DictReader(fh):
                if (r0.get("session_no") or "").strip():
                    prev_no[r0["session_id"]] = int(r0["session_no"])
    next_no = max(prev_no.values(), default=0) + 1
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "session_no", "calib_file", "calibration_id", "check_file", "点検方式", "excluded_files", "用途"]); w.writeheader()
        import re as _re2
        for srow in sessions:
            sid = srow.get("session_id", "")
            if not sid:
                continue
            if sid not in prev_no:
                prev_no[sid] = next_no; next_no += 1
            cf = srow.get("校正原本", "") or f"{sid}_calib.wav"
            m_ex = _re2.search(r"除外[:：]\s*([^／\n]+)", srow.get("備考", "") or "")
            w.writerow({"session_id": sid, "session_no": prev_no[sid], "calib_file": cf, "calibration_id": Path(cf).stem, "check_file": srow.get("点検原本", ""),
                        "点検方式": srow.get("点検方式", ""), "excluded_files": " ".join(m_ex.group(1).split()) if m_ex else "",
                        "用途": srow.get("用途", "") or "最終評価"})
    if tuning:
        tpath = out_path.with_name(out_path.stem + "_tuning.csv")
        with open(tpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(tuning)
        print(f"用途=調整用 {len(tuning)} 行は最終評価の入力に含めず -> {tpath}")
    for wmsg in warns:
        print("[WARN]", wmsg)
    for notice in video_notices:
        print("[VIDEO NOTICE]", notice)
    n_take = len({(r["session_id"], r["take_id"]) for r in rows})
    print(f"{len(rows)} 行 / {n_take} テイク / {len({r['session_id'] for r in rows})} セッション -> {out_path}")
    print("次: step19 --calib raw/<校正原本> --laeq <LAeq_dB> --gain-only --out <calib-dir>（校正 id = 校正原本の stem）→ step19b --calib-dir <calib-dir> → step19c --ann ... → step20 v3")
    print("#44: --yaw は設置の正面ずれの補正だけ。三脚の向き 180° は音に足さず、step19k の正解に使う。")
    return 1 if warns else 0


if __name__ == "__main__":
    sys.exit(main())
