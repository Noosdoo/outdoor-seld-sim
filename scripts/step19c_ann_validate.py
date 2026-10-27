# -*- coding: utf-8 -*-
"""Step 19c: 実録注釈CSVのスキーマ検証（2026-08-14 実録再設計レビュー反映）。

採点(step20)に回す前に、注釈が採点可能な形になっているかを機械チェックする。
現場で作った注釈の取りこぼしを、帰宅後その日のうちに潰すための道具。

検査項目:
  S1 必須列の存在（計画列 take_id / 区分 / 状態を含む。--cut 指定時は
     orig_file / orig_duration_s / cut_offset_s / scored も必須）
  S2 (clip_id, event_id) の一意性
  S3 class が既知クラス∪none／quadrant が F/B/L/R（正例のみ）
  S4 t_start ≤ t_cpa、いずれも [0, --dur] の範囲内
  S5 距離クラス(car_drive/kick/bike/bicycle)の採点対象行に有限・非負の 横距離m がある（単一値 2.0 か 幅 1.5-2.5・R11）。負例の本（G・負例露出）に偶然入った相手の行は S3/S5 の対象外（計画外・2026-10-08）
  S6 cut_offset_s / orig_duration_s が有効、orig_file が非空、scored が0/1
     （**負例行も含め全行**）
  S7 **負例の担当区間が原録音上で先頭0sから末尾まで隙間なく・重なりなくタイルすること**
     （step19b --mode negative の重複分割が正しく効いているかの一発検査。
       ここが崩れていると誤警告の二重計上/取りこぼしが起きる）
  S8 事前登録との突き合わせ（切り出しclip数ではなく take_id 単位）。
  S10 台本ID（条件 ID）の照合（B1・Astra V88-03・2026-10-20）: 台本に無い ID・同じ ID の重複（録り直しの前の本）・線・速さ・向き・左右の取り違えを誤りに。
      未収録の ID は一覧に出す（当日分は警告にしない・--strict で全日程の完了を要求）。--no-script-check で飛ばす。--selftest で自己検査。
     区分A〜E・歩行の本数、歩行対比の pair_id ごとの静止/歩行1本ずつを検査。
     100分連続負例は区分=負例露出とし、200本の計画数から分離する

使い方:
  python scripts/step19c_ann_validate.py --ann out/realsmoke/ann_all.csv --cut
  python scripts/step19c_ann_validate.py --ann ann_orig.csv          # 原録音注釈
  python scripts/step19c_ann_validate.py --ann ann_all.csv --cut --strict  # 収録完了後
  （--plan で区分ごとの計画本数を上書き。既定は 2026-08-22 改訂の A〜E各20＋Fバイク10＋歩行100＝計210本。
    --strict は本数不足などの警告も不合格にする＝全収録完了後の最終ゲート用）
  2026-09-06 再監査 N03: 区分の内訳は --subplan "D:crossing=8,backup_beep=4,horn=4,siren<=4" で検査する（必須 '=' はちょうど、任意 '<=' は 0〜上限）。
    D 全体を --opportunity にすると必須 16 本の不足を見逃すので、D には --subplan を使う
  2026-09-06（監査 R04/W8）: take_id は session_id 列があれば (session_id, take_id) の組で一意（別セッションの take 1 が衝突しない）。
    --cut のとき calibration_id が空の行は警告（校正の引継ぎ）。--opportunity D=8 のように機会枠を指定すると、
    その区分の不足は「上限目標に届かず」の情報表示だけで警告にしない（サイレン等の遭遇枠。0 本でも評価は成立）。
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa",
             "take_id", "区分", "状態"]
CUT_COLS = ["orig_file", "orig_duration_s", "cut_offset_s", "scored"]
CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "car_drive", "crossing",
           "kick", "bike", "bicycle"}   # bicycle = 2026-10-08 変更票 #48
DIST_CLASSES = {"car_drive", "kick", "bike", "bicycle"}   # bicycle = 2026-10-08 変更票 #48
WARN_CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "crossing"}   # 2026-10-12 #59 r6: t_start = 鳴り始め（0 は未確定の印）
QUADS = {"F", "B", "L", "R"}
LATERAL_KEYS = ("横距離m", "横距離", "lateral_m")
PLAN_DEFAULT = {"A": 32, "B": 4, "C": 16, "D": 8, "E": 12, "H": 60, "G": 20, "歩行": 32}   # 2026-10-27: E = E2 8 ＋ 走りながらのクラクション E3 4（本人 14:13 案 A）   # 2026-10-08 変更票 #50（F キック → H 自転車）・2026-10-14 #68（左右をそろえて A 22→24・B 5→8・H 24→28）・2026-10-17 #69（条件ごとに 1 本ずつ・学内 168 本）。D はやめた（#56）が既定には 8 を残す＝命令に D=0 を書く（AS10-01 の試験の前提）。命令の --plan が上書きする
OPPORTUNITY = set()      # --opportunity で指定した機会枠の区分（不足を不合格にしない）
# --subplan "D:crossing=8,backup_beep=4,horn=4,siren<=4": 区分の内訳。'=' は必須（ちょうど）、'<=' は任意（0〜上限）。
# 区分の総数は [必須の和, 必須の和＋任意の上限の和] を許容（再監査 N03: D 全体を機会枠にしない）
SUBPLAN = {}             # kind -> {cls: (kind_of_constraint, n)}   # 2026-09-05: F バイク 10 本を追加（ハンドブック 2026-08-22 改訂の計 210 本）
EXPOSURE_KIND = "負例露出"
UNPLANNED_KINDS = {"G", EXPOSURE_KIND}   # 負例の本の区分。ここに偶然入った相手の行は計画外（象限・横距離を求めない・2026-10-08）
STATES = {"静止", "歩行"}
EPS = 1e-6

errs, warns = [], []
SCRIPT_ID_COL = "台本ID"
E2_SIDE = {"前": 180, "左": 90, "後": 0, "右": 270}   # #69 r40（記録シート V122〜）: クラクション E2 の三脚の向き
E2_DIST_TOL_M = 0.75                                  # E2 の音源とマイクの水平距離が台本（3／7 m）からこれ以上ずれたら取り違え（置き直しの小さなずれは通す）
E2_LINE_TOL_M = 0.30                                  # E2 は音源もマイクも 2.0 m の線の上（y が同じ）。これ以上違えば警告（H25-01）・「0 とみなす」幅にも使う
E2_MIX_DY_M = 1.0                                     # 取り違え（マイクの座標の抜け・原点の二重引き）は y の差が約 2.0 m で片方が 0。この差を超えて片方が 0 なら誤り（敵対レビュー 中 2・2026-10-25）


def expected_takes() -> dict:
    """実録の台本（`md/design/実録_全168本の台本_2026-10-17.md`・#69 r14・学内 172 本）の 台本ID → 条件（B1・Astra V88-03・2026-10-20 Fable）。
    番号の決まり: 奇数 = 左（三脚 0°）・偶数 = 右（180°）。4 本の並びは 後→前・左 → 後→前・右 → 前→後・左 → 前→後・右。
    A2: 時速 15 km は 01〜16（2.0／2.5／3.0／5.0 m × 4）・時速 10 km は 17〜32。B2: 2.0 m × 4。C2: 1.0 m（ベル 10 m 01〜04・5 m 05〜08）・3.0 m（10 m 09〜12・5 m 13〜16）。
    E2: 3 m 01〜04・7 m 05〜08（前・左・後・右）。E2 だけ三脚の向きで前後左右を作る（#69 r40: 車は 2.0 m の線の北に南向き・マイクは (0, +2.0)）= 前 180・左 90・後 0・右 270。H2: 線 1.0／1.5／2.0／3.0／5.0 × 12（ゆっくり・ふつう・速い × 4 本）。G: 20。W01〜16: 静止・歩行の 2 本ずつ（奇数 = 相手が歩く人の左 = 180°・偶数 = 右 = 0°）。"""
    E = {}
    four = [("後→前", 0), ("後→前", 180), ("前→後", 0), ("前→後", 180)]
    for n in range(1, 33):
        g, k = (n - 1) // 4, (n - 1) % 4
        E[f"A2-{n:02d}"] = dict(kind="A", line=[2.0, 2.5, 3.0, 5.0][g % 4], speed="15" if n <= 16 else "10", direction=four[k][0], side=four[k][1])
    for n in range(1, 5):
        E[f"B2-{n:02d}"] = dict(kind="B", line=2.0, direction=four[n - 1][0], side=four[n - 1][1])
    for n in range(1, 17):
        g, k = (n - 1) // 4, (n - 1) % 4
        E[f"C2-{n:02d}"] = dict(kind="C", line=1.0 if n <= 8 else 3.0, bell=10 if g in (0, 2) else 5, direction=four[k][0], side=four[k][1])
    for n in range(1, 9):
        pos = ["前", "左", "後", "右"][(n - 1) % 4]
        E[f"E2-{n:02d}"] = dict(kind="E", dist=3.0 if n <= 4 else 7.0, position=pos, side=E2_SIDE[pos])   # 2026-10-25（地図 §8-3）: 旧は「右 = 180・ほか 0」のままだった
    for n in range(1, 5):   # 2026-10-27（本人 14:13 案 A）: 走りながらのクラクション E3 = A2 と同じ車の通過（2.0 m・時速 15 km）で手前の ∓8 で 1 回鳴らす（区分 E・並びは A2 と同じ）
        E[f"E3-{n:02d}"] = dict(kind="E", moving=True, line=2.0, speed="15", direction=four[n - 1][0], side=four[n - 1][1])
    for n in range(1, 61):
        g, j = (n - 1) // 12, (n - 1) % 12
        E[f"H2-{n:02d}"] = dict(kind="H", line=[1.0, 1.5, 2.0, 3.0, 5.0][g], speed=["ゆっくり", "ふつう", "速い"][j // 4], direction=four[j % 4][0], side=four[j % 4][1])
    for n in range(1, 21):
        E[f"G-{n:02d}"] = dict(kind="G")
    for n in range(1, 17):
        cp = ("bike_bell", 3.0) if n <= 4 else ("bicycle", 2.0) if n <= 8 else ("car_drive", 3.0) if n <= 12 else ("car_drive", 5.0)
        E[f"W{n:02d}"] = dict(kind="歩行", counterpart=cp[0], lateral=cp[1], side=180 if n % 2 == 1 else 0, takes=2)
    return E


def _lead_num(v: str):
    """「15」「15 km/h」「時速15km」のような文字列の最初の数字"""
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)", str(v or ""))
    return float(m.group(1)) if m else None


import re as _re_mod
_OLD_SCRIPT_ID = _re_mod.compile(r"^[A-H]-\d{2}$")   # 10/15 の旧の台本の番号（C-01・H-03 など・#69 の前）。新しい台本は A2-・B2-・C2-・E2-・H2-・G-・W


def _split_script_state(sid: str) -> tuple[str, str]:
    """記録シートの歩く比較の台本ID は「W01-静止」「W01-歩行」（組 W01 の 2 本・HTML の台本の定義）。台本の表は組の番号（W01・2 本）なので、
    状態を分けて返す（「W01」だけの書き方もそのまま通す）。2026-10-25: 台本ID を注釈に通したあと、本物の 10/22 の歩く比較 16 本が「台本に無い」になっていた"""
    import re as _re
    m = _re.match(r"^(W\d{2})-(静止|歩行)$", sid)
    return (m.group(1), m.group(2)) if m else (sid, "")


def _norm_dir(v: str) -> str:
    v = (v or "").strip().replace("->", "→").replace("後ろ", "後")
    return {"back_to_front": "後→前", "front_to_back": "前→後"}.get(v, v)


def check_script_ids(rows, plan, strict_cover: bool = False):
    """S10（B1・V88-03）: 台本ID の集合・重複・実際の条件（線・速さ・向き・左右）の照合。台本ID が 1 行も無ければ飛ばす（旧様式）。"""
    E = expected_takes()
    by_take = defaultdict(list)
    for r in rows:
        tid = r.get("take_id", "").strip(); sid = r.get("session_id", "").strip()
        if not tid:
            continue
        if sid and not tid.startswith(sid):
            tid = f"{sid}/{tid}"
        by_take[tid].append(r)
    if not any((r.get(SCRIPT_ID_COL) or "").strip() for r in rows):
        print(f"  S10 {SCRIPT_ID_COL} の列が無い／全部空 → 条件 ID の照合は飛ばす（10/15 の本は対応表で 台本ID を埋める）"); return
    seen = defaultdict(list)
    for tid, rs in sorted(by_take.items()):
        # 2026-10-25: 条件ID（記録シートの対応表の新しい番号）があればそれを先に見る。10/15 の本は台本ID が旧の番号（C-05 など）で、条件ID が C2-13 など。
        # 新しい台本の本は条件ID = 台本ID。旧の番号で条件ID が空の本は、対応表で数えない本（C-01〜04・H-07・08・11・12）
        ids = {((r.get("条件ID") or "").strip() or (r.get(SCRIPT_ID_COL) or "").strip()).upper() for r in rs} - {""}
        if len(ids) > 1:
            err(f"S10 {tid}: 同じテイクの {SCRIPT_ID_COL}／条件ID が不一致 {sorted(ids)}"); continue
        if not ids:
            warn(f"S10 {tid}: {SCRIPT_ID_COL} が空（対応表で埋めるか、評価に使わないなら台帳で外す）"); continue
        sid_full = next(iter(ids)); sid_, st_ = _split_script_state(sid_full); exp = E.get(sid_)
        if exp is None and _OLD_SCRIPT_ID.match(sid_full):          # 新しい G-01〜20 は台本の表にあるのでここに来ない
            warn(f"S10 {tid}: 旧の台本の番号 '{sid_full}' で条件ID が空＝対応表で数えない本（評価に使わないなら台帳で外す）"); continue
        if exp is None:
            err(f"S10 {tid}: 台本に無い {SCRIPT_ID_COL} '{sid_full}'"); continue
        seen[sid_].append((tid, st_))
        r0 = rs[0]
        st_row = (r0.get("状態") or "").strip()
        if st_ and st_row and st_row != st_:
            err(f"S10 {tid} {sid_full}: 状態 '{st_row}' が台本ID の {st_} と違う")
        pos = [r for r in rs if r["class"].strip() != "none"]
        kind = r0.get("区分", "").strip()
        if exp["kind"] != kind and not (exp["kind"] == "歩行" and kind == "歩行"):
            err(f"S10 {tid} {sid_}: 区分 '{kind}' が台本の {exp['kind']} と違う")
        if "line" in exp and pos:
            got = lateral_of(pos[0]); lm = fnum(pos[0].get("line_m", ""))
            if lm is not None and abs(lm - exp["line"]) > EPS:
                err(f"S10 {tid} {sid_}: 線 line_m {lm:g} が台本の {exp['line']:g} m と違う（取り違え）")
            elif lm is None and got is not None and abs(got - exp["line"]) > EPS:
                err(f"S10 {tid} {sid_}: 横距離m {got:g} が台本の {exp['line']:g} m と違う（取り違え）")
        if "direction" in exp and pos:
            d = _norm_dir(pos[0].get("direction", ""))
            # 2026-10-25（敵対レビュー 重大 1）: 注釈の direction は現場の向き（記録シートの fieldDirection・#69 r33）。三脚 180° の本は
            # 台本の向き（マイクから見た向き）を裏返して書いてあるので、同じ決まりで戻してから比べる（旧は 180° の本を全部「向き違い」にしていた）
            h = fnum((r0.get("三脚の向き") or "").strip().removesuffix("°"))
            d_mic = {"前→後": "後→前", "後→前": "前→後"}.get(d, d) if h is not None and int(h) == 180 else d
            if d and d_mic != exp["direction"]:
                err(f"S10 {tid} {sid_}: 向き '{d}'（現場の向き" + (f"・三脚 180° なのでマイクから見ると '{d_mic}'" if d_mic != d else "")
                    + f"）が台本の {exp['direction']}（マイクから見た向き）と違う")
        if "side" in exp and exp["kind"] != "歩行":
            raw = (r0.get("三脚の向き") or "").strip().removesuffix("°")
            if raw and fnum(raw) is not None and int(fnum(raw)) != exp["side"]:
                name = exp["position"] if "position" in exp else ("左" if exp["side"] == 0 else "右")   # E2 の 0° は後（左ではない）・E3 は A2 と同じ左右
                err(f"S10 {tid} {sid_}: 三脚の向き {raw} が台本の {exp['side']}（{name}）と違う")
        if "speed" in exp and exp["kind"] == "H" and pos:
            sp = (pos[0].get("走行速さ") or "").strip()
            if sp and sp != exp["speed"]:
                err(f"S10 {tid} {sid_}: 速さ '{sp}' が台本の {exp['speed']} と違う")
        if "bell" in exp and pos:
            b = fnum(pos[0].get("ベル目印_m", ""))
            if b is not None and abs(abs(b) - exp["bell"]) > EPS:
                err(f"S10 {tid} {sid_}: ベルの目印 {b:g} が台本の ±{exp['bell']} m と違う")
        if (exp["kind"] == "A" or exp.get("moving")) and pos:                           # R72-06: A の速さ（時速 15／10）・E3 の速さ（時速 15）
            sp = _lead_num(pos[0].get("速度", "") or pos[0].get("走行速さ", ""))
            if sp is not None and abs(sp - float(exp["speed"])) > EPS:
                err(f"S10 {tid} {sid_}: {exp['kind']} の速さ {sp:g} が台本の時速 {exp['speed']} km と違う")
        if exp["kind"] == "E" and "dist" in exp and pos:                                 # R72-06: E2 の距離（3／7 m）と位置（前・左・後・右 = F/L/B/R）・E3（走りながら）は線と向きと左右で見る
            p0 = pos[0]
            sx, sy, mx, my = (fnum(p0.get(k, "")) for k in ("x_m", "y_m", "mic_x_m", "mic_y_m"))
            if None not in (sx, sy, mx, my):
                # 記録シート V123〜（H25-01・2026-10-25）: x_m・y_m = 音源の現場の座標、mic_x_m・mic_y_m = マイクの現場の座標 → 水平距離は差から
                # 2026-10-25（敵対レビュー 中 2）: 取り違えの形そのものを見る。y の差が約 2.0 m で片方が 0 のときだけ誤り。
                # 置けなかったときの直し（三脚を y = 1.5 にしか置けない・車を線から少し外して止めた）は警告だけ（本人 10/24「厳格にしない」）
                dy = abs(sy - my)
                if dy > E2_MIX_DY_M and abs(my) < E2_LINE_TOL_M:
                    err(f"S10 {tid} {sid_}: クラクションのマイクの y {my:g} が原点のまま・音源の y {sy:g} → マイクの座標（マイク_x_m・マイク_y_m）が抜けた疑い"
                        f"（Version 123 の E2 はマイク (0, +2.0)）")
                elif dy > E2_MIX_DY_M and abs(sy) < E2_LINE_TOL_M:
                    err(f"S10 {tid} {sid_}: クラクションの音源の y {sy:g} が 0・マイクの y {my:g} → x_m・y_m がマイクから見た位置のまま（Version 122 の形）で"
                        f"マイクの座標も入っている＝原点を二重に引く疑い")
                elif dy > E2_LINE_TOL_M:
                    warn(f"S10 {tid} {sid_}: クラクションの音源の y {sy:g} とマイクの y {my:g} が {dy:.2f} m 違う（どちらも 2.0 m の線の上のはず）。"
                         f"置けなかったときの直しならそのまま・書き間違いなら記録を直す")
                dm = ((sx - mx) ** 2 + (sy - my) ** 2) ** 0.5
                if abs(dm - exp["dist"]) > E2_DIST_TOL_M:
                    err(f"S10 {tid} {sid_}: クラクションの距離 {dm:.2f} m（音源 − マイク）が台本の {exp['dist']:g} m と違う")
            else:
                dm = fnum(p0.get("line_m", "")) or lateral_of(p0)
                if dm is not None and abs(dm - exp["dist"]) > EPS:
                    err(f"S10 {tid} {sid_}: クラクションの距離 {dm:g} が台本の {exp['dist']:g} m と違う")
            q = (pos[0].get("quadrant") or pos[0].get("象限") or "").strip()
            qe = {"前": "F", "左": "L", "後": "B", "右": "R"}[exp["position"]]
            if q and q != qe:
                err(f"S10 {tid} {sid_}: クラクションの位置 {q} が台本の {exp['position']}（{qe}）と違う")
            h_e2 = fnum((r0.get("三脚の向き") or "").strip().removesuffix("°"))
            if None not in (sx, sy, mx, my, h_e2) and q and (abs(sx - mx) > EPS or abs(sy - my) > EPS):
                # Astra MAP25-06（10/25）: 座標から求めたマイクから見た向き（step19k と同じ式 wrap(方位_現場 − 三脚の向き)）と象限を比べる。
                # 距離と y だけでは音源の x の符号の間違い（後 −3 を +3 と書く）が見つからない。現場の少しのずれは許すので注意だけ
                az = (math.degrees(math.atan2(sy - my, sx - mx)) - h_e2 + 180.0) % 360.0 - 180.0
                qc = "F" if abs(az) <= 45 else "L" if 45 < az < 135 else "R" if -135 < az < -45 else "B"
                if qc != q:
                    warn(f"S10 {tid} {sid_}: 座標（音源 ({sx:g}, {sy:g})・マイク ({mx:g}, {my:g})・三脚 {h_e2:g}°）から求めた向き {qc}（方位 {az:+.0f}°）が"
                         f"象限 {q} と違う → 音源の x の符号の書き間違いか、三脚の向きの書き間違いかを確かめる")
        if exp["kind"] == "歩行" and pos:                                                # R72-06: 歩く比較の相手のクラス・横距離（相手の線 − 歩く人の線）・側
            cls_ = pos[0]["class"].strip()
            if cls_ != exp["counterpart"]:
                err(f"S10 {tid} {sid_}: 歩く比較の相手 {cls_} が台本の {exp['counterpart']} と違う")
            lm = fnum(pos[0].get("line_m", "")); wl = fnum(pos[0].get("mic_y_m", ""))
            lat = (abs(lm - wl) if lm is not None and wl is not None else lateral_of(pos[0]))
            if lat is not None and abs(lat - exp["lateral"]) > EPS:
                err(f"S10 {tid} {sid_}: 歩く比較の横距離 {lat:g}（相手の線 − 歩く人の線）が台本の {exp['lateral']:g} m と違う")
            raw = (r0.get("三脚の向き") or "").strip().removesuffix("°")
            if raw and fnum(raw) is not None and int(fnum(raw)) != exp["side"]:
                err(f"S10 {tid} {sid_}: 歩く比較の向き {raw} が台本の {exp['side']}（{'相手が左' if exp['side'] == 180 else '相手が右'}）と違う")
    for sid_, tl in sorted(seen.items()):
        allow = E[sid_].get("takes", 1)
        sts = [s for _, s in tl if s]
        dup = sorted({s for s in sts if sts.count(s) > 1})            # 歩く比較: 同じ状態（静止か歩行）の 2 回録り
        if len(tl) > allow or dup:
            err(f"S10 {sid_}: 同じ台本ID が {len(tl)} テイク（{', '.join(t for t, _ in tl)}・許すのは {allow}"
                + (f"・同じ状態 {'・'.join(dup)} が 2 回以上" if dup else "") + "）→ 録り直しの前の本を台帳で外す")
    done = {k for k, v in seen.items() if len(v) >= E[k].get("takes", 1)}
    by_kind = defaultdict(lambda: [0, 0])
    for k, e in E.items():
        by_kind[e["kind"]][1] += 1; by_kind[e["kind"]][0] += int(k in done)
    print("  S10 台本ID の済み: " + " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(by_kind.items())) + f"  計 {len(done)}/{len(E)}")
    missing = sorted(k for k in E if k not in done)
    if missing:
        msg = f"S10 未収録の台本ID {len(missing)} 本: " + " ".join(missing[:40]) + (" …" if len(missing) > 40 else "")
        if strict_cover:
            warn(msg)
        else:
            print("  " + msg + "（当日分の未完了は警告にしない・全日程の完了は --strict）")


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
    """横距離: 単一値か幅（1.5-2.5）。幅は (lo, hi) の lo を返す（有効性の判定用・R11）。"""
    for k in LATERAL_KEYS:
        if row.get(k) not in (None, ""):
            t = str(row[k]).strip().replace("〜", "-").replace("～", "-").replace("－", "-")
            t = "".join(chr(ord(c) - 0xFEE0) if "０" <= c <= "９" or c == "．" else c for c in t)
            parts = t.split("-")
            if len(parts) not in (1, 2):
                return None
            xs = [fnum(x) for x in parts]
            if any(x is None or x < 0 for x in xs):
                return None
            if len(xs) == 2 and not xs[0] < xs[1]:
                return None
            return xs[0]
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
        take_id = r.get("take_id", "").strip()
        kind = r.get("区分", "").strip()
        state = r.get("状態", "").strip()
        if not take_id:
            err(f"S1 {tag}: take_id が空です（物理的な収録テイクを識別できません）")
        if kind not in set(plan) | {EXPOSURE_KIND}:
            err(f"S8 {tag}: 区分は A〜F/歩行/{EXPOSURE_KIND} "
                f"（現在 '{kind}'）")
        if state not in STATES:
            err(f"S8 {tag}: 状態は 静止/歩行（現在 '{state}'）")
        if kind in set(plan) - {"歩行"} and state != "静止":
            err(f"S8 {tag}: 共通試験の区分{kind}は状態=静止でなければなりません")
        # 対象が途中で通った行も負例露出に残す。step20 が計画外として扱う。
        if kind == "歩行" and not r.get("pair_id", "").strip():
            err(f"S8 {tag}: 歩行対比には pair_id が必須です")
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
        mask_only = bool((r.get("t_end") or "").strip()) or (r.get("t_end_unknown") or "").strip() == "1"   # 長回しの相手の区間（R72-02）: クリップの外にはみ出してよい・始まり ≤ 終わり
        if mask_only:
            te = fnum(r.get("t_end", ""))
            if te is not None and te + EPS < t0:
                err(f"S4 {tag}: 区間の終わり t_end({te}) が始まり t_start({t0}) より前")
        elif t0 < -EPS or t1 > dur + EPS:
            err(f"S4 {tag}: 時刻がクリップ長 [0,{dur}] の外です ({t0},{t1})")
        # 変更票 #59 r6・Astra FR11-01（2026-10-12）: 正例の時刻 0 は「ラップ秒が空のまま変換された」印（切り出し後は採点対象の行だけ見る。最接近がクリップ外の scored=0 の行は 0 に丸まりうる）
        if cls != "none" and (not cut or is_scored(r)) and (t1 <= EPS or (cls in WARN_CLASSES and t0 <= EPS)):
            err(f"S4 {tag}: 正例の時刻が 0 です（ラップ秒が空のまま変換された疑い・いつ を埋めて step19d を通し直す・変更票 #59 r6）")
        if cut:
            # 負例行も含めて全行を検査する（class=noneで抜けると、担当区間の
            # 根拠になる cut_offset_s が壊れていても素通りしてしまう）
            if not r.get("orig_file", "").strip():
                err(f"S6 {tag}: orig_file が空です")
            if fnum(r.get("cut_offset_s")) is None or fnum(r["cut_offset_s"]) < -EPS:
                err(f"S6 {tag}: cut_offset_s が非負の数値ではありません "
                    f"（現在 '{r.get('cut_offset_s','')}'）")
            odur = fnum(r.get("orig_duration_s"))
            if odur is None or odur <= 0:
                err(f"S6 {tag}: orig_duration_s が正の有限値ではありません "
                    f"（現在 '{r.get('orig_duration_s','')}'）")
            sv = str(r.get("scored", "")).strip().lower()
            if sv not in ("0", "1"):
                err(f"S6 {tag}: scored の値が不正です（'{r.get('scored')}'）。"
                    "0/1 のどちらかを指定してください")
        if cls == "none":
            if not is_scored(r):
                err(f"S6 {tag}: 負例行に scored=0 は使えません"
                    "（露出時間と誤警告の担当区間が消えます）")
            continue
        if is_scored(r):
            if kind in UNPLANNED_KINDS:
                # 負例の本（区分 G・負例露出）に偶然入った相手 = 計画外。記録は「クラスと秒だけ」（本人 2026-10-08 13:43・変更票 #47 A-3・
                # 記録シート Version 45）で、象限と横距離は求めない。step20 v3 は主要評価に入れず窓のマスクだけ効かせる（Opus 監査 Version 44 J2）
                continue
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
            odur = fnum(r.get("orig_duration_s"))
            if None in (off, t0, t1, odur):
                continue
            segs[r["orig_file"].strip()].append((off + t0, off + t1,
                                                 r["clip_id"].strip(), odur))
        for orig, ss in sorted(segs.items()):
            ss.sort()
            total = sum(b - a for a, b, _, _ in ss)
            durations = {round(x[3], 6) for x in ss}
            expected_end = ss[0][3]
            if len(durations) != 1:
                err(f"S7 {orig}: orig_duration_s がクリップ間で不一致です "
                    f"({sorted(durations)})")
            if abs(ss[0][0]) > EPS:
                err(f"S7 {orig}: 担当が原録音の先頭0.000sから始まっていません"
                    f"（{ss[0][0]:.3f}sから）→ 冒頭の誤警告を取りこぼします")
            if abs(ss[-1][1] - expected_end) > EPS:
                err(f"S7 {orig}: 担当終端 {ss[-1][1]:.3f}s が原録音長"
                    f" {expected_end:.3f}s に届いていません"
                    "→ 録音末尾の誤警告を取りこぼします")
            for (a1, b1, c1, _), (a2, b2, c2, _) in zip(ss, ss[1:]):
                if a2 < b1 - EPS:
                    err(f"S7 {orig}: 担当区間が重複 {c1}[{a1:.3f},{b1:.3f}) と "
                        f"{c2}[{a2:.3f},{b2:.3f}) → 誤警告を二重計上します")
                elif a2 > b1 + EPS:
                    err(f"S7 {orig}: 担当区間に隙間 [{b1:.3f},{a2:.3f}) "
                        f"（{c1}と{c2}の間）→ その区間の誤警告を取りこぼします")
            print(f"  S7 {orig}: {len(ss)}クリップ 担当合計 {total:.2f}s "
                  f"（{ss[0][0]:.2f}〜{ss[-1][1]:.2f}s / 原録音{expected_end:.2f}s）")

    # ---- S8 事前登録との突き合わせ（物理テイク単位） ----
    clips = {}
    takes = {}
    take_signature = {}
    for i, r in enumerate(rows, start=2):
        clips.setdefault(r["clip_id"].strip(), r)
        tid = r.get("take_id", "").strip()
        if not tid:
            continue
        sid = r.get("session_id", "").strip()
        if sid and not tid.startswith(sid):
            tid = f"{sid}/{tid}"           # 別セッションの同じ take 番号を区別（監査 R04）
        physical_source = (r.get("orig_file", "").strip() if cut
                           else r.get("clip_id", "").strip())
        sig = (r.get("区分", "").strip(), r.get("状態", "").strip(),
               r.get("pair_id", "").strip(), physical_source)
        if tid in take_signature and take_signature[tid] != sig:
            err(f"S8 L{i}: take_id={tid} の区分/状態/pair_id/原録音が行間で不一致です "
                f"({take_signature[tid]} vs {sig})")
        take_signature.setdefault(tid, sig)
        takes.setdefault(tid, r)

    got = Counter(r.get("区分", "").strip() for r in takes.values())
    take_cls = {}
    for r in rows:
        tid = r.get("take_id", "").strip()
        if not tid:
            continue
        sid = r.get("session_id", "").strip()
        if sid and not tid.startswith(sid):
            tid = f"{sid}/{tid}"
        if r["class"].strip() != "none" and tid not in take_cls:
            take_cls[tid] = r["class"].strip()
    for k in sorted(plan):
        n = got.get(k, 0)
        if k in SUBPLAN:
            req = {c: v for c, (op, v) in SUBPLAN[k].items() if op == "="}
            opt = {c: v for c, (op, v) in SUBPLAN[k].items() if op == "<="}
            lo, hi = sum(req.values()), sum(req.values()) + sum(opt.values())
            got_cls = Counter(take_cls.get(t, "") for t, r in takes.items() if r.get("区分", "").strip() == k)
            for c, v in req.items():
                if got_cls.get(c, 0) != v:
                    warn(f"S8 区分{k} の必須 {c}: {got_cls.get(c, 0)}テイク（計画 {v}テイク）")
            for c, v in opt.items():
                if got_cls.get(c, 0) > v:
                    warn(f"S8 区分{k} の任意 {c}: {got_cls.get(c, 0)}テイク（上限 {v}テイク）")
                elif got_cls.get(c, 0) < v:
                    print(f"  S8 区分{k} の任意 {c}: {got_cls.get(c, 0)}テイク（上限目標 {v}。不足は不合格にしない・0 本なら未評価と報告）")
            if not (lo <= n <= hi):
                warn(f"S8 区分{k}: {n}テイク（内訳の計画 {lo}〜{hi}テイク）")
            else:
                print(f"  S8 区分{k}: {n}テイク（内訳 " + " ".join(f"{c}={got_cls.get(c, 0)}" for c in SUBPLAN[k]) + f"・許容 {lo}〜{hi}）")
        elif n != plan[k]:
            if k in OPPORTUNITY and n < plan[k]:
                print(f"  S8 区分{k}: {n}テイク（機会枠の上限目標 {plan[k]}。不足は不合格にしない・0 本のクラスは未評価と報告）")
            else:
                warn(f"S8 区分{k}: {n}テイク（計画 {plan[k]}テイク）")
    if cut:
        n_nocal = sum(1 for r in rows if not r.get("calibration_id", "").strip())
        if n_nocal:
            warn(f"S9 calibration_id が空の行 {n_nocal} 件（step19 --calib の id を step19b --calibration-id で渡す。監査 R01）")
    planned_total = sum(got.get(k, 0) for k in plan)
    print("  S8 区分別テイク数: "
          + " ".join(f"{k}={got.get(k,0)}" for k in sorted(plan))
          + f"  計{planned_total}テイク（計画 {sum(plan.values())}テイク）"
          + f" / {EXPOSURE_KIND}={got.get(EXPOSURE_KIND,0)}テイク")

    # 歩行対比は件数だけでなく、pair_idごとに静止1・歩行1を要求する。
    walk = [r for r in takes.values() if r.get("区分", "").strip() == "歩行"]
    pairs = defaultdict(list)
    for r in walk:
        pairs[r.get("pair_id", "").strip()].append(r)
    for pair_id, pair_rows in sorted(pairs.items()):
        st = Counter(r.get("状態", "").strip() for r in pair_rows)
        if not pair_id or st != Counter({"静止": 1, "歩行": 1}):
            err(f"S8 歩行対比 pair_id='{pair_id}' は静止1・歩行1でなければなりません "
                f"（現在 静止{st.get('静止',0)}・歩行{st.get('歩行',0)}・計{len(pair_rows)}）")
    st = Counter(r.get("状態", "").strip() for r in takes.values())
    print("  S8 状態別テイク数: " + " ".join(f"{k}={v}" for k, v in sorted(st.items())))
    if "--no-script-check" not in sys.argv:
        check_script_ids(rows, plan, strict_cover=("--strict" in sys.argv))      # S10（B1・2026-10-20）

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
    neg_all_s = sum((fnum(r["t_cpa"]) or 0) - (fnum(r["t_start"]) or 0)
                    for r in rows if r["class"].strip() == "none")
    exposure_s = sum((fnum(r["t_cpa"]) or 0) - (fnum(r["t_start"]) or 0)
                     for r in rows
                     if r["class"].strip() == "none"
                     and r.get("区分", "").strip() == EXPOSURE_KIND)
    print(f"\n  クリップ {len(clips)} / 行 {len(rows)}"
          f"（正例 {len(pos)}・うち採点対象 {len(scored)}）")
    print("  GT区分: " + " ".join(f"{k}={tier[k]}" for k in
                                  ("critical", "caution", "safe", "warn")))
    print(f"  クラス: " + " ".join(f"{k}={v}" for k, v in
                                   sorted(Counter(r['class'].strip() for r in pos).items())))
    print(f"  負例担当 合計{neg_all_s/60.0:.1f}分 / "
          f"うち区分={EXPOSURE_KIND} {exposure_s/60.0:.1f}分（事前登録 ≥100分）")
    if exposure_s < 100 * 60 - EPS:
        warn(f"S8 区分={EXPOSURE_KIND}が事前登録の100分に届いていません"
             f"（{exposure_s/60.0:.1f}分）")


def selftest() -> int:
    E = expected_takes(); ok = []
    ok.append(("台本ID は 144 ＋ 16 組（学内 176 本・10/27 に E3 の 4 本）", len(E) == 160 and sum(e.get("takes", 1) for e in E.values()) == 176))
    ok.append(("E3-01 = 走りながら・2.0 m・時速 15・後→前・左 / E3-04 = 前→後・右", E["E3-01"] == dict(kind="E", moving=True, line=2.0, speed="15", direction="後→前", side=0)
               and E["E3-04"] == dict(kind="E", moving=True, line=2.0, speed="15", direction="前→後", side=180)))
    ok.append(("H2-09 = 1.0 m・速い・後→前・左 / H2-45 = 3.0 m・速い・後→前・左（台本の例）",
               E["H2-09"] == dict(kind="H", line=1.0, speed="速い", direction="後→前", side=0) and E["H2-45"] == dict(kind="H", line=3.0, speed="速い", direction="後→前", side=0)))
    ok.append(("A2-17 = 時速 10・2.0 m・後→前・左 / A2-32 = 10・5.0 m・前→後・右", E["A2-17"]["speed"] == "10" and E["A2-17"]["line"] == 2.0 and E["A2-17"]["side"] == 0
               and E["A2-32"] == dict(kind="A", line=5.0, speed="10", direction="前→後", side=180)))
    ok.append(("C2-02 = 1.0 m・ベル 10・後→前・右 / C2-16 = 3.0 m・ベル 5・前→後・右", E["C2-02"] == dict(kind="C", line=1.0, bell=10, direction="後→前", side=180)
               and E["C2-16"] == dict(kind="C", line=3.0, bell=5, direction="前→後", side=180)))
    ok.append(("E2-08 = 7 m・右・270°（#69 r40）/ W13 = 車 5 m・相手が左（180°）", E["E2-08"] == dict(kind="E", dist=7.0, position="右", side=270) and E["W13"]["counterpart"] == "car_drive" and E["W13"]["lateral"] == 5.0 and E["W13"]["side"] == 180))
    ok.append(("E2 の向き = 前 180・左 90・後 0・右 270（3 m と 7 m で同じ）", [E[f"E2-{n:02d}"]["side"] for n in range(1, 9)] == [180, 90, 0, 270, 180, 90, 0, 270]))
    errs.clear(); warns.clear()
    rows = [dict(clip_id="c1", event_id="1", take_id="01", session_id="s", **{"class": "bicycle"}, 区分="H", 状態="静止", 横距離m="1.0", line_m="1.0", direction="後→前", 走行速さ="速い", **{"三脚の向き": "0", "台本ID": "H2-09"}),
            dict(clip_id="c2", event_id="1", take_id="02", session_id="s", **{"class": "bicycle"}, 区分="H", 状態="静止", 横距離m="1.5", line_m="1.5", direction="後→前", 走行速さ="速い", **{"三脚の向き": "0", "台本ID": "H2-10"}),   # 線と向きが台本（1.0・右）と違う
            dict(clip_id="c3", event_id="1", take_id="03", session_id="s", **{"class": "bicycle"}, 区分="H", 状態="静止", 横距離m="1.0", line_m="1.0", direction="後→前", 走行速さ="速い", **{"三脚の向き": "0", "台本ID": "H2-09"}),   # 重複
            dict(clip_id="c4", event_id="1", take_id="04", session_id="s", **{"class": "bicycle"}, 区分="H", 状態="静止", 横距離m="1.0", line_m="1.0", direction="後→前", 走行速さ="速い", **{"三脚の向き": "0", "台本ID": "H9-99"})]   # 台本に無い
    check_script_ids(rows, PLAN_DEFAULT)
    ok.append((f"取り違え・重複・台本に無い ID を誤りに（{len(errs)} 件）", sum("線 line_m 1.5" in e for e in errs) == 1 and sum("三脚の向き 0 が台本の 180" in e for e in errs) == 1
               and sum("同じ台本ID が 2 テイク" in e for e in errs) == 1 and sum("台本に無い" in e for e in errs) == 1))
    errs.clear(); warns.clear()
    check_script_ids([dict(r, **{"台本ID": ""}) for r in rows[:1]], PLAN_DEFAULT)
    ok.append(("台本ID が 1 行も無ければ飛ばす（旧様式）", not errs and not warns))
    errs.clear(); warns.clear()
    rows2 = [dict(clip_id="a1", event_id="1", take_id="11", session_id="s", **{"class": "car_drive"}, 区分="A", 状態="静止", 横距離m="2.0", line_m="2.0", direction="後→前", 速度="10", **{"三脚の向き": "0", "台本ID": "A2-01"}),   # 速さ 10（台本 15）
             dict(clip_id="e5", event_id="1", take_id="12", session_id="s", **{"class": "horn"}, 区分="E", 状態="静止", 横距離m="3.0", line_m="3.0", quadrant="F", **{"三脚の向き": "180", "台本ID": "E2-05"}),   # 距離 3（台本 7）・向きは台本どおり 180（2026-10-25: 旧の行は 0）
             dict(clip_id="e2", event_id="1", take_id="13", session_id="s", **{"class": "horn"}, 区分="E", 状態="静止", 横距離m="3.0", line_m="3.0", quadrant="F", **{"三脚の向き": "90", "台本ID": "E2-02"}),   # 位置 F（台本 左 L）・向きは台本どおり 90
             dict(clip_id="w1", event_id="1", take_id="14", session_id="s", **{"class": "car_drive"}, 区分="歩行", 状態="静止", pair_id="P1", 横距離m="8.0", line_m="2.0", mic_y_m="10", **{"三脚の向き": "0", "台本ID": "W01"}),   # 相手 車（台本 ベル）・横 8（台本 3）・向き 0（台本 180）
             dict(clip_id="w1ok", event_id="1", take_id="15", session_id="s", **{"class": "bike_bell"}, 区分="歩行", 状態="歩行", pair_id="P1", 横距離m="3.0", line_m="2.0", mic_y_m="5", **{"三脚の向き": "180", "台本ID": "W01"})]   # 正しい
    check_script_ids(rows2, PLAN_DEFAULT)
    ok.append((f"R72-06: A の速さ・E の距離と位置・W の相手／横距離／向きの取り違えを誤りに（{len(errs)} 件）・正しい W01 は通る",
               sum("A の速さ" in e for e in errs) == 1 and sum("クラクションの距離" in e for e in errs) == 1 and sum("クラクションの位置" in e for e in errs) == 1
               and sum("歩く比較の相手" in e for e in errs) == 1 and sum("歩く比較の横距離" in e for e in errs) == 1 and sum("歩く比較の向き" in e for e in errs) == 1 and not any("s/15" in e for e in errs)
               and not any("三脚の向き" in e and ("s/12" in e or "s/13" in e) for e in errs)))
    # Astra H25-01（2026-10-25 Opus・Fable の確認待ち）: 記録シート V123 の E2（x_m・y_m = 音源の現場の座標・mic_x_m・mic_y_m = マイクの現場の座標）
    errs.clear(); warns.clear()
    e2 = lambda tk, sid, h, x, y, q, mx="0", my="2": dict(clip_id=f"v{tk}", event_id="1", take_id=tk, session_id="v", **{"class": "horn"}, 区分="E", 状態="静止", 横距離m="", line_m="",
                                                        quadrant=q, x_m=x, y_m=y, mic_x_m=mx, mic_y_m=my, **{"三脚の向き": h, "台本ID": sid})
    check_script_ids([e2("21", "E2-01", "180", "-3", "2.0", "F"), e2("22", "E2-07", "0", "-7", "2.0", "B"),          # 正しい（前 3 m・後 7 m）
                      e2("23", "E2-03", "0", "-3", "0", "B"),                                                       # 原点を二重に引いた形（音源がマイクから見た位置のまま）
                      e2("24", "E2-06", "90", "-3", "2.0", "L"),                                                     # 距離 3（台本 7）
                      e2("25", "E2-04", "90", "-3", "2.0", "R")], PLAN_DEFAULT)                                     # 向き 90（台本 右 270）
    ok.append((f"H25-01: V123 の E2 は位置から距離を照合・原点の二重引きと向きの取り違えを誤りに（{len(errs)} 件）・正しい 2 本は通る",
               not any(("v/21" in e or "v/22" in e) for e in errs)
               and sum("v/23" in e and "原点を二重に引く" in e for e in errs) == 1
               and sum("v/24" in e and "クラクションの距離 3.00 m" in e for e in errs) == 1
               and sum("v/25" in e and "三脚の向き 90 が台本の 270（右）" in e for e in errs) == 1))
    for name, cnd in ok:
        print(("PASS " if cnd else "FAIL ") + name)
    k = sum(cnd for _, cnd in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    ann = Path(_arg("--ann"))
    cut = "--cut" in sys.argv
    dur = float(_arg("--dur", "10.0")) if cut else float(_arg("--dur", "1e9"))
    plan = dict(PLAN_DEFAULT)
    if _arg("--plan"):
        for kv in _arg("--plan").split(","):
            k, v = kv.split("=")
            plan[k.strip()] = int(v)
    if _arg("--opportunity"):
        for kv in _arg("--opportunity").split(","):
            k, v = kv.split("=") if "=" in kv else (kv, None)
            OPPORTUNITY.add(k.strip())
            if v is not None:
                plan[k.strip()] = int(v)
    if _arg("--subplan"):
        # 例: D:crossing=8,backup_beep=4,horn=4,siren<=4  （複数の区分は ';' で区切る）
        for part in _arg("--subplan").split(";"):
            kind, spec = part.split(":", 1)
            d = {}
            for item in spec.split(","):
                if "<=" in item:
                    c, v = item.split("<="); d[c.strip()] = ("<=", int(v))
                else:
                    c, v = item.split("="); d[c.strip()] = ("=", int(v))
            SUBPLAN[kind.strip()] = d
            plan[kind.strip()] = sum(v for op, v in d.values())     # 総数の上限（満数）
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
