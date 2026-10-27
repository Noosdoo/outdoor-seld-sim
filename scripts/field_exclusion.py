# -*- coding: utf-8 -*-
"""開発用の実録（台帳で使わないの本）の除外区間の扱い（変更票 #75 決め直し 6・Astra R5-30-01・本人 10/27 00:49「6 ok」・2026-10-27 Opus）。
採点の段（学内の 19 本）から呼ぶ部品。結果を見てから本ごとに扱いを変えない（全クラス・全モデルに同じに当てる）。

  1. 時刻: 除外区間は全長の録音の秒で [入り, 出)。出が空なら録音の末尾。切り出しの秒へは cut_offset を引いて直す（to_clip）。
  2. フレームの指標（検出・向き・距離・ベル）: 採点するフレーム（暖機のあと）から、除外区間に少しでも掛かるフレームを引いた差集合だけを使う
     （frame_metric）。除外中の予測と正解は分母・分子の両方から外し、有効フレーム数（秒）を併記。正解の残りが 0 なら欠測（None）。
  3. 通知の成否（通過・ベルの「届いた」）: 必要な履歴（成功の窓の始まりの前 4.0 秒＝暖機）と成功の窓 [鳴り始め − 1, + 4] を全部観察できる
     事象だけを通常の分母に入れる。除外区間がそのどこかに重なる事象は「部分観察」（partial）として別記し、成功にも失敗にも混ぜない（notification）。
  4. 推論は元の時間軸のまま（音を切り詰めたり連結したりしない）。通知の履歴はリセットしない。除外中の発火は数えず、窓の成功にも使わない。
  5. 対象外の本数（部分観察の事象・欠測の本）を表の頭に出す（summarize）。
フレーム k は [k·0.1, (k+1)·0.1) 秒。除外区間 [a, b) に少しでも掛かるフレームは除外（中心でなく重なりで決める＝広めに外す）。
試験: scripts/test_field_exclusion.py（架空例 5 種＋時刻の直し）
"""
from __future__ import annotations

RES = 0.1
NF = 100
WARMUP_S = 4.0
WIN = (-1.0, 4.0)
EPS = 1e-9


def to_clip(intervals_rec, cut_offset: float, rec_len: float) -> list:
    """全長の録音の秒の [入り, 出)（出が空・None なら録音の末尾）→ 切り出しの秒の [a, b)。入り ≥ 出 や非有限は止める。"""
    out = []
    for a, b in intervals_rec:
        a = float(a)
        b = float(rec_len) if b in (None, "") else float(b)
        if not (a == a and b == b) or not a < b:
            raise ValueError(f"除外区間が不正: [{a}, {b})")
        out.append((a - float(cut_offset), b - float(cut_offset)))
    return out


def frames_in(intervals_clip, nf: int = NF, res: float = RES) -> set:
    """[a, b) に少しでも掛かるフレーム番号の集合（フレーム k = [k·res, (k+1)·res)）。"""
    s = set()
    for a, b in intervals_clip:
        for k in range(nf):
            if k * res < b - EPS and (k + 1) * res > a + EPS:
                s.add(k)
    return s


def scored_frames(warmup_s: float = WARMUP_S, nf: int = NF, res: float = RES) -> set:
    return {k for k in range(nf) if k * res >= warmup_s - EPS}


def frame_metric(gt_frames, det_frames, excl_frames, scored=None) -> dict:
    """差集合の検出の指標。gt_frames = 正解のあるフレーム・det_frames = 予測のあるフレーム（同じクラス）。
    戻り: n_valid（採点できるフレーム数）・valid_s・n_gt・n_hit・recall（n_gt が 0 なら None＝欠測）・n_fp（正解の無い有効フレームでの予測）・n_excluded。"""
    scored = scored_frames() if scored is None else set(scored)
    valid = scored - set(excl_frames)
    g = set(gt_frames) & valid
    d = set(det_frames) & valid
    return {"n_valid": len(valid), "valid_s": round(len(valid) * RES, 3), "n_gt": len(g), "n_hit": len(g & d),
            "recall": (len(g & d) / len(g)) if g else None, "n_fp": len(d - g), "n_excluded": len(scored & set(excl_frames))}


def notification(onset_s: float, fire_times, intervals_clip, win=WIN, warmup_s: float = WARMUP_S) -> dict:
    """通知の成否。観察が要る範囲 = [窓の始まり − 暖機, 窓の終わり]。除外区間がそこに重なれば partial（成功にも失敗にも混ぜない）。
    全部観察できれば、除外中でない発火が窓の中にあれば success、無ければ fail。除外中の発火は使わない。"""
    w0, w1 = onset_s + win[0], onset_s + win[1]
    need = (w0 - warmup_s, w1)
    for a, b in intervals_clip:
        if a < need[1] + EPS and b > need[0] + EPS:
            return {"status": "partial", "window": (w0, w1), "need": need, "reason": f"除外区間 [{a:.2f}, {b:.2f}) が観察の要る範囲に重なる"}
    ok_fires = [t for t in fire_times if not any(a - EPS <= t < b - EPS for a, b in intervals_clip)]
    hit = [t for t in ok_fires if w0 - EPS <= t <= w1 + EPS]
    return {"status": "success" if hit else "fail", "window": (w0, w1), "need": need, "first_fire": (min(hit) if hit else None)}


def summarize(take_results: list) -> dict:
    """表の頭に出す数: 本数・欠測の本（フレームの正解の残りが 0）・通知の部分観察の事象・通常の分母の事象。"""
    n_missing = sum(1 for t in take_results if t.get("frame", {}).get("recall") is None)
    ev = [e for t in take_results for e in t.get("notifications", [])]
    return {"n_takes": len(take_results), "n_missing_takes": n_missing,
            "n_events": len(ev), "n_partial": sum(e["status"] == "partial" for e in ev),
            "n_scored_events": sum(e["status"] in ("success", "fail") for e in ev), "n_success": sum(e["status"] == "success" for e in ev)}
