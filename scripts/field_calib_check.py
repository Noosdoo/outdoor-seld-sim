# -*- coding: utf-8 -*-
"""実録の較正の値の照らし合わせ（変更票 #69 r43・2026-10-25 Opus・Fable の確認待ち）。
敵対レビュー（md/audit/敵対的レビュー_Opus55_2026-10-25_Opusの実装.md）の中 1・軽 7 を直した版（2026-10-25 Opus）。

決まり（r43）: ふだんは騒音計の画面の 1 分の LAeq（記録シートの値）を使う。毎晩、騒音計の 1 秒ごとの記録から
録音と同じ秒（暗騒音区間・ふつう 5〜60 s）のエネルギー平均も出して比べ、差が --tol（既定 0.5 dB）を超えた日だけ 1 秒の記録の値に替える。
ただし差が --stop-db（既定 1.4 dB＝運用上の止める線（クラス 2 の参照条件〔94 dB・1 kHz〕の ±1.4 dB を借りた値・現場の差の許容を保証しない・Astra MAP25-05／P25））を超えたら、1 秒の記録の値に自動では替えずに止める
（本人と原因を確かめてから値を決める・本人 10/25 15:56 の決定 (b)）。

やること:
  1 録音（H3-VR の校正の WAV）の W チャンネルの 1 秒ごとの A 特性レベル（spl_a・本番の約束）を出す
  2 騒音計の 1 秒ごとの記録（TXT: 番号 タブ 日付 タブ 時刻 タブ 値）を読む。--slm がフォルダなら中の .TXT を全部試す
    （拡張子の大文字・小文字は問わず、同じファイルは 1 回だけ）
  3 録音の録り始め（bext の時刻）と騒音計の時刻で合わせる。騒音計の記録は行の番号ではなく時刻（秒）で引く
    （記録の途中に抜けがあっても秒がずれない）。録音の i 秒 ↔ 騒音計の時刻「録り始め + i + k 秒」として、
    時計どおり（k = 0）から ±30 秒の範囲で、窓の中で重なった秒の相関が一番高い k を選ぶ
    （H3-VR と騒音計の時計は数秒ずれる・本物の校正で 10/15 は +1 s・10/22 は +3 s）。録音に bext が無ければ照合しない
  4 窓（暗騒音区間）の中で重なった秒だけで、騒音計の値のエネルギー平均（dB → 強さ → 平均 → dB）を出す
  5 判定（上から順に当てはめる）
     照合できない: 録音に bext が無い・どのずれでも騒音計の記録と重ならない・相関が --min-corr より弱い → 使う値は画面の値（確かめる）
     要確認: 選んだずれが時計どおりから 10 s 以上離れている・選んだずれで窓の全部が騒音計の記録と重ならない・
             1 秒ごとの変化（隣の秒との差）どうしの相関が --min-dcorr より弱い → 使う値は画面の値（確かめる）
             （最後のものは、ゆっくり変わる暗騒音どうしがたまたま似て、ずれた秒でも相関が高く出る誤りを止めるため。下の MIN_DCORR）
     画面: 画面の値（--laeq）との差が --tol 以内 → 画面の値
     止める: 差の大きさが --stop-db を超えた → 1 秒の記録の値に自動では替えない（本人と原因を確かめてから値を決める）。使う値なし・ゲインなし
     1 秒の記録: 差が --tol を超え --stop-db 以下 → 1 秒の記録の値に替える（本人に知らせる・r43）。画面の値が無いときも、上の検査を通れば 1 秒の記録の値
     （秒の合わせが「照合できない」「要確認」のときはそちらが先で、差の大きさでは判定しない）
  6 使う値と窓で、step19 と同じ計算の較正のゲイン（calib_gain_db）を出す（使う値が無ければゲインも空）
台帳は書き換えない（値と判定を出すだけ・台帳を直すのは本人の了承のあと・r41 と同じ形で history/ に元を残す）。

終了コード（判定ごと。夜の命令で「画面」以外のときに止まって見るため）:
  0 = 画面（差が --tol 以内・そのまま）
  3 = 1 秒の記録（差が --tol を超え --stop-db 以下 → 1 秒の記録の値に替えて本人に知らせる。画面の値が無いときもこれ）
  4 = 要確認（ずれが時計どおりから 10 s 以上・窓の一部しか重ならない・変化どうしの相関が弱い）→ 使う値は画面の値
  5 = 照合できない（bext が無い・騒音計の記録が重ならない・相関が弱い）→ 使う値は画面の値
  6 = 止める（差の大きさが --stop-db を超えた → 自動では替えない・本人と原因を確かめてから値を決める）→ 使う値なし（laeq_use・gain_db は空）
  2 = 入力の誤り（窓が整数の秒でない・窓が録音より長い・ファイルが無い・--stop-db が --tol より小さい）・1 = 道具の中の思わぬエラー
  4・5 で画面の値（--laeq）も無いときは、使う値なし（ゲインも空）。

出力の項目の意味（一部）:
  shift_from_clock_s = 選んだずれ k（騒音計の時刻 − 録音の時刻〔bext + 録音の秒〕）。合わせはこれで決める
  shift_s = 騒音計の記録の行の番号（0 から）− 録音の秒（窓の最初の重なった秒で・前の版と同じ読み方の参考。記録に抜けがあると途中から変わる）
  corr ／ dcorr = 選んだずれでの 1 秒ごとの値どうしの相関 ／ 1 秒ごとの変化（隣の秒との差）どうしの相関
  n_overlap ／ n_window = 窓の中で騒音計の記録と重なった秒 ／ 窓の秒

使い方:
  python scripts/field_calib_check.py --wav raw/261029_001.WAV --slm raw/slm/20261029/ --laeq 48.3 --window 5-60 [--tol 0.5] [--stop-db 1.4] [--min-corr 0.6] [--min-dcorr 0.5] [--json out.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.calibration import spl_a          # noqa: E402  本番の約束（step19 と同じ）
import step19_realsmoke_convert as cv               # noqa: E402  calib_gain_db（step19 と同じ計算）

FS = 24000
SEARCH_S = 30          # 時計どおりから探すずれの幅 [s]
CLOCK_TOL_S = 10       # 選んだずれがこれ以上時計どおりから離れたら「要確認」[s]（本物の校正の時計のずれは 1〜3 s）
# 相関の下限の既定（中 1 の直しで 0.5 → 0.6 に上げた）。決め方:
#  - 本物の校正の正しい合わせの相関は 10/22 が 0.957、10/15 が 0.711（10/15 は録音の 1 秒の枠を 0.1 秒刻みでずらしても 0.712 にしかならず、
#    秒の途中のずれのせいではない＝本物では「正しい合わせは 0.9 以上」は成り立たない）。下限は 0.711 より余裕をもって低くする
#    （0.7 にすると 10/15 の型の日が 0.01 の差で照合できないになる）。架空の正しい合わせは、騒音計の値の雑音 0.5 dB まで最小 0.84。
#  - 誤った合わせを止めるのは主に形の検査（時刻で引く・bext が無ければ照合しない・時計から 10 s 以上は要確認・窓の全部が重ならなければ要確認）
#    と下の差分の相関。レビューの再現（r6b_calib_realistic.py の a〜d・各 40 回）は形の検査だけで「1 秒の記録に替える」が 0 回になる。
#  - 値どうしの相関だけでは誤りを止めきれない（Opus の 10/25 の試し・台本はリポジトリの外・r6b と同じ作り方）:
#    騒音計の記録が切れずに続き録音機の時計が 30 s 超ずれた架空の形（240 回）で、形の検査を
#    通った誤った合わせ 42 回の相関は最大 0.951（ゆっくり変わる暗騒音どうしはずれた秒でも似る）。0.6 にしてもそのうち 23 回は通る。
#    だから下限は「明らかに弱い」を落とす分（0.5 → 0.6）にとどめ、誤りは差分の相関（MIN_DCORR）で止める。
#  下限を上げすぎたときの失敗は「照合できない → 画面の値」＝ふだんの手順に戻るだけ。
MIN_CORR = 0.6
# 1 秒ごとの変化（隣の秒との差）どうしの相関の下限の既定（中 1 の直しで足した）。決め方:
#  - 同じ音を測っていれば、ゆっくりした流れを除いた秒ごとの上がり下がりも合う。本物の正しい合わせは 10/22 が 0.73、10/15 が 0.735。
#  - 上の架空の形で形の検査を通った誤った合わせ 42 回の差分の相関は最大 0.41・中央 0.21（値どうしの相関は最大 0.951）→ 0.5 で全部止まる。
#  - 架空の正しい合わせは、騒音計の値の雑音 0.1 dB で最小 0.94、0.5 dB で最小 0.40（このときは要確認 → 画面の値＝ふだんの手順）。
MIN_DCORR = 0.5
# 画面の値と 1 秒の記録の値の差がこれを超えたら、1 秒の記録の値に自動では替えずに止める（本人 10/25 15:56 の決定 (b)）。
# 1.4 dB は運用上の止める線（クラス 2 の参照条件〔94 dB・1 kHz〕の ±1.4 dB を借りた値・現場の差の許容を保証しない・Astra MAP25-05／P25）。差が 0.5〜1.4 dB なら今までどおり 1 秒の記録の値に替える。
STOP_DB = 1.4
JUDGE_EXIT = {"画面": 0, "1秒の記録": 3, "要確認": 4, "照合できない": 5, "止める": 6}
EXIT_INPUT = 2


def wav_start(path: Path):
    """WAV の bext の録り始め（日付＋時刻）。無ければ None。中身（音）は読まない。"""
    with open(path, "rb") as f:
        if f.read(12)[8:12] != b"WAVE":
            return None
        while True:
            ck = f.read(8)
            if len(ck) < 8:
                return None
            cid, sz = ck[:4], struct.unpack("<I", ck[4:])[0]
            if cid == b"bext":
                b = f.read(sz)
                try:
                    return dt.datetime.strptime(b[320:338].decode("latin1"), "%Y-%m-%d%H:%M:%S")
                except ValueError:
                    return None
            f.seek(sz + (sz & 1), 1)


def rec_levels(path: Path) -> np.ndarray:
    """録音の W の 1 秒ごとの A 特性レベル [dB・本番の約束]（24 kHz に落としてから）。"""
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    w = x[:, 0]
    if sr != FS:
        if sr % FS:
            raise ValueError(f"{path.name}: 想定外の fs={sr}")
        w = resample_poly(w, 1, sr // FS)
    n = len(w) // FS
    return np.array([spl_a(w[i * FS:(i + 1) * FS], FS) for i in range(n)])


def read_slm(path: Path):
    """騒音計の 1 秒ごとの記録 → [(datetime, 値)]。読めない行は飛ばす。"""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.strip().split("\t")
            if len(p) < 4:
                continue
            try:
                t = dt.datetime.strptime(p[1].strip() + " " + p[2].strip().split(".")[0], "%Y/%m/%d %H:%M:%S")
                v = float(p[3])
            except ValueError:
                continue
            if math.isfinite(v):                         # Astra MAP25-09（10/25）: nan・inf の行は読めない行と同じく飛ばす（窓が欠ければ要確認になる）
                out.append((t, v))
    return out


def slm_files(path: Path) -> list:
    """--slm がフォルダなら中の .TXT（拡張子の大文字・小文字を問わない）を 1 回ずつ、ファイルならそれだけ。
    軽 7: *.TXT と *.txt を別々に探すと、Windows では同じファイルに 2 回当たっていた。フォルダの一覧から拡張子で選ぶので
    同じファイルは 1 回だけ出る（field_ledger_draft の _files と同じく、同じファイルを 2 回数えない）。"""
    if not path.is_dir():
        return [path]
    return sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"), key=lambda p: p.name.lower())


def int_window(window) -> tuple:
    """窓（暗騒音区間）を整数の秒 (a, b) に。騒音計の記録は 1 秒ごとなので、小数の秒は合わせられない → エラー
    （軽 7: 前は「5.5-60」を黙って 5 に切り捨てていた）。5.0-60.0 のように整数の値なら受ける。"""
    try:
        a, b = (float(x) for x in window)
    except (TypeError, ValueError):
        raise ValueError(f"窓は「始まり-終わり」の 2 つの数で書く（例 5-60）: {window!r}") from None
    if not (math.isfinite(a) and math.isfinite(b) and a.is_integer() and b.is_integer()):
        raise ValueError(f"窓 {a:g}-{b:g} は整数の秒で書く（騒音計の記録は 1 秒ごとで、小数の秒は合わせられない・例 5-60）")
    a, b = int(a), int(b)
    if a < 0 or b <= a:
        raise ValueError(f"窓 {a}-{b} は 0 ≤ 始まり < 終わり にする")
    return a, b


def parse_window(text: str) -> tuple:
    parts = str(text).split("-")
    if len(parts) != 2:
        raise ValueError(f"窓は「始まり-終わり」で書く（例 5-60）: {text!r}")
    return int_window(parts)


def energy_mean(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(10 * np.log10(np.mean(10 ** (v / 10))))


def _corr(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0                                       # Astra MAP25-09: 有限でない値（無音の秒の −inf など）が混じれば相関なし＝照合できない側へ
    return float(np.corrcoef(x, y)[0, 1])


def diff_corr(rec: np.ndarray, rec_s, vals) -> float:
    """1 秒ごとの変化（隣の秒との差）どうしの相関。録音の秒が続いている所（i と i+1 の両方が重なった）だけで。"""
    ii = np.asarray(rec_s); vv = np.asarray(vals, dtype=float)
    cont = np.where(np.diff(ii) == 1)[0]
    return _corr(rec[ii[cont + 1]] - rec[ii[cont]], vv[cont + 1] - vv[cont])


def align(rec: np.ndarray, slm, window, start: dt.datetime | None, search: int = SEARCH_S, min_overlap: int = 40):
    """録音の秒 i ↔ 騒音計の時刻 start + i + k 秒（k = ずれ）。騒音計の記録は時刻で引く（行の番号は使わない＝記録の途中の抜けで秒がずれない）。
    k は時計どおり（0）から ±search 秒で、窓の中で重なった秒の相関が一番高い所。重なりが窓の 7 割（40 秒が上限）に満たないずれは比べない。
    返り値: dict(k, corr, dcorr, rec_s, slm_rows) か None（録り始めが無い・どのずれでも重なりが足りない）。slm_rows は記録の行の番号（0 から）、
    dcorr は選んだずれでの 1 秒ごとの変化どうしの相関。"""
    if start is None or not slm:
        return None
    a, b = window
    at = {}
    for n, (t, v) in enumerate(slm):
        at.setdefault(t, (n, v))                            # 同じ時刻が 2 行あれば最初の行
    need = min(min_overlap, int(0.7 * (b - a)))
    best = None
    for k in range(-search, search + 1):
        hit = [(i, at.get(start + dt.timedelta(seconds=i + k))) for i in range(a, b)]
        hit = [(i, h) for i, h in hit if h is not None]
        if len(hit) < max(need, 2):
            continue
        ii = np.array([i for i, _ in hit]); vv = np.array([h[1] for _, h in hit], dtype=float)
        c = _corr(rec[ii], vv)
        if best is None or c > best["corr"]:
            best = dict(k=k, corr=c, rec_s=ii.tolist(), slm_rows=[h[0] for _, h in hit], _vals=vv)
    if best is not None:
        best["dcorr"] = diff_corr(rec, best["rec_s"], best.pop("_vals"))
    return best


def check(wav: Path, slm_paths, laeq_display: float | None, window, tol: float = 0.5, min_corr: float = MIN_CORR,
          clock_tol: float = CLOCK_TOL_S, min_dcorr: float = MIN_DCORR, stop_db: float = STOP_DB, convention: str | None = None) -> dict:
    a, b = int_window(window)
    for name, v in (("--laeq", laeq_display), ("--tol", tol), ("--stop-db", stop_db), ("--min-corr", min_corr), ("--min-dcorr", min_dcorr), ("clock_tol", clock_tol)):
        if v is not None and not math.isfinite(float(v)):           # Astra MAP25-09（10/25）: nan の画面の値が差の比較をすり抜けて「1 秒の記録」になっていた
            raise ValueError(f"{name} が有限の数でない: {v!r}")
    if not stop_db >= tol:
        raise ValueError(f"--stop-db {stop_db:g} は --tol {tol:g} 以上にする")
    rec = rec_levels(wav)
    if b > len(rec):
        raise ValueError(f"窓の終わり {b} s が録音の長さ（1 秒の枠で {len(rec)} s 分）を超える: {wav.name}")
    start = wav_start(wav)
    res = dict(wav=wav.name, wav_start=(start.strftime("%Y-%m-%d %H:%M:%S") if start else ""), window=f"{a}-{b}",
               laeq_display=laeq_display, tol_db=tol, stop_db=stop_db, min_corr=min_corr, min_dcorr=min_dcorr, clock_tol_s=clock_tol)
    if laeq_display is not None:
        fallback, fb_text = dict(use="画面", laeq_use=laeq_display), "画面の値（確かめる）"
    else:
        fallback, fb_text = dict(use="", laeq_use=None), "画面の値も無いので使う値なし（確かめる）"
    cands = []
    if start is not None:
        for sp in slm_paths:
            slm = read_slm(sp)
            al = align(rec, slm, (a, b), start)
            if al is not None:
                cands.append((al["corr"], sp, slm, al))
    if start is None:
        res.update(judgment="照合できない", status=f"録音に録り始めの時刻（bext）が無く、騒音計の時刻と照合できない → {fb_text}", **fallback)
    elif not slm_paths:
        res.update(judgment="照合できない", status=f"騒音計の 1 秒の記録（.TXT）が無い → {fb_text}", **fallback)
    elif not cands:
        res.update(judgment="照合できない", status=f"騒音計の記録と録音の秒が重ならない（時計どおり ±{SEARCH_S} s の範囲で） → {fb_text}", **fallback)
    else:
        corr, sp, slm, al = max(cands, key=lambda c: c[0])
        k = al["k"]
        vals = [slm[j][1] for j in al["slm_rows"]]
        laeq_log = energy_mean(vals)
        res.update(slm=sp.name, shift_s=al["slm_rows"][0] - al["rec_s"][0], shift_from_clock_s=k, corr=round(corr, 3),
                   dcorr=round(al["dcorr"], 3), n_overlap=len(vals), n_window=b - a, rec_s=f"{al['rec_s'][0]}〜{al['rec_s'][-1] + 1}",
                   slm_time=f"{slm[al['slm_rows'][0]][0]:%H:%M:%S}〜{slm[al['slm_rows'][-1]][0]:%H:%M:%S}",
                   laeq_log=round(laeq_log, 2), slm_min=min(vals), slm_max=max(vals))
        d = None
        if laeq_display is not None:
            d = laeq_display - laeq_log
            res["diff_db"] = round(d, 2)
        ref = f"（参考: 重なった秒の 1 秒の記録 {laeq_log:.2f} dB" + (f"・画面との差 {d:+.2f} dB" if d is not None else "") + "）"
        why = []
        if abs(k) >= clock_tol:
            why.append(f"選んだずれ {k:+d} s が時計どおりから {clock_tol:g} s 以上離れている（録音機か騒音計の時計のずれ・本当のずれは探す範囲 ±{SEARCH_S} s の外かもしれない）")
        if len(vals) < b - a:
            why.append(f"選んだずれでは窓 {a}〜{b} s の {b - a} 秒のうち {len(vals)} 秒しか騒音計の記録と重ならない（記録の始まり・終わり・途中の抜け）")
        if al["dcorr"] < min_dcorr:
            why.append(f"1 秒ごとの変化どうしの相関 {al['dcorr']:.2f} が下限 {min_dcorr:g} より弱い（値どうしの相関 {corr:.2f} は、"
                       "ゆっくり変わる暗騒音がたまたま似ただけで秒が合っていないのかもしれない）")
        if corr < min_corr:
            res.update(judgment="照合できない", status=f"相関 {corr:.2f} が下限 {min_corr:g} より弱く、秒を合わせられない → {fb_text}{ref}", **fallback)
        elif why:
            res.update(judgment="要確認", status="要確認: " + "・".join(why) + f" → {fb_text}{ref}", **fallback)
        elif d is None:
            res.update(judgment="1秒の記録", status=f"画面の値なし → 1 秒の記録の値（推定: 1 秒ごとの値の種類は未確認・相関 {corr:.2f}・ずれ {k:+d} s・窓の全部が重なる）",
                       use="1秒の記録", laeq_use=round(laeq_log, 1))
        elif abs(d) <= tol:
            res.update(judgment="画面", status=f"差 {d:+.2f} dB は ±{tol:g} 以内 → 画面の値", use="画面", laeq_use=laeq_display)
        elif abs(d) > stop_db:                     # 本人 10/25 15:56 の決定 (b): 差が大きいときは自動で替えない
            res.update(judgment="止める", status=f"止める: 差 {d:+.2f} dB が ±{stop_db:g}（運用上の止める線）を超える → "
                                                 f"1 秒の記録の値 {laeq_log:.2f} dB に自動では替えない（本人と原因を確かめてから値を決める・使う値なし）",
                       use="", laeq_use=None)
        else:
            res.update(judgment="1秒の記録", status=f"差 {d:+.2f} dB が ±{tol:g} を超える（±{stop_db:g} 以下）→ 1 秒の記録の値に替える（推定: 1 秒ごとの値の種類〔1 秒の LAeq か FAST／SLOW の指示値か〕は未確認・本人に知らせる・r43・Astra MAP25-05）",
                       use="1秒の記録", laeq_use=round(laeq_log, 1))
    res["exit_code"] = JUDGE_EXIT[res["judgment"]]
    res["convention"] = cv.convention_of(convention)[0]                # 先行分 §3: ゲインの約束（既定 v9_143・合成にそろえるなら v12_149）
    res["gain_db"] = round(cv.calib_gain_db(wav, float(res["laeq_use"]), (a, b), convention=res["convention"]), 2) if res.get("laeq_use") is not None else None
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実録の較正の値の照らし合わせ（#69 r43）",
                                 epilog="終了コード: 0 = 画面・3 = 1 秒の記録（替える・本人に知らせる）・4 = 要確認・5 = 照合できない"
                                        "（4・5 の使う値は画面の値）・6 = 止める（差が --stop-db を超えた・使う値なし）・2 = 入力の誤り")
    ap.add_argument("--wav", required=True); ap.add_argument("--slm", required=True, help="騒音計の 1 秒の記録（TXT）かそのフォルダ")
    ap.add_argument("--laeq", type=float, default=None, help="記録シートに書いた画面の LAeq")
    ap.add_argument("--window", default="5-60", help="暗騒音区間（録音の秒・整数・例 5-60）")
    ap.add_argument("--tol", type=float, default=0.5, help="この差 [dB] 以内なら画面の値（既定 0.5）")
    ap.add_argument("--stop-db", type=float, default=STOP_DB,
                    help=f"差の大きさがこれ [dB] を超えたら 1 秒の記録の値に自動では替えずに止める（既定 {STOP_DB:g}＝運用上の止める線・クラス 2 の参照条件の値を借りたもの）")
    ap.add_argument("--min-corr", type=float, default=MIN_CORR, help=f"値どうしの相関の下限（既定 {MIN_CORR:g}・弱いと照合できない）")
    ap.add_argument("--min-dcorr", type=float, default=MIN_DCORR, help=f"1 秒ごとの変化どうしの相関の下限（既定 {MIN_DCORR:g}・弱いと要確認）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--convention", default=None, help="ゲインの約束: v9_143（既定・旧）／v12_149（合成の約束にそろえる・先行分 §3）。照らし合わせの判定は変わらない")
    a = ap.parse_args(argv)
    wav, sp = Path(a.wav), Path(a.slm)
    try:
        for p in (wav, sp):
            if not p.exists():
                raise ValueError(f"見つからない: {p}")
        res = check(wav, slm_files(sp), a.laeq, parse_window(a.window), a.tol, a.min_corr, min_dcorr=a.min_dcorr, stop_db=a.stop_db, convention=a.convention)
    except ValueError as e:
        print(f"入力の誤り: {e}")
        return EXIT_INPUT
    for k, v in res.items():
        print(f"{k}: {'' if v is None else v}")           # 使う値・ゲインが無いときは空で出す
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)     # 記録シートの夜の命令（Version 129）は out/realsmoke/calib_check_<id>.json に書く（夜の最初はまだ無い）
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
