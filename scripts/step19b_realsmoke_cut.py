# -*- coding: utf-8 -*-
"""Step 19b: 原録音 → 解析用10秒クリップの切り出し（2026-08-14 実録再設計レビュー反映）。

背景: 現場では1イベントを20〜30秒で録る（10秒ちょうどを狙って録るのは危険で、
現実的でもない）。一方モデル入力は学習と同じ10秒でなければならない。
本スクリプトが両者を橋渡しし、**注釈の時刻を再基準化**して採点可能な形にする。

2モード:
  event    1イベント=1クリップ。最接近(t_cpa)がクリップ内 --cpa-at 秒（既定8.0）に
           来るよう切り出す（最接近前8秒・後2秒）。原録音の端でクランプされた場合も
           cut_offset_s に実値が残るので時刻参照は壊れない。
  negative 長時間負例を --overlap 秒（既定1.0）の重複つき10秒へ分割する。
           重複を設ける理由は、通知の連続フレーム条件（距離トリガ2フレーム・
           警告音3フレーム）がクリップ境界で分断され**検知漏れ**になるのを防ぐため。
           同時に、重複帯の発火が**二重計上**されないよう、各クリップの担当区間
           （注釈の t_start〜t_cpa）が原録音上で**隙間なく・重なりなく**タイルする
           ように出力する。担当は原則「前クリップ」（後クリップの先頭 --overlap 秒は
           担当外）。採点側 step20 の負例窓は半開区間 [t_start, t_cpa) で数えるため、
           境界時刻ちょうどの発火も一意に1回だけ数えられる。

較正ゲインの継承:
  既定は step19_realsmoke_convert.py が出力した**較正済み24kHz**ファイルを入力に取る。
  切り出しは較正後の波形をスライスするだけなので、LAeqゲインは自動的に継承される。
  長時間の負例など、原本(96kHz)を丸ごとメモリに載せられない場合は
  `step19 --gain-only` でゲインだけ先に求め、本スクリプトに `--gain-db` で渡す
  （このとき入力は96kHz原本でよく、切り出した10秒だけを24kHzへ変換する）。
  24kHz変換済みでない入力では --gain-db/--pitch/--roll/--yaw の明示指定を必須とし、
  指定忘れをエラーにする（値が0のときも各引数を明示する）。

注釈の計画列:
  take_id / 区分 / 状態は必須。歩行区分では pair_id も必須。
  切り出し後は orig_file / orig_duration_s / cut_offset_s / scored を自動付与する。
  連続負例100分は区分=負例露出とし、計画120テイクとは別に数える。

使い方:
  # ①イベント切り出し（較正済み24kHzを入力）
  python scripts/step19b_realsmoke_cut.py --mode event \
      --in out/realsmoke/converted --ann ann_orig.csv \
      --out out/realsmoke/clips --ann-out out/realsmoke/ann_clips.csv

  # ②長時間負例の分割（96kHz原本＋校正 json）。--in が 1 ファイルなら照合はそのファイルだけ（--only a.wav,b.wav でも同じ）
  python scripts/step19b_realsmoke_cut.py --mode negative \
      --in raw/neg_shizuka.wav --ann ann_orig.csv --calib-dir out/realsmoke/conv --pitch 0 --roll 0 --yaw 0 \
      --out out/realsmoke/clips --ann-out out/realsmoke/ann_all.csv --append
  # 出力名（2026-09-22〜）= eval_field_clip<セッション番号 2 桁><セッション内の通し番号 3 桁>。番号は step19d の <ann>_sessions.csv の session_no と切り出し順。
  # 一覧 eval_field_index.md を注釈の隣に出す。同じ原本名が複数セッションにあるときは --session <session_id> で 1 セッションずつ（旧: session__原本_eN）
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 傾き補正の回転行列は step19 の定義を**唯一の正**として共有する（二重定義で
# 座標系がずれるのを防ぐ）。原本96kHzから直接切り出す経路でも、通常変換と
# まったく同じ回転を適用しなければ、その録音だけ方位がずれる。
_spec = importlib.util.spec_from_file_location(
    "_s19conv", Path(__file__).with_name("step19_realsmoke_convert.py"))
_s19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s19)
rot_matrix = _s19.rot_matrix

FS_OUT = 24000
DUR = 10.0
CPA_AT = 8.0
OVERLAP = 1.0
MARGIN_S = 0.25          # リサンプルの過渡を逃がす前後マージン
EPS = 1e-9

# 注釈スキーマ（step19c_ann_validate.py と共有する正）
BASE_COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa"]
PLAN_COLS = ["take_id", "pair_id", "区分", "状態"]
CUT_COLS = ["orig_file", "orig_duration_s", "cut_offset_s", "scored"]
# 2026-09-06（監査 R01/R10）: 校正の引継ぎと履歴不足のフラグ。calibration_id は step19 --calib の id、gain_db は適用した補正量、
# history_short は「最接近がクリップ内 cpa_at − 0.5 s より早い」（録音開始が遅く CPA 前の履歴が足りない）= 1
CALIB_COLS = ["calibration_id", "gain_db", "history_short"]
DIST_CLASSES = {"car_drive", "kick", "bike", "bicycle"}   # bicycle = 2026-10-08 変更票 #48


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def orig_key(name: str) -> str:
    """ファイル名/クリップ名を原録音キーに正規化（_conv 接尾を落とす）。"""
    stem = Path(name).stem
    return stem[:-5] if stem.endswith("_conv") else stem


def require_raw_metadata(input_fs: int, argv=None, have_calib: bool = False) -> None:
    """24kHz変換済みでない入力では、較正値と3軸角の明示指定を必須にする。

    have_calib=True（--calib-dir の有効な校正 json がある）なら --gain-db は不要（補正量は json から取る・再監査2 Q01）。

    角度0°やゲイン0dB自体は有効なので値ではなく、CLIで明示された事実を検査する。
    これにより原本を直接切る際の「指定し忘れ」を黙って通さない。"""
    if input_fs == FS_OUT:
        return
    argv = sys.argv if argv is None else argv
    required = ("--pitch", "--roll", "--yaw") if have_calib else ("--gain-db", "--pitch", "--roll", "--yaw")
    missing = [name for name in required if name not in argv]
    if missing:
        raise ValueError("原本から直接切り出す場合は較正ゲインと回転角を明示してください: "
                         + " ".join(missing))


# ------------------------------------------------------------------ 切り出し計画
def plan_event_cut(t_cpa: float, length_s: float, dur: float = DUR,
                   cpa_at: float = CPA_AT) -> float:
    """最接近が clip 内 cpa_at 秒に来る開始オフセット[s]。原録音の端でクランプ。

    返り値は必ず 0 <= off <= length-dur。length < dur は呼び出し側で弾く。"""
    if length_s + EPS < dur:
        raise ValueError(f"原録音が短すぎます（{length_s:.2f}s < {dur:.2f}s）")
    off = t_cpa - cpa_at
    return float(min(max(off, 0.0), length_s - dur))


def plan_negative_split(length_s: float, dur: float = DUR,
                        overlap: float = OVERLAP):
    """長時間負例の分割計画。

    返り値: [(start_s, own_start_local, own_end_local)]
      start_s          : 原録音上の切り出し開始[s]（音声は必ず dur 秒ぶん取る）
      own_start_local  : そのクリップが誤警告を数える担当区間の開始[s]（クリップ内時刻）
      own_end_local    : 同・終了[s]
    担当区間は原録音上で隙間なく・重なりなくタイルする（合計 = length_s）。"""
    if length_s + EPS < dur:
        raise ValueError(f"負例が短すぎます（{length_s:.2f}s < {dur:.2f}s）")
    assert 0.0 <= overlap < dur, f"overlapは[0,dur)で指定してください: {overlap}"
    step = dur - overlap
    starts = []
    s = 0.0
    while s + dur < length_s - EPS:
        starts.append(s)
        s += step
    starts.append(min(s, length_s - dur) if starts else 0.0)
    # 末尾クリップは length-dur へ寄せる（常に dur 秒の入力を確保する）
    starts[-1] = length_s - dur
    starts = [x for i, x in enumerate(starts) if i == 0 or x > starts[i - 1] + EPS]

    # 担当境界 = 「後クリップの先頭 overlap 秒が終わる時刻」。前クリップはそこまでを
    # 担当する（=重複帯は前クリップ帰属）。末尾クリップは録音終端まで担当する。
    plan = []
    for i, st in enumerate(starts):
        own_start_abs = 0.0 if i == 0 else st + overlap
        own_end_abs = (starts[i + 1] + overlap) if i + 1 < len(starts) else length_s
        own_end_abs = min(own_end_abs, st + dur)
        plan.append((float(st), float(own_start_abs - st), float(own_end_abs - st)))
    return plan


# ------------------------------------------------------------------ 注釈の再基準化
def rebase_rows(rows, off: float, dur: float, orig: str, new_clip: str,
                target_event=None, own=None, orig_duration_s=None,
                calibration_id: str = "", gain_db: float = 0.0, cpa_at: float = CPA_AT):
    """原録音時刻の注釈行を、切り出しクリップの時刻へ移す。

    重要: **1つの物理イベントはちょうど1クリップでだけ採点される**ようにする。
    切り出し窓は重なるので、素朴に「クリップに入っていれば採点」とすると同じ車が
    2回数えられ、対応あり検定の独立性が壊れる。採点担当の決め方:
      event 系   : target_event（その切り出しの主対象）だけ scored=1
      negative系 : own=(担当開始,担当終了) を渡し、**最接近がその担当区間に入る**
                   行だけ scored=1（担当区間は原録音上でタイルするので一意）
    採点しない行も**消さずに scored=0 で残す**（消すと本物のイベントを誤警告に
    数えてしまう＝マスク専用として必要）。t_start がクリップ外へはみ出す場合は
    0 にクランプする（音は先頭から入っているので採点は成立する）。"""
    out = []
    for r in rows:
        if r["class"].strip() == "none":
            continue
        t0, t1 = float(r["t_start"]), float(r["t_cpa"])
        # Astra R72-02（2026-10-20）: 区間の終わり（長回しの相手の t_end・終わり不明なら録音末尾まで）と動き出し t_move_s も原録音の秒 → クリップの秒へ移す。交差の判定は実際の区間の終端で
        t_end = float(r["t_end"]) if (r.get("t_end") or "").strip() else None
        unknown = (r.get("t_end_unknown") or "").strip() == "1"
        end_abs = float("inf") if unknown else (t_end if t_end is not None else t1)
        if end_abs < off - EPS or t0 > off + dur + EPS:
            continue                      # 交差なし=このクリップとは無関係
        if target_event is not None:
            scored = int(str(r.get("event_id", "")) == str(target_event))
        elif own is not None:
            scored = int((off + own[0] - EPS) <= t1 < (off + own[1] - EPS))
        else:
            scored = int((off - EPS) <= t1 <= (off + dur + EPS))
        nr = dict(r)
        nr["clip_id"] = new_clip
        if t_end is not None or unknown:
            nr["t_start"] = f"{t0 - off:.2f}"; nr["t_cpa"] = f"{t1 - off:.2f}"      # マスク専用の区間はクランプしない（切り出しの外へはみ出す分は step20 が区間の交わりで扱う）
        else:
            nr["t_start"] = f"{min(max(t0 - off, 0.0), dur):.2f}"
            nr["t_cpa"] = f"{min(max(t1 - off, 0.0), dur):.2f}"
        nr["t_end"] = (f"{t_end - off:.2f}" if t_end is not None else "")
        nr["t_end_unknown"] = ("1" if unknown else "")
        if (r.get("t_move_s") or "").strip():
            nr["t_move_s"] = f"{float(r['t_move_s']) - off:.2f}"                                  # 0 に丸めない（動き出しの前後の判定に使う）
        nr["orig_file"] = orig
        nr["orig_duration_s"] = (f"{orig_duration_s:.3f}"
                                 if orig_duration_s is not None else "")
        nr["cut_offset_s"] = f"{off:.3f}"
        nr["scored"] = str(scored)
        nr["calibration_id"] = calibration_id or r.get("calibration_id", "")
        nr["gain_db"] = f"{gain_db:.2f}"
        nr["history_short"] = str(int(scored == 1 and (t1 - off) < cpa_at - 0.5))
        out.append(nr)
    return out


def negative_row(new_clip: str, orig: str, off: float, own0: float, own1: float,
                 template=None, orig_duration_s=None, calibration_id: str = "", gain_db: float = 0.0):
    r = dict(template or {})
    r.update({"calibration_id": calibration_id or r.get("calibration_id", ""), "gain_db": f"{gain_db:.2f}", "history_short": "0"})
    r.update({"clip_id": new_clip, "event_id": "n1", "class": "none",
              "quadrant": "", "t_start": f"{own0:.3f}", "t_cpa": f"{own1:.3f}",
              "orig_file": orig,
              "orig_duration_s": (f"{orig_duration_s:.3f}"
                                  if orig_duration_s is not None else ""),
              "cut_offset_s": f"{off:.3f}", "scored": "1"})
    r.setdefault("trial", "neg")
    return r


# ------------------------------------------------------------------ 音声の切り出し
def cut_audio(src: Path, start_s: float, dur_s: float, out_path: Path,
              gain_db: float = 0.0, fs_out: int = FS_OUT,
              pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0) -> Path:
    """原録音の [start_s, start_s+dur_s) を切り出して fs_out で書き出す。

    ストリーム読みなので長時間ファイルでもメモリに載る。入力fsが fs_out と
    異なる場合のみ、前後 MARGIN_S のマージンを付けてリサンプルしてから
    厳密な区間を取り出す（境界の過渡を持ち込まないため）。

    pitch/roll/yaw: **原本(96kHz)から直接切り出す経路で必須**。step19の通常変換で
    掛かるはずの傾き補正をここで掛けないと、その録音だけ方位座標が他とずれる。
    step19変換済みファイルを入力にする場合は既に補正済みなので 0 のままにする。"""
    with sf.SoundFile(str(src)) as f:
        fs, n_total, nch = f.samplerate, len(f), f.channels
        assert nch == 4, f"{src.name}: 4ch(AmbiX)ではありません ch={nch}"
        assert fs in (96000, 48000, 24000), f"{src.name}: 想定外のfs={fs}"
        a = int(round(start_s * fs))
        n = int(round(dur_s * fs))
        assert a >= 0 and a + n <= n_total + 1, \
            f"{src.name}: 切り出し範囲が録音長を超えています"
        n = min(n, n_total - a)
        m = 0 if fs == fs_out else int(round(MARGIN_S * fs))
        a0, b0 = max(0, a - m), min(n_total, a + n + m)
        f.seek(a0)
        buf = f.read(b0 - a0, dtype="float64", always_2d=True)
    lead = a - a0
    down = fs // fs_out
    if down > 1:
        buf = resample_poly(buf, 1, down, axis=0)
        lead = int(round(lead / down))
        n = int(round(n / down))
    y = buf[lead:lead + n]
    if len(y) < n:                        # 端数はゼロ詰め（末尾クリップのみ発生しうる）
        y = np.vstack([y, np.zeros((n - len(y), 4))])
    if pitch or roll or yaw:              # 傾き補正（W不変・ch順 W,Y,Z,X）
        y = y.copy()
        R = rot_matrix(pitch, roll, yaw)
        xyz = y[:, [3, 1, 2]] @ R.T
        y[:, 3], y[:, 1], y[:, 2] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if gain_db:
        y = y * (10.0 ** (gain_db / 20.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, fs_out, subtype="PCM_24")
    return out_path


# ------------------------------------------------------------------ main
WARN_CLASSES = {"siren", "horn", "backup_beep", "bike_bell", "crossing"}


def unfilled_positive_rows(rows) -> list[str]:
    """時刻が未確定の正例（変更票 #59 r6・Astra FR11-01・2026-10-12）: class ≠ none で t_cpa が空か 0 以下、警告音は t_start も 0 以下。
    ラップ秒が空のまま変換された注釈（旧 step19d は 0 秒にして進めた）を切り出さないための門。負例（none）は見ない。"""
    import math as _math

    def _f(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if _math.isfinite(x) else None       # nan／inf は未確定（Astra NT12-02）
    bad = []
    for r in rows:
        cls = (r.get("class") or "").strip()
        if cls in ("", "none"):
            continue
        t1, t0 = _f(r.get("t_cpa")), _f(r.get("t_start"))
        why = None
        if t1 is None or t1 <= 0:
            why = f"t_cpa='{r.get('t_cpa', '')}'"
        elif cls in WARN_CLASSES and (t0 is None or t0 <= 0):
            why = f"t_start='{r.get('t_start', '')}'（警告音の鳴り始めが未確定）"
        if why:
            bad.append(f"{(r.get('take_id') or r.get('clip_id') or '?')} event {r.get('event_id', '')} {cls}: {why}")
    return bad


def _load_ann(path: Path):
    """注釈を原録音キーで引く。orig_file 列（任意の原本名・再監査 N01）があればそれを、無ければ clip_id を使う。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by_orig = {}
    for r in rows:
        src = (r.get("orig_file") or "").strip() or r["clip_id"]
        by_orig.setdefault(orig_key(src), []).append(r)
    return rows, by_orig


def load_calib_dir(calib_dir: Path) -> dict:
    """step19 --calib が書いた calibration_<id>.json を全部読む: id -> 記録（gain_db, laeq_window_s, calib_file）（再監査 N04・Q02）。"""
    import json
    out = {}
    for f in sorted(Path(calib_dir).glob("calibration_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["gain_db"] = float(rec["gain_db"])
        out[rec["calibration_id"]] = rec
    return out


def _parse_win(txt: str):
    a, b = txt.replace("〜", "-").split("-")
    return float(a), float(b)


def calib_for_rows(rows, calib_map: dict, cli_id: str, cli_gain: float):
    """その原本の注釈行から校正 id を 1 つに決め、補正量を返す。行間で id が違う／CLI と食い違う／json に無いときは停止。"""
    ids = {(r.get("calibration_id") or "").strip() for r in rows} - {""}
    if len(ids) > 1:
        raise ValueError(f"同じ原本の注釈行で calibration_id が複数あります: {sorted(ids)}")
    cid = next(iter(ids)) if ids else cli_id
    if cli_id and ids and cli_id != cid:
        raise ValueError(f"--calibration-id {cli_id} と注釈の calibration_id {cid} が違います（校正の取り違え防止・N04）")
    if calib_map:
        if cid not in calib_map:
            raise ValueError(f"calibration_id {cid!r} の json が --calib-dir にありません（step19 --calib で作る）")
        rec = calib_map[cid]; gain = float(rec["gain_db"])
        if cli_gain and abs(cli_gain - gain) > 0.05:
            raise ValueError(f"--gain-db {cli_gain:+.2f} と json の補正量 {gain:+.2f}（{cid}）が違います")
        # 記録紙の暗騒音区間（騒音計で測った区間）と json の窓が同じ区間か（再監査2 Q02）
        jw = rec.get("laeq_window_s")
        for wtxt in {(r.get("暗騒音区間_秒") or "").strip() for r in rows} - {""}:
            try:
                a, b = _parse_win(wtxt)
            except ValueError:
                raise ValueError(f"暗騒音区間_秒 '{wtxt}' の形式が違います（例 0-60）")
            if jw and (abs(a - float(jw[0])) > 0.5 or abs(b - float(jw[1])) > 0.5):
                raise ValueError(f"校正 {cid}: 記録紙の暗騒音区間 {wtxt} と json の窓 {float(jw[0]):g}-{float(jw[1]):g} が違います"
                                 f"（騒音計で測った区間を step19 --laeq-window に渡す・再監査2 Q02）")
        return cid, gain
    return cid, cli_gain


def _write_index(ann_path: Path) -> Path:
    """eval_field_index.md（注釈 CSV の隣）: 1 本 1 行で「名前 → セッション・テイク・事象・クラス・区分・横距離・状態・原本・切り出し位置」。
    名前（eval_field_clip<SS><NNN>）から中身を引くための表（2026-09-22・命名 A の手当て）。採点対象の事象の行を優先する。"""
    rows = list(csv.DictReader(open(ann_path, encoding="utf-8-sig")))
    by = {}
    for r in rows:
        c = r["clip_id"]
        if c not in by or (by[c].get("scored") != "1" and r.get("scored") == "1"):
            by[c] = r
    L = ["# 実録クリップの一覧（自動生成: step19b_realsmoke_cut.py）", "",
         "名前 = `eval_field_clip<セッション番号 2 桁><セッション内の通し番号 3 桁>`（セッション番号は session.csv の登録順・通し番号は切り出し順）。1 本 1 行（採点対象の事象を優先）。", "",
         "| clip_id | session_id | take | event | class | 区分/trial | 横距離m | 状態 | 原本 | 切り出し位置 s |",
         "|" + " --- |" * 10]
    for c in sorted(by):
        r = by[c]
        L.append(f"| {c} | {r.get('session_id', '')} | {r.get('take_id', '')} | {r.get('event_id', '')} | {r.get('class', '')} | "
                 f"{r.get('区分', '') or r.get('trial', '')} | {r.get('横距離m', '')} | {r.get('状態', '')} | {r.get('orig_file', '')} | {r.get('cut_offset_s', '')} |")
    p = ann_path.with_name("eval_field_index.md")
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def _write_ann(path: Path, rows, append: bool = False):
    """--append 時は既存CSVの行を先に読み込み、列を和集合にして書き直す。

    負例の原録音が複数ファイルに分かれる（静穏30分・繁華街30分…）ため、
    同じ注釈CSVへ追記できないと採点前の統合作業が手作業になる。"""
    if append and path.exists():
        prev = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        # 同じ clip_id が別セッションの既存行を上書きしない（再監査2 Q03）
        prev_sess = {r["clip_id"]: r.get("session_id", "") for r in prev if r.get("session_id")}
        for r in rows:
            ps = prev_sess.get(r["clip_id"])
            if ps and r.get("session_id") and ps != r.get("session_id"):
                raise ValueError(f"--append: clip_id {r['clip_id']} は既存の別セッション {ps} の行と衝突します（再監査2 Q03）")
        seen = {(r["clip_id"], r.get("event_id", "")) for r in rows}
        rows = [r for r in prev
                if (r["clip_id"], r.get("event_id", "")) not in seen] + list(rows)
    cols = list(BASE_COLS)
    for c in PLAN_COLS:
        if any(c in r for r in rows):
            cols.append(c)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    for k in CUT_COLS:
        if k not in cols:
            cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> int:
    mode = _arg("--mode", "event")
    src = Path(_arg("--in"))
    ann_path = Path(_arg("--ann"))
    out_dir = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "clips")))
    ann_out = Path(_arg("--ann-out", str(out_dir / f"ann_{mode}.csv")))
    dur = float(_arg("--dur", str(DUR)))
    cpa_at = float(_arg("--cpa-at", str(CPA_AT)))
    overlap = float(_arg("--overlap", str(OVERLAP)))
    gain_db = float(_arg("--gain-db", "0"))
    calibration_id = _arg("--calibration-id", "")
    cli_calib_id = calibration_id          # CLI で宣言した id（ファイルごとの id とは別に保持する）
    calib_map = {}
    if _arg("--calib-dir"):
        cdir = Path(_arg("--calib-dir"))
        calib_map = load_calib_dir(cdir) if cdir.is_dir() else {}
        if not calib_map:
            print(f"❌ --calib-dir {cdir} に calibration_*.json がありません（step19 --calib で作る）。0 dB で続行しない（再監査2 Q02）")
            return 1
    allow_unmatched = "--allow-unmatched" in sys.argv
    only_set = {orig_key(x.strip()) for x in _arg("--only", "").split(",") if x.strip()}   # この呼出しで扱う原本だけ（再監査2 Q01）
    session_filter = _arg("--session", "")                                                # 同名原本が複数セッションにあるとき（再監査2 Q03）
    if gain_db and not calibration_id and not calib_map:
        print("  ⚠️ --gain-db を渡すときは --calibration-id（step19 --calib の id）か --calib-dir も渡してください（校正の引継ぎ・監査 R01/N04）")
    pitch = float(_arg("--pitch", "0"))
    roll = float(_arg("--roll", "0"))
    yaw = float(_arg("--yaw", "0"))
    append = "--append" in sys.argv
    assert mode in ("event", "negative"), f"--mode は event|negative: {mode}"
    # 2026-09-22 実録クリップの命名（本人決定・案 A）: eval_field_clip<セッション番号 2 桁><セッション内の通し番号 3 桁>
    sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.clip_ids import is_new as _is_new, make_field as _make_field, parse_field as _parse_field
    rot = dict(pitch=pitch, roll=roll, yaw=yaw)
    print(f"# mode={mode} gain={gain_db:+.2f}dB "
          f"回転(pitch/roll/yaw)={pitch:g}/{roll:g}/{yaw:g}"
          + ("  ※原本から切る場合は step19 と同じ回転角を必ず渡すこと"
             if gain_db and not (pitch or roll or yaw) else ""))

    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".flac")) \
        if src.is_dir() else [src]
    if only_set:
        files = [f for f in files if orig_key(f.name) in only_set]
    assert files, f"音声が見つかりません: {src}"
    _all_rows, by_orig = _load_ann(ann_path)
    if session_filter:
        _all_rows = [r for r in _all_rows if (r.get("session_id") or "") == session_filter]
        by_orig = {}
        for r in _all_rows:
            by_orig.setdefault(orig_key((r.get("orig_file") or "").strip() or r["clip_id"]), []).append(r)
        print(f"--session {session_filter}: 注釈 {len(_all_rows)} 行に限定")
    dup = {k: sorted({r.get("session_id", "") for r in rs}) for k, rs in by_orig.items() if len({r.get("session_id", "") for r in rs}) > 1}
    if dup:
        print(f"❌ 同じ原本名が複数セッションにあります {dict(list(dup.items())[:3])} → --session <session_id> で 1 セッションずつ切り出す（再監査2 Q03）")
        return 1
    # 変更票 #59 r6・Astra FR11-01（2026-10-12）: 時刻が未確定の正例は切り出さない（ラップ秒が空のまま 0 秒で切ると、相手の音がクリップから消え、本数の検査も通ってしまう）
    bad = unfilled_positive_rows(_all_rows)
    if bad:
        print(f"❌ 時刻が未確定の正例が {len(bad)} 行あります（t_cpa が空か 0 以下・警告音は t_start も）→ 切り出さない。動画の目印と録音の波形から ラップ秒 を埋め、step19d を通し直す（変更票 #59 r6）")
        for m in bad[:30]:
            print("   " + m)
        return 1

    out_rows, n_clip = [], 0
    matched_origs, skipped_files = set(), []
    calib_names = {orig_key(r.get("calib_file", "")) for r in _all_rows if r.get("calib_file")}
    check_names = {orig_key(r.get("check_file", "")) for r in _all_rows if r.get("check_file")}   # 点検録音（step19e で使う・監査 H01）
    excl_names = {orig_key(x) for r in _all_rows for x in (r.get("excluded_files", "") or "").split()}   # 記録紙の備考「除外: ...」（監査 H02/H04）
    # step19d が注釈の隣に置くセッション表（<ann>_sessions.csv）。事象が 0 行のセッションでも校正・点検・除外の原本名が分かる
    side = ann_path.with_name(ann_path.stem + "_sessions.csv")
    if not side.exists() and ann_path.stem.endswith("_tuning"):
        # step19d splits annotations by purpose, but emits one shared session table.
        # An explicitly supplied tuning-specific table still takes precedence.
        side = ann_path.with_name(ann_path.stem.removesuffix("_tuning") + "_sessions.csv")
    sess_no = {}                                  # session_id -> セッション番号（step19d の表の session_no・登録順）
    if side.exists():
        with open(side, encoding="utf-8-sig", newline="") as fh:
            srows = list(csv.DictReader(fh))
        calib_names |= {orig_key(r.get("calib_file", "")) for r in srows if r.get("calib_file")}
        check_names |= {orig_key(r.get("check_file", "")) for r in srows if r.get("check_file")}
        excl_names |= {orig_key(x) for r in srows for x in (r.get("excluded_files", "") or "").split()}
        sess_no = {r["session_id"]: int(r["session_no"]) for r in srows if (r.get("session_no") or "").strip()}
        print(f"セッション表 {side.name}: {len(srows)} セッション（校正 {len(calib_names)}・点検 {len(check_names)}・除外 {len(excl_names)} 本を切り出し対象から外す）")
    for r in _all_rows:                           # 表に無いセッションは出現順に末尾の番号（step19d を通し直すのが正）
        sid = (r.get("session_id") or "").strip()
        if sid and sid not in sess_no:
            sess_no[sid] = max(sess_no.values(), default=0) + 1
            print(f"  ⚠️ セッション表に session_no が無い: {sid} → 仮に {sess_no[sid]} 番。step19d を通し直して表を作ってください")
    counters = {}                                 # セッション番号 -> 最後に使った通し番号（--append なら既存の注釈から続ける）
    if append and ann_out.exists():
        for r in csv.DictReader(open(ann_out, encoding="utf-8-sig")):
            c = (r.get("clip_id") or "").strip()
            if _is_new(c) and "_field_" in c:
                s_no, k_no = _parse_field(c)
                counters[s_no] = max(counters.get(s_no, 0), k_no)

    def _sid_of(rows_, r_):
        sid = ((r_ or {}).get("session_id") or "").strip() or ((rows_[0].get("session_id") or "").strip() if rows_ else "")
        if not sid:
            raise ValueError("注釈に session_id が無い行があります（実録の命名にはセッションが要る。step19d の出力を使う）")
        return sid

    def _next_clip(sid):
        s_no = sess_no[sid]
        counters[s_no] = counters.get(s_no, 0) + 1
        return _make_field(s_no, counters[s_no]), s_no
    for path in files:
        orig = orig_key(path.name)
        if orig.endswith("_calib") or orig in calib_names or orig in check_names:
            continue                      # 校正録音（step19 --calib）・点検録音（step19e）は切り出し対象ではない
        if orig in excl_names:
            print(f"  除外（記録紙の備考の指示）: {path.name}")
            continue
        with sf.SoundFile(str(path)) as f:
            length = len(f) / f.samplerate
            input_fs = f.samplerate
        require_raw_metadata(input_fs, have_calib=bool(calib_map))
        rows = by_orig.get(orig, [])
        base = ((rows[0].get("clip_id") or "").strip() or orig) if rows else orig   # 出力名の土台 = 注釈の clip_id（session__原本・再監査2 Q03）
        if not rows:
            skipped_files.append(path.name)
        else:
            matched_origs.add(orig)
        file_gain = gain_db
        if rows:
            calibration_id, file_gain = calib_for_rows(rows, calib_map, cli_calib_id, gain_db)
            gain_db_file = file_gain
        else:
            gain_db_file = gain_db
        if mode == "event":
            events = [r for r in rows if r["class"].strip() != "none"]
            if not events:
                print(f"  skip {path.name}: 正例イベントの注釈がありません")
                continue
            for r in events:
                ev = str(r.get("event_id", "1"))
                off = plan_event_cut(float(r["t_cpa"]), length, dur, cpa_at)
                new_clip, s_no = _next_clip(_sid_of(rows, r))          # 旧: f"{base}_e{ev}"（原録音名＋事象番号）。同じ入力で通し直せば同じ名前（上書き）
                cut_audio(path, off, dur, out_dir / f"{new_clip}.flac", gain_db_file,
                          **rot)
                added = rebase_rows(rows, off, dur, orig, new_clip,
                                    target_event=ev, orig_duration_s=length,
                                    calibration_id=calibration_id, gain_db=gain_db_file, cpa_at=cpa_at)
                for x in added:
                    x["session_no"] = str(s_no)
                out_rows += added
                n_clip += 1
                print(f"  {new_clip}: off={off:.3f}s CPA→{float(r['t_cpa'])-off:.2f}s"
                      + ("  ⚠端でクランプ" if abs(off - (float(r['t_cpa']) - cpa_at)) > 1e-6
                         else ""))
        else:
            tmpl = next((r for r in rows if r["class"].strip() == "none"), None)
            if tmpl is None:
                # 2026-10-25（敵対レビュー 重大 2）: 負例の行（class=none）が無い原本は負例として切らない。旧は正例の本も 10 秒に分けて
                # 区分・take_id の無い none の行を作り（同じ車を 2 回採点）、注釈の無い録音では _sid_of で止まっていた。注釈の無い録音は下の照合で数える
                print(f"  skip {path.name}: 負例（class=none）の注釈が無い（正例の本か、注釈の無い録音）→ 負例としては切らない")
                continue
            plan = plan_negative_split(length, dur, overlap)
            for i, (off, own0, own1) in enumerate(plan):
                sid = _sid_of(rows, tmpl)
                new_clip, s_no = _next_clip(sid)                       # 旧: f"{base}_s{i:03d}"
                cut_audio(path, off, dur, out_dir / f"{new_clip}.flac", gain_db_file,
                          **rot)
                nrow = negative_row(new_clip, orig, off, own0, own1, tmpl,
                                    orig_duration_s=length, calibration_id=calibration_id, gain_db=gain_db_file)
                nrow.setdefault("session_id", sid); nrow["session_no"] = str(s_no)
                added = rebase_rows(rows, off, dur, orig, new_clip,
                                    own=(own0, own1), orig_duration_s=length,
                                    calibration_id=calibration_id, gain_db=gain_db_file, cpa_at=cpa_at)
                for x in added:
                    x["session_no"] = str(s_no)
                out_rows.append(nrow)
                out_rows += added
                n_clip += 1
            cov = sum(b - a for _, a, b in plan)
            print(f"  {orig}: {len(plan)}クリップ 担当合計={cov:.2f}s "
                  f"(録音長={length:.2f}s, 差={cov-length:+.3f}s)")

    _write_ann(ann_out, out_rows, append=append)
    idx = _write_index(ann_out)
    print(f"\n{mode}: {n_clip}クリップ / 注釈{len(out_rows)}行 -> {out_dir}")
    print(f"注釈CSV -> {ann_out}" + ("（既存行へ追記）" if append else "") + f" / 一覧 -> {idx.name}")
    # 原本と注釈の対応の照合（再監査 N01）: 注釈があるのに音声が無い原本、音声があるのに注釈が無いファイル
    want = {k for k, rs in by_orig.items() if any(r["class"].strip() != "none" for r in rs)} if mode == "event" else set(by_orig)
    if only_set or not src.is_dir():
        want &= {orig_key(f.name) for f in files}   # 1 回の処理対象だけを照合（全体の完全性は step19c --cut で見る・再監査2 Q01）
    missing_audio = sorted(want - matched_origs)
    if skipped_files or missing_audio:
        print(f"⚠️ 対応の不一致: 注釈があるのに音声が見つからない原本 {len(missing_audio)} 件 {missing_audio[:5]} / "
              f"注釈が無くて飛ばした音声 {len(skipped_files)} 件 {skipped_files[:5]}")
        if not allow_unmatched:
            print("   → 原本名（orig_file）と音声ファイル名の stem を合わせてください。意図どおりなら --allow-unmatched で続行")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
