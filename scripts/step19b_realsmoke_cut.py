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

使い方:
  # ①イベント切り出し（較正済み24kHzを入力）
  python scripts/step19b_realsmoke_cut.py --mode event \
      --in out/realsmoke/converted --ann ann_orig.csv \
      --out out/realsmoke/clips --ann-out out/realsmoke/ann_clips.csv

  # ②長時間負例の分割（96kHz原本＋別途求めたゲインを入力）
  python scripts/step19b_realsmoke_cut.py --mode negative \
      --in raw/neg_shizuka.wav --ann ann_orig.csv --gain-db 12.3 \
      --out out/realsmoke/clips --ann-out out/realsmoke/ann_neg.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FS_OUT = 24000
DUR = 10.0
CPA_AT = 8.0
OVERLAP = 1.0
MARGIN_S = 0.25          # リサンプルの過渡を逃がす前後マージン
EPS = 1e-9

# 注釈スキーマ（step19c_ann_validate.py と共有する正）
BASE_COLS = ["clip_id", "event_id", "trial", "class", "quadrant", "t_start", "t_cpa"]
CUT_COLS = ["orig_file", "cut_offset_s", "scored"]
DIST_CLASSES = {"car_drive", "kick", "bike"}


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def orig_key(name: str) -> str:
    """ファイル名/クリップ名を原録音キーに正規化（_conv 接尾を落とす）。"""
    stem = Path(name).stem
    return stem[:-5] if stem.endswith("_conv") else stem


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
                target_event=None, own=None):
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
        if t1 < off - EPS or t0 > off + dur + EPS:
            continue                      # 交差なし=このクリップとは無関係
        if target_event is not None:
            scored = int(str(r.get("event_id", "")) == str(target_event))
        elif own is not None:
            scored = int((off + own[0] - EPS) <= t1 < (off + own[1] - EPS))
        else:
            scored = int((off - EPS) <= t1 <= (off + dur + EPS))
        nr = dict(r)
        nr["clip_id"] = new_clip
        nr["t_start"] = f"{min(max(t0 - off, 0.0), dur):.2f}"
        nr["t_cpa"] = f"{min(max(t1 - off, 0.0), dur):.2f}"
        nr["orig_file"] = orig
        nr["cut_offset_s"] = f"{off:.3f}"
        nr["scored"] = str(scored)
        out.append(nr)
    return out


def negative_row(new_clip: str, orig: str, off: float, own0: float, own1: float,
                 template=None):
    r = dict(template or {})
    r.update({"clip_id": new_clip, "event_id": "n1", "class": "none",
              "quadrant": "", "t_start": f"{own0:.3f}", "t_cpa": f"{own1:.3f}",
              "orig_file": orig, "cut_offset_s": f"{off:.3f}", "scored": "1"})
    r.setdefault("trial", "neg")
    return r


# ------------------------------------------------------------------ 音声の切り出し
def cut_audio(src: Path, start_s: float, dur_s: float, out_path: Path,
              gain_db: float = 0.0, fs_out: int = FS_OUT) -> Path:
    """原録音の [start_s, start_s+dur_s) を切り出して fs_out で書き出す。

    ストリーム読みなので長時間ファイルでもメモリに載る。入力fsが fs_out と
    異なる場合のみ、前後 MARGIN_S のマージンを付けてリサンプルしてから
    厳密な区間を取り出す（境界の過渡を持ち込まないため）。"""
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
    if gain_db:
        y = y * (10.0 ** (gain_db / 20.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, fs_out, subtype="PCM_24")
    return out_path


# ------------------------------------------------------------------ main
def _load_ann(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by_orig = {}
    for r in rows:
        by_orig.setdefault(orig_key(r["clip_id"]), []).append(r)
    return rows, by_orig


def _write_ann(path: Path, rows):
    cols = list(BASE_COLS)
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
    assert mode in ("event", "negative"), f"--mode は event|negative: {mode}"

    files = sorted(list(src.glob("*.flac")) + list(src.glob("*.wav"))) \
        if src.is_dir() else [src]
    assert files, f"音声が見つかりません: {src}"
    _, by_orig = _load_ann(ann_path)

    out_rows, n_clip = [], 0
    for path in files:
        orig = orig_key(path.name)
        with sf.SoundFile(str(path)) as f:
            length = len(f) / f.samplerate
        rows = by_orig.get(orig, [])
        if mode == "event":
            events = [r for r in rows if r["class"].strip() != "none"]
            if not events:
                print(f"  skip {path.name}: 正例イベントの注釈がありません")
                continue
            for r in events:
                ev = str(r.get("event_id", "1"))
                off = plan_event_cut(float(r["t_cpa"]), length, dur, cpa_at)
                new_clip = f"{orig}_e{ev}"
                cut_audio(path, off, dur, out_dir / f"{new_clip}.flac", gain_db)
                out_rows += rebase_rows(rows, off, dur, orig, new_clip,
                                        target_event=ev)
                n_clip += 1
                print(f"  {new_clip}: off={off:.3f}s CPA→{float(r['t_cpa'])-off:.2f}s"
                      + ("  ⚠端でクランプ" if abs(off - (float(r['t_cpa']) - cpa_at)) > 1e-6
                         else ""))
        else:
            plan = plan_negative_split(length, dur, overlap)
            tmpl = next((r for r in rows if r["class"].strip() == "none"), None)
            for i, (off, own0, own1) in enumerate(plan):
                new_clip = f"{orig}_s{i:03d}"
                cut_audio(path, off, dur, out_dir / f"{new_clip}.flac", gain_db)
                out_rows.append(negative_row(new_clip, orig, off, own0, own1, tmpl))
                out_rows += rebase_rows(rows, off, dur, orig, new_clip,
                                        own=(own0, own1))
                n_clip += 1
            cov = sum(b - a for _, a, b in plan)
            print(f"  {orig}: {len(plan)}クリップ 担当合計={cov:.2f}s "
                  f"(録音長={length:.2f}s, 差={cov-length:+.3f}s)")

    _write_ann(ann_out, out_rows)
    print(f"\n{mode}: {n_clip}クリップ / 注釈{len(out_rows)}行 -> {out_dir}")
    print(f"注釈CSV -> {ann_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
