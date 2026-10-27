# -*- coding: utf-8 -*-
"""実録の台帳の下書き（録った日の夜・2026-10-25 Opus・Fable の確認待ち）。
敵対レビュー（md/audit/敵対的レビュー_Opus55_2026-10-25_Opusの実装.md）の軽 6 を直した版（2026-10-25 Opus）。

10/15・10/22 は使い捨てのスクリプトで作った台帳（raw/ledger/ledger_<日付>.csv）の、機械的にできる所だけを道具にした。
特別な場合（押し間違い・番号のずれ・校正の取り違え）の最後の判断は人（Opus）が行い、下書きを見て ledger_<日付>.csv を作る。

やること（中身＝音と映像は読まない・人の名前が入りうる欄〔helper など・使わないの理由の文〕は写さない）:
  1 録音の一覧 wav_<日付>.csv: raw/<YYMMDD>_*.WAV の bext の録り始め・長さ（10/22 と同じ列）。
     RF64（4 GB 超の WAV）は ds64 の長さで読む。読めない録音（WAV の見出しが無い・壊れた）も行を残し（中身の欄は空）、疑いの一覧に理由を出す
  2 動画の一覧 mov_<日付>.csv: raw/video/*.MOV のうち撮り始め（mvhd・日本時間）がその日のもの。
     mvhd の読めない動画はどの日の撮影か分からないので、行を残し（時刻の欄は空）、疑いの一覧に出す（黙って落とさない）
  3 台帳の下書き ledger_<日付>_draft.csv（列 録音,開始,秒,動画,時刻が重なる動画,役割）: 記録シートのバックアップ（helmet_note_backup_<日付>.json）から
     本（takes の file）→「本 <セッション> <台本ID>」・使わない（excluded）→「使わない <セッション>: 理由あり」（理由の文は写さない・空なら「理由なし」）・
     校正（calibFile）→「校正 <セッション>」・点検（checks）→「点検 <セッション>（向き°・方式）」・どれにも無い →「要確認: 記録シートに無い」・
     2 つ以上の役割 →「要確認: 役割が重なる」
  4 疑いの一覧 ledger_<日付>_check.txt:
     - H3-VR の時計のずれ（押した時刻と録り始めの差が一番多く集まる所の中央値）と、推定に使った本の数・支えた本の数（集まった所の ±2 s に入った本）。
       支えが 3 本未満、または |ずれ| > 10 s なら疑いに出す（本が 1 本だけの日は、ずれがその本の差そのものになり、その本の番号ずれを吸い込んで出せない）
     - 番号ずれ: 時計のずれを差し引いて、押した時刻に一番近い録音が記録シートの録音と違う本
     - 動画: 本の動画の撮影の時間が録音の時間と重ならない本。録音の時刻が読めない（bext が無い）・動画の時刻が読めない（mvhd が無い）・
       動画が見つからない、は理由を分けて出す（動画の名前の大文字・小文字は区別しない）
     - 短い録音（5 秒未満＝押し間違いの可能性）・60 秒より短い校正・読めない録音・RF64 の録音・記録シートにあるのに録音が無い
     - 録音機の設定（bext の coding history の Rec Mode・Mic Position・Correction Filter）が 10/15・10/22 の 191 本（AmbiX・Upright・Off）と違う本
     - 割れの疑い（フルスケール近くの標本が 4 個以上で、50 ms 以上続く割れ＝時刻つき）・一瞬の割れだけの録音（拍手・指パッチンの見込み・1 行にまとめる）・
       リミッターの疑い（山が −2 dBFS 付近に張り付く）。音の山だけを読む（中身は聞かない・--no-peaks で読まない）。
       どれも「疑い」まで（Astra P25-10・10/25）: 50 ms は音の種類を分けない（短いホーンも拍手もある）・ゆがんでいない −2 dBFS の大きな音でも張り付きの個数は 1,000 を超える（架空の正弦波で 41,600）→ 録り直しや除外を自動で決めない・本体の設定（リミッターの ON／OFF・閾値）の記録と本番の窓で確かめる。
       2026-10-25 前提の敵対的レビュー（実録 重大 3・中 4）: クラクションはゲイン 70 で割れる見込み・リミッターの状態はファイルに残らない
出力 4 つ（wav_・mov_・ledger_<日付>_draft.csv・ledger_<日付>_check.txt）は上書きしない: 1 つでもあれば、書き始める前に全部を挙げて止める。
--force のときは、あるものを全部 history/ に写してから書く（ledger_<日付>.csv＝人が作る台帳は、この道具は読みも書きもしない）。
注意: 0 時をまたいで録った本は、H3-VR が次の日の番号（YYMMDD）を付け、動画も次の日の撮影になるので、この日の一覧に入らない
     （記録シートのセッションはこの日のままなので「記録シートにあるが録音が無い」に出る。次の日の日付でも回すか、手で足す）。

使い方:
  python scripts/field_ledger_draft.py --raw raw --date 20261029 [--backup raw/sheet/20261029/helmet_note_backup_20261029.json] [--out-dir <試しの置き場>] [--force]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import statistics
import struct
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
JST = dt.timezone(dt.timedelta(hours=9))
LEDGER_COLS = ["録音", "開始", "秒", "動画", "時刻が重なる動画", "役割"]
WAV_COLS = ["file", "bytes", "mtime", "bext_date", "bext_time", "fmt", "ch", "sr", "bits", "blockalign", "data_bytes", "sec", "rec_settings", "peak_dbfs"]
# 2026-10-25（前提の敵対的レビュー 実録 重大 3・中 4）: 録音機の設定（bext の coding history）と、割れ・リミッターの張り付きを夜に確かめる
EXPECTED_SETTINGS = {"Rec Mode": "AmbiX", "Mic Position": "Upright", "Correction Filter": "Off"}   # 10/15・10/22 の 191 本と同じ
CLIP_LEVEL, CLIP_MIN = 0.999, 4                       # フルスケール近くの標本がこれだけあれば割れ
CLIP_GAP_S, CLIP_LONG_S = 0.02, 0.05                  # 20 ms より近い割れは 1 つにまとめ、50 ms 以上続く割れだけを疑いに出す（拍手・指パッチンの一瞬の割れは切り出しの外なのでまとめて 1 行・10/22 で多くの本にあった）
LIMIT_LEVEL, LIMIT_BAND, LIMIT_MIN = 10 ** (-2 / 20), 0.01, 1000   # H3-VR のリミッターの初期値 −2 dBFS（約 0.794）に山が張り付けばリミッターの疑い。
# ↑ 疑いの目安だけ（Astra P25-10: ゆがんでいない −2 dBFS の正弦波でも n_lim は 41,600 になる・個数は時間・標本化周波数・チャンネル数で変わる）。
# 1000 個: 続く大きな音（クラクション 0.5 s なら 4 ch × 96 kHz で数千〜数万の山）を潰したときの量。10/22 は拍手などの一瞬の山が 0.8 付近に 51〜109 個の本があった（リミッターは OFF＝67 本がフルスケールまで割れている）
MOV_COLS = ["file", "bytes", "mtime", "mvhd_utc", "sec"]
SLIP_TOL_S = 5.0          # 時計のずれを差し引いたあと、押した時刻と録り始めがこれ以上違えば番号ずれの疑い
CLOCK_NEAR_S = 2.0        # 時計のずれの「集まる所」の幅（±）
CLOCK_SUPPORT_MIN = 3     # 時計のずれを支える本がこれより少なければ疑いに出す
CLOCK_BIG_S = 10.0        # |時計のずれ| がこれを超えたら疑いに出す（10/15・10/22 の H3-VR は数秒）


def _mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")


def wav_info(path: Path) -> dict:
    """bext の録り始めと data の長さ（見出しだけ読む）。RF64（4 GB 超）は ds64 の長さを使う（_rf64 = True）。
    読めなければ中身の欄を空にして _problem に理由を入れる（一覧から黙って落とさない・軽 6）。"""
    row = {"file": path.name, "bytes": os.path.getsize(path), "mtime": _mtime(path), **{k: "" for k in WAV_COLS[3:]},
           "_start": None, "_problem": "", "_rf64": False, "_path": path, "_settings": {}, "_hist": False}
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return {**row, "_problem": "WAV の見出しが無い（RIFF／RF64 の WAVE でない）"}
            rf64 = head[:4] == b"RF64"
            fmt = None; bd = bt = ""; ds64_data = None; settings = {}; hist = ""
            while True:
                ck = f.read(8)
                if len(ck) < 8:
                    return {**row, "_problem": "data の塊が見つからない（途中で切れた録音かもしれない）"}
                cid, sz = ck[:4].decode("latin1"), struct.unpack("<I", ck[4:])[0]
                if cid == "ds64":
                    b = f.read(sz); f.seek(sz & 1, 1)
                    if len(b) >= 16:
                        ds64_data = struct.unpack("<Q", b[8:16])[0]
                elif cid == "fmt ":
                    b = f.read(sz); f.seek(sz & 1, 1)
                    if len(b) < 16:
                        return {**row, "_problem": "fmt の塊が短い"}
                    fmt = struct.unpack("<HHIIHH", b[:16])
                elif cid == "bext":
                    b = f.read(sz); bd = b[320:330].decode("latin1"); bt = b[330:338].decode("latin1"); f.seek(sz & 1, 1)
                    hist = b[602:].split(b"\x00")[0].decode("latin1", "replace") if len(b) > 602 else ""   # coding history（H3-VR は「Rec Mode=AmbiX; Mic Position=Upright; …」）
                    for k in EXPECTED_SETTINGS:
                        m_ = re.search(re.escape(k) + r"=([^;\r\n]+)", hist)
                        if m_:
                            settings[k] = m_.group(1).strip()
                elif cid == "data":
                    if fmt is None or fmt[2] == 0 or fmt[4] == 0:
                        return {**row, "_problem": "fmt が無いか壊れている（data より前に fmt が要る）"}
                    if rf64 and sz == 0xFFFFFFFF:
                        if ds64_data is None:
                            return {**row, "_problem": "RF64 なのに ds64 の長さが無い"}
                        sz = ds64_data
                    start = None
                    try:
                        start = dt.datetime.strptime(bd + bt, "%Y-%m-%d%H:%M:%S").replace(tzinfo=JST)
                    except ValueError:
                        pass
                    return {**row, "bext_date": bd, "bext_time": bt, "fmt": fmt[0], "ch": fmt[1], "sr": fmt[2], "bits": fmt[5],
                            "blockalign": fmt[4], "data_bytes": sz, "sec": round(sz / fmt[4] / fmt[2], 2), "_start": start, "_rf64": rf64,
                            "rec_settings": "; ".join(f"{k}={v}" for k, v in settings.items()), "_settings": settings, "_hist": bool(hist.strip())}
                else:
                    f.seek(sz + (sz & 1), 1)
    except (OSError, struct.error) as e:
        return {**row, "_problem": f"読めない（{type(e).__name__}）"}


def wav_peaks(path: Path) -> dict:
    """音の山（全部のチャンネルの |x| の最大）・フルスケール近くの標本の数とその時刻のまとまり（runs = [(始め s, 終わり s)]・
    CLIP_GAP_S より近い割れは 1 つにまとめる）・−2 dBFS 付近の標本の数（割れとリミッターの確かめ・中身は聞かない）。"""
    import numpy as np
    import soundfile as sf
    mx, n_clip, n_lim, runs, pos = 0.0, 0, 0, [], 0
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate; gap = max(1, int(CLIP_GAP_S * sr))
        for blk in f.blocks(blocksize=96000, dtype="float64", always_2d=True):
            a = np.abs(blk)
            if a.size:
                mx = max(mx, float(a.max()))
                n_clip += int((a >= CLIP_LEVEL).sum())
                n_lim += int((np.abs(a - LIMIT_LEVEL) <= LIMIT_BAND).sum())
                for i in (np.nonzero((a >= CLIP_LEVEL).any(axis=1))[0] + pos):
                    if runs and i - runs[-1][1] <= gap:
                        runs[-1][1] = int(i)
                    else:
                        runs.append([int(i), int(i)])
            pos += len(blk)
    return dict(max=mx, n_clip=n_clip, n_lim=n_lim, runs=[(s / sr, (e + 1) / sr) for s, e in runs])


def mov_info(path: Path) -> dict:
    """MOV の moov/mvhd の撮り始め（日本時間）と長さ（見出しだけ読む）。
    読めなければ時刻の欄を空にして _problem に理由を入れる（一覧から黙って落とさない・軽 6）。"""
    size = os.path.getsize(path)
    row = {"file": path.name, "bytes": size, "mtime": _mtime(path), "mvhd_utc": "", "sec": "", "_start": None, "_problem": ""}
    try:
        with open(path, "rb") as f:
            p = 0
            while p + 8 <= size:
                f.seek(p); h = f.read(8)
                if len(h) < 8:
                    break
                sz, typ = struct.unpack(">I4s", h); hdr = 8
                if sz == 1:
                    sz = struct.unpack(">Q", f.read(8))[0]; hdr = 16
                elif sz == 0:
                    sz = size - p
                if sz < hdr:
                    return {**row, "_problem": "箱（atom）の大きさが壊れている"}
                if typ == b"moov":
                    q = p + hdr
                    while q + 8 <= p + sz:
                        f.seek(q); s2, t2 = struct.unpack(">I4s", f.read(8))
                        if s2 < 8:
                            return {**row, "_problem": "moov の中の箱の大きさが壊れている"}
                        if t2 == b"mvhd":
                            b = f.read(32); v = b[0]
                            if v == 0:
                                ct, _mt, ts, du = struct.unpack(">IIII", b[4:20])
                            else:
                                b += f.read(12); ct, _mt = struct.unpack(">QQ", b[4:20]); ts, du = struct.unpack(">IQ", b[20:32])
                            if ts == 0:
                                return {**row, "_problem": "mvhd の timescale が 0"}
                            t = dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=ct)
                            return {**row, "mvhd_utc": t.strftime("%H:%M:%S"), "sec": round(du / ts, 2), "_start": t.astimezone(JST)}
                        q += s2
                p += sz
    except (OSError, struct.error, IndexError, OverflowError) as e:
        return {**row, "_problem": f"読めない（{type(e).__name__}）"}
    return {**row, "_problem": "moov/mvhd が無い（撮り始めが読めない）"}


def roles_from_backup(backup: dict, date: str):
    """{録音: [(セッション, 役割, 中身, 動画, 押した時刻 ms)]}。名前の入りうる欄は読まない／写さない
    （helper などは読まない。使わないの理由は自由記述なので、文は捨てて「理由あり／理由なし」だけ持つ）。"""
    out = {}
    def add(fn, *v):
        if fn:
            out.setdefault(fn, []).append(v)
    for s in (backup.get("state") or {}).get("sessions") or []:
        sid = str(s.get("id") or "")
        if not sid.startswith(date + "_"):
            continue
        suf = sid[len(date) + 1:]
        for t in s.get("takes") or []:
            add(t.get("file"), suf, "本", str(t.get("scriptId") or t.get("script") or ""), t.get("video") or "", t.get("t0"))
        for x in s.get("excluded") or []:
            add(x.get("file"), suf, "使わない", "理由あり" if str(x.get("reason") or "").strip() else "理由なし", x.get("video") or "", x.get("t0"))
        add(s.get("calibFile"), suf, "校正", "", "", None)
        for fn, c in (s.get("checks") or {}).items():
            add(fn, suf, "点検", f"{c.get('h', '')}°・{c.get('mode', '')} 方向" + ("・頭の点検" if fn == s.get("checkFile") else ""), "", None)
    return out


def role_text(entries) -> str:
    kinds = {e[1] for e in entries}
    if len(kinds) > 1 or (kinds == {"本"} and len(entries) > 1):
        return "要確認: 役割が重なる（" + " ／ ".join(f"{e[1]} {e[0]} {e[2]}".strip() for e in entries) + "）"
    e = entries[0]
    if e[1] == "本":
        return f"本 {e[0]} {e[2]}"
    if e[1] == "校正":
        return "校正 " + "・".join(dict.fromkeys(x[0] for x in entries))
    return f"{e[1]} {e[0]}: {e[2]}"


def _to_history(path: Path) -> Path:
    """今ある出力を history/ に写す（同じ秒に 2 回でも前の写しを上書きしない）。"""
    hist = path.parent / "history"; hist.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    dst = hist / f"{path.stem}_before_{stamp}{path.suffix}"
    n = 2
    while dst.exists():
        dst = hist / f"{path.stem}_before_{stamp}_{n}{path.suffix}"; n += 1
    shutil.copy2(path, dst)
    return dst


def guard_outputs(paths, force: bool):
    """書き始める前に出力を全部確かめる（軽 6: 前は wav_・mov_ を書いてから下書きで止まり、check.txt は写さずに上書きしていた）。
    1 つでもあれば --force でない限り全部を挙げて止める。--force なら、あるものを全部 history/ に写す。"""
    have = [p for p in paths if p.exists()]
    if have and not force:
        raise SystemExit("既にある（上書きしない・--force なら history/ に写してから書く）: " + " ".join(str(p) for p in have))
    return [_to_history(p) for p in have]


def write_csv(path: Path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _files(d: Path, stem_glob: str, ext: str) -> list:
    """大文字・小文字の拡張子の両方を、同じファイルを 2 回数えずに（Windows は名前の大小を区別しない）。"""
    seen = {}
    for p in list(d.glob(stem_glob + ext.upper())) + list(d.glob(stem_glob + ext.lower())):
        seen.setdefault(p.name.lower(), p)
    return [seen[k] for k in sorted(seen)]


def draft(raw: Path, date: str, backup_path: Path | None, force: bool = False, out_dir: Path | None = None, peaks: bool = True) -> dict:
    yymmdd = date[2:]
    led = out_dir or (raw / "ledger")
    out_wav, out_mov = led / f"wav_{date}.csv", led / f"mov_{date}.csv"
    out_draft, out_chk = led / f"ledger_{date}_draft.csv", led / f"ledger_{date}_check.txt"
    outs = [out_wav, out_mov, out_draft, out_chk]
    if not force:
        guard_outputs(outs, False)                       # 何か読む前・書く前に止める
    wavs = [wav_info(p) for p in _files(raw, f"{yymmdd}_*", ".WAV")]
    day = dt.datetime.strptime(date, "%Y%m%d").date()
    movs_all = [mov_info(p) for p in _files(raw / "video", "*", ".MOV")]
    movs = [m for m in movs_all if m["_start"] and m["_start"].date() == day]
    movs_bad = [m for m in movs_all if m["_start"] is None]
    mov_by_name = {m["file"].lower(): m for m in movs_all}
    bk = {}
    if backup_path and backup_path.is_file():
        with open(backup_path, encoding="utf-8") as fh:
            bk = json.load(fh)
    roles = roles_from_backup(bk, date)
    flags = []
    # H3-VR の時計のずれ = 本の（録り始め − 押した時刻）が一番多く集まる所（±2 s 以内の数が一番多い値・同じ数なら小さいずれ）の中央値。
    # 番号ずれの本の差は大きく外れるので、ただの中央値より崩れにくい。ただし本が 1〜2 本の日は、ずれがその本の差そのものになるので
    # （その本の番号ずれを吸い込む）、支えた本の数と |ずれ| の大きさを疑いに出す（軽 6）
    by_file = {w["file"]: w for w in wavs}
    diffs = [(by_file[fn]["_start"] - dt.datetime.fromtimestamp(e[4] / 1000, JST)).total_seconds()
             for fn, es in roles.items() for e in es if e[1] == "本" and e[4] and fn in by_file and by_file[fn]["_start"]]
    support = []
    if diffs:
        center = max(diffs, key=lambda d: (sum(abs(x - d) <= CLOCK_NEAR_S for x in diffs), -abs(d)))
        support = [x for x in diffs if abs(x - center) <= CLOCK_NEAR_S]
        clock = statistics.median(support)
    else:
        clock = 0.0
    if not diffs:
        flags.append("時計のずれを推定できない: 押した時刻と録り始めの両方がある本が無い → 0 s として扱った（番号ずれと動画の照合は時計のずれ 0 の前提）")
    elif len(support) < CLOCK_SUPPORT_MIN:
        flags.append(f"時計のずれの支えが少ない: {clock:+.1f} s は押した時刻のある本 {len(diffs)} 本のうち {len(support)} 本だけで決めた"
                     f"（{CLOCK_SUPPORT_MIN} 本未満）→ その本の番号ずれをずれとして吸い込んでいるかもしれない・押した時刻と録り始めの差を目で確かめる")
    if abs(clock) > CLOCK_BIG_S:
        flags.append(f"時計のずれが大きい: {clock:+.1f} s（{CLOCK_BIG_S:g} s を超える）→ H3-VR の時計か、番号ずれを吸い込んでいないかを確かめる")
    for fn, es in roles.items():
        for e in es:
            if e[1] != "本" or not e[4]:
                continue
            press = dt.datetime.fromtimestamp(e[4] / 1000, JST) + dt.timedelta(seconds=clock)
            near = min((w for w in wavs if w["_start"]), key=lambda w: abs((w["_start"] - press).total_seconds()), default=None)
            if near is None:
                continue
            gap = abs((near["_start"] - press).total_seconds())
            if near["file"] != fn and gap <= SLIP_TOL_S:
                flags.append(f"番号ずれの疑い: {e[0]} {e[2]} は記録シートでは {fn}・押した時刻（時計のずれ {clock:+.1f} s を差し引き）では {near['file']}（差 {gap:.1f} s）")
            elif fn in by_file and by_file[fn]["_start"] and abs((by_file[fn]["_start"] - press).total_seconds()) > SLIP_TOL_S:
                flags.append(f"時刻が合わない: {e[0]} {e[2]} の {fn} の録り始めと押した時刻が {abs((by_file[fn]['_start'] - press).total_seconds()):.1f} s 違う")
    rows, short_clip = [], []
    for w in wavs:
        es = roles.get(w["file"])
        role = role_text(es) if es else "要確認: 記録シートに無い"
        vid = next((e[3] for e in (es or []) if e[3]), "")
        a0 = w["_start"] + dt.timedelta(seconds=-clock) if w["_start"] else None          # 録音の時間を記録シート（iPad）の時計にそろえる
        over = []
        if a0 is not None:
            a1 = a0 + dt.timedelta(seconds=w["sec"])
            over = [m["file"] for m in movs if m["_start"] < a1 and m["_start"] + dt.timedelta(seconds=m["sec"]) > a0]
        if w["_problem"]:
            flags.append(f"読めない録音: {w['file']}（{w['_problem']}）→ 長さと録り始めは手で確かめる・役割 = {role}")
        elif w["_start"] is None:
            flags.append(f"録音の時刻が読めない: {w['file']}（bext の録り始めが無い）→ 番号ずれと動画の重なりを確かめられない・役割 = {role}")
        if w["_rf64"]:
            flags.append(f"RF64（4 GB 超）の録音: {w['file']}（長さ {w['sec']} s は ds64 から読んだ）")
        if w["_hist"]:                                                               # coding history がある本だけ（ほかの録音機・試験の WAV は見ない）
            bad = [f"{k}={w['_settings'].get(k, '（無い）')}" for k, v in EXPECTED_SETTINGS.items() if w["_settings"].get(k) != v]
            if bad:
                flags.append(f"録音機の設定が前の日と違う: {w['file']}（{'・'.join(bad)}・10/15・10/22 は AmbiX・Upright・Correction Filter Off）・役割 = {role}")
        if peaks and not w["_problem"]:
            try:
                pk = wav_peaks(w["_path"])
                w["peak_dbfs"] = round(20 * math.log10(pk["max"]), 1) if pk["max"] > 0 else ""
                long_runs = [r for r in pk["runs"] if r[1] - r[0] >= CLIP_LONG_S]
                if pk["n_clip"] >= CLIP_MIN and long_runs:
                    flags.append(f"割れの疑い（続く割れ）: {w['file']} の " + "・".join(f"{a:.2f}〜{b:.2f} s" for a, b in long_runs[:6])
                                 + (" …" if len(long_runs) > 6 else "") + f"（フルスケール近くの標本 {pk['n_clip']} 個）→ 本番の音（クラクションなど）の時刻なら録り直しを考える・役割 = {role}")
                elif pk["n_clip"] >= CLIP_MIN:
                    short_clip.append(f"{w['file']}（{pk['runs'][0][0]:.2f} s" + (f" ほか {len(pk['runs']) - 1} か所" if len(pk["runs"]) > 1 else "") + "）")
                elif pk["max"] <= LIMIT_LEVEL + LIMIT_BAND and pk["n_lim"] >= LIMIT_MIN:
                    flags.append(f"リミッターの疑い: {w['file']} は山が {pk['max']:.3f}（−2 dBFS 付近に {pk['n_lim']} 個が張り付き）→ リミッターが ON だったかもしれない（疑いだけ・ゆがんでいない大きな音でも出る＝本体の設定の記録で確かめる）・役割 = {role}")
            except Exception as e:                                                   # 読めなくても台帳の下書きは止めない
                flags.append(f"山を読めない: {w['file']}（{type(e).__name__}）")
        if role.startswith("本") and vid and a0 is not None:                         # 録音の時刻が読めないときは上の行に出した
            m = mov_by_name.get(vid.lower())                                          # 動画の名前の大文字・小文字は区別しない
            if m is None:
                flags.append(f"動画が見つからない: {w['file']}（{role}）の動画 {vid} が raw/video に無い")
            elif m["_start"] is None:
                flags.append(f"動画の時刻が読めない: {w['file']}（{role}）の動画 {m['file']}（{m['_problem']}）→ 録音と重なるか確かめられない")
            elif vid.lower() not in {o.lower() for o in over}:
                flags.append(f"動画が重ならない: {w['file']}（{role}）の動画 {vid} の撮影の時間が録音と重ならない（重なるのは {' '.join(over) or 'なし'}）")
        if w["sec"] != "" and w["sec"] < 5:
            flags.append(f"短い録音: {w['file']} は {w['sec']} s（押し間違いの可能性）・役割 = {role}")
        if role.startswith("校正") and w["sec"] != "" and w["sec"] < 60:
            flags.append(f"短い校正: {w['file']} は {w['sec']} s（60 s 未満）")
        if role.startswith("要確認"):
            flags.append(f"{role}: {w['file']}（{w['bext_time'] or '時刻なし'}・{w['sec'] if w['sec'] != '' else '?'} s）")
        rows.append({"録音": w["file"], "開始": w["bext_time"], "秒": w["sec"], "動画": vid, "時刻が重なる動画": " ".join(over), "役割": role})
    if short_clip:
        flags.append(f"一瞬の割れだけの録音 {len(short_clip)} 本（50 ms 未満・拍手や指パッチンの見込み＝切り出しの 10 秒の外なら問題ない）: "
                     + " ".join(short_clip[:12]) + (" …" if len(short_clip) > 12 else ""))
    for m in movs_bad:
        flags.append(f"動画の時刻が読めない: {m['file']}（{m['_problem']}）→ どの日の撮影か分からないので mov_{date}.csv に時刻なしで入れた")
    for fn in roles:
        if fn not in by_file:
            flags.append(f"記録シートにあるが録音が無い: {fn}（" + " ／ ".join(f"{e[1]} {e[0]} {e[2]}".strip() for e in roles[fn]) + "）")
    led.mkdir(parents=True, exist_ok=True)
    hist = guard_outputs(outs, force)                    # --force: 書く前に、あるものを全部 history/ へ（無ければ何もしない）
    write_csv(out_wav, WAV_COLS, wavs)
    write_csv(out_mov, MOV_COLS, movs + movs_bad)
    write_csv(out_draft, LEDGER_COLS, rows)
    out_chk.write_text(f"# {date} の台帳の下書きの疑い（{len(flags)} 件）・H3-VR の時計のずれ {clock:+.1f} s"
                       f"（録り始め − 押した時刻・押した時刻のある本 {len(diffs)} 本のうち、集まった所の ±{CLOCK_NEAR_S:g} s に入った {len(support)} 本の中央値）\n"
                       + "\n".join(flags) + "\n", encoding="utf-8")
    return dict(n_wav=len(wavs), n_mov=len(movs), n_mov_unreadable=len(movs_bad), n_take=sum(r["役割"].startswith("本") for r in rows),
                clock_s=round(clock, 1), clock_n=len(diffs), clock_support=len(support), flags=flags, history=[str(h) for h in hist],
                files=[str(out_draft), str(out_chk)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実録の台帳の下書き（録った日の夜）")
    ap.add_argument("--raw", default="raw"); ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--backup", default=None, help="記録シートのバックアップ JSON（既定 raw/sheet/<日付>/helmet_note_backup_<日付>.json）")
    ap.add_argument("--force", action="store_true", help="今ある出力を history/ に写してから書き直す")
    ap.add_argument("--out-dir", default=None, help="書き出し先（既定 raw/ledger・試しはスクラッチパッドへ）")
    ap.add_argument("--no-peaks", action="store_true", help="音の山（割れ・リミッターの張り付き）を読まない（見出しだけで速く回すとき）")
    a = ap.parse_args(argv)
    raw = Path(a.raw)
    bp = Path(a.backup) if a.backup else raw / "sheet" / a.date / f"helmet_note_backup_{a.date}.json"
    if not bp.is_file():
        print(f"注意: 記録シートのバックアップが無い（{bp}）→ 役割は全部「要確認」になる")
    res = draft(raw, a.date, bp, a.force, Path(a.out_dir) if a.out_dir else None, peaks=not a.no_peaks)
    print(f"録音 {res['n_wav']}・動画 {res['n_mov']}（時刻の読めない動画 {res['n_mov_unreadable']}）・本 {res['n_take']}・"
          f"H3-VR の時計のずれ {res['clock_s']:+.1f} s（支えた本 {res['clock_support']}/{res['clock_n']}）・疑い {len(res['flags'])} 件")
    for fl in res["flags"]:
        print(" -", fl)
    if res["history"]:
        print("前のものを写した:", " ".join(res["history"]))
    print("書いた:", " ".join(res["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
