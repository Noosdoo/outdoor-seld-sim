# -*- coding: utf-8 -*-
"""step19n_horn_clap_check.py — クラクション（E2）の本で、拍手の大きな音が解析の切り出し（鳴り始めの 5 秒前〜5 秒後）の中に無いかを検査する
（変更票 #69 r16／r17・Astra R105-02・R108-08・本人 10/19「満たさない本は欠測にする・結果は見ない」・実録の採点の準備 B7・2026-10-20・Fable）。

決まり（採点の前に固定）:
  - 切り出し = [鳴り始め − 5 s, 鳴り始め + 5 s]（step19b は最接近を 8 秒目に置き、警告音は鳴り始め + 3 s を最接近にするので同じ区間）。
  - 合格 = 「拍手の終わり < 鳴り始め − 5 s」かつ「録音の終わり ≥ 鳴り始め + 5 s」。
  - 拍手の終わり = events.csv の 拍手_終わり秒（本人が波形で読んだ値）。無ければ 拍手_録音側秒 + CLAP_TAIL_S（3.0 s・r17 の「基本 3 秒で消える」）を見込みとして使い、「見込み」の印を付ける。
  - 録音の終わり = --wav-csv（raw/ledger/wav_YYYYMMDD.csv の file と sec・録音のヘッダから出した長さ。中身は開かない）。
  - 満たさない本 = 欠測（この検査の出力を step20 の前に見て、注釈から外す）。結果（推論・採点）は見ない。
入力:
  --ann   step19d の注釈 CSV（class・t_start〔録音の秒・鳴り始め〕・orig_file・take_id）
  --events 記録シートの events.csv（原本・拍手_録音側秒・拍手_終わり秒〔任意〕）
  --wav-csv 録音の長さの一覧（file, sec）
  --out   結果 CSV（take_id, orig_file, t_on_s, clap_s, clap_end_s, clap_end_source, rec_len_s, verdict, reason）
使い方: python scripts/step19n_horn_clap_check.py --ann out/realsmoke/ann_orig.csv --events raw/ledger/events_fixed_YYYYMMDD.csv --wav-csv raw/ledger/wav_YYYYMMDD.csv --out out/realsmoke/horn_clap_check.csv
       python scripts/step19n_horn_clap_check.py --selftest
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRE_S, POST_S = 5.0, 5.0          # 切り出し = 鳴り始めの 5 秒前〜5 秒後（step19b）
CLAP_TAIL_S = 3.0                 # 拍手の終わりが未読のときの見込み（#69 r17「基本 3 秒で消える」）
OUT_COLS = ["take_id", "orig_file", "t_on_s", "clap_s", "clap_end_s", "clap_end_source", "rec_len_s", "verdict", "reason"]


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def _f(v: str):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def check_rows(ann: list[dict], events: list[dict], wav: list[dict], cls: str = "horn") -> list[dict]:
    """注釈の horn の行ごとに判定を返す（読むだけ）。"""
    claps = {}
    for r in events:
        o = Path(r.get("原本", "")).name.lower()
        if o:
            claps[o] = (_f(r.get("拍手_録音側秒", "")), _f(r.get("拍手_終わり秒", "")))
    lens = {Path(r.get("file", "")).name.lower(): _f(r.get("sec", "")) for r in wav if r.get("file")}
    out = []
    for r in ann:
        if (r.get("class") or "").strip() != cls:
            continue
        orig = Path(r.get("orig_file", "")).name
        t_on = _f(r.get("t_start", ""))
        clap_s, clap_end = claps.get(orig.lower(), (None, None))
        rec_len = lens.get(orig.lower())
        row = dict(take_id=r.get("take_id", ""), orig_file=orig, t_on_s="" if t_on is None else f"{t_on:.2f}",
                   clap_s="" if clap_s is None else f"{clap_s:.2f}", clap_end_s="", clap_end_source="", rec_len_s="" if rec_len is None else f"{rec_len:.2f}")
        reasons = []
        if t_on is None:
            reasons.append("鳴り始め（t_start）が未確定")
        if clap_s is None and clap_end is None:
            reasons.append("拍手_録音側秒 が無い（拍手を波形で読んでいない）")
        if rec_len is None:
            reasons.append("録音の長さが --wav-csv に無い")
        if reasons:
            out.append({**row, "verdict": "要確認", "reason": "・".join(reasons)}); continue
        if clap_end is not None:
            end, src = clap_end, "拍手_終わり秒（波形）"
        else:
            end, src = clap_s + CLAP_TAIL_S, f"見込み（拍手_録音側秒 + {CLAP_TAIL_S:g} s）"
        row["clap_end_s"] = f"{end:.2f}"; row["clap_end_source"] = src
        bad = []
        if not (end < t_on - PRE_S):
            bad.append(f"拍手の終わり {end:.2f} が切り出しの始まり {t_on - PRE_S:.2f} より前でない")
        if not (rec_len >= t_on + POST_S):
            bad.append(f"録音の終わり {rec_len:.2f} が切り出しの終わり {t_on + POST_S:.2f} より前")
        if bad:
            out.append({**row, "verdict": "欠測", "reason": "・".join(bad)})
        else:
            out.append({**row, "verdict": "合格" if clap_end is not None else "合格（拍手の終わりは見込み・波形で読んで確定する）", "reason": ""})
    return out


def selftest() -> int:
    ann = [dict(take_id="e/1", orig_file="261029_010.WAV", **{"class": "horn"}, t_start="20.00"),   # 拍手 10.0・終わり 11.0 → 合格
           dict(take_id="e/2", orig_file="261029_011.WAV", **{"class": "horn"}, t_start="12.00"),   # 拍手 8.0（終わり未読 → 11.0 見込み）→ 11.0 ≥ 7.0 → 欠測
           dict(take_id="e/3", orig_file="261029_012.WAV", **{"class": "horn"}, t_start="20.00"),   # 録音 24.0 s < 25.0 → 欠測
           dict(take_id="e/4", orig_file="261029_013.WAV", **{"class": "horn"}, t_start="20.00"),   # 拍手 無し → 要確認
           dict(take_id="e/5", orig_file="261029_014.WAV", **{"class": "horn"}, t_start="20.00"),   # 拍手 10.0・終わり未読 → 13.0 < 15.0 → 合格（見込み）
           dict(take_id="c/1", orig_file="261029_020.WAV", **{"class": "car_drive"}, t_start="5.00")]
    events = [dict(原本="261029_010.WAV", 拍手_録音側秒="10.0", 拍手_終わり秒="11.0"), dict(原本="261029_011.WAV", 拍手_録音側秒="8.0"),
              dict(原本="261029_012.WAV", 拍手_録音側秒="10.0", 拍手_終わり秒="11.0"), dict(原本="261029_013.WAV"), dict(原本="261029_014.WAV", 拍手_録音側秒="10.0")]
    wav = [dict(file="261029_010.WAV", sec="30.0"), dict(file="261029_011.WAV", sec="30.0"), dict(file="261029_012.WAV", sec="24.0"),
           dict(file="261029_013.WAV", sec="30.0"), dict(file="261029_014.WAV", sec="30.0")]
    out = check_rows(ann, events, wav)
    v = {r["take_id"]: r["verdict"] for r in out}
    ok = [("S1 horn の行だけ 5 本", len(out) == 5 and "c/1" not in v),
          ("S2 合格（拍手の終わりを波形で読んだ）", v["e/1"] == "合格"),
          ("S3 欠測（拍手の見込みの終わりが切り出しの中）", v["e/2"] == "欠測" and "拍手の終わり 11.00" in [r["reason"] for r in out if r["take_id"] == "e/2"][0]),
          ("S4 欠測（録音が鳴り始め + 5 s より短い）", v["e/3"] == "欠測"),
          ("S5 要確認（拍手の秒が無い）", v["e/4"] == "要確認"),
          ("S6 合格（見込み）の印", v["e/5"].startswith("合格（拍手の終わりは見込み"))]
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    ann, events, wav = _read(Path(_arg("--ann"))), _read(Path(_arg("--events"))), _read(Path(_arg("--wav-csv")))
    out = check_rows(ann, events, wav)
    op = Path(_arg("--out", str(ROOT / "out/realsmoke/horn_clap_check.csv")))
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS); w.writeheader(); w.writerows(out)
    from collections import Counter
    c = Counter(r["verdict"].split("（")[0] for r in out)
    print(f"クラクションの拍手の検査: {len(out)} 本 → " + "・".join(f"{k} {n}" for k, n in c.items()) + f" -> {op}")
    for r in out:
        if r["verdict"] != "合格":
            print(f"  {r['take_id']} {r['orig_file']}: {r['verdict']} {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
