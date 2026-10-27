# -*- coding: utf-8 -*-
"""Step 19a: H3-VR が 2 GB ごとに自動で分けたファイルを 1 本につなぐ（2026-09-26・Fable）。

背景: H3-VR は録音中にファイルが 2 GB を超えると自動で次のファイルを作って録音を続ける（取扱説明書 p.51）。
      96 kHz/24 bit/4 ch は 1.15 MB/s なので約 29 分ごと。負例露出 100 分の連続録音は 4 本に分かれる。
      step19b（負例の分割）は 1 本の原録音を前提にしているので、先にこれでつなぐ。
出力: FLAC（PCM_24・元の fs・4 ch。WAV は 4 GB を超えられないため）。ブロックごとに読み書きするのでメモリに載せない。
      隣に <out>.parts.json（部品の名前・フレーム数・つなぎ目の秒・sha256）を書く。

前提（本人が現物で確かめるまでは仮定）: 分割は隙間なし（サンプルが連続）。つなぎ目で音が途切れていたら、
      つなぎ目の秒（parts.json の boundaries_s）をイベントの時刻と照らして扱いを決める。

使い方:
  python scripts/step19a_concat_split.py --out raw/neg_20261005.flac raw/251005_003.WAV raw/251005_004.WAV raw/251005_005.WAV
  python scripts/step19a_concat_split.py --out raw/neg_20261005.flac --glob "raw/251005_00[3-6].WAV"   # 名前順
  python scripts/step19a_concat_split.py --selftest out/_test_step19a
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOCK = 1 << 20          # 1M フレームずつ（96 kHz・4 ch・float64 で 32 MB）
SPLIT_BYTES = 2 * 1024 ** 3


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def concat(parts: list[Path], out: Path, want_sha: bool = True) -> dict:
    infos = []
    for p in parts:
        with sf.SoundFile(str(p)) as f:
            infos.append({"file": p.name, "frames": len(f), "samplerate": f.samplerate, "channels": f.channels,
                          "subtype": f.subtype, "bytes": p.stat().st_size})
    fs, ch, st = infos[0]["samplerate"], infos[0]["channels"], infos[0]["subtype"]
    for i in infos[1:]:
        if (i["samplerate"], i["channels"], i["subtype"]) != (fs, ch, st):
            raise ValueError(f"形式が違う: {infos[0]['file']}={fs}/{ch}/{st} vs {i['file']}={i['samplerate']}/{i['channels']}/{i['subtype']}")
    warn = []
    for i in infos[:-1]:
        if i["bytes"] < SPLIT_BYTES * 0.98:
            warn.append(f"{i['file']} は {i['bytes'] / 1024 ** 3:.2f} GB で 2 GB に達していない（本当に分割の続きか確かめる）")
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    bounds = []
    with sf.SoundFile(str(out), "w", samplerate=fs, channels=ch, subtype="PCM_24", format="FLAC") as w:
        for p in parts:
            with sf.SoundFile(str(p)) as f:
                while True:
                    blk = f.read(BLOCK, dtype="float64", always_2d=True)
                    if not len(blk):
                        break
                    w.write(blk)
                    total += len(blk)
            bounds.append(total)
    rec = {"out": out.name, "samplerate": fs, "channels": ch, "subtype_in": st, "frames": total, "duration_s": total / fs,
           "parts": infos, "boundaries_s": [b / fs for b in bounds[:-1]], "warnings": warn,
           "assumption": "分割は隙間なし（サンプル連続）と仮定。現物で未確認"}
    if want_sha:
        for i, p in zip(infos, parts):
            i["sha256"] = sha256_of(p)
    (out.with_suffix(out.suffix + ".parts.json")).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def selftest(tmp: Path) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    fs, ch = 96000, 4
    rng = np.random.default_rng(0)
    full = (rng.standard_normal((fs * 3, ch)) * 0.1).astype("float64")
    cuts = [0, fs + 123, 2 * fs + 45, len(full)]
    parts = []
    for k in range(3):
        p = tmp / f"251005_00{k + 3}.WAV"
        sf.write(p, full[cuts[k]:cuts[k + 1]], fs, subtype="PCM_24")
        parts.append(p)
    out = tmp / "neg_test.flac"
    rec = concat(parts, out, want_sha=True)
    y, fs2 = sf.read(out, dtype="float64", always_2d=True)
    q = np.round(full * (2 ** 23)) / (2 ** 23)          # 24 bit に丸めた元
    ok = [("fs とチャンネルが同じ", fs2 == fs and y.shape[1] == ch),
          ("フレーム数が合計と一致", len(y) == len(full) and rec["frames"] == len(full)),
          ("サンプルが一致（24 bit の丸め以内）", np.max(np.abs(y - q)) < 1.5 / (2 ** 23)),
          ("つなぎ目の秒が正しい", [round(b, 6) for b in rec["boundaries_s"]] == [round(cuts[1] / fs, 6), round(cuts[2] / fs, 6)]),
          ("parts.json がある", out.with_suffix(".flac.parts.json").exists() and len(rec["parts"]) == 3 and all("sha256" in i for i in rec["parts"])),
          ("2 GB 未満の部品に注意が出る", len(rec["warnings"]) == 2)]
    bad = tmp / "bad.WAV"
    sf.write(bad, full[:fs], 48000, subtype="PCM_24")
    try:
        concat([parts[0], bad], tmp / "x.flac", want_sha=False); ok.append(("形式違いを拒否", False))
    except ValueError:
        ok.append(("形式違いを拒否", True))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    st = opt("--selftest")
    if st:
        return selftest(Path(st))
    out = opt("--out"); pat = opt("--glob")
    no_sha = "--no-sha" in argv
    if no_sha:
        argv.remove("--no-sha")
    files = [Path(a) for a in argv] if not pat else [Path(p) for p in sorted(glob.glob(pat))]
    if not out or len(files) < 2:
        print(__doc__); return 2
    rec = concat(files, Path(out), want_sha=not no_sha)
    print(f"{Path(out).name}: {rec['duration_s'] / 60:.1f} 分・{rec['frames']} フレーム・つなぎ目 {[round(b, 3) for b in rec['boundaries_s']]} s")
    for w in rec["warnings"]:
        print("注意:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
