# -*- coding: utf-8 -*-
"""本番の描き直し（dataset_outdoor_siren_v2final_v30）の val と、背景の形の比べの val（_v30bg_val）の中身の一致の検査（Fable 10/27 07:3x の 3・2026-10-27 Opus）。

同じ plan（同じ seed）・同じ描画の設定（背景の形 (ii)・帯域ゲート 0 dB）で描くので、val の本の
  foa/<本>.flac・metadata/<本>.csv・metadata_dist/<本>.csv・masks/<本>.csv・work/<本>/scene.json
はバイト単位で同じはず。1 本でも違う・欠けていれば止める。ただし scene.json の stats.gen_seconds（描くのにかかった秒数・壁時計）は描くたびに変わるので外して比べる
（2026-10-27: job 9388 で val の 12,266 本の scene.json がこの 1 か所だけ違って止めた・foa・metadata・metadata_dist・masks は全部同じだった）。
版の記録（GEN_VERSIONS.txt）は、決めた 3 本の sha の行の違いだけ許す:
  scripts/_run_v2final_gen.py（params の写しの書き方・19384d680）・scripts/_atomic_write.py（新しく読み込まれる部品）・
  scripts/step11_v30_render.py（既定を (ii) に固定・53e12c828・描き方の値は比べの val と同じ）。
それ以外の行（描画の連なり・render: v30 FEATURES… の行・plan rows expected・ほかのモジュールの sha）が違えば止める。
使い方: python server_sde/_v30_val_match_check.py --a <本番の置き場> --b <比べの val の置き場> --manifest <val の名簿> --out <結果の json>
終了コード: 0 = 一致・8 = 違いあり（結果の json に一覧）・2 = 入力が無い
試験: scripts/test_v30_val_match_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ALLOWED = ("scripts/_run_v2final_gen.py", "scripts/_atomic_write.py", "scripts/step11_v30_render.py")
FILES = (("foa", ".flac"), ("metadata", ".csv"), ("metadata_dist", ".csv"), ("masks", ".csv"))


def parse_gv(text: str) -> tuple[dict, set]:
    """GEN_VERSIONS → ({ファイル: sha}, それ以外の行の集合)。"""
    shas, other = {}, set()
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            shas[parts[1].strip()] = parts[0]
        elif line.strip():
            other.add(line.strip())
    return shas, other


def gen_diff(a_text: str, b_text: str) -> dict:
    sa, oa = parse_gv(a_text)
    sb, ob = parse_gv(b_text)
    changed = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    return {"changed_files": changed, "not_allowed": [k for k in changed if k not in ALLOWED],
            "other_lines_only_a": sorted(oa - ob), "other_lines_only_b": sorted(ob - oa)}


SCENE_IGNORE = (("stats", "gen_seconds"),)   # 描くのにかかった秒数（壁時計・描くたびに変わる・中身ではない）。2026-10-27: job 9388 で 12,266 本の scene.json がこれだけ違って止めた


def _scene_wo_timing(raw: bytes):
    """scene.json から SCENE_IGNORE の鍵を外した中身（読めなければ None＝バイトで比べる）。"""
    try:
        o = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    for path in SCENE_IGNORE:
        d = o
        for k in path[:-1]:
            d = d.get(k) if isinstance(d, dict) else None
        if isinstance(d, dict):
            d.pop(path[-1], None)
    return o


def compare(a: Path, b: Path, clips: list) -> dict:
    """foa・metadata・metadata_dist・masks はバイトで比べる。scene.json は描くのにかかった秒数（stats.gen_seconds）だけ外して中身で比べる。"""
    mism, miss, timing_only = [], [], 0
    for c in clips:
        for d, ext in FILES + (("work", "/scene.json"),):
            rel = f"{d}/{c}{ext}"
            pa, pb = a / rel, b / rel
            if not pa.is_file() or not pb.is_file():
                miss.append(rel)
                continue
            ra, rb = pa.read_bytes(), pb.read_bytes()
            if ra == rb:
                continue
            if d == "work":
                oa, ob = _scene_wo_timing(ra), _scene_wo_timing(rb)
                if oa is not None and oa == ob:
                    timing_only += 1
                    continue
            mism.append(rel)
    return {"n_clips": len(clips), "n_files": len(clips) * (len(FILES) + 1), "mismatch": mism, "missing": miss,
            "scene_timing_only": timing_only, "scene_ignored_keys": [".".join(p) for p in SCENE_IGNORE]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    x = ap.parse_args(argv)
    for p in (x.a / "GEN_VERSIONS.txt", x.b / "GEN_VERSIONS.txt", x.manifest):
        if not p.is_file():
            print(f"無い: {p}")
            return 2
    with open(x.manifest, encoding="utf-8-sig", newline="") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f)]
    res = compare(x.a, x.b, clips)
    res["gen_versions"] = gen_diff((x.a / "GEN_VERSIONS.txt").read_text(encoding="utf-8"), (x.b / "GEN_VERSIONS.txt").read_text(encoding="utf-8"))
    res["allowed"] = list(ALLOWED)
    g = res["gen_versions"]
    ok = not res["mismatch"] and not res["missing"] and not g["not_allowed"] and not g["other_lines_only_a"] and not g["other_lines_only_b"]
    res["ok"] = ok
    x.out.parent.mkdir(parents=True, exist_ok=True)
    x.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"val の中身の一致: {res['n_clips']} 本・{res['n_files']} ファイル・違い {len(res['mismatch'])}・欠け {len(res['missing'])}・"
          f"scene.json の違いが描いた秒数（stats.gen_seconds）だけ {res['scene_timing_only']} 本（中身は同じ）・"
          f"版の記録の違い {g['changed_files']}（許さない {g['not_allowed']}・ほかの行 {len(g['other_lines_only_a']) + len(g['other_lines_only_b'])}）→ {'一致' if ok else '止める'}")
    return 0 if ok else 8


if __name__ == "__main__":
    sys.exit(main())
