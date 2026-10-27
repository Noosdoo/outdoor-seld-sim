# -*- coding: utf-8 -*-
"""v19（新 4 場面）の生成ドライバ（2026-09-08）。描画は v16 と同一（step11_v19_render 経由の generate_clip_v13）。
出力先は out/dataset_outdoor_siren_v19/ だけ（foa / metadata / masks / work）。v16 以前には書かない。
使い方:
  python scripts/_run_v19_gen.py --plan out/dataset_outdoor_siren_v19/plan/assignment_v19_fast.csv --list
  python scripts/_run_v19_gen.py --plan <csv> --rows 0-62      # シャード生成（サーバ）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json  # noqa: E402
import re  # noqa: E402

import soundfile as sf  # noqa: E402

import step11_v19_render as v19r  # noqa: E402

m9 = v19r.m9
N_FRAMES, N_CH = 240000, 4          # 10 s × 24 kHz・FOA 4ch


def clip_complete(clip: str, row: dict | None = None) -> bool:
    """Require the matching scene, decoded FOA and structurally complete labels."""
    if not re.fullmatch(r"fold\d+_room\d+_mix\d+|(train|val|eval|tune)_(syn|field)_clip\d{5}", clip):   # 2026-09-22 改名: 新旧どちらも可
        return False
    ds = m9.DS
    try:
        scene = json.loads((ds / "work" / clip / "scene.json").read_text(encoding="utf-8"))
        saved = scene["row"]
        if saved["clip_id"] != clip or not scene["sources"]:
            return False
        if row is not None and any(str(saved.get(k)) != str(v) for k, v in row.items()):
            return False
        if saved.get("scenario") == "v19turn" and scene["v19"].get("turn_rev") != 2:
            return False
        with sf.SoundFile(str(ds / "foa" / f"{clip}.flac")) as f:
            if f.frames != N_FRAMES or f.channels != N_CH or f.samplerate != 24000:
                return False
            if f.read(dtype="float32", always_2d=True).shape != (N_FRAMES, N_CH):
                return False
        labels = (ds / "metadata" / f"{clip}.csv").read_text(encoding="utf-8").splitlines()
        if len(labels) != scene["stats"]["n_label_rows"]:
            return False
        for line in labels:
            k, c, tr, az, el = map(int, line.split(","))
            if not (0 <= k < 100 and 0 <= c < 9 and 0 <= tr < 3 and -180 <= az <= 180 and -90 <= el <= 90):   # 2026-10-26: 9 クラス（bicycle = 8・#48）。旧は < 8 で自転車の本を未完了にしていた
                return False
        masks = (ds / "masks" / f"{clip}.csv").read_text(encoding="utf-8").splitlines()
        # 2026-10-26 本体 §2-1・R30-10: v30 の masks は 6 列（frame,class,track,src,snr_a_db,snr_a_bg_db）・帯域ゲート（v30_band・決め直し 4 の既定）は 9 列（＋ snr_band_max_db,band_hz,band_valid・§9-6）
        head, ncol = {"v30": ("frame,class,track,src,snr_a_db,snr_a_bg_db", 6),
                      "v30_band": ("frame,class,track,src,snr_a_db,snr_a_bg_db,snr_band_max_db,band_hz,band_valid", 9)}.get(scene.get("gate_rule"), ("frame,class,snr_a_db", 3))
        if not masks or masks[0] != head:
            return False
        for line in masks[1:]:
            parts = line.split(",")
            if len(parts) != ncol:
                return False
            k, c = parts[0], parts[1]
            if not (0 <= int(k) < 100 and 0 <= int(c) < 9):
                return False
            for v in parts[2:]:
                float(v)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
    return True


def purge_partial(clip: str) -> None:
    """途中で止まった組（foa だけ等）を消してから作り直す。"""
    ds = m9.DS
    if not re.fullmatch(r"fold\d+_room\d+_mix\d+|(train|val|eval|tune)_(syn|field)_clip\d{5}", clip):   # 2026-09-22 改名: 新旧どちらも可
        raise ValueError("invalid clip ID")
    w = ds / "work" / clip
    if w.is_symlink() or not w.resolve().is_relative_to(ds.resolve()):
        raise ValueError("partial work path escapes dataset")
    for p in (ds / "foa" / f"{clip}.flac", ds / "metadata" / f"{clip}.csv", ds / "masks" / f"{clip}.csv", ds / "metadata_dist" / f"{clip}.csv"):
        if p.exists():
            p.unlink()
    w = ds / "work" / clip
    if w.exists():
        for f in w.iterdir():
            f.unlink()
        w.rmdir()


def main() -> None:
    a = sys.argv[1:]
    plan = Path(a[a.index("--plan") + 1]) if "--plan" in a else v19r.PLAN_V19 / "assignment_v19.csv"
    rows = v19r.load_plan_v19(plan)
    assert m9.DS.name == "dataset_outdoor_siren_v19", m9.DS
    assert all(str(r["scenario"]).startswith("v19") for r in rows), "v19 以外の行が混ざっている"
    if "--list" in a:
        print(f"plan {plan}: {len(rows)} rows -> {m9.DS}")
        return
    lo, hi = 0, len(rows) - 1
    if "--rows" in a:
        x, y = a[a.index("--rows") + 1].split("-")
        lo, hi = int(x), int(y)
    part = rows[lo:hi + 1]
    t0 = time.time()
    done = skip = 0
    for i, row in enumerate(part):
        if clip_complete(row["clip_id"], row):
            skip += 1
            continue
        purge_partial(row["clip_id"])           # foa だけ残った組などは作り直す（監査 F7）
        v19r.generate_clip(row)
        assert clip_complete(row["clip_id"], row), f"生成後の検査に失敗: {row['clip_id']}"
        done += 1
        if done % 25 == 0:
            print(f"[{lo}-{hi}] {i + 1}/{len(part)} done={done} skip={skip} {(time.time() - t0) / max(done, 1):.1f}s/clip", flush=True)
    print(f"[{lo}-{hi}] FINISHED done={done} skip={skip} {time.time() - t0:.0f}s -> {m9.DS}", flush=True)


if __name__ == "__main__":
    main()
