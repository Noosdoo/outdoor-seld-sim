# -*- coding: utf-8 -*-
"""_run_v2final_gen.already_done の再利用判定の検査（2026-10-08・Astra 監査 D08-02 の回帰検査）。
  R1 旧 scene.json の row に無い来歴列（twin_of）が今の行で空 → 済み（描き直さない）
  R2 twin_of が今の行で非空（双子の行）で旧 row に無い → 済みにしない（新しい本）
  R3 生成条件の列（seed）が違う → 済みにしない（従来どおり）
  R4 音声・ラベル・masks が 1 つでも欠ける → 済みにしない
  R5〜R7 試し描きの版の照合 smoke_version_diff（RR70-01・2026-10-19）
使い方: PYTHONPATH=scripts:src python scripts/test_run_v2final_gen_reuse.py
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _run_v2final_gen as g  # noqa: E402


def fixture(ds: Path, clip: str, row: dict, missing: str | None = None) -> None:
    (ds / "work" / clip).mkdir(parents=True, exist_ok=True)
    (ds / "work" / clip / "scene.json").write_text(json.dumps({"row": row}), encoding="utf-8")
    for d, ext in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")):
        if d == missing:
            continue
        (ds / d).mkdir(parents=True, exist_ok=True); (ds / d / f"{clip}.{ext}").write_bytes(b"x")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="reuse_")); g.m9.DS = tmp
    old_row = {"clip_id": "train_syn_clip00001", "split": "train", "seed": 1, "n_warnings": 0, "mic_z": "1.5"}
    fixture(tmp, "train_syn_clip00001", old_row)
    res = []
    res.append(("R1 旧 row に twin_of 無し・今の行は twin_of 空 → 済み", g.already_done(dict(old_row, twin_of="")) is True))
    res.append(("R2 今の行は twin_of 非空（双子）→ 済みにしない", g.already_done(dict(old_row, twin_of="train_syn_clip00002")) is False))
    res.append(("R3 seed が違う → 済みにしない", g.already_done(dict(old_row, seed=2, twin_of="")) is False))
    fixture(tmp, "train_syn_clip00003", old_row | {"clip_id": "train_syn_clip00003"}, missing="masks")
    res.append(("R4 masks が無い → 済みにしない", g.already_done(dict(old_row, clip_id="train_syn_clip00003", twin_of="")) is False))
    # RR70-01（Astra 再監査 2026-10-19）: 試し描きの版の照合（plan rows expected の行は除く・sha が 1 本でも違えば止める）
    gv_a = "render chain (import order): a -> b\nplan rows expected: 62102\n0a1b  scripts/step11_v29_render.py\n9f9f  md/design/v2final_params.json\n"
    gv_b = gv_a.replace("0a1b", "0a1c"); gv_c = gv_a.replace("plan rows expected: 62102", "plan rows expected: 49")
    res.append(("R5 同じ版の記録 → 違い無し（再開してよい）", g.smoke_version_diff(gv_a, gv_a) == []))
    res.append(("R6 描画器の sha が違う → 違いが出る（止まる）", len(g.smoke_version_diff(gv_a, gv_b)) == 2))
    res.append(("R7 plan rows expected の行だけ違う → 違い無し", g.smoke_version_diff(gv_a, gv_c) == []))
    ok = 0
    for name, passed in res:
        print(("PASS " if passed else "FAIL ") + name); ok += int(passed)
    print(f"{ok}/{len(res)} " + ("ALL PASS" if ok == len(res) else "NG"))
    return 0 if ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
