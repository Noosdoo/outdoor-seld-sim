# -*- coding: utf-8 -*-
"""scripts/_class_confusion.py の試験（2026-10-27 Opus）。架空の枠だけ。
  C1 対角: 正解 自転車・出した 自転車 → (自転車, 自転車)
  C2 取り違え: 正解 自転車・出した キック → (自転車, キック)・見逃しにも誤報にも数えない
  C3 見逃し: 正解 ベル・何も出さない → (ベル, なし)
  C4 誤報: 正解なし・出した 車 → (なし, 車)・nogt にも数える／正解 自転車を当てつつ余分に 車 → (自転車, 自転車) と (なし, 車)・nogt には数えない
  C5 複数: 正解 {ベル, 自転車}・出した {自転車, 車} → (自転車, 自転車)・(ベル, 車)
  C6 本ごとの種類: 一番多い種類・同じ枠数なら先に出た種類・発火なし → 未検出（不正解）・正答率と取り違えの数
  C7 md の行: 表を使わない（| を含まない）・0 の組は書かない
"""
from __future__ import annotations

import sys
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _class_confusion as CC  # noqa: E402


def main() -> int:
    ok = []
    N = CC.NONE
    M = CC.new_matrix(); CC.add_frame(M, {8}, {8})
    ok.append(("C1 対角", M["m"][8][8] == 1 and sum(map(sum, M["m"])) == 1))
    M = CC.new_matrix(); CC.add_frame(M, {8}, {6})
    ok.append(("C2 取り違え (自転車, キック)", M["m"][8][6] == 1 and M["m"][8][N] == 0 and M["m"][N][6] == 0))
    M = CC.new_matrix(); CC.add_frame(M, {3}, set())
    ok.append(("C3 見逃し (ベル, なし)", M["m"][3][N] == 1))
    M = CC.new_matrix(); CC.add_frame(M, set(), {4}); CC.add_frame(M, {8}, {8, 4})
    ok.append(("C4 誤報: 正解なしの枠 (なし, 車)・nogt 1／当てつつ余分な 車 も (なし, 車)・nogt は増えない",
               M["m"][N][4] == 2 and M["nogt"][4] == 1 and M["m"][8][8] == 1 and M["n_frames"] == 2))
    M = CC.new_matrix(); CC.add_frame(M, {3, 8}, {8, 4})
    ok.append(("C5 複数: (自転車, 自転車)・(ベル, 車)", M["m"][8][8] == 1 and M["m"][3][4] == 1 and M["m"][N][4] == 0))
    fires = {40: {6}, 41: {8}, 42: {8}, 43: {6}, 50: {4}}
    t1 = CC.take_class(fires, range(40, 60))                 # 自転車 2・キック 2（キックが先）・車 1 → キック
    t2 = CC.take_class({45: {8}, 46: {8}, 47: {3}}, range(40, 60))
    t3 = CC.take_class({}, range(40, 60))
    R = CC.take_accuracy([{"gt": 8, "pred": t1}, {"gt": 8, "pred": t2}, {"gt": 3, "pred": t3}])
    ok.append((f"C6 本ごと: 同数はキックが先 → {CC.NAMES.get(t1)}・自転車 → {CC.NAMES.get(t2)}・発火なし → 未検出・正答 {R['correct']}/{R['n']}",
               t1 == 6 and t2 == 8 and t3 is None and R["correct"] == 1 and R["n"] == 3
               and R["per_class"][8] == {"n": 2, "correct": 1, "undetected": 0, "as": {6: 1, 8: 1}} and R["per_class"][3]["undetected"] == 1))
    tb = CC.take_class({k: ({8, 3} if k in (50, 51) else {8}) for k in range(45, 60)}, range(40, 60), ignore={8})   # ベルの本: 自転車（正解にある）は数えない
    tb2 = CC.take_class({k: ({8, 6} if k in (50, 51) else {8}) for k in range(45, 60)}, range(40, 60), ignore={3})  # 正解に無いキックは数える
    ok.append(("C6b 正解にある別の種類は取り違えに数えない（ベルの本の自転車 → ベル）・正解に無い種類は数える",
               tb == 3 and tb2 == 8 and CC.take_class({k: {8} for k in range(45, 60)}, range(40, 60), ignore={8}) is None))
    M = CC.new_matrix(); CC.add_clip(M, {50: {8}, 51: {8}, 52: {3}}, {50: {8}, 51: {6}}, range(40, 60))
    L = CC.md_lines(M) + CC.md_lines_take(R)
    ok.append(("C7 md の行: | を含まない・0 の組を書かない・正解 自転車 の行に 自転車 1・キック 1",
               not any("|" in s for s in L) and any(s.startswith("- 正解 自転車（2 枠）: 自転車 1・キック 1") for s in L)
               and not any("サイレン 0" in s for s in L) and M["m"][N][N] == 0))
    M2 = CC.merge(M, M)
    ok.append(("C8 足し合わせ（帯・全体の集計）", M2["m"][8][8] == 2 and M2["n_frames"] == 2 * M["n_frames"]))
    # C9 台帳の窓: 警告音 = 採点上の鳴り始め ±（−1, +4 s）・距離のクラス = 最初の正解の枠 − 10 〜 物理の最接近 + 10・除外の行は外す
    W = CC.windows_from_ledger([{"cls": "0", "track": "0", "kind": "warn", "score_onset_s": "5.0", "excluded": ""},
                                {"cls": "8", "track": "1", "kind": "dist", "phys_cpa_s": "7.0", "excluded": ""},
                                {"cls": "4", "track": "0", "kind": "dist", "phys_cpa_s": "3.0", "excluded": "warmup_cpa"}], {(8, 1): {50: 3.0, 60: 1.0}})
    ok.append(("C9 台帳の窓", W == [(0, range(40, 91)), (8, range(40, 81))]))
    # C10 採点器の --confusion: 主の md は付けても付けなくても 1 バイトも同じ・別の json と .md ができ、数が合う
    import subprocess
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "meta").mkdir()
        (td / "meta" / "val_syn_clip90101.csv").write_text("".join(f"{k},8,0,90,0,{max(1.0, (75 - k) * 0.2):.2f}\n" for k in range(40, 80)), encoding="utf-8")
        (td / "meta" / "val_syn_clip90102.csv").write_text("".join(f"{k},3,0,-90,0,3.00\n" for k in range(50, 70)), encoding="utf-8")
        (td / "plan.csv").write_text("clip_id,split,mic_z,motion\nval_syn_clip90101,val,1.5,static\nval_syn_clip90102,val,1.5,static\n", encoding="utf-8")
        pred = td / "pred.csv"
        pred.write_text("".join([f"val_syn_clip90101,{k},8,0,85,0,2.0\n" for k in range(50, 70)] + [f"val_syn_clip90101,{k},6,0,85,0,2.0\n" for k in range(70, 75)]
                                + [f"val_syn_clip90102,{k},3,0,-85,0,3.0\n" for k in range(52, 61)] + ["val_syn_clip90102,61,8,0,-85,0,3.0\n"]), encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1")
        same, made = [], []
        for script, extra in (("_perception_eval.py", []), ("_score_unified.py", ["--no-t3-sensitivity"])):
            outs = []
            for tag, cf in (("a", []), ("b", ["--confusion", str(td / f"cf_{script}.json")])):
                md = td / f"{script}.md"                       # 同じ名前（見出しに名前が入るため）に書いて、付けたときと付けないときを比べる
                r = subprocess.run([sys.executable, str(E2E / "scripts" / script), str(md), "--plan", str(td / "plan.csv"), "--split", "val", "--meta", str(td / "meta"),
                                    *extra, *cf, f"m={pred}"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(E2E), env=env)
                outs.append(md.read_bytes() if md.is_file() else b"" + r.stderr.encode())
            same.append(outs[0] == outs[1] and len(outs[0]) > 100)
            js = td / f"cf_{script}.json"
            made.append(js.is_file() and js.with_suffix(".md").is_file())
        import json
        cj = json.loads((td / "cf__score_unified.py.json").read_text(encoding="utf-8"))["m"]
        cp = json.loads((td / "cf__perception_eval.py.json").read_text(encoding="utf-8"))["m"]
        m = cj["confusion"]["matrix"]
        ok.append((f"C10 採点器の --confusion: 主の md は同じ {same}・別の出力 {made}・(自転車, 自転車) {m[8][8]}・(自転車, キック) {m[8][6]}・(自転車, なし) {m[8][9]}・(ベル, ベル) {m[3][3]}・"
                   f"(ベル, 自転車) {m[3][8]}・本ごと {cj['take_accuracy']['correct']}/{cj['take_accuracy']['n']}",
                   all(same) and all(made) and m[8][8] == 20 and m[8][6] == 5 and m[8][9] == 15 and m[3][3] == 9 and m[3][8] == 1 and m[3][9] == 10
                   and cj["take_accuracy"]["correct"] == 2 and cj == cp))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
