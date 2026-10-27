# -*- coding: utf-8 -*-
"""test_v2final_conf_scoring.py — 確定評価の台本 server_sde/v2final_conf.sbatch に足した採点 4 本（監査 Astra 9/28 R02）が、開発入力で通るかの検査。

fold20（確定評価セット）の予測・正解は開かない。同じ命令の形を、開発用の予測（out/_b02_2026-09-24/dev_e149_v12causal.csv・run_meta あり）と
v12 の val 名簿・正解で走らせ、終了コード 0・出力ファイルあり・見出しに規則名（v9c／hold）が写ることを確かめる。
  S1 統一採点器 既定（v9c）: rc 0・見出しに「警告音 v9c」
  S2 統一採点器 --warn-rule hold: rc 0・見出しに「警告音 hold」・本文の主指標は S1 と同一（警告音は主指標の GT 外・発火数だけ違う）
  S3 知覚層 _perception_eval.py: rc 0・json あり
  S4 閾値なし _notify_threshold_sweep.py: rc 0・csv あり
  S5 台本の中の 4 本の命令が、この検査と同じ引数の形（--strict・--split eval・--clip-max 1200・--warn-rule hold）を含む
  S6 変更票 #45（2026-10-02）: 4 本すべてに v1 の予測（v1=$V1PRED）、統一採点器の 2 本にオラクル入力（oracle=）、v1 の予測の照合（cmp）と _gt_as_pred.py がある
  S7 オラクル入力（正解をそのまま予測に）を統一採点器に通すと 至近捕捉・検出率 100%・誤捕捉・距離誤差 0%
使い方: python scripts/test_v2final_conf_scoring.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PRED = "out/_b02_2026-09-24/dev_e149_v12causal.csv"
PLAN = "out/dataset_outdoor_siren_v12/plan/manifest_val_syn.csv"
META = "out/dataset_outdoor_siren_v12/metadata_dist"
CLIP_MAX = "400"


def run(args):
    r = subprocess.run([PY] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="conf_scoring_"))
    try:
        common = ["--plan", PLAN, "--split", "val", "--clip-max", CLIP_MAX, "--meta", META]
        rc1, o1 = run([str(ROOT / "scripts/_score_unified.py"), str(tmp / "u_v9c.md"), "--strict", "--raw"] + common + [f"dev={PRED}"])
        rc2, o2 = run([str(ROOT / "scripts/_score_unified.py"), str(tmp / "u_hold.md"), "--warn-rule", "hold", "--strict", "--raw"] + common + [f"dev={PRED}"])
        t1 = (tmp / "u_v9c.md").read_text(encoding="utf-8") if rc1 == 0 else ""
        t2 = (tmp / "u_hold.md").read_text(encoding="utf-8") if rc2 == 0 else ""
        ok.append(("S1 統一採点器 既定: rc 0・見出しに v9c", rc1 == 0 and "警告音 v9c" in t1.splitlines()[0]))
        ok.append(("S2 統一採点器 hold: rc 0・見出しに hold", rc2 == 0 and "警告音 hold" in t2.splitlines()[0]))
        if rc1 == 0 and rc2 == 0:
            row = lambda t: [l for l in t.splitlines() if l.startswith("| dev |")][0].split("|")
            r1, r2 = row(t1), row(t2)
            # 列: 予測, 本数, 至近到達, 強到達, 注意到達, 安全抑制, リード, ≥2.0s前(全事象・#70), 発火数, 検出率, 捕捉(全), 捕捉(条件), 誤捕捉, 至近推定, 距離誤差
            same = [i for i in range(2, 16) if i != 9 and r1[i] == r2[i]]
            ok.append(("S2 主指標は v9c と hold で同一（発火数の列だけ違ってよい）", len(same) == 13 and r1[9] != r2[9]))
        else:
            print("   stderr:", (o1 or o2)[-400:])
        rc3, o3 = run([str(ROOT / "scripts/_perception_eval.py"), str(tmp / "p.md")] + common + ["--json", str(tmp / "p.json"), "--strict", f"dev={PRED}"])
        ok.append(("S3 知覚層: rc 0・json あり", rc3 == 0 and (tmp / "p.json").exists()))
        if rc3 != 0:
            print("   stderr:", o3[-400:])
        rc4, o4 = run([str(ROOT / "scripts/_notify_threshold_sweep.py"), str(tmp / "s.md")] + common + ["--csv", str(tmp / "s.csv"), "--strict", f"dev={PRED}"])
        ok.append(("S4 閾値なし: rc 0・csv あり", rc4 == 0 and (tmp / "s.csv").exists()))
        if rc4 != 0:
            print("   stderr:", o4[-400:])
        sb = (ROOT / "server_sde/v2final_conf.sbatch").read_text(encoding="utf-8")
        need = ["_score_unified.py $CU/unified_core_v9c.md --strict --raw --plan $CP --split eval --clip-max 1200",
                "_score_unified.py $CU/unified_core_hold.md --warn-rule hold --strict --raw --plan $CP --split eval --clip-max 1200",
                "_perception_eval.py $CU/perception_core.md --plan $CP --split eval --clip-max 1200",
                "_notify_threshold_sweep.py $CU/notify_sweep_core.md --plan $CP --split eval --clip-max 1200"]
        ok.append(("S5 台本に 4 本の命令が同じ形で入っている", all(n in sb for n in need) and "assignment_v12conf.csv" in sb and "metadata_dist" in sb))
        # 変更票 #45（2026-10-02）: v1 の予測を 4 本すべてに、オラクル入力を統一採点器の 2 本に渡し、v1 の予測は 8 月の conf_core.csv と照合してから使う
        lines = [l for l in sb.splitlines() if "v2final=$CPRED" in l]
        ok.append(("S6 台本: 4 本すべてに v1=$V1PRED・統一採点器の 2 本に oracle=・照合と _gt_as_pred.py がある",
                   len(lines) == 4 and all("v1=$V1PRED" in l for l in lines)
                   and sum("oracle=$CU/gt_as_pred_core.csv" in l for l in lines) == 2
                   and all("oracle=" in l for l in lines if "_score_unified.py" in l)
                   and "cmp -s - \"$V1CORE\"" in sb and "_gt_as_pred.py --plan $CP --split eval --clip-max 1200" in sb))
        # S7 オラクル入力: 正解をそのまま予測にすると、統一採点器で 至近捕捉(全GT) 100%・検出率 100%・誤捕捉 0%
        rc5, o5 = run([str(ROOT / "scripts/_gt_as_pred.py"), "--plan", PLAN, "--split", "val", "--clip-max", CLIP_MAX, "--meta", META, "--out", str(tmp / "gt.csv")])
        rc6, o6 = run([str(ROOT / "scripts/_score_unified.py"), str(tmp / "u_or.md"), "--strict", "--raw"] + common + [f"dev={PRED}", f"oracle={tmp / 'gt.csv'}"])
        t6 = (tmp / "u_or.md").read_text(encoding="utf-8") if rc6 == 0 else ""
        raw_or = [l for l in t6.splitlines() if l.startswith("| oracle |") and "/" in l]
        good = False
        if raw_or:
            cells = [c.strip() for c in raw_or[0].split("|")]
            # 列: 予測, 強到達 a/b = x%, 至近到達, 至近捕捉(全GT) cap/n_gt = x%, 検出率 det/n_gt = x%, 距離誤差, ペア, 誤捕捉, ...
            pct = lambda c: c.split("=")[-1].strip()      # Astra RN20-08: --raw の欄は「分子/分母 = x%」（R72-08 で誤捕捉も 0/24710 = 0.0000% の形）か「x%」。右端の % だけ比べる
            cap = pct(cells[4]); det = pct(cells[5]); fp = pct(cells[8]); derr = pct(cells[6])
            good = cap == "100.0000%" and det == "100.0000%" and fp == "0.0000%" and derr == "0.0000%"
        ok.append(("S7 オラクル入力は 至近捕捉・検出率 100%・誤捕捉・距離誤差 0%", rc5 == 0 and rc6 == 0 and good))
        if not (rc5 == 0 and rc6 == 0 and good):
            print("   gt_as_pred:", o5[-300:]); print("   unified:", o6[-300:]); print("   row:", raw_or[:1])
        # S8（Astra RN20-08・RN20-03）: 確定評価 v3 の台本の順番 = 実行の契約（乾式の照合）→ 封印の照合 → 開封の予約 → 推論 → eval v3 で 4 本の採点（表は v2final だけ）
        sb3 = (ROOT / "server_sde/v2final_conf_v3.sbatch").read_text(encoding="utf-8")
        i_contract = sb3.find('_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --dry'); i_seal = sb3.find("[seal] files="); i_res = sb3.find("--reserve-dir"); i_inf = sb3.find("$JOBDIR/_causal_infer_v17.py")
        ok.append(("S8 conf_v3: 契約の照合 < 封印の照合 < 開封の予約 < 推論・4 本の採点は eval v3（--split eval --clip-max $N）・表は --dist-table-for v2final",
                   0 < i_contract < i_seal < i_res < i_inf and sb3.count("--split eval --clip-max $N") >= 4 and "--dist-table-for v2final" in sb3
                   and "V2FINAL_SELECT_JSON" in sb3 and "V2FINAL_DECIDE_JSON" in sb3))
        # S9（変更票 #73 Q2）: 規則だけを替えた採点（--notify-rule detect・v2final だけ・警告音の規則は同じ）が 1 本あり、sha の一覧に _notify_detect.py が入る
        det_lines = [l for l in sb3.splitlines() if "--notify-rule detect" in l and "_score_unified.py" in l]
        ok.append(("S9 conf_v3: --notify-rule detect の採点が 1 本（v2final だけ・--split eval）・sha の一覧に _notify_detect.py",
                   len(det_lines) == 1 and "v2final=$V2PRED" in det_lines[0] and "v1=" not in det_lines[0] and "--split eval --clip-max $N" in det_lines[0]
                   and "scripts/_notify_detect.py" in sb3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
