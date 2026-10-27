# -*- coding: utf-8 -*-
"""scripts/v30_dev_compare.py の試験（2026-10-27 Opus）。架空の val とデモの組だけ（評価の本は使わない）。
  D1 評価の本の置き場（realsmoke・eval_v4・eval_field_）を入力に渡したら止める（終了コード 2）
  D2 乾式: 主は val の統一採点器・知覚層・デモの組の 3 つ、相手 D は知覚層とデモの組だけ（統一採点器に通さない）・予測の無い家の 13 本は飛ばす
  D3 架空の val（2 本）とデモの組（1 本）で通す: 出力のファイルができ、index.md に顔ぶれごとの行（val 知覚層・本ごとの種類・学内のデモの組）と予測の sha
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import v30_dev_compare as V  # noqa: E402


def run(args):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = V.main([str(x) for x in args])
    return rc, buf.getvalue()


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="devcmp_"))
    try:
        (tmp / "meta").mkdir()
        (tmp / "meta" / "val_syn_clip90101.csv").write_text("".join(f"{k},8,0,90,0,{max(1.0, (75 - k) * 0.2):.2f}\n" for k in range(40, 80)), encoding="utf-8")
        (tmp / "meta" / "val_syn_clip90102.csv").write_text("".join(f"{k},3,0,-90,0,3.00\n" for k in range(50, 70)), encoding="utf-8")
        (tmp / "plan.csv").write_text("clip_id,split,mic_z,motion\nval_syn_clip90101,val,1.5,static\nval_syn_clip90102,val,1.5,static\n", encoding="utf-8")
        pv = tmp / "pred_val.csv"
        pv.write_text("".join([f"val_syn_clip90101,{k},8,0,85,0,2.0\n" for k in range(50, 70)] + [f"val_syn_clip90102,{k},3,0,-85,0,3.0\n" for k in range(52, 61)]), encoding="utf-8")
        inf = tmp / "demo" / "infer"
        (inf / "gt").mkdir(parents=True)
        (inf / "gt" / "tune_field_clipA.csv").write_text("".join(f"{k},8,0,90,-10,2.0\n" for k in range(50, 90)), encoding="utf-8")
        with open(inf / "ann_all.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clip_id", "orig_file", "class", "t_start", "t_cpa", "cut_offset_s", "区分"]); w.writeheader()
            w.writerow({"clip_id": "tune_field_clipA", "orig_file": "a.WAV", "class": "bicycle", "t_start": "", "t_cpa": "13.0", "cut_offset_s": "5.0", "区分": "入れる"})
        pd = tmp / "pred_demo.csv"
        pd.write_text("".join(f"tune_field_clipA,{k},8,0,95,-10,2.5\n" for k in range(55, 75)), encoding="utf-8")
        cfg = {"date": "test", "val": {"plan": str(tmp / "plan.csv"), "meta": str(tmp / "meta")}, "demo": {"infer": str(inf)}, "home": {"dir": str(tmp / "home")},
               "members": [{"label": "main", "kind": "main", "val": str(pv), "demo": str(pd), "home": ""},
                           {"label": "baselineD", "kind": "D", "val": str(pv), "demo": str(pd)}]}
        (tmp / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        bad = dict(cfg, members=[dict(cfg["members"][0], demo="out/realsmoke/x.csv")])
        (tmp / "bad.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        rc1, o1 = run(["--config", tmp / "bad.json", "--out", tmp / "o_bad"])
        bad2 = dict(cfg, val=dict(cfg["val"], meta="out/dataset_outdoor_siren_v2final_v30_eval_v4/metadata"))
        (tmp / "bad2.json").write_text(json.dumps(bad2, ensure_ascii=False), encoding="utf-8")
        rc1b, _ = run(["--config", tmp / "bad2.json", "--out", tmp / "o_bad2"])
        ok.append(("D1 評価の本の置き場（realsmoke・eval_v4）を渡したら止める", rc1 == 2 and rc1b == 2 and "評価の本" in o1 and not (tmp / "o_bad").exists()))
        rc2, o2 = run(["--config", tmp / "cfg.json", "--out", tmp / "o", "--dry"])
        lines = [x for x in o2.splitlines() if x.startswith("[dry]")]
        ok.append(("D2 乾式: 主 3 つ（統一・知覚層・デモ）・D 2 つ（知覚層・デモ）・家の 13 本は予測が無いので飛ばす",
                   rc2 == 0 and len(lines) == 5 and sum("main・val 統一採点器" in x for x in lines) == 1 and not any("baselineD・val 統一採点器" in x for x in lines)
                   and not any("家の 13 本" in x for x in lines)))
        rc3, o3 = run(["--config", tmp / "cfg.json", "--out", tmp / "o"])
        idx = (tmp / "o" / "index.md").read_text(encoding="utf-8") if (tmp / "o" / "index.md").is_file() else ""
        files = ["val_unified_main.md", "val_unified_confusion_main.json", "val_perception_main.json", "val_perception_confusion_main.json", "demo_main.json",
                 "val_perception_baselineD.json", "demo_baselineD.json"]
        ok.append(("D3 通す: 出力のファイル・index.md に顔ぶれごとの行（val 知覚層・本ごとの種類・学内のデモの組）と予測の sha・D に統一採点器の出力は無い",
                   rc3 == 0 and all((tmp / "o" / f).is_file() for f in files) and not (tmp / "o" / "val_unified_baselineD.md").exists()
                   and "## main（main）" in idx and "bicycle 検出(20°)" in idx and "- val 知覚層: " in idx and "- 学内のデモの組: 本 1" in idx and "val 本ごとの種類（参考）" in idx
                   and "main・val: `" in idx and "|" not in "\n".join(l for l in idx.splitlines() if not l.startswith("- ") or "rc " not in l)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
