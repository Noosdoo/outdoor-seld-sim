# -*- coding: utf-8 -*-
"""test_field_score_members.py — 実録の採点の顔ぶれの回し方（field_score_members.py・--dry）の検査（2026-10-21）。
  F1 無い相手は飛ばして index に書く・ある相手は 来歴 → step20 → step19k の順
  F2 v2final だけ hold と detect を足す・baselineD は step20 に入れない（距離なし・#73 Q3）・D があれば commonD（6 クラス・_filter_classes → _perception_eval）
  F3 知らないメンバー名は 3・run_log.json に命令が残る
  F4 --sync・--extend-takes を step19k の全部の命令に渡す（2026-10-25・旧は渡していなかった）
  F5 --reads に生の step19h の reads（距離クラスの行に midpoint の印なし）を渡すと 3 で止まる・step19l の reads_mid なら通る（2026-10-25）
  F6 印は本・クラス・track ごと（Astra MAP25-03）: 主・車だけの感度・混ざった感度（車 front・自転車 midpoint）は通る・変換されていない本は除いた写し・取り違えは 3
使い方: python scripts/test_field_score_members.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(args):
    e = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([PY, str(ROOT / "scripts/field_score_members.py"), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=ROOT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="fsm_"))
    try:
        root = tmp / "preds"; tag = "ftest"
        for d in ("infer_field_ftest", "infer_field_ftest_abl_nohyst", "infer_field_ftest_baselineD"):
            (root / d).mkdir(parents=True); (root / d / "val_all_causal.csv").write_text("", encoding="utf-8")
        out = tmp / "out"
        common = ["--tag", tag, "--pred-root", str(root), "--n", "5", "--scoring", "old", "--ann-score", str(tmp / "a.csv"), "--ann-clips", str(tmp / "b.csv"), "--reads", str(tmp / "r.csv"), "--out", str(out), "--dry"]
        rc, o = run(common)
        L = json.loads((out / "run_log.json").read_text(encoding="utf-8")) if (out / "run_log.json").is_file() else {}
        whats = [e["what"] for e in L.get("log", [])]
        idx = (out / "index.md").read_text(encoding="utf-8") if (out / "index.md").is_file() else ""
        ok.append(("F1 無い相手（abl_nocausalft・abl_scratch・abl_noheight・baselineB）は飛ばして index に書く・ある相手は 来歴 → step20 → step19k",
                   rc == 0 and L.get("members_present") == ["v2final", "abl_nohyst", "baselineD"] and set(L.get("members_absent", [])) == {"abl_nocausalft", "abl_scratch", "abl_noheight", "baselineB"}
                   and "abl_scratch" in idx and whats[:2] == ["v2final: 来歴の照合", "v2final: step20 v9c"] and "abl_nohyst: step20 v9c" in whats and "abl_nohyst: step19k 知覚層" in whats
                   and whats.index("abl_nohyst: 来歴の照合") < whats.index("abl_nohyst: step20 v9c") < whats.index("abl_nohyst: step19k 知覚層")))
        d_cmds = [e for e in L.get("log", []) if e["what"].startswith("baselineD:")]
        v2_cmds = [e["what"] for e in L.get("log", []) if e["what"].startswith("v2final:")]
        ok.append(("F2 v2final だけ hold と detect・baselineD は step20 に入れない・D があれば commonD（写し 2 本 → 知覚層・6 クラス）",
                   "v2final: step20 hold（併記）" in v2_cmds and "v2final: step20 detect（規則だけの比較・#73 Q2）" in v2_cmds
                   and all("step20" not in e["what"] for e in d_cmds) and any("step19k" in e["what"] for e in d_cmds)
                   and sum(1 for w in whats if w.startswith("commonD:")) == 3
                   and any("--classes" in e["cmd"] and "0,1,3,4,7,8" in e["cmd"] for e in L.get("log", []))
                   and not any("step20 hold" in w or "step20 detect" in w for w in whats if not w.startswith("v2final:"))))
        rc3, o3 = run(common[:-1] + ["--members", "v2final,xyz", "--dry"])
        ok.append(("F3 知らないメンバー名は 3・run_log.json に命令（文字列）が残る", rc3 == 3 and all(isinstance(c, str) for e in L.get("log", []) for c in e["cmd"])))
        out4 = tmp / "out4"
        rc4, o4 = run([x if x != str(out) else str(out4) for x in common] + ["--sync", str(tmp / "s.csv"), "--extend-takes", "X1.WAV,X2.WAV"])
        L4 = json.loads((out4 / "run_log.json").read_text(encoding="utf-8")) if (out4 / "run_log.json").is_file() else {}
        k4 = [e["cmd"] for e in L4.get("log", []) if "step19k" in e["what"]]
        ok.append(("F4 --sync・--extend-takes を step19k の全部の命令に渡す", rc4 == 0 and len(k4) == 3
                   and all("--sync" in c and c[c.index("--sync") + 1] == str(tmp / "s.csv") and "--extend-takes" in c and c[c.index("--extend-takes") + 1] == "X1.WAV,X2.WAV" for c in k4)))
        raw_reads, mid_reads = tmp / "reads_A.csv", tmp / "reads_mid_A.csv"
        raw_reads.write_text("take,frame,cls,track,x,y,z,t,hidden\nT.WAV,,car,1,10,2.0,,3.0,\nT.WAV,,mic,0,0,5,,3.0,\n", encoding="utf-8")
        mid_reads.write_text("take,frame,cls,track,x,y,z,t,hidden,converted,shift_m\nT.WAV,,car,1,8.685,2.0,,3.0,,midpoint,-1.315\nT.WAV,,mic,0,0,5,,3.0,,,\n", encoding="utf-8")
        rc5r, o5r = run([x if x != str(tmp / "r.csv") else str(raw_reads) for x in common])
        rc5m, _ = run([x if x != str(tmp / "r.csv") else str(mid_reads) for x in common])
        ok.append(("F5 生の reads は 3 で止まる（midpoint の印なし）・reads_mid は通る", rc5r == 3 and "reads_mid" in o5r and rc5m == 0))
        # F6（Astra MAP25-03・2026-10-25）: 印は本・クラス・track ごと。感度は車 front・自転車 midpoint（step19l の reads_front の正式な形）を通す。
        # 変換されていない本は止めずに除いた写しを step19k に渡す。主に front だけ・感度に front 無しは止める
        hdr = "take,frame,cls,track,x,y,z,t,hidden,converted,shift_m\n"
        cases = {
            "main": (hdr + "T1.WAV,,car,1,8.7,2.0,,3.0,,midpoint,-1.3\nT1.WAV,,bicycle,2,9.0,1.0,,4.0,,midpoint,-0.5\n", False),
            "front_car_only": (hdr + "T1.WAV,,car,1,10,2.0,,3.0,,front,\n", True),
            "front_mixed": (hdr + "T1.WAV,,car,1,10,2.0,,3.0,,front,\nT2.WAV,,bicycle,1,9.0,1.0,,4.0,,midpoint,-0.5\n", True),
            "one_unconverted": (hdr + "T1.WAV,,car,1,8.7,2.0,,3.0,,midpoint,-1.3\nT2.WAV,,car,1,10,2.0,,5.0,,,\nT2.WAV,,car,1,,,,6.0,start,,\n", False),
            "main_given_front": (hdr + "T1.WAV,,car,1,10,2.0,,3.0,,front,\n", False),
            "front_given_main": (hdr + "T1.WAV,,car,1,8.7,2.0,,3.0,,midpoint,-1.3\n", True),
        }
        res6 = {}
        for name, (text, front) in cases.items():
            rp = tmp / f"reads_{name}.csv"; rp.write_text(text, encoding="utf-8")
            o6 = tmp / f"out6_{name}"
            args = [x if x != str(tmp / "r.csv") else str(rp) for x in common]
            args = [x if x != str(out) else str(o6) for x in args] + (["--reads-front"] if front else [])
            rc6, log6 = run(args)
            L6 = json.loads((o6 / "run_log.json").read_text(encoding="utf-8")) if (o6 / "run_log.json").is_file() else {}
            kr = [e["cmd"][e["cmd"].index("--reads") + 1] for e in L6.get("log", []) if "step19k" in e["what"]]
            res6[name] = (rc6, kr, log6)
        chk = (o6p := tmp / "out6_one_unconverted" / "reads_checked.csv").is_file() and "T2.WAV" not in o6p.read_text(encoding="utf-8") and "T1.WAV" in o6p.read_text(encoding="utf-8")
        ok.append(("F6 主・車だけの感度・混ざった感度は通る（step19k には元の reads）・変換されていない本 1 本は除いた写しを渡す・主に front だけ／感度に front 無しは 3",
                   all(res6[n][0] == 0 and res6[n][1] and all(k.endswith(f"reads_{n}.csv") for k in res6[n][1]) for n in ("main", "front_car_only", "front_mixed"))
                   and res6["one_unconverted"][0] == 0 and res6["one_unconverted"][1] and all(k.endswith("reads_checked.csv") for k in res6["one_unconverted"][1])
                   and chk and "T2.WAV car track1" in res6["one_unconverted"][2]
                   and res6["main_given_front"][0] == 3 and res6["front_given_main"][0] == 3))
        # F7（Astra R142-04・10/27）: 警告音の読みの印。E3 の horn に horn_offset が無い本・ベルに bell_offset が無い本は除いて知らせる（ある本は残す）
        rp7 = tmp / "reads_warn.csv"
        rp7.write_text(hdr + "T1.WAV,,car_drive,1,8.7,2.0,,3.0,,midpoint,-1.3\nT1.WAV,,horn,1,10.85,2.0,,3.0,,horn_offset,0.85\n"
                       "T2.WAV,,car_drive,1,8.7,2.0,,3.0,,midpoint,-1.3\nT2.WAV,,horn,1,10.0,2.0,,3.0,,,\nT3.WAV,,bike_bell,1,9.7,2.0,,3.0,,bell_offset,-0.3\n", encoding="utf-8")
        o7 = tmp / "out7"
        rc7, log7 = run([x if x != str(tmp / "r.csv") else str(rp7) for x in [y if y != str(out) else str(o7) for y in common]])
        ck7 = (o7 / "reads_checked.csv").read_text(encoding="utf-8") if (o7 / "reads_checked.csv").is_file() else ""
        ok.append(("F7 E3 の horn に horn_offset が無い本（T2）は除いて知らせる・T1 の horn と T3 のベル・T2 の車は残す",
                   rc7 == 0 and "T2.WAV horn track1" in log7 and "T1.WAV,,horn" in ck7 and "T3.WAV,,bike_bell" in ck7 and "T2.WAV,,car_drive" in ck7 and "T2.WAV,,horn" not in ck7))
        # F8（Astra R142-04）: 注釈に警告音があるのに --warn-interval が無ければ 3・付ければ通る・旧の形の再現は --no-warn-interval
        ann8 = tmp / "ann8.csv"; ann8.write_text("clip_id,class\nc1,bike_bell\nc2,car_drive\n", encoding="utf-8")
        base8 = [x if x != str(tmp / "b.csv") else str(ann8) for x in common]
        rc8a, o8a = run([x if x != str(out) else str(tmp / "o8a") for x in base8])
        rc8b, _ = run([x if x != str(out) else str(tmp / "o8b") for x in base8] + ["--warn-interval", str(tmp / "w.csv")])
        rc8c, _ = run([x if x != str(out) else str(tmp / "o8c") for x in base8] + ["--no-warn-interval"])
        ok.append(("F8 警告音の注釈があるのに区間の表が無い → 3・--warn-interval で通る・--no-warn-interval（旧の形）でも通る", rc8a == 3 and "--warn-interval" in o8a and rc8b == 0 and rc8c == 0))
        # F9（Astra R142-05・10/27）: --scoring v30 → step20 の全部の命令に --warmup-v30 --min-history 4.0・区間の表の鳴り終わりをクリップの秒へ移した t_warn_end の写しを渡す
        acols = "clip_id,event_id,class,t_start,t_cpa,t_end,区分,orig_file,cut_offset_s,scored\n"
        ann9 = tmp / "ann9.csv"
        ann9.write_text(acols + "c1,1,bike_bell,4.50,7.50,,W,261022_071.WAV,8.000,1\nc2,1,horn,3.00,6.00,,E,261029_020.WAV,8.000,1\n"
                        "c3,1,horn,3.00,6.00,,E,261029_004.WAV,6.000,1\nc4,1,car_drive,3.00,8.00,,A,261029_005.WAV,0.000,1\nc5,1,horn,2.00,2.00,9.00,G,261029_090.WAV,0.000,1\n", encoding="utf-8")
        w9 = tmp / "warn9.csv"
        w9.write_text("take,class,on_s,end_s\n261022_071.WAV,bike_bell,12.5000,13.1000\n261029_020.WAV,horn,11.0000,11.4000\n261029_004.WAV,horn,9.0000,12.6000\n", encoding="utf-8")
        base9 = [x if x not in (str(tmp / "a.csv"), str(tmp / "b.csv")) else str(ann9) for x in common]
        base9 = [x if x != "old" else "v30" for x in base9]
        o9 = tmp / "out9"
        rc9, log9 = run([x if x != str(out) else str(o9) for x in base9] + ["--warn-interval", str(w9)])
        L9 = json.loads((o9 / "run_log.json").read_text(encoding="utf-8")) if (o9 / "run_log.json").is_file() else {}
        s20 = [e["cmd"] for e in L9.get("log", []) if "step20" in e["what"]]
        A9 = {r["clip_id"]: r for r in csv.DictReader(open(o9 / "ann_score_v30.csv", encoding="utf-8"))} if (o9 / "ann_score_v30.csv").is_file() else {}
        we = {k: (v["t_warn_end"], v["t_end"]) for k, v in A9.items()}
        rc9m, log9m = run([x if x != str(out) else str(tmp / "o9m") for x in base9] + ["--warn-interval", str(tmp / "w.csv")])   # 区間の表が無い（--dry でもファイルが無い → 写しを作らない）
        w9b = tmp / "warn9b.csv"; w9b.write_text("take,class,on_s,end_s\n261022_071.WAV,bike_bell,12.5000,13.1000\n", encoding="utf-8")
        rc9b, log9b = run([x if x != str(out) else str(tmp / "o9b") for x in base9] + ["--warn-interval", str(w9b)])
        rc9n, _ = run([x if x != str(out) else str(tmp / "o9n") for x in base9] + ["--no-warn-interval"])
        rc9s, _ = run([x for x in common if x not in ("--scoring", "old")])
        ok.append(("F9 v30: step20 の全部の命令（v2final の v9c・hold・detect と abl_nohyst の 4 本）に --warmup-v30 --min-history 4.0・--ann は写し・t_warn_end = ベル 5.10／E3 3.40（暖機より前）／E2 6.60（またぐ）・車と計画外（G）は空・t_end は変えない・"
                   "区間の表に無い警告音は 3・v30 に --no-warn-interval は 3・--scoring 無しは止まる",
                   rc9 == 0 and len(s20) == 4 and all("--warmup-v30" in c and c[c.index("--min-history") + 1] == "4.0" and c[c.index("--ann") + 1].endswith("ann_score_v30.csv") for c in s20)
                   and we == {"c1": ("5.10", ""), "c2": ("3.40", ""), "c3": ("6.60", ""), "c4": ("", ""), "c5": ("", "9.00")}
                   and rc9m == 0 and rc9b == 3 and "区間の表に無い" in log9b and "c2 horn" in log9b and rc9n == 3 and rc9s == 2))
        if not ok[-1][1]:
            print((rc9, len(s20), we, rc9m, rc9b, rc9n, rc9s, log9[-300:]))
        # F1 の旧の形: step20 に暖機を付けない
        ok.append(("F10 --scoring old: step20 に --warmup-v30 を付けない", all("--warmup-v30" not in e["cmd"] for e in L.get("log", []))))
        if not all(c for _, c in ok):
            print(o[-1200:]); print(whats)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
