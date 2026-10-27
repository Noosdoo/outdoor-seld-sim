# -*- coding: utf-8 -*-
"""本体 §2-5（§9-1 の 1・2・§9-4 の 1・4・変更票 #75 ⑭⑮⑯⑰）の受け入れ試験: 事象台帳 `_event_ledger_v30.py` と統一採点器の `--events`。
架空の本（scene.json・正解・予測）だけを使う（評価セット・実録は使わない）。

  E1 途切れのある 1 台（正解の枠 30〜54 と 59〜90）が台帳で 1 事象・可聴区間 = 連続ランの一覧 "30-54;59-90"・物理の最接近 6.0 s・水平 2.0 m・注意
     （旧の連続ランなら 2 事象 = 採点器の分母 2 → --events で 1）
  E2 暖機の境目: 通過の物理の最接近 3.9 s は除外（warmup_cpa）・4.1 s は残る／警告音は鳴り終わり 3.9 s を除外（warmup_end）・鳴り始め 3.0〜9.0 s は残り採点上の鳴り始め 4.0
  E3 正解が全部消えた車（ラベル 0）も台帳に入り（all_lost 1）分母に数える（未通知）・除外の理由にしない・summary に除外前・理由・分母・全消失
  E4 警告音の届いた行: 0〜10 s 鳴り続けるサイレン（予測も 0〜99 枠）は、規則を frame 40 から始めると届いた 1/1・frame 0 からだと最初の発火が frame 0 付近で窓の外 → 0/1（監査 RR30-01 の再現例）／
     鳴り始め 5.0 s のサイレンで予測が 9.5 s から = 窓 [4.0, 9.0] の外 → 0/1・5.5 s から → 1/1
  E5 距離の誤差の帯: 予測の距離 = 正解 × 1.1 → 至近・注意・合わせの誤差がどれも 10.0%・定数の行（至近 1.39 m など）が出る・「2.0 秒以上前」は参考の節（主の表の列から外れる）
  E6 --events なしの既定の出力が §2-5 の前（834353c91）の統一採点器と 1 バイトも同じ
  E7（Fable 10/26 16:00）: 通知の帰属の方位は scene の軌道（台帳の az_series）。正解が全部消えた至近の車でも、方向の合う発火があれば通知あり（至近到達に数える）／
     発火なし → 未通知／方向の違う発火（一律 +120°）→ 未通知。ラベルのある枠の軌道の方位は架空の正解（伝わりの遅れなしの近似）と 2° 以内
使い方: python scripts/test_score_unified_v30.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _event_ledger_v30 as L  # noqa: E402

REF_COMMIT = "834353c91"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def car(y, t_cpa, v=5.0, z=0.5, track=0, cls="car_drive"):
    x0 = -v * t_cpa
    return {"class": cls, "kind": "vehicle", "track": track, "t_on": 0.0, "t_off": 10.0, "speed_mps": v,
            "wp": [[0.0, x0, y, z], [10.0, x0 + 10.0 * v, y, z]]}


def warn(cls, t_on, t_off, y=5.0, x=8.0, track=0):
    return {"class": cls, "kind": "vehicle", "track": track, "t_on": t_on, "t_off": t_off, "wp": [[0.0, x, y, 1.0], [10.0, x, y, 1.0]]}


def gt_rows(src, ci, frames, trk=0):
    out = []
    for k in frames:
        t = (k + 0.5) / 10.0
        x0, y = src["wp"][0][1], src["wp"][0][2]
        x = x0 + (src["wp"][1][1] - x0) * t / 10.0
        out.append(f"{k},{ci},{trk},{int(round(math.degrees(math.atan2(y, x))))},0,{math.hypot(x, y):.2f}")
    return out


def build(td: Path):
    """架空の置き場: clip → (sources, 正解の行)。"""
    clips = {
        "val_syn_clip90001": ([car(2.0, 6.0)], lambda s: gt_rows(s[0], 4, list(range(30, 55)) + list(range(59, 91)))),     # 途切れのある 1 台
        "val_syn_clip90002": ([car(2.0, 3.9)], lambda s: gt_rows(s[0], 4, range(20, 60))),                                 # 最接近 3.9 s
        "val_syn_clip90003": ([car(2.0, 4.1)], lambda s: gt_rows(s[0], 4, range(20, 60))),                                 # 最接近 4.1 s
        "val_syn_clip90004": ([warn("siren", 3.0, 9.0)], lambda s: gt_rows(s[0], 0, range(30, 90))),                       # 3.0〜9.0 s
        "val_syn_clip90005": ([warn("siren", 1.0, 3.9)], lambda s: gt_rows(s[0], 0, range(10, 39))),                       # 3.9 s に鳴り終わる
        "val_syn_clip90006": ([car(1.2, 6.0)], lambda s: []),                                                             # 正解が全部消えた車（至近）
        "val_syn_clip90007": ([warn("siren", 0.0, 10.0)], lambda s: gt_rows(s[0], 0, range(0, 100))),                      # 0〜10 s 鳴り続ける
        "val_syn_clip90008": ([warn("siren", 5.0, 10.0)], lambda s: gt_rows(s[0], 0, range(50, 100))),                     # 鳴り始め 5.0
        "val_syn_clip90009": ([car(2.0, 4.0)], lambda s: gt_rows(s[0], 4, range(20, 60))),                                 # 最接近ちょうど 4.0 s（残る）
        "val_syn_clip90010": ([warn("horn", 2.0, 4.0)], lambda s: gt_rows(s[0], 1, range(20, 40))),                        # 鳴り終わりちょうど 4.0 s（外す）
        "val_syn_clip90011": ([warn("horn", 2.0, 4.1)], lambda s: gt_rows(s[0], 1, range(20, 41))),                        # 鳴り終わり 4.1 s（残る・鳴り始め 4.0）
    }
    for d in ("work", "metadata", "meta"):
        (td / d).mkdir()
    for c, (srcs, gtf) in clips.items():
        (td / "work" / c).mkdir()
        (td / "work" / c / "scene.json").write_text(json.dumps({"row": {"clip_id": c}, "mic": {"motion": "static", "mic_z": 1.5}, "sources": srcs}), encoding="utf-8")
        rows = gtf(srcs)
        txt = "".join(r + "\n" for r in rows)
        (td / "metadata" / f"{c}.csv").write_text(txt, encoding="utf-8")
        (td / "meta" / f"{c}.csv").write_text(txt, encoding="utf-8")
    with open(td / "plan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_id", "split", "mic_z"]); w.writerows([[c, "val", "1.5"] for c in clips])
    return list(clips)


def write_pred(path: Path, rows):
    path.write_text("".join(f"{c},{k},{ci},0,{az},0,{d}\n" for c, k, ci, az, d in rows), encoding="utf-8")


def run_scorer(script: Path, td: Path, out: str, pred: Path, extra=()):
    r = subprocess.run([sys.executable, str(script), str(td / out), "--plan", str(td / "plan.csv"), "--split", "val", "--meta", str(td / "meta"),
                        "--no-t3-sensitivity", *extra, f"m={pred}"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT), env=dict(os.environ, PYTHONUTF8="1"))
    return r, ((td / out).read_text(encoding="utf-8") if (td / out).is_file() else "")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clips = build(td)
        rows, summ = L.build(td, clips)
        by = {(r["clip"], r["class"]): r for r in rows}
        e1 = by[("val_syn_clip90001", "car_drive")]
        check("E1 途切れのある 1 台 = 台帳で 1 事象・可聴区間 30-54;59-90・物理の最接近 6.0 s・水平 2.0 m・注意",
              sum(1 for r in rows if r["clip"] == "val_syn_clip90001") == 1 and e1["runs"] == "30-54;59-90" and e1["phys_cpa_s"] == 6.0
              and abs(e1["phys_cpa_h_m"] - 2.0) < 1e-6 and e1["tier"] == "caution" and e1["n_label"] == 57, str({k: e1[k] for k in ("runs", "phys_cpa_s", "phys_cpa_h_m", "tier")}))
        ex = {c: by[(c, k)]["excluded"] for c, k in (("val_syn_clip90002", "car_drive"), ("val_syn_clip90003", "car_drive"), ("val_syn_clip90004", "siren"), ("val_syn_clip90005", "siren"))}
        ex.update({c: by[(c, k)]["excluded"] for c, k in (("val_syn_clip90009", "car_drive"), ("val_syn_clip90010", "horn"), ("val_syn_clip90011", "horn"))})
        check("E2 暖機の境目（3.9・4.0・4.1 s）: 最接近 3.9 除外・4.0 と 4.1 残る／鳴り終わり 3.9 と 4.0 除外・4.1 残る（採点上の鳴り始め 4.0）・3.0〜9.0 は鳴り始め 4.0",
              ex == {"val_syn_clip90002": "warmup_cpa", "val_syn_clip90003": "", "val_syn_clip90004": "", "val_syn_clip90005": "warmup_end",
                     "val_syn_clip90009": "", "val_syn_clip90010": "warmup_end", "val_syn_clip90011": ""}
              and by[("val_syn_clip90004", "siren")]["score_onset_s"] == 4.0 and by[("val_syn_clip90008", "siren")]["score_onset_s"] == 5.0
              and by[("val_syn_clip90011", "horn")]["score_onset_s"] == 4.0, str(ex))
        e3 = by[("val_syn_clip90006", "car_drive")]
        cs = summ["by_class"]["car_drive"]
        check("E3 正解が全部消えた車も台帳に入り分母（all_lost 1・除外しない）・summary に除外前・理由・分母・全消失",
              e3["all_lost"] == 1 and e3["excluded"] == "" and e3["tier"] == "critical" and cs["n_physical"] == 5 and cs["excluded"] == {"warmup_cpa": 1}
              and cs["n_denominator"] == 4 and cs["n_all_lost"] == 1, str(cs))
        with open(td / "ledger.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L.COLS); w.writeheader(); w.writerows(rows)
        # 予測: サイレン 0〜10 s（90007）・5.5 s から（90008 は 95 枠から＝窓の外の版を別の予測で）・車は正解の距離 × 1.1
        pr = []
        for k in range(100):
            pr.append(("val_syn_clip90007", k, 0, 32.0, "nan"))
        for k in range(30, 90):
            pr.append(("val_syn_clip90004", k, 0, 32.0, "nan"))
        for k in range(55, 100):
            pr.append(("val_syn_clip90008", k, 0, 32.0, "nan"))
        for c in ("val_syn_clip90001", "val_syn_clip90003"):
            for ln in (td / "meta" / f"{c}.csv").read_text().splitlines():
                k, ci, trk, az, el, d = ln.split(",")
                pr.append((c, int(k), 4, float(az), round(float(d) * 1.1, 4)))
        write_pred(td / "pred.csv", pr)
        pr_late = [p for p in pr if not (p[0] == "val_syn_clip90008" and p[1] < 95)]
        write_pred(td / "pred_late.csv", pr_late)
        sc = ROOT / "scripts/_score_unified.py"
        r40, md40 = run_scorer(sc, td, "v30.md", td / "pred.csv", ("--events", str(td / "ledger.csv")))
        r0, md0 = run_scorer(sc, td, "v30_rs0.md", td / "pred.csv", ("--events", str(td / "ledger.csv"), "--rule-start", "0"))
        rl, mdl = run_scorer(sc, td, "v30_late.md", td / "pred_late.csv", ("--events", str(td / "ledger.csv")))
        ok_run = r40.returncode == 0 and r0.returncode == 0 and rl.returncode == 0
        siren = lambda md: [ln for ln in md.splitlines() if ln.startswith("| m | siren |")]
        check("E4 届いた行: 鳴り続けるサイレンは frame 40 からで届いた（90004・90007・90008 の 3/3）・frame 0 からは 90007 が窓の外で 2/3・90008 を 9.5 s からにすると 2/3",
              ok_run and siren(md40) and "| 3/3 |" in siren(md40)[0] and "| 2/3 |" in siren(md0)[0] and "| 2/3 |" in siren(mdl)[0],
              (siren(md40) + siren(md0) + siren(mdl) + [r40.stderr[-300:]]).__repr__()[:500])
        drow = [ln for ln in md40.splitlines() if ln.startswith("| m | ") and "%（1.39 m）" in ln]
        check("E5 距離の帯: 予測 = 正解 × 1.1 → 合わせ（≤3.2）の誤差 10.0%・定数の行（至近 1.39 m）・「2.0 秒以上前」は参考の節で主の表に列が無い",
              ok_run and drow and "**10.0%**" in drow[0] and "≥2.0s前" not in md40 and "## 参考（⑯" in md40 and "物理の事象" in md40,
              (drow[:1] + [md40[:200]]).__repr__()[:400])
        # E1 の続き: 旧の連続ラン（--events なし）では途切れのある 1 台が 2 事象 → 注意の分母が 1 多い
        ro, mdo = run_scorer(sc, td, "old.md", td / "pred.csv")
        def caut_den(md):
            for ln in md.splitlines():
                if ln.startswith("| m |") and "%" in ln:
                    return ln
            return ""
        check("E1b 旧の採点（--events なし）は途切れで 2 事象・--events は 1 事象（表の行が違う）", ro.returncode == 0 and caut_den(mdo) != caut_den(md40))
        ref = ROOT / "scripts" / f"_tmp_score_unified_ref_{os.getpid()}.py"
        try:
            ref.write_bytes(subprocess.run(["git", "show", f"{REF_COMMIT}:outdoor_seld_e2e/scripts/_score_unified.py"], cwd=str(ROOT), capture_output=True).stdout)
            rr, mdr = run_scorer(ref, td, "refdir/old.md", td / "pred.csv")   # 見出しに出力の名前が出るので同じ名前で別のフォルダに
            a_, b_ = mdr.splitlines(), mdo.splitlines()
            diff = [f"{i}: {x!r} / {y!r}" for i, (x, y) in enumerate(zip(a_, b_)) if x != y][:2] + ([f"行数 {len(a_)} / {len(b_)}"] if len(a_) != len(b_) else [])
            check(f"E6 --events なしの既定の出力が §2-5 の前（{REF_COMMIT}）と 1 バイトも同じ", rr.returncode == 0 and ro.returncode == 0 and mdr == mdo,
                  (rr.stderr[-300:] + " " + " | ".join(diff))[:600])
        finally:
            ref.unlink(missing_ok=True)
    # E7: 全消失の車（90006・至近・y = 1.2 m・最接近 6.0 s）への通知の帰属（scene の軌道の方位）
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        clips2 = build(td2)
        rows2, _ = L.build(td2, clips2)
        e1r = next(r for r in rows2 if r["clip"] == "val_syn_clip90001")
        azs = e1r["az_series"].split(";")
        gt1 = {int(ln.split(",")[0]): float(ln.split(",")[3]) for ln in (td2 / "meta" / "val_syn_clip90001.csv").read_text().splitlines()}
        close_az = all(abs(((float(azs[k]) - a + 180) % 360) - 180) <= 2.0 for k, a in gt1.items())   # 架空の正解は伝わりの遅れなしの近似（最接近で約 0.8°）＋整数の丸め 0.5°。本物の正解は同じ関数なので丸めの差だけ
        with open(td2 / "ledger.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L.COLS); w.writeheader(); w.writerows(rows2)
        e6 = next(r for r in rows2 if r["clip"] == "val_syn_clip90006")
        a6 = [float(x) if x else None for x in e6["az_series"].split(";")]
        def pred6(sign):
            out = []
            for k in range(40, 75):                            # 4.0〜7.5 s: 近づく 1 台（距離は軌道の水平距離）
                t = (k + 0.5) / 10.0; x = -30.0 + 5.0 * t
                d = math.hypot(x, 1.2)
                out.append(("val_syn_clip90006", k, 4, round(a6[k] if sign > 0 else ((a6[k] + 120.0 + 180.0) % 360.0 - 180.0), 1), round(d, 3)))   # 違う方向 = 一律 +120°（±180 の近くで符号反転だけだと差が小さい）
            return out
        write_pred(td2 / "p_ok.csv", pred6(+1)); write_pred(td2 / "p_none.csv", []); write_pred(td2 / "p_bad.csv", pred6(-1))
        res = {}
        for name in ("p_ok", "p_none", "p_bad"):
            r_, md_ = run_scorer(ROOT / "scripts/_score_unified.py", td2, f"{name}.md", td2 / f"{name}.csv", ("--events", str(td2 / "ledger.csv")))
            row = next((ln for ln in md_.splitlines() if ln.startswith("| m |") and "%" in ln), "")
            res[name] = row.split("|")[3].strip() if row else "?"          # 至近到達の列（分母は至近の事象 = 90006 だけ）
        check("E7 全消失の至近の車: 方向の合う発火 → 通知あり（至近到達 100%）・発火なし → 0%・方向の違う発火 → 0%・ラベルの枠の軌道の方位は架空の正解と 2° 以内",
              res == {"p_ok": "100.0%", "p_none": "0.0%", "p_bad": "0.0%"} and close_az, str(res))
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
