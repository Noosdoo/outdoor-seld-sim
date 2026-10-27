# -*- coding: utf-8 -*-
"""通知の「方位帰属」の許容（方位差・時間差）の感度掃引（2026-09-21）。

なぜ: 第 11 回監査／Astra 監査の軽微「帰属の 30° と ±0.5 s は掃引していない（20° や 45° にしたら
結論が動くか調べていない）」への対応。**結論が動かないことを示すのが目的**で、値を選び直すためではない。

やること: `scripts/_notify_v43_attrib.py` の AZ_TOL（方位差）と T_TOL（フレーム差）を
環境変数（NOTIFY_ATTRIB_AZ_TOL / NOTIFY_ATTRIB_T_TOL）で振り、`scripts/_score_unified.py` を
そのつど走らせて、主要指標がどう動くかを集める。**既定（30.0 / 5）では従来と同一出力**。

使い方:
  python scripts/_attrib_sweep.py <outdir> --plan <manifest.csv> --meta <GTdir> --pred <val_all_causal.csv>
        [--az 10,15,20,25,30,35,40,45,60] [--t 1,2,3,5,7,10] [--label v18_e149]
出力: <outdir>/sweep.csv（1 行 = 1 格子点）・<outdir>/sweep.md（表）・<outdir>/pt_az<..>_t<..>.md（各点の採点）
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
FPS = 10.0  # T_TOL[frame] -> 秒


def _cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_md(md: Path, label: str):
    """採点 md から 丸め後の主表 1 行と、丸め前の整数を拾う。"""
    out = {}
    txt = md.read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in txt:
        if not ln.startswith("|"):
            continue
        c = _cells(ln)
        if not c or c[0] != label:
            continue
        if len(c) >= 15 and c[1].count("(") == 1:          # 主表
            out.update({"至近到達": c[2], "強到達": c[3].replace("*", ""),
                        "注意到達": c[4], "安全抑制": c[5], "リード中央": c[6],
                        "発火数": c[8].replace(",", ""),
                        "検出率": c[9], "距離誤差": c[13 + 1].split()[0].replace("*", "")})
        elif len(c) >= 7 and "=" in c[1]:                   # 丸め前
            m = re.match(r"(\d+)/(\d+)", c[1]); out["n_strong"], out["n_crit"] = (m.group(1), m.group(2)) if m else ("", "")
            m = re.match(r"(\d+)/(\d+)", c[2]); out["n_hit"] = m.group(1) if m else ""
            m = re.match(r"(\d+)/(\d+)", c[4]); out["n_det"], out["n_gt"] = (m.group(1), m.group(2)) if m else ("", "")
            out["pairs"] = c[6].replace(",", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", default="v18_e149")
    ap.add_argument("--az", default="10,15,20,25,30,35,40,45,60")
    ap.add_argument("--t", default="1,2,3,5,7,10")
    a = ap.parse_args()

    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    azs = [float(x) for x in a.az.split(",")]
    ts = [int(x) for x in a.t.split(",")]
    rows = []
    t0 = time.time()
    for az in azs:
        for t in ts:
            pt = od / f"pt_az{az:g}_t{t}.md"
            env = dict(os.environ, NOTIFY_ATTRIB_AZ_TOL=f"{az}", NOTIFY_ATTRIB_T_TOL=f"{t}", NOTIFY_ATTRIB_SWEEP="1")   # SWEEP=1 は掃引の明示（混入防止のガード・R03）
            cmd = [PY, "scripts/_score_unified.py", str(pt), "--strict", "--raw",
                   "--plan", a.plan, "--meta", a.meta, f"{a.label}={a.pred}"]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[FAIL] az={az} t={t}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
                sys.exit(1)
            d = parse_md(pt, a.label)
            d.update({"az_tol_deg": az, "t_tol_frame": t, "t_tol_s": round(t / FPS, 2),
                      "既定": "*" if (az == 30.0 and t == 5) else ""})
            rows.append(d)
            print(f"az={az:>4g} t=±{t/FPS:.1f}s  強到達 {d.get('強到達','?'):>6}  "
                  f"安全抑制 {d.get('安全抑制','?'):>6}  発火 {d.get('発火数','?'):>5}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    cols = ["az_tol_deg", "t_tol_frame", "t_tol_s", "既定", "至近到達", "強到達", "注意到達",
            "安全抑制", "リード中央", "発火数", "検出率", "距離誤差",
            "n_strong", "n_crit", "n_hit", "n_det", "n_gt", "pairs"]
    with (od / "sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    L = ["# 方位帰属の許容（方位差 × 時間差）の感度掃引 — " + a.label, "",
         f"予測= {a.pred}", f"plan= {a.plan} / GT= {a.meta}", "",
         "**目的**: 監査の軽微「帰属の 30° と ±0.5 s を掃引していない」への対応。"
         "**値を選び直すためではなく、結論が動かないことを確かめるため**。既定（30°・±0.5 s）の行に `*`。", "",
         "| 方位差 | 時間差 | | 至近到達 | 強到達 | 注意到達 | 安全抑制 | リード中央 | 発火数 | 検出率 | 距離誤差 | 強 n/N |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in rows:
        L.append("| {az_tol_deg:g}° | ±{t_tol_s}s | {既定} | {至近到達} | {強到達} | {注意到達} | {安全抑制} | "
                 "{リード中央} | {発火数} | {検出率} | {距離誤差} | {n_strong}/{n_crit} |".format(**d))
    L += ["", "**読み方**: 検出率・距離誤差は 1 対 1 の距離対応（別の許容 20°）で決まるので、"
          "帰属の許容を変えても**動かないはず**。動いていたら実装の取り違え。"]
    (od / "sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n-> {od/'sweep.md'} ({len(rows)} 点 / {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
