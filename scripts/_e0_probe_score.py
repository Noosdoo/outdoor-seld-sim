# -*- coding: utf-8 -*-
"""E0: 低周波依存度プローブの採点（素 vs HPF50/100/200）。

既存成果物は一切変更しない。出力は out/e0_lowfreq_probe/ のみ（新規）。
- クラス別 可聴(SNR>=0dB)検出率・方向誤差中央値: 本スクリプト内で自前計算
  （予測は7列 [clip,frame,class,track,az,el,dist] を正しくパース）
- 距離採点: scripts/_score_sde_dist.py を subprocess で条件別に実行
- 通知層v3.2: scripts/step12_notify_v3.py を subprocess で条件別に実行
- 最後に比較表＋事前登録ゲート判定を e0_report.md へ

事前登録ゲート(md/design/E可聴外帯域_計画_2026-08-04.md):
  HPF100で車クラス検出率 -1.0pt以上低下、または車の距離MAE +0.10m以上悪化
  → どちらか成立で「低周波寄与あり」
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "out" / "dataset_outdoor_siren_v11"
OUT = ROOT / "out" / "e0_lowfreq_probe"
OUT.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONDS = {
    "base": ROOT / "out" / "predictions_v11sde_run3" / "val_all.csv",
    "hpf50": ROOT / "out" / "predictions_v11sde_run3_hpf50" / "val_all.csv",
    "hpf100": ROOT / "out" / "predictions_v11sde_run3_hpf100" / "val_all.csv",
    "hpf200": ROOT / "out" / "predictions_v11sde_run3_hpf200" / "val_all.csv",
}
CLASSES_JA = ["サイレン", "クラクション", "バック音", "自転車ベル", "車", "踏切"]
CAR = 4


def ang_err(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(np.deg2rad, [az1, el1, az2, el2])
    c = (np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2))
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def load_pred(path: Path):
    """7列 [clip,frame,class,track,az,el,dist] → {clip: {frame:{class:(az,el)}}}"""
    pred = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 7:
            pred[p[0]][int(p[1])][int(p[2])] = (float(p[4]), float(p[5]))
    return pred


def class_table(pred):
    stats = {c: {"tp": 0, "n": 0, "errs": []} for c in range(6)}
    for clip in sorted(pred):
        gt = defaultdict(dict)
        for line in open(DS / "metadata" / f"{clip}.csv"):
            q = line.strip().split(",")
            if len(q) == 5:
                gt[int(q[0])][int(q[1])] = (float(q[3]), float(q[4]))
        mask = {}
        with open(DS / "masks" / f"{clip}.csv") as f:
            next(f)
            for line in f:
                q = line.strip().split(",")
                mask[(int(q[0]), int(q[1]))] = float(q[2])
        for k, evs in gt.items():
            pk = pred[clip].get(k, {})
            for c, (gaz, gel) in evs.items():
                if mask.get((k, c), 99.0) < 0.0:      # 可聴(SNR>=0dB)のみ
                    continue
                stats[c]["n"] += 1
                if c in pk:
                    stats[c]["tp"] += 1
                    stats[c]["errs"].append(ang_err(gaz, gel, *pk[c]))
    return stats


def main() -> None:
    # ① クラス別可聴検出率（4条件）
    recalls = {}
    for cond, pred_path in CONDS.items():
        assert pred_path.exists(), pred_path
        stats = class_table(load_pred(pred_path))
        recalls[cond] = stats
        print(f"[{cond}] done", flush=True)

    lines = ["# E0: 低周波依存度プローブ（SDE run3 / val 1,200本）", "",
             "素 vs ゼロ位相HPF(50/100/200Hz)。事前登録ゲートは設計書参照。", "",
             "## クラス別 可聴(SNR≥0dB)検出率", "",
             "| クラス | 素 | HPF50 | HPF100 | HPF200 |", "| --- | --- | --- | --- | --- |"]
    for c in range(6):
        row = [CLASSES_JA[c]]
        for cond in CONDS:
            s = recalls[cond][c]
            row.append(f"{s['tp']/s['n']:.1%}" if s["n"] else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## クラス別 方向誤差中央値[deg]", "",
              "| クラス | 素 | HPF50 | HPF100 | HPF200 |", "| --- | --- | --- | --- | --- |"]
    for c in range(6):
        row = [CLASSES_JA[c]]
        for cond in CONDS:
            e = recalls[cond][c]["errs"]
            row.append(f"{np.median(e):.1f}" if e else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    # ② 距離採点＋通知層（条件別サブフォルダ、既存スコアラをargvで）
    py = sys.executable
    for cond, pred_path in CONDS.items():
        sub = OUT / cond
        sub.mkdir(exist_ok=True)
        for script in ["_score_sde_dist.py", "step12_notify_v3.py"]:
            r = subprocess.run([py, str(ROOT / "scripts" / script),
                                str(pred_path), str(sub)],
                               capture_output=True, text=True)
            print(f"[{cond}] {script} rc={r.returncode}", flush=True)
            if r.returncode != 0:
                print(r.stderr[-800:], flush=True)

    # ③ ゲート判定（車クラス）
    rb = recalls["base"][CAR]; r100 = recalls["hpf100"][CAR]
    d_recall = (r100["tp"]/r100["n"] - rb["tp"]/rb["n"]) * 100.0
    lines += ["", "## 事前登録ゲート判定（車クラス・HPF100）", "",
              f"- 可聴検出率の変化: {d_recall:+.2f}pt（ゲート: -1.0pt以上の低下）",
              "- 車の距離MAE変化: 条件別 dist_score.md（base/ と hpf100/）を参照して判定",
              "", "※ 背景騒音は完全ピンク＝実街路より低域が軽い。本結果は",
              "「低周波有用性の上限値」として読む（実行前宣言どおり）。", ""]
    (OUT / "e0_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "e0_report.md", flush=True)


if __name__ == "__main__":
    main()
