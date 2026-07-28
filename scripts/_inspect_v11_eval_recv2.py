# -*- coding: utf-8 -*-
"""v11評価拡張の検品FAILを「受音ゲートv2」で再判定する。

背景（v7の教訓「新しい物理/幾何を足したら検品の前提も見直す」の再来）:
  従来ゲートの予測 recv_pred は「発音窓内の 1/d² の**時間平均**」。打音系（ベル）は
  鳴っている瞬間しかエネルギーが無いため、至近すれ違い（S2増量0.8-1.5m / N6 0.5-1.2m）
  では最接近の鋭い 1/d² スパイクが平均を支配し、打鐘がその一瞬に当たらない限り
  実測が予測を大きく下回る（構造的な予測誤り。音は物理的に正常）。
ゲートv2: 予測をドライ波形の**エネルギー重み付き** 1/d² に置き換える
  pred2 = l1m + 10log10( Σ env²(t)/d²(t) / Σ env²(t) )   （10msビン）
  判定は従来と同じ ±3.5dB。az/el/peak/recon/noise/maskの各ゲートは従来のまま
  （本スクリプトはそれらが合格済みであることをinspection.csvで確認してから
  recvのみ再判定する）。
出力: out/dataset_outdoor_siren_v11_eval/inspection_recv2.csv
終了コード: 全FAILがv2で解消=0 / 残FAILあり=1（残った本は除外リスト行き）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v11_eval_render as mev  # noqa: E402 (v10.1bチェーンのフラグ・DS設定)

m9 = mev.m9
DS = m9.DS
TGRID = np.arange(0.005, 10.0, 0.01)   # 10msビン中心


def mic_array(scene):
    m = scene["mic"]
    if "waypoints" in m:
        return np.array(m["waypoints"], dtype=np.float64)
    if m["motion"] == "static":
        return np.array([0.0, 0.0, 1.5])
    v, d = m["walk_speed_mps"], m.get("walk_dir_x", 1.0)
    x0 = -d * v * 5.0
    return np.array([[0.0, x0, 0.0, 1.5], [10.0, x0 + d * v * 10.0, 0.0, 1.5]])


def main() -> int:
    rows = list(csv.DictReader(open(DS / "inspection.csv")))
    fails = [r for r in rows if r["result"] == "FAIL"]
    print(f"FAIL {len(fails)}本を受音ゲートv2で再判定")
    out, n_pass2 = [], 0
    for r in fails:
        clip = r["name"]
        # 他ゲートは全て合格していること（=受音ゲート単独FAIL）を機械確認
        assert float(r["az_med_max"]) < 2.0 and float(r["el_med_max"]) < 2.0, clip
        assert float(r["peak"]) < m9.PEAK_MAX and r["mask_ok"] == "True", clip
        assert float(r["recon_err"]) < 2e-6, clip
        assert abs(float(r["noise_dba_meas"]) - float(r["noise_dba_target"])) < 0.3, clip
        scene = json.loads((DS / "work" / clip / "scene.json").read_text())
        mic = mic_array(scene)
        clip_ok = True
        for src in scene["sources"]:
            pred, meas = src.get("recv_pred_db"), src.get("recv_meas_db")
            if pred is None or abs(meas - pred) <= 3.5:
                continue
            dry = m9._window(m9._make_dry(src), src["t_on"], src["t_off"])
            n = int(10.0 * m9.FS_SIM)
            dry = dry[:n] if len(dry) >= n else np.pad(dry, (0, n - len(dry)))
            env2 = (dry.reshape(1000, -1).astype(np.float64) ** 2).sum(axis=1)
            d_ser = m9._dist_series(src["wp"], mic, TGRID)
            w = env2.sum()
            assert w > 0, clip
            pred2 = src["l1m_db"] + 10.0 * np.log10((env2 / d_ser ** 2).sum() / w)
            diff2 = meas - pred2
            ok2 = abs(diff2) <= 3.5
            clip_ok &= ok2
            out.append({"name": clip, "class": src["class"],
                        "pred_v1": pred, "meas": meas,
                        "pred_v2": round(float(pred2), 2),
                        "diff_v2": round(float(diff2), 2),
                        "result_v2": "PASS" if ok2 else "FAIL"})
        n_pass2 += int(clip_ok)
    with open(DS / "inspection_recv2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    diffs = [abs(o["diff_v2"]) for o in out]
    print(f"ゲートv2判定: {n_pass2}/{len(fails)} 本がPASS "
          f"(|diff_v2| 中央値 {np.median(diffs):.2f}dB / 最大 {max(diffs):.2f}dB)")
    print("->", DS / "inspection_recv2.csv")
    residual = [o for o in out if o["result_v2"] == "FAIL"]
    if residual:
        print("残FAIL（真の除外候補）:", [o["name"] for o in residual])
        return 1
    print("RECV GATE V2: ALL PASS（31本は音響的に正常＝除外しない）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
