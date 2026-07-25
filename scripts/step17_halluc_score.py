# -*- coding: utf-8 -*-
"""Step 17: 幻覚検定 n=50（車なし×サイレンのクリップで car_drive の誤検出を数える）。

v9.2時代はアドホック集計だったものを正式スクリプト化（2026-07-22、Fable）。
対象 = fold2_room9（交差点サイレン20本、scenario予測）+ fold2_room3（幻覚評価30本、
halluc予測）。全50本とも車なし・サイレンありをscene.jsonで検証してから数える。

指標（v9.2最終報 md/results/v9_2_results_2026-07-19.md 1節と同定義）:
  1. 幻覚ありクリップ数（車predフレームが1つでもある）
  2. 車predフレーム総数（/クリップ）
  3. 車の誤通知発火（step12ルールv1通過、当事者に届く誤振動）
  4. サイレン通知 n/50（本来の検出が生きているか）

使い方: python scripts/step17_halluc_score.py \
    [--ds out/dataset_outdoor_siren_v10] [--pred out/predictions_v10_2] \
    [--out out/step12_notify_v10_2] [--title v10.2 run1]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import soundfile as sf  # noqa: E402

from outdoor_seld.calibration import frame_spl_a  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "step12", ROOT / "scripts" / "step12_notify_v9.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

CAR, SIREN = 4, 0


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


DS = ROOT / _arg("--ds", "out/dataset_outdoor_siren_v10")
# fold2_room3（幻覚評価30本）はv10.2追補の独立フォルダ側に生成されている
DS_ADD = ROOT / _arg("--ds-add", "out/dataset_outdoor_siren_v10_2_add")
PRED = ROOT / _arg("--pred", "out/predictions_v10_2")
OUT = ROOT / _arg("--out", "out/step12_notify_v10_2")
TITLE = _arg("--title", "v10.2 run1")


def _ds_of(clip: str) -> Path:
    """クリップの実体がある方のデータセットルートを返す。"""
    for ds in (DS, DS_ADD):
        if (ds / "work" / clip / "scene.json").exists():
            return ds
    raise FileNotFoundError(f"{clip} のscene.jsonがどちらのDSにもない")


def load_pred_multi(path):
    out = defaultdict(lambda: defaultdict(list))
    if not Path(path).exists():
        return out
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def main():
    pred = load_pred_multi(PRED / "scenario_all.csv")
    for k, v in load_pred_multi(PRED / "halluc_all.csv").items():
        pred[k] = v
    clips = ([f"fold2_room9_mix{i+1:03d}" for i in range(20)]
             + [f"fold2_room3_mix{i+1:03d}" for i in range(30)])

    n_hal_clip = car_frames = 0
    car_fires_total = 0
    siren_ok = 0
    frames_per_clip = []
    for clip in clips:
        ds = _ds_of(clip)
        scene = json.loads((ds / "work" / clip / "scene.json").read_text())
        classes = [s["class"] for s in scene["sources"]]
        assert "car_drive" not in classes, f"{clip}: 車が存在する（前提違反）"
        assert "siren" in classes, f"{clip}: サイレンがない（前提違反）"
        p = pred.get(clip, {})
        nf = sum(1 for evs in p.values() for e in evs if e[0] == CAR)
        frames_per_clip.append(nf)
        car_frames += nf
        n_hal_clip += int(nf > 0)
        mix = np.asarray(sf.read(ds / "foa" / f"{clip}.flac")[0], np.float64).T
        fires = m12.fire_events(p, frame_spl_a(mix[0], 24000))
        car_fires_total += sum(1 for _, c, _ in fires if c == CAR)
        siren_ok += int(any(c == SIREN for _, c, _ in fires))

    n = len(clips)
    rep = [f"# 幻覚検定 n={n}（車なし×サイレン。{TITLE}）", "",
           "参考（v9.2最終報）: 幻覚あり7/50・車フレーム73(1.5/本)・誤通知0/50・サイレン50/50", "",
           f"- 幻覚ありクリップ（車predフレーム>0）: **{n_hal_clip}/{n} "
           f"({n_hal_clip/n:.0%})**",
           f"- 車predフレーム総数: **{car_frames}（{car_frames/n:.1f}/クリップ、"
           f"最大{max(frames_per_clip)}）**",
           f"- 車の誤通知発火（ルールv1通過）: **{car_fires_total}件/{n}本**",
           f"- サイレン通知: **{siren_ok}/{n}**", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "halluc_summary.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
