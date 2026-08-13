# -*- coding: utf-8 -*-
"""【非推奨・使用禁止 2026-08-13】本スクリプトは第11回監査で欠陥3件が確定
（①先頭100フレーム固定走査で長尺負例の誤警告を過小評価 ②event_id無視で
1発火が複数イベントに二重計上 ③旧ルールv1使用）。
正= step20_realsmoke_score_v2.py（全区間走査・イベント窓・v3.4・対応あり統計）。
本ファイルは履歴参照用にのみ残す。

Step 20: 実録スモークの採点 — 観測者注釈CSVと予測CSVから当事者指標を出す。

scene.json（合成の正解）が無い実録のための軽量採点。通知判定は本番と同じ
step12のルールv1（fire_events）を使う。

入力:
  --pred  予測CSV（Colab出力の *_all.csv 形式: stem,frame,class,az,el）
  --ann   注釈CSV（記録紙から起こす。列: clip_id,trial,class,quadrant,t_start,t_cpa
          class=siren/horn/backup_beep/bike_bell/car_drive/crossing
          quadrant=F/R/B/L（前/右/後/左）、t_*=クリップ内秒（±1s精度）
          誤報カウント用の「イベント無し区間」はclass=none行で表す）
  --audio 変換済みflacのフォルダ（通知ルールの受聴レベル計算に使用）

指標: 試行ごとの 通知有無 / リードタイム(t_cpa−通知時刻、±1s粒度) / 方向象限一致。
      class=none 区間は誤通知カウント。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.calibration import frame_spl_a  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "step12", ROOT / "scripts" / "step12_notify_v9.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

CLS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
           "car_drive": 4, "crossing": 5}


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def quadrant_of(az_deg: float) -> str:
    """方位→4象限。規約: az=0前・+90左（反時計回り）。F=|az|≤45, L, B=|az|>135, R。"""
    a = (az_deg + 180.0) % 360.0 - 180.0
    if abs(a) <= 45:
        return "F"
    if abs(a) > 135:
        return "B"
    return "L" if a > 0 else "R"


def load_pred(path: Path):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def main() -> int:
    pred = load_pred(Path(_arg("--pred")))
    ann_path = Path(_arg("--ann"))
    audio_dir = Path(_arg("--audio"))
    out_md = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "score_summary.md")))

    rows = list(csv.DictReader(open(ann_path, encoding="utf-8-sig")))
    fires_cache = {}

    def fires_of(clip):
        if clip not in fires_cache:
            f = audio_dir / f"{clip}.flac"
            assert f.exists(), f"音声なし: {f}"
            mix = np.asarray(sf.read(f)[0], np.float64).T
            fires_cache[clip] = m12.fire_events(pred.get(clip, {}),
                                                frame_spl_a(mix[0], 24000))
        return fires_cache[clip]

    results = []
    n_false = 0
    exposure_s = 0.0
    for r in rows:
        clip, cls = r["clip_id"], r["class"].strip()
        fires = fires_of(clip)
        if cls == "none":
            t0f, t1f = float(r["t_start"]), float(r["t_cpa"])
            exposure_s += (t1f - t0f)
            n_false += sum(1 for k, c, _ in fires
                           if t0f <= m12.emit_time(k) <= t1f)
            continue
        ci = CLS_IDX[cls]
        t_cpa = float(r["t_cpa"])
        hits = [(m12.emit_time(k), az) for k, c, az in fires if c == ci]
        notified = bool(hits)
        lead = round(t_cpa - hits[0][0], 2) if hits else None
        quad_ok = (quadrant_of(hits[0][1]) == r["quadrant"].strip()) if hits else None
        results.append({"trial": r["trial"], "class": cls, "notified": notified,
                        "lead": lead, "quad_ok": quad_ok})

    n = len(results)
    n_notif = sum(1 for x in results if x["notified"])
    leads = [x["lead"] for x in results if x["lead"] is not None]
    quads = [x["quad_ok"] for x in results if x["quad_ok"] is not None]
    rep = [f"# 実録スモーク採点（pred={Path(_arg('--pred')).name}）", "",
           f"- 試行 {n}件: 通知 {n_notif}/{n}",
           (f"- リードタイム中央値 {np.median(leads):.1f}s（範囲 {min(leads):.1f}"
            f"〜{max(leads):.1f}s、注釈±1s精度）" if leads else "- リードタイム: n/a"),
           (f"- 方向4象限一致 {sum(quads)}/{len(quads)}" if quads else "- 象限: n/a"),
           f"- 誤通知（class=none区間）: {n_false}件 / {exposure_s/3600:.2f}h", "",
           "| 試行 | クラス | 通知 | リード[s] | 象限一致 |", "| --- | --- | --- | --- | --- |"]
    for x in results:
        rep.append(f"| {x['trial']} | {x['class']} | {'○' if x['notified'] else '×'} | "
                   f"{x['lead'] if x['lead'] is not None else '—'} | "
                   f"{'○' if x['quad_ok'] else ('×' if x['quad_ok'] is not None else '—')} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:8]))
    print("->", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
