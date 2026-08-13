# -*- coding: utf-8 -*-
"""実録経路の単体テスト（第11回監査 再合格条件1-3の検証）。

T1: step19が96kHz入力を受理し24kHz/長さ1/4で出力する
T2: 長尺負例の全区間走査（50秒地点の発火を検出。旧実装の10秒固定では0件になる）
T3: event_id単位のイベント窓（1発火は最大1イベントにのみ割当）
T4: v3.4距離トリガの方位連結±60°（40°差=発火 / 70°差=非発火）
T5: 統計（McNemar厳密・Poisson区間・paired bootstrap）の数値検算
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v2 = _load("s20v2", "step20_realsmoke_score_v2.py")
s19 = _load("s19rc", "step19_realsmoke_convert.py")

fails = []


def check(name, cond, note=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        fails.append(name)


# ---------------- T1: 96kHz変換 ----------------
with tempfile.TemporaryDirectory() as td:
    import soundfile as sf
    td = Path(td)
    fs = 96000
    t = np.arange(2 * fs) / fs
    w = 0.05 * np.sin(2 * np.pi * 1000 * t)
    wav = np.stack([w, 0.3 * w, 0.1 * w, 0.2 * w], axis=1)
    src = td / "take96k.wav"
    sf.write(src, wav, fs)
    out = s19.convert(src, td / "conv", laeq=50.0, win=(0.0, 1.0),
                      pitch=0, roll=0, yaw=0)
    y, sr_out = sf.read(out)
    check("T1 96kHz受理", sr_out == 24000 and abs(len(y) - 2 * 24000) <= 4,
          f"(fs={sr_out}, n={len(y)})")

# ---------------- T2: 長尺負例の全区間走査 ----------------
pred = {"c1": {500: [(4, 10.0, 1.0)], 501: [(4, 12.0, 1.0)]}}
rows = [{"clip_id": "c1", "event_id": "1", "trial": "n1", "class": "none",
         "quadrant": "", "t_start": "0", "t_cpa": "120"}]
ev, neg = v2.evaluate(rows, pred, link_deg=60.0, has_dist=True)
check("T2 長尺負例で50s地点の発火を検出",
      neg["n_false"] == 1 and abs(neg["exposure_s"] - 120.0) < 1e-9,
      f"(n_false={neg['n_false']}, exp={neg['exposure_s']}s; 旧実装なら0件)")

# ---------------- T3: event_id窓と貪欲割当 ----------------
pred3 = {"c2": {49: [(4, 0.0, 1.0)], 50: [(4, 2.0, 1.0)]}}
rows3 = [
    {"clip_id": "c2", "event_id": "1", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "3", "t_cpa": "8"},
    {"clip_id": "c2", "event_id": "2", "trial": "t1", "class": "car_drive",
     "quadrant": "F", "t_start": "20", "t_cpa": "25"},
]
ev3, _ = v2.evaluate(rows3, pred3, link_deg=60.0, has_dist=True)
got = {e["event_id"]: e["notified"] for e in ev3}
check("T3 1発火は1イベントのみ通知成功",
      got.get("1") is True and got.get("2") is False, f"({got})")

# ---------------- T4: ±60°連結 ----------------
h_in = v2.dist_triggers_var({10: [(10.0, 1.4)], 11: [(50.0, 1.4)]}, 1.5, 20, 60.0)
h_out = v2.dist_triggers_var({10: [(10.0, 1.4)], 11: [(80.0, 1.4)]}, 1.5, 20, 60.0)
check("T4 方位連結（40°=発火/70°=非発火）",
      len(h_in) == 1 and len(h_out) == 0, f"(in={h_in}, out={h_out})")

# ---------------- T5: 統計 ----------------
p_mc = v2.mcnemar_exact(0, 8)
lo0, hi0 = v2.poisson_rate_ci(0, 100 / 60.0, alpha=0.10)  # 片側95%上限に相当
bs = v2.paired_bootstrap_median_diff([1, 2, 3, 4], [0, 1, 2, 3], n_boot=2000, seed=0)
check("T5a McNemar厳密 p(0,8)=2^-7",
      abs(p_mc - 2 * (0.5 ** 8)) < 1e-9, f"(p={p_mc:.6f})")
check("T5b Poisson片側95%上限（0件/100分）≈1.80回/h",
      abs(hi0 - 1.797) < 0.02, f"(hi={hi0:.3f})")
check("T5c paired bootstrap（定差1s）CI=[1,1]",
      bs is not None and abs(bs[0] - 1.0) < 1e-9 and abs(bs[1] - 1.0) < 1e-9
      and abs(bs[2] - 1.0) < 1e-9, f"({bs})")

print()
if fails:
    print(f"NG: {len(fails)}件 {fails}")
    sys.exit(1)
print("ALL PASS (7 checks)")
sys.exit(0)
