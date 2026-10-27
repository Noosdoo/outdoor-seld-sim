# -*- coding: utf-8 -*-
"""本体 §3-2（入力の倍率の入口）・§9-4 の 1（推論のヒステリシスを frame 40 から）・§9-4 の 2（倍率の決め方）の受入れ試験（2026-10-26）。
PSELDNets の本体は使わず、`models.components.model_module.BaseModelModule` の形だけの代わりで gain_patch を確かめる。

  P1 既定（SDE_INPUT_GAIN_DB 未設定 = 0）: apply は何もしない（standardize・on_save_checkpoint は元の関数のまま）・meta_settings は辞書を変えない・ckpt に鍵を書かない
  P1b 明示の 0（SDE_INPUT_GAIN_DB=0）: standardize は包まない（計算は旧と bit 一致）・ckpt に sde_input_gain_db 0・meta_settings に input_gain_db 0（来歴の約束・Fable 16:2x）。
      空文字は未設定と同じ
  P2 倍率 20 dB: standardize が入力を 10 倍してから元の standardize を呼ぶ（1 回だけ・2 回 apply しても 1 回）・ckpt に sde_input_gain_db = 20・meta_settings に input_gain_db
  P3 相手 D と重ねない: SDE_INPUT_GAIN_DB と BASELINE_GAIN_DB が両方 0 以外なら apply は止まる・D の台本にも同じ止め
  P4 ckpt の照合: 鍵の無い ckpt と倍率 0 → 通る・ckpt 20 と倍率 20 → 通る・ckpt 20 と倍率 0／鍵なしと倍率 20 → 止まる（推論の check_ckpt と契約の check_input_gain）
  P5 5 本の道: 学習 2 本と推論 3 本が sde_patch のすぐ後で gain_patch.apply()・推論 3 本が ckpt を読んだすぐ後で check_ckpt・run_meta の sde に meta_settings
  H1 ヒステリシスを frame 40 から（hysteresis_patch.apply_with_start）: frame 39 に長さ 0.8・40〜99 に 0.4・入り 0.7・保持 0.3 → 旧（start 0）は 39〜99 の 61 枠が「いる」・
     新（start 40）は 0 枠（40 からは入りを超えない）。start 0 は旧の apply と同じ・入りだけ（保持 = 入り）でも start より前は落ちる
  H2 v17 の HYST_START_FRAME: 既定 0・明示なら 0 でも run_meta の settings に hyst_start_frame（未設定の run の identity は旧のまま）・gain_patch の sha も明示のときだけ
  P6 v2final_train.sbatch: 倍率を明示すれば 0 でも gain_patch を写して sha を追記（本体の sha の tee の後ろ・tee -a）・run に input_gain.json
  G1 倍率の決め方（v30_gain_decide.decide）: 1 pt 以内が 1 つ（最良がそのまま g1）／2 つ以上（小さい方が g1・g2 = g1 を除く最良）／全候補同点（最小・g2 なし）／
     seed 2 で逆転（2 seed の平均で g2 が 1 pt 超良い → g2・ちょうど 1 pt は g1）
  G3 止める所（§9-3 の受け入れ）: seed 1 の候補の欠損（--candidates とそろわない）・F の非有限・seed 2 に g1／g2 の片方が無い → 止まる。
     seed 1 で g1 と g2 が同点（全候補同点ではない）→ seed 2 の通常の規則
  G2 F の計算: フレーム 40〜99 だけ（39 以前の誤検出は数えない）・正解の無いクラスは平均から外す・命令で decision.json に規則・候補数・未丸めの F・分岐・最終
使い方: PSELDNet/PSELDNets/.venv の python（torch が要る）で python scripts/test_gain_patch_v30.py
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_sde"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def fake_models():
    """models.components.model_module.BaseModelModule の形だけ（standardize は入力をそのまま返す・on_save_checkpoint は何もしない）。"""
    class BaseModelModule:
        def standardize(self, batch_x):
            return batch_x

        def on_save_checkpoint(self, checkpoint):
            return None
    m = types.ModuleType("models.components.model_module"); m.BaseModelModule = BaseModelModule
    sys.modules["models"] = types.ModuleType("models"); sys.modules["models.components"] = types.ModuleType("models.components")
    sys.modules["models.components.model_module"] = m
    return BaseModelModule


def load_gain(env: dict):
    for k in ("SDE_INPUT_GAIN_DB", "BASELINE_GAIN_DB"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("gain_patch", None)
    return importlib.import_module("gain_patch")


def main() -> int:
    # P1
    B = fake_models(); orig_std, orig_save = B.standardize, B.on_save_checkpoint
    gp = load_gain({})
    r = gp.apply(); ck = {}; B().on_save_checkpoint(ck)
    d = {"distance_mode": "log"}
    check("P1 既定（倍率 0）: apply は何もしない・standardize と on_save_checkpoint は元のまま・meta_settings は辞書を変えない・ckpt に鍵なし",
          r is False and B.standardize is orig_std and B.on_save_checkpoint is orig_save and gp.meta_settings(d) == d and "sde_input_gain_db" not in ck)
    # P1b
    B = fake_models(); orig_std = B.standardize
    gp = load_gain({"SDE_INPUT_GAIN_DB": "0"})
    r = gp.apply(); ck = {}; B().on_save_checkpoint(ck)
    gpe = load_gain({"SDE_INPUT_GAIN_DB": ""})
    check("P1b 明示の 0: standardize は包まない・ckpt に sde_input_gain_db 0・meta_settings に input_gain_db 0・空文字は未設定と同じ",
          r is False and B.standardize is orig_std and ck.get("sde_input_gain_db", "なし") == 0.0 and gp.meta_settings(d) == dict(d, input_gain_db=0.0)
          and gp.EXPLICIT and not gpe.EXPLICIT and gpe.meta_settings(d) == d, str(ck))
    # P2
    B = fake_models()
    gp = load_gain({"SDE_INPUT_GAIN_DB": "20"})
    r1 = gp.apply(); r2 = gp.apply()
    y = B().standardize(np.ones(4))
    ck = {}; B().on_save_checkpoint(ck)
    check("P2 倍率 20 dB: standardize の入力が 10 倍（2 回 apply しても 1 回）・ckpt に sde_input_gain_db 20・meta_settings に input_gain_db 20",
          r1 and r2 and np.allclose(y, 10.0) and ck.get("sde_input_gain_db") == 20.0 and gp.meta_settings({"a": 1}) == {"a": 1, "input_gain_db": 20.0}, str(y))
    # P3
    B = fake_models()
    gp = load_gain({"SDE_INPUT_GAIN_DB": "20", "BASELINE_GAIN_DB": "40"})
    try:
        gp.apply(); stopped = False
    except RuntimeError:
        stopped = True
    dsrc = (ROOT / "server_sde/_baseline_D_pretrain170_infer.py").read_text(encoding="utf-8")
    check("P3 主の倍率と相手 D の倍率を同じプロセスで重ねない（apply が止まる・D の台本にも止め）", stopped and "SDE_INPUT_GAIN_DB" in dsrc and "二重に掛けない" in dsrc)
    # P4
    gp0 = load_gain({})
    gp20 = load_gain({"SDE_INPUT_GAIN_DB": "20"})
    def stops(f, *a):
        try:
            f(*a); return False
        except Exception:
            return True
    ok4 = gp0.check_ckpt({}) == 0.0 and gp20.check_ckpt({"sde_input_gain_db": 20.0}) == 20.0 and stops(gp0.check_ckpt, {"sde_input_gain_db": 20.0}) and stops(gp20.check_ckpt, {})
    try:
        import torch
        import _conf_exec_contract as CC
        with tempfile.TemporaryDirectory() as td:
            p20, p0 = Path(td) / "g20.ckpt", Path(td) / "g0.ckpt"
            torch.save({"sde_input_gain_db": 20.0, "state_dict": {}}, p20); torch.save({"state_dict": {}}, p0)
            ok4c = CC.check_input_gain(p20, 20) == 20.0 and CC.check_input_gain(p0, 0) == 0.0 and stops(CC.check_input_gain, p20, 0) and stops(CC.check_input_gain, p0, 20)
        det = ""
    except ImportError as e:
        ok4c, det = False, f"torch が無い: {e}"
    check("P4 ckpt の照合（推論の check_ckpt・契約の check_input_gain）: 鍵なし＋0・20＋20 は通る・20＋0・鍵なし＋20 は止まる", ok4 and ok4c, det)
    # P5
    srcs = {n: (ROOT / "server_sde" / n).read_text(encoding="utf-8") for n in ("_train_sde17.py", "_train_causal17.py", "_causal_infer_v17.py", "_causal_infer_long18.py", "_causal_infer_scores.py")}
    order_ok = all(s.index("sde_patch.apply_train_patches()") < s.index("gain_patch.apply()") < s.index("sde_patch.apply_train_patches()") + 900 for s in srcs.values())
    infer_ok = all("gain_patch.check_ckpt(_ck)" in srcs[n] and "gain_patch.meta_settings(" in srcs[n] for n in ("_causal_infer_v17.py", "_causal_infer_long18.py", "_causal_infer_scores.py"))
    check("P5 5 本の道: 学習 2 本と推論 3 本が sde_patch の直後に gain_patch.apply()・推論 3 本が ckpt の照合と run_meta の sde に meta_settings", order_ok and infer_ok)
    # H1
    import hysteresis_patch as HP
    C, T = 9, 100
    doa = np.zeros((3, T, 4 * C), dtype=np.float32)
    doa[0, 39, 0] = 0.8                     # クラス 0・方位 0°（x 成分）
    doa[0, 40:, 0] = 0.4
    sed_in = np.sqrt(doa[:, :, 0:C] ** 2 + doa[:, :, C:2 * C] ** 2 + doa[:, :, 2 * C:3 * C] ** 2) > 0.7
    old = HP.apply_with_start(doa, sed_in, C, 0.7, 0.3, 20.0, start=0)
    new = HP.apply_with_start(doa, sed_in, C, 0.7, 0.3, 20.0, start=40)
    same_old = np.array_equal(old, HP.apply(doa, C, 0.7, 0.3, 20.0))
    one = HP.apply_with_start(doa, sed_in, C, 0.7, 0.7, 20.0, start=40)
    check("H1 ヒステリシスを frame 40 から: 旧 61 枠（39〜99）・新 0 枠・start 0 は旧の apply と同じ・入りだけでも start より前は落ちる",
          int(old[:, :, 0].any(axis=0).sum()) == 61 and int(new.sum()) == 0 and same_old and int(one.sum()) == 0 and int(sed_in[:, :40].sum()) == 1,
          f"旧 {int(old[:, :, 0].any(axis=0).sum())} 枠・新 {int(new.sum())} 枠")
    # H2
    v17 = srcs["_causal_infer_v17.py"]
    check("H2 v17: HYST_START_FRAME は既定 0・明示なら 0 でも run_meta の settings に hyst_start_frame・後処理は 0 以外のとき apply_with_start・gain_patch の sha は明示のとき",
          'os.environ.get("HYST_START_FRAME", "0")' in v17 and 'if HYST_START > 0:' in v17 and "if HYST_START_EXPLICIT:" in v17 and 'META["settings"]["hyst_start_frame"] = HYST_START' in v17
          and 'os.environ.get("HYST_START_FRAME", "").strip() != ""' in v17 and "apply_with_start(doa_np, sed_np" in v17 and "if gain_patch.EXPLICIT:" in v17)
    # P6
    sb = (ROOT / "server_sde/v2final_train.sbatch").read_text(encoding="utf-8").replace("\r\n", "\n")
    i_main = sb.index("| tee $E2E/v2final_train_sha256_job"); i_gain = sb.index("sha256sum $PS/gain_patch.py | tee -a")
    check("P6 v2final_train.sbatch: 明示なら 0 でも gain_patch を写して sha を本体の sha の後ろに追記・run に input_gain.json",
          i_main < i_gain and sb.count('if [ -n "${SDE_INPUT_GAIN_DB:-}" ]; then') == 2 and '!= "0"' not in sb[sb.index("V2FINAL_EXP_SUFFIX"):])
    # G1
    import v30_gain_decide as GD
    a = GD.decide({0: 0.50, 20: 0.60, 30: 0.55, 40: 0.58}, None)
    b = GD.decide({0: 0.50, 20: 0.595, 30: 0.60, 40: 0.592}, None)
    c = GD.decide({0: 0.5, 20: 0.5, 30: 0.5, 40: 0.5}, None)
    d1 = GD.decide({0: 0.50, 20: 0.595, 30: 0.60, 40: 0.592}, {20: 0.58, 30: 0.60})        # 平均 g1=0.5875・g2=0.60 → 差 1.25 pt > 1 → g2
    d2 = GD.decide({0: 0.50, 20: 0.595, 30: 0.60, 40: 0.592}, {20: 0.600, 30: 0.615})      # 平均 0.5975 と 0.6075 → 差 1.00 pt（ちょうど）→ g1
    check("G1 倍率の決め方: 1 pt 以内 1 つ → g1 = 最良・2 つ以上 → 小さい方が g1・g2 = g1 を除く最良・全候補同点 → 最小で g2 なし・seed 2 で 1 pt 超 → g2・ちょうど 1 pt → g1",
          (a["g1"], a["g2"]) == (20, 40) and a["final"] is None and (b["g1"], b["g2"]) == (20, 30) and (c["g1"], c["g2"], c["final"]) == (0, None, 0)
          and d1["final"] == 30 and d2["final"] == 20, str([a, b, c, d1["final"], d2["final"]])[:400])
    # G2
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "gt").mkdir()
        clips = ["val_syn_clip00012", "val_syn_clip00024"]
        (td / "gt" / f"{clips[0]}.csv").write_text("".join(f"{k},4,0,30,0,2.00\n" for k in range(50, 60)), encoding="utf-8")
        (td / "gt" / f"{clips[1]}.csv").write_text("", encoding="utf-8")
        with open(td / "man.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["clip_id"]); w.writerows([[c] for c in clips])
        def pred(name, rows):
            (td / name).write_text("".join(f"{c},{k},{ci},0,30,0,2.0\n" for c, k, ci in rows), encoding="utf-8")
        pred("g0.csv", [(clips[0], k, 4) for k in range(50, 55)] + [(clips[0], k, 4) for k in range(0, 39)])          # 39 以前の誤検出は数えない → F = 2·5/(10+5) = 0.667
        pred("g20.csv", [(clips[0], k, 4) for k in range(50, 60)] + [(clips[1], 70, 2)])                               # 正解の無いクラス 2 の誤検出は平均から外れる・F(4) = 1.0
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts/v30_gain_decide.py"), "--gt", str(td / "gt"), "--manifest", str(td / "man.csv"),
                            "--seed1", f"0={td / 'g0.csv'}", f"20={td / 'g20.csv'}", "--candidates", "0,20", "--out", str(td / "decision.json")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        r_miss = subprocess.run([sys.executable, str(ROOT / "scripts/v30_gain_decide.py"), "--gt", str(td / "gt"), "--manifest", str(td / "man.csv"),
                                 "--seed1", f"0={td / 'g0.csv'}", f"20={td / 'g20.csv'}", "--out", str(td / "decision_miss.json")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        dj = json.loads((td / "decision.json").read_text(encoding="utf-8")) if r.returncode == 0 else {}
        f0, f20 = dj.get("seed1", {}).get("0", {}).get("mean_f"), dj.get("seed1", {}).get("20", {}).get("mean_f")
        check("G2 F はフレーム 40〜99 だけ（39 以前の誤検出は数えない）・正解の無いクラスは外す・decision.json に規則・候補数・未丸めの F・分岐・最終（seed 2 待ち）",
              r.returncode == 0 and abs(f0 - 2 * 5 / 15) < 1e-9 and abs(f20 - 1.0) < 1e-9 and dj["n_candidates"] == 2 and "rule" in dj
              and dj["decision"]["g1"] == 20 and dj["decision"]["g2"] == 0 and dj["decision"]["final"] is None and dj["seed1"]["20"]["n_fire_frames_nogt"] == 1,
              (r.stderr[-300:] if r.returncode else str(dj.get("decision"))))
    # G3
    def stops(f, *a):
        try:
            f(*a); return False
        except Exception:
            return True
    t = GD.decide({0: 0.50, 20: 0.60, 30: 0.60, 40: 0.40}, {20: 0.58, 30: 0.63})          # seed 1 は g1 = 20 と g2 = 30 が同点 → seed 2 の平均 0.59 と 0.615 → 2.5 pt → 30
    check("G3 止める: 候補の欠損（--candidates とそろわない）・F の非有限（seed 1／seed 2）・seed 2 に片方が無い。seed 1 で g1 と g2 が同点 → seed 2 の通常の規則",
          r_miss.returncode != 0 and "候補の欠損" in (r_miss.stderr + r_miss.stdout) and not (td / "decision_miss.json").exists()
          and stops(GD.decide, {0: float("nan"), 20: 0.5}, None) and stops(GD.decide, {0: 0.5, 20: 0.6}, {0: 0.5, 20: float("inf")})
          and stops(GD.decide, {0: 0.5, 20: 0.6, 30: 0.55}, {20: 0.6}) and (t["g1"], t["g2"], t["final"]) == (20, 30, 30), str(t))
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
