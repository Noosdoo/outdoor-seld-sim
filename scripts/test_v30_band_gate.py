# -*- coding: utf-8 -*-
"""帯域ゲート（決め直し 4・本人 10/26 19:32 で v30 の既定 = 入・線 0 dB・仕様 §9-5／§9-6）の受け入れ試験。学習の本（train）の試し描きと架空の信号だけを使う。

  B1 判定量の理論値: 1 kHz の正弦（振幅 0.01）と白色の背景（σ 0.001）で、一番有利な帯 = 1,000 Hz・SN 比の中央値（100 枠）が理論値
     10·log10(A²N ÷ (4·n_bin·σ²))（N = 2,400・n_bin = 1,000 Hz の帯に入る bin の数）から ±0.5 dB・全枠 valid
  B9 端の場合（§9-6 の 1・Astra R4-30-01）: ゼロ／ゼロ → 全枠 valid = False・−999・0 Hz・通らない／stem だけゼロ → valid・大きな負・通らない／
     背景だけゼロ → valid・大きな正・通る／NaN・Inf の入力 → BandGateError で止まる／0 dB ちょうど（stem = 背景）→ 0.0 で線 0 を通る／
     同じクラスの 2 音源（自分 1 kHz・相手 1 kHz を背景に）→ 20·log10(A ÷ B) ±0.2 dB
  B10 Astra の反例を塞ぐ: 分析の道具の帯域関数を NaN を返す形・BandGateError を出す形に差し替えると、0% と出さずに止まる（StopError）
  B2 切: FEATURES["band_gate"] = False で描いた 2 本（車だけの本・電車＋警報機の本）の foa・metadata・masks が、帯域ゲートを足す前（74c7718c6）の
     出力と 1 バイトも同じ（v30 の広帯域 −5 の道を守る）
  B3 入（T 0）: masks 9 列（＋ snr_band_max_db・band_hz・band_valid）・最初の 6 列は切と同じ・scene の gate_rule "v30_band"・線 0・広帯域 −5・
     来歴 gate_band（band_gate.py の内容 hash・線・旗・枠長・床）・音源ごとに gate_band_max_db／gate_band_hz_at_max／gate_band_n_invalid・
     保存した stem と mix から計算し直した帯の値と band_valid が合う（差 最大 0.15 dB）・車だけの本の正解の枠は「隙間埋め 2 をした 帯の通過」の中（丸めの境界 ±0.005 dB は含める）
  B4 入（T +3）: 正解の枠は T 0 の部分集合・gate_line_db 3
  B5 小標本の道具が 9 列の masks を読み、帯 0／+3 の率・帯の照合（停止線 A = −30 dB 以上で 99% 点 0.05・最大 0.25 dB／B = 線 0 ±3 dB で 99% 点 0.05・最大 0.10 dB の内側・Fable 19:5x／20:0x）・境界枠数・来歴（band_gate の hash）を出す
  B6 生成の完了判定（_run_v19_gen.clip_complete）が 切（6 列）も 入（v30_band・9 列）も完了と読む
  B7 v30 の既定が入（決め直し 4・§9-6 の 2）: 環境変数 V30_GATE_BAND_DB の無いプロセスで import すると FEATURES["band_gate"] = True・GATE_BAND_DB = 0.0・
     旗に触らずに描いた本の scene が v30_band・線 0・masks 9 列
  B8 丸めの扱い（§9-6 の 3・Astra R4-30-03）: 一覧の 4 本（S1・S3・S4・S5 の先頭・S3 は同じクラスの 2 音源の本）を既定で描き、
     (a) 未丸めの通過率（保存した stem と mix から band_gate で計算し直した値・§8 の数え方）と (b) 丸めた masks で数えた通過率を別々に出し、
     食い違う枠は計算し直しの差の範囲（|値| ≤ 0.15 dB）だけ・±0.005 dB の境界枠数を併記・(c) 後処理後の最終ラベルの率（n_label_frames ÷ 鳴っている枠）も別に出す。
     同じクラスの 2 音源は masks の src で別々に記録され、それぞれが相手を背景にした計算し直しと合う。完全一致は求めない
使い方: DynamicSound の venv で python scripts/test_v30_band_gate.py（10 本ほど描くので 4 分ほど）
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REF_COMMIT = "74c7718c6"
REF = {"train_syn_clip11266": {"foa": "b18fb3d56776bdd9c7ed563a0a226be574ca53530bfdb6ef71b1c662f06920ac",
                               "metadata": "cd8d865bd76adf29f2d2141f265e806da323ef9eee7474ae9a5a55a84c609c85",
                               "masks": "acc54e07eef8dd1960004c96ab86ff74e2e2685dec11a83c40e93ca11f670155"},
       "train_syn_clip68112": {"foa": "bdb0e87184a5071b89e38ec6c36a0f4dd5478b70f7a4676b44398af971678e0a",
                               "metadata": "9772d3be278d3ab8e9e68159048992c459573981de17f7a9fb3a2a07c223b3a4",
                               "masks": "a8da04ac11958fd168dfe4d64decc6fae7d7c571a8c385e0a6b00780fe048291"}}
CAR_ONLY, TRAIN_BELL = "train_syn_clip11266", "train_syn_clip68112"
HEAD9 = "frame,class,track,src,snr_a_db,snr_a_bg_db,snr_band_max_db,band_hz,band_valid"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read9(p: Path) -> dict:
    """src → frame → (帯の値, valid)。"""
    m = defaultdict(dict)
    for l in p.read_text(encoding="utf-8").splitlines()[1:]:
        q = l.split(",")
        m[int(q[3])][int(q[0])] = (float(q[6]), int(q[8]))
    return m


def main() -> int:
    from outdoor_seld import band_gate as BG
    fs, N = 24000, 2400
    t = np.arange(fs * 10) / fs
    rng = np.random.default_rng(0)
    # B1
    A, sig = 0.01, 0.001
    best, hz, valid = BG.band_snr_frames(A * np.sin(2 * np.pi * 1000.0 * t), rng.normal(0, sig, len(t)), fs)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    lo, hi = BG.band_edges()
    k1000 = int(np.argmin(np.abs(BG.FC - 1000.0)))
    nbin = int(((f >= lo[k1000]) & (f < hi[k1000])).sum())
    theory = 10 * np.log10(A * A * N / (4 * nbin * sig * sig))
    check("B1 1 kHz の正弦＋白色: 一番有利な帯 = 1,000 Hz・SN 比の中央値が理論値 ±0.5 dB・全枠 valid",
          np.all(hz == BG.FC[k1000]) and abs(float(np.median(best)) - theory) <= 0.5 and len(best) == 100 and bool(valid.all()),
          f"中央値 {np.median(best):.2f}・理論 {theory:.2f} dB・bin {nbin}")
    # B9
    z = np.zeros(len(t)); sine = A * np.sin(2 * np.pi * 1000.0 * t); noise = rng.normal(0, sig, len(t))
    b_zz, h_zz, v_zz = BG.band_snr_frames(z, z, fs)
    b_s0, _, v_s0 = BG.band_snr_frames(z, noise, fs)
    b_b0, _, v_b0 = BG.band_snr_frames(sine, z, fs)

    def raises(fn):
        try:
            fn(); return False
        except BG.BandGateError:
            return True
    nan_x = sine.copy(); nan_x[100] = np.nan
    inf_x = noise.copy(); inf_x[5] = np.inf
    b_eq, _, v_eq = BG.band_snr_frames(noise, noise, fs)
    B_ = 0.004
    b_2, h_2, v_2 = BG.band_snr_frames(sine, B_ * np.sin(2 * np.pi * 1000.0 * t) + rng.normal(0, 1e-6, len(t)), fs)
    r9 = {"zero_zero": (not v_zz.any()) and np.all(b_zz == BG.INVALID_DB) and np.all(h_zz == 0.0) and not BG.passes(b_zz, v_zz, 0.0).any(),
          "stem_zero": bool(v_s0.all()) and float(b_s0.max()) < -100 and not BG.passes(b_s0, v_s0, 0.0).any(),
          "bg_zero": bool(v_b0.all()) and float(b_b0.min()) > 100 and bool(BG.passes(b_b0, v_b0, 0.0).all()),
          "nan_stops": raises(lambda: BG.band_snr_frames(nan_x, noise, fs)), "inf_stops": raises(lambda: BG.band_snr_frames(sine, inf_x, fs)),
          "exact_0db": bool(v_eq.all()) and np.all(b_eq == 0.0) and bool(BG.passes(b_eq, v_eq, 0.0).all()),
          "same_class_two": bool(v_2.all()) and abs(float(np.median(b_2)) - 20 * np.log10(A / B_)) <= 0.2 and np.all(h_2 == BG.FC[k1000])}
    check("B9 端の場合: ゼロ／ゼロ（無効・通らない）・stem だけゼロ・背景だけゼロ・NaN と Inf で止まる・0 dB ちょうどは通る・同じクラスの 2 音源",
          all(r9.values()), str(r9))
    import _run_v2final_gen as g
    import step11_v30_render as v30
    import soundfile as sf
    import step11_v9_render as m9
    plan = {r["clip_id"]: r for r in g.load_plan()}
    tmp = Path(tempfile.mkdtemp(prefix="band_gate_"))
    prev = (v30.FEATURES["band_gate"], v30.GATE_BAND_DB)
    bg_prev = v30.FEATURES["bg_shape"]
    v30.FEATURES["bg_shape"] = False      # 10/27 から既定は背景の形を入 → 74c7718c6 の参照と小標本の道具（描き方 off）にそろえて切を明示（B7・B8 は既定に戻して描く）
    try:
        def render(name, band, T, clips):
            ds = tmp / name
            v30.FEATURES["band_gate"], v30.GATE_BAND_DB = band, T
            g.point_to(ds)
            for c in clips:
                v30.generate_clip(dict(plan[c]))
            return ds
        off = render("off", False, 0.0, [CAR_ONLY, TRAIN_BELL])
        got = {c: {d: sha(off / d / f"{c}.{'flac' if d == 'foa' else 'csv'}") for d in ("foa", "metadata", "masks")} for c in REF}
        check(f"B2 切: 2 本の foa・metadata・masks が帯域ゲートを足す前（{REF_COMMIT}）と 1 バイトも同じ", got == REF, str({c: [k for k in v if v[k] != REF[c][k]] for c, v in got.items()}))
        on0 = render("on0", True, 0.0, [CAR_ONLY, TRAIN_BELL])
        ok3, det3 = True, []
        for c in (CAR_ONLY, TRAIN_BELL):
            lines_on = (on0 / "masks" / f"{c}.csv").read_text(encoding="utf-8").splitlines()
            lines_off = (off / "masks" / f"{c}.csv").read_text(encoding="utf-8").splitlines()
            head_ok = lines_on[0] == HEAD9
            six_ok = [",".join(l.split(",")[:6]) for l in lines_on[1:]] == lines_off[1:]
            sc = json.loads((on0 / "work" / c / "scene.json").read_text(encoding="utf-8"))
            gb = sc.get("gate_band") or {}
            rule_ok = ((sc.get("gate_rule"), sc.get("gate_line_db"), sc.get("gate_broadband_line_db")) == ("v30_band", 0.0, -5.0)
                       and gb.get("code_sha256") == BG.code_sha256() and gb.get("line_db") == 0.0 and gb.get("frame_sec") == 0.1 and gb.get("floor") == 1e-30
                       and all(k in s for s in sc["sources"] for k in ("gate_band_max_db", "gate_band_hz_at_max", "gate_band_n_invalid")))
            mix = sf.read(str(on0 / "foa" / f"{c}.flac"), dtype="float64", always_2d=True)[0][:, 0]
            mx = 0.0; vmis = 0
            m = read9(on0 / "masks" / f"{c}.csv")
            for i, s in enumerate(sc["sources"]):
                st = sf.read(str(on0 / "work" / c / f"src{i}_{s['class']}_withrefl_24k.flac"), dtype="float64", always_2d=True)[0][:, 0]
                n = min(len(mix), len(st))
                bb, _, bv = BG.band_snr_frames(st[:n], mix[:n] - st[:n], fs)
                for k, (v, vv) in m[i].items():
                    vmis += int(bool(vv) != bool(bv[k]))
                    if bv[k] and v >= -30.0:
                        mx = max(mx, abs(bb[k] - v))
            ok3 &= head_ok and six_ok and rule_ok and mx <= 0.15 and vmis == 0
            det3.append(f"{c}: 見出し {head_ok}・6 列 {six_ok}・規約と来歴 {rule_ok}・帯の差 最大 {mx:.4f}・valid の食い違い {vmis}")
        mc = read9(on0 / "masks" / f"{CAR_ONLY}.csv")[0]
        kk = sorted(mc)
        pass_r = np.array([(mc[k][1] == 1 and mc[k][0] >= 0.0) or abs(mc[k][0]) <= 0.005 for k in kk])   # 丸めの境界 ±0.005 dB は含める
        aud = m9._fill_gaps(pass_r, m9.GAP_FILL)
        lab = [int(l.split(",")[0]) for l in (on0 / "metadata" / f"{CAR_ONLY}.csv").read_text(encoding="utf-8").splitlines() if l.strip()]
        sub_ok = bool(lab) and all(aud[k] for k in lab)
        check("B3 入（T 0）: 9 列・最初の 6 列は切と同じ・規約 v30_band／線 0／広帯域 −5・来歴 gate_band（hash・線・枠長・床）・帯の値と valid が計算し直しと合う・車だけの本の正解 ⊆ 隙間埋めした通過",
              ok3 and sub_ok, "／".join(det3) + f"／正解 {len(lab)} 枠")
        on3 = render("on3", True, 3.0, [CAR_ONLY])
        lab3 = [int(l.split(",")[0]) for l in (on3 / "metadata" / f"{CAR_ONLY}.csv").read_text(encoding="utf-8").splitlines() if l.strip()]
        sc3 = json.loads((on3 / "work" / CAR_ONLY / "scene.json").read_text(encoding="utf-8"))
        check("B4 入（T +3）: 正解の枠は T 0 の部分集合・gate_line_db 3",
              set(lab3) <= set(lab) and len(lab3) <= len(lab) and sc3.get("gate_line_db") == 3.0, f"T0 {len(lab)} 枠・T3 {len(lab3)} 枠")
        import _run_v19_gen as RG
        cc = {}
        for name, ds in (("off", off), ("on0", on0)):
            g.point_to(ds)
            cc[name] = all(RG.clip_complete(c) for c in (CAR_ONLY, TRAIN_BELL))
        check("B6 生成の完了判定（clip_complete）: 切（6 列）も入（v30_band・9 列）も完了と読む", cc == {"off": True, "on0": True}, str(cc))
        import v30_m5_small_sample as M5
        try:
            r5 = M5.analyse_arm(on0, [CAR_ONLY, TRAIN_BELL], "off"); err5 = None
        except M5.StopError as e:
            r5, err5 = {}, str(e)
        cls = r5.get("classes", {}); bd = r5.get("band_diff_vs_gate", {})
        check("B5 小標本の道具: 9 列の masks を読み、帯 0／+3 の率・帯の照合（停止線 A〔−30 dB 以上: 99% 0.05・最大 0.25〕と B〔線 0 ±3 dB: 99% 0.05・最大 0.10〕の内側）・境界枠数・来歴（band_gate の hash）を出す",
              err5 is None and all("keep_band_0" in v and "keep_band_3" in v and v["keep_band_3"] <= v["keep_band_0"] for v in cls.values())
              and bd.get("max", 9) <= M5.BAND_DIFF_MAX and bd.get("p99", 9) <= M5.BAND_DIFF_P99_MAX and "boundary_frames_pm0.005" in bd and r5.get("band_diff_near_line", {}).get("max", 9) <= M5.BAND_NEAR_MAX
              and r5.get("band_gate", {}).get("code_sha256") == BG.code_sha256() and r5["gate_rule"][0] == "v30_band",
              (err5 or str({k: (round(v["keep_band_0"], 3), round(v["keep_band_3"], 3)) for k, v in cls.items()})) + f"・帯の差 99% {bd.get('p99')}・最大 {bd.get('max')}・境界 {bd.get('boundary_frames_pm0.005')}")
        # B10: Astra の反例（帯域関数が NaN を返しても止まらず 0% と出る）を塞ぐ
        orig = BG.band_snr_frames
        r10 = {}
        try:
            BG.band_snr_frames = lambda s, b, fs_, *a: (np.full(len(s) // 2400, np.nan), np.zeros(len(s) // 2400), np.ones(len(s) // 2400, bool))
            try:
                M5.analyse_arm(on0, [CAR_ONLY], "off"); r10["nan_return"] = False
            except M5.StopError:
                r10["nan_return"] = True

            def _raise(*a, **k):
                raise BG.BandGateError("試験の非有限")
            BG.band_snr_frames = _raise
            try:
                M5.analyse_arm(on0, [CAR_ONLY], "off"); r10["raises"] = False
            except M5.StopError:
                r10["raises"] = True
        finally:
            BG.band_snr_frames = orig
        check("B10 Astra の反例を塞ぐ: 帯域関数が NaN を返す／BandGateError を出す → 分析の道具は 0% と出さずに止まる", all(r10.values()) and len(r10) == 2, str(r10))
        # B7
        import os as _os
        import subprocess as _sp
        env = {k: v for k, v in _os.environ.items() if k != "V30_GATE_BAND_DB"}
        env["PYTHONPATH"] = _os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "src")])
        r7 = _sp.run([sys.executable, "-c", "import step11_v30_render as v;print(v.FEATURES['band_gate'], v.GATE_BAND_DB)"], capture_output=True, text=True, env=env, cwd=str(ROOT))
        v30.FEATURES["band_gate"], v30.GATE_BAND_DB = prev                       # このプロセスの既定に戻してから、旗に触らずに描く
        v30.FEATURES["bg_shape"] = bg_prev
        ds7 = tmp / "default"
        g.point_to(ds7)
        v30.generate_clip(dict(plan[CAR_ONLY]))
        sc7 = json.loads((ds7 / "work" / CAR_ONLY / "scene.json").read_text(encoding="utf-8"))
        h7 = (ds7 / "masks" / f"{CAR_ONLY}.csv").read_text(encoding="utf-8").splitlines()[0]
        check("B7 v30 の既定が入: import した既定 = band_gate True・線 0.0・旗に触らずに描いた本は v30_band・線 0・masks 9 列",
              r7.stdout.split() == ["True", "0.0"] and prev == (True, 0.0) and (sc7.get("gate_rule"), sc7.get("gate_line_db")) == ("v30_band", 0.0) and h7 == HEAD9,
              f"{r7.stdout.strip()} {r7.stderr[-200:]}")
        # B8
        import csv as _csv
        L = list(_csv.DictReader(open(ROOT / "md/design/v30_m5_小標本_本の一覧_2026-10-26.csv", encoding="utf-8")))
        pick = [next(r["clip_id"] for r in L if r["stratum"] == st) for st in ("S1", "S3", "S4", "S5")]
        ds8 = tmp / "m5default"
        g.point_to(ds8)
        for c in pick:
            v30.generate_clip(dict(plan[c]))
        n_fr = n_un = n_ro = n_flip = n_flip_bad = 0; n_bd = 0; n_lab = 0; same_class_ok = []
        for c in pick:
            sc = json.loads((ds8 / "work" / c / "scene.json").read_text(encoding="utf-8"))
            mix = sf.read(str(ds8 / "foa" / f"{c}.flac"), dtype="float64", always_2d=True)[0][:, 0]
            m = read9(ds8 / "masks" / f"{c}.csv")
            seen = defaultdict(list)
            for i, s in enumerate(sc["sources"]):
                if s.get("no_label") or (s.get("params") or {}).get("train_car"):
                    continue
                st = sf.read(str(ds8 / "work" / c / f"src{i}_{s['class']}_withrefl_24k.flac"), dtype="float64", always_2d=True)[0][:, 0]
                n = min(len(mix), len(st))
                bb, _, bv = BG.band_snr_frames(st[:n], mix[:n] - st[:n], fs)
                k0, k1 = max(int(float(s["t_on"]) * 10), 0), min(max(int(float(s["t_off"]) * 10), 1), 100)
                ks = [k for k in range(k0, k1) if k in m[i]]
                un = BG.passes(bb[ks], bv[ks], 0.0)                                       # (a) 未丸め
                ro = np.array([m[i][k][1] == 1 and m[i][k][0] >= 0.0 for k in ks])        # (b) 丸めた masks
                fl = [k for k, a_, b_ in zip(ks, un, ro) if a_ != b_]
                n_fr += len(ks); n_un += int(un.sum()); n_ro += int(ro.sum()); n_flip += len(fl)
                n_flip_bad += sum(1 for k in fl if abs(m[i][k][0]) > 0.15)
                n_bd += sum(1 for k in ks if abs(m[i][k][0]) <= 0.005)
                n_lab += int(s.get("n_label_frames") or 0)                                 # (c) 後処理後の最終ラベル
                dd = [abs(bb[k] - m[i][k][0]) for k in ks if bv[k] and m[i][k][0] >= -30.0]
                seen[s["class"]].append((i, float(max(dd)) if dd else 0.0))
            for cl, lst in seen.items():
                if len(lst) >= 2:
                    same_class_ok.append(len({i for i, _ in lst}) == len(lst) and all(d <= 0.15 for _, d in lst))
        check("B8 丸めの扱い: 未丸めの通過率と丸めた masks の通過率を別々に出し、食い違いは計算し直しの差の範囲（|値| ≤ 0.15 dB）だけ・±0.005 dB の境界枠数と最終ラベルの率を併記・同じクラスの 2 音源は src で別々に合う",
              n_fr > 0 and n_flip_bad == 0 and same_class_ok and all(same_class_ok),
              f"枠 {n_fr}・未丸め {100 * n_un / n_fr:.1f}%・丸め {100 * n_ro / n_fr:.1f}%・食い違い {n_flip}（0.15 dB 超 {n_flip_bad}）・境界 ±0.005 {n_bd}・最終ラベル {100 * n_lab / n_fr:.1f}%・同クラス 2 音源 {same_class_ok}")
    finally:
        v30.FEATURES["band_gate"], v30.GATE_BAND_DB = prev
        v30.FEATURES["bg_shape"] = bg_prev
        shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
