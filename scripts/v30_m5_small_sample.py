# -*- coding: utf-8 -*-
"""−5 dB の小標本の検査（本体 §9-1 の 8・§9-4 の 6・2026-10-26）。手順と止める条件は md/design/v30_m5_小標本_計画_2026-10-26.md に固定（実行の前にコミット）。
一覧（md/design/v30_m5_小標本_本の一覧_2026-10-26.csv・v30_m5_sample_list.py）の 120 本を、v30 の入口で 2 通りに試し描きする:
  off = 背景の形なし（今の FEATURES・本人の試聴の前）／on = 背景の形あり（FEATURES["bg_shape"] = True・試聴の返事の前に比べるため）
数えるもの（音源ごと・正解のある音源の「鳴っている枠」= [t_on, t_off) の 0.1 s の枠）:
  - 新しい背景（暗騒音＋雨＋同じ場面の自分以外の全 stem・電車の車両も）の広帯域 A 特性 SN 比（masks の snr_a_db）で区切った枠の割合（< −10・−10〜−5・−5〜0・0〜5・≥ 5）
  - 残存率 = 新しい定義で −5 以上の枠の割合（クラス別）。分けて見るため: 旧の量（暗騒音＋雨・snr_a_bg_db）で 0 以上（v29 の線）・新しい定義で 0 以上・旧の量で −5 以上
  - 一番有利な帯: −10〜−5 と −5〜0 の枠で、1/3 オクターブ帯（125 Hz〜10 kHz・20 帯）の SN 比の最大（背景 = 自分以外の全部／暗騒音＋雨だけ）の中央値・10%・90%・よく勝つ帯
  - ゲート本体の関数との照合: 保存した stem と mix から calibration と同じ frame_spl_a で SN 比を計算し直し、masks の snr_a_db との差（masks ≥ −30 dB の枠）の分位点と最大の clip
止める条件（どれか 1 つで止める・JSON に error）: 定義の食い違い（照合の差の 99% 点 > 0.10 dB か最大 > 1.0 dB）・非有限（鳴っている枠の masks か計算し直した値）・
  版の混在（scene の gate_rule／gate_line_db／gate_bg が v30・−5.0・noise+others でない・背景の形の有無が描き方と違う・masks の見出しが 6 列でない・描いたコードの sha が途中で変わった）
使い方: python scripts/v30_m5_small_sample.py --list <一覧 csv> --work <描く置き場> --out <json> [--arms off,on] [--analyse-only] [--clips a,b（試走だけ・一覧の外）]
決め直し 4（本人 10/26 19:32）のあと: 描くときのゲートは v30 の既定（帯域ゲート 0 dB・masks 8 列・gate_rule v30_band）。off／on は背景の形だけの切り替え。
  10/26 の本番の 240 本は帯域ゲートを足す前の描画（広帯域 −5・6 列）で、帯の数は下の形で描き直さずに数えた（結果 md §8）。
§9-6 の 1・3（Astra R4-30-01・03）: 帯域の判定量・入力のパワー・保存した snr_band_max_db の非有限は止める（band_gate.BandGateError → 止める）。全帯が無効な枠は通らない（valid）。
  帯の照合（8／9 列の masks があるとき）の停止線（Fable 10/26 19:5x・**6 本の実測〔99% 点 0.036・最大 0.174・線 ±3 で 0.025／0.040〕を見てから置いた**・
  最初の案 99% 点 0.02／最大 0.15 は 24 bit で保存した音からの計算し直しの量子化で正当な描画でも超えた）: 両方を止める条件にする
    A = 帯の値 ≥ −30 dB の枠で 99% 点 0.05 dB・最大 0.25 dB（量子化を超える食い違い）
    B = 線の近く（帯の値 0 ±3 dB）の枠で 99% 点 0.05 dB・最大 0.10 dB（判定に効く範囲は最大で厳しく見る・試験の 2 本〔207 枠・99% 点 0.0355・最大 0.040〕で
        99% 点 0.03 を超えたので Fable 20:0x に案 1 = 99% 点は A と同じ 0.05）
  描画の中の判定は丸めていない値で行う（masks は小数 2 桁・照合の差は計算し直しの側の量子化と丸め）。丸めた masks の値が線 ±0.005 dB にある境界枠の数を併記（band_boundary）。
  CODE に src/outdoor_seld/band_gate.py。結果の JSON の band_gate に内容 hash・帯と床の約束・数えた線（0・+3）・枠長。
帯域ゲートの列（Fable 10/26 18:3x の下ごしらえ・描き直さずに数える）: 保存した stem と mix から outdoor_seld.band_gate.band_snr_frames（描画の帯域ゲートと同じ関数）で
  一番有利な 1/3 オクターブ帯の SN 比を計算し、帯域ゲート T_band = 0／+3 dB の残存率（keep_band_0・keep_band_3）を並べる。masks が 8 列（帯域ゲートで描いた本）なら
  snr_band_max_db と計算し直した値の差も出す（band_diff_vs_gate・止める条件には入れない）。scene の gate_rule は v30（線 −5）か v30_band（線 = 帯の線）のどちらかに揃っていること。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINE = -5.0
BINS = [(-999.0, -10.0), (-10.0, -5.0), (-5.0, 0.0), (0.0, 5.0), (5.0, 999.0)]
NEAR = [(-10.0, -5.0), (-5.0, 0.0)]
DIFF_P99_MAX, DIFF_MAX = 0.10, 1.0
BAND_DIFF_P99_MAX, BAND_DIFF_MAX = 0.05, 0.25       # §9-6 の 1・A: 帯の値 ≥ −30 dB の枠の照合の停止線（Fable 19:5x・6 本の実測を見てから置いた）
BAND_NEAR_DB, BAND_NEAR_P99_MAX, BAND_NEAR_MAX = 3.0, 0.05, 0.10   # B: 線の近く（帯の値 0 ±3 dB）の枠の停止線（Fable 20:0x 案 1・枠が少ないと 99% 点 ≈ 最大なので最大 0.10 で厳しく見る）
BAND_LINES = (0.0, 3.0)                              # 数える帯の線（0 = 決め直し 4 の本線・+3 = 分析の併記だけ）
BOUNDARY_DB = 0.005                                  # §9-6 の 3: 丸めた masks で数えるときの境界（±0.005 dB）
CMP_FLOOR = -30.0
TRAIN_CAR_REF = "train_car（参考・対象外）"
FS, SPF = 24000, 2400
FREQS = np.fft.rfftfreq(SPF, 1.0 / FS)
FC = 1000.0 * 2.0 ** (np.arange(-9, 11) / 3.0)
BAND_MASKS = [(FREQS >= fc / 2 ** (1 / 6)) & (FREQS < fc * 2 ** (1 / 6)) for fc in FC]
CODE = ["scripts/step11_v30_render.py", "scripts/step11_v13_render.py", "scripts/step11_v12_render.py", "scripts/step11_v9_render.py",
        "src/outdoor_seld/noise_v30.py", "src/outdoor_seld/calibration.py", "src/outdoor_seld/labels.py", "src/outdoor_seld/geometry.py",
        "src/outdoor_seld/band_gate.py"]                # §9-6 の 1: 帯域ゲートの関数も版の対象


class StopError(Exception):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_ids() -> dict:
    return {c: _sha(ROOT / c) for c in CODE}


def render(rows: list[dict], work: Path, arm: str) -> None:
    import _run_v2final_gen as g
    import step11_v30_render as v30
    ds = work / arm
    ds.mkdir(parents=True, exist_ok=True)
    cj = ds / "CODE.json"
    ids = code_ids()
    if cj.is_file() and json.loads(cj.read_text(encoding="utf-8")) != ids:
        raise StopError(f"版の混在: {arm} の置き場を描いたコードの sha と今のコードが違う（{cj}）")
    cj.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
    prev = v30.FEATURES["bg_shape"]
    v30.FEATURES["bg_shape"] = (arm == "on")
    try:
        g.point_to(ds)
        for r in rows:
            if (ds / "work" / r["clip_id"] / "scene.json").is_file():
                continue
            v30.generate_clip(dict(r))
    finally:
        v30.FEATURES["bg_shape"] = prev
    if code_ids() != ids:
        raise StopError(f"版の混在: {arm} を描いている間にコードが変わった")


def band_db(x):
    n = len(x) // SPF
    P = np.abs(np.fft.rfft(x[:n * SPF].reshape(n, SPF), axis=1)) ** 2
    return 10.0 * np.log10(np.maximum(np.stack([P[:, m].sum(axis=1) for m in BAND_MASKS], axis=1), 1e-30))


MASK_HEAD6 = "frame,class,track,src,snr_a_db,snr_a_bg_db"
MASK_HEAD8 = MASK_HEAD6 + ",snr_band_max_db,band_hz"
MASK_HEAD9 = MASK_HEAD8 + ",band_valid"


def read_masks(p: Path):
    """src → frame → (新の広帯域 SN 比, 旧の量, 帯の SN 比 or None, valid or None)。6 列（v30）・8 列（下ごしらえ）・9 列（v30_band・§9-6）。
    保存した値の非有限は止める（§9-6 の 1）。"""
    with open(p, newline="", encoding="utf-8") as f:
        head = f.readline().strip()
        if head not in (MASK_HEAD6, MASK_HEAD8, MASK_HEAD9):
            raise StopError(f"版の混在: masks の見出しが 6／8／9 列のどれでもない: {p.name} {head!r}")
        m = defaultdict(dict)
        for line in f:
            q = line.strip().split(",")
            vals = [float(x) for x in q[4:]]
            if not all(np.isfinite(vals)):
                raise StopError(f"非有限: {p.name} の masks の行 {line.strip()!r}")
            m[int(q[3])][int(q[0])] = (vals[0], vals[1], (vals[2] if len(q) >= 8 else None), (int(q[8]) if len(q) >= 9 else None))
    return m


def analyse_arm(ds: Path, clips: list[str], arm: str) -> dict:
    import soundfile as sf
    from outdoor_seld import calibration as cal
    from outdoor_seld import band_gate
    acc = defaultdict(lambda: defaultdict(list))
    diffs, worst = [], (0.0, None)
    bdiffs, bworst = [], (0.0, None)
    bnear, bnworst = [], (0.0, None)
    n_boundary = {f"{t:g}": 0 for t in BAND_LINES}; n_round_flip = 0; n_invalid = 0
    per_src = []
    rule0 = None
    for clip in clips:
        wdir = ds / "work" / clip
        scene = json.loads((wdir / "scene.json").read_text(encoding="utf-8"))
        rule = (scene.get("gate_rule"), scene.get("gate_line_db"), scene.get("gate_bg"))
        if not (rule == ("v30", LINE, "noise+others") or (rule[0] == "v30_band" and rule[2] == "noise+others" and scene.get("gate_broadband_line_db") == LINE)):
            raise StopError(f"版の混在: {clip} のゲートの規約 {rule}")
        rule0 = rule0 or rule
        if rule != rule0:
            raise StopError(f"版の混在: {clip} のゲートの規約 {rule} が先の本 {rule0} と違う")
        if ("shape" in (scene.get("noise") or {})) != (arm == "on"):
            raise StopError(f"版の混在: {clip} の背景の形の有無が描き方（{arm}）と違う")
        masks = read_masks(ds / "masks" / f"{clip}.csv")
        mix = sf.read(str(ds / "foa" / f"{clip}.flac"), dtype="float64", always_2d=True)[0][:, 0]
        srcs = scene["sources"]
        stems = [sf.read(str(wdir / f"src{i}_{s['class']}_withrefl_24k.flac"), dtype="float64", always_2d=True)[0][:, 0] for i, s in enumerate(srcs)]
        n = min([len(mix)] + [len(x) for x in stems])
        mix = mix[:n]; stems = [x[:n] for x in stems]
        bg_noise = mix - np.sum(stems, axis=0) if stems else mix
        B_noise = band_db(bg_noise)
        for i, s in enumerate(srcs):
            others = mix - stems[i]
            re = np.asarray(cal.frame_spl_a(stems[i], FS)) - np.asarray(cal.frame_spl_a(others, FS))
            try:                                                              # 帯域ゲートの判定量（描画と同じ関数・描き直さずに数える・非有限は止める）
                bb, _bhz, bvalid = band_gate.band_snr_frames(stems[i], others, FS)
            except band_gate.BandGateError as e:
                raise StopError(f"非有限: {clip} src{i} の帯域の判定量（{e}）")
            if not (np.all(np.isfinite(bb)) and len(bb) == len(bvalid)):
                raise StopError(f"非有限: {clip} src{i} の帯域の判定量（関数の返り値）")
            mm = masks.get(i, {})
            for k, (v, vb, vband, vvalid) in mm.items():
                if k < len(re) and v >= CMP_FLOOR:
                    if not (np.isfinite(v) and np.isfinite(re[k])):
                        raise StopError(f"非有限: {clip} src{i} 枠 {k}（masks {v}・計算し直し {re[k]}）")
                    d = abs(float(re[k]) - v); diffs.append(d)
                    if d > worst[0]:
                        worst = (d, f"{clip}/src{i}/{s['class']}/枠{k}")
                if vband is not None and k < len(bb):
                    if vvalid is not None and bool(vvalid) != bool(bvalid[k]):
                        raise StopError(f"定義の食い違い: {clip} src{i} 枠 {k} の band_valid（masks {vvalid}・計算し直し {int(bvalid[k])}）")
                    if bvalid[k] and vband >= CMP_FLOOR:
                        db_ = abs(float(bb[k]) - vband); bdiffs.append(db_)
                        if db_ > bworst[0]:
                            bworst = (db_, f"{clip}/src{i}/{s['class']}/枠{k}")
                        if abs(vband) <= BAND_NEAR_DB:                        # B: 線 0 の近く
                            bnear.append(db_)
                            if db_ > bnworst[0]:
                                bnworst = (db_, f"{clip}/src{i}/{s['class']}/枠{k}")
                    for t in BAND_LINES:                                      # §9-6 の 3: 丸めた値が線 ±0.005 dB にある境界枠
                        if abs(vband - t) <= BOUNDARY_DB:
                            n_boundary[f"{t:g}"] += 1
                    if (bool(vvalid if vvalid is not None else True) and vband >= 0.0) != bool(band_gate.passes(bb[k:k + 1], bvalid[k:k + 1], 0.0)[0]):
                        n_round_flip += 1                                     # 丸めた masks での通過と、未丸めの計算し直しでの通過が違う枠（線 0）
            if s.get("no_label"):
                continue
            k0, k1 = max(int(float(s["t_on"]) * 10), 0), min(max(int(float(s["t_off"]) * 10), 1), 100)
            ks = [k for k in range(k0, k1) if k in mm]
            if not ks:
                continue
            new = np.array([mm[k][0] for k in ks]); old = np.array([mm[k][1] for k in ks])
            band = np.array([bb[k] if (k < len(bb) and bvalid[k]) else band_gate.INVALID_DB for k in ks])   # 無効な枠は通らない
            n_invalid += sum(1 for k in ks if k < len(bvalid) and not bvalid[k])
            if not (np.all(np.isfinite(new)) and np.all(np.isfinite(old))):
                raise StopError(f"非有限: {clip} src{i} の鳴っている枠の masks")
            # 仕様 §9-1 の 8: 電車の車両（class crossing の params.train_car・正解の付く車両も）は対象から外し背景に残す（mix − 自分 に入っている）。参考として別に数える
            cls = TRAIN_CAR_REF if (s.get("params") or {}).get("train_car") else s["class"]
            a = acc[cls]
            a["new"] += new.tolist(); a["old"] += old.tolist(); a["band"] += band.tolist()
            B_s = band_db(stems[i]); B_o = band_db(others)
            for k, v in zip(ks, new):
                for lo, hi in NEAR:
                    if lo <= v < hi and k < len(B_s):
                        a[f"best_all_{lo:g}"].append(float((B_s[k] - B_o[k]).max()))
                        a[f"best_noise_{lo:g}"].append(float((B_s[k] - B_noise[k]).max()))
                        a[f"fc_{lo:g}"].append(float(FC[int((B_s[k] - B_o[k]).argmax())]))
            per_src.append({"clip": clip, "src": i, "class": cls, "n_active": len(ks), "keep_new": float(np.mean(new >= LINE)), "keep_old": float(np.mean(old >= 0.0)),
                            "keep_band_0": float(np.mean(band >= 0.0)), "keep_band_3": float(np.mean(band >= 3.0)),
                            "n_label_frames": s.get("n_label_frames"), "audible_frac": s.get("audible_frac")})
    if not diffs:
        raise StopError("照合できる枠が無い（masks ≥ −30 dB の枠が 0）")
    dq = {f"p{q}": float(np.percentile(diffs, q)) for q in (50, 90, 99)}
    dq["max"] = float(max(diffs)); dq["worst"] = worst[1]; dq["n"] = len(diffs)
    res = {"diff_vs_gate": dq, "classes": {}, "n_clips": len(clips), "gate_rule": list(rule0) if rule0 else None,
           "band_gate": {"code_sha256": band_gate.code_sha256(), "contract": band_gate.CONTRACT, "lines_counted": list(BAND_LINES),
                         "n_invalid_active_frames": n_invalid, "rates": "未丸めの計算し直し（valid かつ ≥ 線）・隙間埋めの前（最終ラベルの率は別）"}}
    if bdiffs:
        res["band_diff_vs_gate"] = {**{f"p{q}": float(np.percentile(bdiffs, q)) for q in (50, 90, 99)}, "max": float(max(bdiffs)), "worst": bworst[1], "n": len(bdiffs),
                                    "stop_line": {"p99": BAND_DIFF_P99_MAX, "max": BAND_DIFF_MAX}, "boundary_frames_pm0.005": n_boundary, "round_flip_frames_line0": n_round_flip}
        if bnear:
            res["band_diff_near_line"] = {"p99": float(np.percentile(bnear, 99)), "max": float(max(bnear)), "worst": bnworst[1], "n": len(bnear),
                                          "range_db": [-BAND_NEAR_DB, BAND_NEAR_DB], "stop_line": {"p99": BAND_NEAR_P99_MAX, "max": BAND_NEAR_MAX}}
    import step11_v30_render as v30
    for cls, a in sorted(acc.items()):
        new = np.array(a["new"]); old = np.array(a["old"])
        r = {"gated": (cls in v30.GATED_V30) or cls == TRAIN_CAR_REF, "reference_only": cls == TRAIN_CAR_REF, "n_frames": int(len(new)),
             "bins_new": {f"{lo:g},{hi:g}": float(np.mean((new >= lo) & (new < hi))) for lo, hi in BINS},
             "keep_new_m5": float(np.mean(new >= LINE)), "keep_old_0": float(np.mean(old >= 0.0)),
             "keep_new_0": float(np.mean(new >= 0.0)), "keep_old_m5": float(np.mean(old >= LINE)),
             "keep_band_0": float(np.mean(np.array(a["band"]) >= 0.0)), "keep_band_3": float(np.mean(np.array(a["band"]) >= 3.0))}
        for lo, _ in NEAR:
            b = np.array(a.get(f"best_all_{lo:g}", [])); bn = np.array(a.get(f"best_noise_{lo:g}", [])); fc = np.array(a.get(f"fc_{lo:g}", []))
            if len(b):
                vals, cnt = np.unique(fc, return_counts=True)
                r[f"best_band_{lo:g}"] = {"n": int(len(b)), "all_med": float(np.median(b)), "all_p10": float(np.percentile(b, 10)), "all_p90": float(np.percentile(b, 90)),
                                          "noise_med": float(np.median(bn)), "top_fc": [float(v) for v in vals[np.argsort(-cnt)[:2]]]}
        res["classes"][cls] = r
    res["per_source"] = per_src
    if dq["p99"] > DIFF_P99_MAX or dq["max"] > DIFF_MAX:
        raise StopError(f"定義の食い違い: 照合の差 99% 点 {dq['p99']:.3f} dB（上限 {DIFF_P99_MAX}）・最大 {dq['max']:.3f} dB（上限 {DIFF_MAX}・{dq['worst']}）")
    if bdiffs:
        bq = res["band_diff_vs_gate"]
        if bq["p99"] > BAND_DIFF_P99_MAX or bq["max"] > BAND_DIFF_MAX:
            raise StopError(f"定義の食い違い（帯・A）: 帯の照合の差 99% 点 {bq['p99']:.3f} dB（上限 {BAND_DIFF_P99_MAX}）・最大 {bq['max']:.3f} dB（上限 {BAND_DIFF_MAX}・{bq['worst']}）")
        bn = res.get("band_diff_near_line")
        if bn and (bn["p99"] > BAND_NEAR_P99_MAX or bn["max"] > BAND_NEAR_MAX):
            raise StopError(f"定義の食い違い（帯・B 線 0 ±{BAND_NEAR_DB:g} dB）: 99% 点 {bn['p99']:.3f} dB（上限 {BAND_NEAR_P99_MAX}）・最大 {bn['max']:.3f} dB（上限 {BAND_NEAR_MAX}・{bn['worst']}）")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--clips", default="", help="試走だけ（一覧の外の本・本番の数には使わない）")
    a = ap.parse_args(argv)
    import _run_v2final_gen as g
    plan = {r["clip_id"]: r for r in g.load_plan()}
    if a.clips:
        clips = [c for c in a.clips.split(",") if c]
    else:
        with open(a.list, newline="", encoding="utf-8") as f:
            clips = [r["clip_id"] for r in csv.DictReader(f)]
    if any(c.startswith(("eval_", "val_")) for c in clips):
        raise SystemExit("評価の本と val は使わない")
    out = {"list": str(a.list), "list_sha256": _sha(a.list) if a.list.is_file() else None, "trial_clips": bool(a.clips), "arms": {}, "code": code_ids(),
           "rule": "本体 §9-1 の 8・md/design/v30_m5_小標本_計画_2026-10-26.md"}
    try:
        for arm in a.arms.split(","):
            if not a.analyse_only:
                render([plan[c] for c in clips], a.work, arm)
            out["arms"][arm] = analyse_arm(a.work / arm, clips, arm)
            d = out["arms"][arm]["diff_vs_gate"]
            print(f"[{arm}] 照合の差 中央 {d['p50']:.4f}・99% {d['p99']:.4f}・最大 {d['max']:.4f} dB（{d['worst']}）")
            for cls, r in out["arms"][arm]["classes"].items():
                print(f"  {cls:12s} {'ゲート' if r['gated'] else '    '} 枠 {r['n_frames']:5d}・新 −5 以上 {100 * r['keep_new_m5']:5.1f}%・旧 0 以上 {100 * r['keep_old_0']:5.1f}%"
                      f"（新 0 以上 {100 * r['keep_new_0']:5.1f}%・旧 −5 以上 {100 * r['keep_old_m5']:5.1f}%）・帯 0 以上 {100 * r['keep_band_0']:5.1f}%・帯 +3 以上 {100 * r['keep_band_3']:5.1f}%")
    except StopError as e:
        out["error"] = str(e)
        print("止める:", e)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("->", a.out)
    return 3 if "error" in out else 0


if __name__ == "__main__":
    sys.exit(main())
