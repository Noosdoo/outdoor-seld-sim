# -*- coding: utf-8 -*-
"""先行分 §1（決定 ⑩・2026-10-26 Fable 仕様）の受け入れ試験: マイクの正面を進む向きに回す（描画 v30 の m9.MIC_HEADING）。

A. 関数（速い）: mic_heading_deg の規約（静止 0・±x・止まる・最初から止まる・+y・範囲外）／記録の形と読み戻し／R_z(−ψ) の回し方
   （ψ = 0 は bit 一致・180 は X→−X・Y→−Y がちょうど）／方位の回し方がベクトルの回し方と同じ／frame_label_rows の heading（None と 0 が同じ）／
   方位の円周平均（±180 をまたぐ・打ち消し合うと None・欠測）。
A'. 受音時刻の向き（描画・速い）: +x に歩いて 5 s で +y に曲がるマイクと 95 m 後ろの音源（遅れ約 0.28 s）で、曲がった直後の FOA とラベルの方位が
   受音時刻の向き（90）で回っている（放射時刻の向き 0 ではない）。
B. 描画（遅い・CPU で 4 本 × 切／入）: 本番の生成ドライバ（_run_v2final_gen の鎖 v9 → … → v29）と同じ道で、v30 の入口（v30.generate_clip）の中で
   回転（FEATURES["mic_heading"]）だけを切／入にして描く。import しただけでは m9.MIC_HEADING が切のまま・入口を抜けると切に戻る（B0）。
   静止（00001）と dirx = +1 の歩行（00030）は FOA・ラベル・masks が bit 一致。dirx = −1 の歩行（00009 車が世界の左・00083 プリロールのサイレン）は
   masks と W が bit 一致・ラベルの方位だけが 180 度ずれる・車の最接近の方位が −90 付近（マイク座標の右）・描いた FOA の intensity_vector_doa と
   ラベルの方位が最接近の前後 1 秒の中央値で 3 度以内。scene.json は新しい鍵（mic_heading_deg・side_mic）と生成の秒数のほかは同じ。
   距離ラベル（step19_dist_labels_v16 の照合の門）は入の本でも通り、距離の列は切と同じ。
   B11（Astra R30-08）: v29 単独・v30 単独・同じプロセスで v29 → v30 の順に import の 3 通り（別プロセス）で、v29 の入口（v19r.generate_clip）の出力が
   bit 一致（回転とクラクションの Δ が効く本で）。v30 の入口の出力は v29 と違う（効いている証拠）。
出力は一時フォルダ（終わったら消す・--keep で残す）。リポジトリの out/ には書かない。

使い方: PYTHONPATH=scripts:src python scripts/test_render_mic_heading.py [--quick]（--quick = A だけ）
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.geometry import (az_to_mic_frame, azel_deg_to_unit, circular_mean_deg,  # noqa: E402
                                   mic_heading_deg, mic_heading_from_record, mic_heading_record,
                                   rotate_to_mic_frame, unit_to_azel_deg)
from outdoor_seld.labels import frame_label_rows  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'NG'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def wrap(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ A. 関数 ----
def part_a() -> None:
    t = np.linspace(-1.0, 11.0, 241)
    check("A1 静止 → ψ = 0", np.all(mic_heading_deg(np.array([0.0, 0.0, 1.5]), t) == 0.0))
    walk_p = np.array([[0.0, -6.0, 0.0, 1.5], [10.0, 6.0, 0.0, 1.5]])
    walk_m = np.array([[0.0, 6.0, 0.0, 1.5], [10.0, -6.0, 0.0, 1.5]])
    check("A2 歩行 +x → 0・範囲外も 0", np.all(mic_heading_deg(walk_p, t) == 0.0))
    check("A3 歩行 −x → 180（範囲外も）", np.all(mic_heading_deg(walk_m, t) == 180.0))
    stop_x = np.array([[0.0, -5.0, 0.0, 1.5], [5.0, 0.0, 0.0, 1.5], [10.0, 0.0, 0.0, 1.5]])
    check("A4 walk_stop_x（+x に歩いて止まる）→ 0 のまま", np.all(mic_heading_deg(stop_x, t) == 0.0))
    cross_y = np.array([[0.0, 0.0, -6.0, 1.5], [5.0, 0.0, -0.5, 1.5], [10.0, 0.0, -0.5, 1.5]])
    check("A5 walk_cross_y（+y に歩いて止まる）→ 90 のまま（止まった後も）", np.all(mic_heading_deg(cross_y, t) == 90.0))
    late = np.array([[0.0, 0.0, 0.0, 1.5], [3.0, 0.0, 0.0, 1.5], [10.0, 0.0, -7.0, 1.5]])
    check("A6 最初から止まって −y に動く → 最初も −90", np.all(mic_heading_deg(late, t) == -90.0))
    turn = np.array([[0.0, 0.0, 0.0, 1.5], [4.0, 4.0, 0.0, 1.5], [6.0, 4.0, 0.0, 1.5], [10.0, 4.0, 4.0, 1.5]])
    h = mic_heading_deg(turn, np.array([1.0, 4.0, 5.0, 5.999, 6.0, 9.0, 12.0]))
    check("A7 +x → 止まる（直前の 0 を保つ）→ +y（90）・区間の始まりで切り替わる", list(h) == [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 90.0], str([float(x) for x in h]))
    allstill = np.array([[0.0, 1.0, 1.0, 1.5], [10.0, 1.0, 1.0, 1.5]])
    check("A8 経路の全部が静止 → 0", np.all(mic_heading_deg(allstill, t) == 0.0))
    rec_ok = True
    for arr in (walk_p, walk_m, stop_x, cross_y, late, turn, allstill):
        rec = mic_heading_record(arr)
        json.loads(json.dumps(rec))
        rec_ok &= bool(np.array_equal(mic_heading_from_record(rec, t), mic_heading_deg(arr, t)))
    check("A9 記録の形（数か [[t0, ψ], …]）から読み戻した ψ が mic_heading_deg と同じ", rec_ok,
          f"walk_m={mic_heading_record(walk_m)} turn={mic_heading_record(turn)}")
    rng = np.random.default_rng(0)
    u = azel_deg_to_unit(rng.uniform(-180, 180, 500), rng.uniform(-60, 60, 500))
    check("A10 ψ = 0 の回し方は bit 一致（同じ配列）", rotate_to_mic_frame(u, 0.0) is u
          and np.array_equal(rotate_to_mic_frame(u, np.zeros(500)), u))
    r180 = rotate_to_mic_frame(u, 180.0)
    check("A11 ψ = 180 → X→−X・Y→−Y・Z そのまま（ちょうど）", np.array_equal(r180[:, 0], -u[:, 0]) and np.array_equal(r180[:, 1], -u[:, 1])
          and np.array_equal(r180[:, 2], u[:, 2]))
    r90 = rotate_to_mic_frame(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 90.0)
    check("A12 ψ = 90（+y を向く）→ 世界の前は右（0,−1,0）・世界の左は前（1,0,0）", np.allclose(r90, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-15), str(r90.round(6).tolist()))
    psi = rng.uniform(-180, 180, 500)
    az0, el0 = unit_to_azel_deg(u)
    az_v, el_v = unit_to_azel_deg(rotate_to_mic_frame(u, psi))
    az_s = az_to_mic_frame(az0, psi)
    check("A13 方位の回し方 = ベクトルの回し方（同じ行列）・仰角は変わらない", np.max(np.abs(wrap(az_s - az_v))) < 1e-9 and np.max(np.abs(el_v - el0)) < 1e-9,
          f"max|Δaz|={np.max(np.abs(wrap(az_s - az_v))):.2e}")
    check("A14 az_mic = wrap(az − ψ)・範囲 (−180, 180]", np.max(np.abs(wrap(az_s - (az0 - psi)))) < 1e-9 and np.all(az_s > -180.0) and np.all(az_s <= 180.0))
    a = az_to_mic_frame(np.array([np.nan, 10.0]), np.array([30.0, 30.0]))
    check("A15 NaN（未到達）は NaN のまま", np.isnan(a[0]) and abs(a[1] + 20.0) < 1e-9)
    wp = np.array([[0.0, -20.0, 2.0, 0.5], [10.0, 20.0, 2.0, 0.5]])
    mic = np.array([[0.0, 6.0, 0.0, 1.5], [10.0, -6.0, 0.0, 1.5]])
    r_none, d_none = frame_label_rows(wp, mic, 10.0, 4, 0)
    r_zero, _ = frame_label_rows(wp, mic, 10.0, 4, 0, heading_deg=np.zeros(100))
    r_s0, _ = frame_label_rows(wp, mic, 10.0, 4, 0, heading_deg=0.0)
    check("A16 frame_label_rows: heading None ＝ 0（配列）＝ 0（数）", r_none == r_zero == r_s0, f"{len(r_none)} 行")
    r_180, d_180 = frame_label_rows(wp, mic, 10.0, 4, 0, heading_deg=mic_heading_deg(mic, (np.arange(100) + 0.5) * 0.1))
    same_frames = [x[0] for x in r_none] == [x[0] for x in r_180] and [x[4] for x in r_none] == [x[4] for x in r_180]
    daz = np.array([wrap(b[3] - a_[3] - 180.0) for a_, b in zip(r_none, r_180)])
    check("A17 dirx = −1 の heading: 行・仰角は同じで方位だけ 180 ずれる（丸めの 1 度以内）", same_frames and np.all(np.abs(daz) <= 1.0)
          and np.allclose(d_180["az_world"], d_none["az"], equal_nan=True), f"max|Δ|={np.max(np.abs(daz)) if len(daz) else 0}")
    c1 = circular_mean_deg([179.0, -179.0, 178.0, -178.0])
    check("A18 円周平均: ±180 をまたぐ（179・−179・178・−178 → 180）", c1 is not None and abs(wrap(c1 - 180.0)) < 1e-9 and c1 > -180.0, str(c1))
    check("A19 円周平均: 打ち消し合う角度（90・−90／0・120・−120）→ 決まらない（None）",
          circular_mean_deg([90.0, -90.0]) is None and circular_mean_deg([0.0, 120.0, -120.0]) is None)
    c3 = circular_mean_deg([np.nan, 10.0, np.nan, 20.0])
    check("A20 円周平均: 欠測（NaN）は外す・全部欠測／空は None", c3 is not None and abs(c3 - 15.0) < 1e-9
          and circular_mean_deg([np.nan, np.nan]) is None and circular_mean_deg([]) is None, str(c3))


def part_a_reception() -> None:
    """受音時刻の向き: 曲がった直後（受音 5.0〜5.28 s）は放射時刻がまだ曲がる前（< 5 s）だが、回すのは受音時刻の ψ = 90。"""
    import _run_v2final_gen as g
    import step11_v30_render as v30
    from outdoor_seld.foa import intensity_vector_doa
    m9 = g.m9
    mic = np.array([[0.0, 0.0, 0.0, 1.5], [5.0, 5.0, 0.0, 1.5], [10.0, 5.0, 5.0, 1.5]])     # +x に 1 m/s → 5 s から +y に 1 m/s
    wp = np.array([[0.0, -95.0, 0.0, 1.5], [10.0, -95.0, 0.0, 1.5]])                        # 後ろ 95〜100 m（遅れ約 0.28〜0.29 s）
    dry = np.random.default_rng(7).standard_normal(int(m9.CLIP * m9.FS_SIM)) * 0.05
    c = m9.sound_speed(m9.TEMP_C)
    with v30.active():
        stem_d, _ = m9._render_stem(dry, wp, mic, c)
        rows, dbg = m9.frame_label_rows(wp, mic, clip_len_sec=m9.CLIP, class_idx=4, c=c, heading_deg=m9.label_heading(mic))
    _, az_iv, _, _ = intensity_vector_doa(stem_d, m9.FS_OUT)
    lab = {r[0]: r[3] for r in rows}
    after = [51, 52]                      # 受音 5.1〜5.3 s: 受音時刻の ψ = 90 → 後ろの音源は左（+90）・放射時刻の ψ（0）なら 180
    before = [45, 46, 47, 48]
    ok_lab = all(abs(wrap(lab[k] - 90.0)) <= 5.0 for k in after) and all(abs(wrap(lab[k] - 180.0)) <= 5.0 for k in before)
    ok_iv = all(abs(wrap(az_iv[k] - 90.0)) <= 8.0 for k in after) and all(abs(wrap(az_iv[k] - 180.0)) <= 8.0 for k in before)
    check("A21 受音時刻の向き: 曲がった直後のラベルと FOA（intensity 法）が ψ = 90 で回っている（放射時刻の ψ = 0 なら 180）・曲がる前は 180",
          ok_lab and ok_iv and not m9.MIC_HEADING,
          f"ラベル 前 {[lab[k] for k in before]}・後 {[lab[k] for k in after]}／FOA 前 {[round(float(az_iv[k]), 1) for k in before]}・後 {[round(float(az_iv[k]), 1) for k in after]}")


# ------------------------------------------------------------------ B. 描画 ----
CLIPS = {"train_syn_clip00001": ("static", None), "train_syn_clip00030": ("walk", 1.0),
         "train_syn_clip00009": ("walk", -1.0), "train_syn_clip00083": ("walk", -1.0)}
VOLATILE = {"gen_seconds"}
NEW_KEYS = {"mic_heading_deg", "side_mic"}


def _strip(o, drop):
    if isinstance(o, dict):
        return {k: _strip(v, drop) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [_strip(v, drop) for v in o]
    return o


def _read_meta(p: Path) -> list:
    return [tuple(int(v) for v in ln.split(",")) for ln in p.read_text().splitlines() if ln.strip()]


def part_b(keep: bool) -> None:
    import soundfile as sf
    import _run_v2final_gen as g
    import step11_v30_render as v30
    from outdoor_seld.foa import intensity_vector_doa
    m9 = g.m9
    st = [m9.MIC_HEADING]
    with v30.active():
        st.append(m9.MIC_HEADING)
    st.append(m9.MIC_HEADING)
    check("B0 v30 を import しただけでは m9.MIC_HEADING は切・v30 の入口（active）の中だけ入・抜けると切に戻る", st == [False, True, False], str(st))
    rows = {r["clip_id"]: r for r in g.load_plan() if r["clip_id"] in CLIPS}
    tmp = Path(tempfile.mkdtemp(prefix="test_render_mic_heading_"))
    print("一時フォルダ:", tmp, flush=True)
    try:
        for flag, sub in ((False, "off"), (True, "on")):
            v30.FEATURES["mic_heading"] = flag              # v30 の入口の中で回転だけを切／入（ほかの v30 の直しは両方で同じ）
            g.point_to(tmp / sub)
            for cid in CLIPS:
                v30.generate_clip(dict(rows[cid]))
        v30.FEATURES["mic_heading"] = True
        c = m9.sound_speed(m9.TEMP_C)
        for cid, (motion, dirx) in CLIPS.items():
            A, B = tmp / "off", tmp / "on"
            sa = json.loads((A / "work" / cid / "scene.json").read_text())
            sb = json.loads((B / "work" / cid / "scene.json").read_text())
            check(f"B1 {cid} 場面の種類（{motion} dirx={dirx}）", sa["mic"]["motion"] == motion and sa["mic"].get("walk_dir_x") == dirx)
            check(f"B2 {cid} scene.json は新しい鍵と秒数のほか同じ（乱数・速度・横距離・t_on/t_off・受音量）",
                  _strip(sa, VOLATILE) == _strip(sb, VOLATILE | NEW_KEYS) and not any(k in json.dumps(sa) for k in ('"side_mic"', '"mic_heading_deg"')))
            want = 0.0 if dirx in (None, 1.0) else 180.0
            check(f"B3 {cid} mic_heading_deg = {want}・side_mic が全部の音源にある", sb["mic"].get("mic_heading_deg") == want
                  and all(x.get("side_mic") in ("L", "R", "") for x in sb["sources"]), str([(x["class"], x.get("side_mic")) for x in sb["sources"]]))
            masks_same = (A / "masks" / f"{cid}.csv").read_bytes() == (B / "masks" / f"{cid}.csv").read_bytes()
            fa, _ = sf.read(A / "foa" / f"{cid}.flac", dtype="float64", always_2d=True)
            fb, _ = sf.read(B / "foa" / f"{cid}.flac", dtype="float64", always_2d=True)
            ma, mb = _read_meta(A / "metadata" / f"{cid}.csv"), _read_meta(B / "metadata" / f"{cid}.csv")
            if want == 0.0:
                same = all((A / d / f"{cid}.{e}").read_bytes() == (B / d / f"{cid}.{e}").read_bytes()
                           for d, e in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")))
                check(f"B4 {cid} ψ = 0: FOA・ラベル・masks が bit 一致", same)
                continue
            check(f"B5 {cid} masks と W が bit 一致（W は向きで変わらない）", masks_same and np.array_equal(fa[:, 0], fb[:, 0]))
            # 混ぜた音には回さない拡散の背景（暗騒音・雨）が入るので、符号の反転は音源ごとの stem で見る
            flip = []
            for i, src in enumerate(sb["sources"]):
                for kind in ("direct", "withrefl"):
                    f = f"src{i}_{src['class']}_{kind}_24k.flac"
                    xa, _ = sf.read(A / "work" / cid / f, dtype="float64", always_2d=True)
                    xb, _ = sf.read(B / "work" / cid / f, dtype="float64", always_2d=True)
                    flip.append(np.array_equal(xa[:, 0], xb[:, 0]) and np.array_equal(xa[:, 2], xb[:, 2])
                                and np.max(np.abs(xb[:, 1] + xa[:, 1])) <= 2.0 ** -22 and np.max(np.abs(xb[:, 3] + xa[:, 3])) <= 2.0 ** -22)
            check(f"B6 {cid} 音源の stem（direct・反射込み）: W・Z は bit 一致・X→−X・Y→−Y（24 bit の丸めの範囲）", len(flip) > 0 and all(flip), f"{sum(flip)}/{len(flip)} 本")
            rows_same = [(r[0], r[1], r[2], r[4]) for r in ma] == [(r[0], r[1], r[2], r[4]) for r in mb]
            daz = np.array([wrap(b[3] - a[3] - 180.0) for a, b in zip(ma, mb)])
            check(f"B7 {cid} ラベル: 行・クラス・track・仰角は同じで方位だけ 180 ずれる（丸め 1 度以内）", rows_same and len(daz) > 0 and np.all(np.abs(daz) <= 1.0),
                  f"{len(ma)} 行 max|Δ|={np.max(np.abs(daz)) if len(daz) else None}")
            # intensity 法: 1 音源の direct ステムで、最接近の前後 1 秒のフレームの方位をラベルと比べる
            for i, src in enumerate(sb["sources"]):
                if src["class"] not in ("car_drive", "siren") or src.get("track", 0) != 0:
                    continue
                stem, _ = sf.read(B / "work" / cid / f"src{i}_{src['class']}_direct_24k.flac", dtype="float64", always_2d=True)
                tf, az_iv, _, en = intensity_vector_doa(stem.T, 24000)
                ci = g.v13.v12.CLASS_IDX_V12[src["class"]]
                lab = {r[0]: r[3] for r in mb if r[1] == ci and r[2] == 0}
                k0 = int(round(src["t_min_dist_s"] / 0.1))
                ks = [k for k in range(k0 - 10, k0 + 11) if k in lab and k < len(az_iv) and np.isfinite(az_iv[k])]
                d = wrap(np.array([az_iv[k] - lab[k] for k in ks]))
                med = circular_mean_deg(d) if len(d) else None                 # 円周統計（Astra R30-09）
                med = float("nan") if med is None else med
                check(f"B8 {cid} {src['class']}: 描いた FOA の intensity 方位とラベルの方位の差（最接近 ±1 s・円周平均）が 3 度以内", len(d) >= 5 and abs(med) <= 3.0,
                      f"中央値 {med:+.2f}°・{len(d)} 枠・最接近の枠のラベル {lab.get(k0)}°・side_mic={src.get('side_mic')}")
                if cid == "train_syn_clip00009" and src["class"] == "car_drive":
                    lat_left = np.mean([p[2] for p in src["wp"]]) > 0
                    check(f"B9 {cid} 車は世界の左（+y）→ マイク座標の右（最接近の方位 −90 付近・side_mic = R）",
                          lat_left and lab.get(k0) is not None and abs(wrap(lab[k0] + 90.0)) <= 20.0 and src.get("side_mic") == "R",
                          f"y={src['wp'][0][2]:.2f} 最接近の方位 {lab.get(k0)}°")
        # 距離ラベル: v16 の照合の門（方位を位置から計算し直して metadata と全行比べる）が入の本で通り、距離の列が切と同じ
        import step19_dist_labels_v16 as s16
        dist = {}
        for sub in ("off", "on"):
            ds = tmp / sub
            s16.DS = ds; s16.s19.DS = ds; s16.s19.OUT = ds / "metadata_dist"; s16.m9.DS = ds; s16.m9.WORK = ds / "work"
            for cid in CLIPS:
                s16.regen_clip_v15(cid, c)
                dist[(sub, cid)] = [ln.split(",") for ln in (ds / "metadata_dist" / f"{cid}.csv").read_text().splitlines()]
        okd = all([r[:3] + r[4:] for r in dist[("off", cid)]] == [r[:3] + r[4:] for r in dist[("on", cid)]] for cid in CLIPS)
        check("B10 距離ラベル（v16 の照合の門）が入の本でも通り、方位のほか（距離の列）は切と同じ", okd)
        part_b_imports(tmp)
    finally:
        if keep:
            print("残した:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


IMPORT_CLIPS = ("train_syn_clip00009", "train_syn_clip00208")      # dirx = −1 の歩行（回転が効く）・静止のクラクション（Δ が効く）


def child(mode: str, out: str) -> int:
    """B11 の子プロセス: mode = v29（v29 の鎖だけ）／v30（v30 を最初に import）／v29v30（v29 の鎖 → v30 の順）。v29 の入口で描く。"""
    if mode == "v30":
        import step11_v30_render as v30  # noqa: F401  v30 を最初に読む（中で v29 の鎖を読む）
    import _run_v2final_gen as g          # 生成ドライバの設定（D1 SIREN_MIX・出力先）
    if mode == "v29v30":
        import step11_v30_render as v30  # noqa: F401
    rows = {r["clip_id"]: r for r in g.load_plan() if r["clip_id"] in IMPORT_CLIPS}
    g.point_to(Path(out) / "v29path")
    for cid in IMPORT_CLIPS:
        g.v19r.generate_clip(dict(rows[cid]))
    if mode == "v29v30":
        g.point_to(Path(out) / "v30path")
        for cid in IMPORT_CLIPS:
            v30.generate_clip(dict(rows[cid]))
        g.point_to(Path(out) / "v29path_again")              # §9-4 の 6: v30 → v29 の順にもう一度 v29 を描く（入口を抜けたら元に戻るか）
        for cid in IMPORT_CLIPS:
            g.v19r.generate_clip(dict(rows[cid]))
    print("child-ok", mode, "MIC_HEADING", g.m9.MIC_HEADING)
    return 0


def part_b_imports(tmp: Path) -> None:
    import subprocess
    import os
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "src")]), PYTHONUTF8="1")
    for mode in ("v29", "v30", "v29v30"):
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", mode, str(tmp / f"imp_{mode}")],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or "child-ok" not in r.stdout:
            check(f"B11 子プロセス {mode} が走る", False, (r.stdout + r.stderr)[-800:])
            return
    same, diff = [], []
    for cid in IMPORT_CLIPS:
        for d, e in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")):
            b = [(tmp / f"imp_{m}" / "v29path" / d / f"{cid}.{e}").read_bytes() for m in ("v29", "v30", "v29v30")]
            b.append((tmp / "imp_v29v30" / "v29path_again" / d / f"{cid}.{e}").read_bytes())
            same.append(b[0] == b[1] == b[2] == b[3])
        diff.append((tmp / "imp_v29v30" / "v30path" / "foa" / f"{cid}.flac").read_bytes() != (tmp / "imp_v29" / "v29path" / "foa" / f"{cid}.flac").read_bytes())
    check("B11 v29 単独・v30 単独・v29 → v30 の順の 3 通りと、同じプロセスで v30 → v29 の順にもう一度描いた v29 で、v29 の入口の出力（FOA・ラベル・masks）が bit 一致（回転と Δ が効く 2 本）",
          all(same), f"{sum(same)}/{len(same)}")
    check("B12 同じプロセスの v30 の入口の出力は v29 と違う（回転・Δ が効いている）", all(diff), str(diff))


def main() -> int:
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        return child(sys.argv[i + 1], sys.argv[i + 2])
    part_a()
    part_a_reception()
    if "--quick" not in sys.argv:
        part_b("--keep" in sys.argv)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(ng)}/{len(RESULTS)} OK" + (f"  NG: {ng}" if ng else ""))
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
