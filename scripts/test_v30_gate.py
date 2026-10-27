# -*- coding: utf-8 -*-
"""本体 §2-1（正解の線 −5 dB と背景の定義）・§2-2（踏切のゲート）の受け入れ試験（2026-10-26・v30 の入口の中だけ）。CPU で plan の本を少し描く。

  G1 音源 1 つの本（自転車だけ・キックだけ）: v30 の snr_a_db（背景 = 暗騒音＋雨＋他の音源〔無い〕）と v29 の snr_a_db の差 ≤ 0.01 dB・
     snr_a_bg_db は v29 の snr_a_db と同じ文字・masks の見出しは 5 列・scene.json に gate_rule v30・gate_line_db −5・gate_bg noise+others
  G2 2 音源の本（車＋バック音）: 相手（車）を +12 dB にすると、自分（バック音）の snr_a_db が下がり、正解が消える枠がある（消えた枠は元の部分集合）。snr_a_bg_db は変わらない
  G3 線 −5 の数え（val の自転車の試し描き 5 本）: 自転車の正解の枠 = 「masks の snr_a_db ≥ −5 を隙間埋め（GAP_FILL 2）したもの」∩「音が届いている枠」・
     n_label_frames と一致・v29（線 0・背景 = 暗騒音）より減らない
  G4 ゲートを切った v30（FEATURES gate_v30 = False）は、静止のマイク・クラクションなしの本で v29 の FOA・ラベル・masks と bit 一致（差し込み口は切なら何もしない）
  G5 踏切（§2-2）: 電車の重なる本で、v29 にあった踏切の正解のうち消えた枠がある・消えた枠は全部 v30 の snr_a_db の隙間埋めの外・
     t_first_audible_s（鳴り始め）= 最初に正解の付いた枠
出力は一時フォルダ（終わったら消す・--keep で残す）。
決め直し 4（本人 10/26 19:32）で v30 の既定は帯域ゲート（FEATURES["band_gate"] = True）になった。この試験は**広帯域 −5 の道（帯域ゲート切）**の定義を確かめるので、
始めに FEATURES["band_gate"] = False にして終わりに戻す。既定の帯域ゲートの試験は scripts/test_v30_band_gate.py。
使い方: PYTHONPATH=scripts:src python scripts/test_v30_gate.py
"""
from __future__ import annotations

import csv
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

RESULTS = []
ONE_SRC = ["train_syn_clip34008", "train_syn_clip04806"]
TWO_SRC = "train_syn_clip00170"           # 車＋バック音（静止）
VAL_BIKE = ["val_syn_clip34005", "val_syn_clip37005", "val_syn_clip34010", "val_syn_clip37010", "val_syn_clip34015"]
TRAIN_CROSS = "train_syn_clip06602"       # 電車 1 両＋踏切（静止）
STATIC_NOHORN = ["train_syn_clip34008", "train_syn_clip00067"]


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'NG'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def masks(ds: Path, clip: str) -> list[dict]:
    with open(ds / "masks" / f"{clip}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def meta(ds: Path, clip: str) -> list[tuple]:
    return [tuple(int(v) for v in ln.split(",")) for ln in (ds / "metadata" / f"{clip}.csv").read_text().splitlines() if ln.strip()]


def scene(ds: Path, clip: str) -> dict:
    return json.loads((ds / "work" / clip / "scene.json").read_text())


def main() -> int:
    import _run_v2final_gen as g
    import step11_v30_render as v30
    m9, v13 = g.m9, g.v13
    band_prev = v30.FEATURES["band_gate"]
    v30.FEATURES["band_gate"] = False                     # 広帯域 −5 の道（帯域ゲート切）の試験・既定は帯域ゲート（決め直し 4）
    bg_prev = v30.FEATURES["bg_shape"]
    v30.FEATURES["bg_shape"] = False                      # 10/27 から既定は背景の形を入 → v29 との bit 一致と旧の値の比べは形を切って描く
    need = set(ONE_SRC) | {TWO_SRC, TRAIN_CROSS} | set(VAL_BIKE) | set(STATIC_NOHORN)
    rows = {r["clip_id"]: r for r in g.load_plan() if r["clip_id"] in need}
    tmp = Path(tempfile.mkdtemp(prefix="test_v30_gate_"))
    print("一時フォルダ:", tmp, flush=True)
    try:
        A, B = tmp / "v29", tmp / "v30"
        g.point_to(A)
        for c in sorted(need):
            g.v19r.generate_clip(dict(rows[c]))
        g.point_to(B)
        for c in sorted(need):
            v30.generate_clip(dict(rows[c]))
        # G1
        for c in ONE_SRC:
            ma, mb = masks(A, c), masks(B, c)
            hdr = list(mb[0].keys())
            d = max(abs(float(x["snr_a_db"]) - float(y["snr_a_db"])) for x, y in zip(ma, mb))
            same_bg = all(x["snr_a_db"] == y["snr_a_bg_db"] for x, y in zip(ma, mb)) and len(ma) == len(mb)
            sb = scene(B, c)
            check(f"G1 {c}（音源 1 つ）: 新旧の snr_a_db の差 ≤ 0.01 dB・snr_a_bg_db = 旧の値・masks 6 列（src つき）・scene に gate_rule／gate_line_db／gate_bg",
                  d <= 0.01 and same_bg and hdr == ["frame", "class", "track", "src", "snr_a_db", "snr_a_bg_db"]
                  and (sb.get("gate_rule"), sb.get("gate_line_db"), sb.get("gate_bg")) == ("v30", -5.0, "noise+others"), f"最大差 {d:.4f} dB")
        # G2: 相手（車）を +12 dB
        orig = v13.sample_scene_v13

        def louder(row):
            s = orig(row)
            for src in s["sources"]:
                if src["class"] == "car_drive":
                    src["l1m_db"] = float(src["l1m_db"]) + 12.0
            return s
        C2 = tmp / "v30_loud"
        g.point_to(C2)
        v13.sample_scene_v13 = louder
        try:
            with v30.active():
                v13.generate_clip_v13(dict(rows[TWO_SRC]))          # ピークの引き直しの梯子を通さない（場面が変わらないように）
        finally:
            v13.sample_scene_v13 = orig
        mn = [r for r in masks(B, TWO_SRC) if r["class"] == "2"]; ml = [r for r in masks(C2, TWO_SRC) if r["class"] == "2"]
        dn = np.array([float(r["snr_a_db"]) for r in mn]); dl = np.array([float(r["snr_a_db"]) for r in ml])
        bg_same = [r["snr_a_bg_db"] for r in mn] == [r["snr_a_bg_db"] for r in ml]
        ln = {r[0] for r in meta(B, TWO_SRC) if r[1] == 2}; ll = {r[0] for r in meta(C2, TWO_SRC) if r[1] == 2}
        check("G2 車を +12 dB → バック音の snr_a_db が下がり、正解が消える枠がある（元の部分集合）・snr_a_bg_db（暗騒音＋雨だけ）は同じ",
              np.all(dl <= dn + 1e-9) and np.max(dn - dl) > 3.0 and len(ln - ll) > 0 and ll <= ln and bg_same,
              f"最大の下がり {np.max(dn - dl):.1f} dB・消えた枠 {len(ln - ll)}（{len(ln)} → {len(ll)}）")
        # G3: 線 −5 の数え（val の自転車）
        from outdoor_seld.labels import frame_label_rows
        ok3, det = True, []
        for c in VAL_BIKE:
            sb = scene(B, c); mb = masks(B, c)
            for i, src in enumerate(sb["sources"]):
                if src["class"] != "bicycle":
                    continue
                trk = int(src.get("track", 0) or 0)
                snr = np.array([float(r["snr_a_db"]) for r in mb if r["class"] == "8" and int(r["track"]) == trk])
                aud = m9._fill_gaps(snr >= -5.0, m9.GAP_FILL)
                with v30.active():
                    mic = v30.sample_scene(dict(rows[c]))["_mic_arr"]
                wp = np.array(src["wp"], float)
                pre = float(src.get("preroll_s", 0.0))
                act_from = -pre if pre > 0 else src["t_on"]
                if pre > 0:
                    wp = v13._extend_back(wp, pre)
                act, _ = frame_label_rows(wp, mic, clip_len_sec=m9.CLIP, class_idx=8, track_idx=trk, source_active_from=act_from,
                                          source_active_until=src["t_off"], c=m9.sound_speed(m9.TEMP_C))
                want = {r[0] for r in act if aud[r[0]]}
                got = {r[0] for r in meta(B, c) if r[1] == 8 and r[2] == trk}
                old = {r[0] for r in meta(A, c) if r[1] == 8 and r[2] == trk}
                ok3 &= (got == want and len(got) == src["n_label_frames"] and len(got) >= len(old))
                det.append(f"{c}: {len(old)}→{len(got)}")
        check("G3 val の自転車 5 本: 正解の枠 = 隙間埋め（snr_a_db ≥ −5）∩ 音が届いている枠・n_label_frames と一致・v29 より減らない", ok3, "・".join(det))
        # G4: ゲートを切った v30 は v29 と bit 一致（静止・クラクションなし）
        C4 = tmp / "v30_nogate"
        g.point_to(C4)
        v30.FEATURES["gate_v30"] = False
        try:
            for c in STATIC_NOHORN:
                v30.generate_clip(dict(rows[c]))
        finally:
            v30.FEATURES["gate_v30"] = True
        same4 = all((A / d / f"{c}.{e}").read_bytes() == (C4 / d / f"{c}.{e}").read_bytes()
                    for c in STATIC_NOHORN for d, e in (("foa", "flac"), ("metadata", "csv"), ("masks", "csv")))
        check("G4 ゲートを切った v30（静止・クラクションなし 2 本）は v29 と FOA・ラベル・masks が bit 一致", same4)
        # G5: 踏切（電車の重なる本）
        sb = scene(B, TRAIN_CROSS); mb = masks(B, TRAIN_CROSS)
        old5 = {(r[0], r[2]) for r in meta(A, TRAIN_CROSS) if r[1] == 5}
        new5 = {(r[0], r[2]) for r in meta(B, TRAIN_CROSS) if r[1] == 5}
        lost = old5 - new5
        consistent, firsts = True, []
        for src in sb["sources"]:
            if src["class"] != "crossing" or src.get("no_label"):
                continue
            trk = int(src.get("track", 0) or 0)
            snr = np.array([float(r["snr_a_db"]) for r in mb if r["class"] == "5" and int(r["track"]) == trk][:100])
            aud = m9._fill_gaps(snr >= -5.0, m9.GAP_FILL)
            consistent &= all(not aud[k] for k, t in lost if t == trk)
            fr = sorted(k for k, t in new5 if t == trk)
            firsts.append((src.get("t_first_audible_s"), round(fr[0] * 0.1, 2) if fr else None))
        n_train = sum(1 for x in sb["sources"] if x.get("no_label"))
        # R30-10: src 列で同じ (class 5, track 0) の電車の車両と警報機が分かれ、src は scene の音源の番号と一致する
        src_ok = all(sb["sources"][int(r["src"])]["class"] == {"5": "crossing"}.get(r["class"], sb["sources"][int(r["src"])]["class"]) for r in mb) \
            and len({r["src"] for r in mb}) == len(sb["sources"])
        check("G5 踏切（電車 1 両と重なる本）: v29 の踏切の正解のうち消えた枠がある・消えた枠は全部 v30 の隙間埋めの外・鳴り始め = 最初に正解の付いた枠",
              n_train >= 1 and len(lost) > 0 and consistent and all(a == b for a, b in firsts) and src_ok,
              f"電車 {n_train} 両・踏切の正解 {len(old5)} → {len(new5)}（消えた {len(lost)}）・鳴り始め {firsts}")
        # 読み手が列名で読む（R30-10）: 完了の判定 clip_complete・val_gate_* の数え（DictReader）が v30 の 6 列の masks で動く
        import _run_v19_gen as G19
        G19.m9.DS = B
        bad_cc = [c for c in sorted(need) if not G19.clip_complete(c)]
        cc = not bad_cc
        import subprocess as _sp
        rv = _sp.run([sys.executable, str(ROOT / "scripts/val_gate_by_class.py"), "--masks", str(B / "masks"), "--meta", str(B / "metadata")],
                     capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
        check("G6 読み手: clip_complete が v30 の 6 列の masks を受ける・val_gate_by_class が列名で読んで走る", cc and rv.returncode == 0 and "Traceback" not in rv.stderr,
              (f"未完了と判定 {bad_cc} " + ((rv.stdout[-200:] + rv.stderr[-300:]) if rv.returncode else "")))
    finally:
        v30.FEATURES["band_gate"] = band_prev
        v30.FEATURES["bg_shape"] = bg_prev
        if "--keep" in sys.argv:
            print("残した:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(ng)}/{len(RESULTS)} OK" + (f"  NG: {ng}" if ng else ""))
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
