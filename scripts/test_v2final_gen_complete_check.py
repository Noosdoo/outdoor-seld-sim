# -*- coding: utf-8 -*-
"""server_sde/_v2final_gen_complete_check.py の検査（再監査 Astra 10/1 RA01・10/6 RF01 の反例を含む）。

人工の束（plan 6 本・4 種の成果物・HDF5 ルート）を一時フォルダに作り、
  C0 そろっていれば OK（clip 1・2 は正解あり = csv の行と h5 の活動が一致、他は正解なし = csv 0 バイト・h5 は全部 0）
  C1 unrelated.h5（中身は HDF5 でない）だけが新しくてラベル h5 が無い → NG（RA01 の反例）
  C2 ラベル h5 の片方だけ古い（更新されていない）→ NG
  C3 adpit_dist に plan の clip が 1 本欠ける → NG
  C4 adpit_dist の adpit/dist が空 → NG
  C5 foa の 1 本が 0 バイト → NG
  C6 特徴 h5（clip 単位でない）が空 → NG、キーがあれば OK
  C7 plan の行数が期待と違う → NG
  C8〜C12（監査 A 10/1 F2・F12）: clip ごとの特徴 h5 が半分欠け／1 本も無い（要求する構成のとき）・adpit/se が無い・adpit/dist が NaN・特徴 h5 の中身が空 → NG
  C13〜C15: accdoa のラベル h5 が無い／track の活動配列が無い → NG、正解なしの 0 バイト csv は OK・masks の 0 バイトは NG
  C20〜C22（Astra 監査 10/7 V05・L02）: h5 を 1 フレームに短く切る → NG（長さは csv の最後のフレーム+1 と一致が規約）。4 音源のフレームで track を全消し → NG。csv に clip の長さ以上のフレーム → NG
  C16〜C19（再監査 10/6 RF01）: 正解 csv を残して 4 種の h5 の活動を全部 0 に → NG。csv が空なのに h5 に活動 → NG。
      h5 の活動フレームが csv と 1 フレームずれる → NG。metadata と metadata_dist の先頭 5 列が食い違う → NG。
      --no-content なら C16 は通る（飛ばした印が出る）
を確かめる。h5py が要るので PSELDNets の venv で走らせる:
    & 'C:\\Users\\satos\\research\\PSELDNet\\PSELDNets\\.venv\\Scripts\\python.exe' scripts\\test_v2final_gen_complete_check.py
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server_sde"))
import _v2final_gen_complete_check as CC  # noqa: E402

try:
    import h5py
except Exception as e:  # pragma: no cover
    print("h5py が無い:", e, "→ PSELDNets の venv で実行する")
    sys.exit(2)

IDS = [f"train_syn_clip{i:05d}" for i in range(1, 5)] + ["val_syn_clip00005", "val_syn_clip30001"]
T_PRE = time.time() - 100  # 「前処理の開始」はいまより 100 秒前
T, C = 10, 8
# 正解あり 2 本: clip -> [(frame, cls, trk, az, el, dist)]（dist は距離クラスだけ・サーバの実物と同じ 6 列）
# IDS[1] の frame 3〜5 は 4 音源（クラクション・車・キック・バイク）= track 形式は上位 3 枠だけ残す（PSELDNets の仕様・L02 の検査用）
ROWS = {
    IDS[0]: [(f, 4, 0, 176 - f, 0, 2.0 + 0.1 * f) for f in range(1, 10)],                       # 車 1 台・frame 1〜9
    IDS[1]: ([(f, 1, 0, 30, 0, None) for f in range(1, 6)] + [(f, 4, 0, -20, -1, 3.0) for f in range(3, 7)]
             + [(f, 6, 0, 90, 0, 1.5) for f in range(3, 6)] + [(f, 7, 0, -90, 0, 4.0) for f in range(3, 6)]),
}


def _write_meta(ds: Path, c: str, rows, dist: bool):
    p = ds / ("metadata_dist" if dist else "metadata") / f"{c}.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        for fr, cls, trk, az, el, d in rows:
            cols = [fr, cls, trk, az, el] + ([("" if d is None else d)] if dist else [])
            f.write(",".join(str(x) for x in cols) + "\n")


def make_fixture(root: Path, *, n_ids=6, feature_keys=("feat",), clip_features=False):
    ds = root / "ds"
    (ds / "plan").mkdir(parents=True)
    with open(ds / "plan" / "assignment_v2final.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "split", "scenario"])
        for c in IDS[:n_ids]:
            w.writerow([c, "val" if c.startswith("val") else "train", "x"])
    for sub, ext, _nonempty in CC.REQUIRED_SUBS:
        (ds / sub).mkdir()
        for c in IDS[:n_ids]:
            if sub in ("metadata", "metadata_dist"):
                _write_meta(ds, c, ROWS.get(c, []), dist=(sub == "metadata_dist"))   # 正解なしは 0 バイト（実物と同じ）
            else:
                (ds / sub / f"{c}.{ext}").write_bytes(b"x")
    h5root = root / "h5"
    for kind in CC.LABEL_KINDS:          # サーバの実物と同じ 4 種（adpit・adpit_dist・accdoa・track）と同じ配列の形
        d = h5root / "label" / kind / "dev"
        d.mkdir(parents=True)
        with h5py.File(d / "outdoor_siren_v12.h5", "w") as hf:
            for c in IDS[:n_ids]:
                rows = ROWS.get(c, [])
                Tc = (max(r[0] for r in rows) + 1) if rows else T      # PSELDNets の規約: 配列の長さ = csv の最後のフレーム + 1
                if kind in ("adpit", "adpit_dist"):
                    se = np.zeros((Tc, 6, C), dtype=bool); dist = np.zeros((Tc, 6, C), dtype=np.float32)
                    for fr, cls, trk, az, el, dd in rows:
                        se[fr, 0, cls] = True
                        if dd is not None:
                            dist[fr, 0, cls] = dd
                    hf.create_dataset(f"{c}/adpit/se", data=se)
                    hf.create_dataset(f"{c}/adpit/azi", data=np.zeros((Tc, 6, C), dtype=np.int16))
                    if kind == "adpit_dist":
                        hf.create_dataset(f"{c}/adpit/dist", data=dist)
                elif kind == "accdoa":
                    se = np.zeros((Tc, C), dtype=bool)
                    for fr, cls, *_ in rows:
                        se[fr, cls] = True
                    hf.create_dataset(f"{c}/accdoa/se", data=se)
                    hf.create_dataset(f"{c}/accdoa/azi", data=np.zeros((Tc, C), dtype=np.int16))
                else:  # track: フレームごとに csv の行順で 3 枠まで（4 本目以降は落ちる = 実物の extract_track_label と同じ）
                    se = np.zeros((Tc, 3, C), dtype=bool)
                    used = {}
                    for fr, cls, *_ in rows:
                        k = used.get(fr, 0)
                        if k < 3:
                            se[fr, k, cls] = True
                            used[fr] = k + 1
                    hf.create_dataset(f"{c}/sed_label", data=se)
                    hf.create_dataset(f"{c}/doa_label", data=np.zeros((Tc, 3, 3), dtype=np.float32))
    fd = h5root / "features" / "dev"
    fd.mkdir(parents=True)
    with h5py.File(fd / "feat.h5", "w") as hf:
        for k in feature_keys:
            hf.create_dataset(k, data=np.zeros(3))
    # clip ごとの特徴 h5（置き方によっては data/<fs>fs/.../dev/<ds>/<clip>.h5・キー feature。v2final の実物は持たないので既定の検査では要求しない）
    cd = h5root / "data" / "24000fs" / "wav" / "dev" / "outdoor_siren_v12"
    cd.mkdir(parents=True)
    if clip_features:
        for c in IDS[:n_ids]:
            with h5py.File(cd / f"{c}.h5", "w") as hf:
                hf.create_dataset("feature", data=np.zeros((4, 100), dtype=np.float32))
    return ds, h5root


def run(ds, h5root, n=6, require_clip_features=False, content=True):
    msgs = []
    bad = CC.check(n, h5root, T_PRE, ds, ds / "plan" / "assignment_v2final.csv", "dev", log=msgs.append,
                   require_clip_features=require_clip_features, content=content, frames=T)
    return bad


def _label(h5, kind):
    return h5 / "label" / kind / "dev" / "outdoor_siren_v12.h5"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="v2fcc_"))
    results = []
    try:
        # C0
        ds, h5 = make_fixture(tmp / "c0")
        bad = run(ds, h5)
        results.append(("C0 そろっていれば OK", bad == [], bad))
        # C1: RA01 の反例 = ラベル h5 を全部消し、unrelated.h5（HDF5 でない）だけを新しく置く
        ds, h5 = make_fixture(tmp / "c1")
        shutil.rmtree(h5 / "label")
        shutil.rmtree(h5 / "features")
        (h5 / "unrelated.h5").write_bytes(b"not an HDF5")
        bad = run(ds, h5)
        ok = any("必要なラベル h5 が無い" in b for b in bad) and any("開けない" in b for b in bad)
        results.append(("C1 unrelated.h5 だけ → NG（ラベル無し・開けない）", ok, bad))
        # C2: adpit の h5 だけ古い
        ds, h5 = make_fixture(tmp / "c2")
        old = T_PRE - 3600
        os.utime(_label(h5, "adpit"), (old, old))
        bad = run(ds, h5)
        results.append(("C2 片方のラベル h5 が古い → NG", any("更新されていない" in b for b in bad), bad))
        # C3: adpit_dist に 1 本欠け
        ds, h5 = make_fixture(tmp / "c3")
        with h5py.File(_label(h5, "adpit_dist"), "a") as hf:
            del hf[IDS[2]]
        bad = run(ds, h5)
        results.append(("C3 adpit_dist に clip 欠け → NG", any("clip が無い 1" in b for b in bad), bad))
        # C4: adpit/dist が空
        ds, h5 = make_fixture(tmp / "c4")
        with h5py.File(_label(h5, "adpit_dist"), "a") as hf:
            del hf[f"{IDS[0]}/adpit/dist"]
            hf.create_dataset(f"{IDS[0]}/adpit/dist", data=np.zeros((0, 6, C), dtype=np.float32))
        bad = run(ds, h5)
        results.append(("C4 adpit/dist が空 → NG", any("フレーム数 0" in b or "無いか空" in b for b in bad), bad))
        # C5: foa の 1 本が 0 バイト
        ds, h5 = make_fixture(tmp / "c5")
        (ds / "foa" / f"{IDS[1]}.flac").write_bytes(b"")
        bad = run(ds, h5)
        results.append(("C5 foa が 0 バイト → NG", any(b.startswith("foa 欠け 1") for b in bad), bad))
        # C6: 特徴 h5 が空 → NG
        ds, h5 = make_fixture(tmp / "c6", feature_keys=())
        bad = run(ds, h5)
        results.append(("C6 特徴 h5 が空 → NG", any("が空" in b for b in bad), bad))
        # C7: plan の行数が違う
        ds, h5 = make_fixture(tmp / "c7")
        bad = run(ds, h5, n=7)
        results.append(("C7 plan の行数が期待と違う → NG", any("行数" in b for b in bad), bad))
        # C8（監査 A F2・A2）: clip ごとの特徴 h5 を要求する構成で、半分欠ける → NG
        ds, h5 = make_fixture(tmp / "c8", clip_features=True)
        cd = h5 / "data" / "24000fs" / "wav" / "dev" / "outdoor_siren_v12"
        for c in IDS[:3]:
            (cd / f"{c}.h5").unlink()
        bad = run(ds, h5, require_clip_features=True)
        results.append(("C8 特徴 h5 が半分欠け（要求する構成）→ NG", any("特徴 h5 の欠け 3" in b for b in bad), bad))
        # C9（監査 A F2・A3）: 要求する構成で 1 本も無い → NG。要求しない既定（v2final の実物）では OK
        ds, h5 = make_fixture(tmp / "c9", clip_features=True)
        shutil.rmtree(h5 / "data")
        bad = run(ds, h5, require_clip_features=True)
        bad0 = run(ds, h5)
        results.append(("C9 特徴 h5 が 1 本も無い → 要求時 NG・既定 OK", any("特徴 h5 の欠け 6" in b for b in bad) and bad0 == [], (bad, bad0)))
        # C13: ラベル h5 が adpit・adpit_dist だけで accdoa・track が無い → NG（サーバの実物は 4 本）
        ds, h5 = make_fixture(tmp / "c13")
        shutil.rmtree(h5 / "label" / "accdoa")
        bad = run(ds, h5)
        results.append(("C13 accdoa のラベル h5 が無い → NG", any("label/accdoa" in b for b in bad), bad))
        # C15: metadata・metadata_dist の 0 バイト（正解の無い本）は OK、masks の 0 バイトは NG
        ds, h5 = make_fixture(tmp / "c15")
        (ds / "metadata" / f"{IDS[2]}.csv").write_bytes(b"")
        (ds / "metadata_dist" / f"{IDS[2]}.csv").write_bytes(b"")
        bad = run(ds, h5)
        (ds / "masks" / f"{IDS[3]}.csv").write_bytes(b"")
        bad2 = run(ds, h5)
        results.append(("C15 正解なしの 0 バイト csv は OK・masks の 0 バイトは NG", bad == [] and any(b.startswith("masks 欠け 1") for b in bad2), (bad, bad2)))
        # C14: track の活動配列（sed_label）が無い → NG（内容の照合が「活動配列なし」で止める）
        ds, h5 = make_fixture(tmp / "c14")
        with h5py.File(_label(h5, "track"), "a") as hf:
            del hf[f"{IDS[0]}/sed_label"]
        bad = run(ds, h5)
        results.append(("C14 track の sed_label が無い → NG", any("活動配列なし 1" in b for b in bad), bad))
        # C10（監査 A F12・A7）: adpit の clip に adpit/se が無い → NG
        ds, h5 = make_fixture(tmp / "c10")
        with h5py.File(_label(h5, "adpit"), "a") as hf:
            del hf[f"{IDS[0]}/adpit/se"]
        bad = run(ds, h5)
        results.append(("C10 adpit/se が無い → NG", any("adpit/se 無し" in b for b in bad), bad))
        # C11（監査 A F12・A8）: adpit_dist の距離が NaN → NG
        ds, h5 = make_fixture(tmp / "c11")
        with h5py.File(_label(h5, "adpit_dist"), "a") as hf:
            hf[f"{IDS[-1]}/adpit/dist"][...] = np.nan
        bad = run(ds, h5)
        results.append(("C11 adpit/dist が NaN → NG", any("非有限値" in b for b in bad), bad))
        # C12: 特徴 h5 の中身が空 → NG（clip 名の h5 があれば要求の有無にかかわらず中身を見る）
        ds, h5 = make_fixture(tmp / "c12", clip_features=True)
        p = h5 / "data" / "24000fs" / "wav" / "dev" / "outdoor_siren_v12" / f"{IDS[1]}.h5"
        with h5py.File(p, "w") as hf:
            hf.create_dataset("feature", data=np.zeros((0, 100), dtype=np.float32))
        bad = run(ds, h5)
        results.append(("C12 特徴 h5 の中身が空 → NG", any("特徴なし" in b for b in bad), bad))
        # C16（RF01 の 2 行目）: 正解 csv を残し、同じ clip の 4 種の h5 の活動を全部 0 にする（キー・形・更新時刻は維持）→ NG。--no-content なら通る
        ds, h5 = make_fixture(tmp / "c16")
        for kind in CC.LABEL_KINDS:
            with h5py.File(_label(h5, kind), "a") as hf:
                g = hf[IDS[0]]
                for key in ("adpit/se", "adpit/dist", "accdoa/se", "sed_label"):
                    if key in g:
                        g[key][...] = 0
        bad = run(ds, h5)
        bad_nc = run(ds, h5, content=False)
        ok = sum(1 for b in bad if "活動が食い違う 1" in b) == 4 and bad_nc == []
        results.append(("C16 正解 csv ありで h5 の活動を 0 に → 4 種とも NG・--no-content なら通る", ok, (bad, bad_nc)))
        # C17: csv が空（正解なし）なのに h5 に活動 → NG
        ds, h5 = make_fixture(tmp / "c17")
        with h5py.File(_label(h5, "accdoa"), "a") as hf:
            hf[f"{IDS[2]}/accdoa/se"][2, 4] = True
        bad = run(ds, h5)
        results.append(("C17 csv が空なのに h5 に活動 → NG", any("accdoa" in b and "活動が食い違う 1" in b for b in bad), bad))
        # C18: 活動フレームが 1 つずれる（frame 9 → 0 に移す）→ NG（本数が同じでも集合が違う）
        ds, h5 = make_fixture(tmp / "c18")
        with h5py.File(_label(h5, "adpit"), "a") as hf:
            se = hf[f"{IDS[0]}/adpit/se"][()]
            se[9, 0, 4] = False; se[0, 0, 4] = True
            hf[f"{IDS[0]}/adpit/se"][...] = se
        bad = run(ds, h5)
        results.append(("C18 活動フレームが 1 つずれる → NG（adpit だけ）", sum(1 for b in bad if "活動が食い違う 1" in b) == 1 and any("adpit" in b and "adpit_dist" not in b and "活動が食い違う 1" in b for b in bad), bad))
        # C19: metadata と metadata_dist の先頭 5 列が食い違う → NG
        ds, h5 = make_fixture(tmp / "c19")
        _write_meta(ds, IDS[0], ROWS[IDS[0]][:-1], dist=False)
        bad = run(ds, h5)
        results.append(("C19 metadata と metadata_dist が食い違う → NG", any("先頭 5 列が食い違う本 1" in b for b in bad), bad))
        # C20（Astra 10/7 V05）: 正解 csv を残し、4 種の h5 を「無活動の frame 0 だけ（1 フレーム）」に短く切る → NG（期待長 10 と違う）
        ds, h5 = make_fixture(tmp / "c20")
        for kind in CC.LABEL_KINDS:
            with h5py.File(_label(h5, kind), "a") as hf:
                g = hf[IDS[0]]
                for key in ("adpit/se", "adpit/azi", "adpit/dist", "accdoa/se", "accdoa/azi", "sed_label", "doa_label"):
                    if key in g:
                        arr = g[key][()][:1]
                        del g[key]
                        g.create_dataset(key, data=np.zeros_like(arr))
        bad = run(ds, h5)
        results.append(("C20 h5 を 1 フレームに短く切る → 4 種とも NG", sum(1 for b in bad if "活動が食い違う 1" in b) == 4 and any("フレーム数 1 ≠ csv の最後のフレーム+1 = 10" in b for b in bad), bad))
        # C21（Astra 10/7 L02）: 4 音源のフレーム（IDS[1] の 3〜5）で track を全部 0 に → NG（上位 3 枠が残るはず）
        ds, h5 = make_fixture(tmp / "c21")
        with h5py.File(_label(h5, "track"), "a") as hf:
            se = hf[f"{IDS[1]}/sed_label"][()]
            se[3:6] = False
            hf[f"{IDS[1]}/sed_label"][...] = se
        bad = run(ds, h5)
        results.append(("C21 4 音源のフレームで track を全消し → NG", any("track" in b and "活動が食い違う 1" in b for b in bad), bad))
        # C22: csv に期待長の範囲外のフレーム（frame 12）を足す → NG（捨てずに止める）
        ds, h5 = make_fixture(tmp / "c22")
        _write_meta(ds, IDS[0], ROWS[IDS[0]] + [(12, 4, 0, 10, 0, 5.0)], dist=True)
        _write_meta(ds, IDS[0], ROWS[IDS[0]] + [(12, 4, 0, 10, 0, 5.0)], dist=False)
        bad = run(ds, h5)
        results.append(("C22 csv に範囲外のフレーム → NG", any("範囲外のフレーム" in b for b in bad), bad))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = 0
    for name, ok, bad in results:
        print(("合格 " if ok else "不合格 ") + name + ("" if ok else f"  bad={bad}"))
        n_ok += ok
    print(f"{n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
