# -*- coding: utf-8 -*-
"""物理ablationスイッチのスモークテスト v2（第9回監査の条件を消化した増強版）。

検証内容:
  1) ABLATE未設定で core＋ext3種（キック・バイク・列車）の
     foa/metadata/masks が既存conf実物と**バイト同一**（read_bytes比較。
     v1の「sha先頭64bit・core1本・foaのみ」から拡張=監査[低]/[中]対応）
  2) 4条件で生成が通り、音声がfullとも相互とも異なる（sha256全桁）
  3) 非ゲートクラス(サイレン)の行が no_airabs/no_1r/no_ground で不変
  4) no_doppler: 発音区間境界の単体検証（歩行マイク含む・独立オラクル・
     半開区間 [t_on, t_off) を assert = 監査[中]境界指摘の回帰テスト）
  5) step19距離ラベルを full/no_1r/no_doppler で実行し、
     (frame,class,track)キー集合が各モードのmetadataと一致（下流波及の検証）
使い方: python scripts/_ablate_smoke.py <出力ベースdir>
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["", "no_doppler", "no_airabs", "no_1r", "no_ground"]
EXT_IDX = {"core": 0, "kick": 1200, "bike": 1425, "train": 1650}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_of(p: Path, cls=None):
    out = []
    for line in open(p, encoding="utf-8"):
        q = line.strip().split(",")
        if len(q) == 5 and (cls is None or int(q[1]) == cls):
            out.append(tuple(int(v) for v in q))
    return out


def _nd_boundary_unit():
    """no_doppler境界の単体検証（レンダ不要・独立オラクル）。"""
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from outdoor_seld import ablate
    from outdoor_seld.geometry import (SOUND_SPEED_20C, receiver_positions_at,
                                       solve_emission_times)
    from outdoor_seld.labels import frame_label_rows
    assert ablate.MODE == "no_doppler"
    c = SOUND_SPEED_20C
    rng = np.random.default_rng(7)
    for case in ("static", "walk"):
        # 音源: 10秒直線移動。マイク: 静止 or 歩行
        p0 = np.array([30.0, 5.0, 1.0])
        v = np.array([-4.0, 0.5, 0.0])
        wp = np.array([[0.0, *p0], [10.0, *(p0 + 10 * v)]])
        if case == "static":
            mic = np.array([0.0, 0.0, 1.5])
        else:
            mic = np.array([[0.0, -1.0, 0.0, 1.5], [10.0, 6.0, 1.0, 1.5]])
        # 独立オラクル: 参照点の中央値規約を自前で再計算（fastsimを使わない）
        fs = ablate.NODOP_FS
        n = int(round(10.0 * fs))
        tr = np.arange(n) / fs
        te, ps_te = solve_emission_times(tr, wp, mic, c)
        fin = np.isfinite(te)
        te_ref = float(np.median(te[fin]))
        i_ref = int(np.argmin(np.abs(np.where(fin, te, np.inf) - te_ref)))
        ps_ref = ps_te[i_ref]
        t_frames = (np.arange(100) + 0.5) * 0.1
        mic_at = (receiver_positions_at(t_frames, mic) if mic.ndim == 2
                  else np.broadcast_to(mic, (100, 3)))
        delay = np.linalg.norm(mic_at - ps_ref[None, :], axis=1) / c
        t_nd = t_frames - delay
        # t_off をあるフレームの t_nd にピッタリ合わせ、半開区間で除外されることを確認
        k_edge = 50
        t_on, t_off = 0.0, float(t_nd[k_edge])
        rows, _ = frame_label_rows(wp, mic, 10.0, class_idx=0,
                                   source_active_from=t_on,
                                   source_active_until=t_off, c=c)
        got = {r[0] for r in rows}
        az_ok = np.isfinite(
            solve_emission_times(t_frames, wp, mic, c)[0])
        expect = {k for k in range(100)
                  if az_ok[k] and t_on <= t_nd[k] < t_off}
        assert got == expect, (case, sorted(got ^ expect)[:5])
        assert k_edge not in got, f"{case}: t_nd==t_off の境界フレームが除外されていない"
    print("ND_BOUNDARY_UNIT: PASS (static/walk・半開区間・独立オラクル一致)")


def child(base: Path):
    mode = os.environ.get("ABLATE", "")
    tag = mode or "full"
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import step11_v12_conf_render as cf
    ds = base / tag
    cf.m9.DS = ds
    cf.m9.WORK = ds / "work"
    for sub in ("foa", "metadata", "masks", "work"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    plan = cf.load_plan_v12conf()
    which = list(EXT_IDX) if mode == "" else ["core"]
    info = {"mode": tag, "clips": {}}
    for name in which:
        row = plan[EXT_IDX[name]]
        cf.m12.generate_clip_v12(row)
        cid = row["clip_id"]
        info["clips"][name] = {
            "clip": cid,
            "foa": sha(ds / "foa" / f"{cid}.flac"),
            "ds": str(ds)}
    if mode == "no_doppler":
        _nd_boundary_unit()
    print("SMOKE_JSON:" + json.dumps(info))


def _run_step19(ds: Path):
    """scratchデータセットにstep19距離ラベルを実行し、キー集合を返す。"""
    spec = importlib.util.spec_from_file_location(
        "s19", ROOT / "scripts" / "step19_dist_labels_v12.py")
    s19 = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]   # argparseはmain()内でも走るため
    try:
        spec.loader.exec_module(s19)
        s19.DS = ds
        s19.OUT = ds / "metadata_dist"
        s19.main()
    finally:
        sys.argv = argv
    keys_meta, keys_dist = set(), set()
    for p in (ds / "metadata").glob("*.csv"):
        for r in rows_of(p):
            keys_meta.add((p.stem, r[0], r[1], r[2]))
    for p in (ds / "metadata_dist").glob("*.csv"):
        for line in open(p, encoding="utf-8"):
            q = line.strip().split(",")
            if len(q) == 6:
                keys_dist.add((p.stem, int(q[0]), int(q[1]), int(q[2])))
    return keys_meta, keys_dist


def main():
    base = Path(sys.argv[-1]).resolve()
    if "--child" in sys.argv:
        child(base)
        return
    base.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode in MODES:
        env = dict(os.environ, ABLATE=mode, PYTHONIOENCODING="utf-8")
        out = subprocess.run(
            [sys.executable, __file__, "--child", str(base)],
            env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if out.returncode != 0:
            print(out.stdout[-3000:])
            print(out.stderr[-3000:])
            raise SystemExit(f"child failed for mode={mode!r}")
        for l in out.stdout.splitlines():
            if l.startswith("ND_BOUNDARY_UNIT"):
                print("[4]", l)
        line = [l for l in out.stdout.splitlines()
                if l.startswith("SMOKE_JSON:")][0]
        res[mode or "full"] = json.loads(line[len("SMOKE_JSON:"):])

    conf = ROOT / "out" / "dataset_outdoor_siren_v12_conf"
    # 1) 既定経路: core+ext3種の foa/metadata/masks がconf実物とバイト同一
    ds_full = Path(res["full"]["clips"]["core"]["ds"])
    for name, c_ in res["full"]["clips"].items():
        cid = c_["clip"]
        for sub, ext in (("foa", ".flac"), ("metadata", ".csv"),
                         ("masks", ".csv")):
            a = (ds_full / sub / f"{cid}{ext}").read_bytes()
            b = (conf / sub / f"{cid}{ext}").read_bytes()
            assert a == b, f"既定経路の不一致: {name}/{sub}"
        print(f"[1] 既定経路バイト同一({name}: {cid}): PASS")

    # 2) 音声の相違（sha256全桁）
    hashes = {m: r["clips"]["core"]["foa"] for m, r in res.items()}
    print("[2] core foa sha256:")
    for m, h in hashes.items():
        print(f"    {m:>10}: {h[:32]}…")
    assert len(set(hashes.values())) == len(hashes)

    # 3) サイレン行不変（非ゲートクラス）
    full_meta = ds_full / "metadata" / (res["full"]["clips"]["core"]["clip"] + ".csv")
    full_siren = rows_of(full_meta, cls=0)
    for m in ("no_airabs", "no_1r", "no_ground"):
        p = Path(res[m]["clips"]["core"]["ds"]) / "metadata" / \
            (res[m]["clips"]["core"]["clip"] + ".csv")
        assert rows_of(p, cls=0) == full_siren
        print(f"[3] サイレン行不変({m}): PASS")
    ndp = Path(res["no_doppler"]["clips"]["core"]["ds"]) / "metadata" / \
        (res["no_doppler"]["clips"]["core"]["clip"] + ".csv")
    nd = rows_of(ndp, cls=0)
    print(f"[3] no_dopplerサイレン行: {len(full_siren)}→{len(nd)}行 "
          f"対称差{len(set(nd) ^ set(full_siren))}行（一定遅延規約の移動）")

    # 5) step19キー整合（full / no_1r / no_doppler）
    for m in ("full", "no_1r", "no_doppler"):
        ds = Path(res[m]["clips"]["core"]["ds"])
        km, kd = _run_step19(ds)
        assert km == kd, f"step19キー不一致({m}): {sorted(km ^ kd)[:4]}"
        print(f"[5] step19キー整合({m}): PASS ({len(kd)}キー)")
    print("ALL SMOKE v2 CHECKS DONE")


if __name__ == "__main__":
    main()
