# -*- coding: utf-8 -*-
"""step11_v23_render.py — 警告音の直し（2026-10-12・Fable・依頼文 `md/handoff/Fable依頼_合成の音を現実に近づける_2026-10-10.md` r5 §1・§10・順番の 3）。

v22（`step11_v22_render.py`）の上に重ねる新しい版。既存の版のファイルは変えない。変えるのは**音だけ**で、場面の乱数の消費は変えない
（同じ seed の場面は同じ軌道・同じ時刻・同じ暗騒音）。意図して変える列は下に書く。音色のための乱数は独立
（np.random.default_rng([seed, 20261012, 印, 元の値から作った整数])・元の rng は 1 回も余計に消費しない・r5 §8）。

1. クラクション（`outdoor_seld.alert_sounds_v23.make_horn_v23`）: 押しの列をクリップの絶対時刻で描く。
   - 走る車（v13 の `_rewrite_horn`）: 回数・押しの長さ・間を Takada 2013/2014 の 6 型と §10 の長さで引き直す（独立の乱数）→
     **変わる列: params["presses"]（t_on からの相対）・t_off・n_honk・horn_type・grammar**。t_on は元のまま（列が窓の終わりを越えるときだけ早める）。
   - 止まった車（v19_static_horn・1 押し = 1 窓・今の plan には無い）: 窓の中は連続音。窓は変えない。
   - 音色（独立の乱数・params に残す）: f_lo・f_hi（2 音のとき）・dual・alpha・onset_sec・release_tau_sec。
2. バック音（`make_backup_beep_v23`）: 先頭の音を鳴動窓の頭 t_on に。型 3 つ（単音スピーカー式・単音圧電式・2 音チャイム）を同じ確率。
   **変わる列: params["kind"] とその型のパラメータ**。窓は変えない。
3. ベルの高さ（`bell_f0_v23`）: 3 型とも JBPI 2023 表 3 の種類 → 試料 → ±3%。**変わる列: params["f0"]・params["bell_sample"]・ディンディン型は n_rings = 1**。
   ピン型の範囲の検査（v22 の PIN_F0）は単打ベル 4 種の幅（2,383〜3,017 Hz）に差し替える。
4. ベルつき走行（v19bellbike）のベルの高さ z: 自転車の軌道の 0.1〜0.4 m → ハンドルの高さ U(0.95, 1.10) m（事実 = パナソニックの車種の 965〜1,085 mm・幅は仮定）。
   **変わる列: ベルの wp の z だけ**（xy・速さ・鳴らす時刻は変えない）。v22 の check_bellbike（軌道が同じ）は xy と時刻だけ比べる形に差し替える。

音の大きさの合わせ方（最大値・速さの曲線）はこの版には入れていない（本人の決定 10/12 15:28 のあとに別の版で）。

使い方:
  python scripts/step11_v23_render.py selftest                 # 代表の場面を引いて検査（描画なし）
  python scripts/step11_v23_render.py audition <出力dir>        # 試聴用の wav（旧版と新版の並べ比べ・README つき）
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import step11_v22_render as v22  # noqa: E402
from outdoor_seld import alert_sounds_v23 as a23  # noqa: E402

m9, v13, v19r = v22.m9, v22.v13, v22.v19r
SALT = 20261012
BELLBIKE_Z = (0.95, 1.10)
PLAN = ROOT / "out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv"


def _irng(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


def _kind_of(p: dict) -> str:
    bt = p.get("bell_type", "ding")
    return "pin" if bt == "pin" else ("ring" if bt == "ring" else "ding")


# ---------------------------------------------------------------- 1〜3: params（音色）を差し替える
def _warn_params_v23(prev):
    def fn(cls: str, rng: np.random.Generator, seed: int) -> dict:
        p = prev(cls, rng, seed)                       # 元の抽選（乱数の消費は同じ）
        if cls == "horn":
            p = dict(p, **a23.horn_params_v23(_irng(seed, SALT, 1, round(p["f_lo"] * 1000))))
        elif cls == "backup_beep":
            p = dict(p, **a23.backup_params_v23(_irng(seed, SALT, 2, round(p["freq"] * 1000))))
        elif cls == "bike_bell":
            kind = _kind_of(p)
            f0, sid = a23.bell_f0_v23(kind, _irng(seed, SALT, 3, round(p["f0"] * 1000)))
            p = dict(p, f0=f0, bell_sample=sid, v23=True)
            if kind == "ding":
                p["n_rings"] = 1                       # 2 回目の打音は無し（本人 10/10 16:43）
        return p
    fn._v23 = True
    return fn


def _bell_params_of_type_v23(prev):
    def fn(kind: str, rng: np.random.Generator, seed: int) -> dict:
        p = prev(kind, rng, seed)
        f0, sid = a23.bell_f0_v23(kind, _irng(seed, SALT, 3, round(p["f0"] * 1000)))
        p = dict(p, f0=f0, bell_sample=sid, v23=True)
        if kind == "ding":
            p["n_rings"] = 1
        return p
    fn._v23 = True
    return fn


# ---------------------------------------------------------------- 1: 走る車の押しの列（窓の終わりだけ変わる）
def _rewrite_horn_v23(prev):
    def fn(src: dict, rng: np.random.Generator) -> dict:
        s = prev(src, rng)                             # 元の乱数の消費（n・honk・gap・lead）はそのまま
        p = s["params"]
        plan = a23.horn_press_plan(_irng(p["audio_seed"], SALT, 6, round(p["f_lo"] * 1000)))
        t_on, total = float(s["t_on"]), float(plan["total"])
        if t_on + total > m9.CLIP - 0.3:
            t_on = max(0.3, m9.CLIP - 0.3 - total)
        s["t_on"], s["t_off"] = round(t_on, 3), round(min(t_on + total, m9.CLIP - 0.1), 3)
        s["params"] = dict(p, presses=[list(x) for x in plan["presses"]])
        s["n_honk"], s["horn_type"], s["grammar"] = plan["n"], plan["type"], "v23_presses"
        return s
    fn._v23 = True
    return fn


# ---------------------------------------------------------------- 1・2: ドライ音
def _make_dry_v23(prev):
    def fn(src: dict) -> np.ndarray:
        cls, p = src["class"], src.get("params", {})
        if cls == "horn" and p.get("v23"):
            t_on, t_off = float(src.get("t_on", 0.0)), float(src.get("t_off", m9.CLIP))
            rel = p.get("presses") or [[0.0, t_off - t_on]]         # 押しの列が無い（止まった車 = 1 押し 1 窓）なら窓いっぱいの連続音
            presses = [(t_on + float(a), float(L)) for a, L in rel]
            return a23.make_horn_v23(m9.CLIP, m9.FS_SIM, np.random.default_rng(p["audio_seed"]), presses,
                                     f_lo=p["f_lo"], f_hi=(p["f_hi"] if p.get("dual") else None), alpha=p["alpha"],
                                     onset_sec=p["onset_sec"], release_tau_sec=p["release_tau_sec"])
        if cls == "backup_beep" and p.get("v23"):
            return a23.make_backup_beep_v23(m9.CLIP, m9.FS_SIM, float(src.get("t_on", 0.0)), p)
        return prev(src)
    fn._v23 = True
    return fn


# ---------------------------------------------------------------- 4: ベルつき走行のベルの高さ
def _bellbike_v23(prev):
    def fn(row: dict) -> dict:
        s = prev(row)
        bell = next(src for src in s["sources"] if src["class"] == "bike_bell")
        z = float(_irng(row["seed"], SALT, 7).uniform(*BELLBIKE_Z))
        bell["wp"] = [[pt[0], pt[1], pt[2], z] for pt in bell["wp"]]
        bell["bell_z"] = round(z, 3)
        s.setdefault("v23", {})["bell_z"] = round(z, 3)
        return s
    fn._v23 = True
    return fn


def check_bellbike_v23(s: dict) -> None:
    """v22.check_bellbike の差し替え: 軌道は xy と時刻が同じで、z はハンドルの高さ。"""
    srcs = s["sources"]
    assert len(srcs) == 2 and srcs[0]["class"] == v22.CLASS_BICYCLE and srcs[1]["class"] == "bike_bell"
    b = srcs[1]
    assert [[pt[0], pt[1], pt[2]] for pt in b["wp"]] == [[pt[0], pt[1], pt[2]] for pt in srcs[0]["wp"]], "ベルと走行音の軌道（xy・時刻）が違う"
    assert all(BELLBIKE_Z[0] - 1e-6 <= pt[3] <= BELLBIKE_Z[1] + 1e-6 for pt in b["wp"]), "ベルの高さがハンドルの高さの外"
    kind = _kind_of(b["params"])
    assert kind in ("ding", "ring", "pin")
    lo, hi = a23.bell_f0_range(kind)
    assert lo - 1e-6 <= b["params"]["f0"] <= hi + 1e-6, f"{kind} の基音が範囲外"
    assert 0.0 <= b["t_on"] < b["t_off"] <= m9.CLIP, "鳴動窓が不正"
    assert b["params"].get("align_strike") and v22.strike_at_window_start(b), "鳴動窓の頭に打音が無い（D08-01）"
    assert abs(b["t_on"] - srcs[0]["t_cpa_s"]) <= v22.BELL_START_WINDOW_S + 1e-6 or b["t_on"] in (0.2,), "鳴り始めが最接近の ±4 s の外"


# ---------------------------------------------------------------- 登録（import 時に 1 回）
if not getattr(m9._warn_params, "_v23", False):
    m9._warn_params = _warn_params_v23(m9._warn_params)
if not getattr(v22.bell_params_of_type, "_v23", False):
    v22.bell_params_of_type = _bell_params_of_type_v23(v22.bell_params_of_type)
if not getattr(v13._rewrite_horn, "_v23", False):
    v13._rewrite_horn = _rewrite_horn_v23(v13._rewrite_horn)
if not getattr(m9._make_dry, "_v23", False):
    m9._make_dry = _make_dry_v23(m9._make_dry)
if not getattr(v22.sample_v22bellbike, "_v23", False):
    v22.sample_v22bellbike = _bellbike_v23(v22.sample_v22bellbike)
    v19r._SAMPLERS19[v22.SCEN_BELLBIKE] = v22.sample_v22bellbike
v22.check_bellbike = check_bellbike_v23
v22.PIN_F0 = a23.bell_f0_range("pin")
v22.EXTRA_BELL_KEYS = tuple(v22.EXTRA_BELL_KEYS) + ("bell_sample", "v23")
generate_clip = v22.generate_clip
sample_scene_v19 = v22.sample_scene_v19


# ---------------------------------------------------------------- 検査
def check_horn_dry(src: dict, dry: np.ndarray) -> None:
    p = src["params"]; t_on, t_off = float(src["t_on"]), float(src["t_off"])
    rel = p.get("presses") or [[0.0, t_off - t_on]]
    segs = a23.sounding_segments(dry, m9.FS_SIM)
    assert len(segs) == len(rel), f"押し {len(rel)} 回に対して鳴る区間 {len(segs)}（{segs}）"
    for (a, L), (s0, s1) in zip(rel, segs):
        assert abs(s0 - (t_on + a)) <= 0.010, f"押しの開始 {t_on + a:.3f} に対して鳴り始め {s0:.3f}"
        assert 0.0 <= s1 - (t_on + a + L) <= 0.15, f"押しの終わり {t_on + a + L:.3f} に対して鳴り終わり {s1:.3f}"
    assert s1 <= t_off + 0.15 + 1e-6, "最後の押しが窓の終わりを越える"


def check_backup_dry(src: dict, dry: np.ndarray) -> None:
    t_on = float(src["t_on"])
    segs = a23.sounding_segments(dry, m9.FS_SIM, rel_db=-20.0)
    assert segs and abs(segs[0][0] - t_on) <= 0.005, f"先頭の音 {segs[0][0] if segs else None} が t_on {t_on} に合わない"
    k0 = int((t_on - 0.01) * m9.FS_SIM)
    assert k0 <= 0 or float(np.max(np.abs(dry[:k0]))) == 0.0, "t_on より前に音がある"
    assert src["params"]["kind"] in a23.BACKUP_KINDS


def check_bell_params(p: dict) -> None:
    kind = _kind_of(p)
    lo, hi = a23.bell_f0_range(kind)
    assert lo - 1e-6 <= p["f0"] <= hi + 1e-6, f"{kind} の基音 {p['f0']:.0f} が範囲 {lo:.0f}〜{hi:.0f} の外"
    sp, j = p["bell_sample"].split("-")
    assert sp in a23.BELL_TYPE_SPECIES[kind] and 1 <= int(j) <= len(a23.BELL_SAMPLES[sp]), "試料の印が型に合わない"
    base = a23.BELL_SAMPLES[sp][int(j) - 1]
    assert a23.BELL_JITTER[0] - 1e-9 <= p["f0"] / base <= a23.BELL_JITTER[1] + 1e-9, "±3% の外"
    if kind == "ding":
        assert p.get("n_rings") == 1, "ディンディン型が 1 打になっていない"


def _plan_rows(pred, n: int) -> list:
    out = []
    with open(PLAN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if pred(r):
                r["seed"] = int(r["seed"]); r["n_warnings"] = int(r.get("n_warnings") or 0); out.append(r)
                if len(out) >= n:
                    break
    return out


_V22_DUMP = r"""
import json, sys
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import step11_v22_render as v22
rows = json.loads(sys.stdin.read())
out = []
for r in rows:
    s = v22.sample_scene_v19(r)
    out.append({"noise": s["noise"]["dba"], "sources": [{"class": x["class"], "kind": x["kind"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"],
                 "t_off": x["t_off"], "l1m": x.get("l1m_db"), "speed": x.get("speed_mps"), "t_cpa": x.get("t_cpa_s")} for x in s["sources"]]})
print(json.dumps(out))
"""


def _strip_for_compare(s: dict) -> dict:
    out = []
    for x in s["sources"]:
        d = {"class": x["class"], "kind": x["kind"], "track": x["track"], "wp": x["wp"], "t_on": x["t_on"], "t_off": x["t_off"],
             "l1m": x.get("l1m_db"), "speed": x.get("speed_mps"), "t_cpa": x.get("t_cpa_s")}
        if x["class"] == "horn" and x.get("grammar") == "v23_presses":
            d["t_off"] = None                                    # 意図して変える列
        if x["class"] == "bike_bell" and s.get("v23", {}).get("bell_z") is not None:
            d["wp"] = [[pt[0], pt[1], pt[2]] for pt in x["wp"]]  # z は意図して変える
        out.append(d)
    return {"noise": s["noise"]["dba"], "sources": out}


def selftest() -> int:
    ok = []
    rows = {}
    rows["horn"] = _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "horn", 4)
    rows["beep"] = _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "backup_beep", 4)
    rows["bell"] = _plan_rows(lambda r: r["scenario"] == "v11core" and r["w1_class"] == "bike_bell", 6)
    rows["bellbike"] = _plan_rows(lambda r: r["scenario"] == "v19bellbike", 4)
    rows["bellpin"] = _plan_rows(lambda r: r["scenario"] == "v19bellpin", 3)
    # 1 クラクション: 押しの列どおりに鳴る・音色の params が付く
    n_h, kinds_seen = 0, set()
    for r in rows["horn"]:
        s = sample_scene_v19(r)
        for src in s["sources"]:
            if src["class"] == "horn":
                assert src["params"].get("v23") and src.get("grammar") == "v23_presses", "v23 の params が無い"
                check_horn_dry(src, m9._make_dry(src)); n_h += 1; kinds_seen.add(src["horn_type"])
    ok.append((f"horn: 走る車のクラクション {n_h} 本で、押しの列どおりに鳴り始め・区間の数が合う（型 {sorted(kinds_seen)}）", n_h >= 4))
    # 2 バック音: 先頭の音が t_on
    n_b, kinds_b = 0, set()
    for r in rows["beep"]:
        s = sample_scene_v19(r)
        for src in s["sources"]:
            if src["class"] == "backup_beep":
                check_backup_dry(src, m9._make_dry(src)); n_b += 1; kinds_b.add(src["params"]["kind"])
    ok.append((f"backup: {n_b} 本で先頭の音が t_on に合い、型が付く（{sorted(kinds_b)}）", n_b >= 4))
    # 3 ベルの高さ・1 打
    n_bell, kinds_bell = 0, set()
    for r in rows["bell"] + rows["bellpin"] + rows["bellbike"]:
        s = sample_scene_v19(r)
        for src in s["sources"]:
            if src["class"] == "bike_bell":
                check_bell_params(src["params"]); n_bell += 1; kinds_bell.add(_kind_of(src["params"]))
                assert v22.strike_at_window_start(src), "鳴動窓の頭に打音が無い"
    ok.append((f"bell: {n_bell} 本で基音が表 3 の試料 ±3%・試料の印・ディンディン型は 1 打・窓の頭に打音（型 {sorted(kinds_bell)}）", n_bell >= 10 and kinds_bell == {"ding", "ring", "pin"}))
    # ディンディン型のドライ音に 2 打目が無い（打音の区間が 1 つ）
    s_d = None
    for r in rows["bell"]:
        s = sample_scene_v19(r)
        for src in s["sources"]:
            if src["class"] == "bike_bell" and _kind_of(src["params"]) == "ding":
                s_d = src; break
        if s_d:
            break
    if s_d is not None:
        dry = m9._make_dry(s_d); t_on = float(s_d["t_on"]); seg = dry[int(t_on * m9.FS_SIM): int(min(m9.CLIP, t_on + 1.5) * m9.FS_SIM)]
        segs = a23.sounding_segments(seg, m9.FS_SIM, rel_db=-25.0, min_gap_sec=0.1)
        ok.append((f"bell ding: 窓の頭から 1.5 s の中の打音の区間が 1 つ（{segs}）", len(segs) == 1))
    # 4 ベルつき走行の高さ
    zs = []
    for r in rows["bellbike"]:
        s = sample_scene_v19(r); v22.check_scene("v19bellbike", s); zs.append(s["v23"]["bell_z"])
    ok.append((f"bellbike: ベルの z がハンドルの高さ {BELLBIKE_Z}（{zs}）・v22 の検査（xy と時刻は同じ軌道・打音・±4 s）が通る", all(BELLBIKE_Z[0] <= z <= BELLBIKE_Z[1] for z in zs)))
    for r in rows["bellpin"]:
        v22.check_scene("v19bellpin", sample_scene_v19(r))
    ok.append(("bellpin: v22 の検査（ピン型・範囲は単打ベル 4 種の幅・打音）が通る", True))
    # 5 場面の乱数の消費が変わっていない: v22 だけを読んだ別プロセスと、音以外の列が一致
    sample = rows["horn"][:2] + rows["beep"][:2] + rows["bell"][:2] + rows["bellbike"][:2] + rows["bellpin"][:1]
    res = subprocess.run([sys.executable, "-c", _V22_DUMP % (str(ROOT / "scripts"), str(ROOT / "src"))], input=json.dumps(sample),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    same = False
    if res.returncode == 0:
        ref = json.loads(res.stdout.strip().splitlines()[-1])
        mine = [_strip_for_compare(sample_scene_v19(r)) for r in sample]
        diffs = []
        for a, b, r in zip(ref, mine, sample):
            if abs(a["noise"] - b["noise"]) > 1e-9 or len(a["sources"]) != len(b["sources"]):
                diffs.append(r["clip_id"]); continue
            for x, y in zip(a["sources"], b["sources"]):
                if y["t_off"] is None:
                    x = dict(x, t_off=None)
                if x["class"] == "bike_bell" and len(y["wp"][0]) == 3:
                    x = dict(x, wp=[[pt[0], pt[1], pt[2]] for pt in x["wp"]])
                if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                    diffs.append(r["clip_id"]); break
        same = not diffs
        ok.append((f"場面の乱数: v22 だけの別プロセスと {len(sample)} 本の軌道・時刻・暗騒音・音量が一致（違いは意図した列だけ: 走る車のクラクションの t_off・ベルつき走行のベルの z）{'' if same else diffs}", same))
    else:
        ok.append(("場面の乱数: 別プロセスの v22 の抽選が動く", False)); print(res.stderr[-800:])
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


# ---------------------------------------------------------------- 試聴
def audition(out_dir: Path) -> int:
    """旧版と新版のドライ音を並べる（音色と鳴り方の比較・大きさはそろえる＝ピーク −3 dBFS）。実録の参照は本人のベル（調整用の家の録音）。"""
    import soundfile as sf
    from outdoor_seld.alert_sounds import make_horn, make_backup_beep, make_bike_bell, make_bike_bell_ring
    from outdoor_seld.bike_bell_v2 import make_bike_bell_v2
    fs = m9.FS_SIM
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["警告音の直し（v23・2026-10-12）の試聴ファイル。旧版と新版のドライ音（伝搬なし）を同じ大きさ（ピーク −3 dBFS）で並べた。"
             "大きさの違いは別の決定なので、ここでは「鳴り方（押しの列・先頭の位置）」と「音色・高さ」を聞いてほしい。wav は git に入れない（README だけ）。", ""]

    def save(name: str, x: np.ndarray, note: str):
        y = x / (np.max(np.abs(x)) + 1e-12) * 10 ** (-3 / 20)
        sf.write(str(out_dir / name), (y * 32767).astype(np.int16), fs, subtype="PCM_16")
        lines.append(f"{name}: {note}"); print("wrote", name)

    rng = np.random.default_rng(12)
    # ---- クラクション
    old = make_horn(6.0, fs, np.random.default_rng(1), f_lo=410.0, f_hi=500.0)
    save("horn_old_window_2press.wav", m9._window(old, 1.0, 2.0) if hasattr(m9, "_window") else old, "旧版: 0 秒から 0.35 s 鳴って 0.15 s 休むの繰り返しを窓 1.0〜2.0 s で切ったもの（1 回の押しのつもりが途切れる）")
    for name, presses, dual, note in (
        ("horn_new_single_short_dual.wav", [(1.0, 0.45)], True, "新版: 単発（短い押し 0.45 s）・2 音"),
        ("horn_new_double_short_single.wav", [(1.0, 0.3), (1.6, 0.4)], False, "新版: 短く 2 回（0.3 s・間 0.3 s・0.4 s）・単音"),
        ("horn_new_single_long_dual.wav", [(1.0, 1.5)], True, "新版: 長く 1 回（1.5 s・連続音）・2 音"),
        ("horn_new_triple_short_dual.wav", [(1.0, 0.2), (1.5, 0.25), (2.05, 0.2)], True, "新版: 短く 3 回・2 音"),
    ):
        p = a23.horn_params_v23(rng); f_hi = p["f_lo"] * 1.22 if dual else None
        x = a23.make_horn_v23(5.0, fs, np.random.default_rng(2), presses, f_lo=p["f_lo"], f_hi=f_hi, alpha=p["alpha"], onset_sec=p["onset_sec"], release_tau_sec=p["release_tau_sec"])
        save(name, x, note + f"（基音 {p['f_lo']:.0f} Hz・倍音の減り α={p['alpha']:.2f}・重心 {a23.spectral_centroid_hz(x, fs):.0f} Hz）")
    # ---- バック音
    save("backup_old_1kHz.wav", make_backup_beep(6.0, fs), "旧版: 1 kHz・0.5 s 鳴って 0.5 s 休む（毎分 60 回）")
    for kind, note in (("speaker", "新版: 単音スピーカー式（700〜1,900 Hz・毎分 60〜100 回・鳴る割合 0.4〜0.6）"),
                       ("piezo", "新版: 単音圧電式（2,100〜3,200 Hz）"),
                       ("chime", "新版: 2 音チャイム（ピン 2,000 → 0.25 s 後にポーン 1,590 Hz・1.25 s ごと・高さと時間 ±10%）")):
        r = np.random.default_rng(abs(hash(kind)) % 1000)
        while True:
            p = a23.backup_params_v23(r)
            if p["kind"] == kind:
                break
        x = a23.make_backup_beep_v23(6.0, fs, 0.5, p)
        detail = f"{p['freq']:.0f} Hz・毎分 {p['rate_per_min']:.0f} 回・割合 {p['duty']:.2f}" if kind != "chime" else f"高さ ×{p['pitch_mult']:.2f}・時間 ×{p['time_mult']:.2f}"
        save(f"backup_new_{kind}.wav", x, note + f"（この例: {detail}・先頭の音は 0.5 s）")
    # ---- ベル
    save("bell_old_ding_3000_2strikes.wav", make_bike_bell(3.0, fs, f0=3000.0, n_rings=2, ring_gap_sec=0.45, repeat_period_sec=10.0), "旧版 ディンディン型: 基音 3,000 Hz・2 打（0.45 s）")
    for f0, sid in ((2185.0, "b-1"), (2657.0, "a-1"), (3536.0, "c-2")):
        save(f"bell_new_ding_{int(f0)}_{sid}.wav", make_bike_bell(3.0, fs, f0=f0, n_rings=1, ring_gap_sec=0.45, repeat_period_sec=10.0), f"新版 ディンディン型: 1 打・基音 {f0:.0f} Hz（JBPI 試料 {sid}）")
    save("bell_old_ring_3000.wav", make_bike_bell_ring(3.0, fs, f0=3000.0), "旧版 チリリン型: 基音 3,000 Hz")
    for f0, sid in ((2185.0, "b-1"), (2657.0, "a-1"), (3337.0, "c-1")):
        save(f"bell_new_ring_{int(f0)}_{sid}.wav", make_bike_bell_ring(3.0, fs, f0=f0), f"新版 チリリン型: 基音 {f0:.0f} Hz（引きベル 試料 {sid}）")
    for f0, sid in ((2457.0, "d-1"), (2929.0, "i-1")):
        save(f"bell_new_pin_{int(f0)}_{sid}.wav", make_bike_bell_v2(3.0, fs, f0=f0, repeat_period_sec=10.0, rng=np.random.default_rng(3)), f"新版 ピン型: 1 打・基音 {f0:.0f} Hz（単打ベル 試料 {sid}）")
    real = ROOT / "raw_demo/261007_003.WAV"
    if real.is_file():
        x, sr = sf.read(str(real), dtype="float64", always_2d=True)
        seg = x[int(54.3 * sr): int(57.3 * sr), 0]
        if sr != fs:
            from scipy.signal import resample_poly
            seg = resample_poly(seg, fs, sr)
        save("real_bell_pin_demo1.wav", seg, "参照: 本人の自転車のベル（10/7 の家の録音・調整用・ピン型・W チャンネル）")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "audition":
        sys.exit(audition(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "audition_2026-10-12"))
    print(__doc__); sys.exit(2)
